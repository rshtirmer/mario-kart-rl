"""PPO training with 8 parallel environments for Mario Kart RL."""

import time
import json
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch
import gymnasium

from .config import Config
from .env import MarioKartEnv
from .agent import CNNPolicy
from .telemetry import Telemetry

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def get_device():
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps") + 1
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def compute_gae(rewards, values, dones, next_values, gamma, gae_lambda):
    """Vectorized GAE for (n_steps, n_envs) shaped arrays."""
    n_steps, n_envs = rewards.shape
    advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
    last_gae = np.zeros(n_envs, dtype=np.float32)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def make_env(state, max_steps):
    """Factory for AsyncVectorEnv."""
    def _init():
        return MarioKartEnv(state=state, max_episode_steps=max_steps)
    return _init


def train():
    cfg = Config()
    device = get_device()
    print(f"Device: {device}")

    project_root = Path(__file__).resolve().parent.parent.parent

    try:
        exp_id = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
        ).decode().strip()
    except Exception:
        exp_id = f"run_{int(time.time())}"
    print(f"Experiment: {exp_id}")

    run_dir = project_root / "runs" / exp_id
    telem = Telemetry(str(run_dir))

    use_wandb = cfg.use_wandb and _wandb is not None
    if use_wandb:
        _wandb.init(project="mario-kart-rl", config=cfg.__dict__, name=exp_id)

    # 8 parallel environments
    n_envs = cfg.n_envs
    print(f"Creating {n_envs} parallel environments...")
    envs = gymnasium.vector.AsyncVectorEnv(
        [make_env(cfg.state, cfg.max_episode_steps) for _ in range(n_envs)]
    )

    obs_shape = envs.single_observation_space.shape  # (4, 52, 160)
    n_actions = envs.single_action_space.n
    print(f"Obs shape: {obs_shape}, Actions: {n_actions}, Envs: {n_envs}")

    agent = CNNPolicy(obs_shape, n_actions, hidden_dim=cfg.hidden_dim).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Rollout storage: (n_steps, n_envs, ...)
    obs_buf = np.zeros((cfg.n_steps, n_envs) + obs_shape, dtype=np.uint8)
    act_buf = np.zeros((cfg.n_steps, n_envs), dtype=np.int64)
    rew_buf = np.zeros((cfg.n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((cfg.n_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((cfg.n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((cfg.n_steps, n_envs), dtype=np.float32)

    global_step = 0
    total_episodes = 0
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    best_lap_time = float("inf")
    start_time = time.time()
    fps_counter_time = time.time()
    fps_counter_steps = 0

    obs, infos = envs.reset()
    print(f"Training for {cfg.training_minutes} minutes with {n_envs} envs...")
    print("---")

    while True:
        elapsed = time.time() - start_time
        if elapsed > cfg.training_minutes * 60:
            break

        for step in range(cfg.n_steps):
            obs_buf[step] = obs

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(device)
                actions, log_probs, _, values = agent.get_action_and_value(obs_tensor)

            act_np = actions.cpu().numpy()
            act_buf[step] = act_np
            logp_buf[step] = log_probs.cpu().numpy()
            val_buf[step] = values.cpu().numpy()

            obs, rewards, terminateds, truncateds, infos = envs.step(act_np)
            rew_buf[step] = rewards
            dones = np.logical_or(terminateds, truncateds)
            done_buf[step] = dones.astype(np.float32)

            global_step += n_envs
            fps_counter_steps += n_envs

            # Save live frames from each env (B/W cropped view)
            for i in range(n_envs):
                telem.save_live_frame(obs[i, 0], env_id=i)

            # Periodic history save (from env 0)
            if global_step % (cfg.frame_interval * n_envs) < n_envs:
                telem.save_frame(global_step, obs[0, 0])

            # Track episode completions (terminal info in _-prefixed keys)
            for i in range(n_envs):
                if dones[i]:
                    total_episodes += 1
                    ep_rew = infos.get("_episode_reward", rewards)[i]
                    ep_len = infos.get("_episode_step", np.zeros(n_envs))[i]
                    episode_rewards.append(float(ep_rew))
                    episode_lengths.append(float(ep_len))

                    t = infos.get("_time_seconds", np.zeros(n_envs))[i]
                    lap_num = infos.get("_lap_number", np.zeros(n_envs))[i]
                    if lap_num > 0 and t > 0:
                        avg_lap = t / lap_num
                        lap_times.append(avg_lap)
                        if avg_lap < best_lap_time:
                            best_lap_time = avg_lap

        # GAE (vectorized)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device)
            _, next_values = agent(obs_tensor)
            next_values = next_values.cpu().numpy()

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_values, cfg.gamma, cfg.gae_lambda
        )

        # Flatten (n_steps, n_envs) -> (n_steps * n_envs)
        total_steps = cfg.n_steps * n_envs
        b_obs = torch.from_numpy(obs_buf.reshape(total_steps, *obs_shape)).to(device)
        b_act = torch.from_numpy(act_buf.reshape(total_steps)).long().to(device)
        b_logp = torch.from_numpy(logp_buf.reshape(total_steps)).to(device)
        b_adv = torch.from_numpy(advantages.reshape(total_steps)).to(device)
        b_ret = torch.from_numpy(returns.reshape(total_steps)).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            indices = np.random.permutation(total_steps)
            for start in range(0, total_steps, cfg.batch_size):
                end = start + cfg.batch_size
                mb_idx = indices[start:end]

                _, new_logp, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_act[mb_idx]
                )

                ratio = torch.exp(new_logp - b_logp[mb_idx])
                pg_loss1 = -b_adv[mb_idx] * ratio
                pg_loss2 = -b_adv[mb_idx] * torch.clamp(
                    ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * ((new_value - b_ret[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        # Telemetry
        now = time.time()
        fps = fps_counter_steps / max(0.001, now - fps_counter_time)
        fps_counter_time = now
        fps_counter_steps = 0
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024

        metrics = {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "fps": fps,
            "total_episodes": total_episodes,
            "peak_memory_mb": mem_mb,
            "elapsed_seconds": elapsed,
            "n_envs": n_envs,
        }
        if episode_rewards:
            metrics["episode_reward"] = float(np.mean(episode_rewards[-20:]))
            metrics["episode_length"] = float(np.mean(episode_lengths[-20:]))
        if lap_times:
            metrics["avg_lap_time"] = float(np.mean(lap_times[-20:]))
            metrics["best_lap_time"] = best_lap_time
        telem.log_step(global_step, metrics)
        if use_wandb:
            _wandb.log(metrics, step=global_step)

    # Save checkpoint
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": cfg.__dict__,
    }, ckpt_path)

    # Final evaluation (single env)
    print("Running final evaluation...")
    eval_env = MarioKartEnv(state=cfg.state, max_episode_steps=cfg.max_episode_steps)
    eval_results = evaluate_agent(eval_env, agent, device, cfg.eval_episodes)
    eval_env.close()

    telem.close()
    if use_wandb:
        _wandb.finish()
    envs.close()

    elapsed_total = time.time() - start_time
    peak_mem = psutil.Process().memory_info().rss / 1024 / 1024

    print("---")
    print(f"avg_lap_time:      {eval_results['avg_lap_time']:.3f}")
    print(f"best_lap_time:     {eval_results['best_lap_time']:.3f}")
    print(f"tracks_wr_beaten:  {eval_results['tracks_wr_beaten']}")
    print(f"training_seconds:  {cfg.training_minutes * 60:.1f}")
    print(f"total_seconds:     {elapsed_total:.1f}")
    print(f"peak_memory_mb:    {peak_mem:.1f}")
    print(f"total_episodes:    {total_episodes}")
    print(f"total_steps_M:     {global_step / 1e6:.1f}")
    print(f"num_params_M:      {n_params / 1e6:.1f}")
    print(f"eval_tracks:       1")

    return eval_results


def evaluate_agent(env, agent, device, n_episodes=5):
    agent.eval()
    lap_times_list = []
    best_lap = float("inf")

    wr_file = Path(__file__).resolve().parent.parent.parent / "records.json"
    wr_data = {}
    if wr_file.exists():
        with open(wr_file) as f:
            wr_data = json.load(f).get("tracks", {})
    mario_circuit_wr = wr_data.get("mario_circuit_1", {}).get("avg_lap_wr", 11.174)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            done = terminated or truncated

        t = info.get("time_seconds", 0)
        lap_num = info.get("lap_number", 0)
        if lap_num > 0 and t > 0:
            avg_lap = t / lap_num
            lap_times_list.append(avg_lap)
            if avg_lap < best_lap:
                best_lap = avg_lap

    agent.train()
    avg_lap_time = float(np.mean(lap_times_list)) if lap_times_list else 999.0
    best_lap_time = best_lap if best_lap < float("inf") else 999.0
    tracks_wr_beaten = 1 if avg_lap_time < mario_circuit_wr else 0

    return {
        "avg_lap_time": avg_lap_time,
        "best_lap_time": best_lap_time,
        "tracks_wr_beaten": tracks_wr_beaten,
    }


if __name__ == "__main__":
    train()
