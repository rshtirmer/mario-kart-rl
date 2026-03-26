"""PPO training loop for Mario Kart RL with telemetry and evaluation."""

import time
import json
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch

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
            t = torch.zeros(1, device="mps")
            _ = t + 1
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        next_val = next_value if t == n_steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    returns = advantages + values
    return advantages, returns


def train():
    cfg = Config()
    device = get_device()
    print(f"Device: {device}")

    # Project root is 3 levels up from this file: src/mariokart/train.py -> project root
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

    env = MarioKartEnv(state=cfg.state, max_episode_steps=cfg.max_episode_steps)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"Obs shape: {obs_shape}, Actions: {n_actions}")

    agent = CNNPolicy(obs_shape, n_actions, hidden_dim=cfg.hidden_dim).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Rollout storage
    obs_buf = np.zeros((cfg.n_steps,) + obs_shape, dtype=np.uint8)
    act_buf = np.zeros(cfg.n_steps, dtype=np.int64)
    rew_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    done_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    logp_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    val_buf = np.zeros(cfg.n_steps, dtype=np.float32)

    global_step = 0
    total_episodes = 0
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    best_lap_time = float("inf")
    start_time = time.time()
    fps_counter_time = time.time()
    fps_counter_steps = 0

    obs, info = env.reset()
    print(f"Training for {cfg.training_minutes} minutes...")
    print("---")

    while True:
        elapsed = time.time() - start_time
        if elapsed > cfg.training_minutes * 60:
            break

        for step in range(cfg.n_steps):
            obs_buf[step] = obs

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)

            act_buf[step] = action.cpu().item()
            logp_buf[step] = log_prob.cpu().item()
            val_buf[step] = value.cpu().item()

            obs, reward, terminated, truncated, info = env.step(act_buf[step])
            rew_buf[step] = reward
            done = terminated or truncated
            done_buf[step] = float(done)

            global_step += 1
            fps_counter_steps += 1

            # Save B/W cropped frame (what agent sees) for dashboard
            # Latest frame always written for live view
            telem.save_live_frame(obs[0])  # first channel of frame stack
            # Periodic save for history replay
            if global_step % cfg.frame_interval == 0:
                telem.save_frame(global_step, obs[0])

            if done:
                total_episodes += 1
                ep_reward = info.get("episode_reward", 0)
                ep_length = info.get("episode_step", 0)
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)

                t = info.get("time_seconds", 0)
                lap_num = info.get("lap_number", 0)
                if lap_num > 0 and t > 0:
                    avg_lap = t / lap_num
                    lap_times.append(avg_lap)
                    if avg_lap < best_lap_time:
                        best_lap_time = avg_lap

                obs, info = env.reset()

        # GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            _, next_value = agent(obs_tensor)
            next_value = next_value.cpu().item()

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, cfg.gamma, cfg.gae_lambda
        )

        # PPO update
        b_obs = torch.from_numpy(obs_buf).to(device)
        b_act = torch.from_numpy(act_buf).long().to(device)
        b_logp = torch.from_numpy(logp_buf).to(device)
        b_adv = torch.from_numpy(advantages).to(device)
        b_ret = torch.from_numpy(returns).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            indices = np.random.permutation(cfg.n_steps)
            for start in range(0, cfg.n_steps, cfg.batch_size):
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
        if global_step % cfg.log_interval < cfg.n_steps:
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
            }
            if episode_rewards:
                metrics["episode_reward"] = float(np.mean(episode_rewards[-10:]))
                metrics["episode_length"] = float(np.mean(episode_lengths[-10:]))
            if lap_times:
                metrics["avg_lap_time"] = float(np.mean(lap_times[-10:]))
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

    # Final evaluation
    print("Running final evaluation...")
    eval_results = evaluate_agent(env, agent, device, cfg.eval_episodes)
    telem.close()
    if use_wandb:
        _wandb.finish()
    env.close()

    elapsed_total = time.time() - start_time
    process = psutil.Process()
    peak_mem = process.memory_info().rss / 1024 / 1024

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
