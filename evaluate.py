"""Evaluation harness -- loads checkpoint, runs episodes, reports metrics vs world records."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from env import MarioKartEnv
from agent import CNNPolicy


def get_device():
    if torch.backends.mps.is_available():
        try:
            t = torch.zeros(1, device="mps")
            _ = t + 1
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def evaluate(checkpoint_path, n_episodes=5, seed=42):
    cfg = Config()
    device = get_device()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore config if available
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Environment
    env = MarioKartEnv(
        state=cfg.state,
        action_repeat=cfg.action_repeat,
        frame_stack=cfg.frame_stack,
        frame_size=cfg.frame_size,
        max_episode_steps=cfg.max_episode_steps,
    )

    # Agent
    agent = CNNPolicy(
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        hidden_dim=cfg.hidden_dim,
        cnn_channels=cfg.cnn_channels,
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    # Load world records
    wr_file = Path(__file__).parent / "records.json"
    with open(wr_file) as f:
        wr_data = json.load(f)["tracks"]

    mario_circuit_wr = wr_data["mario_circuit_1"]["avg_lap_wr"]

    # Run evaluation
    lap_times = []
    completion_pcts = []
    speeds = []

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        obs, info = env.reset()
        done = False
        ep_speeds = []

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            done = terminated or truncated
            ep_speeds.append(info.get("speed", 0))

        t = info.get("time_seconds", 0)
        lap_num = info.get("lap_number", 0)
        total_cp = info.get("total_checkpoints", 1)
        cp = info.get("checkpoint", 0)

        if lap_num > 0 and t > 0:
            lap_times.append(t / lap_num)
        completion_pcts.append((cp + lap_num * total_cp) / max(1, 5 * total_cp) * 100)
        if ep_speeds:
            speeds.append(float(np.mean(ep_speeds)))

    env.close()

    avg_lap = float(np.mean(lap_times)) if lap_times else 999.0
    best_lap = float(min(lap_times)) if lap_times else 999.0
    avg_completion = float(np.mean(completion_pcts))
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    wr_beaten = 1 if avg_lap < mario_circuit_wr else 0

    print("---")
    print(f"avg_lap_time:      {avg_lap:.3f}")
    print(f"best_lap_time:     {best_lap:.3f}")
    print(f"tracks_wr_beaten:  {wr_beaten}")
    print(f"avg_completion_%:  {avg_completion:.1f}")
    print(f"avg_speed:         {avg_speed:.0f}")
    print(f"eval_episodes:     {n_episodes}")
    print(f"track_results:")
    print(f"  mario_circuit_1: {avg_lap:.3f} (wr: {mario_circuit_wr:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run evaluate.py <checkpoint_path> [n_episodes]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    evaluate(ckpt_path, n_eps)
