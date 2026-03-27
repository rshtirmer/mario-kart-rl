"""Hyperparameters for Mario Kart RL training."""

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    state: str = "MarioCircuit1.TimeTrialMario"
    max_episode_steps: int = 2000  # tighter deadline forces faster completion

    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.05
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    n_envs: int = 8
    n_steps: int = 128  # per env, total = 128*8 = 1024 per rollout
    batch_size: int = 128
    n_epochs: int = 3
    training_minutes: int = 30

    # Network
    hidden_dim: int = 256

    # Evaluation
    eval_episodes: int = 10

    # Telemetry
    log_interval: int = 100
    frame_interval: int = 25  # ~1.7s between frames at 118 FPS for smooth video

    # Weights & Biases
    use_wandb: bool = True
