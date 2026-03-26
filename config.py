"""Hyperparameters for Mario Kart RL training."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # Environment
    state: str = "MarioCircuit1"
    action_repeat: int = 4
    frame_stack: int = 4
    frame_size: tuple = (84, 84)
    max_episode_steps: int = 4500  # ~5 min at 4 frame skip

    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    n_steps: int = 128
    batch_size: int = 128
    n_epochs: int = 4
    training_minutes: int = 30

    # Network
    cnn_channels: list = field(default_factory=lambda: [32, 64, 64])
    hidden_dim: int = 512

    # Evaluation
    eval_episodes: int = 5

    # Telemetry
    log_interval: int = 100
    frame_interval: int = 1000
    eval_interval: int = 5000
