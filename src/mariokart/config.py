"""Hyperparameters for Mario Kart RL training."""

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    state: str = "MarioCircuit1.TimeTrialMario"
    max_episode_steps: int = 4500

    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.03
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 3
    training_minutes: int = 30

    # Network
    hidden_dim: int = 256

    # Evaluation
    eval_episodes: int = 5

    # Telemetry
    log_interval: int = 100
    frame_interval: int = 1000

    # Weights & Biases
    use_wandb: bool = True
