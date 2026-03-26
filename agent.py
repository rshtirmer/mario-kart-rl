"""CNN policy network for PPO -- takes stacked grayscale frames, outputs action logits + value."""

import torch
import torch.nn as nn
import numpy as np


class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim=512, cnn_channels=None):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        c, h, w = obs_shape  # (frame_stack, 84, 84)

        # CNN encoder
        layers = []
        in_channels = c
        for i, out_channels in enumerate(cnn_channels):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 8, stride=4))
            elif i == 1:
                layers.append(nn.Conv2d(in_channels, out_channels, 4, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.cnn(dummy)
            self.cnn_out_size = int(np.prod(cnn_out.shape[1:]))

        # Shared FC
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_size, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for output heads
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x):
        # x: (batch, frame_stack, H, W) uint8 -> float32 normalized
        x = x.float() / 255.0
        features = self.cnn(x)
        features = features.reshape(features.size(0), -1)
        hidden = self.fc(features)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        """Sample action (or evaluate given action) and return logprob, entropy, value."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
