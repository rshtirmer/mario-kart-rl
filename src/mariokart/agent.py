"""CNN policy for PPO with separate policy/value heads."""

import torch
import torch.nn as nn
import numpy as np


def _make_cnn(c):
    return nn.Sequential(
        nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
    )


class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim=512):
        super().__init__()
        c, h, w = obs_shape

        # Shared CNN backbone
        self.cnn = _make_cnn(c)

        with torch.no_grad():
            cnn_out_size = int(np.prod(self.cnn(torch.zeros(1, c, h, w)).shape[1:]))

        # Separate heads with own FC layers
        self.policy_fc = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for final policy layer
        nn.init.orthogonal_(self.policy_fc[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_fc[-1].weight, gain=1.0)

    def forward(self, x):
        x = x.float() / 255.0
        features = self.cnn(x).reshape(x.size(0), -1)
        logits = self.policy_fc(features)
        value = self.value_fc(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
