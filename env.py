"""Mario Kart environment wrapper for stable-retro with Gymnasium interface."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Disable pyglet shadow window before any retro imports (headless M1 Mac)
try:
    import pyglet
    pyglet.options["shadow_window"] = False
except ImportError:
    pass

import gymnasium as gym
import numpy as np
import retro
from pathlib import Path
from collections import deque

INTEGRATION_DIR = str(Path(__file__).parent.resolve() / "retro_data")
GAME_NAME = "SuperMarioKart-Snes"

# SNES button order: B Y SELECT START UP DOWN LEFT RIGHT A X L R
BUTTON_NAMES = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

# Discrete action table: each row is a 12-element button array
# 0: accelerate, 1: accel+left, 2: accel+right, 3: accel+hop,
# 4: accel+left+hop, 5: accel+right+hop, 6: coast (nothing)
ACTION_TABLE = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # B + LEFT
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # B + RIGHT
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # B + L (hop)
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # B + LEFT + L (drift left)
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # B + RIGHT + L (drift right)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nothing (coast)
], dtype=np.int8)

# Surface types
SURFACE_ROAD = 64
SURFACE_WALL = 128
SURFACE_OFFROAD = 84
SURFACE_WATER = 92
SURFACE_VOID = 32
SURFACE_DEEP_WATER = 34


class MarioKartEnv(gym.Env):
    """Gymnasium wrapper around stable-retro Super Mario Kart."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        state="MarioCircuit1",
        action_repeat=4,
        frame_stack=4,
        frame_size=(84, 84),
        max_episode_steps=4500,
        render_mode=None,
    ):
        super().__init__()
        self.action_repeat = action_repeat
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Register custom integration path
        retro.data.Integrations.add_custom_path(INTEGRATION_DIR)

        # Create retro env (rgb_array for headless rendering)
        self._env = retro.make(
            GAME_NAME,
            state=state,
            inttype=retro.data.Integrations.CUSTOM,
            render_mode="rgb_array",
        )

        # Spaces
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, frame_size[0], frame_size[1]),
            dtype=np.uint8,
        )

        # Frame buffer for stacking
        self._frames = deque(maxlen=frame_stack)

        # State tracking for reward
        self._prev_checkpoint = 0
        self._prev_lap = 0
        self._prev_coins = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._last_raw_frame = None

    def _preprocess_frame(self, obs):
        """Convert RGB frame to grayscale and resize to frame_size."""
        import cv2
        # Crop bottom minimap (keep top 110 rows of ~224)
        obs = obs[:110, :, :]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]),
                             interpolation=cv2.INTER_AREA)
        return resized

    def _get_stacked_obs(self):
        return np.array(self._frames, dtype=np.uint8)

    def _compute_reward(self, info):
        """Compute shaped reward from RAM values."""
        reward = 0.0

        # Checkpoint progress (main signal)
        lap = max(0, info.get("lap", 128) - 128)
        total_cp = max(1, info.get("total_checkpoints", 1))
        checkpoint = info.get("checkpoint", 0)
        progress = checkpoint + lap * total_cp

        delta_progress = progress - self._prev_checkpoint
        if delta_progress < -100:
            delta_progress = 0  # state reset
        reward += delta_progress * 10.0
        self._prev_checkpoint = progress

        # Speed bonus (encourage going fast)
        speed = info.get("speed", 0)
        if speed > 800:
            reward += 0.1
        elif speed > 500:
            reward += 0.05

        # Surface penalty
        surface = info.get("surface", SURFACE_ROAD)
        if surface == SURFACE_WALL:
            reward -= 1.0
        elif surface in (SURFACE_VOID, SURFACE_DEEP_WATER):
            reward -= 2.0
        elif surface == SURFACE_OFFROAD:
            reward -= 0.3

        # Wrong way penalty
        if info.get("wrong_way", 0) == 0x10:
            reward -= 1.0

        # Coin collection bonus
        coins = info.get("coins", 0)
        if coins > self._prev_coins:
            reward += (coins - self._prev_coins) * 2.0
        self._prev_coins = coins

        # Lap completion bonus (big reward scaled inversely by time)
        new_lap = max(0, info.get("lap", 128) - 128)
        if new_lap > self._prev_lap and self._prev_lap > 0:
            reward += 100.0
        self._prev_lap = new_lap

        return reward

    def _extract_info(self, retro_info):
        """Extract structured info from retro info dict."""
        info = dict(retro_info)
        info["lap_number"] = max(0, info.get("lap", 128) - 128)
        info["is_racing"] = info.get("ext_mode", 0) == 0x1C

        # Compute time from frame count (60fps, each step = action_repeat frames)
        # More reliable than BCD timer which has address issues on this ROM
        info["time_seconds"] = self._step_count * self.action_repeat / 60.0

        return info

    def _retro_reset(self):
        """Handle both old gym and new gymnasium API from retro."""
        result = self._env.reset()
        if isinstance(result, tuple):
            return result[0]  # (obs, info) -> obs
        return result

    def _retro_step(self, action):
        """Handle both 4-tuple and 5-tuple step returns."""
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, term, trunc, info = result
            return obs, reward, term or trunc, info
        return result  # (obs, reward, done, info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._retro_reset()

        self._prev_checkpoint = 0
        self._prev_lap = 0
        self._prev_coins = 0
        self._step_count = 0
        self._total_reward = 0.0

        frame = self._preprocess_frame(obs)
        self._last_raw_frame = obs
        for _ in range(self.frame_stack):
            self._frames.append(frame)

        # Get initial info by stepping once with no-op
        obs2, _, _, info = self._retro_step(np.zeros(12, dtype=np.int8))
        frame2 = self._preprocess_frame(obs2)
        self._frames.append(frame2)
        self._last_raw_frame = obs2

        return self._get_stacked_obs(), self._extract_info(info)

    def step(self, action):
        buttons = ACTION_TABLE[action]

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.action_repeat):
            obs, retro_reward, done, retro_info = self._retro_step(buttons)
            info = self._extract_info(retro_info)
            reward = self._compute_reward(info)
            total_reward += reward

            if done:
                terminated = True
                break

        self._step_count += 1
        self._total_reward += total_reward
        self._last_raw_frame = obs

        frame = self._preprocess_frame(obs)
        self._frames.append(frame)

        if self._step_count >= self.max_episode_steps:
            truncated = True

        info["episode_step"] = self._step_count
        info["episode_reward"] = self._total_reward

        return self._get_stacked_obs(), total_reward, terminated, truncated, info

    def get_raw_frame(self):
        """Return last raw RGB frame (for telemetry)."""
        return self._last_raw_frame

    def close(self):
        self._env.close()
