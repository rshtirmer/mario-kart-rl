"""Hybrid Mario Kart environment: B/W cropped frames + RAM for rewards.
Uses RetroArch snes9x core on ARM64."""

import struct
import numpy as np
import gymnasium as gym
import stable_retro
import cv2
from pathlib import Path
from collections import deque

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTEGRATION_DIR = str(_PROJECT_ROOT / "retro_data")
GAME_NAME = "SuperMarioKart-Snes"
TOP_SPEED = 783.0

# Crop config (from crop tool)
CROP_TOP = 24
CROP_BOT = 107
CROP_LEFT = 0
CROP_RIGHT = 256
FRAME_W = 160  # wide to preserve 256:83 aspect ratio
FRAME_H = 52
FRAME_STACK = 4

# WRAM addresses
RAM = {
    "kart_x":       (136,  "<h"),
    "kart_y":       (140,  "<h"),
    "speed":        (4330, "<h"),
    "direction":    (149,  "B"),
    "checkpoint":   (4316, "B"),
    "lap_size":     (328,  "B"),
    "lap":          (4289, "B"),
    "coins":        (3584, "B"),
    "surface":      (4270, "B"),
    "wrong_way":    (267,  "B"),
    "game_mode":    (181,  "B"),
    "frame_ctr":    (56,   "<h"),
    "course":       (292,  "B"),
    "boost_ctr":    (4298, "<h"),
    "wall_hit":     (4178, "B"),
}

ACTIONS = [
    ["B"],                  # 0: accelerate
    ["B", "LEFT"],          # 1: accel + left
    ["B", "RIGHT"],         # 2: accel + right
    ["B", "L"],             # 3: hop
    ["B", "LEFT", "L"],     # 4: drift left
    ["B", "RIGHT", "L"],    # 5: drift right
    ["LEFT"],               # 6: coast + left
    ["RIGHT"],              # 7: coast + right
    [],                     # 8: coast
]

SURFACE_ROAD = 64
SURFACE_WALL = 128
SURFACE_OFFROAD = 84


def read_ram(mem, name):
    addr, fmt = RAM[name]
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, mem[addr:addr + size])[0]


class MarioKartEnv(gym.Env):
    """Mario Kart env with B/W cropped frame stack observations."""

    def __init__(self, state="MarioCircuit1", max_episode_steps=4500):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(FRAME_STACK, FRAME_H, FRAME_W),
            dtype=np.uint8,
        )
        self._state = state
        self._env = None
        self._buttons = None
        self._action_lut = None
        self._step_count = 0
        self._high_water = 0
        self._wall_steps = 0
        self._total_reward = 0.0
        self._frames = deque(maxlen=FRAME_STACK)
        self._max_episode_steps = max_episode_steps
        self._last_raw_frame = None

    def _make_env(self):
        stable_retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
        self._env = stable_retro.make(
            game=GAME_NAME, state=self._state,
            inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
            render_mode=None,
        )
        self._buttons = self._env.buttons
        self._action_lut = []
        for combo in ACTIONS:
            arr = np.zeros(len(self._buttons), dtype=np.int8)
            for b in combo:
                if b in self._buttons:
                    arr[self._buttons.index(b)] = 1
            self._action_lut.append(arr)

    def _preprocess_frame(self, screen):
        """Crop, grayscale, resize to 84x84."""
        cropped = screen[CROP_TOP:CROP_BOT, CROP_LEFT:CROP_RIGHT]
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized

    def _get_screen(self):
        """Get current screen from emulator."""
        return self._env.unwrapped.em.get_screen()

    def _read_info(self):
        mem = self._env.get_ram()
        info = {}
        for name in RAM:
            info[name] = read_ram(mem, name)
        info["lap_number"] = max(0, info["lap"] - 128)
        info["is_racing"] = info["game_mode"] == 0x1C
        return info

    def _get_obs(self):
        return np.array(self._frames, dtype=np.uint8)

    def _compute_reward(self, info):
        lap_size = max(info["lap_size"], 1)
        progress = info["checkpoint"] + info["lap_number"] * lap_size
        reward = 0.0

        delta = progress - self._high_water
        if delta > 0:
            reward += delta * 10.0
            self._high_water = progress
        elif delta < -100:
            self._high_water = progress

        speed = info["speed"]
        if speed > TOP_SPEED * 0.8:
            reward += 0.2
        elif speed > TOP_SPEED * 0.5:
            reward += 0.1

        if info["surface"] == SURFACE_WALL:
            reward -= 1.0
        if info["wrong_way"] == 0x10:
            reward -= 0.5

        return reward

    def _check_done(self, info):
        if not info["is_racing"]:
            return True
        if info["lap"] >= 133:
            return True
        if self._step_count > 50 and info["speed"] < 100:
            self._wall_steps += 1
            if self._wall_steps > 50:
                return True
        else:
            self._wall_steps = 0
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._env is None:
            self._make_env()
        self._env.reset()

        self._step_count = 0
        self._high_water = 0
        self._wall_steps = 0
        self._total_reward = 0.0
        self._frames.clear()

        # Step once to get initial state
        self._env.step(np.zeros(len(self._buttons), dtype=np.int8))
        info = self._read_info()
        screen = self._get_screen()
        self._last_raw_frame = screen

        frame = self._preprocess_frame(screen)
        for _ in range(FRAME_STACK):
            self._frames.append(frame)

        lap_size = max(info["lap_size"], 1)
        self._high_water = info["checkpoint"] + info["lap_number"] * lap_size

        return self._get_obs(), self._format_info(info)

    def step(self, action):
        buttons = self._action_lut[action]
        total_reward = 0.0
        terminated = False

        for _ in range(4):  # frame skip
            self._env.step(buttons)
            info = self._read_info()
            total_reward += self._compute_reward(info)
            if self._check_done(info):
                terminated = True
                break

        self._step_count += 1
        self._total_reward += total_reward

        screen = self._get_screen()
        self._last_raw_frame = screen
        self._frames.append(self._preprocess_frame(screen))

        truncated = self._step_count >= self._max_episode_steps
        return self._get_obs(), total_reward, terminated, truncated, self._format_info(info)

    def _format_info(self, info):
        info["episode_step"] = self._step_count
        info["episode_reward"] = self._total_reward
        info["time_seconds"] = self._step_count * 4 / 60.0
        return info

    def get_raw_frame(self):
        return self._last_raw_frame

    def close(self):
        if self._env is not None:
            self._env.close()
