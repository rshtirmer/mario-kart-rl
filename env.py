"""RAM-based Mario Kart environment using direct memory reads.
Uses RetroArch snes9x core on ARM64 — reads WRAM directly for observations."""

import struct
import numpy as np
import gymnasium as gym
import stable_retro
from pathlib import Path
from collections import deque

INTEGRATION_DIR = str(Path(__file__).parent.resolve() / "retro_data")
GAME_NAME = "SuperMarioKart-Snes"
TOP_SPEED = 783.0

# WRAM addresses (offsets into 128KB SNES WRAM)
RAM = {
    "kart_x":       (136,  "<h"),   # signed 16-bit
    "kart_y":       (140,  "<h"),
    "speed":        (4330, "<h"),
    "direction":    (149,  "B"),     # unsigned 8-bit
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
HISTORY_LEN = 4


def read_ram(mem, name):
    """Read a value from WRAM by name."""
    addr, fmt = RAM[name]
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, mem[addr:addr + size])[0]


class MarioKartEnv(gym.Env):
    """Super Mario Kart env with RAM observations (32-dim vector)."""

    OBS_SIZE = 12 + HISTORY_LEN * 5

    def __init__(self, state="MarioCircuit1", max_episode_steps=4500):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        self._state = state
        self._env = None
        self._buttons = None
        self._action_lut = None
        self._step_count = 0
        self._high_water = 0
        self._wall_steps = 0
        self._total_reward = 0.0
        self._history = deque(maxlen=HISTORY_LEN)
        self._max_episode_steps = max_episode_steps

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

    def _read_info(self):
        """Read all game state from WRAM."""
        mem = self._env.get_ram()
        info = {}
        for name in RAM:
            info[name] = read_ram(mem, name)
        info["lap_number"] = max(0, info["lap"] - 128)
        info["is_racing"] = info["game_mode"] == 0x1C
        return info

    def _get_obs(self, info):
        """Build 32-dim observation vector."""
        x = info["kart_x"] / 4096.0
        y = info["kart_y"] / 4096.0
        d = info["direction"]
        dir_rad = d * 2 * np.pi / 256.0
        dir_sin, dir_cos = np.sin(dir_rad), np.cos(dir_rad)
        speed = info["speed"] / 1000.0

        lap_size = max(info["lap_size"], 1)
        checkpoint = info["checkpoint"] / lap_size
        lap = info["lap_number"]
        surface = info["surface"]
        on_road = 1.0 if surface == SURFACE_ROAD else 0.0
        on_wall = 1.0 if surface == SURFACE_WALL else 0.0
        on_offroad = 1.0 if surface == SURFACE_OFFROAD else 0.0
        turned = 1.0 if info["wrong_way"] == 0x10 else 0.0
        coins = info["coins"] / 10.0

        current = np.array([
            speed, dir_sin, dir_cos, checkpoint,
            on_road, on_wall, on_offroad, turned,
            x, y, lap / 5.0, coins
        ], dtype=np.float32)

        self._history.append((x, y, dir_sin, dir_cos, speed))
        hist = np.zeros(HISTORY_LEN * 5, dtype=np.float32)
        for i, h in enumerate(self._history):
            hist[i * 5:(i + 1) * 5] = h

        return np.clip(np.concatenate([current, hist]), -2.0, 2.0)

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
        self._history.clear()
        for _ in range(HISTORY_LEN):
            self._history.append((0, 0, 0, 0, 0))

        # Step once to get initial state
        self._env.step(np.zeros(len(self._buttons), dtype=np.int8))
        info = self._read_info()

        lap_size = max(info["lap_size"], 1)
        self._high_water = info["checkpoint"] + info["lap_number"] * lap_size

        return self._get_obs(info), self._format_info(info)

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
        truncated = self._step_count >= self._max_episode_steps

        return self._get_obs(info), total_reward, terminated, truncated, self._format_info(info)

    def _format_info(self, info):
        info["episode_step"] = self._step_count
        info["episode_reward"] = self._total_reward
        info["time_seconds"] = self._step_count * 4 / 60.0
        return info

    def get_raw_frame(self):
        """Return rendered frame for telemetry (works with RetroArch core)."""
        try:
            return self._env.unwrapped.em.get_screen()
        except Exception:
            return None

    def close(self):
        if self._env is not None:
            self._env.close()
