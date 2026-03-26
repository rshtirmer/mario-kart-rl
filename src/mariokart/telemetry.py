"""Lightweight telemetry logger -- appends JSONL metrics and saves frame snapshots."""

import json
import time
from pathlib import Path

import numpy as np


class Telemetry:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = open(self.run_dir / "metrics.jsonl", "a")
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def log_step(self, step: int, metrics: dict):
        """Append one JSON line. Called every N training steps."""
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()

    def save_frame(self, step: int, frame: np.ndarray):
        """Save a B/W frame as PNG for history replay."""
        from PIL import Image
        img = Image.fromarray(frame, mode="L")
        img.save(self.frames_dir / f"{step:010d}.png")

    def save_live_frame(self, frame: np.ndarray):
        """Overwrite the live frame file (dashboard polls this)."""
        from PIL import Image
        img = Image.fromarray(frame, mode="L")
        # Write to temp then rename for atomic update
        tmp = self.run_dir / "live_frame.tmp.png"
        live = self.run_dir / "live_frame.png"
        img.save(tmp)
        tmp.rename(live)

    def close(self):
        self.metrics_file.close()
