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
        """Save a single game frame as compressed PNG via PIL."""
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(self.frames_dir / f"{step:010d}.png")

    def close(self):
        self.metrics_file.close()
