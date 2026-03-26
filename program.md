# autoresearch-mariokart

Autonomous RL research loop for training a Mario Kart agent that beats world records.

## Overview

Unlike a typical autoresearch setup where the environment and eval harness are pre-built, this project starts from near-zero. The agent (you, Claude) must:

1. **Bootstrap phase**: Build the environment wrapper, evaluation harness, and initial training script from scratch.
2. **Research phase**: Iteratively improve the agent via the standard autoresearch experiment loop.
3. **Telemetry**: Emit lightweight data throughout training so the human can monitor progress via a dashboard — without slowing down training.

---

## Phase 0: Bootstrap

Before the experiment loop can begin, you need to build the project. All platform decisions are pre-made in `CLAUDE.md` — do not ask the human.

### 0a. Platform decisions (PRE-MADE)

These are final. Read `CLAUDE.md` for full details.

- **Game**: Super Mario Kart (SNES), USA/NTSC version
- **ROM**: `Super Mario Kart (USA).sfc` in repo root (gitignored)
- **Emulator**: `stable-retro` (maintained fork of gym-retro) with snes9x/libretro core
- **Reference integration**: https://github.com/esteveste/gym-SuperMarioKart-Snes — has working data.json, scenario.json, and Lua reward scripts. Use as starting point.
- **Starting track**: Mario Circuit 1 (track ID 0x07). Expand to more tracks once this works.
- **Character**: Koopa Troopa (best handling for learning)
- **CC class**: 100cc
- **Hardware**: Mac Mini M1, 16GB unified memory. PyTorch MPS backend (NOT CUDA). Fall back to CPU if MPS ops fail.
- **World records**: In `records.json`. Mario Circuit 1 best lap WR: 10.79s (KVD), 5-lap WR: 55.87s (Nori Ishi).

### 0a-1. Agent teams for bootstrap (optional)

If agent teams are enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`), parallelize bootstrap:
- **Teammate 1**: Build `env.py` (emulator wrapper — hardest part)
- **Teammate 2**: Build `agent.py`, `train.py`, `config.py`
- **Teammate 3**: Build `evaluate.py`, `telemetry.py`, `dashboard.py`

Require plan approval before teammates start implementing. The lead coordinates integration and runs end-to-end verification.

### 0b. Project structure

Create the following file structure:

```
autoresearch-mariokart/
├── pyproject.toml          # dependencies (torch, gymnasium, emulator bindings, etc.)
├── README.md
├── records.json            # world record times per track
├── env.py                  # environment wrapper (YOU BUILD THIS)
├── agent.py                # neural network architecture
├── train.py                # RL training loop
├── config.py               # hyperparameters
├── evaluate.py             # evaluation harness
├── telemetry.py            # lightweight logging for dashboard
├── dashboard.py            # standalone monitoring server (runs separately)
├── results.tsv             # experiment log (untracked by git)
├── runs/                   # telemetry output dir
│   └── <experiment_id>/
│       ├── metrics.jsonl   # streaming metrics (one JSON line per log interval)
│       ├── frames/         # sampled frames for video reconstruction
│       └── checkpoints/    # model checkpoints
└── dashboard/
    └── index.html          # static dashboard (or React artifact)
```

### 0c. Building the environment

Build `env.py` as a standard Gymnasium environment. It must:

- Wrap the emulator (headless mode — no display).
- Expose a standard `reset() -> obs, info` / `step(action) -> obs, reward, terminated, truncated, info` interface.
- Define the observation space (screen pixels, downsampled to something reasonable like 84x84 or 160x120).
- Define the action space (discrete: combinations of D-pad + A/B/L/R, or whatever the game needs).
- Implement a base reward function. Start simple:
  - `+10` per checkpoint crossed (address 4316 increments; total checkpoints at address 328)
  - `-1` per frame on surface 128 (wall), 32/40 (off-track), or 34 (deep water)
  - Large bonus for lap completion, scaled inversely by lap time
  - Small speed bonus for maintaining high speed (address 4330 > threshold)
  - Episode ends when race finishes (5 laps) or a timeout is hit, or too many wall hits
- Read game memory to extract: lap time, position on track, speed, lap count, race finished flag. All RAM addresses are documented in `CLAUDE.md`. Key ones:
  - Position: X=136, Y=140 (2-byte signed)
  - Speed: 4330 (2-byte signed magnitude)
  - Checkpoint: 4316 (progress), 328 (total per lap)
  - Lap: 4289 (subtract 128 for actual lap)
  - Timer: 256=ms, 258=sec, 260=min (BCD)
  - Surface: 4270 (64=road, 128=wall, 84=grass/offroad)
  - Wrong way: 267 (0x10 = backwards)

**This is the hardest part of the whole project.** Getting the emulator to run headless, reading RAM for reward signals, and mapping actions correctly is fiddly. The `esteveste/gym-SuperMarioKart-Snes` repo has a working integration — use its data.json and adapt its approach. The human may need to help debug emulator issues.

**Important for M1 Mac**: `stable-retro` works on Apple Silicon. If import fails, try `pip install gym-retro` as fallback. The ROM must be imported with `python -m retro.import .` from the directory containing the .sfc file.

### 0d. Building evaluation

Build `evaluate.py`:

- Loads a checkpoint, runs N episodes (e.g. 5) per track in the eval set.
- Records: lap times, race completion %, average speed.
- Compares best lap times against `records.json`.
- Prints the standard output format (see Output Format below).
- Deterministic: seeds the environment for reproducibility.

### 0e. Building telemetry

Build `telemetry.py` — a lightweight logger that training calls into. **Must not slow down training.**

Design principle: **emit raw data to disk, reconstruct later.** The dashboard reads files; training just appends.

```python
class Telemetry:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.metrics_file = open(self.run_dir / "metrics.jsonl", "a")
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

    def log_step(self, step: int, metrics: dict):
        """Append one JSON line. Called every N training steps (not every step)."""
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()

    def save_frame(self, step: int, frame: np.ndarray):
        """Save a single game frame as compressed PNG. Called infrequently."""
        # Save raw numpy array — dashboard reconstructs video from frame sequence
        np.savez_compressed(self.frames_dir / f"{step:010d}.npz", frame=frame)
```

**What gets logged** (every ~100 training steps — just dict serialization + file append):
- `episode_reward`, `episode_length`, `lap_time` (if episode finished)
- `loss`, `entropy`, `value_loss`, `policy_loss`
- `learning_rate`, `epsilon` (if applicable)
- `fps` (environment frames per second)
- `peak_memory_mb` (unified memory on M1, or VRAM on discrete GPU)

**What gets sampled** (every ~1000 steps or once per episode):
- Single game frame (compressed `.npz`) — the dashboard stitches these into timelapse videos.
- These are NOT full episode recordings. Just snapshots. Full recording would kill throughput.

**Optional (every ~5000 steps):**
- Run a short eval episode (1 lap) and save the full frame sequence for that one episode. This gives the dashboard a proper replay video at key checkpoints. Flag this in metrics as `eval_episode: true`.

### 0f. Building the dashboard

Build `dashboard.py` as a **standalone process** that runs separately from training. It watches the `runs/` directory and serves a web UI.

The dashboard should:
- **Auto-refresh**: Poll `metrics.jsonl` for new lines, update charts in real-time.
- **Show training curves**: reward over time, loss, lap times, FPS.
- **Show frame timelapse**: Reconstruct a video/animation from saved frames so the human can see what the agent is doing.
- **Show experiment history**: Read `results.tsv` and display a table of all experiments with status.
- **Show current vs world record**: Per-track comparison of agent's best time vs WR.
- **Lightweight**: A single Python file serving HTML is fine. The human starts it with `uv run dashboard.py` and opens `localhost:8080`.

### Data flow

```
Training process                    Dashboard process
─────────────────                   ─────────────────
train.py
  ├─ telemetry.log_step() ──────►  runs/<id>/metrics.jsonl  ◄──── dashboard.py reads
  ├─ telemetry.save_frame() ────►  runs/<id>/frames/*.npz   ◄──── dashboard.py reads
  └─ (end) writes run.log          results.tsv              ◄──── dashboard.py reads
```

### Dashboard features

1. **Live training curves** (updates every few seconds):
   - Episode reward (smoothed)
   - Lap time (if episodes are completing)
   - Loss components (policy, value, entropy)
   - FPS / throughput

2. **Frame viewer** (timelapse of saved snapshots):
   - Scrubber showing frames from current experiment
   - Auto-play at adjustable speed
   - Shows the agent's behavior evolving over training

3. **Eval replay** (when available):
   - Full episode recordings from periodic eval runs
   - Side-by-side: current best vs previous best

4. **Experiment history** (from results.tsv):
   - Table: commit, avg_lap_time, tracks_wr, status, description
   - Chart: avg_lap_time over experiments (only `keep` entries)
   - Color-coded: green=keep, red=crash, gray=discard

5. **World record tracker**:
   - Per-track: agent's best time vs WR target
   - Progress bar showing % of gap closed

### 0g. Verification

Before entering the experiment loop, verify the full pipeline end-to-end:

1. `env.py` — can reset and step, observations look right, reward signal is non-trivial.
2. `train.py` — runs for 1 minute without crashing, loss decreases.
3. `evaluate.py` — loads checkpoint, runs eval, prints metrics.
4. `telemetry.py` — writes `metrics.jsonl` and frame snapshots during training.
5. `dashboard.py` — serves and displays data from the test run.

Only after all 5 pass does the human confirm and you enter the experiment loop.

---

## Phase 1: Experiment loop

### Branch setup

1. **Agree on a run tag** (e.g. `mar26`). Must not already exist.
2. `git checkout -b autoresearch/<tag>` from master.
3. Initialize `results.tsv` with header row.
4. First run is always the unmodified baseline.

### What you CAN modify

- `train.py` — RL algorithm, training loop, batch construction, replay buffer, reward shaping (additive to base), curriculum, exploration schedule.
- `agent.py` — network architecture, state encoder, action head, auxiliary heads, input preprocessing.
- `config.py` — all hyperparameters.

### What you CANNOT modify

- `env.py` — observation space, action space, base reward, emulator interface. (If the env has a genuine bug, fix it on master and rebase — don't hack around it in train.py.)
- `evaluate.py` — the eval harness is the ground truth.
- `telemetry.py` — the logging interface is stable. You can call it differently (log more/less often) but don't change the format — the dashboard depends on it.
- `dashboard.py` — it reads the telemetry format; changing it breaks monitoring.
- `pyproject.toml` — no new dependencies.

### Training budget

Each experiment trains for a **fixed time budget of 30 minutes** (wall clock). Launch: `uv run train.py > run.log 2>&1`.

### Telemetry during training

Every experiment writes to `runs/<experiment_id>/` where `experiment_id` is the git short hash. Training must call into `telemetry.py`:

- `log_step()` every 100 training steps.
- `save_frame()` every 1000 steps.
- Optionally: one full eval-episode recording every 5000 steps.

This lets the human watch progress in the dashboard in near-real-time while they sleep.

### Output format

After training + eval, the script prints:

```
---
avg_lap_time:      42.350
best_lap_time:     39.820
tracks_wr_beaten:  0
training_seconds:  1800.2
total_seconds:     1860.5
peak_memory_mb:    12040.8
total_episodes:    48000
total_steps_M:     12.4
num_params_M:      3.2
eval_tracks:       4
track_results:
  mario_circuit:   41.200 (wr: 38.500)
  koopa_beach:     43.500 (wr: 40.100)
  ghost_valley:    42.100 (wr: 39.800)
  rainbow_road:    42.600 (wr: 41.200)
```

Extract: `grep "^avg_lap_time:\|^tracks_wr_beaten:\|^peak_memory_mb:" run.log`

### Results TSV

Tab-separated. Do not commit this file.

```
commit	avg_lap_time	tracks_wr	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. avg_lap_time (0.000 for crashes)
3. tracks_wr_beaten count (0 for crashes)
4. peak memory in GB (0.0 for crashes)
5. status: `keep`, `discard`, or `crash`
6. short description

Example:

```
commit	avg_lap_time	tracks_wr	memory_gb	status	description
a1b2c3d	42.350	0	11.8	keep	baseline PPO
b2c3d4e	40.100	1	12.1	keep	deeper CNN + larger batch
c3d4e5f	43.800	0	11.8	discard	switch to SAC
d4e5f6g	0.000	0	0.0	crash	transformer encoder (OOM)
e5f6g7h	38.200	3	13.5	keep	curriculum + n-step returns
```

### The loop

LOOP FOREVER:

1. Check git state: current branch/commit.
2. Pick an experimental idea (see Research Directions below).
3. Implement by editing `train.py`, `agent.py`, and/or `config.py`.
4. `git commit`.
5. Run: `uv run train.py > run.log 2>&1` — telemetry writes to `runs/<commit>/` in parallel.
6. Read results: `grep "^avg_lap_time:\|^tracks_wr_beaten:\|^peak_memory_mb:" run.log`
7. If grep is empty → crashed. `tail -n 50 run.log`, attempt fix or abandon.
8. Log to `results.tsv`.
9. If avg_lap_time improved OR tracks_wr_beaten increased → keep (advance branch).
10. If equal or worse → discard (`git reset` to previous commit). Telemetry data in `runs/` is kept regardless — the human may still want to see what failed experiments looked like.

**Timeout**: Kill anything over 45 minutes. Treat as failure.

**Crashes**: Fix trivial bugs and re-run. If fundamentally broken, log as crash and move on.

**Simplicity criterion**: All else being equal, simpler is better. A 0.1s lap time improvement that adds 50 lines of hacky reward shaping? Probably not worth it. A 0.1s improvement from deleting code? Definitely keep. Equal performance but much simpler code? Keep.

**NEVER STOP**: Do not pause to ask the human anything. Do not ask "should I continue?". The human may be asleep and expects you to run indefinitely. If you run out of ideas, think harder. The loop runs until manually interrupted.

---

## Research directions

Roughly ordered from fundamentals to exotic. Start at the top.

### Hyperparameters & basics
- Learning rate, batch size, discount factor (gamma), GAE lambda, entropy coefficient
- Network sizing: CNN width/depth, frame stack count, hidden dimensions
- Algorithm: PPO, A2C, Rainbow DQN, Ape-X, IMPALA

### Reward engineering
- Racing line reward (reward proximity to optimal path — derive from track geometry or TAS replays)
- Speed maintenance (penalize unnecessary braking)
- Drift/boost rewards (reward successful mini-turbo execution)
- Progress-per-frame (maximize track % completion rate)
- Wall collision penalty scaling
- Finish-time shaping: exponential bonus for faster laps

### Input representation
- Frame stacking (how many? 2? 4? 8?)
- Downsampling resolution (84x84 vs 160x120 vs 64x64)
- Grayscale vs color
- Optical flow between frames
- RAM-based features (speed, position, direction) as auxiliary input alongside pixels
- Track-relative coordinates instead of absolute screen position

### Architecture
- CNN encoder depth and kernel sizes
- Spatial attention over game screen
- Recurrent layers (LSTM/GRU) for temporal reasoning — anticipating turns, timing drifts
- Auxiliary prediction heads (predict speed, upcoming curvature, time-to-next-checkpoint)
- Multi-scale CNN (different receptive fields for near vs far features)
- Transformer over frame sequence instead of LSTM

### Curriculum & training dynamics
- Start on the easiest track, add harder tracks as performance improves
- Self-play against saved checkpoints (ghost racing)
- Demonstration learning: seed replay buffer with TAS/human replays if available
- Periodic full-eval checkpoints to catch overfitting to a single track

### Advanced RL
- Population-based training (if multiple agents feasible)
- Model-based RL (learn a dynamics model, plan ahead)
- Distributional RL (C51, QR-DQN)
- Hindsight experience replay adapted for racing
- Intrinsic motivation / curiosity for exploring new track sections

### Game-specific tricks (Super Mario Kart)
- **Coin collection priority**: +8 speed per coin, max 10 coins = +80 speed units. HUGE bonus. Reward coin collection early.
- **Mini-turbo mastery**: charge counter at RAM 4298 must reach 128+. Requires ~74 frames of holding L/R + direction + accelerate. Reward boost activation.
- **Infini-boost glitch**: hitting a zip arrow during mini-boost gives persistent elevated speed. Can be exploited.
- **Hop over offroad**: during a hop, kart uses the surface type from where the hop started. Can hop from road across offroad patches.
- **Mushroom boost**: bypasses offroad penalty entirely. Time mushrooms for offroad shortcuts.
- **Surface-aware driving**: offroad kills speed. Penalize time on surface types 84 (grass), 92 (water), 88 (snow).
- **Action repeat**: hold actions for 2-4 frames (the game runs at 60fps, decisions every frame is excessive).
- **RAM-based auxiliary inputs**: expose speed (4330), direction (149), surface (4270), coins (3584), checkpoint (4316), boost counter (4298) alongside pixels.
- **Wrong-way detection**: RAM address 267 = 0x10 means driving backwards. Heavy penalty.
- **Character choice**: Koopa/Toad have best handling (fastest turn response). Good for learning. Switch to Bowser/DK for top speed once agent is competent.

---

## Example timeline

Assuming ~35 minutes per experiment, overnight (8 hours) yields ~14 experiments.

| Hour | Experiments | What's happening |
|------|-------------|-----------------|
| 0    | 1           | Baseline PPO established |
| 1    | 2–3         | Hyperparameter tuning (LR, batch size) |
| 2    | 4–5         | Architecture changes (deeper CNN, frame stacking) |
| 3    | 6–7         | Reward shaping experiments |
| 4    | 8–9         | Curriculum learning, input representation |
| 5    | 10–11       | Combining best ideas |
| 6    | 12–13       | Advanced: recurrent layers, auxiliary heads |
| 7    | 14          | Exotic: distributional RL, model-based |

The human wakes up, opens the dashboard, sees 14 rows in results.tsv, watches timelapse videos of the agent improving, and checks how close they are to world records.
