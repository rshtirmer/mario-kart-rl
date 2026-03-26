# autoresearch-mariokart

Autonomous RL research loop that trains a Super Mario Kart (SNES) agent to beat world records -- powered by Claude Code and the Ralph Loop.

```
Human sleeps. Claude experiments. Agent improves. Dashboard updates.
Wake up to 14+ experiments and timelapse videos of a kart learning to race.
```

## How It Works

This project implements [Karpathy-style autoresearch](https://github.com/karpathy/autoresearch) for reinforcement learning, using the [Ralph Loop](https://ghuntley.com/ralph/) technique to drive Claude Code in a continuous experiment loop.

**Phase 0 - Bootstrap**: Claude builds the entire RL pipeline from scratch -- environment wrapper (gym-retro + SNES emulator), neural network, training loop, evaluation harness, telemetry, and a live dashboard.

**Phase 1 - Experiment Loop**: Claude iterates autonomously -- picks an idea, implements it, trains for 30 minutes, evaluates, keeps improvements or reverts failures, logs everything, and immediately starts the next experiment. No human intervention needed.

```
LOOP FOREVER:
  1. Pick experimental idea (architecture, reward shaping, hyperparams...)
  2. Implement changes to train.py / agent.py / config.py
  3. git commit
  4. Train for 30 min --> telemetry streams to dashboard
  5. Evaluate against world records
  6. Keep if improved, revert if not
  7. Log to results.tsv
  8. GOTO 1
```

## Target

- **Game**: Super Mario Kart (SNES, USA/NTSC)
- **Starting Track**: Mario Circuit 1
- **World Record**: Best lap 10.79s (KVD), 5-lap 55.87s (Nori Ishi)
- **Emulator**: stable-retro (gym-retro fork) with snes9x core
- **Hardware**: Tested on Mac Mini M1 16GB (PyTorch MPS)

## Prerequisites

- [Claude Code](https://claude.ai/claude-code) v2.1.32+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.11+
- Super Mario Kart (USA) ROM (.sfc file) -- you must provide this yourself
- [Ralph Loop plugin](https://github.com/anthropics/claude-code-plugins) for Claude Code

## Setup

1. Clone the repo:
```bash
git clone https://github.com/rshtirmer/mario-kart-rl.git
cd mario-kart-rl
```

2. Place your ROM file in the repo root:
```bash
# Must be named exactly:
Super Mario Kart (USA).sfc
```

3. (Optional) Enable agent teams for parallelized bootstrap:
```bash
# Add to Claude Code settings.json:
# {"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}}
```

## Kickoff

Launch the autonomous research loop with:

```bash
/ralph-loop "Read ralph-prompt.md and follow its instructions exactly. This is iteration $(cat results.tsv 2>/dev/null | wc -l) of an autonomous RL research loop. Read program.md for the full spec. Begin." --max-iterations 200
```

Then walk away. The first iteration bootstraps the entire pipeline (Phase 0). Subsequent iterations run the experiment loop (Phase 1).

## Monitoring

Once training starts, launch the dashboard in a separate terminal:

```bash
uv run dashboard.py
# Open http://localhost:8080
```

The dashboard shows:
- Live training curves (reward, loss, lap times, FPS)
- Frame timelapse of the agent learning to drive
- Experiment history from results.tsv
- World record progress per track

## Project Structure

```
mario-kart-rl/
+-- program.md              # Full spec (the "bible" -- what Claude follows)
+-- ralph-prompt.md         # Ralph Loop prompt (fed each iteration)
+-- CLAUDE.md               # Platform decisions, RAM addresses, hardware config
+-- records.json            # World record times per track
+-- pyproject.toml          # Dependencies (created during bootstrap)
+-- env.py                  # Gymnasium environment wrapping SNES emulator
+-- agent.py                # Neural network architecture
+-- train.py                # RL training loop
+-- config.py               # Hyperparameters
+-- evaluate.py             # Evaluation harness
+-- telemetry.py            # Lightweight metric logging
+-- dashboard.py            # Live monitoring web UI
+-- results.tsv             # Experiment log (gitignored)
+-- runs/                   # Telemetry data per experiment (gitignored)
```

## How the Ralph Loop Works

The [Ralph Loop](https://ghuntley.com/ralph/) feeds the same prompt to Claude Code repeatedly. Each iteration, Claude:

1. Reads ralph-prompt.md (always the same instructions)
2. Reads program.md (the full spec)
3. Reads CLAUDE.md (platform decisions, RAM addresses)
4. Checks results.tsv and git log (sees its own previous work)
5. Decides what to try next
6. Implements, trains, evaluates, logs
7. Loop restarts with the same prompt

The "self-referential" aspect: Claude sees its own git history, results.tsv, and code from previous iterations. Each iteration builds on the last.

## Research Directions

The agent explores (roughly in order):
1. Hyperparameter tuning (LR, batch size, gamma, entropy)
2. Architecture (CNN depth, frame stacking, recurrence)
3. Reward shaping (checkpoints, speed, coins, surface penalties)
4. Input representation (pixels, RAM features, resolution)
5. Game-specific tricks (mini-turbos, coin priority, hop mechanics)
6. Curriculum learning (easy tracks first)
7. Advanced RL (distributional, model-based, population-based)

## Key Technical Details

- **RAM addresses**: Full map of game memory for reward signals -- position, speed, checkpoints, lap times, surface type, coins, items. See CLAUDE.md.
- **Reward function**: Checkpoint-based (+10 per checkpoint) with surface penalties and speed bonuses.
- **Action space**: Discrete SNES controller (D-pad + A/B/L/R combinations).
- **Observation**: Downsampled game frames + optional RAM-based features.

## Acknowledgments

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the autonomous experiment loop concept
- [Geoffrey Huntley's Ralph technique](https://ghuntley.com/ralph/) for the iterative Claude loop
- [esteveste/gym-SuperMarioKart-Snes](https://github.com/esteveste/gym-SuperMarioKart-Snes) for the gym-retro integration reference
- [Data Crystal](https://datacrystal.tcrf.net/wiki/Super_Mario_Kart/RAM_map) for RAM mapping
- [mkwrs.com](https://mkwrs.com/smk/) for world record data
- [TASVideos](https://tasvideos.org/GameResources/SNES/SuperMarioKart) for game mechanics research
