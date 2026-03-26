# 🏎️ Mario Kart RL — Autonomous Research Loop

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C) ![RL](https://img.shields.io/badge/RL-PPO-green) ![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-lightgrey) ![Status](https://img.shields.io/badge/Status-Bootstrapping-yellow)

Autonomous RL research loop that trains an AI to play Super Mario Kart (SNES) and beat world records. Runs as a continuous experiment loop -- pick an idea, train for 30 min, evaluate, keep or discard, repeat.

---

## Key Features

- **Fully autonomous** -- runs overnight without human intervention
- **Real-time dashboard** at `localhost:8080` with live training curves and frame timelapse
- **PPO with CNN policy** on Apple Silicon (MPS backend)
- **Automatic experiment tracking** in `results.tsv` with full git history
- **Frame-by-frame telemetry** for replay analysis and debugging

---

## Quick Start

```bash
# Install dependencies
uv sync

# Import ROM (place Super Mario Kart (USA).sfc in repo root)
cp "Super Mario Kart (USA).sfc" retro_data/SuperMarioKart-Snes/rom.sfc

# Start training
uv run python train.py > run.log 2>&1 &

# Monitor progress
uv run python dashboard.py
```

Then open [http://localhost:8080](http://localhost:8080) to watch the agent learn.

---

## Architecture

| File | Description |
|------|-------------|
| `env.py` | Gymnasium environment wrapping the SNES emulator via stable-retro |
| `agent.py` | CNN policy network and PPO algorithm implementation |
| `train.py` | Main training loop with rollout collection and optimization |
| `config.py` | Hyperparameters, action space, and reward shaping configuration |
| `evaluate.py` | Evaluation harness -- runs the agent and measures lap times |
| `telemetry.py` | Lightweight metric logger streaming data to the dashboard |
| `dashboard.py` | Live web UI showing training curves, frames, and experiment history |

Supporting files:

| File | Description |
|------|-------------|
| `program.md` | Full project spec -- the "bible" that Claude follows each iteration |
| `ralph-prompt.md` | Ralph Loop prompt fed to Claude Code on every iteration |
| `records.json` | World record lap times per track (target to beat) |
| `results.tsv` | Experiment log with metrics from every run |

---

## World Records (Target)

Human world records for Super Mario Kart (SNES, NTSC Time Trial). The goal is to match or beat these times.

| Track | Best Lap | Holder | 5-Lap Time | Holder |
|-------|----------|--------|------------|--------|
| Mario Circuit 1 | 10.79s | KVD | 55.87s | Nori Ishi |
| Donut Plains 1 | 13.16s | ScouB | 67.54s | Firewaster |
| Ghost Valley 1 | 11.63s | KVD | 59.09s | KVD |
| Bowser Castle 1 | 16.85s | KVD | 85.00s | KVD |
| Mario Circuit 2 | 11.12s | ScouB | 59.92s | Sami |

Source: [mkwrs.com/smk](https://mkwrs.com/smk/)

---

## How It Works

This project implements autonomous RL research using the [Ralph Loop](https://ghuntley.com/ralph/) technique to drive Claude Code in a continuous experiment loop.

```
LOOP FOREVER:
  1. Pick an experimental idea (architecture, reward shaping, hyperparams)
  2. Implement changes to the codebase
  3. git commit
  4. Train for 30 minutes with telemetry streaming to the dashboard
  5. Evaluate against world record lap times
  6. Keep if improved, revert if not
  7. Log results to results.tsv
  8. GOTO 1
```

**Phase 0 -- Bootstrap**: Claude builds the entire RL pipeline from scratch -- environment wrapper, neural network, training loop, evaluation harness, telemetry, and dashboard.

**Phase 1 -- Experiment Loop**: Claude iterates autonomously, exploring hyperparameter tuning, architecture changes, reward shaping, input representations, and game-specific tricks like mini-turbos and coin collection.

The loop is self-referential: each iteration reads its own git history, `results.tsv`, and previous code to decide what to try next. No human steering required.

---

## Acknowledgments

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) -- autonomous experiment loop concept
- [Geoffrey Huntley's Ralph technique](https://ghuntley.com/ralph/) -- iterative Claude loop
- [esteveste/gym-SuperMarioKart-Snes](https://github.com/esteveste/gym-SuperMarioKart-Snes) -- gym-retro integration reference
- [Data Crystal](https://datacrystal.tcrf.net/wiki/Super_Mario_Kart/RAM_map) -- RAM mapping
- [mkwrs.com](https://mkwrs.com/smk/) -- world record data
- [TASVideos](https://tasvideos.org/GameResources/SNES/SuperMarioKart) -- game mechanics research
