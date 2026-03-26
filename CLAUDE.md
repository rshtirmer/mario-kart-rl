# Mario Kart RL - Autoresearch

## Platform Decisions (DO NOT ASK THE HUMAN - these are final)

- Game: Super Mario Kart (SNES), USA version (NTSC)
- ROM: `Super Mario Kart (USA).sfc` in repo root (gitignored)
- Emulator: `stable-retro` (maintained fork of gym-retro) with snes9x/libretro core
- Reference integration: https://github.com/esteveste/gym-SuperMarioKart-Snes (data.json, scenario.json, reward scripts)
- Starting track: Mario Circuit 1 (track ID 0x07)
- Starting character: Koopa Troopa (best handling, good for learning)
- CC class: 100cc (good balance of speed and learnable difficulty)
- Python: 3.11 via uv

## Hardware Constraints

- Mac Mini M1, 16GB unified memory
- PyTorch MPS backend (NOT CUDA) -- use `device = "mps" if torch.backends.mps.is_available() else "cpu"`
- Some MPS ops may not be supported -- fall back to CPU for those, don't crash
- No discrete GPU, no VRAM -- `peak_vram_mb` in output means peak memory usage (use `psutil` or `resource` module)
- Training budget may need to be shorter than 30 min if throughput is low on M1 -- adjust in config.py

## Key RAM Addresses (gym-retro decimal offsets)

```
Position:     X=136 (2B signed), Y=140 (2B signed)
Speed:        4330 (2B signed, overall magnitude)
Direction:    149 (1B unsigned, camera angle: 0=N, 64=E, 128=S, 192=W)
Checkpoint:   4316 (1B, progress through track)
Total CPs:    328 (1B, checkpoints per lap)
Lap:          4289 (1B, subtract 128 for actual lap number)
Coins:        3584 (1B, 0-10)
Surface:      4270 (1B, see surface table below)
Rank:         4160 (1B, current race position)
Item:         3440 (1B, item held)
Timer:        256=ms, 258=sec, 260=min (2B BCD each)
Course:       292 (1B, track ID)
Game Mode:    54 (1B)
Ext Mode:     181 (1B, 0x1C = racing)
Wrong Way:    267 (0x10 = driving backwards)
Boost Ctr:    4298 (2B, mini-turbo charge, need 128+ to fire)
Mushroom:     4174 (58=active, 26=inactive)
Wall Hit:     4178 (0=none, 7=collision)
Lakitu:       7224 (rescue timer)
```

## Surface Types

```
64  = Road (normal)
68  = Bowser Castle road
70  = Donut Plains track
84  = Off-road (grass)
92  = Shallow water
128 = Wall collision
32  = Off-track / void (fall off)
34  = Deep water
```

## Track IDs

```
0x07 = Mario Circuit 1      0x0F = Mario Circuit 2
0x10 = Ghost Valley 1       0x13 = Donut Plains 1
0x11 = Bowser Castle 1      0x0D = Koopa Beach 1
0x05 = Rainbow Road
```

## World Records (Time Trial, NTSC)

Mario Circuit 1: 5-lap 0'55"87 (Nori Ishi), best lap 0'10"79 (KVD)
Donut Plains 1:  5-lap 1'07"54 (Firewaster), best lap 0'13"16 (ScouB)
Ghost Valley 1:  5-lap 0'59"09 (KVD), best lap 0'11"63 (KVD)
Bowser Castle 1: 5-lap 1'25"00 (KVD), best lap 0'16"85 (KVD)
Mario Circuit 2: 5-lap 0'59"92 (Sami), best lap 0'11"12 (ScouB)

## Game Mechanics Notes (for reward shaping)

- Coins: +8 speed per coin to top speed (max 10 coins = +80 speed units). Huge bonus.
- Mini-turbo: charge at $10CA must reach 128+. Requires holding L/R + direction + accelerate for ~74+ frames.
- Mushroom: sets 4174 to 58, allows full speed on offroad.
- Hopping: kart uses the surface from where hop started (can hop from road over offroad).
- Skidding does NOT slow the kart down.
- Wall collision detected at 4178 (7=collision). Costs time but doesn't always reduce speed.
- Infini-boost glitch: hitting zip arrow during mini-boost gives persistent elevated speed.
- Off-road: reduces top speed. Letting off gas on offroad actually slows deceleration less than holding gas.

## Agent Teams

Agent teams are available for parallelizing bootstrap work. Enable with:
```json
{"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}}
```

During Phase 0 bootstrap, consider using agent teams to parallelize:
- Teammate 1: env.py (emulator wrapper, hardest part)
- Teammate 2: agent.py + train.py + config.py
- Teammate 3: evaluate.py + telemetry.py + dashboard.py

During Phase 1, agent teams can run competing experiments in parallel using git worktrees.

## Dependencies (for pyproject.toml)

Core: torch, gymnasium, stable-retro (or gym-retro), numpy, Pillow
Optional: psutil (memory tracking), opencv-python-headless (frame processing)
Dashboard: aiohttp or flask (lightweight web server)
