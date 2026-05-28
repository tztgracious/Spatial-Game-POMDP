# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2-player zero-sum POMDP modelling FPS game microstructure, solved via multi-agent self-play PPO. The key mechanic is **Line-of-Sight (LoS) discontinuity** — players can only observe each other when no wall intersects the line between them — which creates the partial observability that makes this a POMDP rather than a fully observable MDP.

## Planned Module Structure

The project is spec'd out in `GUIDE.md` but not yet implemented. The intended layout:

```
core/geometry.py          # Vector cross-product segment-intersection for LoS checks
env/arena.py              # Gymnasium/PettingZoo environment (step, reset, state/action spaces)
env/reward_shaping.py     # Reward functions + potential-based shaping + action penalties
agents/ppo_baseline.py    # PPO agent wired to stable-baselines3 or Ray RLlib
training/self_play_loop.py # Self-play pool with historical checkpoint sampling
utils/callbacks.py        # Curriculum learning scheduler (decay potential reward over timesteps)
analysis/value_heatmap.py # Critic V(s) heatmap across (x,y) grid
analysis/trajectory_plot.py # Trajectory plots at 10k/50k/500k training steps
```

## Key Design Decisions

**State space** `S ∈ ℝⁿ`: both agents' (x, y) positions, HP, and LoS boolean.

**Action space** (discrete): move up/down/left/right (step Δd), fire, stay.

**State transition** `P(s'|s,a)`: deterministic movement, probabilistic hit (when LoS=True and fire action).

**Reward structure**:
- Kill: +10, Death: -10, Time step: -0.01
- Potential-based guidance (decayed via curriculum): reward proportional to distance reduction toward the corner zone
- Blind-fire penalty: extra negative reward for firing when LoS=False

**Self-play pool**: current agent trains against both the latest opponent and random historical checkpoints (~20% probability) to prevent catastrophic forgetting.

## Performance Constraint

Single `env.step()` must complete in **< 1ms** — the LoS geometry check is the bottleneck. Implement `core/geometry.py` with pure NumPy vector math (no Python loops over wall segments if avoidable).

## Expected Dependencies

```
gymnasium
pettingzoo
stable-baselines3   # or ray[rllib]
torch
numpy
matplotlib
tensorboard
```

## Development Phases

See `GUIDE.md` for the full 5-phase plan. Phases in order: environment engine → self-play training loop → reward shaping → visualization/analysis.

---

## Git Workflow Rules

- **Never push directly to `main`.** All work happens on feature/phase branches.
- Branch naming: `phase-1-env-engine`, `phase-2-self-play`, etc.
- Every phase or logical unit of work ends with a PR to `main` describing what was done.
- PR titles follow: `[PhaseN] Short description of changes`

## Testing Rules

- Every new function or class gets unit tests in `tests/` mirroring the source path (e.g., `core/geometry.py` → `tests/test_geometry.py`).
- Integration tests go in `tests/test_integration.py` — they exercise multi-module flows end-to-end.
- Run tests: `pytest tests/ -v`
- Run a single test file: `pytest tests/test_geometry.py -v`
- Run a single test: `pytest tests/test_geometry.py::TestSegmentsIntersect::test_basic_cross -v`
- Performance regression test: `pytest tests/ -v -k "performance"` — enforces the < 1ms step budget.

## Project Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the package in editable mode for clean imports
```
