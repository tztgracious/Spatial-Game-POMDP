# Spatial-Game-POMDP

A 2-player zero-sum POMDP modelling FPS game microstructure, solved via multi-agent self-play PPO. The key mechanic is **Line-of-Sight (LoS) discontinuity** — agents only observe each other when no wall intersects their connecting segment — which is what makes this a POMDP rather than a fully-observable MDP.

For the full design rationale see [`GUIDE.md`](GUIDE.md); for tooling conventions see [`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest tests/                            # 192 tests
python -m training.train_phase3          # full 500k-step self-play run (~3 min)
python -m analysis.value_heatmap         # plot V(s) for the trained critic
python -m analysis.trajectory_plot       # plot policy progression across checkpoints
```

## Formal definition

### State

Each step exposes two 7-element observation blocks (one per agent). For agent $i$ with opponent $j = 1 - i$:

$$o_i = ( x_i, y_i, h_i / H_{\max}, \mathbf{1}_{\mathrm{LoS}}, \tilde{x}_j, \tilde{y}_j, \tilde{h}_j / H_{\max} )$$

where $(\tilde{x}_j, \tilde{y}_j, \tilde{h}_j) = (x_j, y_j, h_j)$ when LoS is open and $(0, 0, 0)$ when blocked. This masking is what makes the problem a POMDP — each agent's observation is a *partial* view of the underlying world state.

### Action

Per agent, $a \in \{0, 1, 2, 3, 4, 5\}$ — stay / up / down / left / right / fire. Both agents act simultaneously each step (`MultiDiscrete([6, 6])`).

### Transition

The transition kernel $P(s' \mid s, a)$ has two parts:

- **Movement** is deterministic: positions translate by $\Delta d = 0.5$ in the chosen direction, clipped to arena bounds.
- **Combat** is probabilistic. When agent $i$ fires with LoS open, the hit probability uses the **Option C 近大远小** formula:

$$P_{\mathrm{hit}} = \exp\left( -\frac{\| p_j - p_i \|}{\sigma} \right) \cdot \max\left( \epsilon_{\min}, \frac{\arctan\left( d_{\mathrm{peek}} / \| c - p_i \| \right)}{\pi / 2} \right)$$

where $c$ is the **active corner** (the wall endpoint inside the shooter–target sightline) and $d_{\mathrm{peek}}$ is the target's perpendicular distance to the shadow ray $p_i \to c$. Smaller peek → smaller exposure → smaller hit probability. Computed in `core/geometry.los_and_active_corner` and `core/geometry.hit_probability`.

### Reward

For each agent, per step:

$$r = R_k \cdot \mathbf{1}_{\mathrm{kill}} - R_t - R_b \cdot \mathbf{1}_{\mathrm{blind\text{-}fire}} + w \cdot ( \gamma \Phi(s') - \Phi(s) )$$

where the blind-fire indicator fires when the agent shoots without LoS, and $\Phi(s) = -\| p - z \|$ is the **potential function** — the negative distance to a chosen target zone $z$ (typically a wall corner). The shaping term is **Ng–Russell–Russell potential-based**, so adding it preserves the set of optimal policies. The weight $w$ is decayed linearly to zero by `CurriculumCallback`, so the final policy is optimal under the pure zero-sum reward.

## Results

A 500k-step run (50 self-play generations, curriculum 1.0 → 0.0):

| Plot | Insight |
|---|---|
| `runs/phase3/value_heatmap.png` | Critic's V(s) shows bright ridges at the wall corners — the policy *induced* the LoS topology from rewards alone, without being told about walls explicitly |
| `runs/phase3/trajectories.png` | Side-by-side 10k / 50k / 500k step checkpoints: random walk → directed → deliberate corner peeking |

## Pitfalls

Non-obvious mistakes encountered during development. Documented so future contributors don't repeat them.

### 1. Lower $\gamma$ gives *higher* potential shaping for approach
Because $\Phi(s) = -\| p - z \|$ is **negative**, the term $\gamma \Phi(s') - \Phi(s)$ is dominated by how aggressively $\gamma$ discounts the negative future potential. The intuitive "lower γ should weaken shaping" is wrong. We initially had a failing test on this until we worked through the sign.

### 2. Fixed agent spawn → policy overfits to one trajectory
The original `reset()` always spawned A at $(1, 1)$ and B at $(9, 9)$. PPO then learned "what to do from this exact start" rather than a generalising policy, and the critic's V(s) was only calibrated along the trained trajectory corridor — breaking downstream analyses like value heatmaps. Fix: `randomize_spawn=True` (new default) samples uniformly with a min-separation reject loop. Set `randomize_spawn=False` for tests / evaluation when you need determinism.

### 3. `ModelPool` default `max_size=10` silently evicts early checkpoints
A 500k-step run at `generation_steps=10_000` creates 50 checkpoints; with `max_size=10` only the last 10 survive. For analyses that need historical snapshots (training-progression trajectory plots), bump `pool_kwargs=dict(max_size=60)`. Storage is cheap (~460 KB per checkpoint).

### 4. LoS at a wall endpoint is *blocked*, not grazing
A sightline that exactly grazes a wall **endpoint** is correctly reported as blocked by `los_and_active_corner` (which calls `segments_intersect`). The older standalone `find_active_corner` did not check intersection, so it returned a finite `d_peek` for that degenerate case. After deduplication the discrepancy surfaced — the test now asserts the correct `los=False` semantic at the grazing geometry.

### 5. Critic at OOD positions is interpolation, not evidence
Value heatmaps sweep agent A across the full arena, but the critic is only well-calibrated where the training distribution actually visited. The `randomize_spawn` fix widens the visited region significantly, but cells far from any plausible trajectory should be read as "smooth network interpolation" rather than "learned ground truth."

### 6. The sub-1ms env step budget is enforced by a test
`tests/test_arena.py::TestArenaPerformance::test_step_under_1ms` regresses on per-step latency. Adding any per-wall Python loop in the hot path will fail it — use the single-pass `los_and_active_corner` rather than chaining `has_line_of_sight` + a separate corner search inside `step()`.

## Repository layout

```
core/geometry.py             # LoS + active-corner geometry primitives
env/arena.py                 # 2-player Gymnasium env (POMDP, simultaneous actions)
env/reward_shaping.py        # Potential-based shaping (Ng-Russell-Russell)
utils/callbacks.py           # CurriculumCallback for linear shaping-weight decay
agents/ppo_agent.py          # SB3 PPO factory (MLP [128, 128])
training/self_play_loop.py   # SelfPlayWrapper + ModelPool + run_self_play
training/train_phase3.py     # Entry point: 500k-step self-play with curriculum
analysis/value_heatmap.py    # Critic V(s) sweep across the arena
analysis/trajectory_plot.py  # Policy evolution across checkpoints
tests/                       # pytest — 192 tests
```

## Testing

```bash
pytest tests/                          # full suite
pytest tests/ -k performance           # 1ms step-budget regression
pytest tests/test_arena.py -v          # arena unit tests
```
