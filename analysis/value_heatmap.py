"""
Value heatmap V(s) over the arena for a trained PPO critic.

Sweeps agent-0's (x, y) on a grid with the opponent fixed at a chosen point,
constructs the same 7-element observation the policy was trained on (POMDP:
opponent block zeroed when LoS is blocked), and queries the critic for V(s).

Run:
    python -m analysis.value_heatmap                       # default 500k checkpoint
    python -m analysis.value_heatmap --checkpoint <path>   # any specific gen
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from core.geometry import has_line_of_sight
from env.arena import ArenaEnv, OBS_SIZE


def compute_value_grid(
    model: PPO,
    env: ArenaEnv,
    opp_pos: Tuple[float, float],
    resolution: int = 60,
    own_hp_frac: float = 1.0,
    opp_hp_frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a resolution × resolution grid of V(s) values.

    Each cell (i, j) corresponds to agent-0 standing at (xs[i], ys[j]) with
    the opponent fixed at opp_pos and both agents at full HP.  The LoS bit and
    opponent-position channels are filled in to exactly match arena._compute_obs.

    Returns
    -------
    xs : (resolution,) array of x coordinates
    ys : (resolution,) array of y coordinates
    V  : (resolution, resolution) array indexed [y_idx, x_idx] for imshow
    """
    xs = np.linspace(0.0, env.width, resolution, dtype=np.float32)
    ys = np.linspace(0.0, env.height, resolution, dtype=np.float32)

    obs_batch = np.zeros((resolution * resolution, OBS_SIZE), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            los = has_line_of_sight((float(x), float(y)), opp_pos, env.walls)
            idx = j * resolution + i
            obs_batch[idx, 0] = x
            obs_batch[idx, 1] = y
            obs_batch[idx, 2] = own_hp_frac
            if los:
                obs_batch[idx, 3] = 1.0
                obs_batch[idx, 4] = opp_pos[0]
                obs_batch[idx, 5] = opp_pos[1]
                obs_batch[idx, 6] = opp_hp_frac
            # else: cols 3-6 stay 0 (POMDP masking)

    with torch.no_grad():
        tensor = torch.as_tensor(obs_batch, device=model.device)
        values = model.policy.predict_values(tensor).cpu().numpy().reshape(resolution, resolution)
    return xs, ys, values


def plot_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    V: np.ndarray,
    walls,
    opp_pos: Tuple[float, float],
    out_path: Path,
    title: str = "V(s) — agent A across arena",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        V,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    for (a, b) in walls:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="white", linewidth=4)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=2)
    ax.scatter(
        [opp_pos[0]], [opp_pos[1]],
        c="red", s=240, marker="X", edgecolors="white", linewidths=1.5,
        label=f"opponent B @ ({opp_pos[0]:.1f}, {opp_pos[1]:.1f})",
    )
    plt.colorbar(im, ax=ax, label="V(s)")
    ax.set_xlabel("agent A — x")
    ax.set_ylabel("agent A — y")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V(s) heatmap for a trained critic.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/gen_00500000.zip",
        help="Path to PPO checkpoint (.zip).",
    )
    parser.add_argument("--opp-x", type=float, default=8.0)
    parser.add_argument("--opp-y", type=float, default=5.0)
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument(
        "--out",
        default="runs/phase3/value_heatmap.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = ArenaEnv(randomize_spawn=False)
    opp_pos = (args.opp_x, args.opp_y)

    xs, ys, V = compute_value_grid(model, env, opp_pos=opp_pos, resolution=args.resolution)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        xs, ys, V, env.walls, opp_pos, out_path,
        title=f"V(s) — {Path(args.checkpoint).stem}",
    )
    print(
        f"Saved {out_path}  |  V range: [{V.min():.3f}, {V.max():.3f}]  "
        f"|  mean {V.mean():.3f}"
    )


if __name__ == "__main__":
    main()
