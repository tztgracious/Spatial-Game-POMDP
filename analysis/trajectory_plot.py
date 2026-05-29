"""
Trajectory evolution plot: roll out checkpoints at increasing training step
counts to visualise how the policy progresses from random walk → deliberate
corner peeking.

Each subplot shows N self-play episodes (the checkpoint plays both sides) with
agent A in blue and agent B in red.  Spawn positions are fixed across
checkpoints so the comparison is apples-to-apples.

Run:
    python -m analysis.trajectory_plot
    python -m analysis.trajectory_plot --steps 10000 50000 500000 --episodes 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from env.arena import ArenaEnv, OBS_SIZE


def rollout_positions(
    model: PPO,
    n_episodes: int = 4,
    seed: int = 0,
    max_steps: int = 200,
) -> List[np.ndarray]:
    """
    Roll out n_episodes of self-play with `model` playing both sides.

    Spawn is deterministic (corner spawn) so trajectories are directly
    comparable across checkpoints.  Returns a list of (T_i, 2, 2) arrays
    where last dim is (x, y).
    """
    env = ArenaEnv(randomize_spawn=False, max_steps=max_steps)
    trajectories: List[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        positions = [env._positions.copy()]
        terminated = truncated = False
        while not (terminated or truncated):
            # Full obs is [A_block, B_block] — each agent gets its own 7-dim view
            action_a, _ = model.predict(obs[:OBS_SIZE], deterministic=False)
            action_b, _ = model.predict(obs[OBS_SIZE:], deterministic=False)
            obs, _, terminated, truncated, _ = env.step([int(action_a), int(action_b)])
            positions.append(env._positions.copy())
        trajectories.append(np.asarray(positions))

    return trajectories


def plot_evolution(
    checkpoints: Sequence[Tuple[str, Path]],
    walls,
    out_path: Path,
    n_episodes: int = 4,
    width: float = 10.0,
    height: float = 10.0,
) -> None:
    fig, axes = plt.subplots(
        1, len(checkpoints),
        figsize=(5.2 * len(checkpoints), 5.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (label, path) in zip(axes, checkpoints):
        model = PPO.load(str(path), device="cpu")
        trajectories = rollout_positions(model, n_episodes=n_episodes)

        for t in trajectories:
            ax.plot(t[:, 0, 0], t[:, 0, 1], color="tab:blue", alpha=0.55, linewidth=1.4)
            ax.plot(t[:, 1, 0], t[:, 1, 1], color="tab:red", alpha=0.55, linewidth=1.4)
            ax.scatter(t[0, 0, 0], t[0, 0, 1], color="tab:blue",
                       s=60, marker="o", edgecolors="white", linewidths=1, zorder=3)
            ax.scatter(t[0, 1, 0], t[0, 1, 1], color="tab:red",
                       s=60, marker="o", edgecolors="white", linewidths=1, zorder=3)
            ax.scatter(t[-1, 0, 0], t[-1, 0, 1], color="tab:blue",
                       s=60, marker="X", edgecolors="white", linewidths=1, zorder=3)
            ax.scatter(t[-1, 1, 0], t[-1, 1, 1], color="tab:red",
                       s=60, marker="X", edgecolors="white", linewidths=1, zorder=3)

        for (a, b) in walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=4)

        ax.set_title(label)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Self-play trajectories — {n_episodes} eps each, A=blue B=red, ●=spawn ✕=end",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trajectory evolution across training.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[10_000, 50_000, 500_000],
        help="Training step counts whose checkpoints to load.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--out", default="runs/phase3/trajectories.png")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints: List[Tuple[str, Path]] = []
    for step in args.steps:
        path = ckpt_dir / f"gen_{step:08d}.zip"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {path}.  Run training with a larger ModelPool "
                f"max_size if early checkpoints were evicted."
            )
        checkpoints.append((f"{step:,} steps", path))

    env = ArenaEnv(randomize_spawn=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_evolution(
        checkpoints, env.walls, out_path,
        n_episodes=args.episodes,
        width=env.width, height=env.height,
    )
    print(f"Saved {out_path}  |  {len(checkpoints)} checkpoints × {args.episodes} episodes")


if __name__ == "__main__":
    main()
