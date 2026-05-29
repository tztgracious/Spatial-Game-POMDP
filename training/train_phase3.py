"""
Phase 3 self-play training entry point.

Wires RewardShaper (target = upper wall corner (5, 7)) and CurriculumCallback
(linear weight decay 1.0 -> 0.0 across total_timesteps) into run_self_play.
"""

from __future__ import annotations

import logging

from env.arena import ArenaEnv
from env.reward_shaping import RewardShaper
from training.self_play_loop import run_self_play
from utils.callbacks import CurriculumCallback


TOTAL_TIMESTEPS = 500_000
GENERATION_STEPS = 10_000
TARGET_ZONE = (5.0, 7.0)   # upper endpoint of default vertical wall
INITIAL_WEIGHT = 1.0
FINAL_WEIGHT = 0.0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    shaper = RewardShaper(
        target_zone=TARGET_ZONE,
        gamma=0.99,
        initial_weight=INITIAL_WEIGHT,
    )
    curriculum = CurriculumCallback(
        shaper=shaper,
        total_timesteps=TOTAL_TIMESTEPS,
        initial_weight=INITIAL_WEIGHT,
        final_weight=FINAL_WEIGHT,
        verbose=1,
    )

    run_self_play(
        total_timesteps=TOTAL_TIMESTEPS,
        generation_steps=GENERATION_STEPS,
        checkpoint_dir="checkpoints",
        log_dir="runs/phase3",
        arena_kwargs=dict(
            shaper=shaper,
            distance_scale=8.0,    # Option C distance falloff
            min_exposure=0.2,
            randomize_spawn=True,  # generalisation
        ),
        # Retain every generation for Phase 4 trajectory analysis.
        # 50 gens * ~460KB ≈ 23MB total.
        pool_kwargs=dict(max_size=60),
        callbacks=[curriculum],
        seed=0,
    )


if __name__ == "__main__":
    main()
