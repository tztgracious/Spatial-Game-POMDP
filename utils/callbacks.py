"""
Curriculum learning callback for SB3 training (GUIDE §Phase 3).

CurriculumCallback linearly decays RewardShaper.weight from initial_weight
to final_weight over the course of training, smoothly transitioning agents
from guided potential-based exploration to pure zero-sum competition.

weight(t) = initial + (final - initial) * clamp(t / total_timesteps, 0, 1)
"""

from __future__ import annotations

import logging
from stable_baselines3.common.callbacks import BaseCallback
from env.reward_shaping import RewardShaper

logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """
    Decays RewardShaper.weight linearly from initial_weight → final_weight.

    At t = 0             : weight = initial_weight  (full potential guidance)
    At t = total_timesteps: weight = final_weight   (shaping off, pure zero-sum)

    Logs shaping_weight to TensorBoard on every rollout end when verbose > 0.
    """

    def __init__(
        self,
        shaper: RewardShaper,
        total_timesteps: int,
        initial_weight: float = 1.0,
        final_weight: float = 0.0,
        log_interval: int = 1_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.shaper = shaper
        self.total_timesteps = total_timesteps
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)
        self.shaper.set_weight(weight)
        if self.verbose and self.num_timesteps % self.log_interval == 0:
            logger.info(
                "Curriculum  step=%d  progress=%.3f  shaping_weight=%.4f",
                self.num_timesteps, progress, weight,
            )
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is not None:
            self.logger.record("curriculum/shaping_weight", self.shaper.weight)
