"""
Potential-based reward shaping for arena self-play (GUIDE §Phase 3).

RewardShaper computes Φ(s) = -‖pos - target_zone‖ and the per-step shaping
bonus r = weight * (γ·Φ(s') - Φ(s)).

The weight is mutable so CurriculumCallback (utils/callbacks.py) can linearly
decay it to zero over training, progressively removing the guidance signal
until the agents compete under pure zero-sum rewards.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


class RewardShaper:
    """
    Potential-based shaping that guides agents toward a target zone.

    Potential : Φ(s) = -‖pos - target_zone‖   (higher = closer = better)
    Shaping   : r = weight * (γ · Φ(s') - Φ(s))

    Positive r means the agent moved closer to the zone; negative means it
    retreated.  Once weight reaches 0 the method returns 0 immediately.
    """

    def __init__(
        self,
        target_zone: Tuple[float, float],
        gamma: float = 0.99,
        initial_weight: float = 1.0,
    ) -> None:
        self.target_zone = np.array(target_zone, dtype=float)
        self.gamma = gamma
        self._weight = float(initial_weight)

    # --- Properties ---

    @property
    def weight(self) -> float:
        return self._weight

    def set_weight(self, weight: float) -> None:
        """Update shaping weight; clamped to [0, ∞)."""
        self._weight = float(np.clip(weight, 0.0, None))

    # --- Core methods ---

    def potential(self, pos) -> float:
        """Φ(s) — negative distance to target zone."""
        return -float(np.linalg.norm(np.asarray(pos, dtype=float) - self.target_zone))

    def shaping(self, old_pos, new_pos) -> float:
        """
        One-step shaping bonus for moving from old_pos to new_pos.

        Returns 0.0 immediately when weight == 0 to skip computation once
        curriculum has fully phased out the guidance signal.
        """
        if self._weight == 0.0:
            return 0.0
        return self._weight * (
            self.gamma * self.potential(new_pos) - self.potential(old_pos)
        )
