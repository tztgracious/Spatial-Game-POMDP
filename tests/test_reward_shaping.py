"""Unit tests for env/reward_shaping.py"""

import pytest
import numpy as np
from env.reward_shaping import RewardShaper


@pytest.fixture
def shaper():
    return RewardShaper(target_zone=(5.0, 5.0), gamma=0.99, initial_weight=1.0)


class TestRewardShaperPotential:
    def test_potential_at_target_is_zero(self, shaper):
        assert shaper.potential((5.0, 5.0)) == pytest.approx(0.0)

    def test_potential_negative_away_from_target(self, shaper):
        assert shaper.potential((0.0, 5.0)) < 0.0

    def test_potential_closer_is_greater(self, shaper):
        p_close = shaper.potential((4.0, 5.0))  # dist=1
        p_far = shaper.potential((0.0, 5.0))    # dist=5
        assert p_close > p_far

    def test_potential_uses_euclidean_distance(self, shaper):
        # dist from (2,1) to (5,5) = sqrt(9+16) = 5
        assert shaper.potential((2.0, 1.0)) == pytest.approx(-5.0)


class TestRewardShaperWeight:
    def test_initial_weight(self, shaper):
        assert shaper.weight == pytest.approx(1.0)

    def test_set_weight(self, shaper):
        shaper.set_weight(0.5)
        assert shaper.weight == pytest.approx(0.5)

    def test_set_weight_clamps_negative_to_zero(self, shaper):
        shaper.set_weight(-2.0)
        assert shaper.weight == pytest.approx(0.0)

    def test_set_weight_zero(self, shaper):
        shaper.set_weight(0.0)
        assert shaper.weight == pytest.approx(0.0)

    def test_set_weight_large(self, shaper):
        shaper.set_weight(10.0)
        assert shaper.weight == pytest.approx(10.0)


class TestRewardShaperShaping:
    def test_positive_when_approaching(self, shaper):
        # Moving from x=0 → x=1, both at y=5: closer to (5,5)
        s = shaper.shaping(old_pos=(0.0, 5.0), new_pos=(1.0, 5.0))
        assert s > 0.0

    def test_negative_when_retreating(self, shaper):
        # Moving from x=4 → x=3: farther from (5,5)
        s = shaper.shaping(old_pos=(4.0, 5.0), new_pos=(3.0, 5.0))
        assert s < 0.0

    def test_zero_when_stationary(self, shaper):
        # gamma < 1 so there's a tiny discount, but PBRS still gives small value
        # When old==new: r = w*(gamma*Phi(s) - Phi(s)) = w*(gamma-1)*Phi(s)
        # This is non-zero unless Phi(s)=0 (at target zone).
        # At target zone itself it should be zero.
        s = shaper.shaping(old_pos=(5.0, 5.0), new_pos=(5.0, 5.0))
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_zero_when_weight_is_zero(self, shaper):
        shaper.set_weight(0.0)
        s = shaper.shaping(old_pos=(0.0, 0.0), new_pos=(9.9, 9.9))
        assert s == pytest.approx(0.0)

    def test_scales_linearly_with_weight(self, shaper):
        s1 = shaper.shaping((0.0, 5.0), (1.0, 5.0))
        shaper.set_weight(2.0)
        s2 = shaper.shaping((0.0, 5.0), (1.0, 5.0))
        assert s2 == pytest.approx(2.0 * s1)

    def test_potential_based_reward_shaping_formula(self, shaper):
        # r = w * (gamma * Phi(s') - Phi(s))
        old, new = (0.0, 5.0), (1.0, 5.0)
        expected = shaper.weight * (shaper.gamma * shaper.potential(new) - shaper.potential(old))
        assert shaper.shaping(old, new) == pytest.approx(expected)

    def test_gamma_affects_shaping_magnitude(self):
        # r = w * (γ·Φ(s') - Φ(s)); Φ is negative here.
        # Lower γ discounts the negative future potential more, raising shaping for approach.
        shaper_low_gamma = RewardShaper((5.0, 5.0), gamma=0.5, initial_weight=1.0)
        shaper_high_gamma = RewardShaper((5.0, 5.0), gamma=0.99, initial_weight=1.0)
        old, new = (0.0, 5.0), (1.0, 5.0)
        # Both should be positive (approach), but low gamma gives higher value here
        assert shaper_low_gamma.shaping(old, new) > shaper_high_gamma.shaping(old, new) > 0
