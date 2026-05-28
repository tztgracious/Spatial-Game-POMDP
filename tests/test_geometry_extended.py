"""Unit tests for Phase 3 geometry additions:
los_and_active_corner and hit_probability (Option C).
"""

import math
import pytest
import numpy as np
from core.geometry import (
    los_and_active_corner,
    hit_probability,
    has_line_of_sight,
)


# ---------------------------------------------------------------------------
# los_and_active_corner — combined single-pass function
# ---------------------------------------------------------------------------

class TestLosAndActiveCorner:
    """Combined LoS check + active-corner search.

    Standard scenario:
      Wall from (5,0) to (5,7) — vertical segment in lower-centre of arena.
      A = (2, 8) — above the wall.
      B peeks around the upper endpoint C = (5, 7).
    """

    def _arr(self, x, y):
        return np.array([x, y], dtype=float)

    # ---- LoS basics ----

    def test_blocked_los_returns_false(self):
        walls = [((5.0, 0.0), (5.0, 10.0))]  # full vertical wall
        los, corner, dpk_AB, dpk_BA = los_and_active_corner(
            self._arr(2, 5), self._arr(8, 5), walls
        )
        assert los is False
        assert corner is None
        assert dpk_AB == float("inf")
        assert dpk_BA == float("inf")

    def test_open_field_los_true_no_corner(self):
        los, corner, _, _ = los_and_active_corner(
            self._arr(0, 5), self._arr(9, 5), []
        )
        assert los is True
        assert corner is None

    def test_matches_has_line_of_sight(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        A, B = self._arr(2, 8), self._arr(7, 7)
        los_combined, _, _, _ = los_and_active_corner(A, B, walls)
        los_separate = has_line_of_sight(tuple(A), tuple(B), walls)
        assert los_combined == los_separate

    def test_same_point_returns_true_no_corner(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        los, corner, _, _ = los_and_active_corner(
            self._arr(5, 5), self._arr(5, 5), walls
        )
        assert los is True
        assert corner is None

    # ---- Active corner selection ----

    def test_detects_corner_between_a_and_b(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        _, corner, _, _ = los_and_active_corner(
            self._arr(2, 8), self._arr(7, 7), walls
        )
        assert corner is not None

    def test_selects_upper_corner_for_above_peek(self):
        # A above wall, B peeking past upper corner (5,7)
        walls = [((5.0, 0.0), (5.0, 7.0))]
        _, corner, _, _ = los_and_active_corner(
            self._arr(2, 8), self._arr(6, 7.5), walls
        )
        assert corner is not None
        assert corner[1] == pytest.approx(7.0, abs=0.1)

    def test_corner_beyond_b_is_excluded(self):
        # Wall endpoint at (12,5) is beyond B=(8,5) — t > 0.95, excluded
        walls = [((12.0, 5.0), (12.0, 8.0))]
        _, corner, _, _ = los_and_active_corner(
            self._arr(0, 5), self._arr(8, 5), walls
        )
        assert corner is None

    def test_corner_behind_a_is_excluded(self):
        # Wall endpoint at (-2,5) is behind A=(0,5) — t < 0.05, excluded
        walls = [((-2.0, 5.0), (-2.0, 8.0))]
        _, corner, _, _ = los_and_active_corner(
            self._arr(0, 5), self._arr(8, 5), walls
        )
        assert corner is None

    def test_selects_corner_closest_to_sightline(self):
        walls = [
            ((5.0, 5.05), (5.0, 8.0)),   # lower endpoint (5,5.05) near line y=5
            ((5.0, 0.0),  (5.0, 2.0)),   # upper endpoint (5,2) far from y=5
        ]
        _, corner, _, _ = los_and_active_corner(
            self._arr(0, 5), self._arr(9, 5), walls
        )
        assert corner is not None
        assert corner[1] == pytest.approx(5.05, abs=0.1)

    def test_same_corner_used_for_both_directions(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        _, corner, _, _ = los_and_active_corner(
            self._arr(2, 8), self._arr(7, 7), walls
        )
        _, corner_rev, _, _ = los_and_active_corner(
            self._arr(7, 7), self._arr(2, 8), walls
        )
        assert corner is not None and corner_rev is not None
        np.testing.assert_array_almost_equal(corner, corner_rev)

    # ---- d_peek correctness ----

    def test_b_on_shadow_ray_is_blocked(self):
        # B beyond C along the A→C ray makes the sightline graze the wall
        # endpoint C — the combined function reports this as blocked LoS.
        A = self._arr(0, 8)
        C = np.array([5.0, 7.0])
        ray = C - A
        ray_unit = ray / np.linalg.norm(ray)
        B_on_ray = A + np.linalg.norm(ray) * 1.2 * ray_unit
        walls = [((5.0, 0.0), (5.0, 7.0))]
        los, corner, d_peek_AB, _ = los_and_active_corner(A, B_on_ray, walls)
        assert los is False
        assert corner is None
        assert d_peek_AB == float("inf")

    def test_d_peek_positive_when_b_off_shadow_ray(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        _, _, d_peek_AB, _ = los_and_active_corner(
            self._arr(2, 8), self._arr(6, 7.5), walls
        )
        assert d_peek_AB > 0.0

    def test_more_stepout_gives_larger_d_peek(self):
        walls = [((5.0, 0.0), (5.0, 7.0))]
        A = self._arr(2, 8)
        _, _, d_small, _ = los_and_active_corner(A, self._arr(5.3, 7.3), walls)
        _, _, d_large, _ = los_and_active_corner(A, self._arr(8.0, 7.5), walls)
        assert d_large > d_small

    def test_d_peek_AB_and_BA_differ(self):
        """Asymmetric peek angles — d_peek_AB and d_peek_BA should differ."""
        walls = [((5.0, 0.0), (5.0, 7.0))]
        los, _, dpk_AB, dpk_BA = los_and_active_corner(
            self._arr(2, 8), self._arr(7, 7.5), walls
        )
        assert los is True
        assert dpk_AB != pytest.approx(dpk_BA)


# ---------------------------------------------------------------------------
# hit_probability — distance falloff (unchanged from Phase 3 base)
# ---------------------------------------------------------------------------

class TestHitProbabilityDistance:
    def test_no_scales_full_hit(self):
        assert hit_probability((0, 0), (9, 9), []) == pytest.approx(1.0)

    def test_base_accuracy_respected(self):
        assert hit_probability((0, 0), (9, 9), [], base_accuracy=0.7) == pytest.approx(0.7)

    def test_distance_scale_none_no_decay(self):
        p = hit_probability((0, 0), (9, 9), [], distance_scale=None)
        assert p == pytest.approx(1.0)

    def test_distance_scale_decays_with_range(self):
        p_close = hit_probability((4.5, 5), (5.5, 5), [], distance_scale=8.0)
        p_far   = hit_probability((1.0, 5), (9.0, 5), [], distance_scale=8.0)
        assert p_close > p_far

    def test_at_distance_scale_equals_exp_neg1(self):
        p = hit_probability((0, 0), (8, 0), [], distance_scale=8.0)
        assert p == pytest.approx(math.exp(-1), rel=1e-5)

    def test_probability_bounded(self):
        for d in [0, 1, 5, 10, 100]:
            p = hit_probability((0, 0), (float(d), 0), [], distance_scale=5.0)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# hit_probability — Option C 近大远小 exposure
# ---------------------------------------------------------------------------

class TestHitProbabilityOptionC:
    """
    Corner scenario:
      Wall from (5,0) to (5,7).
      A = (2,8) — above the wall, can see B peeking around upper corner (5,7).
    """

    def setup_method(self):
        self.walls = [((5.0, 0.0), (5.0, 7.0))]
        self.A = (2.0, 8.0)

    def test_open_field_full_exposure(self):
        p = hit_probability((0, 0), (5, 5), [])
        assert p == pytest.approx(1.0)

    def test_no_active_corner_full_exposure(self):
        # Wall is completely to the side — no corner on sightline
        walls = [((0.0, 0.0), (0.0, 10.0))]  # left border, not between A and B
        p = hit_probability((2, 5), (8, 5), walls)
        assert p == pytest.approx(1.0)

    def test_min_exposure_when_d_peek_near_zero(self):
        # B at (5.3, 7.0) — almost on the shadow ray through upper corner (5,7)
        # d_peek will be very small → peek_fraction ≈ min_exposure
        p = hit_probability(self.A, (5.3, 7.0), self.walls, min_exposure=0.2)
        assert p <= 0.25  # close to min_exposure

    def test_exposure_increases_with_stepout(self):
        p_small = hit_probability(self.A, (5.5, 7.2), self.walls, min_exposure=0.0)
        p_large = hit_probability(self.A, (8.0, 7.5), self.walls, min_exposure=0.0)
        assert p_large > p_small

    def test_近大远小_farther_corner_harder_to_hit(self):
        """Core 近大远小 test: same d_peek, different dist(A,C) → different hit prob.

        A_near is close to corner C — same physical step-out d gives a larger
        shadow angle → B is easier to hit.
        A_far is far from corner C — same d_peek subtends a smaller angle → harder to hit.
        """
        # We fix the wall and pick two shooter positions at different distances from corner C=(5,7)
        A_near = (4.0, 8.0)   # dist to C=(5,7) ≈ sqrt(1+1) = 1.41
        A_far  = (0.0, 8.0)   # dist to C=(5,7) ≈ sqrt(25+1) = 5.10
        B = (6.0, 7.5)        # same peeking target for both

        p_near = hit_probability(A_near, B, self.walls, min_exposure=0.0)
        p_far  = hit_probability(A_far,  B, self.walls, min_exposure=0.0)

        # Shooter close to corner → B appears to subtend larger angle → easier to hit
        assert p_near > p_far

    def test_min_exposure_floor_respected(self):
        for min_exp in [0.1, 0.2, 0.3]:
            p = hit_probability(self.A, (5.3, 7.0), self.walls, min_exposure=min_exp)
            assert p >= min_exp - 1e-6

    def test_distance_and_exposure_compound(self):
        # Both factors active: combined < either alone
        p_no_dist  = hit_probability(self.A, (7.0, 7.5), self.walls,
                                     distance_scale=None, min_exposure=0.0)
        p_no_exp   = hit_probability(self.A, (7.0, 7.5), [],
                                     distance_scale=8.0, min_exposure=0.0)
        p_both     = hit_probability(self.A, (7.0, 7.5), self.walls,
                                     distance_scale=8.0, min_exposure=0.0)
        assert p_both <= min(p_no_dist, p_no_exp)

    def test_result_bounded_0_1(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            A = tuple(rng.uniform(0, 10, 2))
            B = tuple(rng.uniform(0, 10, 2))
            p = hit_probability(A, B, self.walls,
                                distance_scale=5.0, min_exposure=0.1)
            assert 0.0 <= p <= 1.0
