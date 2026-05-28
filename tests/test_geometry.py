"""Unit tests for core/geometry.py"""

import pytest
import numpy as np
from core.geometry import segments_intersect, has_line_of_sight, _on_segment


class TestOnSegment:
    def test_point_at_midpoint(self):
        assert _on_segment(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 0.0])
        )

    def test_point_at_endpoint(self):
        assert _on_segment(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([2.0, 0.0])
        )

    def test_point_off_segment(self):
        assert not _on_segment(
            np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0])
        )


class TestSegmentsIntersect:
    def test_simple_cross(self):
        # + shape: horizontal (0,1)-(2,1) crosses vertical (1,0)-(1,2)
        assert segments_intersect((0, 1), (2, 1), (1, 0), (1, 2))

    def test_parallel_horizontal(self):
        assert not segments_intersect((0, 0), (2, 0), (0, 1), (2, 1))

    def test_parallel_vertical(self):
        assert not segments_intersect((1, 0), (1, 2), (2, 0), (2, 2))

    def test_collinear_non_overlapping(self):
        assert not segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))

    def test_collinear_overlapping(self):
        assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0))

    def test_t_intersection_endpoint_on_middle(self):
        # T-shape: (1,1) is on segment (0,1)-(2,1)
        assert segments_intersect((0, 1), (2, 1), (1, 1), (1, 2))

    def test_shared_endpoint(self):
        assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 0))

    def test_diagonal_cross(self):
        assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))

    def test_non_intersecting_diagonal(self):
        assert not segments_intersect((0, 0), (1, 1), (2, 0), (3, 1))

    def test_very_short_segments(self):
        # Near-degenerate but still crossing
        assert segments_intersect((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0))

    def test_segments_touching_at_corner(self):
        # L-shape touching at (1,1)
        assert segments_intersect((0, 1), (1, 1), (1, 1), (1, 0))


class TestHasLineOfSight:
    """Scenario: 10x10 arena with a vertical wall from (5,3) to (5,7)."""

    def setup_method(self):
        self.walls = [((5.0, 3.0), (5.0, 7.0))]

    def test_clear_same_side_left(self):
        assert has_line_of_sight((1.0, 5.0), (4.0, 5.0), self.walls)

    def test_clear_same_side_right(self):
        assert has_line_of_sight((6.0, 5.0), (9.0, 5.0), self.walls)

    def test_blocked_horizontal_through_wall(self):
        assert not has_line_of_sight((2.0, 5.0), (8.0, 5.0), self.walls)

    def test_clear_above_wall(self):
        # y=8 is above the wall's y-extent (3..7)
        assert has_line_of_sight((2.0, 8.0), (8.0, 8.0), self.walls)

    def test_clear_below_wall(self):
        assert has_line_of_sight((2.0, 1.0), (8.0, 1.0), self.walls)

    def test_clear_diagonal_bypasses_wall(self):
        # Diagonal from (1,1) to (9,9) passes through (5,5), which is on the wall
        # The wall goes from (5,3) to (5,7) so (5,5) IS on the wall -> blocked
        assert not has_line_of_sight((1.0, 1.0), (9.0, 9.0), self.walls)

    def test_no_walls_always_visible(self):
        assert has_line_of_sight((0.0, 0.0), (9.9, 9.9), [])

    def test_empty_walls_list(self):
        assert has_line_of_sight((2.0, 5.0), (8.0, 5.0), [])

    def test_multiple_walls_first_blocks(self):
        walls = [
            ((3.0, 0.0), (3.0, 10.0)),
            ((7.0, 0.0), (7.0, 10.0)),
        ]
        assert not has_line_of_sight((1.0, 5.0), (9.0, 5.0), walls)

    def test_multiple_walls_path_clears_both(self):
        walls = [
            ((3.0, 5.0), (3.0, 10.0)),  # upper half of x=3
            ((7.0, 0.0), (7.0, 5.0)),   # lower half of x=7
        ]
        # A straight horizontal path at y=2 only hits the lower wall at x=7
        assert not has_line_of_sight((1.0, 2.0), (9.0, 2.0), walls)

    def test_symmetry(self):
        # LoS should be symmetric: A sees B iff B sees A
        a, b = (2.0, 5.0), (8.0, 5.0)
        assert has_line_of_sight(a, b, self.walls) == has_line_of_sight(b, a, self.walls)

    def test_same_point_always_visible(self):
        assert has_line_of_sight((5.0, 5.0), (5.0, 5.0), self.walls)
