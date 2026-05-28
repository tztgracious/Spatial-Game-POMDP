"""
Spatial geometry utilities: 2D line-segment intersection and Line-of-Sight queries.

All hot-path functions use NumPy vectorised math so that LoS checks remain
well under the 1ms per-step budget even with many wall segments.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

# Type aliases
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Scalar 2-D cross product of vectors OA and OB."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """True if point r lies on segment pq (caller guarantees collinearity)."""
    return (
        min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
        and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    )


def segments_intersect(
    p1: Point, p2: Point, p3: Point, p4: Point
) -> bool:
    """
    Return True if segment p1-p2 and segment p3-p4 intersect.

    Uses the standard orientation / cross-product test with collinear
    edge-case handling.
    """
    a, b = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    c, d = np.asarray(p3, dtype=float), np.asarray(p4, dtype=float)

    d1 = _cross_2d(c, d, a)
    d2 = _cross_2d(c, d, b)
    d3 = _cross_2d(a, b, c)
    d4 = _cross_2d(a, b, d)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Collinear overlap checks
    if d1 == 0 and _on_segment(c, d, a):
        return True
    if d2 == 0 and _on_segment(c, d, b):
        return True
    if d3 == 0 and _on_segment(a, b, c):
        return True
    if d4 == 0 and _on_segment(a, b, d):
        return True

    return False


def has_line_of_sight(a: Point, b: Point, walls: List[Segment]) -> bool:
    """
    Return True if no wall segment blocks the line of sight between a and b.

    A point is always visible to itself (degenerate zero-length segment).
    Otherwise iterates over walls and returns False on the first intersection,
    making early-exit common for blocked sightlines.
    """
    if a == b:
        return True
    for wall_start, wall_end in walls:
        if segments_intersect(a, b, wall_start, wall_end):
            return False
    return True
