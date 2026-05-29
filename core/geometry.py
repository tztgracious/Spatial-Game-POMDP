"""
Spatial geometry utilities: 2D line-segment intersection and Line-of-Sight queries.

All hot-path functions use NumPy vectorised math so that LoS checks remain
well under the 1ms per-step budget even with many wall segments.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

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


def los_and_active_corner(
    A: np.ndarray,
    B: np.ndarray,
    walls: List[Segment],
) -> Tuple[bool, Optional[np.ndarray], float, float]:
    """
    Combined single-pass LoS check + active corner for both shooting directions.

    Iterates walls exactly once to:
      1. Detect any blocking segment (returns early on first intersection).
      2. Track the wall endpoint closest to the sightline AB — this is the
         corner both agents are peeking around.  Because perpendicular distance
         to a line is direction-independent, the same corner is optimal for
         A-shoots-B and B-shoots-A; only d_peek differs.

    Returns
    -------
    (los, corner, d_peek_AB, d_peek_BA)
      los       — True if A can see B.
      corner    — active corner ndarray, or None when no qualifying endpoint exists.
      d_peek_AB — B's step-out past A's shadow ray A→corner (use when A fires at B).
      d_peek_BA — A's step-out past B's shadow ray B→corner (use when B fires at A).
    When los is False all other values are (None, inf, inf).
    """
    _INF = float("inf")
    if np.allclose(A, B):
        return True, None, _INF, _INF

    AB = B - A
    AB_len = float(np.linalg.norm(AB))
    if AB_len < 1e-9 or not walls:
        return True, None, _INF, _INF

    AB_unit = AB / AB_len
    AB_perp = np.array([-AB_unit[1], AB_unit[0]])

    best_corner: Optional[np.ndarray] = None
    best_perp: float = _INF

    for (p1, p2) in walls:
        # --- LoS check: early-exit on first blocking wall ---
        if segments_intersect(tuple(A), tuple(B), p1, p2):
            return False, None, _INF, _INF

        # --- Active corner: endpoint closest to sightline AB ---
        for raw in (p1, p2):
            C = np.asarray(raw, dtype=float)
            AC = C - A
            t = float(np.dot(AC, AB_unit)) / AB_len
            if not (0.05 < t < 0.95):
                continue
            perp = abs(float(np.dot(AC, AB_perp)))
            if perp < best_perp:
                best_perp = perp
                best_corner = C.copy()

    if best_corner is None:
        return True, None, _INF, _INF

    # d_peek_AB: B's perpendicular distance to shadow ray A → corner
    AC = best_corner - A
    AC_len = float(np.linalg.norm(AC))
    if AC_len < 1e-9:
        return True, best_corner, 0.0, 0.0
    AC_unit = AC / AC_len
    d_peek_AB = abs(float((B - A)[0] * AC_unit[1] - (B - A)[1] * AC_unit[0]))

    # d_peek_BA: A's perpendicular distance to shadow ray B → corner
    BC = best_corner - B
    BC_len = float(np.linalg.norm(BC))
    if BC_len < 1e-9:
        return True, best_corner, d_peek_AB, 0.0
    BC_unit = BC / BC_len
    d_peek_BA = abs(float((A - B)[0] * BC_unit[1] - (A - B)[1] * BC_unit[0]))

    return True, best_corner, d_peek_AB, d_peek_BA


def hit_probability(
    shooter: Point,
    target: Point,
    walls: List[Segment],
    base_accuracy: float = 1.0,
    distance_scale: Optional[float] = None,
    min_exposure: float = 0.2,
    _corner: Optional[np.ndarray] = None,
    _d_peek: Optional[float] = None,
) -> float:
    """
    P(hit) when shooter fires at target — Option C implementation.

    Distance falloff (distance_scale is not None)
        P *= exp(-dist(A,B) / distance_scale)

    近大远小 — corner peek exposure
        peek_fraction = max(min_exposure, arctan(d_peek / dist(A,C)) / (π/2))
        where C is the active corner and d_peek is B's step-out past A's shadow ray.
        No active corner → peek_fraction = 1.0 (fully exposed).

    _corner / _d_peek : pre-computed values from los_and_active_corner().
        When supplied the geometry pass is skipped — use these when calling
        inside arena.step() where los_and_active_corner() was already run.
    """
    A_arr = np.asarray(shooter, dtype=float)
    B_arr = np.asarray(target, dtype=float)
    p = float(base_accuracy)

    if distance_scale is not None:
        dist_AB = float(np.linalg.norm(B_arr - A_arr))
        p *= float(np.exp(-dist_AB / distance_scale))

    if _corner is not None and _d_peek is not None:
        active_corner, d_peek = _corner, _d_peek
    else:
        _, active_corner, d_peek, _ = los_and_active_corner(A_arr, B_arr, walls)

    if active_corner is not None:
        dist_AC = float(np.linalg.norm(active_corner - A_arr))
        if dist_AC > 1e-9:
            peek_angle = float(np.arctan2(d_peek, dist_AC))
            peek_fraction = max(float(min_exposure), peek_angle / (np.pi / 2))
        else:
            peek_fraction = 1.0
        p *= peek_fraction

    return float(np.clip(p, 0.0, 1.0))


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
