from __future__ import annotations

import math
from dataclasses import dataclass

from .detection import HitEvent


@dataclass
class ScoringRing:
    """A concentric scoring ring on the target face."""
    score: int
    radius_fraction: float   # radius as fraction of the display's short axis


# Standard 10-ring target scaled to display proportions
DEFAULT_RINGS: list[ScoringRing] = [
    ScoringRing(score=10, radius_fraction=0.05),
    ScoringRing(score=9,  radius_fraction=0.10),
    ScoringRing(score=8,  radius_fraction=0.15),
    ScoringRing(score=7,  radius_fraction=0.20),
    ScoringRing(score=6,  radius_fraction=0.27),
    ScoringRing(score=5,  radius_fraction=0.34),
    ScoringRing(score=4,  radius_fraction=0.42),
    ScoringRing(score=3,  radius_fraction=0.50),
    ScoringRing(score=2,  radius_fraction=0.60),
    ScoringRing(score=1,  radius_fraction=0.72),
]


def score_hit(pt: tuple[int, int], center: tuple[int, int],
              display_size: tuple[int, int],
              rings: list[ScoringRing] = DEFAULT_RINGS) -> int:
    """
    Score a hit at display coordinates pt against concentric rings.

    center: center of the target in display pixels
    display_size: (width, height)
    Returns the ring score (1–10), or 0 if outside all rings.
    """
    short_axis = min(display_size)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    distance = math.sqrt(dx * dx + dy * dy)

    for ring in rings:
        if distance <= ring.radius_fraction * short_axis:
            return ring.score
    return 0


def compute_group_stats(hits: list[HitEvent]) -> dict:
    """
    Compute shot group statistics from a list of hits.

    Returns a dict with:
      count, center (x, y), spread_px (mean distance from group center),
      extreme_spread_px (max pairwise distance)
    """
    if not hits:
        return {"count": 0}

    pts = [h.display_pt for h in hits]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    spread = sum(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in pts) / len(pts)

    extreme = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            extreme = max(extreme, d)

    return {
        "count": len(hits),
        "center": (cx, cy),
        "spread_px": round(spread, 1),
        "extreme_spread_px": round(extreme, 1),
    }
