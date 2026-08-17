"""Time-to-collision (TTC) calculation.

New module. Combines a distance (from `mapping.py::BEVMapper`, ultimately
derived from Depth Anything V2's depth map -- unmodified, see project's
Model Status docs for the metric-vs-relative depth caveat that also applies
here) with a closing speed (from the new `relative_velocity.py`) into a
standard TTC estimate: how many seconds until the gap closes to zero if
current relative motion continues unchanged.

TTC is only meaningful when the object is actually closing (distance
shrinking over time). An object that is stationary or receding has no
finite collision time from this metric -- that case is represented as
`ttc_seconds=None` (explicitly "not applicable", not zero, not infinity as
a magic number, and not silently omitted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from relative_velocity import RelativeVelocity
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class TTCResult:
    track_id: int
    class_name: str
    distance_m: float
    closing_speed_m_s: float   # positive = closing, <=0 = not closing
    ttc_seconds: Optional[float]  # None if not closing / not enough data


def compute_ttc(
    distance_m: float,
    closing_speed_m_s: float,
    min_closing_speed_m_s: float = 0.15,
    max_ttc_s: float = 60.0,
) -> Optional[float]:
    """Compute time-to-collision in seconds, or None if not applicable.

    Args:
        distance_m: current forward distance to the object (meters, >0).
        closing_speed_m_s: POSITIVE value means the object is approaching
            (gap shrinking); this is `-vz_m_s` from `RelativeVelocity`,
            since `vz_m_s` is negative when closing (see relative_velocity.py).
        min_closing_speed_m_s: below this, the object is treated as
            effectively not closing (avoids a near-zero-denominator
            division producing an absurdly large but technically-finite
            TTC number that looks like a precise prediction but isn't).
        max_ttc_s: cap the returned value at this many seconds -- beyond
            this, the exact number stops being operationally meaningful
            ("collision in 340 seconds" isn't a useful distinction from
            "not closing"), so it's clipped for display/logic sanity.

    Returns:
        TTC in seconds (float, 0 < value <= max_ttc_s), or None if the
        object is not closing meaningfully (safe from a TTC perspective).
    """
    if distance_m <= 0:
        return None
    if closing_speed_m_s < min_closing_speed_m_s:
        return None
    ttc = distance_m / closing_speed_m_s
    return min(ttc, max_ttc_s)


def compute_ttc_for_objects(
    bev_objects: List,
    velocities: Dict[int, RelativeVelocity],
    min_closing_speed_m_s: float = 0.15,
    max_ttc_s: float = 60.0,
) -> List[TTCResult]:
    """Compute a TTCResult for every BEV object that has a valid velocity estimate.

    `bev_objects` is any iterable with `.track_id`, `.class_name`, `.z_m`
    (duck-typed against `mapping.py::BEVObject`, not imported directly).
    Objects with no `track_id`, or no valid entry in `velocities` yet
    (e.g. newly-appeared track, not enough history), are skipped -- not an
    error, just "no TTC available yet for this object this frame".
    """
    results: List[TTCResult] = []
    for obj in bev_objects:
        track_id = getattr(obj, "track_id", None)
        if track_id is None:
            continue
        velocity = velocities.get(track_id)
        if velocity is None or not velocity.valid:
            continue

        closing_speed = -velocity.vz_m_s  # vz negative = closing; flip sign convention
        ttc = compute_ttc(obj.z_m, closing_speed, min_closing_speed_m_s, max_ttc_s)

        results.append(TTCResult(
            track_id=track_id,
            class_name=getattr(obj, "class_name", "object"),
            distance_m=obj.z_m,
            closing_speed_m_s=closing_speed,
            ttc_seconds=ttc,
        ))
    return results
