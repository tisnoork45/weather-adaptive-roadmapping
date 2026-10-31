"""Relative velocity estimation from tracked BEV object positions.

New module -- does not exist anywhere in the current project. Reads
`BEVObject` positions (x_m, z_m) that `mapping.py::BEVMapper` already
produces per frame, and per `track_id` (already provided by the existing,
unmodified `tracking.py::ByteTracker`), estimates how fast that object's
position is changing in real-world meters per second.

IMPORTANT terminology note: this measures RELATIVE velocity -- how fast the
object's distance/lateral position is changing relative to the camera --
not the object's absolute velocity independent of ego-motion. This project
has no odometry/IMU input, so absolute decomposition (their speed vs. our
speed) is not possible, and is not needed: TTC (time-to-collision) is
defined in terms of relative closing velocity, which is exactly what this
module measures. This is not a limitation to work around, it is the
correct quantity for the purpose.

Uses real WALL-CLOCK elapsed time between observations (via
`time.perf_counter()`), not an assumed frame rate, since this pipeline's
per-frame latency is highly variable (confirmed: ~3.8s/frame on CPU) --
using a fixed assumed dt would silently produce wrong velocity units.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class RelativeVelocity:
    """Estimated relative velocity for one tracked object, in meters/second."""

    track_id: int
    vx_m_s: float          # lateral relative velocity, +right
    vz_m_s: float          # forward relative velocity; NEGATIVE = closing (approaching),
                            # POSITIVE = receding (moving away)
    speed_m_s: float        # magnitude of (vx_m_s, vz_m_s)
    valid: bool = True       # False if there wasn't enough recent history to estimate


class RelativeVelocityEstimator:
    """Maintains a short position history per track_id and estimates velocity.

    Call `update()` once per frame with the current frame's BEV objects
    (each with a `track_id`, `x_m`, `z_m`) and a timestamp. Returns a
    velocity estimate per track_id that had enough history.
    """

    def __init__(
        self,
        history_length: int = 5,
        max_gap_seconds: float = 2.0,
        ema_alpha: float = 0.5,
    ) -> None:
        """
        Args:
            history_length: number of recent (timestamp, x_m, z_m) samples
                kept per track, used for a simple linear (oldest-to-newest)
                finite-difference velocity estimate.
            max_gap_seconds: if the time between a track's oldest and newest
                retained sample exceeds this, the estimate is considered
                unreliable (e.g. the track was lost and reacquired, or the
                pipeline stalled) and `valid=False` is returned instead of a
                number computed across too large/uncertain a gap.
            ema_alpha: exponential-moving-average smoothing weight applied
                to the raw finite-difference estimate, blended against the
                previous estimate for the same track, to reduce frame-to-
                frame jitter from BEV/depth noise. 1.0 = no smoothing.
        """
        self.history_length = history_length
        self.max_gap_seconds = max_gap_seconds
        self.ema_alpha = ema_alpha
        self._history: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self._smoothed: Dict[int, Tuple[float, float]] = {}

    def update(self, bev_objects: List, timestamp: Optional[float] = None) -> Dict[int, RelativeVelocity]:
        """Update position history for this frame's objects and return the
        current velocity estimate for every track_id present this frame
        (`valid=False` entries included, for tracks too new/gapped to
        estimate yet -- callers should check `.valid`).

        `bev_objects` is any iterable of objects with `.track_id`, `.x_m`,
        `.z_m` attributes -- i.e. `mapping.py::BEVObject`, passed by
        duck-typing so this module does not need to import mapping.py.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        results: Dict[int, RelativeVelocity] = {}
        seen_ids = set()

        for obj in bev_objects:
            track_id = getattr(obj, "track_id", None)
            if track_id is None:
                continue  # untracked detection (e.g. tracker not enabled) -- skip
            seen_ids.add(track_id)

            if track_id not in self._history:
                self._history[track_id] = deque(maxlen=self.history_length)
            self._history[track_id].append((timestamp, obj.x_m, obj.z_m))

            results[track_id] = self._estimate(track_id)

        self._prune_stale(seen_ids)
        return results

    def _estimate(self, track_id: int) -> RelativeVelocity:
        hist = self._history[track_id]
        if len(hist) < 2:
            return RelativeVelocity(track_id=track_id, vx_m_s=0.0, vz_m_s=0.0, speed_m_s=0.0, valid=False)

        t_old, x_old, z_old = hist[0]
        t_new, x_new, z_new = hist[-1]
        dt = t_new - t_old

        if dt <= 1e-6:
            return RelativeVelocity(track_id=track_id, vx_m_s=0.0, vz_m_s=0.0, speed_m_s=0.0, valid=False)
        if dt > self.max_gap_seconds:
            # Too large a gap between the oldest and newest retained sample
            # (e.g. this track was just re-acquired after a long occlusion)
            # -- report invalid rather than a velocity computed across an
            # unreliable time span.
            return RelativeVelocity(track_id=track_id, vx_m_s=0.0, vz_m_s=0.0, speed_m_s=0.0, valid=False)

        raw_vx = (x_new - x_old) / dt
        raw_vz = (z_new - z_old) / dt

        if track_id in self._smoothed:
            prev_vx, prev_vz = self._smoothed[track_id]
            vx = self.ema_alpha * raw_vx + (1 - self.ema_alpha) * prev_vx
            vz = self.ema_alpha * raw_vz + (1 - self.ema_alpha) * prev_vz
        else:
            vx, vz = raw_vx, raw_vz

        self._smoothed[track_id] = (vx, vz)
        speed = (vx ** 2 + vz ** 2) ** 0.5
        return RelativeVelocity(track_id=track_id, vx_m_s=vx, vz_m_s=vz, speed_m_s=speed, valid=True)

    def _prune_stale(self, seen_ids: set) -> None:
        """Drop history for tracks that no longer appear (avoids unbounded
        memory growth over a long-running session)."""
        stale = set(self._history.keys()) - seen_ids
        for track_id in stale:
            del self._history[track_id]
            self._smoothed.pop(track_id, None)
