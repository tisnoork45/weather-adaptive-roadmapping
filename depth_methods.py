"""Ways of turning Depth Anything V2 output into metres, compared in the KITTI evaluation.

* ``LegacyLinearConversion`` -- a frozen copy of the conversion the pipeline
  used up to commit 0be283f (DepthEstimator._real_infer): min/max-normalise the
  relative output (with EMA-smoothed bounds) and map it linearly onto
  0-80 m, near = 0. It exists only to measure how wrong that was; the
  pipeline no longer uses it.
* ``align_inverse_depth`` -- per-frame least-squares scale and shift of the
  relative (inverse-depth) output against ground truth. Uses the evaluation
  ground truth, so it is an ORACLE upper bound, never a deployable method.
* ``ground_plane_scale`` -- deployable baseline: recover scale and shift of the
  relative output from flat-road geometry, using only the camera matrix and
  the documented camera height. No ground truth is used.
* ``focal_rescale`` -- metric output rescaled by the ratio of this camera's
  focal length to the metric model's training camera (an ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Virtual KITTI 2 camera (the Depth Anything V2 Metric Outdoor training data): fx = fy = 725.0087 px.
VKITTI2_FOCAL_PX = 725.0087


class LegacyLinearConversion:
    """The pre-Task-4 relative-to-"metres" mapping, kept verbatim for comparison.

    ``alpha=1.0`` uses each frame's own min/max (what the pipeline did on its
    first frame and for any isolated image); ``alpha=0.3`` with a persistent
    instance reproduces its smoothing across a sequence.
    """

    def __init__(self, max_depth_m: float = 80.0, alpha: float = 0.3) -> None:
        self.max_depth_m, self.alpha = max_depth_m, alpha
        self._ema_min: Optional[float] = None
        self._ema_max: Optional[float] = None

    def __call__(self, raw_depth: np.ndarray) -> np.ndarray:
        raw_depth = np.asarray(raw_depth, dtype=np.float32)
        d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(raw_depth)
        if self._ema_min is None:
            self._ema_min, self._ema_max = d_min, d_max
        else:
            self._ema_min = self.alpha * d_min + (1 - self.alpha) * self._ema_min
            self._ema_max = self.alpha * d_max + (1 - self.alpha) * self._ema_max
        norm_min, norm_max = self._ema_min, self._ema_max
        if norm_max - norm_min < 1e-6:
            norm_min, norm_max = d_min, d_max
        normalized = np.clip((raw_depth - norm_min) / (norm_max - norm_min), 0.0, 1.0)
        return (1.0 - normalized) * self.max_depth_m


def _inverse_to_depth(inv: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """1/inv, with non-positive or tiny inverse depth mapped to max_depth (i.e. "far")."""
    out = np.full(inv.shape, max_depth, dtype=np.float64)
    ok = inv > 1.0 / max_depth
    out[ok] = 1.0 / inv[ok]
    return np.clip(out, min_depth, max_depth).astype(np.float32)


def align_inverse_depth(raw_relative: np.ndarray, target_depth_m: np.ndarray, min_depth: float = 1e-3,
                        max_depth: float = 80.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """ORACLE: least-squares s, t with s*d + t ~= 1/target on target's valid (>0) pixels."""
    valid = (target_depth_m > min_depth) & (target_depth_m <= max_depth)
    if valid.sum() < 2:
        raise ValueError("Need at least two valid target pixels to align.")
    d = raw_relative[valid].astype(np.float64)
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, 1.0 / target_depth_m[valid].astype(np.float64), rcond=None)
    pred = _inverse_to_depth(s * raw_relative.astype(np.float64) + t, min_depth, max_depth)
    return pred, {"scale": float(s), "shift": float(t), "n_fit_pixels": int(valid.sum())}


def focal_rescale(metric_depth_m: np.ndarray, fx: float, train_fx: float = VKITTI2_FOCAL_PX,
                  max_depth: float = 80.0) -> np.ndarray:
    """Metric output scaled by fx / train_fx: a narrower lens (larger fx) sees objects larger,
    which a focal-unaware model reads as nearer; this undoes that to first order."""
    return np.clip(metric_depth_m * (fx / train_fx), 1e-3, max_depth).astype(np.float32)


@dataclass
class GroundPlaneFit:
    ok: bool
    slope: float = float("nan")       # k in d = k*v + c over road rows
    intercept: float = float("nan")   # c
    scale: float = float("nan")       # a in d = a/Z + b
    shift: float = float("nan")       # b
    n_rows: int = 0
    r2: float = float("nan")          # how well the line explains the road-region rows
    reason: str = ""


def ground_plane_scale(raw_relative: np.ndarray, K: np.ndarray, camera_height_m: float,
                       roi_rows: Tuple[float, float] = (0.72, 0.98), roi_cols: Tuple[float, float] = (0.40, 0.60),
                       min_depth: float = 1e-3, max_depth: float = 80.0, min_r2: float = 0.5) -> Tuple[Optional[np.ndarray], GroundPlaneFit]:
    """Metric depth from relative inverse depth using flat-road geometry. No ground truth used.

    For a level camera at height h above a flat road, a road pixel in row v
    lies at depth Z = fy*h / (v - cy), i.e. 1/Z is linear in v. If the model's
    output is affine in inverse depth, d = a/Z + b, then over road rows
    d = k*v + c with k = a/(fy*h) and c = b - k*cy. Fitting k and c on a fixed
    region just ahead of the car gives a = k*fy*h and b = c + k*cy, and
    Z = a / (d - b) everywhere. Assumes zero pitch/roll (horizon at cy) and
    that the region is mostly road; frames where the fit is implausible
    (output not increasing towards the camera, or R^2 < min_r2 -- a threshold
    fixed before any evaluation) are reported as failures rather than filled in.
    """
    h, w = raw_relative.shape
    r0, r1 = int(roi_rows[0] * h), int(roi_rows[1] * h)
    c0, c1 = int(roi_cols[0] * w), int(roi_cols[1] * w)
    fy, cy = float(K[1, 1]), float(K[1, 2])
    rows = np.arange(r0, r1)
    rows = rows[rows > cy + 1]                               # only rows below the horizon
    if rows.size < 10:
        return None, GroundPlaneFit(ok=False, reason="region of interest is not below the horizon")
    d_rows = np.median(raw_relative[rows, c0:c1].astype(np.float64), axis=1)

    keep = np.ones(rows.size, dtype=bool)
    for _ in range(3):                                        # least squares with MAD outlier rejection
        k, c = np.polyfit(rows[keep], d_rows[keep], 1)
        resid = d_rows - (k * rows + c)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-12
        new_keep = np.abs(resid) <= 3.0 * 1.4826 * mad
        if new_keep.sum() < 10 or (new_keep == keep).all():
            break
        keep = new_keep

    fitted = d_rows[keep]
    ss_tot = float(np.sum((fitted - fitted.mean()) ** 2))
    ss_res = float(np.sum((fitted - (k * rows[keep] + c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 * max(float(np.mean(fitted ** 2)), 1e-300) else float("nan")
    common = dict(slope=float(k), intercept=float(c), n_rows=int(keep.sum()), r2=float(r2))
    if not np.isfinite(r2):
        return None, GroundPlaneFit(ok=False, reason="no depth variation in the road region", **common)
    if not np.isfinite(k) or k <= 0:
        return None, GroundPlaneFit(ok=False, reason="output does not increase towards the camera on the road region",
                                    **common)
    if r2 < min_r2:
        return None, GroundPlaneFit(ok=False, reason=f"road region is not explained by a flat plane (R^2={r2:.2f})",
                                    **common)
    a = k * fy * camera_height_m
    b = c + k * cy
    pred = _inverse_to_depth((raw_relative.astype(np.float64) - b) / a, min_depth, max_depth)
    return pred, GroundPlaneFit(ok=True, scale=float(a), shift=float(b), **common)
