"""Depth-accuracy metrics for evaluating predicted depth maps against ground truth.

Two protocols, kept separate and labelled because their definitions differ:

* ``devkit_metrics`` -- reproduces the official KITTI depth-prediction devkit
  (devkit/cpp/evaluate_depth.cpp + io_depth.h, 2017): predictions are quantised
  to the uint16/256 PNG format, invalid (0) predictions are filled with the
  devkit's ``interpolateBackground``, every pixel with ground truth > 0 is
  scored, there is no depth cap, and SqRel is mean(err^2 / gt^2).
* ``standard_metrics`` -- the metrics most depth papers report (Eigen et al.
  2014 and successors): ground truth limited to (min_depth, max_depth],
  predictions clamped to the same range, SqRel = mean(err^2 / gt), SILog in
  both the x100 benchmark form and the lambda = 0.5 form used by the Depth
  Anything V2 codebase, plus delta < 1.25^k accuracies.

Both are computed per image; ``aggregate`` averages over images, which is how
the devkit's statMean reports results. These are *pixel-wise depth* metrics:
they say nothing on their own about the accuracy of object distances.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

DEVKIT_SCALE = 256.0  # KITTI depth PNGs store metres * 256 as uint16; 0 = invalid


def quantize_like_devkit_png(depth_m: np.ndarray) -> np.ndarray:
    """Round-trip a depth map through the devkit's uint16 PNG encoding.

    Values that would encode as 0 come back as 0 (invalid), exactly as the
    devkit would read them.
    """
    encoded = np.clip(np.round(np.asarray(depth_m, dtype=np.float64) * DEVKIT_SCALE), 0, 65535)
    return encoded.astype(np.uint16).astype(np.float64) / DEVKIT_SCALE


def interpolate_background(depth: np.ndarray) -> np.ndarray:
    """Fill invalid pixels (< 0) the way the devkit's DepthImage::interpolateBackground does.

    Per row: interior runs of invalid pixels get the minimum of their two valid
    neighbours; leading/trailing runs copy the nearest valid pixel. Then per
    column: rows still invalid (no valid pixel in the whole row) copy the
    nearest valid pixel above/below. Returns a new array; the input is unchanged.
    """
    d = np.array(depth, dtype=np.float64, copy=True)
    height, width = d.shape
    if (d >= 0).all():
        return d  # dense prediction: nothing to fill

    for v in range(height):
        row = d[v]
        valid = row >= 0
        if not valid.any():
            continue
        idx = np.flatnonzero(valid)
        # Interior gaps between consecutive valid pixels (loop over gaps only).
        for g in np.flatnonzero(np.diff(idx) > 1):
            left, right = idx[g], idx[g + 1]
            row[left + 1:right] = min(row[left], row[right])
        row[:idx[0]] = row[idx[0]]
        row[idx[-1] + 1:] = row[idx[-1]]

    for u in range(width):
        col = d[:, u]
        valid = col >= 0
        if not valid.any():
            continue
        idx = np.flatnonzero(valid)
        col[:idx[0]] = col[idx[0]]
        col[idx[-1] + 1:] = col[idx[-1]]
    return d


def devkit_metrics(pred_m: np.ndarray, gt_m: np.ndarray) -> Dict[str, float]:
    """Per-image errors exactly as defined in the KITTI devkit's depthError().

    ``gt_m`` uses 0 for "no ground truth". ``pred_m`` is quantised to the PNG
    format first; any pixel that becomes 0 is treated as invalid and filled.
    Accumulation is in float64 (the C++ uses float32), so results can differ
    from the binary in the last few digits.
    """
    if pred_m.shape != gt_m.shape:
        raise ValueError(f"Prediction shape {pred_m.shape} != ground-truth shape {gt_m.shape}")
    pred = quantize_like_devkit_png(pred_m)
    pred[pred == 0] = -1.0
    pred = interpolate_background(pred)

    valid = gt_m > 0
    n = int(valid.sum())
    if n == 0:
        raise ValueError("Ground truth has no valid pixels.")
    gt = gt_m[valid].astype(np.float64)
    p = pred[valid]

    err = np.abs(gt - p)
    err_inv = np.abs(1.0 / gt - 1.0 / p)
    log_diff = np.log(gt) - np.log(p)
    mean_sq_log = float(np.mean(log_diff ** 2))
    silog = float(np.sqrt(max(mean_sq_log - float(np.mean(log_diff)) ** 2, 0.0)))
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "inv_mae": float(np.mean(err_inv)),
        "inv_rmse": float(np.sqrt(np.mean(err_inv ** 2))),
        "log_mae": float(np.mean(np.abs(log_diff))),
        "log_rmse": float(np.sqrt(mean_sq_log)),
        "silog": silog,
        "silog_x100": 100.0 * silog,
        "abs_rel": float(np.mean(err / gt)),
        "sq_rel_devkit": float(np.mean(err ** 2 / gt ** 2)),
        "n_pixels": n,
    }


def standard_metrics(pred_m: np.ndarray, gt_m: np.ndarray, min_depth: float = 1e-3,
                     max_depth: float = 80.0) -> Dict[str, float]:
    """Per-image standard depth metrics on ground truth in (min_depth, max_depth]."""
    if pred_m.shape != gt_m.shape:
        raise ValueError(f"Prediction shape {pred_m.shape} != ground-truth shape {gt_m.shape}")
    valid = (gt_m > min_depth) & (gt_m <= max_depth)
    n = int(valid.sum())
    if n == 0:
        raise ValueError("Ground truth has no pixels inside the evaluation range.")
    gt = gt_m[valid].astype(np.float64)
    p = np.clip(pred_m[valid].astype(np.float64), min_depth, max_depth)

    thresh = np.maximum(gt / p, p / gt)
    diff = p - gt
    log_diff = np.log(p) - np.log(gt)
    mean_sq_log = float(np.mean(log_diff ** 2))
    mean_log = float(np.mean(log_diff))
    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "sq_rel": float(np.mean(diff ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(mean_sq_log)),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(gt)))),
        "silog_x100": 100.0 * float(np.sqrt(max(mean_sq_log - mean_log ** 2, 0.0))),
        "silog_dav2": float(np.sqrt(max(mean_sq_log - 0.5 * mean_log ** 2, 0.0))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
        "n_pixels": n,
    }


def range_bin_metrics(pred_m: np.ndarray, gt_m: np.ndarray,
                      edges: Sequence[float] = (0.0, 10.0, 20.0, 40.0, 80.0)) -> Dict[str, Dict[str, float]]:
    """Standard metrics restricted to ground-truth depth bins (lo, hi]; empty bins are omitted."""
    out: Dict[str, Dict[str, float]] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        if ((gt_m > max(lo, 1e-3)) & (gt_m <= hi)).any():
            out[f"{lo:g}-{hi:g}m"] = standard_metrics(pred_m, gt_m, min_depth=max(lo, 1e-3), max_depth=hi)
    return out


def aggregate(per_image: Iterable[Mapping[str, float]], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Mean of each metric over images (pixel counts are summed instead)."""
    rows: List[Mapping[str, float]] = list(per_image)
    if not rows:
        return {}
    keys = list(keys or rows[0].keys())
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(vals.sum()) if k == "n_pixels" else float(vals.mean())
    return out


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], n_resamples: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI for the mean of per-image values, resampling whole clusters (e.g. drives).

    Frames from one drive (and left/right camera pairs) are strongly correlated,
    so resampling images independently would understate the uncertainty.
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    names = np.unique(clusters)
    sums = np.array([values[clusters == c].sum() for c in names])
    counts = np.array([(clusters == c).sum() for c in names])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(n_resamples, len(names)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n_clusters": int(len(names)),
    }
