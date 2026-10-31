"""Matching, estimator replay and metrics for the offline KITTI velocity evaluation.

The protocol these implement -- subsets, metrics, acceptance thresholds -- is
fixed in task5_evaluation_protocol.md; change it there first, never only here.

Estimators are replayed on cached per-frame pipeline outputs (detections after
tracking, BEV positions, metric depth), so every estimator sees identical
inputs. An estimator is wrapped in an adapter with
``process(frame: CachedFrame) -> {track_id: VelocityEstimate}`` and a
``reset()`` called at the start of every sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking import iou_batch
from ttc_calculator import compute_ttc

MATCH_IOU = 0.5
# Which pipeline (COCO) classes may match which KITTI types. Riders are detected as
# person and/or bicycle; vans and pick-ups often come out of COCO models as "truck".
COMPATIBLE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "Car": ("car", "truck", "bus"),
    "Van": ("car", "truck", "bus"),
    "Truck": ("truck", "bus", "car"),
    "Tram": ("bus", "truck", "train"),
    "Misc": ("car", "truck", "bus", "motorcycle", "bicycle"),
    "Pedestrian": ("person",),
    "Person": ("person",),
    "Cyclist": ("person", "bicycle", "motorcycle"),
}
TTC_BANDS = (("CRITICAL", 1.5), ("WARNING", 3.0), ("CAUTION", 5.0))   # config.yaml risk thresholds, clear weather
RANGE_BINS = ((3.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0))


# =============================================================================
# Cached inputs and estimator adapters
# =============================================================================


@dataclass
class CachedObject:
    """One tracked detection as the production pipeline produced it."""
    track_id: Optional[int]
    class_name: str
    box: Tuple[float, float, float, float]
    confidence: float
    has_bev: bool = False
    x_m: float = np.nan          # BEVMapper position (only when has_bev)
    z_m: float = np.nan


@dataclass
class CachedFrame:
    sequence: str
    frame: int
    timestamp: float
    objects: List[CachedObject]
    weather_label: str = "unknown"
    load_image: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False)
    load_depth: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False)
    online: Optional[dict] = field(default=None, repr=False)   # production estimator's output when cached
    camera: Optional[object] = field(default=None, repr=False)   # camera.CameraModel of the sequence
    _memo: dict = field(default_factory=dict, repr=False)

    @property
    def bev_objects(self) -> List[CachedObject]:
        """The objects that reached BEVMapper's output, in the pipeline's order (what the estimator received)."""
        return [o for o in self.objects if o.has_bev]

    def image(self) -> np.ndarray:
        if "image" not in self._memo:
            self._memo["image"] = self.load_image()
        return self._memo["image"]

    def gray(self) -> np.ndarray:
        if "gray" not in self._memo:
            import cv2
            self._memo["gray"] = cv2.cvtColor(self.image(), cv2.COLOR_BGR2GRAY)
        return self._memo["gray"]

    def depth(self) -> np.ndarray:
        """The cached metric depth map (float16 on disk, returned as float32)."""
        if "depth" not in self._memo:
            if self.load_depth is None:
                raise RuntimeError(f"{self.sequence}/{self.frame}: no cached depth map")
            self._memo["depth"] = self.load_depth()
        return self._memo["depth"]

    def release(self) -> None:
        self._memo.clear()


@dataclass
class VelocityEstimate:
    vx_m_s: float
    vz_m_s: float
    valid: bool
    sigma_vz: float = np.nan     # 1-sigma uncertainty of vz, for estimators that provide one
    z_m: float = np.nan          # the estimator's own distance, if it measures one


class B0Adapter:
    """The frozen production estimator (velocity_b0.py), fed exactly what the pipeline feeds it."""

    name = "B0"

    def __init__(self, history_length: int, max_gap_seconds: float, ema_alpha: float) -> None:
        self._kwargs = dict(history_length=history_length, max_gap_seconds=max_gap_seconds, ema_alpha=ema_alpha)
        self.reset()

    def reset(self) -> None:
        from velocity_b0 import RelativeVelocityEstimator
        self._estimator = RelativeVelocityEstimator(**self._kwargs)

    def process(self, frame: CachedFrame) -> Dict[int, VelocityEstimate]:
        out = self._estimator.update(frame.bev_objects, timestamp=frame.timestamp)
        return {tid: VelocityEstimate(v.vx_m_s, v.vz_m_s, v.valid) for tid, v in out.items()}


def replay(adapter, frames: Iterable[CachedFrame]) -> List[Dict[int, VelocityEstimate]]:
    """Run one estimator over one sequence's frames (in order) from a fresh state."""
    adapter.reset()
    return [adapter.process(f) for f in frames]


def replay_many(adapters: Mapping[str, object], frames: Sequence[CachedFrame]
                ) -> Tuple[Dict[str, List[Dict[int, VelocityEstimate]]], Dict[str, List[float]]]:
    """Run several estimators frame by frame over one sequence, each from a fresh state.

    Image and depth reads are done before the clock starts (in the live pipeline
    both are already in memory), so the per-frame times are compute only.
    Frame data is released after each frame to bound memory.
    """
    import time
    for a in adapters.values():
        a.reset()
    out = {name: [] for name in adapters}
    times = {name: [] for name in adapters}
    needs = {n for a in adapters.values() for n in getattr(a, "needs", ())}
    for f in frames:
        if "gray" in needs:
            f.gray()
        if "depth" in needs:
            f.depth()
        for name, a in adapters.items():
            t0 = time.perf_counter()
            out[name].append(a.process(f))
            times[name].append((time.perf_counter() - t0) * 1000.0)
        f.release()
    return out, times


# =============================================================================
# Matching pipeline objects to ground truth
# =============================================================================


def match_frame(det_boxes: np.ndarray, det_classes: Sequence[str], gt_boxes: np.ndarray, gt_types: Sequence[str],
                iou_threshold: float = MATCH_IOU) -> List[Tuple[int, int, float]]:
    """One-to-one matches (det_index, gt_index, iou): Hungarian on IoU over class-compatible pairs >= threshold."""
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return []
    iou = iou_batch(np.asarray(det_boxes, dtype=np.float32), np.asarray(gt_boxes, dtype=np.float32)).astype(np.float64)
    allowed = np.array([[dc in COMPATIBLE_CLASSES.get(gt, ()) for gt in gt_types] for dc in det_classes])
    score = np.where(allowed & (iou >= iou_threshold), iou, 0.0)
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if score[r, c] > 0]


# =============================================================================
# Metrics
# =============================================================================


def ttc_band(ttc_s: Optional[float]) -> str:
    """Band of a TTC under the clear-weather thresholds; NONE = not closing or TTC >= 5 s."""
    if ttc_s is None:
        return "NONE"
    for name, limit in TTC_BANDS:
        if ttc_s < limit:
            return name
    return "NONE"


def estimated_ttc(distance_m: float, vz_m_s: float) -> Optional[float]:
    return compute_ttc(distance_m, -vz_m_s)


def error_stats(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {"n": 0}
    a = np.abs(err)
    return {"n": int(err.size), "mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_ae": float(np.median(a)), "p90_ae": float(np.percentile(a, 90)), "bias": float(err.mean())}


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[str], n_resamples: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI of a pooled mean, resampling whole sequences (frames of one sequence are correlated)."""
    values, clusters = np.asarray(values, dtype=np.float64), np.asarray(clusters)
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_clusters": 0}
    names = np.unique(clusters)
    sums = np.array([values[clusters == c].sum() for c in names])
    counts = np.array([(clusters == c).sum() for c in names])
    picks = np.random.default_rng(seed).integers(0, len(names), size=(n_resamples, len(names)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 100 * alpha / 2)),
            "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))), "n_clusters": int(len(names))}


def paired_cluster_bootstrap(values_a: Sequence[float], values_b: Sequence[float], clusters: Sequence[str],
                             n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """CI of mean(b) - mean(a) and of the relative change (mean(b) - mean(a)) / mean(a), over paired
    observations, resampling whole sequences so that both methods always see the same resample."""
    a, b, clusters = (np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64),
                      np.asarray(clusters))
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs the same observations for both methods")
    names = np.unique(clusters)
    sa = np.array([a[clusters == c].sum() for c in names])
    sb = np.array([b[clusters == c].sum() for c in names])
    n = np.array([(clusters == c).sum() for c in names])
    picks = np.random.default_rng(seed).integers(0, len(names), size=(n_resamples, len(names)))
    ma, mb = sa[picks].sum(axis=1) / n[picks].sum(axis=1), sb[picks].sum(axis=1) / n[picks].sum(axis=1)
    diff, rel = mb - ma, (mb - ma) / ma
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "diff_ci_low": float(np.percentile(diff, lo)), "diff_ci_high": float(np.percentile(diff, hi)),
            "rel_change": float((b.mean() - a.mean()) / a.mean()),
            "rel_ci_low": float(np.percentile(rel, lo)), "rel_ci_high": float(np.percentile(rel, hi)),
            "n": int(a.size), "n_clusters": int(len(names))}


def calibration_stats(err: np.ndarray, sigma: np.ndarray) -> Dict[str, object]:
    """How well a stated 1-sigma uncertainty describes the actual closing-speed errors.

    For Gaussian errors with correct sigma, |e|/sigma <= 1 for 68.3% and <= 1.96 for 95%
    of rows, and the RMSE within each sigma quintile equals the mean sigma there."""
    ok = np.isfinite(sigma) & (sigma > 0)
    e, s = np.asarray(err)[ok], np.asarray(sigma)[ok]
    if e.size == 0:
        return {"n": 0}
    zs = np.abs(e) / s
    edges = np.quantile(s, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        bins.append({"sigma_lo": float(lo), "sigma_hi": float(hi), "n": int(m.sum()), "mean_sigma": float(s[m].mean()),
                     "rmse": float(np.sqrt(np.mean(e[m] ** 2))), "mae": float(np.mean(np.abs(e[m])))})
    return {"n": int(e.size), "frac_within_1sigma": float(np.mean(zs <= 1.0)),
            "frac_within_1.96sigma": float(np.mean(zs <= 1.96)), "median_abs_z": float(np.median(zs)),
            "spearman_sigma_vs_abs_err": float(_spearman(s, np.abs(e))), "by_sigma_quintile": bins}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def gating_curve(rows: List[Mapping], estimator: str, thresholds: Sequence[float]) -> List[Dict[str, float]]:
    """Coverage and closing-speed MAE when only estimates with sigma_vz <= threshold count as valid."""
    n_total = len(rows)
    out = []
    for tau in thresholds:
        keep = [r for r in rows if r["matched"] and r[f"{estimator}_valid"]
                and np.isfinite(r.get(f"{estimator}_sigma", np.nan)) and r[f"{estimator}_sigma"] <= tau]
        e = np.array([-r[f"{estimator}_vz"] - r["gt_closing"] for r in keep])
        out.append({"sigma_max": float(tau), "coverage_end_to_end": len(keep) / n_total if n_total else float("nan"),
                    "n": len(keep), "mae": float(np.mean(np.abs(e))) if e.size else float("nan")})
    return out


def summarize_rows(rows: List[Mapping], estimator: str, n_boot: int = 2000, seed: int = 0) -> Dict[str, object]:
    """Metrics for one estimator over evaluation rows (dicts from the harness's per-object table).

    Every row is an eligible ground-truth observation; ``matched`` says whether a
    pipeline track was matched to it and ``<est>_valid`` whether that estimator
    produced a valid velocity for the track in that frame.
    """
    n_total = len(rows)
    matched = [r for r in rows if r["matched"]]
    valid = [r for r in matched if r[f"{estimator}_valid"]]
    seqs = [r["sequence"] for r in valid]
    est_closing = np.array([-r[f"{estimator}_vz"] for r in valid])
    gt_closing = np.array([r["gt_closing"] for r in valid])
    e_close = est_closing - gt_closing
    e_lat = np.array([r[f"{estimator}_vx"] - r["gt_rel_vx"] for r in valid])

    out: Dict[str, object] = {
        "n_eligible_gt": n_total, "n_matched": len(matched), "n_valid": len(valid),
        "detection_rate": len(matched) / n_total if n_total else float("nan"),
        "coverage_end_to_end": len(valid) / n_total if n_total else float("nan"),
        "coverage_given_matched": len(valid) / len(matched) if matched else float("nan"),
        "closing_speed": error_stats(e_close), "lateral_speed": error_stats(e_lat),
        "closing_speed_mae_ci": cluster_bootstrap_mean(np.abs(e_close), seqs, n_boot, seed),
        "closing_speed_bias_ci": cluster_bootstrap_mean(e_close, seqs, n_boot, seed),
    }
    strong = np.abs(gt_closing) >= 1.0
    out["sign_accuracy_gt_ge_1ms"] = float(np.mean(np.sign(est_closing[strong]) == np.sign(gt_closing[strong]))) \
        if strong.any() else float("nan")
    out["n_sign_rows"] = int(strong.sum())

    z = np.array([r["gt_z_near"] for r in valid])
    out["by_range"] = {f"{lo:g}-{hi:g} m": error_stats(e_close[(z >= lo) & (z < hi)]) for lo, hi in RANGE_BINS}
    ego_moving = np.array([r["ego_moving"] for r in valid], dtype=bool)
    out["by_ego_motion"] = {"ego_moving": error_stats(e_close[ego_moving]),
                            "ego_stationary": error_stats(e_close[~ego_moving])}
    turning = np.array([r["ego_turning"] for r in valid], dtype=bool)
    out["by_turning"] = {"turning": error_stats(e_close[turning]), "straight": error_stats(e_close[~turning])}
    occ = np.array([r["occluded"] for r in valid])
    out["by_occlusion"] = {f"occluded={k}": error_stats(e_close[occ == k]) for k in (0, 1)}

    # TTC bands under the pipeline's own TTC function and clear-weather thresholds, for objects in the
    # ego corridor only (cars parked beside the road give short "TTCs" that no risk logic should act on).
    bands = [n for n, _ in TTC_BANDS] + ["NONE"]
    conf = {g: {e: 0 for e in bands} for g in bands}
    in_path = [r for r in valid if r["gt_in_corridor"]]
    for r in in_path:
        g = ttc_band(r["gt_ttc"])
        e = ttc_band(estimated_ttc(r["pipeline_z_m"], r[f"{estimator}_vz"]))
        conf[g][e] += 1
    gt_none = sum(conf["NONE"].values())
    gt_urgent = sum(sum(conf[g].values()) for g in ("CRITICAL", "WARNING"))
    out["ttc_band_confusion"] = conf
    out["false_urgent_rate"] = (conf["NONE"]["CRITICAL"] + conf["NONE"]["WARNING"]) / gt_none if gt_none else float("nan")
    out["false_critical_rate"] = conf["NONE"]["CRITICAL"] / gt_none if gt_none else float("nan")
    out["missed_urgent_rate"] = sum(conf[g]["NONE"] + conf[g]["CAUTION"] for g in ("CRITICAL", "WARNING")) / gt_urgent \
        if gt_urgent else float("nan")
    out["n_gt_urgent"] = gt_urgent
    out["n_in_corridor"] = len(in_path)
    out["band_agreement"] = sum(conf[b][b] for b in bands) / len(in_path) if in_path else float("nan")

    # Secondary (protocol amendment 1): bands with the estimator's own distance, when it measures one.
    own = [r for r in in_path if np.isfinite(r.get(f"{estimator}_z", np.nan))]
    if own:
        conf_own = {g: {e: 0 for e in bands} for g in bands}
        for r in own:
            conf_own[ttc_band(r["gt_ttc"])][ttc_band(estimated_ttc(r[f"{estimator}_z"], r[f"{estimator}_vz"]))] += 1
        none_own = sum(conf_own["NONE"].values())
        urgent_own = sum(sum(conf_own[g].values()) for g in ("CRITICAL", "WARNING"))
        out["own_distance_ttc"] = {
            "n": len(own),
            "false_urgent_rate": (conf_own["NONE"]["CRITICAL"] + conf_own["NONE"]["WARNING"]) / none_own
            if none_own else float("nan"),
            "missed_urgent_rate": sum(conf_own[g]["NONE"] + conf_own[g]["CAUTION"] for g in ("CRITICAL", "WARNING"))
            / urgent_own if urgent_own else float("nan"),
            "band_agreement": sum(conf_own[b][b] for b in bands) / len(own)}
        zo = np.array([r[f"{estimator}_z"] for r in valid if np.isfinite(r.get(f"{estimator}_z", np.nan))])
        zg = np.array([r["gt_z_near"] for r in valid if np.isfinite(r.get(f"{estimator}_z", np.nan))])
        out["own_distance_abs_rel_vs_z_near"] = float(np.mean(np.abs(zo - zg) / zg))

    sig = np.array([r.get(f"{estimator}_sigma", np.nan) for r in valid], dtype=np.float64)
    if np.isfinite(sig).any():
        out["calibration"] = calibration_stats(e_close, sig)

    # Distance input quality (the same for every estimator that uses the pipeline's BEV positions).
    zp = np.array([r["pipeline_z_m"] for r in matched])
    zn = np.array([r["gt_z_near"] for r in matched])
    good = np.isfinite(zp)
    out["distance_abs_rel_vs_z_near"] = float(np.mean(np.abs(zp[good] - zn[good]) / zn[good])) if good.any() \
        else float("nan")
    return out
