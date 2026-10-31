"""Offline velocity evaluation on the KITTI tracking training set (Task 5).

Two steps, so that every velocity estimator is compared on identical inputs:

1. ``cache``: run the unchanged production PerceptionPipeline (metric depth,
   strict) once per frame of every sequence and store what it produced --
   raw detections, tracked detections, BEV positions, the online velocity
   estimates of the production estimator, the weather label and the metric
   depth map (float16). A fresh pipeline is built for every sequence, as if
   the app were started on it. Frames are timestamped t = frame * period,
   the nominal 0.1 s unless --frame-period says otherwise (see kitti_tracking.py
   for why the true spacing is ~0.1035 s).

2. ``evaluate``: replay estimators (the frozen B0 and, with --estimators, the
   experimental ones in velocity_experimental.py) on the cache, check that B0's
   replay reproduces the online production estimates exactly, match pipeline
   tracks to KITTI ground truth, and write per-object rows and summaries with
   sequence-level bootstrap CIs, paired comparisons with B0, latency and
   uncertainty calibration. Experimental estimators may not be scored on the
   test split without --score-test (protocol section 2).

Protocol (subsets, metrics, thresholds, dev/test split):
task5_evaluation_protocol.md. Outputs go outside the repository by default
(/workspace/experiments/kitti_velocity/...): KITTI is CC BY-NC-SA 3.0, and
nothing derived from it may be committed.

Usage:
    python -B evaluate_kitti_velocity.py cache [--sequences 0000 0001] [--limit-frames 20]
    python -B evaluate_kitti_velocity.py evaluate --cache <cache dir> [--split dev|test|all]
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from kitti_tracking import KITTI_TRACKING_FPS, KITTI_TRACKING_ROOT, KittiTrackingDataset, KittiTrackingSequence
from models import DEPTH_KIND_METRIC
from kitti_velocity_gt import ego_kinematics, is_primary_eligible, is_vru_eligible, object_kinematics
from utils import get_logger, load_config
from velocity_eval import (
    CachedFrame, CachedObject, gating_curve, match_frame, paired_cluster_bootstrap, replay_many, summarize_rows,
)
from velocity_experimental import build_estimators

logger = get_logger(__name__)

DEFAULT_OUT_ROOT = Path("/workspace/experiments/kitti_velocity")
PROTOCOL_FILE = Path(__file__).with_name("task5_evaluation_protocol.md")
SPLITS = {"dev": lambda name: int(name) % 2 == 0, "test": lambda name: int(name) % 2 == 1, "all": lambda name: True}
EGO_MOVING_M_S = 1.0         # |camera forward speed| above which the ego vehicle counts as moving
EGO_TURNING_RAD_S = 0.1      # |yaw rate| above which it counts as turning (~5.7 deg/s)
CACHE_FORMAT = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> Dict[str, object]:
    """Commit of the project; $PROJECT_GIT_DIR points at the repository when running from a copy of it."""
    repo = os.environ.get("PROJECT_GIT_DIR", str(Path(__file__).parent))

    def run(*cmd):
        return subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True, check=False).stdout.strip()
    changed = [line for line in run("status", "--porcelain").splitlines() if line]
    return {"repo": repo, "commit": run("rev-parse", "HEAD"), "dirty": bool(changed), "changed_files": changed}


SOURCE_FILES = ("evaluate_kitti_velocity.py", "velocity_eval.py", "kitti_velocity_gt.py", "kitti_tracking.py",
                "velocity_b0.py", "relative_velocity.py", "pipeline.py", "mapping.py", "tracking.py", "models.py",
                "camera.py", "ttc_calculator.py", "risk_engine.py", "fusion.py", "utils.py", "config.yaml")


def source_hashes() -> Dict[str, str]:
    """SHA-256 of the code that produced a run (works without git, e.g. from a sandbox copy)."""
    here = Path(__file__).parent
    return {name: sha256_file(here / name) for name in SOURCE_FILES if (here / name).exists()}


def versions() -> Dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": str(torch.version.cuda),
            "numpy": np.__version__, "opencv": cv2.__version__}


# =============================================================================
# Step 1: cache the production pipeline's per-frame outputs
# =============================================================================


class _RecordingDetector:
    """Passes everything through to the real detector and keeps a copy of its raw output
    (the tracker later assigns track IDs to, and drops, detections in place)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last: List = []

    def infer(self, frame):
        result = self.inner.infer(frame)
        self.last = [copy.deepcopy(d) for d in result.detections]
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def evaluation_config():
    """config.yaml with the two in-memory overrides the evaluation needs; the file itself is not touched."""
    cfg = load_config("config.yaml")
    overrides = {"depth_estimation.mode": "metric", "pipeline.strict": True}
    cfg.depth_estimation.mode = "metric"
    cfg.pipeline.strict = True
    return cfg, overrides


def _det_dict(d) -> dict:
    return {"box": [float(v) for v in d.box], "class_id": int(d.class_id), "class_name": d.class_name,
            "confidence": float(d.confidence), "track_id": d.track_id}


def _align_bev(detections, bev_objects) -> List[int]:
    """Index into bev_objects for each detection (-1 if BEVMapper skipped it).

    BEVMapper emits objects in detection order and only skips detections with
    no usable depth, so an in-order walk recovers the correspondence; it is
    checked to consume every BEV object."""
    out, j = [], 0
    for d in detections:
        if j < len(bev_objects) and (bev_objects[j].track_id, bev_objects[j].class_name, bev_objects[j].confidence) \
                == (d.track_id, d.class_name, d.confidence):
            out.append(j)
            j += 1
        else:
            out.append(-1)
    if j != len(bev_objects):
        raise RuntimeError("could not align BEV objects with detections")
    return out


def cache_sequence(seq: KittiTrackingSequence, cfg, out_dir: Path, limit_frames: int, save_depth: bool) -> dict:
    from pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(cfg)
    if pipeline.depth_estimator.output_kind != DEPTH_KIND_METRIC:
        raise RuntimeError("the evaluation needs metric depth, but the depth estimator is not in metric mode")
    recorder = _RecordingDetector(pipeline.detector)
    pipeline.detector = recorder

    n = min(seq.n_frames, limit_frames) if limit_frames else seq.n_frames
    w, h = seq.image_size
    out_dir.mkdir(parents=True, exist_ok=False)
    depth_mm = np.lib.format.open_memmap(out_dir / "depth_f16.npy", mode="w+", dtype=np.float16,
                                         shape=(n, h, w)) if save_depth else None
    frames, latencies = [], []
    for k in range(n):
        t = float(seq.timestamps[k])
        image = seq.load_image(k)
        start = time.perf_counter()
        out = pipeline.process(image, timestamp=t, camera=seq.camera)
        latencies.append((time.perf_counter() - start) * 1000.0)
        dets = out.detections.detections
        bev = out.bev.objects if out.bev is not None else []
        if out.depth.depth_map.shape != (h, w):
            raise RuntimeError(f"{seq.name}/{k}: depth map {out.depth.depth_map.shape} != image {(h, w)}")
        if depth_mm is not None:
            depth_mm[k] = out.depth.depth_map.astype(np.float16)
        bev_idx = _align_bev(dets, bev)
        objects = []
        for d, j in zip(dets, bev_idx):
            obj = _det_dict(d)
            obj["bev"] = None if j < 0 else {"x_m": bev[j].x_m, "z_m": bev[j].z_m}
            objects.append(obj)
        frames.append({
            "frame": k, "timestamp": t, "weather": out.weather.label, "depth_kind": out.depth.kind,
            "raw_detections": [_det_dict(d) for d in recorder.last],
            "objects": objects,
            "online_velocities": {str(tid): [v.vx_m_s, v.vz_m_s, v.speed_m_s, bool(v.valid)]
                                  for tid, v in out.relative_velocities.items()},
            "latency_ms": latencies[-1],
        })
        if k % 100 == 0:
            logger.info("cache %s: frame %d/%d (%.0f ms)", seq.name, k, n, latencies[-1])
    if depth_mm is not None:
        depth_mm.flush()
        del depth_mm
    with gzip.open(out_dir / "frames.json.gz", "wt") as fh:
        json.dump({"sequence": seq.name, "n_frames": n, "image_size": [w, h], "frames": frames}, fh)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"n_frames": n, "complete": n == seq.n_frames, "image_size": [w, h],
            "camera": {"fx": seq.camera.fx, "fy": seq.camera.fy, "cx": seq.camera.cx, "cy": seq.camera.cy},
            "latency_ms_median": float(np.median(latencies)), "latency_ms_mean": float(np.mean(latencies))}


def cmd_cache(args) -> None:
    cfg, overrides = evaluation_config()
    met = Path(cfg.depth_estimation.metric_weights_path)
    met_sha = sha256_file(met)
    if met_sha != cfg.depth_estimation.metric_sha256:
        raise RuntimeError(f"Metric checkpoint hash {met_sha} != recorded {cfg.depth_estimation.metric_sha256}")
    dataset = KittiTrackingDataset(args.root, sequences=args.sequences, frame_period_s=args.frame_period)
    git = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + (git["commit"][:7] or "nogit") \
        + ("-dirty" if git["dirty"] else "") + (f"-limit{args.limit_frames}" if args.limit_frames else "")
    out = Path(args.out or DEFAULT_OUT_ROOT / "cache" / run_id)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "cache_format": CACHE_FORMAT, "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "git": git, "source_sha256": source_hashes(), "versions": versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_root": str(dataset.root), "frame_period_s": dataset.frame_period_s,
        "timestamps": "t = frame * frame_period_s (reconstructed; the benchmark has no timestamp files)",
        "config_overrides_in_memory": overrides,
        "config": {k: dict(cfg[k]) for k in ("detection", "tracking", "depth_estimation", "velocity", "risk", "mapping")},
        "checkpoints": {"detection": {"path": cfg.detection.weights_path,
                                      "sha256": sha256_file(Path(cfg.detection.weights_path))},
                        "depth_metric": {"path": str(met), "resolved": str(met.resolve()), "sha256": met_sha},
                        "weather": {"path": cfg.weather_classification.weights_path,
                                    "sha256": sha256_file(Path(cfg.weather_classification.weights_path))}},
        "depth_cache": "float16 metric depth per frame (depth_f16.npy); BEV positions were computed from the "
                       "float32 map before quantisation" if not args.no_depth else "not saved",
        "licence": "KITTI, CC BY-NC-SA 3.0 -- never commit anything in this directory",
        "sequences": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for seq in dataset:
        t0 = time.perf_counter()
        info = cache_sequence(seq, cfg, out / seq.name, args.limit_frames, save_depth=not args.no_depth)
        info["wall_s"] = time.perf_counter() - t0
        manifest["sequences"][seq.name] = info
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("cached %s: %d frames in %.0f s", seq.name, info["n_frames"], info["wall_s"])
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out)


# =============================================================================
# Step 2: evaluate estimators on the cache
# =============================================================================


def load_cached_frames(cache_dir: Path, seq: KittiTrackingSequence) -> List[CachedFrame]:
    with gzip.open(cache_dir / seq.name / "frames.json.gz", "rt") as fh:
        data = json.load(fh)
    depth_path = cache_dir / seq.name / "depth_f16.npy"
    depth = np.load(depth_path, mmap_mode="r") if depth_path.exists() else None
    frames = []
    for f in data["frames"]:
        k = f["frame"]
        if f["timestamp"] != float(seq.timestamps[k]):
            raise RuntimeError(f"{seq.name}/{k}: cached timestamp {f['timestamp']} != {seq.timestamps[k]}")
        objects = [CachedObject(track_id=o["track_id"], class_name=o["class_name"], box=tuple(o["box"]),
                                confidence=o["confidence"], has_bev=o["bev"] is not None,
                                x_m=o["bev"]["x_m"] if o["bev"] else np.nan,
                                z_m=o["bev"]["z_m"] if o["bev"] else np.nan) for o in f["objects"]]
        frames.append(CachedFrame(
            sequence=seq.name, frame=k, timestamp=f["timestamp"], objects=objects, weather_label=f["weather"],
            load_image=(lambda k=k: seq.load_image(k)),
            load_depth=None if depth is None else (lambda k=k: np.asarray(depth[k], dtype=np.float32)),
            online=f["online_velocities"], camera=seq.camera,
        ))
    return frames


def check_b0_equivalence(frames: List[CachedFrame], b0_out) -> Dict[str, int]:
    """B0 replayed on the cache must reproduce the production estimator's online output bit for bit."""
    n_values, mismatches = 0, 0
    for f, est in zip(frames, b0_out):
        online = {int(t): v for t, v in f.online.items()}
        if set(online) != set(est):
            mismatches += 1
            continue
        for tid, (vx, vz, _speed, valid) in online.items():
            e = est[tid]
            n_values += 1
            if not (e.vx_m_s == vx and e.vz_m_s == vz and e.valid == valid):
                mismatches += 1
    return {"n_compared": n_values, "n_mismatched": mismatches}


def evaluate_sequence(seq: KittiTrackingSequence, frames: List[CachedFrame], estimates: Dict[str, list]) -> List[dict]:
    gt_rows = object_kinematics(seq)
    ego = ego_kinematics(seq)
    by_frame: Dict[int, list] = {}
    for r in gt_rows:
        by_frame.setdefault(r.frame, []).append(r)
    rows = []
    for f in frames:
        gts = by_frame.get(f.frame, [])
        dets = [o for o in f.objects if o.track_id is not None]
        matches = match_frame(np.array([o.box for o in dets]).reshape(-1, 4), [o.class_name for o in dets],
                              np.array([g.bbox for g in gts]).reshape(-1, 4), [g.type for g in gts])
        det_for_gt = {gi: (di, iou) for di, gi, iou in matches}
        e = ego[f.frame]
        for gi, g in enumerate(gts):
            subset = "primary" if is_primary_eligible(g) else "vru" if is_vru_eligible(g) else None
            if subset is None:
                continue
            row = {
                "sequence": seq.name, "frame": f.frame, "subset": subset, "gt_track": g.track_id, "type": g.type,
                "occluded": g.occluded, "truncated": g.truncated, "gt_x": g.x, "gt_z": g.z, "gt_z_near": g.z_near,
                "gt_rel_vx": g.rel_vx, "gt_rel_vz": g.rel_vz, "gt_closing": g.closing_speed,
                "gt_rel_vz_cd": g.rel_vz_cd, "gt_abs_vx": g.abs_vx, "gt_abs_vz": g.abs_vz, "gt_ttc": g.ttc,
                "gt_in_corridor": g.in_corridor, "ego_vz": e.vz, "ego_yaw_rate": e.yaw_rate,
                "ego_moving": bool(abs(e.vz) > EGO_MOVING_M_S) if e.has_window else False,
                "ego_turning": bool(abs(e.yaw_rate) > EGO_TURNING_RAD_S) if e.has_window else False,
                "weather": f.weather_label, "matched": gi in det_for_gt,
            }
            det = dets[det_for_gt[gi][0]] if gi in det_for_gt else None
            row.update({"det_track": det.track_id if det else None, "det_class": det.class_name if det else None,
                        "iou": det_for_gt[gi][1] if det else np.nan, "has_bev": bool(det and det.has_bev),
                        "pipeline_x_m": det.x_m if det else np.nan, "pipeline_z_m": det.z_m if det else np.nan})
            for name, per_frame in estimates.items():
                v = per_frame[f.frame].get(det.track_id) if det else None
                row[f"{name}_valid"] = bool(v is not None and v.valid)
                row[f"{name}_vx"] = v.vx_m_s if v is not None and v.valid else np.nan
                row[f"{name}_vz"] = v.vz_m_s if v is not None and v.valid else np.nan
                row[f"{name}_sigma"] = v.sigma_vz if v is not None and v.valid else np.nan
                row[f"{name}_z"] = v.z_m if v is not None and v.valid else np.nan
            rows.append(row)
    return rows


def comparison_table(summaries: dict, paired: dict, latency: dict, subset: str) -> List[str]:
    """One row per estimator: the headline numbers next to each other (paired change vs B0 included)."""
    lines = [f"### {subset}: estimators side by side", "",
             "| estimator | coverage | closing MAE [95% CI] | paired MAE change vs B0 [95% CI] (n) | median AE | "
             "bias | lateral MAE | sign acc | false-urgent | missed-urgent | within 1σ / 1.96σ | ms/frame |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in summaries.items():
        sub, name = key.split(" / ")
        if sub != subset:
            continue
        c, ci = s["closing_speed"], s["closing_speed_mae_ci"]
        p = paired.get(key)
        pc = (f"{100 * p['rel_change']:+.1f}% [{100 * p['rel_ci_low']:+.1f}, {100 * p['rel_ci_high']:+.1f}] "
              f"({p['n']})") if p else "--"
        cal = s.get("calibration")
        cs = f"{cal['frac_within_1sigma']:.2f} / {cal['frac_within_1.96sigma']:.2f}" if cal and cal.get("n") else "--"
        lines.append(
            f"| {name} | {s['coverage_end_to_end']:.3f} | {c.get('mae', float('nan')):.3f} "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | {pc} | {c.get('median_ae', float('nan')):.3f} | "
            f"{c.get('bias', float('nan')):+.3f} | {s['lateral_speed'].get('mae', float('nan')):.3f} | "
            f"{s['sign_accuracy_gt_ge_1ms']:.3f} | {s['false_urgent_rate']:.3f} | {s['missed_urgent_rate']:.3f} | "
            f"{cs} | {latency[name]['median_ms']:.2f} |")
    return lines + [""]


def write_markdown(path: Path, manifest: dict, summaries: dict, paired: dict = None, latency: dict = None) -> None:
    lines = [f"# KITTI tracking velocity evaluation `{manifest['run_id']}`", "",
             f"Cache: `{manifest['cache_dir']}`  ", f"Commit: `{manifest['git']['commit']}`"
             f"{' (dirty)' if manifest['git']['dirty'] else ''}  ",
             f"Protocol SHA-256: `{manifest['protocol_sha256']}`  ",
             f"B0 replay vs online production estimates: {manifest['b0_equivalence']}", "",
             "Closing speed = -dZ/dt in the camera frame (positive = approaching). Errors in m/s; "
             "CIs are 95% percentile bootstrap over sequences.", ""]
    if latency:
        for subset in ("primary", "vru"):
            lines += comparison_table(summaries, paired or {}, latency, subset)
    for key, s in summaries.items():
        c, ci = s["closing_speed"], s["closing_speed_mae_ci"]
        lines += [f"## {key}", "",
                  f"- eligible GT observations {s['n_eligible_gt']}, matched {s['n_matched']} "
                  f"(detection rate {s['detection_rate']:.3f}), valid velocity {s['n_valid']} "
                  f"(end-to-end coverage {s['coverage_end_to_end']:.3f}, given matched {s['coverage_given_matched']:.3f})",
                  f"- closing-speed MAE {c.get('mae', float('nan')):.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], "
                  f"RMSE {c.get('rmse', float('nan')):.3f}, median {c.get('median_ae', float('nan')):.3f}, "
                  f"P90 {c.get('p90_ae', float('nan')):.3f}, bias {c.get('bias', float('nan')):+.3f} "
                  f"[{s['closing_speed_bias_ci']['ci_low']:+.3f}, {s['closing_speed_bias_ci']['ci_high']:+.3f}]",
                  f"- lateral-speed MAE {s['lateral_speed'].get('mae', float('nan')):.3f}",
                  f"- sign accuracy (|GT closing| >= 1 m/s, n={s['n_sign_rows']}) {s['sign_accuracy_gt_ge_1ms']:.3f}",
                  f"- TTC bands: false-urgent rate {s['false_urgent_rate']:.3f}, false-critical rate "
                  f"{s['false_critical_rate']:.3f}, missed-urgent rate {s['missed_urgent_rate']:.3f} "
                  f"(in-corridor n={s['n_in_corridor']}, GT urgent n={s['n_gt_urgent']}), band agreement {s['band_agreement']:.3f}",
                  f"- distance input |z_pipeline - z_near| / z_near: {s['distance_abs_rel_vs_z_near']:.3f}", "",
                  "| range | n | MAE | bias |", "|---|---|---|---|"]
        for rng, st in s["by_range"].items():
            lines.append(f"| {rng} | {st['n']} | {st.get('mae', float('nan')):.3f} | {st.get('bias', float('nan')):+.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def cmd_evaluate(args) -> None:
    if args.split != "dev" and set(args.estimators) != {"B0"} and not args.score_test:
        raise RuntimeError("experimental estimators are tuned on dev; scoring them on test (or all) needs "
                           "--score-test, once per frozen configuration (protocol section 2)")
    if "B0" not in args.estimators:
        raise RuntimeError("B0 must be part of every run (equivalence check and paired comparisons)")
    cache_dir = Path(args.cache)
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
    if cache_manifest.get("cache_format") != CACHE_FORMAT:
        raise RuntimeError(f"cache format {cache_manifest.get('cache_format')} != {CACHE_FORMAT}")
    names = [n for n in sorted(cache_manifest["sequences"]) if SPLITS[args.split](n)]
    if not names:
        raise RuntimeError(f"no cached sequences in split {args.split!r}")
    dataset = KittiTrackingDataset(cache_manifest["dataset_root"], sequences=names,
                                   frame_period_s=cache_manifest["frame_period_s"])
    vcfg = cache_manifest["config"]["velocity"]

    git = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"_{args.split}_" + (git["commit"][:7] or "nogit") \
        + ("-dirty" if git["dirty"] else "")
    out = Path(args.out or DEFAULT_OUT_ROOT / "eval" / run_id)
    out.mkdir(parents=True, exist_ok=False)

    adapters = build_estimators(vcfg, args.estimators)
    latency: Dict[str, List[float]] = {name: [] for name in adapters}
    all_rows: List[dict] = []
    equivalence = {"n_compared": 0, "n_mismatched": 0}
    for seq in dataset:
        frames = load_cached_frames(cache_dir, seq)
        estimates, times = replay_many(adapters, frames)
        for name, ts in times.items():
            latency[name] += ts
        eq = check_b0_equivalence(frames, estimates["B0"])
        equivalence = {k: equivalence[k] + eq[k] for k in equivalence}
        all_rows += evaluate_sequence(seq, frames, estimates)
        logger.info("evaluated %s: %d frames, B0 equivalence %s", seq.name, len(frames), eq)
    if equivalence["n_mismatched"]:
        raise RuntimeError(f"B0 replay does not reproduce the online production estimates: {equivalence}")

    summaries = {}
    for subset in ("primary", "vru"):
        rows = [r for r in all_rows if r["subset"] == subset]
        for name in adapters:
            summaries[f"{subset} / {name}"] = summarize_rows(rows, name, n_boot=args.bootstrap, seed=args.seed)
    # Paired comparisons against B0 on observations where both are valid.
    paired, gating = {}, {}
    for subset in ("primary", "vru"):
        for name in adapters:
            if name == "B0":
                continue
            rows = [r for r in all_rows if r["subset"] == subset and r["B0_valid"] and r[f"{name}_valid"]]
            if rows:
                paired[f"{subset} / {name}"] = paired_cluster_bootstrap(
                    [abs(-r["B0_vz"] - r["gt_closing"]) for r in rows],
                    [abs(-r[f"{name}_vz"] - r["gt_closing"]) for r in rows],
                    [r["sequence"] for r in rows], n_resamples=args.bootstrap, seed=args.seed)
            sub_rows = [r for r in all_rows if r["subset"] == subset]
            if any(np.isfinite(r[f"{name}_sigma"]) for r in sub_rows):
                gating[f"{subset} / {name}"] = gating_curve(sub_rows, name, [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 1e9])
    latency_summary = {name: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
                              "mean_ms": float(np.mean(v))} for name, v in latency.items()}

    manifest = {
        "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "git": git, "source_sha256": source_hashes(), "versions": versions(), "cache_dir": str(cache_dir),
        "cache_run_id": cache_manifest["run_id"],
        "cache_git": cache_manifest["git"], "split": args.split, "sequences": names,
        "frame_period_s": dataset.frame_period_s,
        "protocol_file": str(PROTOCOL_FILE), "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "estimators": list(adapters), "b0_equivalence": equivalence, "latency_ms": latency_summary,
        "ego_thresholds": {"moving_m_s": EGO_MOVING_M_S, "turning_rad_s": EGO_TURNING_RAD_S},
        "bootstrap": {"n": args.bootstrap, "seed": args.seed, "cluster": "sequence"},
        "licence": "KITTI, CC BY-NC-SA 3.0 -- never commit anything in this directory",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps({"summaries": summaries, "paired_vs_B0": paired,
                                                  "gating_curves": gating, "latency_ms": latency_summary}, indent=2))
    with gzip.open(out / "per_object.csv.gz", "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()) if all_rows else ["empty"])
        writer.writeheader()
        writer.writerows(all_rows)
    write_markdown(out / "summary.md", manifest, summaries, paired, latency_summary)
    print(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline KITTI tracking velocity evaluation.")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache", help="run the production pipeline once per frame and cache its outputs")
    c.add_argument("--root", default=KITTI_TRACKING_ROOT, help="extracted KITTI tracking 'training' directory")
    c.add_argument("--sequences", nargs="*", default=None, help="sequence names (default: all)")
    c.add_argument("--limit-frames", type=int, default=0, help="cache only the first N frames per sequence")
    c.add_argument("--no-depth", action="store_true", help="do not store the float16 depth maps")
    c.add_argument("--frame-period", type=float, default=1.0 / KITTI_TRACKING_FPS,
                   help="seconds between frames for timestamps (default: nominal 0.1)")
    c.add_argument("--out", default=None)
    e = sub.add_parser("evaluate", help="replay estimators on a cache and score them against ground truth")
    e.add_argument("--cache", required=True)
    e.add_argument("--split", choices=sorted(SPLITS), default="dev")
    e.add_argument("--bootstrap", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--estimators", nargs="+", default=["B0"], help="names from velocity_experimental.build_estimators")
    e.add_argument("--score-test", action="store_true", help="allow experimental estimators on test/all (once!)")
    e.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    {"cache": cmd_cache, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
