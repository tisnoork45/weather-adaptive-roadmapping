"""Offline evaluation of experimental ego-motion estimators on KITTI tracking (Task 5, Stage 3).

Replays the estimators in ego_motion.py over the cached pipeline outputs (metric depth maps and
object boxes) and the original images, and scores them against OXTS-derived camera motion
(kitti_velocity_gt.ego_kinematics: forward speed = camera velocity along its own z axis, yaw rate
about the camera's y axis, positive = turning right; 5-frame least-squares slopes). OXTS is used
here only as ground truth; no estimator sees it.

Acceptance (task5_evaluation_protocol.md, section 6): E1 forward-speed MAE <= 0.5 m/s, E2 yaw-rate
MAE <= 0.02 rad/s, E3 false "moving" rate on stationary-ego frames <= 5%. Timing is the cache's
(nominal 0.1 s unless the cache says otherwise), for estimates and ground truth alike.

Usage:
    python -B evaluate_kitti_egomotion.py --cache <cache dir> [--split dev] [--estimators ...]
Scoring the test split requires --score-test (approval first; protocol section 2).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from ego_motion import build_ego_estimators
from evaluate_kitti_velocity import (
    CACHE_FORMAT, DEFAULT_OUT_ROOT, SPLITS, git_state, load_cached_frames, sha256_file, source_hashes, versions,
)
from kitti_tracking import KittiTrackingDataset
from kitti_velocity_gt import ego_kinematics
from utils import get_logger
from velocity_eval import cluster_bootstrap_mean, replay_many

logger = get_logger(__name__)

STATIONARY_M_S = 0.1          # GT |forward speed| below this: ego stationary (E3 population)
MOVING_M_S = 1.0              # GT |forward speed| above this: clearly moving (missed-motion rate)
STRAIGHT_RAD_S = 0.05
TURNING_RAD_S = 0.1
ACCEL_M_S2 = 0.5
EGO_SOURCES = ("ego_motion.py", "evaluate_kitti_egomotion.py", "tests_ego_motion.py")


def gt_rows(seq) -> List[dict]:
    ego = ego_kinematics(seq)
    v = np.array([e.vz if e.has_window else np.nan for e in ego])
    t = seq.timestamps
    acc = np.full(len(v), np.nan)
    acc[2:-2] = (v[4:] - v[:-4]) / (t[4:] - t[:-4])
    rows = []
    for e, a in zip(ego, acc):
        if not e.has_window:
            continue
        moving = abs(e.vz) >= MOVING_M_S
        rows.append({"sequence": seq.name, "frame": e.frame, "gt_speed": e.vz, "gt_yaw_rate": e.yaw_rate,
                     "gt_accel": a, "stationary": abs(e.vz) < STATIONARY_M_S, "moving": moving,
                     "turning": abs(e.yaw_rate) >= TURNING_RAD_S,
                     "straight": moving and abs(e.yaw_rate) < STRAIGHT_RAD_S,
                     "accelerating": moving and np.isfinite(a) and abs(a) >= ACCEL_M_S2})
    return rows


def summarize(rows: List[dict], name: str, n_boot: int, seed: int) -> Dict[str, object]:
    valid = [r for r in rows if r[f"{name}_valid"]]
    seqs = [r["sequence"] for r in valid]
    es = np.array([r[f"{name}_speed"] - r["gt_speed"] for r in valid])
    ey = np.array([r[f"{name}_yaw"] - r["gt_yaw_rate"] for r in valid])
    out: Dict[str, object] = {
        "n_frames": len(rows), "n_valid": len(valid), "coverage": len(valid) / len(rows) if rows else np.nan,
        "E1_speed_mae": cluster_bootstrap_mean(np.abs(es), seqs, n_boot, seed),
        "E2_yaw_mae": cluster_bootstrap_mean(np.abs(ey), seqs, n_boot, seed),
        "speed_bias": float(es.mean()) if es.size else np.nan, "yaw_bias": float(ey.mean()) if ey.size else np.nan,
        "speed_p95_ae": float(np.percentile(np.abs(es), 95)) if es.size else np.nan,
        "yaw_p95_ae": float(np.percentile(np.abs(ey), 95)) if ey.size else np.nan,
        "speed_rel_error_moving": float(np.median(np.abs(es[[r["moving"] for r in valid]]) /
                                                   np.abs([r["gt_speed"] for r in valid if r["moving"]])))
        if any(r["moving"] for r in valid) else np.nan,
    }
    stat = [r for r in valid if r["stationary"]]
    out["E3_false_moving"] = cluster_bootstrap_mean([float(bool(r[f"{name}_moving"])) for r in stat],
                                                    [r["sequence"] for r in stat], n_boot, seed) if stat else None
    out["n_stationary_valid"] = len(stat)
    out["n_stationary_total"] = sum(r["stationary"] for r in rows)
    mov = [r for r in valid if r["moving"]]
    out["missed_motion_rate"] = float(np.mean([not r[f"{name}_moving"] for r in mov])) if mov else np.nan
    by = {}
    for cond in ("stationary", "straight", "turning", "accelerating"):
        sub = [r for r in valid if r[cond]]
        by[cond] = {"n": len(sub), "n_total": sum(r[cond] for r in rows),
                    "speed_mae": float(np.mean([abs(r[f"{name}_speed"] - r["gt_speed"]) for r in sub])) if sub else np.nan,
                    "yaw_mae": float(np.mean([abs(r[f"{name}_yaw"] - r["gt_yaw_rate"]) for r in sub])) if sub else np.nan}
    out["by_condition"] = by
    bins = {}
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 20)):
        sub = [r for r in valid if lo <= abs(r["gt_speed"]) < hi]
        bins[f"{lo}-{hi} m/s"] = {"n": len(sub), "speed_mae": float(np.mean([abs(r[f"{name}_speed"] - r["gt_speed"])
                                                                            for r in sub])) if sub else np.nan}
    out["by_speed"] = bins
    for key, gt_key, sig_key in (("speed", "gt_speed", f"{name}_sigma_speed"), ("yaw", "gt_yaw_rate", f"{name}_sigma_yaw")):
        s = np.array([r[sig_key] for r in valid], dtype=float)
        if np.isfinite(s).any():
            e = np.array([r[f"{name}_{key}"] - r[gt_key] for r in valid])
            ok = np.isfinite(s) & (s > 0)
            z = np.abs(e[ok]) / s[ok]
            out[f"calibration_{key}"] = {"n": int(ok.sum()), "within_1sigma": float(np.mean(z <= 1)),
                                         "within_1.96sigma": float(np.mean(z <= 1.96)),
                                         "k68": float(np.median(z) / 0.6745)}
    per_seq = {}
    for sname in sorted({r["sequence"] for r in rows}):
        sv = [r for r in valid if r["sequence"] == sname]
        per_seq[sname] = {"n": len(sv), "speed_mae": float(np.mean([abs(r[f"{name}_speed"] - r["gt_speed"]) for r in sv]))
                          if sv else np.nan,
                          "yaw_mae": float(np.mean([abs(r[f"{name}_yaw"] - r["gt_yaw_rate"]) for r in sv])) if sv else np.nan}
    out["per_sequence"] = per_seq
    e1, e2, e3 = out["E1_speed_mae"]["mean"], out["E2_yaw_mae"]["mean"], (out["E3_false_moving"] or {}).get("mean")
    out["pass"] = {"E1": bool(e1 <= 0.5), "E2": bool(e2 <= 0.02), "E3": None if e3 is None else bool(e3 <= 0.05)}
    return out


def evaluate(cache_dir: Path, names: List[str], split: str, estimator_params: dict = None):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("cache_format") != CACHE_FORMAT:
        raise RuntimeError("unexpected cache format")
    seqs = [n for n in sorted(manifest["sequences"]) if SPLITS[split](n)]
    ds = KittiTrackingDataset(manifest["dataset_root"], sequences=seqs, frame_period_s=manifest["frame_period_s"])
    ests = build_ego_estimators(names, estimator_params)
    rows, latency = [], {n: [] for n in names}
    for seq in ds:
        frames = load_cached_frames(cache_dir, seq)
        out, times = replay_many(ests, frames)
        for n, ts in times.items():
            latency[n] += ts
        for r in gt_rows(seq):
            k = r["frame"]
            for n in names:
                e = out[n][k]
                r[f"{n}_valid"] = bool(e.valid)
                r[f"{n}_speed"], r[f"{n}_yaw"] = e.speed, e.yaw_rate
                r[f"{n}_moving"] = e.moving
                r[f"{n}_sigma_speed"], r[f"{n}_sigma_yaw"] = e.sigma_speed, e.sigma_yaw
                r[f"{n}_n_inliers"], r[f"{n}_parallax_px"] = e.n_inliers, e.parallax_px
            rows.append(r)
        logger.info("ego-motion %s: %d frames", seq.name, len(frames))
    return rows, latency, manifest, seqs


def main() -> None:
    p = argparse.ArgumentParser(description="Offline KITTI ego-motion evaluation (experimental).")
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=sorted(SPLITS), default="dev")
    p.add_argument("--estimators", nargs="+", default=["EM0_flow_depth_baseline", "EM1_pnp_raw", "EM2_pnp_filtered"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--score-test", action="store_true", help="allow test/all (only after approval, once)")
    args = p.parse_args()
    if args.split != "dev" and not args.score_test:
        raise RuntimeError("ego-motion estimators are designed on dev; scoring test/all needs --score-test "
                           "and prior approval (protocol section 2)")
    cache_dir = Path(args.cache)
    git = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"_ego_{args.split}_" + (git["commit"][:7] or "nogit")
    out = Path(args.out or DEFAULT_OUT_ROOT / "stage3" / run_id)
    out.mkdir(parents=True, exist_ok=False)
    rows, latency, cache_manifest, seqs = evaluate(cache_dir, args.estimators, args.split)
    summaries = {n: summarize(rows, n, args.bootstrap, args.seed) for n in args.estimators}
    lat = {n: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))} for n, v in latency.items()}
    here = Path(__file__).parent
    manifest = {
        "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "git": git,
        "versions": versions(), "split": args.split, "sequences": seqs, "estimators": args.estimators,
        "source_sha256": {**source_hashes(), **{f: sha256_file(here / f) for f in EGO_SOURCES if (here / f).exists()}},
        "protocol_sha256": sha256_file(here / "task5_evaluation_protocol.md"),
        "cache_dir": str(cache_dir), "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "cache_checkpoints": cache_manifest["checkpoints"], "frame_period_s": cache_manifest["frame_period_s"],
        "ground_truth": "OXTS-derived camera motion (kitti_velocity_gt.ego_kinematics), evaluation only",
        "conditions": {"stationary_m_s": STATIONARY_M_S, "moving_m_s": MOVING_M_S, "straight_rad_s": STRAIGHT_RAD_S,
                       "turning_rad_s": TURNING_RAD_S, "accel_m_s2": ACCEL_M_S2},
        "latency_ms": lat, "licence": "KITTI, CC BY-NC-SA 3.0 -- never commit anything in this directory",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, default=float))
    with gzip.open(out / "per_frame.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Ego-motion evaluation `{run_id}` ({args.split})", "",
             "| estimator | coverage | E1 speed MAE [95% CI] | E2 yaw MAE [95% CI] | E3 false-moving [95% CI] (n) | "
             "missed motion | ms/frame | pass E1/E2/E3 |", "|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        e1, e2, e3 = s["E1_speed_mae"], s["E2_yaw_mae"], s["E3_false_moving"]
        e3s = f"{e3['mean']:.3f} [{e3['ci_low']:.3f}, {e3['ci_high']:.3f}] ({s['n_stationary_valid']})" if e3 else "--"
        lines.append(f"| {n} | {s['coverage']:.3f} | {e1['mean']:.3f} [{e1['ci_low']:.3f}, {e1['ci_high']:.3f}] | "
                     f"{e2['mean']:.4f} [{e2['ci_low']:.4f}, {e2['ci_high']:.4f}] | {e3s} | {s['missed_motion_rate']:.3f} | "
                     f"{lat[n]['median_ms']:.1f} | {s['pass']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "CHECKSUMS.sha256", "w") as fh:
        for f in sorted(out.iterdir()):
            if f.name != "CHECKSUMS.sha256":
                fh.write(f"{sha256_file(f)}  {f.name}\n")
    print(out)


if __name__ == "__main__":
    main()
