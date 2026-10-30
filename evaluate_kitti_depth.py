"""Evaluate depth methods against LiDAR ground truth on KITTI's selected validation set.

Runs every method on all 1,000 frames of val_selection_cropped and writes, to
outputs/experiments/kitti_depth/<timestamp>_<commit>/ (git-ignored, because
KITTI is CC BY-NC-SA 3.0):

    manifest.json           what was run: commit, checkpoint/dataset hashes, config, versions
    per_image_metrics.csv   one row per frame x method x protocol
    summary.json/.md        means with 95% CIs (bootstrap over the 13 drives)
    range_bins.csv, per_camera.csv, ground_plane_fit.csv, runtime.json, qualitative/*.png

Methods: metric ViT-S (+ focal-rescale ablation); the pipeline's old linear
0-80 m conversion of relative depth (per frame, and with its EMA across each
drive); relative depth aligned per frame to the ground truth (ORACLE upper
bound -- uses the evaluation ground truth); and a ground-plane baseline
(camera matrix + KITTI's documented 1.65 m camera height, no ground truth).

These are PIXEL-WISE DEPTH metrics on a favourable camera domain (KITTI's
focal length is close to the metric model's Virtual KITTI 2 training camera).
They do not measure object-distance accuracy and do not show generalisation
to road.mp4 or other cameras.

Usage:
    python -B evaluate_kitti_depth.py                 # full run, strict models, CUDA if available
    python -B evaluate_kitti_depth.py --limit 20      # quick smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from depth_methods import (
    VKITTI2_FOCAL_PX, LegacyLinearConversion, align_inverse_depth, focal_rescale, ground_plane_scale,
)
from depth_metrics import cluster_bootstrap_ci, devkit_metrics, range_bin_metrics, standard_metrics
from kitti_depth import KITTI_CAMERA_HEIGHT_M, KittiDepthSelection
from models import DepthEstimator
from utils import get_logger, load_config

logger = get_logger(__name__)

METHODS = {
    "metric": "Metric DA-V2 ViT-S (Virtual KITTI 2), output used as-is",
    "metric_focal": "Metric DA-V2 ViT-S x fx/725.0 (focal-length rescale ablation)",
    "legacy_per_frame": "Relative DA-V2 ViT-S + old linear 0-80 m conversion, per-frame min/max",
    "legacy_ema": "Relative DA-V2 ViT-S + old linear 0-80 m conversion, EMA bounds across each drive/camera",
    "ground_plane": "Relative DA-V2 ViT-S scaled by flat-ground geometry (K + 1.65 m camera height)",
    "oracle_aligned_gt": "ORACLE: relative DA-V2 ViT-S, per-frame LSQ scale/shift in inverse depth to the GT",
    "oracle_aligned_lidar": "ORACLE: same alignment fitted to the raw projected LiDAR, evaluated on the GT",
}
DEPLOYABLE = ["metric", "metric_focal", "legacy_per_frame", "legacy_ema", "ground_plane"]
KEY_METRICS = {
    "standard": ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog_x100", "silog_dav2", "log10", "delta1", "delta2", "delta3"],
    "devkit": ["mae", "rmse", "inv_mae", "inv_rmse", "log_mae", "log_rmse", "silog_x100", "abs_rel", "sq_rel_devkit"],
    "standard_vs_lidar": ["abs_rel", "rmse", "delta1"],
}
GP_ROI = {"roi_rows": (0.72, 0.98), "roi_cols": (0.40, 0.60), "min_r2": 0.5}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KITTI depth-selection evaluation of depth methods.")
    p.add_argument("--root", default=None, help="depth_selection root (default: config.yaml datasets.*)")
    p.add_argument("--out", default=None, help="output directory (default: outputs/experiments/kitti_depth/<run>)")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N frames (0 = all)")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples over drives")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-predictions", action="store_true", help="also write devkit-format uint16 PNGs")
    p.add_argument("--skip-dataset-hash", action="store_true", help="do not re-hash the 2 GB dataset zip")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> Dict[str, object]:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    changed = [line for line in run("git", "status", "--porcelain").splitlines() if line]
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(changed), "changed_files": changed}


def colorize(depth_m: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    norm = np.clip(255 - depth_m / max_depth * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[depth_m <= 0] = 0
    return out


def write_qualitative(path: Path, image, gt, preds: Dict[str, np.ndarray]) -> None:
    panels = [("image", image), ("ground truth (LiDAR)", colorize(gt))]
    for name in ("metric", "legacy_per_frame", "ground_plane", "oracle_aligned_gt"):
        if name in preds:
            panels.append((name, colorize(preds[name])))
    if "metric" in preds:
        err = np.zeros_like(gt)
        v = gt > 0
        err[v] = np.abs(preds["metric"][v] - gt[v]) / gt[v]
        e = cv2.applyColorMap(np.clip(err / 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        e[~v] = 0
        panels.append(("metric |err|/gt (0-50%, GT pixels only)", e))
    rows = []
    for title, img in panels:
        img = img.copy()
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(img)
    cv2.imwrite(str(path), np.vstack(rows))


def summarize(rows: List[dict], names_filter=None, n_boot=2000, seed=0) -> Dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        if names_filter is None or r["name"] in names_filter:
            groups[(r["method"], r["protocol"])].append(r)
    out: Dict[str, dict] = {}
    for (method, protocol), rs in sorted(groups.items()):
        drives = [r["drive"] for r in rs]
        entry = {"n_frames": len(rs), "n_drives": len(set(drives)), "n_pixels": int(sum(r["n_pixels"] for r in rs))}
        for k in KEY_METRICS[protocol]:
            entry[k] = cluster_bootstrap_ci([r[k] for r in rs], drives, n_resamples=n_boot, seed=seed)
        out.setdefault(method, {})[protocol] = entry
    return out


def fmt_ci(e: dict, digits: int = 3) -> str:
    return f"{e['mean']:.{digits}f} [{e['ci_low']:.{digits}f}, {e['ci_high']:.{digits}f}]"


def write_markdown(path: Path, manifest: dict, summary: dict, common: dict, gp_stats: dict,
                   bins: List[dict], per_cam: List[dict], runtime: dict) -> None:
    L = []
    L.append("# KITTI depth evaluation (val_selection_cropped)\n")
    L.append(f"Run: `{manifest['run_id']}` · commit `{manifest['git']['commit'][:7]}`"
             f"{' (uncommitted changes)' if manifest['git']['dirty'] else ''} · {manifest['n_frames']} frames, "
             f"{manifest['n_drives']} drives · device {manifest['device']}\n")
    L.append("> **Scope.** These are *pixel-wise depth* metrics against LiDAR ground truth. They do **not** "
             "measure object-distance accuracy (this subset has no 3D boxes), and good depth metrics do not by "
             "themselves establish accurate object distances.\n>\n"
             "> **Favourable camera domain.** KITTI's focal length (707–722 px) is close to the metric model's "
             "Virtual KITTI 2 training camera (725 px), and KITTI is clear daytime only. These results do **not** "
             "show generalisation to road.mp4 or any other camera.\n>\n"
             "> **Oracles.** `oracle_*` rows use ground truth to fit scale/shift and are upper bounds, not "
             "deployable methods.\n")
    L.append(f"Means over images with 95% confidence intervals from a bootstrap over the {manifest['n_drives']} "
             f"drives ({manifest['args']['bootstrap']} resamples, seed {manifest['args']['seed']}); "
             "with few drives the intervals are wide.\n")

    def table(protocol, metrics, title, src=summary, digits=3):
        L.append(f"\n## {title}\n")
        L.append("| Method | frames | " + " | ".join(metrics) + " |")
        L.append("|---|---|" + "---|" * len(metrics))
        for m in METHODS:
            if m in src and protocol in src[m]:
                e = src[m][protocol]
                L.append(f"| `{m}` | {e['n_frames']} | " + " | ".join(fmt_ci(e[k], digits) for k in metrics) + " |")

    table("standard", ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog_x100", "delta1", "delta2", "delta3"],
          "Standard protocol (GT in (0.001, 80] m, predictions clamped)")
    table("devkit", ["mae", "rmse", "inv_rmse", "log_rmse", "silog_x100", "abs_rel", "sq_rel_devkit"],
          "Official devkit protocol (all GT > 0, no cap, devkit SqRel = mean(err²/gt²); inverse errors in 1/m)")
    table("standard_vs_lidar", ["abs_rel", "rmse", "delta1"],
          "Secondary: standard protocol against raw projected LiDAR instead of annotated GT")

    L.append("\n## Ground-plane baseline coverage\n")
    L.append(f"Fit succeeded on {gp_stats['ok']} of {gp_stats['total']} frames; failures: "
             + (", ".join(f"{k} ({v})" for k, v in gp_stats["reasons"].items()) or "none")
             + ". Failed frames are excluded from its rows (never filled in). "
             f"Camera height {KITTI_CAMERA_HEIGHT_M} m (KITTI documentation), region rows {GP_ROI['roi_rows']}, "
             f"columns {GP_ROI['roi_cols']}, minimum R² {GP_ROI['min_r2']}.\n")
    if common:
        L.append("\n### All methods on the frames where the ground-plane fit succeeded\n")
        table("standard", ["abs_rel", "rmse", "rmse_log", "delta1"], "Standard protocol, common subset", src=common)

    L.append("\n## Error by ground-truth depth range (standard protocol, mean over frames with pixels in the bin)\n")
    L.append("| Method | bin | frames | abs_rel | rmse | delta1 |")
    L.append("|---|---|---|---|---|---|")
    for b in bins:
        L.append(f"| `{b['method']}` | {b['bin']} | {b['n_frames']} | {b['abs_rel']:.3f} | {b['rmse']:.3f} | {b['delta1']:.3f} |")

    L.append("\n## By camera (standard protocol)\n")
    L.append("| Method | camera | frames | abs_rel | rmse | delta1 |")
    L.append("|---|---|---|---|---|---|")
    for c in per_cam:
        L.append(f"| `{c['method']}` | {c['camera']} | {c['n_frames']} | {c['abs_rel']:.3f} | {c['rmse']:.3f} | {c['delta1']:.3f} |")

    L.append("\n## Runtime\n")
    for k, v in runtime.items():
        L.append(f"- {k}: {v}")
    L.append("\n## Method definitions\n")
    for m, d in METHODS.items():
        L.append(f"- `{m}`: {d}")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config("config.yaml")
    root = Path(args.root or cfg.datasets.kitti_depth_selection_root)
    dataset = KittiDepthSelection(root)
    samples = dataset.samples[: args.limit] if args.limit else dataset.samples

    git = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + (git["commit"][:7] or "nogit") \
        + ("-dirty" if git["dirty"] else "") + (f"-limit{args.limit}" if args.limit else "")
    out = Path(args.out or Path("outputs/experiments/kitti_depth") / run_id)
    (out / "qualitative").mkdir(parents=True, exist_ok=False)

    depth_cfg = cfg.depth_estimation
    rel_path, met_path = Path(depth_cfg.weights_path), Path(depth_cfg.metric_weights_path)
    met_sha = sha256_file(met_path)
    if met_sha != depth_cfg.metric_sha256:
        raise RuntimeError(f"Metric checkpoint hash {met_sha} != recorded {depth_cfg.metric_sha256}")
    zip_path = root.parent.parent / "data_depth_selection.zip"
    dataset_sha = None if args.skip_dataset_hash or not zip_path.exists() else sha256_file(zip_path)

    # strict=True: a missing or broken model must stop the evaluation, never fall back.
    common_kw = dict(encoder=depth_cfg.encoder, input_size=depth_cfg.input_size, max_depth_m=depth_cfg.max_depth_m,
                     device_preference=args.device, strict=True)
    rel_model = DepthEstimator(weights_path=str(rel_path), mode="relative", **common_kw)
    met_model = DepthEstimator(weights_path=str(met_path), mode="metric", **common_kw)
    device = str(rel_model.device)

    manifest = {
        "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "args": vars(args), "git": git, "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "opencv": cv2.__version__},
        "checkpoints": {"relative": {"path": str(rel_path), "sha256": sha256_file(rel_path)},
                        "metric": {"path": str(met_path), "resolved": str(met_path.resolve()), "sha256": met_sha}},
        "dataset": {"root": str(root), "split": "val_selection_cropped", "zip": str(zip_path), "zip_sha256": dataset_sha,
                    "licence": "KITTI, CC BY-NC-SA 3.0 -- outputs derived from it must not be committed"},
        "n_frames": len(samples), "n_drives": len({s.drive for s in samples}),
        "methods": METHODS, "ground_plane": {"camera_height_m": KITTI_CAMERA_HEIGHT_M,
                                              "height_source": "KITTI documentation (not fitted to ground truth)", **GP_ROI},
        "metric_focal_train_fx": VKITTI2_FOCAL_PX, "input_size": depth_cfg.input_size,
        "config_depth_estimation": dict(depth_cfg),
        "caveats": ["pixel-wise depth only; object-distance accuracy is not evaluated",
                    "favourable camera domain; no generalisation claim beyond KITTI",
                    "oracle_* methods use the evaluation ground truth"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("KITTI depth evaluation %s: %d frames -> %s", run_id, len(samples), out)

    rows: List[dict] = []
    gp_rows: List[dict] = []
    bin_acc = defaultdict(list)
    legacy_seq: Dict[tuple, LegacyLinearConversion] = {}
    times = {"relative": [], "metric": []}
    qual_idx = set(np.linspace(0, len(samples) - 1, 6).round().astype(int).tolist())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def timed(model, image, key):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model.predict_raw(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times[key].append((time.perf_counter() - t0) * 1000.0)
        return result

    for i, s in enumerate(samples):
        image, gt, lidar, K = s.load_image(), s.load_gt(), s.load_velodyne(), s.load_intrinsics()
        rel = timed(rel_model, image, "relative")
        met = timed(met_model, image, "metric")
        if rel.shape != gt.shape or met.shape != gt.shape:
            raise RuntimeError(f"{s.name}: prediction shape {rel.shape}/{met.shape} != ground truth {gt.shape}")

        preds = {"metric": met, "metric_focal": focal_rescale(met, fx=K[0, 0]),
                 "legacy_per_frame": LegacyLinearConversion(alpha=1.0)(rel)}
        key = (s.drive, s.camera)                       # samples are sorted by drive, frame, camera
        preds["legacy_ema"] = legacy_seq.setdefault(key, LegacyLinearConversion(alpha=0.3))(rel)
        preds["oracle_aligned_gt"], _ = align_inverse_depth(rel, gt)
        preds["oracle_aligned_lidar"], _ = align_inverse_depth(rel, lidar)
        gp_pred, fit = ground_plane_scale(rel, K, KITTI_CAMERA_HEIGHT_M, **GP_ROI)
        gp_rows.append({"name": s.name, "drive": s.drive, "camera": s.camera, **fit.__dict__})
        if gp_pred is not None:
            preds["ground_plane"] = gp_pred

        base = {"name": s.name, "drive": s.drive, "camera": s.camera}
        for method, pred in preds.items():
            rows.append({**base, "method": method, "protocol": "standard", **standard_metrics(pred, gt)})
            rows.append({**base, "method": method, "protocol": "devkit", **devkit_metrics(pred, gt)})
            rows.append({**base, "method": method, "protocol": "standard_vs_lidar", **standard_metrics(pred, lidar)})
            for bin_name, m in range_bin_metrics(pred, gt).items():
                bin_acc[(method, bin_name)].append(m)
            if args.save_predictions:
                d = out / "predictions_devkit" / method
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{s.name}.png"), np.clip(np.round(pred * 256.0), 1, 65535).astype(np.uint16))
        if i in qual_idx:
            write_qualitative(out / "qualitative" / f"{i:04d}_{s.name}.png", image, gt, preds)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d frames", i + 1, len(samples))

    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "per_image_metrics.csv", index=False)
    pd.DataFrame(gp_rows).to_csv(out / "ground_plane_fit.csv", index=False)

    summary = summarize(rows, n_boot=args.bootstrap, seed=args.seed)
    gp_ok = {r["name"] for r in gp_rows if r["ok"]}
    common = summarize([r for r in rows if r["protocol"] == "standard"], names_filter=gp_ok,
                       n_boot=args.bootstrap, seed=args.seed) if gp_ok else {}
    reasons = defaultdict(int)
    for r in gp_rows:
        if not r["ok"]:
            reasons[r["reason"].split(" (R^2")[0]] += 1
    gp_stats = {"ok": len(gp_ok), "total": len(gp_rows), "reasons": dict(reasons)}

    bins = []
    for (method, bin_name), ms in sorted(bin_acc.items()):
        bins.append({"method": method, "bin": bin_name, "n_frames": len(ms), "n_pixels": int(sum(m["n_pixels"] for m in ms)),
                     **{k: float(np.mean([m[k] for m in ms])) for k in ("abs_rel", "rmse", "delta1")}})
    pd.DataFrame(bins).to_csv(out / "range_bins.csv", index=False)

    per_cam = []
    for method in METHODS:
        for cam in ("02", "03"):
            rs = [r for r in rows if r["method"] == method and r["protocol"] == "standard" and r["camera"] == cam]
            if rs:
                per_cam.append({"method": method, "camera": cam, "n_frames": len(rs),
                                **{k: float(np.mean([r[k] for r in rs])) for k in ("abs_rel", "rmse", "delta1")}})
    pd.DataFrame(per_cam).to_csv(out / "per_camera.csv", index=False)

    runtime = {
        "relative_ms_per_frame_median": round(float(np.median(times["relative"])), 1),
        "metric_ms_per_frame_median": round(float(np.median(times["metric"])), 1),
        "relative_ms_per_frame_mean": round(float(np.mean(times["relative"])), 1),
        "metric_ms_per_frame_mean": round(float(np.mean(times["metric"])), 1),
        "note": "model forward incl. pre/post-processing, CUDA-synchronised; input 1216x352 -> 518x1792 tokens grid",
        "peak_gpu_memory_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    result = {"summary": summary, "common_subset_ground_plane": common, "ground_plane_coverage": gp_stats,
              "deployable_methods": DEPLOYABLE, "oracle_methods": [m for m in METHODS if m.startswith("oracle")]}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_markdown(out / "summary.md", manifest, summary, common, gp_stats, bins, per_cam, runtime)
    logger.info("Done: %s", out / "summary.md")
    print(out)


if __name__ == "__main__":
    main()
