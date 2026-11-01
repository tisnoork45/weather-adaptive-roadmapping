"""Golden-output regression checks for the perception pipeline.

Run from the repository root (tests_regression.py runs these in a subprocess):

    python -B scripts/regression/golden.py record  --mode relative   # write a new golden (road.mp4)
    python -B scripts/regression/golden.py compare --mode metric     # exit 0 = identical, 1 = differs, 2 = skipped
    python -B scripts/regression/golden.py adopt   --mode relative --source FILE --provenance "..."
    python -B scripts/regression/golden.py kitti   --out DIR          # KITTI 0000 frames 0-49 vs the Task 5 cache
    python -B scripts/regression/golden.py reset   --mode metric      # reset() and shared-model determinism

A golden is the full per-frame output of PerceptionPipeline on road.mp4 frames 0..N-1
(detections, track IDs, weather, a hash of the depth map, BEV, velocities, TTC, risk),
fed exactly as run.py feeds a video (video-timeline timestamps). The comparison is
exact: on one machine the pipeline is bit-reproducible.

Goldens and their manifests live in $WARM_REGRESSION_DIR (default outputs/regression/,
git-ignored). Each manifest records an environment fingerprint -- SHA-256 of the video,
config.yaml and checkpoints, plus the torch version and GPU. GPU results are only
bit-identical on the same software and hardware, so `compare` reports "skipped" (exit 2)
instead of a false regression when the fingerprint does not match.

The KITTI check runs the frozen Task 5 harness (evaluate_kitti_velocity.cache_sequence)
into a scratch directory and compares it with the frozen cache, which is only read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "outputs" / "regression"
VIDEO = ROOT / "road.mp4"
DEFAULT_FRAMES = 240
KITTI_ROOT = "/workspace/datasets/kitti/tracking/extracted/training"
KITTI_CACHE = "/workspace/experiments/kitti_velocity/cache/20261031-190612_1892061-dirty"
EXIT_OK, EXIT_DIFFERS, EXIT_SKIPPED = 0, 1, 2


def regression_dir() -> Path:
    return Path(os.environ.get("WARM_REGRESSION_DIR", str(DEFAULT_DIR)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _array_hash(a) -> str:
    import numpy as np

    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest() + f"|{a.dtype}|{a.shape}"


def dump_output(out) -> Dict[str, Any]:
    """Everything the pipeline computes for one frame, as plain JSON values."""
    d = {
        "weather": [out.weather.label, out.weather.confidence, out.weather.probabilities],
        "weather_raw": [out.weather_raw.label, out.weather_raw.confidence, out.weather_raw.probabilities],
        "dets": [[[float(v) for v in x.box], int(x.class_id), x.class_name, float(x.confidence), x.track_id]
                 for x in out.detections.detections],
        "depth": [out.depth.kind, _array_hash(out.depth.depth_map), float(out.depth.mean_depth),
                  None if out.depth.valid_mask is None else _array_hash(out.depth.valid_mask)],
        "fusion": [out.fusion.sensor_weights, out.fusion.module_confidences, out.fusion.overall_scene_confidence,
                   out.fusion.weather_label],
        "bev": None if out.bev is None else [_array_hash(out.bev.bev_image),
                                             [[o.track_id, o.class_name, o.x_m, o.z_m, o.confidence]
                                              for o in out.bev.objects]],
        "vel": {str(k): [v.vx_m_s, v.vz_m_s, v.speed_m_s, bool(v.valid)] for k, v in out.relative_velocities.items()},
        "risk": None if out.risk is None else [out.risk.overall_risk_level, out.risk.weather_label,
                                               out.risk.active_thresholds_s,
                                               [[r.track_id, r.class_name, r.distance_m, r.ttc_seconds, r.risk_level]
                                                for r in out.risk.per_object]],
        "camera": None if out.camera is None else [out.camera.fx, out.camera.fy, out.camera.cx, out.camera.cy,
                                                   out.camera.width, out.camera.height, out.camera.calibrated],
    }
    return json.loads(json.dumps(d))


def diff(a: Any, b: Any, path: str = "", out: Optional[List] = None) -> List:
    """Exact structural differences between two JSON values: (path, a, b) tuples."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            out.append((path, "keys", sorted(a.keys() ^ b.keys())))
        for k in a.keys() & b.keys():
            diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len", (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, a, b))
    return out


def video_config(mode: str):
    """config.yaml with run.py's command-line overrides for this mode (in memory only)."""
    from utils import load_config

    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.depth_estimation["mode"] = mode
    if mode == "metric":
        cfg.pipeline["strict"] = True
    return cfg


def fingerprint(mode: str) -> Dict[str, Any]:
    import torch

    cfg = video_config(mode)
    depth = cfg.depth_estimation.metric_weights_path if mode == "metric" else cfg.depth_estimation.weights_path
    checkpoints = [cfg.detection.weights_path, depth, cfg.weather_classification.weights_path]
    return {
        "video_sha256": sha256_file(VIDEO) if VIDEO.exists() else None,
        "config_sha256": sha256_file(ROOT / "config.yaml"),
        "checkpoints_sha256": {c: sha256_file(ROOT / c) if (ROOT / c).exists() else None for c in checkpoints},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_video(mode: str, n_frames: int) -> List[Dict[str, Any]]:
    import cv2

    from pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(video_config(mode))
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        for k in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(dump_output(pipeline.process(frame, timestamp=k / fps)))
    finally:
        cap.release()
    return frames


def _paths(mode: str):
    d = regression_dir()
    return d / f"road_{mode}.json.gz", d / f"road_{mode}.manifest.json"


def _write_golden(mode: str, frames: List[Dict], provenance: str) -> None:
    data_path, manifest_path = _paths(mode)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_path, "wt") as fh:
        json.dump({"mode": mode, "frames": frames}, fh)
    manifest = {"mode": mode, "n_frames": len(frames), "video": VIDEO.name, "created_utc":
                datetime.now(timezone.utc).isoformat(timespec="seconds"), "provenance": provenance,
                "data_sha256": sha256_file(data_path), "fingerprint": fingerprint(mode)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {data_path} ({len(frames)} frames)")


def cmd_record(args) -> int:
    if not VIDEO.exists():
        print(f"SKIP: {VIDEO} not found")
        return EXIT_SKIPPED
    _write_golden(args.mode, run_video(args.mode, args.frames), args.provenance or "recorded by golden.py record")
    return EXIT_OK


def cmd_adopt(args) -> int:
    """Store an existing per-frame dump (same format) as the golden, with this environment's fingerprint."""
    with gzip.open(args.source, "rt") as fh:
        data = json.load(fh)
    if data.get("mode") != args.mode:
        print(f"ERROR: {args.source} is a {data.get('mode')!r} dump, not {args.mode!r}")
        return EXIT_DIFFERS
    _write_golden(args.mode, data["frames"], args.provenance)
    return EXIT_OK


def cmd_compare(args) -> int:
    import torch

    data_path, manifest_path = _paths(args.mode)
    if not (data_path.exists() and manifest_path.exists()):
        print(f"SKIP: no golden for mode {args.mode!r} in {regression_dir()} (see scripts/regression/README.md)")
        return EXIT_SKIPPED
    if not VIDEO.exists():
        print(f"SKIP: {VIDEO} not found")
        return EXIT_SKIPPED
    if not torch.cuda.is_available():
        print("SKIP: the goldens were recorded on CUDA; no GPU available")
        return EXIT_SKIPPED
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(data_path) != manifest["data_sha256"]:
        print(f"ERROR: {data_path} does not match its manifest's SHA-256 (corrupted or edited golden)")
        return EXIT_DIFFERS
    now = fingerprint(args.mode)
    changed = [k for k in now if now[k] != manifest["fingerprint"].get(k)]
    if changed:
        print(f"SKIP: environment differs from the golden's ({', '.join(changed)}); "
              "an exact comparison would not be meaningful")
        return EXIT_SKIPPED
    with gzip.open(data_path, "rt") as fh:
        golden = json.load(fh)["frames"]
    current = run_video(args.mode, manifest["n_frames"])
    differences = diff(golden, current)
    print(f"road.mp4 {args.mode}: {len(current)}/{len(golden)} frames, {len(differences)} differences")
    for d in differences[:20]:
        print("   ", d)
    return EXIT_OK if not differences else EXIT_DIFFERS


def cmd_kitti(args) -> int:
    import numpy as np
    import torch

    cache = Path(os.environ.get("KITTI_VELOCITY_CACHE", KITTI_CACHE))
    if not (cache / "0000" / "frames.json.gz").exists() or not Path(KITTI_ROOT).exists():
        print(f"SKIP: KITTI tracking data or the Task 5 cache ({cache}) is not available")
        return EXIT_SKIPPED
    if not torch.cuda.is_available():
        print("SKIP: the Task 5 cache was produced on CUDA; no GPU available")
        return EXIT_SKIPPED
    import evaluate_kitti_velocity as ev
    from kitti_tracking import KittiTrackingDataset

    out = Path(args.out)
    cfg, _ = ev.evaluation_config()
    seq = next(iter(KittiTrackingDataset(KITTI_ROOT, sequences=["0000"], frame_period_s=0.1)))
    ev.cache_sequence(seq, cfg, out / "0000", args.frames, save_depth=True)   # writes only under --out
    with gzip.open(out / "0000" / "frames.json.gz", "rt") as fh:
        new = json.load(fh)["frames"]
    with gzip.open(cache / "0000" / "frames.json.gz", "rt") as fh:
        old = json.load(fh)["frames"][:len(new)]
    for f in new + old:
        f.pop("latency_ms")
    differences = diff(old, new)
    depth_new = np.load(out / "0000" / "depth_f16.npy")
    depth_old = np.load(cache / "0000" / "depth_f16.npy", mmap_mode="r")[:len(new)]
    depth_same = bool(np.array_equal(depth_new, depth_old))
    print(f"KITTI 0000 frames 0-{len(new) - 1} vs {cache.name}: {len(differences)} differences, "
          f"depth identical: {depth_same}")
    for d in differences[:20]:
        print("   ", d)
    return EXIT_OK if not differences and depth_same else EXIT_DIFFERS


def cmd_reset(args) -> int:
    """Real models: a pipeline gives identical outputs before and after reset(), and two
    pipelines sharing one thread-safe model set, run in parallel threads, each reproduce it."""
    import threading

    import cv2
    import torch

    from pipeline import PerceptionPipeline

    if not VIDEO.exists():
        print(f"SKIP: {VIDEO} not found")
        return EXIT_SKIPPED
    if not torch.cuda.is_available():
        print("SKIP: needs a CUDA GPU (the check is bit-exact)")
        return EXIT_SKIPPED
    cfg = video_config(args.mode)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = [cap.read()[1] for _ in range(args.frames)]
    cap.release()

    def run(p):
        return [dump_output(p.process(f, timestamp=k / fps)) for k, f in enumerate(frames)]

    first = PerceptionPipeline(cfg)
    solo = run(first)
    first.reset()
    again = run(first)
    shared = first.models.thread_safe()
    a, b = PerceptionPipeline(cfg, models=shared), PerceptionPipeline(cfg, models=shared)
    results = {}
    threads = [threading.Thread(target=lambda n, p: results.__setitem__(n, run(p)), args=(n, p))
               for n, p in (("a", a), ("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    checks = {"after reset()": diff(solo, again), "thread A (shared models)": diff(solo, results.get("a")),
              "thread B (shared models)": diff(solo, results.get("b"))}
    for name, d in checks.items():
        print(f"road.mp4 {args.mode}, {len(frames)} frames, {name}: {len(d)} differences")
    return EXIT_OK if not any(checks.values()) else EXIT_DIFFERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("record", "compare", "adopt"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["relative", "metric"], required=True)
        if name == "record":
            p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
            p.add_argument("--provenance", default="")
        if name == "adopt":
            p.add_argument("--source", required=True)
            p.add_argument("--provenance", required=True)
    k = sub.add_parser("kitti")
    k.add_argument("--out", required=True, help="scratch directory for the new cache (must not exist)")
    k.add_argument("--frames", type=int, default=50)
    r = sub.add_parser("reset")
    r.add_argument("--mode", choices=["relative", "metric"], default="metric")
    r.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    return {"record": cmd_record, "compare": cmd_compare, "adopt": cmd_adopt, "kitti": cmd_kitti,
            "reset": cmd_reset}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
