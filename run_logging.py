"""Opt-in structured run recording: a manifest plus per-frame and per-object CSV files.

    recorder = RunRecorder.start(cfg, pipeline, source="road.mp4", fps=23.997)
    ...
    output = pipeline.process(frame, timestamp=t)       # inference first,
    recorder.record(output, frame_index, t)             # then recording reads its result
    ...
    recorder.close("complete")

writes ``<paths.outputs_dir>/runs/<run_id>/``:

    manifest.json   what ran: git state, command, resolved config + SHA-256, checkpoint
                    SHA-256s, device, package versions, depth mode, strict, camera, source
    frames.csv      one row per processed frame (weather, counts, depth kind, risk, latencies)
    objects.csv     one row per detected object per frame (box, track, BEV position,
                    relative velocity, TTC, risk -- metric columns empty without metric depth)
    INCOMPLETE      present until the recording is closed as "complete"

Recording is outside the perception path: it only reads a finished
PipelineOutput, and every error it hits is logged once and then disables the
recorder (status "failed"), so live inference is never interrupted. A run
whose manifest still says "recording" was not closed (process killed or
crashed) and must be treated as incomplete. Nothing is written unless a
recorder is started.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1
INCOMPLETE_MARKER = "INCOMPLETE"
STATUS_RECORDING, STATUS_COMPLETE, STATUS_INTERRUPTED, STATUS_FAILED = "recording", "complete", "interrupted", "failed"

FRAME_COLUMNS = [
    "frame_index", "timestamp_s", "weather_label", "weather_confidence", "weather_raw_label",
    "weather_raw_confidence", "n_detections", "n_tracked", "depth_kind", "depth_mean", "bev_available",
    "n_bev_objects", "n_valid_velocities", "overall_risk_level", "min_ttc_s", "scene_confidence",
    "latency_total_ms", "latency_weather_ms", "latency_detection_ms", "latency_tracking_ms", "latency_depth_ms",
    "latency_fusion_ms", "latency_bev_ms", "latency_risk_ms", "fps",
]
OBJECT_COLUMNS = [
    "frame_index", "timestamp_s", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
    "bev_x_m", "bev_z_m", "vx_m_s", "vz_m_s", "speed_m_s", "velocity_valid", "distance_m", "ttc_s", "risk_level",
]
VERSION_PACKAGES = ["torch", "torchvision", "numpy", "opencv-python", "ultralytics", "streamlit", "PyYAML",
                    "scipy", "pandas", "open3d", "Pillow"]
FLUSH_EVERY_FRAMES = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write via a temporary file in the same directory and rename, so a crash never leaves half a manifest."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            res = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
            return res.stdout.strip() if res.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
            "changed_files": None if status is None else status.splitlines()}


def package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _checkpoint_info(path: str) -> Dict[str, Any]:
    p = Path(path)
    info: Dict[str, Any] = {"path": path, "exists": p.exists()}
    if p.exists():
        info.update(resolved=str(p.resolve()), size_bytes=p.stat().st_size, sha256=sha256_file(p))
    return info


def _model_status(model: Any) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    fallback = bool(getattr(model, "_use_fallback", False)) or getattr(model, "model", True) is None
    return {"backend": "heuristic fallback" if fallback else "neural network",
            "device": str(getattr(model, "device", "")) or None}


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _align_bev(detections: List[Any], bev_objects: List[Any]) -> List[int]:
    """Index into bev_objects for each detection (-1 if it has none).

    BEVMapper emits objects in detection order and only skips detections
    without usable depth, so an in-order walk recovers the correspondence
    (the same rule the Task 5 harness uses)."""
    out, j = [], 0
    for d in detections:
        if j < len(bev_objects) and (bev_objects[j].track_id, bev_objects[j].class_name,
                                     bev_objects[j].confidence) == (d.track_id, d.class_name, d.confidence):
            out.append(j)
            j += 1
        else:
            out.append(-1)
    return out


class RunRecorder:
    """See the module docstring. Use :meth:`start`; it returns None if the run directory cannot be created."""

    def __init__(self, run_dir: Path, manifest: Dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.frames_recorded = 0
        self.objects_recorded = 0
        self.active = True
        self._camera_recorded = False
        self._frames_fh = open(run_dir / "frames.csv", "w", newline="", encoding="utf-8")
        self._objects_fh = open(run_dir / "objects.csv", "w", newline="", encoding="utf-8")
        self._frames = csv.writer(self._frames_fh)
        self._objects = csv.writer(self._objects_fh)
        self._frames.writerow(FRAME_COLUMNS)
        self._objects.writerow(OBJECT_COLUMNS)

    # ------------------------------------------------------------------ start
    @classmethod
    def start(cls, config: Any, pipeline: Any, source: Any, fps: Optional[float] = None,
              root: Optional[str] = None, command: Optional[List[str]] = None,
              config_path: str = "config.yaml", cli_overrides: Optional[Dict[str, Any]] = None,
              run_id: Optional[str] = None) -> Optional["RunRecorder"]:
        """Create the run directory and write the initial manifest. Never raises: on any
        error it logs why and returns None, and the caller simply runs without recording."""
        try:
            return cls._start(config, pipeline, source, fps, root, command, config_path, cli_overrides, run_id)
        except Exception:
            logger.exception("Run recording could not start; continuing without recording.")
            return None

    @classmethod
    def _start(cls, config, pipeline, source, fps, root, command, config_path, cli_overrides, run_id):
        repo = Path(__file__).resolve().parent
        git = git_state(repo)
        if root is None:
            root = str(Path((config.get("paths") or {}).get("outputs_dir", "outputs")) / "runs")
        base_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_"
                             + ((git["commit"] or "nogit")[:7]) + ("-dirty" if git["dirty"] else ""))
        runs = Path(root)
        runs.mkdir(parents=True, exist_ok=True)
        run_dir = None
        for attempt in range(100):          # never reuse (overwrite) an existing run directory
            candidate = runs / (base_id if attempt == 0 else f"{base_id}-{attempt}")
            try:
                candidate.mkdir()
                run_dir = candidate
                break
            except FileExistsError:
                continue
        if run_dir is None:
            raise FileExistsError(f"no free run directory for {base_id!r} under {runs}")
        (run_dir / INCOMPLETE_MARKER).write_text(
            "This recording has not been closed as complete. See manifest.json 'status'.\n", encoding="utf-8")

        resolved = _jsonable(dict(config))
        canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
        depth_cfg, pipe_cfg = config.depth_estimation, config.pipeline
        depth_mode = getattr(pipeline, "depth_mode", depth_cfg.get("mode", "relative"))
        checkpoints = {
            "detection": _checkpoint_info(config.detection.weights_path),
            "depth": _checkpoint_info(depth_cfg.metric_weights_path if depth_mode == "metric"
                                      else depth_cfg.weights_path),
        }
        if getattr(pipeline, "weather_classifier", None) is not None:
            checkpoints["weather"] = _checkpoint_info(config.weather_classification.weights_path)
        cam = config.camera
        calibrated = bool(cam.get("calibrated", False))
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001 -- informational only
            gpu = None
        cfg_file = Path(config_path)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": run_dir.name,
            "status": STATUS_RECORDING,
            "started_utc": _utc_now(),
            "finished_utc": None,
            "frames_recorded": 0,
            "objects_recorded": 0,
            "command": list(command if command is not None else sys.argv),
            "cwd": os.getcwd(),
            "project": _jsonable(dict(config.get("project") or {})),
            "git": git,
            "config": {
                "path": str(cfg_file),
                "file_sha256": sha256_file(cfg_file) if cfg_file.exists() else None,
                "cli_overrides_in_memory": _jsonable(cli_overrides or {}),
                "resolved_sha256": hashlib.sha256(canonical).hexdigest(),
                "resolved": resolved,
            },
            "checkpoints": checkpoints,
            "models": {
                "detection": _model_status(getattr(pipeline, "detector", None)),
                "depth": _model_status(getattr(pipeline, "depth_estimator", None)),
                "weather": _model_status(getattr(pipeline, "weather_classifier", None)),
            },
            "device": {"preference": config.device.preference, "gpu": gpu},
            "versions": package_versions(),
            "platform": platform.platform(),
            "pipeline": {
                "depth_mode": depth_mode,
                "strict": bool(pipe_cfg.get("strict", False)),
                "run_weather_classification": bool(pipe_cfg.get("run_weather_classification", True)),
                "run_tracking": bool(pipe_cfg.run_tracking),
                "run_depth_estimation": bool(pipe_cfg.run_depth_estimation),
                "run_bev_mapping": bool(pipe_cfg.run_bev_mapping),
            },
            "camera": {
                "calibrated": calibrated,
                "config_intrinsics": _jsonable(dict(cam.intrinsics)),
                "config_size": [cam.width, cam.height],
                "first_frame": None,
            },
            "source": {"description": str(source), "fps": fps,
                       "timestamps": "video timeline (frame_index / fps)" if fps else "wall clock"},
            "outputs_note": (
                "Monocular-camera estimates only (no LiDAR, radar or IMU; forward-facing camera only). "
                + ("Depth is relative (unitless): distance, velocity, TTC and risk columns are empty."
                   if depth_mode != "metric" else
                   "Distances, velocities and TTC come from metric monocular depth"
                   + ("." if calibrated else " with UNCALIBRATED placeholder intrinsics: approximate only."))
            ),
            "files": {"frames": "frames.csv", "objects": "objects.csv"},
            "error": None,
        }
        _write_json_atomic(run_dir / "manifest.json", manifest)
        recorder = cls(run_dir, manifest)
        logger.info("Recording run to %s", run_dir)
        return recorder

    # ----------------------------------------------------------------- record
    def record(self, output: Any, frame_index: int, timestamp: Optional[float]) -> None:
        """Append this frame's rows. Never raises; on error the recorder stops (status 'failed')."""
        if not self.active:
            return
        try:
            self._record(output, frame_index, timestamp)
        except Exception as exc:  # noqa: BLE001 -- recording must never stop inference
            logger.exception("Run recording failed at frame %d; inference continues without recording.",
                             frame_index)
            self._finish(STATUS_FAILED, f"recording error at frame {frame_index}: {exc!r}")

    def _record(self, output: Any, frame_index: int, timestamp: Optional[float]) -> None:
        if not self._camera_recorded and output.camera is not None:
            c = output.camera
            self.manifest["camera"]["first_frame"] = {
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "width": c.width, "height": c.height,
                "calibrated": bool(c.calibrated), "source": getattr(c, "source", None)}
            self._camera_recorded = True

        dets = output.detections.detections
        bev_objects = output.bev.objects if output.bev is not None else []
        velocities = output.relative_velocities or {}
        risk_by_track = {r.track_id: r for r in output.risk.per_object} if output.risk is not None else {}
        ttcs = [r.ttc_seconds for r in risk_by_track.values() if r.ttc_seconds is not None]
        m = output.metrics or {}
        weather_raw = output.weather_raw
        self._frames.writerow([_fmt(v) for v in [
            frame_index, timestamp, output.weather.label, output.weather.confidence,
            weather_raw.label if weather_raw is not None else None,
            weather_raw.confidence if weather_raw is not None else None,
            len(dets), sum(d.track_id is not None for d in dets), output.depth.kind, output.depth.mean_depth,
            int(output.bev is not None), len(bev_objects), sum(bool(v.valid) for v in velocities.values()),
            output.risk.overall_risk_level if output.risk is not None else None, min(ttcs) if ttcs else None,
            output.fusion.overall_scene_confidence, output.total_latency_ms,
            m.get("weather_latency_ms"), m.get("detection_latency_ms"), m.get("tracking_latency_ms"),
            m.get("depth_latency_ms"), m.get("fusion_latency_ms"), m.get("bev_latency_ms"),
            m.get("risk_latency_ms"), m.get("fps"),
        ]])
        for d, j in zip(dets, _align_bev(dets, bev_objects)):
            bev = bev_objects[j] if j >= 0 else None
            vel = velocities.get(d.track_id) if d.track_id is not None else None
            risk = risk_by_track.get(d.track_id) if d.track_id is not None else None
            x1, y1, x2, y2 = (float(v) for v in d.box)
            self._objects.writerow([_fmt(v) for v in [
                frame_index, timestamp, d.track_id, int(d.class_id), d.class_name, float(d.confidence),
                x1, y1, x2, y2,
                bev.x_m if bev is not None else None, bev.z_m if bev is not None else None,
                vel.vx_m_s if vel is not None else None, vel.vz_m_s if vel is not None else None,
                vel.speed_m_s if vel is not None else None, (int(bool(vel.valid)) if vel is not None else None),
                risk.distance_m if risk is not None else None, risk.ttc_seconds if risk is not None else None,
                risk.risk_level if risk is not None else None,
            ]])
            self.objects_recorded += 1
        self.frames_recorded += 1
        if self.frames_recorded % FLUSH_EVERY_FRAMES == 0:
            self._flush()

    def _flush(self) -> None:
        self._frames_fh.flush()
        self._objects_fh.flush()

    # ------------------------------------------------------------------ close
    def close(self, status: str = STATUS_COMPLETE, error: Optional[str] = None) -> None:
        """Finalise the manifest. Never raises. Only status 'complete' removes the INCOMPLETE marker."""
        if not self.active:
            return
        self._finish(status, error)

    def _finish(self, status: str, error: Optional[str]) -> None:
        self.active = False
        for fh in (self._frames_fh, self._objects_fh):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.manifest.update(status=status, finished_utc=_utc_now(), frames_recorded=self.frames_recorded,
                                 objects_recorded=self.objects_recorded, error=error)
            _write_json_atomic(self.run_dir / "manifest.json", self.manifest)
            if status == STATUS_COMPLETE:
                (self.run_dir / INCOMPLETE_MARKER).unlink(missing_ok=True)
            logger.info("Run recording %s: %d frames, %d objects in %s", status, self.frames_recorded,
                        self.objects_recorded, self.run_dir)
        except Exception:  # noqa: BLE001
            logger.exception("Could not finalise the run manifest in %s (the run stays marked incomplete).",
                             self.run_dir)
