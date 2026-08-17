"""Shared utilities: config loading, logging, metrics, visualization, video IO, device selection."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

# =============================================================================
# Configuration loading
# =============================================================================


class ConfigDict(dict):
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigDict(value)
        if isinstance(value, list):
            return [ConfigDict._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Configuration key '{item}' not found.") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


def load_config(path: str = "config.yaml") -> ConfigDict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ConfigDict(raw)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Logging
# =============================================================================

_CONFIGURED = False


def configure_logging(level="INFO", log_to_file=True, log_file="logs/pipeline.log",
                       max_bytes=5 * 1024 * 1024, backup_count=3) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    if log_to_file:
        log_dir = os.path.dirname(log_file) or "."
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name if name else "weather_adaptive_mapping")


logger = get_logger(__name__)


# =============================================================================
# Device selection
# =============================================================================


def get_device(preference: str = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in ("mps", "auto") and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Performance metrics
# =============================================================================


@dataclass
class LatencyStats:
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def update(self, latency_ms: float) -> None:
        self.history.append(latency_ms)

    @property
    def mean(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0

    @property
    def last(self) -> float:
        return self.history[-1] if self.history else 0.0


class FPSTracker:
    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._timestamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


class MetricsTracker:
    def __init__(self, stages: List[str], window: int = 60) -> None:
        self.fps_tracker = FPSTracker(window=window)
        self.stage_latency: Dict[str, LatencyStats] = {
            stage: LatencyStats(history=deque(maxlen=window)) for stage in stages
        }
        self.frame_count: int = 0
        self.object_count_history: Deque[int] = deque(maxlen=window)

    def record_stage(self, stage: str, latency_ms: float) -> None:
        if stage not in self.stage_latency:
            self.stage_latency[stage] = LatencyStats(history=deque(maxlen=60))
        self.stage_latency[stage].update(latency_ms)

    def record_frame(self, object_count: int) -> float:
        self.frame_count += 1
        self.object_count_history.append(object_count)
        return self.fps_tracker.tick()

    def total_latency_ms(self) -> float:
        return sum(stats.last for stats in self.stage_latency.values())

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {
            "fps": round(self.fps_tracker.fps, 2),
            "frame_count": self.frame_count,
            "total_latency_ms": round(self.total_latency_ms(), 2),
        }
        for stage, stats in self.stage_latency.items():
            summary[f"{stage}_latency_ms"] = round(stats.last, 2)
            summary[f"{stage}_latency_avg_ms"] = round(stats.mean, 2)
        return summary


# =============================================================================
# Video / camera IO
# =============================================================================


class VideoStream:
    def __init__(self, source: Union[int, str], width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "VideoStream":
        self.cap = cv2.VideoCapture(self.source)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: {self.source}")
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            raise RuntimeError("VideoStream not opened. Call .open() first.")
        return self.cap.read()

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def generate_synthetic_frame(width: int = 1280, height: int = 720, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height // 2):
        shade = int(135 + (y / (height / 2)) * 60)
        frame[y, :] = (shade, shade - 20, shade - 40)
    frame[height // 2:, :] = (60, 60, 60)
    horizon_y = height // 2
    for offset in (-1, 1):
        pts = np.array([[width // 2 + offset * 40, horizon_y], [width // 2 + offset * 260, height]], dtype=np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 4)
    for _ in range(rng.integers(1, 4)):
        x = int(rng.integers(50, width - 150))
        y = int(rng.integers(horizon_y + 20, height - 100))
        w_box, h_box = int(rng.integers(60, 140)), int(rng.integers(40, 80))
        color = tuple(int(c) for c in rng.integers(30, 220, size=3))
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, -1)
    return frame


# =============================================================================
# Visualization
# =============================================================================

_PALETTE: List[Tuple[int, int, int]] = [
    (66, 135, 245), (52, 235, 116), (235, 64, 52), (235, 192, 52),
    (188, 52, 235), (52, 235, 235), (235, 52, 149), (140, 235, 52),
    (52, 88, 235), (235, 119, 52),
]

# BGR colors per risk level, used by draw_hud and draw_risk_panel.
RISK_LEVEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "SAFE": (60, 200, 60),
    "CAUTION": (0, 210, 255),
    "WARNING": (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}


def color_for_id(identifier: int) -> Tuple[int, int, int]:
    return _PALETTE[identifier % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
    track_ids: Sequence[int] = None,
    distances: Optional[Sequence[Optional[float]]] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """UNCHANGED from the previous fusion-wiring patch (alpha/distances)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    target = frame if alpha >= 0.999 else frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        tid = track_ids[i] if track_ids is not None else i
        color = color_for_id(tid)
        cv2.rectangle(target, (x1, y1), (x2, y2), color, 2)

        label = labels[i] if i < len(labels) else "obj"
        score = scores[i] if i < len(scores) else 0.0
        tag = f"{label} {score:.2f}"
        if track_ids is not None:
            tag = f"ID:{tid} {tag}"
        if distances is not None and i < len(distances) and distances[i] is not None:
            tag = f"{tag} {distances[i]:.1f}m"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(target, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(target, tag, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if alpha < 0.999:
        cv2.addWeighted(target, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_lanes(
    frame: np.ndarray,
    lanes: Sequence[Sequence[Tuple[int, int]]],
    alpha: float = 1.0,
) -> np.ndarray:
    """UNCHANGED from the previous fusion-wiring patch."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    target = frame if alpha >= 0.999 else frame.copy()

    for idx, lane in enumerate(lanes):
        if len(lane) < 2:
            continue
        color = color_for_id(idx + 5)
        pts = np.array(lane, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(target, [pts], isClosed=False, color=color, thickness=4)
        for pt in lane:
            cv2.circle(target, (int(pt[0]), int(pt[1])), 3, color, -1)

    if alpha < 0.999:
        cv2.addWeighted(target, alpha, frame, 1 - alpha, 0, frame)
    return frame


def overlay_depth_map(frame: np.ndarray, depth_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, depth_color, alpha, 0)


def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


def draw_hud(frame: np.ndarray, weather: str, fps: float, object_count: int,
             latency_ms: float, weights: Dict[str, float], risk_level: Optional[str] = None) -> np.ndarray:
    """Draw the corner HUD panel.

    NEW: optional `risk_level` parameter (default None -- omitting it
    produces byte-identical output to before this parameter existed, same
    backward-compatibility pattern as the earlier `alpha`/`distances`
    additions to draw_detections/draw_lanes).
    """
    extra_line = 1 if risk_level is not None else 0
    panel_h = 34 + 22 * (2 + len(weights) + extra_line)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    y = 22
    cv2.putText(frame, f"Weather: {weather}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {latency_ms:.1f}ms", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Objects: {object_count}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    y += 22
    if risk_level is not None:
        risk_color = RISK_LEVEL_COLORS.get(risk_level, (200, 200, 200))
        cv2.putText(frame, f"RISK: {risk_level}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2, cv2.LINE_AA)
        y += 22
    for name, weight in weights.items():
        cv2.putText(frame, f"{name}: {weight:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1, cv2.LINE_AA)
        y += 20
    return frame


def draw_risk_panel(frame: np.ndarray, risk, max_objects: int = 5) -> np.ndarray:
    """NEW. Draw a dedicated risk panel (top-right corner): overall risk
    level, color-coded, plus the highest-risk tracked objects with their
    distance and TTC. `risk` is a `risk_engine.RiskAssessment` or None (a
    None risk is a no-op, returns `frame` unchanged -- safe to call even on
    frames where risk data isn't available, e.g. BEV mapping was disabled).
    """
    if risk is None:
        return frame

    h, w = frame.shape[:2]
    panel_w = 260
    order = {"CRITICAL": 0, "WARNING": 1, "CAUTION": 2, "SAFE": 3}
    sorted_objects = sorted(risk.per_object, key=lambda o: order.get(o.risk_level, 4))[:max_objects]
    panel_h = 40 + 20 * (len(sorted_objects) + 1)

    x0 = w - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    overall_color = RISK_LEVEL_COLORS.get(risk.overall_risk_level, (200, 200, 200))
    cv2.putText(frame, f"RISK: {risk.overall_risk_level}", (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, overall_color, 2, cv2.LINE_AA)

    y = y0 + 46
    for obj in sorted_objects:
        color = RISK_LEVEL_COLORS.get(obj.risk_level, (200, 200, 200))
        ttc_txt = f"{obj.ttc_seconds:.1f}s" if obj.ttc_seconds is not None else "--"
        line = f"ID{obj.track_id} {obj.class_name}: {obj.distance_m:.1f}m TTC={ttc_txt}"
        cv2.putText(frame, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 20

    return frame
