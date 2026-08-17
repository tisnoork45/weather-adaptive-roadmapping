"""Dynamic per-module confidence estimation and weather-adaptive sensor fusion.

Two closely related concerns merged into one file:
  1. ConfidenceEstimator: turns each module's raw output into a single 0-1
     confidence score using model score + weather penalty + temporal stability.
  2. AdaptiveFusionEngine: combines static reliability priors, the live
     weather condition, and per-module confidence into smoothed sensor
     weights and one overall scene confidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Dynamic confidence estimation
# =============================================================================


@dataclass
class ModuleConfidence:
    module_name: str
    model_score: float
    weather_penalty: float
    stability_score: float
    final_confidence: float


class StabilityTracker:
    """Tracks a scalar summary statistic per module across frames to estimate jitter."""

    def __init__(self, history_length: int = 30) -> None:
        self._history: Dict[str, Deque[float]] = {}
        self.history_length = history_length

    def update(self, module_name: str, value: float) -> float:
        if module_name not in self._history:
            self._history[module_name] = deque(maxlen=self.history_length)
        hist = self._history[module_name]
        hist.append(value)

        if len(hist) < 3:
            return 0.75

        arr = np.array(hist, dtype=np.float32)
        mean_abs = float(np.mean(np.abs(arr))) + 1e-6
        variation = float(np.std(np.diff(arr))) / mean_abs
        return float(np.clip(1.0 - variation, 0.0, 1.0))


class ConfidenceEstimator:
    """Computes per-module dynamic confidence scores for the current frame."""

    def __init__(
        self,
        detection_signal_weights: Optional[Dict[str, float]] = None,
        lane_signal_weights: Optional[Dict[str, float]] = None,
        depth_signal_weights: Optional[Dict[str, float]] = None,
        history_length: int = 30,
    ) -> None:
        self.detection_signal_weights = detection_signal_weights or {
            "model_score": 0.5, "weather_penalty": 0.3, "box_stability": 0.2,
        }
        self.lane_signal_weights = lane_signal_weights or {
            "model_score": 0.5, "weather_penalty": 0.35, "curve_consistency": 0.15,
        }
        self.depth_signal_weights = depth_signal_weights or {
            "model_score": 0.4, "weather_penalty": 0.4, "edge_consistency": 0.2,
        }
        self.stability_tracker = StabilityTracker(history_length=history_length)

    def estimate_detection_confidence(self, mean_detection_score: float,
                                       weather_modifier: float, num_detections: int) -> ModuleConfidence:
        stability = self.stability_tracker.update("detection_count", float(num_detections))
        w = self.detection_signal_weights
        final = w["model_score"] * mean_detection_score + w["weather_penalty"] * weather_modifier \
            + w["box_stability"] * stability
        return ModuleConfidence("camera_detection", mean_detection_score, weather_modifier,
                                 stability, float(np.clip(final, 0.0, 1.0)))

    def estimate_lane_confidence(self, mean_lane_score: float,
                                  weather_modifier: float, num_lanes: int) -> ModuleConfidence:
        stability = self.stability_tracker.update("lane_count", float(num_lanes))
        w = self.lane_signal_weights
        final = w["model_score"] * mean_lane_score + w["weather_penalty"] * weather_modifier \
            + w["curve_consistency"] * stability
        return ModuleConfidence("camera_lane", mean_lane_score, weather_modifier,
                                 stability, float(np.clip(final, 0.0, 1.0)))

    def estimate_depth_confidence(self, depth_map_std: float,
                                   weather_modifier: float, max_depth_m: float) -> ModuleConfidence:
        normalized_std = float(np.clip(depth_map_std / (max_depth_m * 0.5), 0.0, 1.0))
        stability = self.stability_tracker.update("depth_std", depth_map_std)
        w = self.depth_signal_weights
        final = w["model_score"] * normalized_std + w["weather_penalty"] * weather_modifier \
            + w["edge_consistency"] * stability
        return ModuleConfidence("camera_depth", normalized_std, weather_modifier,
                                 stability, float(np.clip(final, 0.0, 1.0)))


# =============================================================================
# Adaptive sensor fusion
# =============================================================================


@dataclass
class FusionResult:
    sensor_weights: Dict[str, float] = field(default_factory=dict)
    module_confidences: Dict[str, float] = field(default_factory=dict)
    overall_scene_confidence: float = 0.0
    weather_label: str = "unknown"


class AdaptiveFusionEngine:
    """Computes dynamic, weather-aware sensor weights and a fused scene confidence."""

    def __init__(self, base_weights: Dict[str, float],
                 weather_modifiers: Dict[str, Dict[str, float]], ema_alpha: float = 0.25) -> None:
        self.base_weights = dict(base_weights)
        self.weather_modifiers = weather_modifiers
        self.ema_alpha = ema_alpha
        self._smoothed_weights: Optional[Dict[str, float]] = None

    def get_weather_modifier(self, weather_label: str, module_key: str) -> float:
        modifiers = self.weather_modifiers.get(weather_label)
        if modifiers is None:
            logger.debug("No weather modifier for '%s'; defaulting to 1.0.", weather_label)
            return 1.0
        return float(modifiers.get(module_key, 1.0))

    def compute_raw_weights(self, weather_label: str,
                             module_confidences: List[ModuleConfidence]) -> Dict[str, float]:
        raw_weights: Dict[str, float] = {}
        for mc in module_confidences:
            base = self.base_weights.get(mc.module_name, 0.5)
            modifier = self.get_weather_modifier(weather_label, mc.module_name)
            raw = 0.5 * (base * modifier) + 0.5 * mc.final_confidence
            raw_weights[mc.module_name] = float(np.clip(raw, 0.0, 1.0))
        return raw_weights

    def smooth_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        if self._smoothed_weights is None:
            self._smoothed_weights = dict(raw_weights)
            return dict(self._smoothed_weights)
        smoothed = {
            k: self.ema_alpha * v + (1 - self.ema_alpha) * self._smoothed_weights.get(k, v)
            for k, v in raw_weights.items()
        }
        self._smoothed_weights = smoothed
        return dict(smoothed)

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 1e-6:
            n = len(weights) or 1
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    def fuse(self, weather_label: str, module_confidences: List[ModuleConfidence]) -> FusionResult:
        raw_weights = self.compute_raw_weights(weather_label, module_confidences)
        smoothed_weights = self.smooth_weights(raw_weights)
        module_conf_map = {mc.module_name: mc.final_confidence for mc in module_confidences}

        normalized = self.normalize_weights(smoothed_weights)
        overall_confidence = sum(normalized[k] * module_conf_map.get(k, 0.0) for k in normalized)

        return FusionResult(
            sensor_weights=smoothed_weights, module_confidences=module_conf_map,
            overall_scene_confidence=float(np.clip(overall_confidence, 0.0, 1.0)),
            weather_label=weather_label,
        )
