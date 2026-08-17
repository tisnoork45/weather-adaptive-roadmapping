"""Risk classification from TTC, distance, and current weather condition.

New module. Combines the new `ttc_calculator.py` output with the existing,
unmodified `models.py::WeatherClassifier` output to produce a discrete risk
level per object and an overall frame-level risk level.

Weather-adaptive thresholds: the SAME time-to-collision value should be
treated as more dangerous in poor visibility/traction conditions (fog,
rain, snow, night) than in clear weather, since reaction margin and
stopping distance are worse. This is implemented by multiplying the base
TTC thresholds by a per-weather factor from config.yaml (>1.0 = thresholds
become larger = a given TTC value escalates to a higher risk level sooner
in that weather).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ttc_calculator import TTCResult
from utils import get_logger

logger = get_logger(__name__)

RISK_LEVELS = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
_RISK_ORDER = {level: i for i, level in enumerate(RISK_LEVELS)}


@dataclass
class ObjectRisk:
    track_id: int
    class_name: str
    distance_m: float
    ttc_seconds: Optional[float]
    risk_level: str


@dataclass
class RiskAssessment:
    per_object: List[ObjectRisk] = field(default_factory=list)
    overall_risk_level: str = "SAFE"
    weather_label: str = "unknown"
    active_thresholds_s: Dict[str, float] = field(default_factory=dict)


class RiskEngine:
    """Classifies TTC/distance into a discrete risk level, weather-adjusted."""

    def __init__(
        self,
        ttc_thresholds_s: Dict[str, float],
        weather_multipliers: Dict[str, float],
        min_safe_distance_m: float = 5.0,
    ) -> None:
        """
        Args:
            ttc_thresholds_s: base thresholds in seconds, keys "CRITICAL",
                "WARNING", "CAUTION" (e.g. {"CRITICAL": 1.5, "WARNING": 3.0,
                "CAUTION": 5.0}). An object's TTC below the "CRITICAL"
                threshold -> CRITICAL; below "WARNING" -> WARNING; below
                "CAUTION" -> CAUTION; otherwise (or TTC is None, i.e. not
                closing) -> SAFE from the TTC criterion (see also the
                distance-based floor below).
            weather_multipliers: per-weather-label multiplier applied to
                every base threshold (e.g. {"fog": 1.6, "clear": 1.0}). A
                multiplier > 1.0 makes thresholds larger, so the same TTC
                value escalates to a worse risk level sooner in that
                weather. Unrecognized/missing weather labels default to 1.0
                (no adjustment).
            min_safe_distance_m: even when an object is NOT closing (no
                finite TTC, e.g. a stopped vehicle directly ahead), being
                very close is still worth flagging -- this is a distance-
                only floor applied regardless of TTC, so a stationary near
                obstacle isn't reported as SAFE just because it isn't
                actively closing this instant.
        """
        self.ttc_thresholds_s = dict(ttc_thresholds_s)
        self.weather_multipliers = dict(weather_multipliers)
        self.min_safe_distance_m = min_safe_distance_m

    def get_weather_adjusted_thresholds(self, weather_label: str) -> Dict[str, float]:
        multiplier = self.weather_multipliers.get(weather_label, 1.0)
        return {level: base * multiplier for level, base in self.ttc_thresholds_s.items()}

    def assess(self, weather_label: str, ttc_results: List[TTCResult]) -> RiskAssessment:
        thresholds = self.get_weather_adjusted_thresholds(weather_label)

        per_object: List[ObjectRisk] = []
        for r in ttc_results:
            level = self._classify(r.distance_m, r.ttc_seconds, thresholds)
            per_object.append(ObjectRisk(
                track_id=r.track_id, class_name=r.class_name,
                distance_m=r.distance_m, ttc_seconds=r.ttc_seconds, risk_level=level,
            ))

        overall = self._worst_level([o.risk_level for o in per_object])
        return RiskAssessment(
            per_object=per_object, overall_risk_level=overall,
            weather_label=weather_label, active_thresholds_s=thresholds,
        )

    def _classify(self, distance_m: float, ttc_s: Optional[float], thresholds: Dict[str, float]) -> str:
        if ttc_s is not None:
            if ttc_s < thresholds.get("CRITICAL", 0.0):
                return "CRITICAL"
            if ttc_s < thresholds.get("WARNING", 0.0):
                return "WARNING"
            if ttc_s < thresholds.get("CAUTION", 0.0):
                return "CAUTION"
        # Distance-only floor: catches close-but-not-actively-closing objects
        # (e.g. a stopped car directly ahead) that TTC alone would miss.
        if distance_m < self.min_safe_distance_m * 0.5:
            return "WARNING"
        if distance_m < self.min_safe_distance_m:
            return "CAUTION"
        return "SAFE"

    @staticmethod
    def _worst_level(levels: List[str]) -> str:
        if not levels:
            return "SAFE"
        return max(levels, key=lambda level: _RISK_ORDER.get(level, 0))
