"""End-to-end weather-adaptive perception pipeline + dashboard render helpers.

`PerceptionPipeline` orchestrates, per frame: weather classification, object
detection, tracking, depth estimation, dynamic confidence estimation,
adaptive fusion, and BEV mapping. The dashboard-rendering functions at the
bottom of this file are thin UI helpers used by app.py (Streamlit); two of
them -- `draw_primary_object_panel` and `draw_ego_speed_panel` -- are plain
cv2 pixel-drawing functions (not HTML) and are also reused by run.py's
OpenCV window, so both UIs share one implementation instead of duplicating
it. They are kept in the same file as the pipeline (rather than a separate
`dashboard` package) since together they are still small.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import cv2

from fusion import AdaptiveFusionEngine, ConfidenceEstimator, FusionResult, ModuleConfidence
from mapping import BEVMapper, BEVResult
from models import (
    DepthEstimator, DepthResult, DetectionResult,
    WeatherClassifier, WeatherResult, YOLODetector,
)
from relative_velocity import RelativeVelocity, RelativeVelocityEstimator
from risk_engine import RiskAssessment, RiskEngine
from tracking import ByteTracker
from ttc_calculator import compute_ttc_for_objects
from utils import ConfigDict, MetricsTracker, RISK_LEVEL_COLORS, draw_detections, get_logger

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    """Aggregated output of one full pipeline pass over a single frame."""

    frame: np.ndarray
    weather: WeatherResult
    detections: DetectionResult
    depth: DepthResult
    fusion: FusionResult
    bev: Optional[BEVResult]
    # NEW: populated only when BEV mapping is enabled and produced at least
    # one tracked object with a valid depth-derived position; otherwise
    # relative_velocities stays {} and risk stays None (no crash either way).
    relative_velocities: Dict[int, RelativeVelocity] = field(default_factory=dict)
    risk: Optional[RiskAssessment] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


class PerceptionPipeline:
    """High-level façade wiring together every perception module.

    Instantiated once (loading all models), then call `.process(frame)` for
    every incoming camera frame.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        device_pref = config.device.preference

        logger.info("Initializing perception pipeline modules...")

        self.detector = YOLODetector(
            weights_path=config.detection.weights_path,
            confidence_threshold=config.detection.confidence_threshold,
            iou_threshold=config.detection.iou_threshold,
            input_size=config.detection.input_size,
            classes_of_interest=config.detection.classes_of_interest,
            device_preference=device_pref,
        )

        self.depth_estimator = DepthEstimator(
            weights_path=config.depth_estimation.weights_path,
            encoder=config.depth_estimation.encoder,
            input_size=config.depth_estimation.input_size,
            max_depth_m=config.depth_estimation.max_depth_m,
            device_preference=device_pref,
        )

        self.weather_classifier = WeatherClassifier(
            weights_path=config.weather_classification.weights_path,
            classes=config.weather_classification.classes,
            input_size=config.weather_classification.input_size,
            normalize_mean=config.weather_classification.normalize_mean,
            normalize_std=config.weather_classification.normalize_std,
            device_preference=device_pref,
        )

        self.tracker = ByteTracker(
            track_thresh=config.tracking.track_thresh,
            track_buffer=config.tracking.track_buffer,
            match_thresh=config.tracking.match_thresh,
            frame_rate=config.tracking.frame_rate,
            min_box_area=config.tracking.min_box_area,
        )

        self.confidence_estimator = ConfidenceEstimator(
            detection_signal_weights=config.confidence_estimation.detection_signal_weights,
            depth_signal_weights=config.confidence_estimation.depth_signal_weights,
            history_length=config.confidence_estimation.history_length,
        )

        self.fusion_engine = AdaptiveFusionEngine(
            base_weights=config.fusion.base_weights,
            weather_modifiers=config.fusion.weather_modifiers,
            ema_alpha=config.fusion.ema_alpha,
        )

        self.bev_mapper = BEVMapper(
            fx=config.camera.intrinsics.fx, fy=config.camera.intrinsics.fy,
            cx=config.camera.intrinsics.cx, cy=config.camera.intrinsics.cy,
            bev_range_m=config.mapping.bev_range_m, bev_width_m=config.mapping.bev_width_m,
            resolution_px_per_m=config.mapping.bev_resolution_px_per_m,
        )

        # NEW: relative velocity (from BEV object positions) -> TTC -> risk.
        # Reads the `velocity:`/`risk:` sections already present in
        # config.yaml. Purely additive -- nothing above this line changed.
        self.velocity_estimator = RelativeVelocityEstimator(
            history_length=config.velocity.history_length,
            max_gap_seconds=config.velocity.max_gap_seconds,
            ema_alpha=config.velocity.ema_alpha,
        )
        self.risk_engine = RiskEngine(
            ttc_thresholds_s=config.risk.ttc_thresholds_s,
            weather_multipliers=config.risk.weather_multipliers,
            min_safe_distance_m=config.risk.min_safe_distance_m,
        )

        self.pipeline_cfg = config.pipeline
        self.max_depth_m = config.depth_estimation.max_depth_m
        self.metrics = MetricsTracker(
            stages=["weather", "detection", "tracking", "depth", "fusion", "bev", "risk"]
        )
        self._frame_index = 0
        logger.info("All perception pipeline modules initialized successfully.")

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PipelineOutput:
        """Run the full perception stack on a single BGR frame.

        Args:
            frame: BGR frame.
            timestamp: the real-world (or video-timeline) time, in seconds,
                that this frame represents -- used only by the relative-
                velocity estimator to compute elapsed time between
                observations of the same tracked object. Optional and
                backward compatible: if omitted (None), falls back to
                wall-clock `time.perf_counter()` inside
                `RelativeVelocityEstimator.update()`, which is correct for a
                live camera (every processed frame IS the current real
                moment) or the synthetic demo stream, but is WRONG for a
                prerecorded video on a slow CPU pipeline, where processing
                one frame can take several real seconds despite the video's
                own timeline advancing by a fraction of a second -- callers
                driving a video file should compute and pass an explicit
                video-timeline timestamp (frame_index / video_fps) instead.
                See run.py's main loop for the reference implementation.
        """
        overall_start = time.perf_counter()
        self._frame_index += 1

        weather_result = self.weather_classifier.infer(frame)
        self.metrics.record_stage("weather", weather_result.inference_time_ms)

        detection_result = self.detector.infer(frame)
        self.metrics.record_stage("detection", detection_result.inference_time_ms)

        tracked_start = time.perf_counter()
        if self.pipeline_cfg.run_tracking:
            detection_result.detections = self.tracker.update(detection_result.detections)
        self.metrics.record_stage("tracking", (time.perf_counter() - tracked_start) * 1000.0)

        if self.pipeline_cfg.run_depth_estimation:
            depth_result = self.depth_estimator.infer(frame)
        else:
            depth_result = DepthResult(depth_map=np.zeros(frame.shape[:2], dtype=np.float32))
        self.metrics.record_stage("depth", depth_result.inference_time_ms)

        fusion_start = time.perf_counter()
        module_confidences = self._estimate_confidences(weather_result, detection_result, depth_result)
        fusion_result = self.fusion_engine.fuse(weather_result.label, module_confidences)
        self.metrics.record_stage("fusion", (time.perf_counter() - fusion_start) * 1000.0)

        bev_result = None
        if self.pipeline_cfg.run_bev_mapping:
            bev_result = self.bev_mapper.build_bev(
                frame, depth_result.depth_map, detection_result.detections,
            )
            self.metrics.record_stage("bev", bev_result.inference_time_ms)

        # NEW: relative velocity -> TTC -> weather-adjusted risk. Depends on
        # BEV's per-object (x_m, z_m) positions, so if BEV mapping is
        # disabled (or produced zero valid positions this frame) there is
        # nothing to compute risk from -- that is not an error, it just
        # means relative_velocities stays {} and risk stays None for this
        # frame, exactly like "no detections" already means an empty
        # DetectionResult rather than a crash.
        risk_stage_start = time.perf_counter()
        relative_velocities: Dict[int, RelativeVelocity] = {}
        risk_result: Optional[RiskAssessment] = None
        if bev_result is not None:
            relative_velocities = self.velocity_estimator.update(bev_result.objects, timestamp=timestamp)
            ttc_results = compute_ttc_for_objects(bev_result.objects, relative_velocities)
            risk_result = self.risk_engine.assess(weather_result.label, ttc_results)
        self.metrics.record_stage("risk", (time.perf_counter() - risk_stage_start) * 1000.0)

        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        self.metrics.record_frame(object_count=len(detection_result))

        return PipelineOutput(
            frame=frame, weather=weather_result, detections=detection_result,
            depth=depth_result, fusion=fusion_result, bev=bev_result,
            relative_velocities=relative_velocities, risk=risk_result,
            metrics=self.metrics.summary(), total_latency_ms=total_latency_ms,
        )

    def _estimate_confidences(self, weather_result: WeatherResult, detection_result: DetectionResult,
                               depth_result: DepthResult) -> List[ModuleConfidence]:
        mean_det_score = float(np.mean(detection_result.scores())) if len(detection_result) else 0.0
        detection_conf = self.confidence_estimator.estimate_detection_confidence(
            mean_detection_score=mean_det_score,
            weather_modifier=self.fusion_engine.get_weather_modifier(weather_result.label, "camera_detection"),
            num_detections=len(detection_result),
        )

        depth_std = float(np.std(depth_result.depth_map)) if depth_result.depth_map.size else 0.0
        depth_conf = self.confidence_estimator.estimate_depth_confidence(
            depth_map_std=depth_std,
            weather_modifier=self.fusion_engine.get_weather_modifier(weather_result.label, "camera_depth"),
            max_depth_m=self.max_depth_m,
        )

        return [detection_conf, depth_conf]


# =============================================================================
# Streamlit dashboard rendering helpers (used by app.py)
#
# UI-ONLY layer: every function here reads already-computed PipelineOutput /
# RiskEngine state and formats it for display. None of them run inference,
# compute TTC/velocity/risk, or duplicate any perception logic -- see
# PerceptionPipeline.process() above for where those numbers actually come
# from. Panels return either an HTML string (rendered via
# `st.markdown(..., unsafe_allow_html=True)`), a numpy image, or a
# pandas.DataFrame; app.py is responsible for writing each one into a stable
# `st.empty()` placeholder with exactly ONE call per frame (never several
# sequential st.* calls into the same container), which is what prevents
# per-frame UI accumulation.
# =============================================================================


def _bgr_to_css_rgb(bgr) -> str:
    b, g, r = bgr
    return f"rgb({r},{g},{b})"


def _risk_status_color(level: Optional[str]) -> str:
    if level is None:
        return "#8a8f98"
    return _bgr_to_css_rgb(RISK_LEVEL_COLORS.get(level, (200, 200, 200)))


def _progress_bar_html(label: str, value_0_1: float, right_text: str = "") -> str:
    pct = max(0.0, min(1.0, value_0_1)) * 100.0
    right = right_text or f"{pct:.0f}%"
    return (
        f'<div class="bar-row"><span class="bar-label">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="bar-value">{right}</span></div>'
    )


def _format_distance(distance_m: Optional[float]) -> str:
    return f"{distance_m:.1f} m" if distance_m is not None else "N/A"


def _format_ttc(ttc_seconds: Optional[float]) -> str:
    return f"{ttc_seconds:.1f} s" if ttc_seconds is not None else "N/A"


def _format_relative_speed(velocity: Optional[RelativeVelocity]) -> str:
    """Unit-conversion / formatting only -- reads the existing
    `RelativeVelocity.vz_m_s` field (backend-computed) and applies the same
    closing-speed sign convention `ttc_calculator.py` already documents
    (positive = closing), just converted to km/h and worded for display.
    """
    if velocity is None or not velocity.valid:
        return "N/A"
    closing_kmh = -velocity.vz_m_s * 3.6
    if closing_kmh > 0.3:
        direction = "Approaching"
    elif closing_kmh < -0.3:
        direction = "Receding"
    else:
        direction = "Stable"
    return f"{closing_kmh:+.1f} km/h ({direction})"


DASHBOARD_CSS = """
<style>
.stApp { background-color: #0a0e13; }
.hud-header { display:flex; justify-content:space-between; align-items:flex-start;
    padding: 2px 0 12px 0; border-bottom: 1px solid rgba(0,229,255,0.18); margin-bottom: 12px; }
.hud-title { font-family: 'Courier New', monospace; font-size: 1.55rem; font-weight:700;
    color:#e8f6ff; letter-spacing:1.5px; }
.hud-subtitle { font-size:0.78rem; color:#7a8896; margin-top:2px; letter-spacing:0.5px; }
.hud-status-row { display:flex; flex-direction:column; align-items:flex-end; gap:5px; }
.hud-status-badge { font-family:'Courier New',monospace; font-size:0.85rem; font-weight:700;
    padding:4px 14px; border:1px solid; border-radius:3px; letter-spacing:1.5px; }
.hud-status-badge .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }
.hud-status-sub { font-size:0.68rem; color:#7a8896; letter-spacing:0.5px; }
.active-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:#00e5ff; margin:0 5px; }

.metric-row { display:flex; justify-content:space-between; background:rgba(16,20,26,0.75);
    border:1px solid rgba(0,229,255,0.12); border-radius:4px; padding:10px 20px; margin-bottom:12px; }
.metric-cell { text-align:left; }
.metric-label { font-size:0.62rem; color:#7a8896; letter-spacing:1.5px; font-family:'Courier New',monospace; }
.metric-value { font-size:1.15rem; color:#e8f6ff; font-weight:700; font-family:'Courier New',monospace; margin-top:2px;}

.hud-card { background:rgba(16,20,26,0.75); border:1px solid rgba(0,229,255,0.12);
    border-radius:4px; padding:14px 16px; margin-bottom: 12px; min-height: 100%; }
.hud-card-title { font-size:0.72rem; color:#00e5ff; letter-spacing:1.5px; font-weight:700;
    margin-bottom:12px; font-family:'Courier New',monospace; }

.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.bar-label { flex: 0 0 108px; font-size:0.68rem; color:#c7d3dd; font-family:'Courier New',monospace; }
.bar-track { flex:1; height:7px; background:rgba(255,255,255,0.08); border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; background:linear-gradient(90deg,#0091a8,#00e5ff); }
.bar-value { flex:0 0 44px; text-align:right; font-size:0.68rem; color:#e8f6ff; font-family:'Courier New',monospace; }

.diag-row { display:flex; justify-content:space-between; font-size:0.75rem; color:#c7d3dd;
    font-family:'Courier New',monospace; padding:2px 0; }
.diag-row.total { border-top:1px solid rgba(255,255,255,0.12); margin-top:6px; padding-top:7px;
    font-weight:700; color:#e8f6ff; }

.status-line { display:flex; justify-content:space-between; align-items:center; font-size:0.68rem;
    padding:2px 0 8px 0; font-family:'Courier New',monospace; color:#c7d3dd; }
.status-tag { font-size:0.6rem; padding:1px 9px; border-radius:8px; letter-spacing:0.5px; }

.risk-guide-row { display:flex; align-items:center; gap:10px; font-size:0.72rem;
    font-family:'Courier New',monospace; color:#c7d3dd; padding:4px 0; }
.risk-guide-dot { width:9px; height:9px; border-radius:50%; flex:0 0 9px; }
.risk-guide-label { flex:0 0 78px; font-weight:700; }
</style>
"""


def render_dashboard_css() -> str:
    """Injected once via st.markdown at app startup -- never re-injected
    per frame, so it cannot contribute to per-frame accumulation."""
    return DASHBOARD_CSS


def render_header(st, title: str) -> None:
    """Static title block -- call ONCE, before the live loop starts. The
    live road-status badge is separate (render_status_badge_html), since
    that part changes every frame and must go through its own placeholder.
    """
    st.markdown(
        f"""<div class="hud-header">
            <div>
                <div class="hud-title">{title.upper()}</div>
                <div class="hud-subtitle">Autonomous Vehicle Perception System</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_status_badge_html(output: PipelineOutput) -> str:
    """Live road-status badge, sourced from output.risk.overall_risk_level
    (the actual backend RiskEngine result). If risk is unavailable this
    frame (e.g. BEV mapping disabled), shows N/A rather than a fabricated
    status -- never defaults to SAFE by omission."""
    level = output.risk.overall_risk_level if output.risk is not None else None
    label = level if level is not None else "N/A"
    color = _risk_status_color(level)
    return f"""
    <div class="hud-status-row">
        <div class="hud-status-badge" style="border-color:{color};color:{color};">
            <span class="dot" style="background:{color};"></span>ROAD STATUS: {label}
        </div>
        <div class="hud-status-sub">VISION PERCEPTION<span class="active-dot"></span>ACTIVE</div>
    </div>"""


def render_metric_row_html(output: PipelineOutput) -> str:
    """WEATHER / CONFIDENCE / OBJECTS / FPS / LATENCY, all read directly
    from PipelineOutput -- CONFIDENCE here is the weather classifier's own
    confidence (output.weather.confidence), matching what's shown again,
    in more detail, in the weather card. Overall fusion scene-confidence is
    shown separately inside the Sensor Status panel."""
    items = [
        ("WEATHER", output.weather.label.upper()),
        ("CONFIDENCE", f"{output.weather.confidence * 100:.0f}%"),
        ("OBJECTS", f"{len(output.detections)}"),
        ("FPS", f"{output.metrics.get('fps', 0.0):.1f}"),
        ("LATENCY", f"{output.total_latency_ms:.0f} ms"),
    ]
    cells = "".join(
        f'<div class="metric-cell"><div class="metric-label">{k}</div>'
        f'<div class="metric-value">{v}</div></div>'
        for k, v in items
    )
    return f'<div class="metric-row">{cells}</div>'


def _select_primary_object(output: PipelineOutput):
    """Picks the single most relevant tracked object for the on-camera HUD
    card: worst risk level first (CRITICAL > WARNING > CAUTION > SAFE),
    same priority ordering already used by utils.draw_risk_panel. Returns
    (ObjectRisk, detection_confidence_or_None, RelativeVelocity_or_None),
    or None if there is nothing to show."""
    if output.risk is None or not output.risk.per_object:
        return None
    order = {"CRITICAL": 0, "WARNING": 1, "CAUTION": 2, "SAFE": 3}
    obj_risk = sorted(output.risk.per_object, key=lambda o: order.get(o.risk_level, 4))[0]
    det_conf = None
    for d in output.detections.detections:
        if d.track_id == obj_risk.track_id:
            det_conf = d.confidence
            break
    velocity = output.relative_velocities.get(obj_risk.track_id)
    return obj_risk, det_conf, velocity


def draw_primary_object_panel(frame: np.ndarray, output: PipelineOutput) -> np.ndarray:
    """Compact translucent HUD card for the highest-risk tracked object
    (Rule 11): class/ID/confidence, distance, relative speed, TTC, status.
    All values read straight from output.risk / output.relative_velocities /
    output.detections -- nothing computed here. Distinguishes three honest
    states: risk unavailable (BEV disabled), risk available but zero
    objects, and a real featured object. Public (no leading underscore):
    shared by both app.py's camera feed and run.py's OpenCV window.
    """
    h, w = frame.shape[:2]
    # 340px (widened from 270px): at the "Relative Speed" row's font sizes,
    # the label ("Relative Speed", ~98px) plus its longest real value
    # ("+123.4 km/h (Approaching)", ~205px) needs ~327px including margins
    # -- 270px let the value text overlap the label on that row. Measured
    # directly against cv2.getTextSize, not guessed.
    panel_w = 340
    x0, y0 = w - panel_w - 14, 14

    if output.risk is None:
        lines = ["OBJECT STATUS", "N/A - BEV mapping disabled"]
        panel_h = 20 + 24 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 22, 26), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)
        y = y0 + 22
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i == 0 else (170, 170, 170)
            scale = 0.5 if i == 0 else 0.42
            cv2.putText(frame, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 24
        return frame

    primary = _select_primary_object(output)
    overlay = frame.copy()

    if primary is None:
        panel_h = 56
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 22, 26), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)
        cv2.putText(frame, "NO TRACKED OBJECTS", (x0 + 12, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    obj_risk, det_conf, velocity = primary
    color = RISK_LEVEL_COLORS.get(obj_risk.risk_level, (200, 200, 200))
    rows = [
        ("Distance", _format_distance(obj_risk.distance_m)),
        ("Relative Speed", _format_relative_speed(velocity)),
        ("Time to Collision", _format_ttc(obj_risk.ttc_seconds)),
        ("Status", obj_risk.risk_level),
    ]
    panel_h = 40 + 24 * len(rows)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 22, 26), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), color, 1)

    title = f"{obj_risk.class_name.upper()} #{obj_risk.track_id}"
    cv2.putText(frame, title, (x0 + 12, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if det_conf is not None:
        conf_txt = f"{det_conf * 100:.0f}%"
        (tw, _), _ = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, conf_txt, (x0 + panel_w - tw - 12, y0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    y = y0 + 50
    for label, value in rows:
        cv2.putText(frame, label, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        val_color = color if label == "Status" else (255, 255, 255)
        cv2.putText(frame, value, (x0 + panel_w - tw - 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 1, cv2.LINE_AA)
        y += 24
    return frame


def draw_ego_speed_panel(frame: np.ndarray, center_x: bool = False) -> np.ndarray:
    """Small (~150x64px) panel -- deliberately far from the road center /
    primary object panel. The backend has no odometry/speed sensor anywhere
    (no such field exists on PipelineOutput, config.yaml, or any perception
    module -- relative_velocity.py's own docstring states this project has
    no odometry/IMU input), so this always reads N/A. Never fabricates a
    vehicle speed.

    `center_x`: if False (default), bottom-left corner -- byte-identical to
    this function's original, pre-rename position (app.py's camera feed
    keeps this default, so the Streamlit dashboard's layout is unchanged).
    If True, horizontally centered along the bottom edge instead -- used by
    run.py's OpenCV window, which places a separate sensor-status panel in
    the bottom-left corner and would otherwise overlap it.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 150, 64
    x0 = (w - panel_w) // 2 if center_x else 14
    y0 = h - panel_h - 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 22, 26), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (110, 110, 110), 1)
    cv2.putText(frame, "EGO SPEED", (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, "N/A", (x0 + 10, y0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def render_camera_feed(output: PipelineOutput) -> np.ndarray:
    """The dominant dashboard element. Reuses utils.draw_detections (its
    alpha/distances params already existed, just weren't wired to a caller
    before now) plus two small HUD overlays for the primary-risk-object card
    and the ego-speed indicator. Draws NOTHING that isn't already present on
    PipelineOutput.
    """
    annotated = output.frame.copy()
    dets = output.detections
    track_ids = [d.track_id if d.track_id is not None else i for i, d in enumerate(dets.detections)]

    # Per-detection distance, aligned by track_id (not by list position --
    # BEVMapper.project_objects_to_bev() silently drops detections with
    # invalid depth, so BEV objects are a subset of detections, never a
    # positional match). None where no valid BEV position exists this frame.
    bev_distance_by_track = {}
    if output.bev is not None:
        for obj in output.bev.objects:
            if obj.track_id is not None:
                bev_distance_by_track[obj.track_id] = obj.z_m
    distances = [bev_distance_by_track.get(d.track_id) for d in dets.detections]

    # Detection opacity driven by the ACTUAL current adaptive sensor
    # weights -- makes the project's core weather-adaptive-fusion thesis
    # visible directly on the camera feed (lower trust in fog/rain/night ->
    # visibly fainter overlays), using an already-computed backend value.
    detection_alpha = output.fusion.sensor_weights.get("camera_detection", 1.0)

    annotated = draw_detections(
        annotated, dets.boxes(), dets.labels(), dets.scores(), track_ids,
        distances=distances, alpha=detection_alpha,
    )
    annotated = draw_primary_object_panel(annotated, output)
    annotated = draw_ego_speed_panel(annotated)
    return annotated


def render_weather_card_html(output: PipelineOutput) -> str:
    """ONE compact weather card (Rule 14) -- not a chart. Uses the actual
    WeatherClassifier output; performs no inference."""
    label = output.weather.label.upper()
    conf_pct = output.weather.confidence * 100.0
    probs = sorted(output.weather.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    bars = "".join(_progress_bar_html(cls.upper(), p, f"{p * 100:.0f}%") for cls, p in probs)
    return f"""
    <div class="hud-card">
        <div class="hud-card-title">WEATHER</div>
        <div style="font-size:1.25rem;font-weight:700;color:#e8f6ff;font-family:'Courier New',monospace;">{label}</div>
        <div style="font-size:0.72rem;color:#7a8896;margin-bottom:10px;">{conf_pct:.0f}% confidence</div>
        {bars}
    </div>"""


def render_sensor_status_html(output: PipelineOutput, pipeline_cfg) -> str:
    """Rule 13: per-module confidence where a real one exists
    (fusion.module_confidences, for detection/depth), ACTIVE/INACTIVE
    (from the real config.pipeline flags) where it doesn't (tracking, BEV --
    neither produces a scalar confidence anywhere in the backend)."""
    conf = output.fusion.module_confidences

    def _row(label: str, value: Optional[float], active: bool) -> str:
        status_color = "#00e5ff" if active else "#5a6472"
        status_text = "ACTIVE" if active else "INACTIVE"
        tag = f'<span class="status-tag" style="color:{status_color};border:1px solid {status_color};">{status_text}</span>'
        if value is None:
            return f'<div class="status-line"><span>{label}</span>{tag}</div>'
        bar = _progress_bar_html(label, value, f"{value * 100:.0f}%")
        return f'{bar}<div style="text-align:right;margin-top:-5px;">{tag}</div>'

    rows = (
        _row("Camera Detection", conf.get("camera_detection"), True)
        + _row("Depth Estimation", conf.get("camera_depth"), pipeline_cfg.run_depth_estimation)
        + _row("Object Tracking", None, pipeline_cfg.run_tracking)
        + _row("BEV Mapping", None, pipeline_cfg.run_bev_mapping)
    )
    scene_conf = output.fusion.overall_scene_confidence * 100.0
    return f"""
    <div class="hud-card">
        <div class="hud-card-title">SENSOR STATUS</div>
        {rows}
        <div style="font-size:0.65rem;color:#7a8896;margin-top:4px;">Overall scene confidence: {scene_conf:.0f}%</div>
    </div>"""


def render_risk_guide_html(output: PipelineOutput, risk_engine: Optional[RiskEngine]) -> str:
    """Rule 15: NEVER hardcodes TTC thresholds -- calls the real
    RiskEngine.get_weather_adjusted_thresholds() (an existing, unmodified
    backend method) for the CURRENT weather label, so fog/rain/snow
    visibly change the displayed thresholds, exactly matching what
    RiskEngine.assess() actually used to classify this frame's objects.
    """
    thresholds = {}
    if risk_engine is not None:
        thresholds = risk_engine.get_weather_adjusted_thresholds(output.weather.label)
    critical, warning, caution = thresholds.get("CRITICAL"), thresholds.get("WARNING"), thresholds.get("CAUTION")

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.1f}s" if v is not None else "N/A"

    rows = [
        ("CRITICAL", RISK_LEVEL_COLORS["CRITICAL"], f"TTC &lt; {_fmt(critical)}"),
        ("WARNING", RISK_LEVEL_COLORS["WARNING"], f"{_fmt(critical)} &ndash; {_fmt(warning)}"),
        ("CAUTION", RISK_LEVEL_COLORS["CAUTION"], f"{_fmt(warning)} &ndash; {_fmt(caution)}"),
        ("SAFE", RISK_LEVEL_COLORS["SAFE"], f"TTC &ge; {_fmt(caution)}"),
    ]
    rows_html = "".join(
        f'<div class="risk-guide-row"><span class="risk-guide-dot" style="background:{_bgr_to_css_rgb(color)};"></span>'
        f'<span class="risk-guide-label">{level}</span><span>{desc}</span></div>'
        for level, color, desc in rows
    )
    return f"""
    <div class="hud-card">
        <div class="hud-card-title">RISK LEVEL GUIDE</div>
        {rows_html}
        <div style="font-size:0.62rem;color:#7a8896;margin-top:8px;">Weather-adjusted for: {output.weather.label.upper()}</div>
    </div>"""


def render_sensor_weights_html(output: PipelineOutput) -> str:
    """Rule 18: actual output.fusion.sensor_weights, no invented values."""
    weights = output.fusion.sensor_weights
    label_map = {"camera_detection": "Detection", "camera_depth": "Depth"}
    bars = "".join(
        _progress_bar_html(label_map.get(k, k), v, f"{v:.2f}") for k, v in weights.items()
    ) or '<div style="color:#7a8896;font-size:0.72rem;">No fusion data yet.</div>'
    return f"""
    <div class="hud-card">
        <div class="hud-card-title">ADAPTIVE SENSOR WEIGHTS</div>
        {bars}
    </div>"""


def render_depth_panel(output: PipelineOutput) -> np.ndarray:
    """Unchanged from the original implementation."""
    from utils import depth_to_colormap

    return depth_to_colormap(output.depth.depth_map)


def render_bev_panel(output: PipelineOutput) -> np.ndarray:
    """Unchanged from the original implementation."""
    if output.bev is not None:
        return output.bev.bev_image
    return np.zeros((400, 300, 3), dtype=np.uint8)


def render_latency_html(output: PipelineOutput) -> str:
    """Rule 19: moved out of the main view into a diagnostics expander;
    now also includes the 'risk' stage (added to MetricsTracker's stages
    list during the TTC/risk backend integration, but the original chart
    here hardcoded a stage list from before that existed, so it never
    surfaced) alongside the pre-existing stages."""
    stages = ["weather", "detection", "tracking", "depth", "fusion", "bev", "risk"]
    rows = "".join(
        f'<div class="diag-row"><span>{s.capitalize()}</span><span>{output.metrics[f"{s}_latency_ms"]:.1f} ms</span></div>'
        for s in stages if f"{s}_latency_ms" in output.metrics
    )
    total_s = output.total_latency_ms / 1000.0
    fps = output.metrics.get("fps", 0.0)
    frame_count = output.metrics.get("frame_count", 0)
    return f"""
    <div class="hud-card">
        <div class="hud-card-title">PER-MODULE LATENCY</div>
        {rows}
        <div class="diag-row total"><span>Total</span><span>{total_s:.2f} s</span></div>
        <div style="font-size:0.65rem;color:#7a8896;margin-top:8px;">
            FPS: {fps:.2f} &nbsp;|&nbsp; Frames processed: {frame_count}
        </div>
    </div>"""


def render_object_table_df(output: PipelineOutput):
    """Rule 20: full detail table, relocated to an expander below the fold.
    Extends the original columns with Distance/TTC/Relative Speed/Risk --
    all sourced from output.risk.per_object / output.relative_velocities by
    track_id lookup, "N/A" where a given object has no valid entry yet
    (e.g. brand-new track, or risk unavailable this frame). Returns None if
    there are no detections, so the caller can show its own empty state.
    """
    import pandas as pd

    dets = output.detections.detections
    if not dets:
        return None

    risk_by_track = {o.track_id: o for o in output.risk.per_object} if output.risk is not None else {}
    vel_by_track = output.relative_velocities or {}

    rows = []
    for d in dets:
        obj_risk = risk_by_track.get(d.track_id)
        velocity = vel_by_track.get(d.track_id)
        rows.append({
            "Track ID": d.track_id if d.track_id is not None else "-",
            "Class": d.class_name,
            "Confidence": round(d.confidence, 3),
            "Distance": _format_distance(obj_risk.distance_m) if obj_risk else "N/A",
            "Relative Speed": _format_relative_speed(velocity),
            "TTC": _format_ttc(obj_risk.ttc_seconds) if obj_risk else "N/A",
            "Risk": obj_risk.risk_level if obj_risk else "N/A",
        })
    return pd.DataFrame(rows)
