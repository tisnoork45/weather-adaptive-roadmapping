"""End-to-end weather-adaptive perception pipeline + dashboard render helpers.

NEW in this version: relative velocity, TTC, and risk assessment. After BEV
mapping produces per-object (x_m, z_m) positions (unchanged, from the
existing mapping.py), this pipeline now:
  1. Feeds those positions into `relative_velocity.py` (new) to get a
     closing/lateral velocity per tracked object.
  2. Feeds distance + velocity into `ttc_calculator.py` (new) to get a
     time-to-collision per object.
  3. Feeds weather + TTC into `risk_engine.py` (new) to get a weather-
     adjusted risk level per object and an overall frame-level risk.

This is purely additive: every existing stage (weather, detection,
tracking, lane, depth, confidence estimation, adaptive fusion, BEV) is
UNCHANGED. If BEV mapping is disabled in config.yaml, risk/velocity are
simply not computed for that frame (`relative_velocities={}`, `risk=None`)
rather than erroring, since they depend on BEV's (x_m, z_m) output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fusion import AdaptiveFusionEngine, ConfidenceEstimator, FusionResult, ModuleConfidence
from mapping import BEVMapper, BEVResult
from models import (
    DepthEstimator, DepthResult, DetectionResult, LaneDetector, LaneResult,
    WeatherClassifier, WeatherResult, YOLODetector,
)
from relative_velocity import RelativeVelocity, RelativeVelocityEstimator
from risk_engine import RiskAssessment, RiskEngine
from tracking import ByteTracker
from ttc_calculator import compute_ttc_for_objects
from utils import ConfigDict, MetricsTracker, get_logger

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    """Aggregated output of one full pipeline pass over a single frame."""

    frame: np.ndarray
    weather: WeatherResult
    detections: DetectionResult
    lanes: LaneResult
    depth: DepthResult
    fusion: FusionResult
    bev: Optional[BEVResult]
    object_distances: List[Optional[float]] = field(default_factory=list)
    relative_velocities: Dict[int, RelativeVelocity] = field(default_factory=dict)
    risk: Optional[RiskAssessment] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


class PerceptionPipeline:
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
        self.lane_detector = LaneDetector(
            weights_path=config.lane_detection.weights_path,
            backbone=config.lane_detection.backbone,
            input_width=config.lane_detection.input_width,
            input_height=config.lane_detection.input_height,
            num_lanes=config.lane_detection.num_lanes,
            num_row_anchors=config.lane_detection.num_row_anchors,
            num_col_anchors=config.lane_detection.num_col_anchors,
            confidence_threshold=config.lane_detection.confidence_threshold,
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
            lane_signal_weights=config.confidence_estimation.lane_signal_weights,
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

        # --- NEW: relative velocity + risk engine ---------------------------------
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
            stages=["weather", "detection", "tracking", "lane", "depth", "fusion", "bev", "risk"]
        )
        self._frame_index = 0
        logger.info("All perception pipeline modules initialized successfully.")

    def process(self, frame: np.ndarray) -> PipelineOutput:
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

        lane_result = self.lane_detector.infer(frame) if self.pipeline_cfg.run_lane_detection else LaneResult()
        self.metrics.record_stage("lane", lane_result.inference_time_ms)

        if self.pipeline_cfg.run_depth_estimation:
            depth_result = self.depth_estimator.infer(frame)
        else:
            depth_result = DepthResult(depth_map=np.zeros(frame.shape[:2], dtype=np.float32))
        self.metrics.record_stage("depth", depth_result.inference_time_ms)

        fusion_start = time.perf_counter()
        module_confidences = self._estimate_confidences(weather_result, detection_result, lane_result, depth_result)
        fusion_result = self.fusion_engine.fuse(weather_result.label, module_confidences)
        self.metrics.record_stage("fusion", (time.perf_counter() - fusion_start) * 1000.0)

        object_distances = self.bev_mapper.get_object_distances(
            detection_result.detections, depth_result.depth_map
        )

        bev_result = None
        if self.pipeline_cfg.run_bev_mapping:
            bev_result = self.bev_mapper.build_bev(
                frame, depth_result.depth_map, detection_result.detections, lanes_pixel=lane_result.lanes,
            )
            self.metrics.record_stage("bev", bev_result.inference_time_ms)

        # --- NEW: relative velocity -> TTC -> weather-adjusted risk -------------------
        # Depends on BEV's (x_m, z_m) object positions -- if BEV mapping is
        # disabled this frame, there is nothing to compute risk from, so
        # this stage is skipped (not an error) and risk/velocities are empty.
        risk_start = time.perf_counter()
        relative_velocities: Dict[int, RelativeVelocity] = {}
        risk_result: Optional[RiskAssessment] = None
        if bev_result is not None:
            relative_velocities = self.velocity_estimator.update(bev_result.objects)
            ttc_results = compute_ttc_for_objects(bev_result.objects, relative_velocities)
            risk_result = self.risk_engine.assess(weather_result.label, ttc_results)
        self.metrics.record_stage("risk", (time.perf_counter() - risk_start) * 1000.0)

        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
        self.metrics.record_frame(object_count=len(detection_result))

        return PipelineOutput(
            frame=frame, weather=weather_result, detections=detection_result, lanes=lane_result,
            depth=depth_result, fusion=fusion_result, bev=bev_result, object_distances=object_distances,
            relative_velocities=relative_velocities, risk=risk_result,
            metrics=self.metrics.summary(), total_latency_ms=total_latency_ms,
        )

    def _estimate_confidences(self, weather_result, detection_result, lane_result, depth_result) -> List[ModuleConfidence]:
        mean_det_score = float(np.mean(detection_result.scores())) if len(detection_result) else 0.0
        detection_conf = self.confidence_estimator.estimate_detection_confidence(
            mean_detection_score=mean_det_score,
            weather_modifier=self.fusion_engine.get_weather_modifier(weather_result.label, "camera_detection"),
            num_detections=len(detection_result),
        )
        mean_lane_score = float(np.mean(lane_result.confidences)) if lane_result.confidences else 0.0
        lane_conf = self.confidence_estimator.estimate_lane_confidence(
            mean_lane_score=mean_lane_score,
            weather_modifier=self.fusion_engine.get_weather_modifier(weather_result.label, "camera_lane"),
            num_lanes=len(lane_result),
        )
        depth_std = float(np.std(depth_result.depth_map)) if depth_result.depth_map.size else 0.0
        depth_conf = self.confidence_estimator.estimate_depth_confidence(
            depth_map_std=depth_std,
            weather_modifier=self.fusion_engine.get_weather_modifier(weather_result.label, "camera_depth"),
            max_depth_m=self.max_depth_m,
        )
        return [detection_conf, lane_conf, depth_conf]


# =============================================================================
# Rendering helpers (shared by run.py and app.py)
#
# CORRECTION: an earlier patch to this file accidentally dropped several of
# these functions (render_header, render_metric_row, render_weather_panel,
# render_sensor_weights_panel, render_depth_panel, render_bev_panel,
# render_latency_breakdown, render_object_table) that app.py imports and
# depends on. If you applied that patch, app.py would fail with an
# ImportError. This version restores all of them, in full, alongside the
# fusion-alpha wiring and the new risk-panel wiring -- verify after applying
# by actually starting the dashboard (see the testing section).
# =============================================================================


def render_header(st, title: str) -> None:
    st.markdown(
        f"""<div style="padding: 0.75rem 0 0.25rem 0;">
            <h1 style="margin-bottom:0;">{title}</h1>
            <p style="color:#888;margin-top:0;">Weather-adaptive multi-sensor perception for autonomous vehicles</p>
        </div>""",
        unsafe_allow_html=True,
    )


def render_metric_row(st, output: PipelineOutput) -> None:
    cols = st.columns(6)
    cols[0].metric("FPS", f"{output.metrics.get('fps', 0.0):.1f}")
    cols[1].metric("Total Latency", f"{output.total_latency_ms:.1f} ms")
    cols[2].metric("Objects Detected", f"{len(output.detections)}")
    cols[3].metric("Weather", output.weather.label.capitalize())
    cols[4].metric("Scene Confidence", f"{output.fusion.overall_scene_confidence:.2f}")
    cols[5].metric("Risk", output.risk.overall_risk_level if output.risk else "N/A")


def render_camera_feed(output: PipelineOutput) -> np.ndarray:
    """Return an annotated BGR frame: detections + lanes drawn with opacity
    driven by the current weather-adaptive sensor weights, real depth-
    derived distance text on each box, and the risk panel overlaid in the
    corner (NEW).
    """
    from utils import draw_detections, draw_lanes, draw_risk_panel

    annotated = output.frame.copy()
    dets = output.detections
    track_ids = [d.track_id if d.track_id is not None else i for i, d in enumerate(dets.detections)]

    detection_alpha = output.fusion.sensor_weights.get("camera_detection", 1.0)
    lane_alpha = output.fusion.sensor_weights.get("camera_lane", 1.0)

    annotated = draw_detections(
        annotated, dets.boxes(), dets.labels(), dets.scores(), track_ids,
        distances=output.object_distances, alpha=detection_alpha,
    )
    annotated = draw_lanes(annotated, output.lanes.lanes, alpha=lane_alpha)
    annotated = draw_risk_panel(annotated, output.risk)
    return annotated


def render_weather_panel(st, output: PipelineOutput) -> None:
    import pandas as pd

    st.subheader("Weather Classification")
    probs = output.weather.probabilities
    if probs:
        df = pd.DataFrame({"condition": list(probs.keys()), "probability": list(probs.values())}) \
            .sort_values("probability", ascending=False)
        st.bar_chart(df.set_index("condition"))
    st.caption(
        f"Predicted: **{output.weather.label}** (confidence {output.weather.confidence:.2f}, "
        f"{output.weather.inference_time_ms:.1f} ms)"
    )


def render_sensor_weights_panel(st, output: PipelineOutput) -> None:
    import pandas as pd

    st.subheader("Adaptive Sensor Weights")
    weights = output.fusion.sensor_weights
    if weights:
        df = pd.DataFrame({"module": list(weights.keys()), "weight": list(weights.values())})
        st.bar_chart(df.set_index("module"))
    conf = output.fusion.module_confidences
    st.caption(" | ".join(f"{k}: {v:.2f}" for k, v in conf.items()))


def render_depth_panel(output: PipelineOutput) -> np.ndarray:
    from utils import depth_to_colormap

    return depth_to_colormap(output.depth.depth_map)


def render_bev_panel(output: PipelineOutput) -> np.ndarray:
    if output.bev is not None:
        return output.bev.bev_image
    return np.zeros((400, 300, 3), dtype=np.uint8)


def render_latency_breakdown(st, output: PipelineOutput) -> None:
    import pandas as pd

    st.subheader("Per-Module Latency (ms)")
    stages = ["weather", "detection", "tracking", "lane", "depth", "fusion", "bev", "risk"]
    rows = [{"stage": s, "latency_ms": output.metrics[f"{s}_latency_ms"]}
            for s in stages if f"{s}_latency_ms" in output.metrics]
    if rows:
        st.bar_chart(pd.DataFrame(rows).set_index("stage"))


def render_object_table(st, output: PipelineOutput) -> None:
    import pandas as pd

    st.subheader("Detected Objects")
    dets = output.detections.detections
    if not dets:
        st.info("No objects currently detected.")
        return
    risk_by_track = {}
    if output.risk is not None:
        risk_by_track = {o.track_id: o for o in output.risk.per_object}
    rows = []
    for d in dets:
        obj_risk = risk_by_track.get(d.track_id)
        rows.append({
            "Track ID": d.track_id if d.track_id is not None else "-",
            "Class": d.class_name,
            "Confidence": round(d.confidence, 3),
            "Box (x1,y1,x2,y2)": [round(v, 1) for v in d.box],
            "TTC (s)": round(obj_risk.ttc_seconds, 1) if obj_risk and obj_risk.ttc_seconds is not None else "-",
            "Risk": obj_risk.risk_level if obj_risk else "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
