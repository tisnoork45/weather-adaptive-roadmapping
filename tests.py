from __future__ import annotations

"""Test suite for the Weather Adaptive Road Mapping perception pipeline.

Run with:
    pytest tests.py -v

All tests run on CPU with synthetic frames only -- no GPU, dataset, or
pretrained weights required -- since every model wrapper gracefully falls
back to a classical CV / heuristic implementation when weights are absent.

Set TEST_DEVICE=cuda (or auto) to run the pipeline tests on the GPU, so the
real Depth Anything V2 / YOLO11x paths are exercised there too:
    TEST_DEVICE=cuda pytest tests.py -v
"""

import os

import cv2
import numpy as np
import pytest

from fusion import AdaptiveFusionEngine, ConfidenceEstimator, ModuleConfidence
from models import Detection, DetectionResult, YOLODetector
from pipeline import PerceptionPipeline, PipelineOutput
from tracking import ByteTracker, cxcyah_to_xyxy, iou_batch, xyxy_to_cxcyah
from utils import generate_synthetic_frame, load_config

TEST_DEVICE = os.environ.get("TEST_DEVICE", "cpu")

# =============================================================================
# Detection tests
# =============================================================================


@pytest.fixture(scope="module")
def detector() -> YOLODetector:
    # allow_download=False: this fixture exists to exercise the classical
    # fallback detector. Previously ultralytics silently auto-downloaded real
    # YOLO11x weights here, so the fallback was never actually tested.
    return YOLODetector(weights_path="weights/nonexistent_yolo11x.pt",
                         confidence_threshold=0.3, device_preference="cpu", allow_download=False)


def test_detector_uses_fallback_when_weights_missing(detector: YOLODetector) -> None:
    assert detector._use_fallback


def test_detector_initializes(detector: YOLODetector) -> None:
    assert detector is not None


def test_detection_on_synthetic_frame(detector: YOLODetector) -> None:
    frame = generate_synthetic_frame(width=640, height=480, seed=1)
    result = detector.infer(frame)
    assert isinstance(result, DetectionResult)
    assert result.inference_time_ms >= 0.0
    for det in result.detections:
        assert isinstance(det, Detection)
        x1, y1, x2, y2 = det.box
        assert x2 >= x1 and y2 >= y1
        assert 0.0 <= det.confidence <= 1.0


def test_detection_result_helpers(detector: YOLODetector) -> None:
    frame = generate_synthetic_frame(width=640, height=480, seed=2)
    result = detector.infer(frame)
    assert len(result.boxes()) == len(result) == len(result.labels()) == len(result.scores())


@pytest.mark.skipif(not os.path.exists("weights/depth_anything_v2_vits.pth"),
                    reason="Depth Anything V2 weights not present")
def test_depth_real_inference_runs_on_requested_cpu_device() -> None:
    """Regression: depth_anything_v2/dpt.py used to send its input to CUDA
    whenever CUDA existed, even with the model on CPU, so every frame failed
    and silently fell back. strict=True makes such a failure raise instead of
    being swallowed by infer()'s fallback."""
    from models import DEPTH_KIND_RELATIVE, DepthEstimator

    estimator = DepthEstimator(device_preference="cpu", input_size=154, strict=True)
    result = estimator.infer(generate_synthetic_frame(width=640, height=480, seed=3))
    assert result.kind == DEPTH_KIND_RELATIVE
    assert result.depth_map.shape == (480, 640)


def test_detection_center_and_dims() -> None:
    det = Detection(box=[10.0, 20.0, 50.0, 80.0], class_id=2, class_name="car", confidence=0.9)
    assert det.center == [30.0, 50.0]
    assert det.width == 40.0
    assert det.height == 60.0


def test_detector_handles_blank_frame(detector: YOLODetector) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert isinstance(detector.infer(blank), DetectionResult)


# =============================================================================
# Fusion / confidence tests
# =============================================================================

BASE_WEIGHTS = {"camera_detection": 0.9, "camera_lane": 0.85, "camera_depth": 0.8}
WEATHER_MODIFIERS = {
    "clear": {"camera_detection": 1.0, "camera_lane": 1.0, "camera_depth": 1.0},
    "fog": {"camera_detection": 0.55, "camera_lane": 0.5, "camera_depth": 0.45},
}


def _sample_confidences():
    return [
        ModuleConfidence("camera_detection", 0.8, 1.0, 0.7, 0.8),
        ModuleConfidence("camera_lane", 0.7, 1.0, 0.6, 0.7),
        ModuleConfidence("camera_depth", 0.6, 1.0, 0.5, 0.6),
    ]


def test_confidence_estimator_ranges() -> None:
    estimator = ConfidenceEstimator()
    det = estimator.estimate_detection_confidence(0.8, 1.0, 3)
    lane = estimator.estimate_lane_confidence(0.6, 0.5, 2)
    depth = estimator.estimate_depth_confidence(15.0, 0.7, 80.0)
    for result in (det, lane, depth):
        assert 0.0 <= result.final_confidence <= 1.0


def test_fusion_weights_valid_range() -> None:
    engine = AdaptiveFusionEngine(BASE_WEIGHTS, WEATHER_MODIFIERS, ema_alpha=0.5)
    result = engine.fuse("clear", _sample_confidences())
    assert set(result.sensor_weights.keys()) == set(BASE_WEIGHTS.keys())
    assert all(0.0 <= w <= 1.0 for w in result.sensor_weights.values())
    assert 0.0 <= result.overall_scene_confidence <= 1.0


def test_fusion_weights_decrease_in_fog() -> None:
    engine_clear = AdaptiveFusionEngine(BASE_WEIGHTS, WEATHER_MODIFIERS, ema_alpha=1.0)
    engine_fog = AdaptiveFusionEngine(BASE_WEIGHTS, WEATHER_MODIFIERS, ema_alpha=1.0)
    result_clear = engine_clear.fuse("clear", _sample_confidences())
    result_fog = engine_fog.fuse("fog", _sample_confidences())
    for key in BASE_WEIGHTS:
        assert result_fog.sensor_weights[key] <= result_clear.sensor_weights[key] + 1e-6


def test_normalize_weights_sums_to_one() -> None:
    normalized = AdaptiveFusionEngine.normalize_weights({"a": 0.2, "b": 0.3, "c": 0.5})
    assert abs(sum(normalized.values()) - 1.0) < 1e-6


def test_normalize_weights_handles_zero_total() -> None:
    normalized = AdaptiveFusionEngine.normalize_weights({"a": 0.0, "b": 0.0})
    assert abs(sum(normalized.values()) - 1.0) < 1e-6


# =============================================================================
# Tracking tests
# =============================================================================


def _make_detection(x1, y1, x2, y2, score=0.9, class_name="car") -> Detection:
    return Detection(box=[x1, y1, x2, y2], class_id=2, class_name=class_name, confidence=score)


def test_box_conversion_roundtrip() -> None:
    box = np.array([10.0, 20.0, 60.0, 100.0], dtype=np.float32)
    recovered = cxcyah_to_xyxy(xyxy_to_cxcyah(box))
    assert np.allclose(box, recovered, atol=1e-3)


def test_iou_batch_identical_and_disjoint() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert np.isclose(iou_batch(a, a)[0, 0], 1.0)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert np.isclose(iou_batch(a, b)[0, 0], 0.0)


def test_tracker_assigns_new_ids() -> None:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
    tracked = tracker.update([_make_detection(10, 10, 50, 50), _make_detection(200, 200, 260, 260)])
    ids = [d.track_id for d in tracked]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == 2


def test_tracker_maintains_id_across_frames() -> None:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.5, frame_rate=30)
    first_id = tracker.update([_make_detection(10, 10, 60, 60, score=0.9)])[0].track_id
    second_id = tracker.update([_make_detection(14, 12, 64, 62, score=0.9)])[0].track_id
    assert first_id is not None and second_id == first_id


def test_tracker_handles_empty_detections() -> None:
    assert ByteTracker().update([]) == []


def test_tracker_low_confidence_recovery() -> None:
    tracker = ByteTracker(track_thresh=0.6, track_buffer=30, match_thresh=0.5, frame_rate=30)
    original_id = tracker.update([_make_detection(10, 10, 60, 60, score=0.9)])[0].track_id
    tracked2 = tracker.update([_make_detection(12, 11, 62, 61, score=0.3)])
    assert len(tracked2) == 1
    assert tracked2[0].track_id == original_id


# =============================================================================
# Full pipeline integration tests
# =============================================================================


@pytest.fixture(scope="module")
def pipeline() -> PerceptionPipeline:
    config = load_config("config.yaml")
    config.device.preference = TEST_DEVICE
    return PerceptionPipeline(config)


def test_pipeline_initializes(pipeline: PerceptionPipeline) -> None:
    assert pipeline.detector and pipeline.depth_estimator
    assert pipeline.weather_classifier and pipeline.tracker and pipeline.fusion_engine and pipeline.bev_mapper


def test_pipeline_process_returns_valid_output(pipeline: PerceptionPipeline) -> None:
    frame = generate_synthetic_frame(width=640, height=480, seed=42)
    output = pipeline.process(frame)

    assert isinstance(output, PipelineOutput)
    assert output.frame.shape == frame.shape
    assert output.total_latency_ms >= 0.0
    assert output.weather.label in pipeline.weather_classifier.classes
    assert 0.0 <= output.weather.confidence <= 1.0
    for det in output.detections.detections:
        assert len(det.box) == 4
    assert output.depth.depth_map.shape[:2] == frame.shape[:2]
    assert 0.0 <= output.fusion.overall_scene_confidence <= 1.0
    assert set(output.fusion.sensor_weights.keys()) == {"camera_detection", "camera_depth"}
    if output.bev is not None:
        assert output.bev.bev_image.ndim == 3


def test_pipeline_multiple_frames(pipeline: PerceptionPipeline) -> None:
    outputs = [pipeline.process(generate_synthetic_frame(640, 480, seed=i)) for i in range(5)]
    assert len(outputs) == 5
    assert outputs[-1].metrics["fps"] >= 0.0


def test_pipeline_handles_all_black_frame(pipeline: PerceptionPipeline) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = pipeline.process(frame)
    assert isinstance(output, PipelineOutput)
    assert output.total_latency_ms >= 0.0
"""NEW test functions to APPEND to the existing tests.py (do not replace the
file -- add these functions alongside the existing ones; no existing test
is modified).
"""


from dataclasses import dataclass

from relative_velocity import RelativeVelocityEstimator
from risk_engine import RiskEngine
from ttc_calculator import compute_ttc, compute_ttc_for_objects


@dataclass
class _FakeBEVObject:
    track_id: int
    x_m: float
    z_m: float
    class_name: str = "car"
    confidence: float = 0.9


# =============================================================================
# relative_velocity.py
# =============================================================================


def test_relative_velocity_recovers_known_closing_speed() -> None:
    estimator = RelativeVelocityEstimator(history_length=5, max_gap_seconds=2.0, ema_alpha=1.0)
    t, z = 0.0, 50.0
    closing_speed = 5.0
    for dt in (0.30, 0.45, 0.28, 0.50, 0.33):  # variable real dt, not fixed frame rate
        t += dt
        z -= closing_speed * dt
        result = estimator.update([_FakeBEVObject(1, 2.0, z)], timestamp=t)
    vel = result[1]
    assert vel.valid
    assert abs(vel.vx_m_s - 0.0) < 0.05
    assert abs(vel.vz_m_s - (-closing_speed)) < 0.05


def test_relative_velocity_large_gap_is_invalid() -> None:
    estimator = RelativeVelocityEstimator(history_length=5, max_gap_seconds=2.0)
    estimator.update([_FakeBEVObject(1, 2.0, 50.0)], timestamp=0.0)
    estimator.update([_FakeBEVObject(1, 2.0, 45.0)], timestamp=0.5)
    result = estimator.update([_FakeBEVObject(1, 2.0, 44.0)], timestamp=5.5)  # large stall
    assert result[1].valid is False


def test_relative_velocity_prunes_lost_tracks() -> None:
    estimator = RelativeVelocityEstimator()
    estimator.update([_FakeBEVObject(1, 0, 0), _FakeBEVObject(2, 0, 0)], timestamp=0.0)
    assert set(estimator._history.keys()) == {1, 2}
    estimator.update([_FakeBEVObject(1, 0, 0)], timestamp=0.1)
    assert set(estimator._history.keys()) == {1}


def test_relative_velocity_needs_two_points() -> None:
    estimator = RelativeVelocityEstimator()
    result = estimator.update([_FakeBEVObject(1, 0, 10)], timestamp=0.0)
    assert result[1].valid is False


# =============================================================================
# ttc_calculator.py
# =============================================================================


def test_ttc_basic_math() -> None:
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=5.0) == 10.0
    assert abs(compute_ttc(distance_m=3.0, closing_speed_m_s=6.0) - 0.5) < 1e-9


def test_ttc_none_when_not_closing() -> None:
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=-2.0) is None
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=0.01) is None


def test_ttc_capped_at_max() -> None:
    assert compute_ttc(distance_m=1000.0, closing_speed_m_s=0.2, max_ttc_s=60.0) == 60.0


def test_ttc_for_objects_skips_invalid_or_missing_velocity() -> None:
    from relative_velocity import RelativeVelocity

    objects = [
        _FakeBEVObject(track_id=1, x_m=0.0, z_m=20.0),
        _FakeBEVObject(track_id=2, x_m=0.0, z_m=30.0),  # no velocity entry
        _FakeBEVObject(track_id=3, x_m=0.0, z_m=15.0),  # velocity present but invalid
    ]
    velocities = {
        1: RelativeVelocity(track_id=1, vx_m_s=0.0, vz_m_s=-8.0, speed_m_s=8.0, valid=True),
        3: RelativeVelocity(track_id=3, vx_m_s=0.0, vz_m_s=-8.0, speed_m_s=8.0, valid=False),
    }
    results = compute_ttc_for_objects(objects, velocities)
    assert len(results) == 1
    assert results[0].track_id == 1
    assert abs(results[0].ttc_seconds - 2.5) < 1e-9


# =============================================================================
# risk_engine.py
# =============================================================================

_THRESHOLDS = {"CRITICAL": 1.5, "WARNING": 3.0, "CAUTION": 5.0}
_WEATHER_MULT = {"clear": 1.0, "fog": 1.6, "rain": 1.3, "night": 1.25, "snow": 1.5, "overcast": 1.05}


def _make_engine() -> RiskEngine:
    return RiskEngine(_THRESHOLDS, _WEATHER_MULT, min_safe_distance_m=5.0)


def test_risk_engine_ttc_boundaries() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    cases = [(1.0, "CRITICAL"), (2.0, "WARNING"), (4.0, "CAUTION"), (10.0, "SAFE")]
    for ttc, expected in cases:
        r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=ttc)
        result = engine.assess("clear", [r])
        assert result.per_object[0].risk_level == expected


def test_risk_engine_weather_adaptive_thresholds_genuinely_change_risk() -> None:
    """The core weather-adaptive requirement: identical TTC must classify
    differently depending on weather, not just carry a different number
    somewhere unused."""
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=2.0)
    clear_level = engine.assess("clear", [r]).per_object[0].risk_level
    fog_level = engine.assess("fog", [r]).per_object[0].risk_level
    assert clear_level == "WARNING"
    assert fog_level == "CRITICAL"
    assert clear_level != fog_level


def test_risk_engine_distance_floor_catches_stationary_close_object() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=2, class_name="car", distance_m=2.0, closing_speed_m_s=-1.0, ttc_seconds=None)
    result = engine.assess("clear", [r])
    assert result.per_object[0].risk_level == "WARNING"


def test_risk_engine_overall_is_worst_object() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r_safe = TTCResult(track_id=3, class_name="person", distance_m=40.0, closing_speed_m_s=1.0, ttc_seconds=40.0)
    r_critical = TTCResult(track_id=4, class_name="car", distance_m=5.0, closing_speed_m_s=5.0, ttc_seconds=1.0)
    result = engine.assess("clear", [r_safe, r_critical])
    assert result.overall_risk_level == "CRITICAL"


def test_risk_engine_no_objects_is_safe() -> None:
    engine = _make_engine()
    result = engine.assess("clear", [])
    assert result.overall_risk_level == "SAFE"
    assert result.per_object == []


def test_risk_engine_unrecognized_weather_defaults_to_no_adjustment() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=2.0)
    result = engine.assess("some_unknown_weather_label", [r])
    assert result.per_object[0].risk_level == "WARNING"  # same as clear (multiplier defaults to 1.0)


# =============================================================================
# Pipeline-level TTC/velocity/risk integration
#
# These tests exercise PerceptionPipeline.process()'s actual wiring of
# relative_velocity.py -> ttc_calculator.py -> risk_engine.py (added after
# the standalone-module tests above, which only ever construct those three
# classes directly and never go through the pipeline). They specifically
# guard against the bug this integration was built to fix: passing no
# timestamp (or a wall-clock timestamp on a slow CPU pipeline) made every
# relative-velocity estimate permanently invalid, because the gap between
# samples was measured in real processing seconds (~3-4s/frame) rather than
# video-timeline seconds (~0.04s/frame at ~24fps).
# =============================================================================

from mapping import BEVObject  # noqa: E402


@pytest.fixture(scope="module")
def pipeline_no_bev() -> PerceptionPipeline:
    """A second pipeline instance with BEV mapping disabled, to prove the
    velocity/TTC/risk stage degrades to empty/None instead of crashing when
    its only data source (BEV object positions) never exists."""
    config = load_config("config.yaml")
    config.device.preference = TEST_DEVICE
    config.pipeline.run_bev_mapping = False
    return PerceptionPipeline(config)


def test_pipeline_output_carries_risk_fields(pipeline: PerceptionPipeline) -> None:
    """Basic plumbing check: the fields exist, are the right types, and
    processing several frames with realistic video-timeline-spaced
    timestamps doesn't crash."""
    for i in range(3):
        frame = generate_synthetic_frame(640, 480, seed=100 + i)
        timestamp = i / 24.0  # ~24fps video-timeline spacing
        output = pipeline.process(frame, timestamp=timestamp)
        assert isinstance(output.relative_velocities, dict)
        assert output.risk is None or output.risk.overall_risk_level in (
            "SAFE", "CAUTION", "WARNING", "CRITICAL",
        )


def test_pipeline_process_without_timestamp_still_works(pipeline: PerceptionPipeline) -> None:
    """Backward compatibility: existing callers (app.py, carla_sim.py) call
    pipeline.process(frame) with no timestamp argument at all. This must
    keep working exactly as before (falls back to wall-clock internally)."""
    frame = generate_synthetic_frame(640, 480, seed=7)
    output = pipeline.process(frame)
    assert isinstance(output, PipelineOutput)
    assert isinstance(output.relative_velocities, dict)


def test_pipeline_zero_detections_is_safe_no_crash(pipeline: PerceptionPipeline) -> None:
    """An all-black frame produces zero detections from the classical
    fallback detector (no edges to find) -- risk must not crash and must
    not report anything worse than SAFE when there is nothing to assess."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = pipeline.process(frame, timestamp=0.0)
    assert len(output.detections) == 0
    if output.risk is not None:
        assert output.risk.overall_risk_level == "SAFE"
        assert output.risk.per_object == []


def test_pipeline_bev_disabled_yields_empty_risk_not_crash(pipeline_no_bev: PerceptionPipeline) -> None:
    """If BEV mapping is off, there are no (x_m, z_m) positions to compute
    velocity/TTC/risk from -- this must degrade gracefully, not crash."""
    frame = generate_synthetic_frame(640, 480, seed=9)
    output = pipeline_no_bev.process(frame, timestamp=0.0)
    assert output.bev is None
    assert output.relative_velocities == {}
    assert output.risk is None


def test_relative_velocity_becomes_valid_with_video_timeline_spacing(pipeline: PerceptionPipeline) -> None:
    """Reproduces and proves the fix for the actual reported bug: at
    video-timeline spacing (~1/24s apart, not several wall-clock seconds
    apart), the SAME track_id observed across consecutive frames produces a
    valid velocity estimate well within max_gap_seconds -- using the
    pipeline's OWN config-loaded velocity_estimator instance, not a fresh
    test-only one, so this catches real config-wiring bugs too."""
    estimator = pipeline.velocity_estimator
    fps = 23.997  # data/sample_videos/road.mp4's actual reported fps
    z = 30.0
    closing_speed = 4.0
    result = None
    for frame_number in range(6):
        timestamp = frame_number / fps
        z -= closing_speed / fps
        result = estimator.update([BEVObject(track_id=999001, class_name="car",
                                              x_m=0.0, z_m=z, confidence=0.9)], timestamp=timestamp)
    assert result[999001].valid is True
    assert result[999001].vz_m_s < 0  # negative = closing, matching the fabricated approach


def test_end_to_end_ttc_risk_escalation_with_pipeline_config(pipeline: PerceptionPipeline) -> None:
    """Deterministic, synthetic proof of the full relative-velocity -> TTC
    -> weather-adjusted-risk path, using the pipeline's real config-loaded
    velocity_estimator and risk_engine (not fresh objects), since the real
    road.mp4 clip does not reliably contain an actively-closing object and
    it would be dishonest to claim it demonstrated one. Drives a fabricated
    object closing from 40m to a few meters at a constant, known speed and
    confirms the risk level actually escalates as TTC and distance shrink,
    and that identical kinematics classify worse in fog than in clear
    weather (the project's core weather-adaptive claim)."""
    from relative_velocity import RelativeVelocity
    from ttc_calculator import compute_ttc_for_objects

    estimator = RelativeVelocityEstimator(
        history_length=pipeline.velocity_estimator.history_length,
        max_gap_seconds=pipeline.velocity_estimator.max_gap_seconds,
        ema_alpha=1.0,  # no smoothing, so the fabricated speed is recovered exactly
    )
    fps = 24.0
    closing_speed = 12.0  # m/s -- fast enough to walk through every risk level
    z = 40.0
    track_id = 999002

    levels_seen = []
    for frame_number in range(60):
        timestamp = frame_number / fps
        z = max(z - closing_speed / fps, 0.5)
        bev_objects = [BEVObject(track_id=track_id, class_name="car", x_m=0.0, z_m=z, confidence=0.9)]
        velocities = estimator.update(bev_objects, timestamp=timestamp)
        ttc_results = compute_ttc_for_objects(bev_objects, velocities)
        risk = pipeline.risk_engine.assess("clear", ttc_results)
        if risk.per_object:
            levels_seen.append(risk.per_object[0].risk_level)

    # Risk must have escalated beyond SAFE at some point as the object closed.
    assert any(level != "SAFE" for level in levels_seen)
    assert "CRITICAL" in levels_seen

    # Same kinematics, worse weather -> same-or-worse risk level (the
    # weather multiplier must actually change the classification, not just
    # be carried around unused).
    mid_ttc_results = compute_ttc_for_objects(
        [BEVObject(track_id=track_id, class_name="car", x_m=0.0, z_m=20.0, confidence=0.9)],
        {track_id: RelativeVelocity(track_id=track_id, vx_m_s=0.0, vz_m_s=-closing_speed,
                                     speed_m_s=closing_speed, valid=True)},
    )
    clear_level = pipeline.risk_engine.assess("clear", mid_ttc_results).per_object[0].risk_level
    fog_level = pipeline.risk_engine.assess("fog", mid_ttc_results).per_object[0].risk_level
    order = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}
    assert order[fog_level] >= order[clear_level]
    assert fog_level != clear_level


# =============================================================================
# Weather classifier: preprocessing parity + training pipeline
# =============================================================================


def _capturing_weather_classifier(tmp_path):
    """A WeatherClassifier whose 'model' records the exact tensor it receives."""
    import torch
    from models import WeatherClassifier

    classifier = WeatherClassifier(weights_path=str(tmp_path / "missing.pth"), device_preference="cpu")
    captured = {}

    class _Capture(torch.nn.Module):
        def forward(self, x):
            captured["input"] = x.detach().cpu().clone()
            return torch.zeros(x.shape[0], len(classifier.classes))

    classifier.model, classifier._use_fallback = _Capture(), False
    return classifier, captured


def test_weather_training_and_inference_preprocessing_are_identical(tmp_path) -> None:
    """Regression: training (PIL + torchvision Resize) and WeatherClassifier
    (cv2.resize INTER_LINEAR) used to preprocess differently, costing ~1.6
    accuracy points in the deployed pipeline. Both must now produce the exact
    same tensor from the same image file."""
    import torch
    from models import WeatherEvalTransform, load_weather_rgb

    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), generate_synthetic_frame(width=1280, height=720, seed=7))
    classifier, captured = _capturing_weather_classifier(tmp_path)

    classifier._neural_infer(cv2.imread(str(path)))  # what the live pipeline does with a BGR frame
    training_tensor = WeatherEvalTransform(classifier.input_size, classifier.mean, classifier.std)(load_weather_rgb(path))

    assert captured["input"].shape == (1, 3, classifier.input_size, classifier.input_size)
    assert torch.equal(captured["input"][0], training_tensor)


def test_weather_train_transform_output_matches_eval_shape() -> None:
    from models import WeatherTrainTransform

    rgb = cv2.cvtColor(generate_synthetic_frame(width=1280, height=720, seed=8), cv2.COLOR_BGR2RGB)
    out = WeatherTrainTransform(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(rgb)
    assert out.shape == (3, 224, 224) and out.dtype.is_floating_point


def test_train_weather_model_smoke(tmp_path) -> None:
    """End-to-end: tiny synthetic dataset -> checkpoint -> WeatherClassifier
    loads it without falling back; evaluate_test=False never touches test/."""
    import json
    from models import DEFAULT_WEATHER_CLASSES, WeatherClassifier, train_weather_model

    data = tmp_path / "ds"
    for split, n in (("train", 2), ("val", 1), ("test", 1)):
        for c, cls in enumerate(DEFAULT_WEATHER_CLASSES):
            (data / split / cls).mkdir(parents=True)
            for i in range(n):
                name = f"dawn_{cls}-{i}.jpg" if cls == "fog" and i == 0 else f"{cls}-{i}.jpg"
                frame = generate_synthetic_frame(width=640, height=480, seed=c * 10 + i)
                cv2.imwrite(str(data / split / cls / name), frame)

    ckpt, run_dir = tmp_path / "w.pth", tmp_path / "run"
    summary = train_weather_model(
        data_dir=str(data), output_path=str(ckpt), epochs=1, batch_size=4, num_workers=0,
        pretrained=False, evaluate_test=False, run_dir=str(run_dir),
    )
    report = json.loads((run_dir / "report.json").read_text())
    assert summary["best_epoch"] == 1 and "test" not in report and "test_accuracy" not in summary
    assert set(report["validation"]["by_source"]) == {"bdd", "dawn"}

    classifier = WeatherClassifier(weights_path=str(ckpt), device_preference="cpu")
    assert not classifier._use_fallback
    assert classifier.infer(generate_synthetic_frame(width=640, height=480, seed=1)).label in DEFAULT_WEATHER_CLASSES


# =============================================================================
# Weather temporal smoothing + BEV intrinsics scaling
# =============================================================================


def _weather(label: str, p: float = 0.9):
    from models import DEFAULT_WEATHER_CLASSES, WeatherResult

    rest = (1.0 - p) / (len(DEFAULT_WEATHER_CLASSES) - 1)
    probs = {c: (p if c == label else rest) for c in DEFAULT_WEATHER_CLASSES}
    return WeatherResult(label=label, confidence=p, probabilities=probs)


def test_weather_smoother_suppresses_short_burst_and_follows_sustained_change() -> None:
    from models import WeatherTemporalSmoother

    s, fps, t = WeatherTemporalSmoother(time_constant_s=1.0), 24.0, 0.0
    for _ in range(48):                      # 2 s of clear
        out = s.update(_weather("clear"), t); t += 1 / fps
    for _ in range(12):                      # 0.5 s glare burst read as snow
        out = s.update(_weather("snow", 0.8), t); t += 1 / fps
        assert out.label == "clear"
    for _ in range(72):                      # 3 s of real rain
        out = s.update(_weather("rain"), t); t += 1 / fps
    assert out.label == "rain"
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-6


def test_weather_smoother_resets_on_backwards_time_and_can_be_disabled() -> None:
    from models import WeatherTemporalSmoother

    s = WeatherTemporalSmoother(time_constant_s=1.0)
    for i in range(48):
        s.update(_weather("clear"), i / 24)
    assert s.update(_weather("night"), 0.0).label == "night"     # new video restarts at t=0
    off = WeatherTemporalSmoother(time_constant_s=0.0)
    off.update(_weather("clear"), 0.0)
    assert off.update(_weather("fog"), 0.04).label == "fog"


def test_bev_intrinsics_scale_with_frame_size() -> None:
    from mapping import BEVMapper

    def x_at(width: int, height: int, u_frac: float) -> float:
        bev = BEVMapper(fx=1000, fy=1000, cx=640, cy=360, reference_width=1280, reference_height=720)
        depth = np.full((height, width), 20.0, dtype=np.float32)
        u = u_frac * width
        det = Detection(box=[u - 10, 0.5 * height, u + 10, 0.8 * height], class_id=2, class_name="car",
                        confidence=0.9, track_id=1)
        return bev.build_bev(np.zeros((height, width, 3), np.uint8), depth, [det]).objects[0].x_m

    assert abs(x_at(1024, 576, 0.5)) < 0.05                     # centred object is straight ahead
    assert abs(x_at(1024, 576, 0.75) - x_at(1280, 720, 0.75)) < 0.05   # same metres at any resolution
    unscaled = BEVMapper(fx=1000, fy=1000, cx=640, cy=360)      # no reference size: unchanged behaviour
    unscaled._use_frame_size(1024, 576)
    assert (unscaled.fx, unscaled.cx) == (1000, 640)


def test_pipeline_output_has_smoothed_and_raw_weather(pipeline: PerceptionPipeline) -> None:
    out = pipeline.process(generate_synthetic_frame(width=640, height=480, seed=3), timestamp=100.0)
    assert out.weather_raw is not None
    assert out.weather.label in out.weather.probabilities


# =============================================================================
# Tracker persistence: lost tracks are re-associated (ByteTrack track pool)
# =============================================================================


def _det(box, conf=0.9):
    return Detection(box=list(box), class_id=2, class_name="car", confidence=conf)


def _moving_box(i, x0=100.0, speed=4.0):
    return (x0 + speed * i, 200.0, x0 + 80 + speed * i, 260.0)


def test_tracker_recovers_same_id_after_missed_detections() -> None:
    """Regression: lost tracks were never matched again, so a 1-frame detection
    dropout permanently ended a track (23 of 42 IDs on road.mp4 were such fragments)."""
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    ids = []
    for i in range(20):
        if 8 <= i < 11:                      # detector misses the object for 3 frames
            tracker.update([])
            continue
        ids.append(tracker.update([_det(_moving_box(i))])[0].track_id)
    assert len(set(ids)) == 1


def test_tracker_does_not_give_a_distant_new_object_a_lost_id() -> None:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    first = [tracker.update([_det(_moving_box(i))])[0].track_id for i in range(5)][-1]
    tracker.update([])
    far = tracker.update([_det((900.0, 500.0, 980.0, 560.0))])[0].track_id
    assert far != first


def test_tracker_lost_track_expires_after_buffer() -> None:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=5, match_thresh=0.8, frame_rate=30)
    first = [tracker.update([_det((100.0, 200.0, 180.0, 260.0))])[0].track_id for _ in range(5)][-1]
    for _ in range(7):                       # longer than the 5-frame buffer
        tracker.update([])
    assert tracker.update([_det((100.0, 200.0, 180.0, 260.0))])[0].track_id != first


def test_tracker_low_confidence_detection_does_not_revive_lost_track() -> None:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    for _ in range(5):
        tracker.update([_det((100.0, 200.0, 180.0, 260.0))])
    tracker.update([])
    assert tracker.update([_det((100.0, 200.0, 180.0, 260.0), conf=0.3)])[0].track_id is None


def test_untracked_detection_is_drawn_without_an_id(monkeypatch) -> None:
    """Untracked detections used to be labelled with their list index ("ID:1"),
    which looked like -- and could collide with -- a real track ID."""
    import utils

    texts = []
    real_put_text = utils.cv2.putText
    monkeypatch.setattr(utils.cv2, "putText", lambda img, text, *a, **k: (texts.append(text), real_put_text(img, text, *a, **k))[1])
    utils.draw_detections(np.zeros((200, 300, 3), np.uint8), [[10, 40, 80, 90], [120, 40, 200, 90]],
                          ["car", "car"], [0.9, 0.4], [7, None])
    assert texts == ["ID:7 car 0.90", "car 0.40"]
