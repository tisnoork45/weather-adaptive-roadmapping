from __future__ import annotations

"""Tests for depth units, masks, calibration, output sizes and evaluation correctness.

Run with:
    pytest tests_depth.py -v

Tests that need the metric checkpoint or the KITTI depth-selection data are
skipped when those files are absent. TEST_DEVICE selects cpu (default) or cuda.
"""

import os

import numpy as np
import pytest

from depth_metrics import (
    cluster_bootstrap_ci, devkit_metrics, interpolate_background, quantize_like_devkit_png,
    range_bin_metrics, standard_metrics,
)
from utils import load_config

TEST_DEVICE = os.environ.get("TEST_DEVICE", "cpu")


def _gt_with_holes(seed: int = 0, shape=(40, 60)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gt = rng.uniform(2.0, 70.0, size=shape)
    gt[rng.random(shape) < 0.6] = 0.0          # sparse, like LiDAR ground truth
    return gt


# =============================================================================
# Evaluation metrics
# =============================================================================


def test_perfect_prediction_has_zero_error() -> None:
    gt = _gt_with_holes()
    pred = np.where(gt > 0, gt, 30.0)
    std = standard_metrics(pred, gt)
    assert std["abs_rel"] < 1e-12 and std["rmse"] < 1e-12 and std["silog_x100"] < 1e-6
    assert std["delta1"] == std["delta2"] == std["delta3"] == 1.0
    dk = devkit_metrics(pred, gt)
    # devkit quantises predictions to 1/256 m, so errors are at most half a step
    assert dk["mae"] <= 0.5 / 256 + 1e-12 and dk["abs_rel"] < 1e-3


def test_constant_scale_error_is_scale_invariant_only_in_silog() -> None:
    gt = _gt_with_holes(1)
    pred = np.where(gt > 0, 2.0 * gt, 10.0)
    std = standard_metrics(pred, gt, max_depth=200.0)
    assert std["abs_rel"] == pytest.approx(1.0)
    assert std["rmse_log"] == pytest.approx(np.log(2.0))
    assert std["silog_x100"] == pytest.approx(0.0, abs=1e-4)   # 0 up to float cancellation
    assert std["delta1"] == std["delta2"] == std["delta3"] == 0.0   # 2 > 1.25^3 = 1.95
    dk = devkit_metrics(pred, gt)
    assert dk["silog"] == pytest.approx(0.0, abs=1e-3)
    assert dk["log_rmse"] == pytest.approx(np.log(2.0), abs=1e-3)


def test_devkit_and_standard_sqrel_differ_as_defined() -> None:
    gt = np.array([[2.0, 0.0, 10.0]])
    pred = np.array([[3.0, 5.0, 8.0]])       # errors 1 and 2 on the valid pixels
    std = standard_metrics(pred, gt)
    dk = devkit_metrics(pred, gt)
    assert std["sq_rel"] == pytest.approx((1 / 2 + 4 / 10) / 2)          # mean(err^2 / gt)
    assert dk["sq_rel_devkit"] == pytest.approx((1 / 4 + 4 / 100) / 2)  # mean(err^2 / gt^2)
    assert std["abs_rel"] == pytest.approx(dk["abs_rel"]) == pytest.approx((1 / 2 + 2 / 10) / 2)


def test_silog_definitions() -> None:
    gt = np.array([[5.0, 10.0, 20.0, 40.0]])
    pred = np.array([[6.0, 9.0, 25.0, 30.0]])
    d = np.log(pred) - np.log(gt)
    std = standard_metrics(pred, gt)
    assert std["silog_x100"] == pytest.approx(100 * np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))
    assert std["silog_dav2"] == pytest.approx(np.sqrt(np.mean(d ** 2) - 0.5 * np.mean(d) ** 2))
    assert devkit_metrics(pred, gt)["silog"] == pytest.approx(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2), abs=1e-3)


def test_standard_metrics_respect_range_and_clamp_predictions() -> None:
    gt = np.array([[0.0, 5.0, 90.0]])        # 0 = invalid, 90 m is outside (0, 80]
    pred = np.array([[1.0, 200.0, 90.0]])    # 200 m prediction is clamped to 80
    std = standard_metrics(pred, gt)
    assert std["n_pixels"] == 1
    assert std["rmse"] == pytest.approx(75.0)


def test_devkit_png_quantisation() -> None:
    q = quantize_like_devkit_png(np.array([0.001, 1.0 + 1 / 512 + 1e-9, 300.0]))
    assert q[0] == 0.0                        # encodes as 0 -> invalid
    assert q[1] == pytest.approx(1.0 + 1 / 256)
    assert q[2] == pytest.approx(65535 / 256)  # uint16 saturation


def test_interpolate_background_matches_devkit_semantics() -> None:
    d = np.array([
        [-1, -1, -1, -1, -1, -1],            # empty top row -> copied from the first valid row below
        [-1, 4.0, -1, -1, 9.0, -1],          # interior gap -> min(4, 9); edges copy nearest
        [-1, -1, -1, -1, -1, -1],            # empty row *between* valid rows: the devkit leaves it invalid
        [2.0, -1, 1.0, -1, -1, 3.0],
    ])
    out = interpolate_background(d)
    np.testing.assert_allclose(out[1], [4, 4, 4, 4, 9, 9])
    np.testing.assert_allclose(out[3], [2, 1, 1, 1, 1, 3])
    np.testing.assert_allclose(out[0], out[1])
    assert (out[2] == -1).all()
    assert d[1, 0] == -1                          # input untouched


def test_devkit_fills_invalid_predictions_instead_of_skipping_them() -> None:
    gt = np.array([[10.0, 10.0, 10.0]])
    pred = np.array([[10.0, 0.0, 20.0]])      # hole -> min(10, 20) = 10
    assert devkit_metrics(pred, gt)["mae"] == pytest.approx(10.0 / 3)


def test_metrics_reject_shape_mismatch_and_empty_gt() -> None:
    with pytest.raises(ValueError):
        standard_metrics(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        devkit_metrics(np.ones((2, 2)), np.zeros((2, 2)))


def test_range_bins_partition_valid_pixels() -> None:
    gt = np.array([[5.0, 15.0, 30.0, 60.0, 0.0]])
    bins = range_bin_metrics(gt.copy() + (gt == 0), gt)
    assert list(bins) == ["0-10m", "10-20m", "20-40m", "40-80m"]
    assert sum(b["n_pixels"] for b in bins.values()) == 4


def test_cluster_bootstrap_is_reproducible_and_brackets_mean() -> None:
    rng = np.random.default_rng(3)
    clusters = np.repeat([f"drive{i}" for i in range(8)], 10)
    values = rng.normal(0.1, 0.02, size=80) + np.repeat(rng.normal(0, 0.03, 8), 10)
    a = cluster_bootstrap_ci(values, clusters, seed=7)
    b = cluster_bootstrap_ci(values, clusters, seed=7)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"] and a["n_clusters"] == 8


# =============================================================================
# KITTI depth-selection loader
# =============================================================================

KITTI_ROOT = os.environ.get("KITTI_DEPTH_ROOT", "/workspace/datasets/kitti/depth_selection/extracted/depth_selection")
K_TEXT = "721.5377 0.0 596.5593 0.0 721.5377 149.854 0.0 0.0 1.0"


def _write_fake_kitti(root, frames=(("2011_09_26_drive_0002_sync", "0000000005", "02"),
                                    ("2011_09_26_drive_0002_sync", "0000000006", "03"))):
    import cv2

    split = root / "val_selection_cropped"
    for sub in ("image", "groundtruth_depth", "velodyne_raw", "intrinsics"):
        (split / sub).mkdir(parents=True, exist_ok=True)
    for drive, frame, cam in frames:
        cv2.imwrite(str(split / "image" / f"{drive}_image_{frame}_image_{cam}.png"), np.zeros((8, 12, 3), np.uint8))
        gt = np.zeros((8, 12), np.uint16)
        gt[5, 3] = int(12.5 * 256)            # 12.5 m
        gt[6, 4] = 1                          # smallest encodable depth: 1/256 m
        cv2.imwrite(str(split / "groundtruth_depth" / f"{drive}_groundtruth_depth_{frame}_image_{cam}.png"), gt)
        cv2.imwrite(str(split / "velodyne_raw" / f"{drive}_velodyne_raw_{frame}_image_{cam}.png"), gt)
        (split / "intrinsics" / f"{drive}_image_{frame}_image_{cam}.txt").write_text(K_TEXT + " \n")
    return split


def test_kitti_loader_pairs_files_and_decodes_depth(tmp_path) -> None:
    from kitti_depth import KittiDepthSelection

    _write_fake_kitti(tmp_path)
    ds = KittiDepthSelection(tmp_path)
    assert len(ds) == 2 and ds.drives == ["2011_09_26_drive_0002_sync"]
    s = ds[1]
    assert (s.drive, s.frame, s.camera) == ("2011_09_26_drive_0002_sync", "0000000006", "03")
    gt = s.load_gt()
    assert gt.dtype == np.float32 and gt[5, 3] == 12.5 and gt[6, 4] == pytest.approx(1 / 256)
    assert (gt > 0).sum() == 2                # 0 stays "no ground truth"
    K = s.load_intrinsics()
    assert K[0, 0] == pytest.approx(721.5377) and K[0, 2] == pytest.approx(596.5593) and K[1, 2] == pytest.approx(149.854)
    assert s.load_image().shape == (8, 12, 3)


def test_kitti_loader_refuses_incomplete_split(tmp_path) -> None:
    from kitti_depth import KittiDepthSelection

    split = _write_fake_kitti(tmp_path)
    next((split / "intrinsics").glob("*.txt")).unlink()
    with pytest.raises(FileNotFoundError, match="missing companion"):
        KittiDepthSelection(tmp_path)


def test_kitti_depth_png_rejects_8bit_and_wrong_dtype(tmp_path) -> None:
    import cv2
    from kitti_depth import read_kitti_depth_png, read_kitti_intrinsics

    cv2.imwrite(str(tmp_path / "eight.png"), np.full((4, 4), 200, np.uint8))
    with pytest.raises(ValueError):
        read_kitti_depth_png(tmp_path / "eight.png")
    cv2.imwrite(str(tmp_path / "low16.png"), np.full((4, 4), 200, np.uint16))
    with pytest.raises(ValueError, match="<= 255"):
        read_kitti_depth_png(tmp_path / "low16.png")
    (tmp_path / "k.txt").write_text("1 2 3 4")
    with pytest.raises(ValueError):
        read_kitti_intrinsics(tmp_path / "k.txt")


@pytest.mark.skipif(not os.path.isdir(KITTI_ROOT), reason="KITTI depth-selection data not present")
def test_real_kitti_validation_split_is_complete() -> None:
    from kitti_depth import KittiDepthSelection

    ds = KittiDepthSelection(KITTI_ROOT)
    assert len(ds) == 1000 and len(ds.drives) == 13
    s = ds[0]
    image, gt, K = s.load_image(), s.load_gt(), s.load_intrinsics()
    assert image.shape == (352, 1216, 3) and gt.shape == (352, 1216)
    assert 0 < gt[gt > 0].min() and gt.max() < 100 and 700 < K[0, 0] < 730


# =============================================================================
# Camera model / calibration
# =============================================================================


def test_camera_from_kitti_matrix_and_back() -> None:
    from camera import CameraModel

    K = np.array([[721.5377, 0, 596.5593], [0, 721.5377, 149.854], [0, 0, 1]])
    cam = CameraModel.from_matrix(K, 1216, 352, source="kitti")
    assert cam.calibrated and (cam.width, cam.height) == (1216, 352)
    np.testing.assert_allclose(cam.matrix(), K)


def test_camera_scaling_keeps_field_of_view() -> None:
    from camera import CameraModel

    cam = CameraModel(fx=1000, fy=1000, cx=640, cy=360, width=1280, height=720)
    s = cam.scaled_to(1024, 576)
    assert (s.fx, s.fy, s.cx, s.cy) == pytest.approx((800, 800, 512, 288))
    assert cam.scaled_to(1280, 720) is cam
    # horizontal field of view is unchanged
    assert np.arctan(s.width / 2 / s.fx) == pytest.approx(np.arctan(cam.width / 2 / cam.fx))


def test_camera_refuses_rescaling_across_aspect_ratios() -> None:
    from camera import CameraModel

    cam = CameraModel(fx=1000, fy=1000, cx=640, cy=360, width=1280, height=720)
    with pytest.raises(ValueError, match="aspect ratio"):
        cam.scaled_to(1216, 352)              # KITTI crop needs its own calibration


def test_camera_from_config_is_marked_uncalibrated_by_default() -> None:
    from camera import CameraModel

    cfg = load_config("config.yaml")
    cam = CameraModel.from_config(cfg.camera)
    assert (cam.width, cam.height, cam.fx) == (1280, 720, 1000.0)
    assert cam.calibrated is False


# =============================================================================
# Depth modes, units, output sizes, masks and strict mode
# =============================================================================

METRIC_WEIGHTS = "weights/depth_anything_v2_metric_vkitti_vits.pth"
METRIC_SHA256 = "9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb"
RELATIVE_WEIGHTS = "weights/depth_anything_v2_vits.pth"
needs_metric = pytest.mark.skipif(not os.path.exists(METRIC_WEIGHTS), reason="metric depth checkpoint not present")
needs_relative = pytest.mark.skipif(not os.path.exists(RELATIVE_WEIGHTS), reason="relative depth checkpoint not present")


def _frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    # plain random BGR image: works at any size (generate_synthetic_frame needs >= ~640x480)
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@needs_metric
def test_metric_checkpoint_matches_recorded_sha256() -> None:
    import hashlib

    h = hashlib.sha256()
    with open(METRIC_WEIGHTS, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == METRIC_SHA256


@needs_metric
@pytest.mark.parametrize("width,height", [(1216, 352), (1024, 576), (641, 479)])
def test_metric_depth_is_metres_and_matches_frame_size(width: int, height: int) -> None:
    from models import DEPTH_KIND_METRIC, DepthEstimator

    est = DepthEstimator(weights_path=METRIC_WEIGHTS, mode="metric", max_depth_m=80.0, input_size=154,
                         device_preference=TEST_DEVICE, strict=True)
    result = est.infer(_frame(width, height))
    d = result.depth_map
    assert result.kind == DEPTH_KIND_METRIC and result.is_metric
    assert d.shape == (height, width) and d.dtype == np.float32
    assert np.isfinite(d).all() and d.min() > 0.0 and d.max() <= 80.0


@needs_relative
@pytest.mark.parametrize("width,height", [(1216, 352), (641, 479)])
def test_relative_depth_is_raw_unitless_output(width: int, height: int) -> None:
    from models import DEPTH_KIND_RELATIVE, DepthEstimator

    est = DepthEstimator(weights_path=RELATIVE_WEIGHTS, mode="relative", input_size=154,
                         device_preference=TEST_DEVICE, strict=True)
    frame = _frame(width, height, seed=1)
    result = est.infer(frame)
    assert result.kind == DEPTH_KIND_RELATIVE and not result.is_metric
    assert result.depth_map.shape == (height, width)
    # returned exactly as the model produced it -- no 0-80 m normalisation
    np.testing.assert_array_equal(result.depth_map, est.predict_raw(frame))


@needs_relative
def test_dashboard_rows_are_masked() -> None:
    from models import DepthEstimator

    est = DepthEstimator(weights_path=RELATIVE_WEIGHTS, input_size=154, device_preference=TEST_DEVICE,
                         strict=True, ignore_bottom_rows_fraction=0.25)
    mask = est.infer(_frame(320, 200)).valid_mask
    assert mask.shape == (200, 320)
    assert mask[:150].all() and not mask[150:].any()
    unmasked = DepthEstimator(weights_path=RELATIVE_WEIGHTS, input_size=154, device_preference=TEST_DEVICE)
    assert unmasked.infer(_frame(320, 200)).valid_mask is None


def test_unknown_depth_mode_is_rejected() -> None:
    from models import DepthEstimator

    with pytest.raises(ValueError, match="mode"):
        DepthEstimator(weights_path="missing.pth", mode="metres", device_preference="cpu")


@pytest.mark.parametrize("mode", ["relative", "metric"])
def test_strict_depth_refuses_missing_weights(tmp_path, mode: str) -> None:
    from models import DEPTH_KIND_HEURISTIC, DepthEstimator

    missing = str(tmp_path / "absent.pth")
    with pytest.raises(RuntimeError, match="strict mode"):
        DepthEstimator(weights_path=missing, mode=mode, device_preference="cpu", strict=True)
    # Non-strict keeps the old fallback, but labels it so nothing reads it as metres.
    lenient = DepthEstimator(weights_path=missing, mode=mode, device_preference="cpu", strict=False)
    result = lenient.infer(_frame(160, 120))
    assert result.kind == DEPTH_KIND_HEURISTIC and not result.is_metric


@needs_relative
def test_strict_depth_raises_on_inference_error(monkeypatch) -> None:
    from models import DEPTH_KIND_HEURISTIC, DepthEstimator

    def boom(frame):
        raise RuntimeError("simulated inference failure")

    strict = DepthEstimator(weights_path=RELATIVE_WEIGHTS, input_size=154, device_preference="cpu", strict=True)
    monkeypatch.setattr(strict, "predict_raw", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        strict.infer(_frame(160, 120))
    lenient = DepthEstimator(weights_path=RELATIVE_WEIGHTS, input_size=154, device_preference="cpu")
    monkeypatch.setattr(lenient, "predict_raw", boom)
    assert lenient.infer(_frame(160, 120)).kind == DEPTH_KIND_HEURISTIC


def test_strict_detector_and_weather_refuse_missing_weights(tmp_path) -> None:
    from models import WeatherClassifier, YOLODetector

    with pytest.raises(RuntimeError, match="strict mode"):
        YOLODetector(weights_path=str(tmp_path / "absent.pt"), allow_download=False,
                     device_preference="cpu", strict=True)
    with pytest.raises(RuntimeError, match="strict mode"):
        WeatherClassifier(weights_path=str(tmp_path / "absent.pth"), device_preference="cpu", strict=True)
    # Explicitly non-strict: the documented fallbacks still work.
    assert YOLODetector(weights_path=str(tmp_path / "absent.pt"), allow_download=False,
                        device_preference="cpu", strict=False)._use_fallback
    assert WeatherClassifier(weights_path=str(tmp_path / "absent.pth"), device_preference="cpu")._use_fallback


# =============================================================================
# Pipeline integration: relative mode never yields metres; metric mode does
# =============================================================================


def _pipeline(mode: str, **overrides):
    from pipeline import PerceptionPipeline

    cfg = load_config("config.yaml")
    cfg.device.preference = TEST_DEVICE
    cfg.depth_estimation["mode"] = mode
    cfg.pipeline["strict"] = True              # tests must never pass on a silent fallback
    for key, value in overrides.items():
        cfg.depth_estimation[key] = value
    return PerceptionPipeline(cfg)


@pytest.fixture(scope="module")
def relative_pipeline():
    return _pipeline("relative")


@pytest.fixture(scope="module")
def metric_pipeline():
    if not os.path.exists(METRIC_WEIGHTS):
        pytest.skip("metric depth checkpoint not present")
    return _pipeline("metric")


def _road_like_frame(seed: int = 5):
    from utils import generate_synthetic_frame
    return generate_synthetic_frame(width=1024, height=576, seed=seed)


def test_config_default_depth_mode_is_still_relative() -> None:
    cfg = load_config("config.yaml")
    assert cfg.depth_estimation.mode == "relative"   # application default unchanged in this task
    assert cfg.pipeline.strict is False


def test_relative_pipeline_reports_no_metric_quantities(relative_pipeline, monkeypatch) -> None:
    import pipeline as pl
    from models import DEPTH_KIND_RELATIVE

    out = None
    for i in range(3):
        out = relative_pipeline.process(_road_like_frame(i), timestamp=i / 24.0)
    assert out.depth.kind == DEPTH_KIND_RELATIVE
    assert out.bev is None and out.risk is None and out.relative_velocities == {}

    texts = []
    real_put_text = pl.cv2.putText
    monkeypatch.setattr(pl.cv2, "putText", lambda img, text, *a, **k: (texts.append(text), real_put_text(img, text, *a, **k))[1])
    pl.draw_primary_object_panel(out.frame.copy(), out)
    assert "N/A - relative depth (no metric distance)" in texts
    table = pl.render_object_table_df(out)
    if table is not None:
        assert set(table["Distance (est.)"]) == {"N/A"} and set(table["TTC (est.)"]) == {"N/A"}


def test_metric_pipeline_produces_metric_bev_and_risk(metric_pipeline) -> None:
    from models import DEPTH_KIND_METRIC

    out = None
    for i in range(3):
        out = metric_pipeline.process(_road_like_frame(i), timestamp=i / 24.0)
    assert out.depth.kind == DEPTH_KIND_METRIC
    assert out.bev is not None and out.risk is not None
    assert out.camera is not None and out.camera.calibrated is False      # config camera = placeholder
    assert (out.camera.width, out.camera.height, out.camera.fx) == (1024, 576, pytest.approx(800.0))
    for obj in out.bev.objects:
        assert 0.0 < obj.z_m <= 80.0


def test_metric_pipeline_uses_a_per_frame_camera(metric_pipeline) -> None:
    from camera import CameraModel

    frame = _road_like_frame(1)
    narrow = CameraModel(fx=2000.0, fy=2000.0, cx=512.0, cy=288.0, width=1024, height=576,
                         source="test", calibrated=True)
    out = metric_pipeline.process(frame, timestamp=10.0, camera=narrow)
    assert out.camera is narrow
    assert (metric_pipeline.bev_mapper.fx, metric_pipeline.bev_mapper.cx) == (2000.0, 512.0)
    with pytest.raises(ValueError, match="depth map"):
        metric_pipeline.process(frame, timestamp=11.0, camera=CameraModel(
            fx=700, fy=700, cx=600, cy=150, width=1216, height=352, calibrated=True))


def test_calibrated_config_camera_refuses_other_aspect_ratio(relative_pipeline) -> None:
    import dataclasses

    frame = np.zeros((480, 640, 3), np.uint8)               # 4:3 vs the 16:9 config camera
    assert relative_pipeline.camera_for(frame).width == 640   # placeholder: per-axis scaling allowed
    calibrated = dataclasses.replace(relative_pipeline.default_camera, calibrated=True)
    with pytest.raises(ValueError, match="aspect ratio"):
        calibrated.scaled_to(640, 480)


def test_bev_skips_objects_whose_depth_is_masked() -> None:
    from camera import CameraModel
    from mapping import BEVMapper
    from models import Detection

    depth = np.full((200, 300), 15.0, dtype=np.float32)
    mask = np.ones((200, 300), dtype=bool)
    mask[150:] = False                                       # "dashboard"
    cam = CameraModel(fx=300, fy=300, cx=150, cy=100, width=300, height=200, calibrated=True)
    on_road = Detection(box=[100, 60, 140, 120], class_id=2, class_name="car", confidence=0.9, track_id=1)
    on_dash = Detection(box=[100, 160, 140, 195], class_id=2, class_name="car", confidence=0.9, track_id=2)
    bev = BEVMapper(fx=1, fy=1, cx=0, cy=0).build_bev(np.zeros((200, 300, 3), np.uint8), depth,
                                                       [on_road, on_dash], camera=cam, valid_mask=mask)
    assert [o.track_id for o in bev.objects] == [1]
    assert bev.objects[0].z_m == pytest.approx(15.0)
    assert bev.objects[0].x_m == pytest.approx((120 - 150) * 15.0 / 300)


def test_display_depth_keeps_near_and_far_orientation() -> None:
    from models import DEPTH_KIND_METRIC, DEPTH_KIND_RELATIVE, DepthResult
    from pipeline import display_depth

    near_far_inverse = np.array([[5.0, 1.0]])                # relative: near = large
    shown = display_depth(DepthResult(depth_map=near_far_inverse, kind=DEPTH_KIND_RELATIVE))
    assert shown[0, 0] < shown[0, 1]                         # display: near = small, like metres
    metric = np.array([[2.0, 30.0]])
    np.testing.assert_array_equal(display_depth(DepthResult(depth_map=metric, kind=DEPTH_KIND_METRIC)), metric)


# =============================================================================
# Comparison methods (legacy conversion, oracle alignment, ground plane, focal rescale)
# =============================================================================


def test_legacy_conversion_reproduces_old_formula() -> None:
    from depth_methods import LegacyLinearConversion

    raw = np.array([[0.0, 2.0, 8.0]], dtype=np.float32)       # inverse depth: 8 = nearest
    np.testing.assert_allclose(LegacyLinearConversion(alpha=1.0)(raw), [[80.0, 60.0, 0.0]])
    seq = LegacyLinearConversion(alpha=0.3)
    seq(raw)
    second = seq(raw * 2.0)                                    # max 16 -> EMA bound 0.3*16 + 0.7*8 = 10.4
    np.testing.assert_allclose(second, [[80.0, 80.0 * (1 - 4 / 10.4), 0.0]], rtol=1e-6)


def test_oracle_alignment_recovers_affine_inverse_depth() -> None:
    from depth_methods import align_inverse_depth

    rng = np.random.default_rng(0)
    gt = rng.uniform(3.0, 70.0, size=(30, 40))
    raw = 2.5 * (1.0 / gt) + 0.04                              # d = a/Z + b
    sparse = np.where(rng.random(gt.shape) < 0.3, gt, 0.0)
    pred, info = align_inverse_depth(raw, sparse)
    np.testing.assert_allclose(pred, gt, rtol=1e-4)
    assert info["scale"] == pytest.approx(1 / 2.5, rel=1e-6)


def _flat_road_scene(fy=720.0, cy=150.0, height=1.65, a=3.0, b=0.05, shape=(352, 1216)):
    h, w = shape
    v = np.arange(h, dtype=np.float64)[:, None].repeat(w, axis=1)
    Z = np.full(shape, 80.0)
    below = v > cy + 0.5
    Z[below] = np.minimum(fy * height / (v[below] - cy), 80.0)
    return Z, a / Z + b


def test_ground_plane_recovers_metric_scale_on_flat_road() -> None:
    from depth_methods import ground_plane_scale

    K = np.array([[720.0, 0, 608.0], [0, 720.0, 150.0], [0, 0, 1]])
    Z, raw = _flat_road_scene()
    pred, fit = ground_plane_scale(raw, K, camera_height_m=1.65)
    assert fit.ok and fit.scale == pytest.approx(3.0, rel=1e-6) and fit.shift == pytest.approx(0.05, rel=1e-6)
    road = Z < 79.0
    np.testing.assert_allclose(pred[road], Z[road], rtol=1e-4)


def test_ground_plane_uses_given_height_and_reports_failure() -> None:
    from depth_methods import ground_plane_scale

    K = np.array([[720.0, 0, 608.0], [0, 720.0, 150.0], [0, 0, 1]])
    Z, raw = _flat_road_scene()
    pred, fit = ground_plane_scale(raw, K, camera_height_m=3.30)    # wrong height -> depths double
    assert fit.ok and np.median(pred[Z < 20] / Z[Z < 20]) == pytest.approx(2.0, rel=1e-3)
    flat = np.ones_like(raw)                                        # no depth structure at all
    pred, fit = ground_plane_scale(flat, K, camera_height_m=1.65)
    assert pred is None and not fit.ok and fit.reason


def test_focal_rescale() -> None:
    from depth_methods import VKITTI2_FOCAL_PX, focal_rescale

    d = np.array([[10.0, 79.0]])
    np.testing.assert_allclose(focal_rescale(d, fx=VKITTI2_FOCAL_PX), d)
    np.testing.assert_allclose(focal_rescale(d, fx=2 * VKITTI2_FOCAL_PX), [[20.0, 80.0]])   # capped at 80


def test_bev_status_is_inactive_without_metric_depth(relative_pipeline) -> None:
    import pipeline as pl

    out = relative_pipeline.process(_road_like_frame(2), timestamp=0.0)
    html = pl.render_module_status_html(out, relative_pipeline.pipeline_cfg)
    bev_part = html.split("BEV Mapping", 1)[1]
    assert "INACTIVE" in bev_part.split("</div>", 2)[0] + bev_part.split("</div>", 2)[1]
