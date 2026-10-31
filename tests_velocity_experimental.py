"""Tests for the experimental Stage 2 velocity components (velocity_experimental.py) and the
Stage 2 additions to the evaluation harness. Synthetic data only.

Run with:
    pytest tests_velocity_experimental.py -v
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from camera import CameraModel
from velocity_eval import CachedFrame, CachedObject, calibration_stats, gating_curve, replay_many
from velocity_experimental import (
    DepthConsistencyGate, KalmanAdapter, KalmanParams, KLTParams, ObjectKalmanFilter, RobustDepthParams,
    ScaleAdapter, build_estimators, klt_scale, robust_object_depth,
)

VCFG = {"history_length": 5, "max_gap_seconds": 2.0, "ema_alpha": 0.5}
CAM = CameraModel(fx=700.0, fy=700.0, cx=320.0, cy=120.0, width=640, height=240, source="test", calibrated=True)


# --- robust depth ------------------------------------------------------------------------------


def test_robust_depth_ignores_background_and_road() -> None:
    depth = np.full((240, 640), 60.0, dtype=np.float32)        # background
    depth[100:200, 200:300] = 20.0                              # the object
    depth[185:200, 200:300] = 8.0                               # road below the bumper, inside the box
    depth[100:200, 200:215] = 60.0                              # background at the box edge
    z, sigma, n = robust_object_depth(depth, (200, 100, 300, 200))
    assert z == pytest.approx(20.0) and n > 1000
    assert sigma == pytest.approx(0.05 * 20.0)                  # floor: no spread among inliers
    # B0's 5x5 bottom-centre window would read the road (8 m) here
    assert np.median(depth[197:200, 248:253]) == 8.0


def test_robust_depth_rejects_tiny_or_empty_regions() -> None:
    depth = np.zeros((240, 640), dtype=np.float32)
    assert np.isnan(robust_object_depth(depth, (10, 10, 60, 60))[0])          # no valid pixels
    assert np.isnan(robust_object_depth(np.ones((240, 640)), (10, 10, 12, 12))[0])  # too small
    depth[:] = 30.0
    z, _, _ = robust_object_depth(depth, (10, 10, 60, 60), RobustDepthParams(percentile=30))
    assert z == pytest.approx(30.0)


# --- depth consistency gate --------------------------------------------------------------------


def test_depth_gate_rejects_spikes_but_not_real_jumps() -> None:
    g = DepthConsistencyGate(k=3.0, rel_sigma=0.05, v_max=45.0, max_rejects=2)
    zs = [30.0, 29.5, 29.0, 5.0, 28.0]                          # 5 m is a depth outlier
    out = [g(0.1 * k, z) for k, z in enumerate(zs)]
    assert [ok for _, ok in out] == [True, True, True, False, True]
    assert out[3][0] == pytest.approx(28.5)                     # replaced by the extrapolation
    g2 = DepthConsistencyGate(max_rejects=2)
    out2 = [g2(0.1 * k, z) for k, z in enumerate([30, 30, 30, 2, 2, 2, 2])]
    assert [ok for _, ok in out2] == [True, True, True, False, False, True, True]   # re-anchors


# --- Kalman filter ---------------------------------------------------------------------------


def test_kalman_converges_with_irregular_dt_and_gates_outliers() -> None:
    rng = np.random.default_rng(0)
    p = KalmanParams(q_lon=0.5, q_lat=0.5)
    R = np.diag([0.2 ** 2, 0.5 ** 2])
    t, x, z = 0.0, 2.0, 40.0
    kf = ObjectKalmanFilter(t, x, z, R, p)
    for k in range(60):
        t += 0.1 if k % 3 else 0.11                             # irregular spacing
        kf.predict(t)
        meas_z = 40.0 - 8.0 * t + rng.normal(0, 0.5)
        if k == 30:
            meas_z = 10.0                                       # gross outlier
        ok = kf.update_position(2.0 + 0.5 * t + rng.normal(0, 0.2), meas_z, R)
        assert ok == (k != 30)
    assert kf.s[3] == pytest.approx(-8.0, abs=0.6) and kf.s[2] == pytest.approx(0.5, abs=0.4)
    assert np.sqrt(kf.P[3, 3]) < 1.5


def test_kalman_velocity_measurement_reduces_uncertainty() -> None:
    p = KalmanParams()
    R = np.diag([0.1, 1.0])
    a, b = ObjectKalmanFilter(0.0, 0, 30, R, p), ObjectKalmanFilter(0.0, 0, 30, R, p)
    for k in range(1, 6):
        for kf in (a, b):
            kf.predict(0.1 * k)
            kf.update_position(0, 30 - 0.5 * k, R)
        b.update_closing_speed(-5.0, 0.3)
    assert b.P[3, 3] < a.P[3, 3] and b.s[3] == pytest.approx(-5.0, abs=0.5)


# --- KLT scale ---------------------------------------------------------------------------------


def _textured(seed=0, size=(240, 640)):
    rng = np.random.default_rng(seed)
    img = (rng.random((size[0] // 4, size[1] // 4)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST), (5, 5), 0)


def _zoom(img, s, centre):
    M = cv2.getRotationMatrix2D(centre, 0, s)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("s_true", [1.0, 1.04, 0.97])
def test_klt_scale_recovers_zoom(s_true) -> None:
    img = _textured()
    c = (320.0, 120.0)
    box0 = (260.0, 80.0, 380.0, 160.0)
    box1 = tuple(np.array([c[0] + (box0[0] - c[0]) * s_true, c[1] + (box0[1] - c[1]) * s_true,
                           c[0] + (box0[2] - c[0]) * s_true, c[1] + (box0[3] - c[1]) * s_true]))
    m = klt_scale(img, _zoom(img, s_true, c), box0, box1)
    assert m is not None and m.n_points >= 20
    assert m.scale == pytest.approx(s_true, abs=0.004) and m.sigma < 0.01


def test_klt_forward_backward_check_drops_bad_tracks() -> None:
    img = _textured(1)
    nxt = img.copy()
    nxt[100:140, 300:340] = _textured(2)[100:140, 300:340]     # a patch that changes completely
    box = (260.0, 80.0, 380.0, 160.0)
    with_fb = klt_scale(img, nxt, box, box, KLTParams(fb_max_px=1.0))
    without = klt_scale(img, nxt, box, box, KLTParams(fb_max_px=float("inf")))
    assert with_fb is not None and without is not None and with_fb.n_points < without.n_points
    assert with_fb.scale == pytest.approx(1.0, abs=0.003)
    assert klt_scale(img, img, (0, 0, 10, 10), (0, 0, 10, 10)) is None      # too small


# --- adapters on a synthetic approaching object -------------------------------------------------


def _approach_frames(n=12, z0=30.0, v=-10.0, dt=0.1):
    """A textured car-sized plane at x=0, approaching at |v| m/s, rendered into a static background."""
    bg = _textured(3)
    tex = _textured(4, (200, 200))
    frames = []
    for k in range(n):
        z = z0 + v * dt * k
        half_w, half_h = 0.9 * CAM.fx / z, 0.75 * CAM.fx / z
        x1, x2, y1, y2 = CAM.cx - half_w, CAM.cx + half_w, CAM.cy - half_h, CAM.cy + half_h
        img = bg.copy()
        patch = cv2.resize(tex, (int(round(x2 - x1)), int(round(y2 - y1))), interpolation=cv2.INTER_LINEAR)
        img[int(round(y1)):int(round(y1)) + patch.shape[0], int(round(x1)):int(round(x1)) + patch.shape[1]] = patch
        depth = np.full(img.shape, 80.0, dtype=np.float32)
        depth[int(y1):int(y2), int(x1):int(x2)] = z
        obj = CachedObject(track_id=7, class_name="car", box=(x1, y1, x2, y2), confidence=0.9, has_bev=True,
                           x_m=0.0, z_m=z)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frames.append(CachedFrame(sequence="s", frame=k, timestamp=dt * k, objects=[obj], camera=CAM,
                                  load_image=lambda bgr=bgr: bgr, load_depth=lambda d=depth: d))
    return frames


def test_scale_adapter_measures_closing_speed_without_depth_differences() -> None:
    frames = _approach_frames()
    out = ScaleAdapter(VCFG)
    out.reset()
    est = [out.process(f) for f in frames]
    assert not est[0][7].valid and all(e[7].valid for e in est[1:])
    assert -est[-1][7].vz_m_s == pytest.approx(10.0, rel=0.08)
    assert abs(est[-1][7].vx_m_s) < 1.0 and est[-1][7].sigma_vz > 0


@pytest.mark.parametrize("name", ["K_kalman", "KS_kalman_scale", "DKS_robust_kalman_scale", "D_robust_depth",
                                  "T_depth_gate"])
def test_depth_based_adapters_track_a_clean_approach(name) -> None:
    frames = _approach_frames(n=15)
    est, times = replay_many(build_estimators(VCFG, [name]), frames)
    last = est[name][-1][7]
    assert last.valid and -last.vz_m_s == pytest.approx(10.0, rel=0.15)
    assert len(times[name]) == 15


def test_sigma_gate_marks_uncertain_estimates_invalid() -> None:
    frames = _approach_frames(n=6)
    gated = KalmanAdapter(VCFG, sigma_gate=0.01)
    gated.reset()
    assert not any(gated.process(f)[7].valid for f in frames)


def test_build_estimators_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_estimators(VCFG, ["B0", "B9"])


# --- calibration and gating metrics ------------------------------------------------------------


def test_calibration_of_correct_gaussian_sigmas() -> None:
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 3.0, 20000)
    err = rng.normal(0, sigma)
    c = calibration_stats(err, sigma)
    assert c["frac_within_1sigma"] == pytest.approx(0.683, abs=0.01)
    assert c["frac_within_1.96sigma"] == pytest.approx(0.95, abs=0.01)
    for b in c["by_sigma_quintile"]:
        assert b["rmse"] == pytest.approx(b["mean_sigma"], rel=0.1)
    over = calibration_stats(err, 0.5 * sigma)
    assert over["frac_within_1sigma"] < 0.45                      # overconfident sigmas are detected


def test_gating_curve_trades_coverage_for_error() -> None:
    rows = [dict(matched=True, X_valid=True, X_vz=-(2.0 + e), X_sigma=s, gt_closing=2.0)
            for e, s in [(0.1, 0.2), (0.5, 0.6), (3.0, 2.5)]] + [dict(matched=False, X_valid=False)]
    c = gating_curve(rows, "X", [0.3, 1.0, 10.0])
    assert [r["n"] for r in c] == [1, 2, 3] and [r["coverage_end_to_end"] for r in c] == [0.25, 0.5, 0.75]
    assert c[0]["mae"] == pytest.approx(0.1) and c[2]["mae"] == pytest.approx(3.6 / 3)


def test_harness_refuses_to_score_test_split_with_experimental_estimators(monkeypatch, tmp_path) -> None:
    import evaluate_kitti_velocity as ev
    monkeypatch.setattr(sys, "argv", ["x", "evaluate", "--cache", str(tmp_path), "--split", "test",
                                      "--estimators", "B0", "K_kalman"])
    with pytest.raises(RuntimeError, match="--score-test"):
        ev.main()
    monkeypatch.setattr(sys, "argv", ["x", "evaluate", "--cache", str(tmp_path), "--split", "dev",
                                      "--estimators", "K_kalman"])
    with pytest.raises(RuntimeError, match="B0 must be part"):
        ev.main()
