"""Synthetic tests for the experimental ego-motion module (ego_motion.py, Task 5 Stage 3).

Run with:
    pytest tests_ego_motion.py -v

Conventions under test: camera x right, y down, z forward; X_t = R X_{t-1} + t; the camera's
displacement is -R^T t and its rotation R^T; forward speed = displacement_z / dt; yaw rate =
yaw_about_y(R^T) / dt, positive = turning right (= -wz of the OXTS).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from camera import CameraModel
from ego_motion import (
    EgoEstimate, FlowDepthBaseline, PnPEgoMotion, PnPParams, background_mask, backproject, build_ego_estimators,
    camera_motion, relative_pose, rotation_parallax, yaw_about_y,
)
from velocity_eval import CachedFrame, CachedObject

CAM = CameraModel(fx=720.0, fy=720.0, cx=620.0, cy=180.0, width=1240, height=370, source="test", calibrated=True)
K = CAM.matrix()


def Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project(X: np.ndarray) -> np.ndarray:
    return np.stack([K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]], axis=1)


def scene(n=600, seed=0) -> np.ndarray:
    """Static points in the t-1 camera frame: a road plane 1.65 m below and building facades."""
    rng = np.random.default_rng(seed)
    road = np.stack([rng.uniform(-8, 8, n // 2), np.full(n // 2, 1.65), rng.uniform(4, 35, n // 2)], 1)
    walls = np.stack([np.where(rng.random(n // 2) < 0.5, -7.0, 7.0), rng.uniform(-4, 1.6, n // 2),
                      rng.uniform(5, 38, n // 2)], 1)
    return np.vstack([road, walls])


def move(X: np.ndarray, forward_m: float, yaw: float):
    """Points seen after the camera drives forward_m along its z axis while yawing by `yaw` (right > 0)."""
    Rc, c = Ry(yaw), np.array([0.0, 0.0, forward_m])
    return (X - c) @ Rc        # = Rc^T (X - c), row-wise


def depth_image(p0: np.ndarray, z: np.ndarray) -> np.ndarray:
    d = np.zeros((CAM.height, CAM.width), dtype=np.float32)
    u, v = np.round(p0[:, 0]).astype(int), np.round(p0[:, 1]).astype(int)
    ok = (u >= 0) & (u < CAM.width) & (v >= 0) & (v < CAM.height)
    d[v[ok], u[ok]] = z[ok]
    return d


def correspondences(forward_m, yaw, seed=0, noise_px=0.0):
    X0 = scene(seed=seed)
    X1 = move(X0, forward_m, yaw)
    p0, p1 = project(X0), project(X1)
    ok = (X1[:, 2] > 1) & (p0[:, 0] > 0) & (p0[:, 0] < CAM.width) & (p0[:, 1] > 0) & (p0[:, 1] < CAM.height) \
        & (p1[:, 0] > 0) & (p1[:, 0] < CAM.width) & (p1[:, 1] > 0) & (p1[:, 1] < CAM.height)
    p0, p1, X0 = p0[ok], p1[ok], X0[ok]
    rng = np.random.default_rng(seed + 1)
    p0r = np.round(p0)                       # depth is sampled at the rounded pixel: use exact pixels
    X0 = backproject(p0r, X0[:, 2], K)
    p1 = project(move(X0, forward_m, yaw)) + rng.normal(0, noise_px, size=p0r.shape)
    return p0r.astype(np.float32), p1.astype(np.float32), depth_image(p0r, X0[:, 2])


# --- conventions ----------------------------------------------------------------------------------


def test_sign_conventions_of_camera_motion() -> None:
    R, t = Ry(0.05).T, -Ry(0.05).T @ np.array([0, 0, 1.0])       # drove 1 m forward, yawed right 0.05 rad
    c, Rc = camera_motion(R, t)
    assert np.allclose(c, [0, 0, 1.0]) and yaw_about_y(Rc) == pytest.approx(0.05)
    # a right turn makes a distant straight-ahead point move left in the image
    X = np.array([[0.0, 0.0, 1000.0]])
    assert project(move(X, 0.0, 0.05))[0, 0] < K[0, 2]


def test_rotation_parallax_is_zero_for_pure_rotation_whatever_the_depth() -> None:
    X0 = scene()
    X1 = move(X0, 0.0, 0.03)
    R = Ry(0.03).T
    par = rotation_parallax(project(X0), project(X1), R, K)
    assert np.max(par) < 1e-6
    par_t = rotation_parallax(project(X0), project(move(X0, 1.0, 0.03)), R, K)
    assert np.median(par_t) > 2.0                                  # translation leaves depth-dependent parallax


# --- relative pose from correspondences ------------------------------------------------------------


@pytest.mark.parametrize("forward, yaw", [(0.0, 0.0), (1.2, 0.0), (1.0, 0.03), (0.6, -0.04), (-0.5, 0.0)])
@pytest.mark.parametrize("rotation", ["pnp", "essential"])
def test_relative_pose_recovers_metric_motion(forward, yaw, rotation) -> None:
    p0, p1, d0 = correspondences(forward, yaw, noise_px=0.2)
    if rotation == "essential" and abs(forward) < 1e-9:
        pytest.skip("the essential matrix is undefined without translation")
    R, t, inl = relative_pose(p0, p1, d0, K, PnPParams(rotation=rotation))
    c, Rc = camera_motion(R, t)
    assert c[2] == pytest.approx(forward, abs=0.03) and abs(c[0]) < 0.05
    # the essential-matrix rotation is less precise than PnP's with good depth (~1.5 mrad at 0.2 px noise)
    assert yaw_about_y(Rc) == pytest.approx(yaw, abs=1e-3 if rotation == "pnp" else 3e-3)
    assert inl.mean() > 0.9


def test_moving_object_points_are_rejected_as_outliers() -> None:
    p0, p1, d0 = correspondences(1.0, 0.0, noise_px=0.2)
    bad = np.arange(0, len(p1), 4)                                 # 25% of points on a car moving with us
    p1 = p1.copy()
    p1[bad] = p0[bad] + np.random.default_rng(3).normal(0, 0.2, (len(bad), 2))
    R, t, inl = relative_pose(p0, p1, d0, K, PnPParams())
    c, _ = camera_motion(R, t)
    assert c[2] == pytest.approx(1.0, abs=0.05) and inl[bad].mean() < 0.1


def test_missing_and_invalid_depth_is_handled() -> None:
    p0, p1, d0 = correspondences(1.0, 0.0)
    assert relative_pose(p0, p1, np.zeros_like(d0), K, PnPParams()) is None          # no depth at all
    d_nan = d0.copy()
    d_nan[d_nan > 0] = np.nan
    assert relative_pose(p0, p1, d_nan, K, PnPParams()) is None                      # NaN depth
    assert relative_pose(p0[:5], p1[:5], d0, K, PnPParams()) is None                 # too few points


def test_background_mask_excludes_grown_boxes() -> None:
    m = background_mask((100, 200), [(50, 20, 70, 60)], grow=0.1)
    assert m[40, 60] == 0 and m[40, 48] == 0 and m[40, 40] == 255 and m[90, 190] == 255


# --- full estimators on rendered images -------------------------------------------------------------


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((CAM.height // 3 + 1, CAM.width // 3 + 1)) * 255).astype(np.uint8)
    img = cv2.resize(img, (CAM.width, CAM.height), interpolation=cv2.INTER_NEAREST)[:CAM.height, :CAM.width]
    return cv2.GaussianBlur(img, (5, 5), 0)


def _frames_two_planes(forward_per_frame: float, yaw_per_frame: float, n: int = 6, dt: float = 0.1):
    """Exact renderings of a static scene of two fronto-parallel textured planes -- the left half of the
    first view at 12 m, the right half at 25 m -- for a camera that drives forward and yaws each frame.
    Each pixel's ray is intersected with the scene in the first camera's frame and the texture is looked
    up where that point appears in the first image; the depth map is the z of that point in the current
    camera frame."""
    tex = _texture()
    W, Hh = CAM.width, CAM.height
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(Hh, dtype=np.float64))
    rays = np.linalg.inv(K) @ np.vstack([us.ravel(), vs.ravel(), np.ones(us.size)])
    frames = []
    for k in range(n):
        Rc, c = Ry(yaw_per_frame * k), np.array([0, 0, forward_per_frame * k])
        rw = Rc @ rays                                     # ray directions in the first camera's frame
        best_u = np.full(us.size, np.nan)
        best_v = np.full(us.size, np.nan)
        best_z = np.full(us.size, np.nan)
        for d, left in ((12.0, True), (25.0, False)):
            s_ = (d - c[2]) / rw[2]
            P = c[:, None] + s_ * rw                       # intersection with the plane z = d
            u0 = K[0, 0] * P[0] / P[2] + K[0, 2]
            v0 = K[1, 1] * P[1] / P[2] + K[1, 2]
            on = (s_ > 0) & ((u0 < W / 2) if left else (u0 >= W / 2)) & np.isnan(best_u)
            best_u[on], best_v[on], best_z[on] = u0[on], v0[on], (s_ * rays[2])[on]
        mapx = best_u.reshape(Hh, W).astype(np.float32)
        mapy = best_v.reshape(Hh, W).astype(np.float32)
        img = cv2.remap(tex, np.nan_to_num(mapx, nan=-1), np.nan_to_num(mapy, nan=-1), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth = np.nan_to_num(best_z.reshape(Hh, W), nan=0.0).astype(np.float32)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frames.append(CachedFrame(sequence="s", frame=k, timestamp=dt * k, objects=[], camera=CAM,
                                  load_image=lambda b=bgr: b, load_depth=lambda dd=depth: dd))
    return frames


@pytest.mark.parametrize("name", ["EM1_pnp_raw", "EM2_pnp_filtered"])
def test_estimators_on_rendered_straight_drive(name) -> None:
    est = build_ego_estimators([name])[name]
    out = [est.process(f) for f in _frames_two_planes(1.0, 0.0)]
    assert not out[0].valid                                         # no previous frame yet
    for e in out[2:]:
        assert e.valid and e.moving and e.speed == pytest.approx(10.0, rel=0.08) and abs(e.yaw_rate) < 0.02


def test_estimators_on_rendered_turn() -> None:
    est = PnPEgoMotion(PnPParams(use_filter=False))
    out = [est.process(f) for f in _frames_two_planes(0.8, 0.01)]
    for e in out[1:]:
        assert e.valid and e.yaw_rate == pytest.approx(0.1, abs=0.015) and e.speed == pytest.approx(8.0, rel=0.1)


def test_baseline_needs_distant_background() -> None:
    """The flow/depth baseline measures yaw on points beyond 30 m; this scene ends at 25 m."""
    est = FlowDepthBaseline()
    out = [est.process(f) for f in _frames_two_planes(0.0, 0.0)]
    assert all(not e.valid and e.reason == "too few far or near points" for e in out[1:])
    est = FlowDepthBaseline(far_m=20.0)
    out = [est.process(f) for f in _frames_two_planes(0.0, 0.0)]
    assert all(e.valid and e.moving is False and abs(e.yaw_rate) < 0.01 for e in out[1:])


@pytest.mark.parametrize("name", ["EM1_pnp_raw", "EM2_pnp_filtered"])
def test_estimators_report_stationary_camera(name) -> None:
    est = build_ego_estimators([name])[name]
    out = [est.process(f) for f in _frames_two_planes(0.0, 0.0)]
    for e in out[1:]:
        assert e.valid and e.moving is False and abs(e.speed) < 0.1 and abs(e.yaw_rate) < 0.01


def test_rotation_only_is_stationary_but_turning() -> None:
    est = PnPEgoMotion(PnPParams(use_filter=False))
    out = [est.process(f) for f in _frames_two_planes(0.0, 0.01)]
    for e in out[1:]:
        assert e.valid and e.moving is False and e.yaw_rate == pytest.approx(0.1, abs=0.01)


def test_missing_frames_and_featureless_images_are_invalid() -> None:
    frames = _frames_two_planes(1.0, 0.0, n=4)
    est = PnPEgoMotion()
    est.process(frames[0])
    frames[2].timestamp = 0.5                                       # a gap: not consecutive
    assert est.process(frames[2]).reason == "no consecutive previous frame"
    blank = CachedFrame(sequence="s", frame=9, timestamp=0.6, objects=[], camera=CAM,
                        load_image=lambda: np.zeros((CAM.height, CAM.width, 3), np.uint8),
                        load_depth=lambda: np.full((CAM.height, CAM.width), 10.0, np.float32))
    e = est.process(blank)
    assert not e.valid or e.reason                                  # featureless: no measurement
    masked = CachedFrame(sequence="s", frame=10, timestamp=0.7, camera=CAM, load_image=frames[3].load_image,
                         load_depth=frames[3].load_depth,
                         objects=[CachedObject(1, "car", (0, 0, CAM.width, CAM.height), 0.9)])
    assert est.measure(0.6, frames[3].gray(), frames[3].depth(), [(0, 0, CAM.width, CAM.height)], 0.7,
                       masked.gray(), [], K).reason == "too few background features"


def test_unknown_estimator_name() -> None:
    with pytest.raises(ValueError):
        build_ego_estimators(["EM9"])
