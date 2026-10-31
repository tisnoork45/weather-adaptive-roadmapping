"""EXPERIMENTAL monocular ego-motion estimation (Task 5, Stage 3). Offline only.

Nothing in the live pipeline imports this module. Inputs are restricted to what
the camera pipeline itself has at run time: consecutive frames, the camera
matrix, the pipeline's metric depth map and its object boxes (to mask objects
that may move), and frame timestamps. No GPS/IMU/OXTS value is ever an input;
KITTI OXTS is used only by the evaluation as ground truth.

Frames and conventions (same as kitti_velocity_gt):

* camera frame: x right, y down, z forward (metres).
* relative pose between frames t-1 and t: (R, t) with X_t = R X_{t-1} + t for a
  static point. The camera's own displacement, expressed in the t-1 camera
  frame, is c = -R^T t, and its rotation (camera t axes in t-1 frame) is R^T.
* forward speed: c_z / dt (m/s, positive = driving forward).
* yaw rate: yaw_about_y(R^T) / dt (rad/s, about the camera's y axis, which
  points down; positive = turning right). This equals -wz of the OXTS.
* dt: the difference of the frames' timestamps (nominal 0.1 s on KITTI).

Estimators:

* ``FlowDepthBaseline`` -- the simple baseline: yaw from the median horizontal
  flow of distant background points, speed from the median depth decrease of
  near background points, stationary if |speed| < 0.5 m/s.
* ``PnPEgoMotion`` -- KLT background features with a forward-backward check;
  3D points from the metric depth at t-1, their positions at t, PnP with
  RANSAC for metric (R, t) (rotation optionally from the essential matrix);
  a stationary test on rotation-compensated parallax (scale free); and an
  optional time-aware Kalman filter on speed and yaw rate that also reports a
  1-sigma uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
MAX_FRAME_GAP_S = 0.15


@dataclass
class EgoEstimate:
    valid: bool
    speed: float = np.nan            # forward speed (m/s)
    yaw_rate: float = np.nan         # rad/s, positive = turning right
    moving: Optional[bool] = None
    sigma_speed: float = np.nan
    sigma_yaw: float = np.nan
    n_inliers: int = 0
    parallax_px: float = np.nan      # median rotation-compensated parallax of the inliers
    reason: str = ""


def yaw_about_y(R: np.ndarray) -> float:
    return float(np.arctan2(R[0, 2], R[2, 2]))


def camera_motion(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(camera displacement in the t-1 frame, camera rotation) from X_t = R X_{t-1} + t."""
    return -R.T @ np.asarray(t).reshape(3), R.T


def background_mask(shape, boxes: Sequence, grow: float = 0.1) -> np.ndarray:
    """uint8 mask, 0 inside (grown) object boxes: those objects may move and must not vote for ego motion."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        gw, gh = grow * (x2 - x1), grow * (y2 - y1)
        mask[int(max(0, y1 - gh)):int(min(h, y2 + gh + 1)), int(max(0, x1 - gw)):int(min(w, x2 + gw + 1))] = 0
    return mask


def track_features(g0: np.ndarray, g1: np.ndarray, mask: Optional[np.ndarray], max_corners: int = 800,
                   fb_max_px: float = 0.5, min_distance: int = 8, win: int = 21, levels: int = 3,
                   predict=None) -> Tuple[np.ndarray, np.ndarray]:
    """KLT correspondences (N, 2) x 2 that pass the forward-backward check.

    ``predict(p0) -> p1_guess`` optionally seeds the forward search (e.g. with the flow implied by the
    previous ego motion and the depth map), so that large motions are not tracked short."""
    p0 = cv2.goodFeaturesToTrack(g0, max_corners, 0.01, min_distance, mask=mask, blockSize=7)
    if p0 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    lk = dict(_LK, winSize=(win, win), maxLevel=levels)
    if predict is not None:
        guess = predict(p0.reshape(-1, 2)).astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    else:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    fb = np.linalg.norm((p0r - p0).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb <= fb_max_px)
    h, w = g0.shape
    p0, p1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    return p0[inside], p1[inside]


def sample_depth(depth: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return depth[v, u].astype(np.float64)


def backproject(pts: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (pts[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pts[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def rotation_parallax(p0: np.ndarray, p1: np.ndarray, R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """|p1 - K R K^-1 p0| per point: the image motion left after removing the rotation.
    For a camera that only rotates this is ~0 whatever the depth -- a scale-free stationarity test."""
    H = K @ R @ np.linalg.inv(K)
    ph = np.hstack([p0, np.ones((len(p0), 1))]) @ H.T
    return np.linalg.norm(ph[:, :2] / ph[:, 2:3] - p1, axis=1)


@dataclass(frozen=True)
class PnPParams:
    min_depth: float = 2.0
    max_depth: float = 40.0          # metric depth beyond this is too uncertain to fix the scale
    reproj_px: float = 1.5
    min_inliers: int = 20
    lk_win: int = 21
    lk_levels: int = 3
    fb_max_px: float = 0.5
    predict_flow: bool = False       # seed KLT with the flow implied by the previous frame's motion + depth
    rotation: str = "pnp"            # "pnp" or "essential" (rotation from the essential matrix)
    trans_max_depth: Optional[float] = None   # if set: translation re-fitted with R fixed on points nearer than
                                              # this (far monocular depth is compressed, which shrinks the scale)
    ground_rescale: bool = False     # rescale depth so that the road below the camera is at camera_height_m
    camera_height_m: float = 1.65    # a configuration constant (KITTI: documented 1.65 m), not a measurement
    scale: str = "pnp"               # "road": translation from dense flow of the road just ahead (plane-induced
                                     # homography with the PnP rotation and the camera height); PnP as fallback
    road_rows: Tuple[float, float] = (0.70, 0.98)
    road_cols: Tuple[float, float] = (0.30, 0.70)
    road_min_pixels: int = 400
    road_max_resid_px: float = 1.5
    stationary_px: float = 0.4       # median rotation-compensated parallax below this = stationary
    use_filter: bool = True
    q_accel: float = 4.0             # speed filter: white-noise acceleration density (m^2/s^3)
    q_yaw: float = 0.05              # yaw-rate filter: random-walk density (rad^2/s^3)
    meas_sigma_speed_abs: float = 0.15
    meas_sigma_speed_rel: float = 0.05
    meas_sigma_yaw: float = 0.01
    gate_chi2: float = 10.8          # 1 dof, 99.9%
    moving_speed: float = 0.3        # filtered speed above this (and parallax test failed) = moving
    sigma_scale: float = 1.0         # calibration multiplier for the reported sigmas (fitted on dev)


def relative_pose(p0: np.ndarray, p1: np.ndarray, depth0: np.ndarray, K: np.ndarray, params: PnPParams
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(R, t, inlier mask) with X_t = R X_{t-1} + t, or None. Metric scale from depth0 (at t-1)."""
    z = sample_depth(depth0, p0)
    good = np.isfinite(z) & (z >= params.min_depth) & (z <= params.max_depth)
    if good.sum() < params.min_inliers:
        return None
    obj = backproject(p0[good], z[good], K)
    img = p1[good].astype(np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=200,
                                             reprojectionError=params.reproj_px, confidence=0.999,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < params.min_inliers:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    if params.rotation == "essential":
        E, emask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
        if E is not None and E.shape == (3, 3):
            _, Re, _, _ = cv2.recoverPose(E, p0, p1, K, mask=emask)
            # metric translation for the fixed rotation: least squares on the PnP inliers. Without enough
            # translation the essential matrix is degenerate; keep the PnP solution if this one is implausible.
            t_e = _translation_given_rotation(obj[inl], img[inl], K, Re, tvec.ravel())
            if np.all(np.isfinite(t_e)) and np.linalg.norm(t_e) < 5.0 and \
                    np.degrees(np.arccos(np.clip((np.trace(Re.T @ R) - 1) / 2, -1, 1))) < 2.0:
                R, tvec = Re, t_e
    if params.trans_max_depth is not None:
        near = obj[inl, 2] <= params.trans_max_depth
        if near.sum() >= params.min_inliers // 2:
            tvec = _translation_given_rotation(obj[inl][near], img[inl][near], K, R, np.asarray(tvec).ravel())
    mask = np.zeros(len(p0), dtype=bool)
    mask[np.flatnonzero(good)[inl]] = True
    return R, np.asarray(tvec, dtype=np.float64).reshape(3), mask


_DIS = None


def road_translation(g0, g1, R, t_init, K, mask, params) -> Optional[Tuple[np.ndarray, float, int]]:
    """Metric translation from the road region: dense flow (DIS) of the road pixels just ahead, fitted
    with the homography H = K (R + t n^T / h) K^-1 of a level road plane (n = (0, 1, 0), y down) at the
    camera height h. R is fixed (from PnP); t is solved by robust (Huber) Gauss-Newton. Returns
    (t, median residual px, n pixels) or None."""
    global _DIS
    h, w = g0.shape
    r0, r1 = int(params.road_rows[0] * h), int(params.road_rows[1] * h)
    c0, c1 = int(params.road_cols[0] * w), int(params.road_cols[1] * w)
    r0 = max(r0, int(K[1, 2]) + 10)                      # below the horizon only
    if r1 - r0 < 10:
        return None
    m = 40                                               # crop with a margin for the motion
    y0, y1, x0, x1 = max(0, r0 - m), min(h, r1 + m), max(0, c0 - m), min(w, c1 + m)
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = _DIS.calc(np.ascontiguousarray(g0[y0:y1, x0:x1]), np.ascontiguousarray(g1[y0:y1, x0:x1]), None)
    vs, us = np.mgrid[r0:r1:3, c0:c1:3]
    keep = mask[vs, us] > 0
    vs, us = vs[keep], us[keep]
    if vs.size < params.road_min_pixels:
        return None
    f = flow[vs - y0, us - x0]
    x = np.stack([us, vs, np.ones_like(us)], 0).astype(np.float64)
    obs = np.stack([us + f[:, 0], vs + f[:, 1]], 1)
    Kinv = np.linalg.inv(K)
    n_over_h = np.array([0.0, 1.0 / params.camera_height_m, 0.0])
    ray = Kinv @ x                                       # (3, N)
    A = K @ R @ ray                                      # rotation part, (3, N)
    s = n_over_h @ ray                                   # n.x / h per pixel, (N,)
    t = np.array(t_init, dtype=np.float64)
    for _ in range(15):
        P = A + np.outer(K @ t, s)                       # K (R + t n^T/h) K^-1 x
        u, v = P[0] / P[2], P[1] / P[2]
        r = np.concatenate([u - obs[:, 0], v - obs[:, 1]])
        # d(u,v)/dt: P = A + (K t) s  ->  dP/dt_j = K[:, j] s
        J = np.zeros((2 * len(u), 3))
        for j in range(3):
            dP = np.outer(K[:, j], s)
            J[:len(u), j] = (dP[0] * P[2] - P[0] * dP[2]) / P[2] ** 2
            J[len(u):, j] = (dP[1] * P[2] - P[1] * dP[2]) / P[2] ** 2
        a = np.abs(r)
        wgt = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))     # Huber, delta = 1 px
        sw = np.sqrt(wgt)
        step, *_ = np.linalg.lstsq(J * sw[:, None], -r * sw, rcond=None)
        t += step
        if np.linalg.norm(step) < 1e-5:
            break
    resid = np.hypot(r[:len(u)], r[len(u):])
    med = float(np.median(resid))
    if not np.all(np.isfinite(t)) or med > params.road_max_resid_px:
        return None
    return t, med, int(len(u))


def road_height(depth: np.ndarray, K: np.ndarray, mask: np.ndarray, rows=(0.80, 1.0), cols=(0.35, 0.65),
                min_pixels: int = 200) -> float:
    """Median height below the camera of the (unmasked) road region just ahead, from the depth map."""
    h, w = depth.shape
    vs, us = np.mgrid[int(rows[0] * h):int(rows[1] * h), int(cols[0] * w):int(cols[1] * w)]
    ok = (mask[vs, us] > 0) & np.isfinite(depth[vs, us]) & (depth[vs, us] > 0.5) & (vs > K[1, 2] + 5)
    if ok.sum() < min_pixels:
        return float("nan")
    z, v = depth[vs, us][ok], vs[ok]
    return float(np.median((v - K[1, 2]) * z / K[1, 1]))


def _translation_given_rotation(obj, img, K, R, t0):
    """Gauss-Newton on t only, minimising reprojection error with R fixed."""
    t = np.array(t0, dtype=np.float64)
    Xr = obj @ R.T
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for _ in range(10):
        X = Xr + t
        z = X[:, 2]
        u, v = fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy
        r = np.concatenate([u - img[:, 0], v - img[:, 1]])
        J = np.zeros((2 * len(z), 3))
        J[:len(z), 0], J[:len(z), 2] = fx / z, -fx * X[:, 0] / z ** 2
        J[len(z):, 1], J[len(z):, 2] = fy / z, -fy * X[:, 1] / z ** 2
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        t += step
        if np.linalg.norm(step) < 1e-6:
            break
    return t


class _Scalar2:
    """Constant-velocity filter on one scalar (state [x, xdot]); used for speed."""

    def __init__(self, x: float, var: float, t: float, q: float) -> None:
        self.s, self.P, self.t, self.q = np.array([x, 0.0]), np.diag([var, 25.0]), t, q

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt > 0:
            F = np.array([[1, dt], [0, 1]])
            Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            self.s, self.P, self.t = F @ self.s, F @ self.P @ F.T + Q, t

    def update(self, z: float, var: float, gate: float) -> bool:
        S = self.P[0, 0] + var
        y = z - self.s[0]
        if y * y / S > gate:
            return False
        K = self.P[:, 0] / S
        self.s = self.s + K * y
        self.P = self.P - np.outer(K, self.P[0, :])
        return True


class _Scalar1:
    """Random-walk filter on one scalar; used for yaw rate."""

    def __init__(self, x: float, var: float, t: float, q: float) -> None:
        self.x, self.P, self.t, self.q = x, var, t, q

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt > 0:
            self.P += self.q * dt
            self.t = t

    def update(self, z: float, var: float, gate: float) -> bool:
        S = self.P + var
        if (z - self.x) ** 2 / S > gate:
            return False
        K = self.P / S
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return True


class PnPEgoMotion:
    """See module docstring. ``process(frame)`` takes a velocity_eval.CachedFrame (gray(), depth(),
    objects, timestamp, camera) and returns an EgoEstimate for that frame."""

    needs = ("gray", "depth")

    def __init__(self, params: PnPParams = PnPParams()) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.prev = None                    # (t, gray, depth, boxes)
        self.last_motion = None             # (R, t, dt) of the previous frame pair, to seed KLT
        self.speed_f: Optional[_Scalar2] = None
        self.yaw_f: Optional[_Scalar1] = None
        self.n_gated = 0

    def measure(self, t0, g0, d0, boxes0, t1, g1, boxes1, K) -> EgoEstimate:
        """One frame pair, no temporal filtering."""
        dt = t1 - t0
        mask = background_mask(g0.shape, boxes0)
        predict = None
        if self.p.predict_flow and self.last_motion is not None:
            Rp, tp, dtp = self.last_motion
            tp = tp * (dt / dtp)

            def predict(pts, Rp=Rp, tp=tp):
                z = sample_depth(d0, pts)
                z = np.where(np.isfinite(z) & (z > 0.5), z, 80.0)
                X = backproject(pts, z, K) @ Rp.T + tp
                X[:, 2] = np.maximum(X[:, 2], 0.5)
                return np.stack([K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]], 1)
        p0, p1 = track_features(g0, g1, mask, fb_max_px=self.p.fb_max_px, win=self.p.lk_win, levels=self.p.lk_levels,
                                predict=predict)
        if len(p0):
            keep = background_mask(g1.shape, boxes1)[np.clip(p1[:, 1].astype(int), 0, g1.shape[0] - 1),
                                                     np.clip(p1[:, 0].astype(int), 0, g1.shape[1] - 1)] > 0
            p0, p1 = p0[keep], p1[keep]
        if len(p0) < self.p.min_inliers:
            self.last_motion = None
            return EgoEstimate(False, reason="too few background features")
        if self.p.ground_rescale:
            hr = road_height(d0, K, mask)
            if np.isfinite(hr) and 0.5 * self.p.camera_height_m < hr < 2.0 * self.p.camera_height_m:
                d0 = d0 * (self.p.camera_height_m / hr)
        res = relative_pose(p0, p1, d0, K, self.p)
        if res is None:
            self.last_motion = None
            return EgoEstimate(False, n_inliers=0, reason="pose estimation failed")
        R, tvec, inl = res
        if self.p.scale == "road":
            rt = road_translation(g0, g1, R, tvec, K, background_mask(g1.shape, boxes0 + list(boxes1)), self.p)
            if rt is not None:
                tvec = rt[0]
        self.last_motion = (R, tvec, dt)
        c, Rc = camera_motion(R, tvec)
        par = float(np.median(rotation_parallax(p0[inl], p1[inl], R, K)))
        stationary = par < self.p.stationary_px
        speed = 0.0 if stationary else float(c[2] / dt)
        return EgoEstimate(True, speed=speed, yaw_rate=yaw_about_y(Rc) / dt, moving=not stationary,
                           n_inliers=int(inl.sum()), parallax_px=par)

    def process(self, frame) -> EgoEstimate:
        t, g, d = frame.timestamp, frame.gray(), frame.depth()
        boxes = [o.box for o in frame.objects]
        K = frame.camera.matrix()
        prev, self.prev = self.prev, (t, g, d, boxes)
        if prev is None or not 0 < t - prev[0] <= MAX_FRAME_GAP_S:
            self.speed_f = self.yaw_f = None
            self.last_motion = None
            return EgoEstimate(False, reason="no consecutive previous frame")
        m = self.measure(prev[0], prev[1], prev[2], prev[3], t, g, boxes, K)
        if not self.p.use_filter:
            return m
        return self._filter(t, m)

    def _filter(self, t: float, m: EgoEstimate) -> EgoEstimate:
        p = self.p
        if self.speed_f is not None:
            self.speed_f.predict(t)
            self.yaw_f.predict(t)
        if m.valid:
            sv2 = (0.05 if not m.moving else np.hypot(p.meas_sigma_speed_abs, p.meas_sigma_speed_rel * m.speed)) ** 2
            sy2 = p.meas_sigma_yaw ** 2
            if self.speed_f is None:
                self.speed_f = _Scalar2(m.speed, sv2, t, p.q_accel)
                self.yaw_f = _Scalar1(m.yaw_rate, sy2, t, p.q_yaw)
            else:
                if not self.speed_f.update(m.speed, sv2, p.gate_chi2):
                    self.n_gated += 1
                if not self.yaw_f.update(m.yaw_rate, sy2, p.gate_chi2):
                    self.n_gated += 1
        if self.speed_f is None:
            return m
        speed, yaw = float(self.speed_f.s[0]), float(self.yaw_f.x)
        moving = bool(m.moving) if m.valid else abs(speed) > p.moving_speed
        if moving and abs(speed) < p.moving_speed:
            moving = False
        return EgoEstimate(True, speed=speed, yaw_rate=yaw, moving=moving,
                           sigma_speed=p.sigma_scale * float(np.sqrt(self.speed_f.P[0, 0])),
                           sigma_yaw=p.sigma_scale * float(np.sqrt(self.yaw_f.P)),
                           n_inliers=m.n_inliers, parallax_px=m.parallax_px,
                           reason=m.reason if not m.valid else "")


class FlowDepthBaseline:
    """Simple baseline: yaw = -atan(median horizontal flow / fx) of far background points (depth > 30 m,
    where translation hardly moves them), speed = median depth decrease of near background points
    (3-30 m), stationary if |speed| < 0.5 m/s. No pose estimation, no filtering."""

    needs = ("gray", "depth")

    def __init__(self, far_m: float = 30.0, near: Tuple[float, float] = (3.0, 30.0), stationary_speed: float = 0.5):
        self.far_m, self.near, self.stationary_speed = far_m, near, stationary_speed
        self.reset()

    def reset(self) -> None:
        self.prev = None

    def process(self, frame) -> EgoEstimate:
        t, g, d = frame.timestamp, frame.gray(), frame.depth()
        boxes = [o.box for o in frame.objects]
        K = frame.camera.matrix()
        prev, self.prev = self.prev, (t, g, d, boxes)
        if prev is None or not 0 < t - prev[0] <= MAX_FRAME_GAP_S:
            return EgoEstimate(False, reason="no consecutive previous frame")
        dt = t - prev[0]
        p0, p1 = track_features(prev[1], g, background_mask(g.shape, prev[3]))
        if len(p0) < 10:
            return EgoEstimate(False, reason="too few background features")
        z0, z1 = sample_depth(prev[2], p0), sample_depth(d, p1)
        far = z0 > self.far_m
        near = (z0 >= self.near[0]) & (z0 <= self.near[1]) & (z1 > 0)
        if far.sum() < 5 or near.sum() < 10:
            return EgoEstimate(False, reason="too few far or near points")
        yaw = -np.arctan(np.median(p1[far, 0] - p0[far, 0]) / K[0, 0]) / dt
        speed = float(np.median(z0[near] - z1[near]) / dt)
        moving = abs(speed) >= self.stationary_speed
        return EgoEstimate(True, speed=speed if moving else 0.0, yaw_rate=float(yaw), moving=moving,
                           n_inliers=int(near.sum()))


def build_ego_estimators(names, params: Optional[dict] = None):
    """Named estimators; ``params`` may also define extra variants as {name: PnPParams kwargs} (dev sweeps)."""
    params = params or {}
    registry = {n: (lambda kw=kw: PnPEgoMotion(PnPParams(**kw))) for n, kw in params.items()
                if n.startswith("SWEEP_")}
    registry.update({
        "EM0_flow_depth_baseline": lambda: FlowDepthBaseline(),
        "EM1_pnp_raw": lambda: PnPEgoMotion(PnPParams(use_filter=False, **params.get("EM1_pnp_raw", {}))),
        "EM1_pnp_essential_raw": lambda: PnPEgoMotion(PnPParams(use_filter=False, rotation="essential")),
        "EM2_pnp_filtered": lambda: PnPEgoMotion(PnPParams(**params.get("EM2_pnp_filtered", {}))),
    })
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown ego estimators {unknown}; known: {sorted(registry)}")
    return {n: registry[n]() for n in names}
