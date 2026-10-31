"""EXPERIMENTAL velocity estimators for the offline KITTI evaluation (Task 5, Stage 2).

Nothing in the live pipeline imports this module. Each component is a separate,
switchable piece so that its contribution can be measured on its own against B0:

* ``robust_object_depth``  -- depth from a larger interior region of the box with
  MAD outlier rejection, instead of B0's 5x5 window at the box's bottom centre.
* ``DepthConsistencyGate`` -- rejects a depth sample that is physically
  inconsistent with the track's recent depths (replaced by the prediction).
* ``ObjectKalmanFilter``   -- time-aware constant-velocity Kalman filter on
  (X, Z) with Mahalanobis gating, coasting over missed frames.
* ``klt_scale``            -- KLT feature tracking inside the box with a
  forward-backward consistency check; returns the image scale change.
  Scale change gives closing speed (Z (s - 1) / dt) and TTC (dt / (s - 1))
  without differencing two noisy depths.
* uncertainty: the Kalman variants and the scale cue report sigma_vz; a
  ``sigma_gate`` marks estimates with larger uncertainty as invalid.

Adapters follow velocity_eval's interface: ``reset()``, ``process(frame)``.
All parameters are defaults chosen before tuning or tuned on the dev split only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from velocity_eval import CachedFrame, CachedObject, VelocityEstimate

MAX_CLOSING_SPEED_M_S = 45.0       # physical bound for |dZ/dt| between a car and anything on the road


# =============================================================================
# Component 1: robust object depth
# =============================================================================


@dataclass(frozen=True)
class RobustDepthParams:
    cols: Tuple[float, float] = (0.2, 0.8)       # central part of the box width
    rows: Tuple[float, float] = (0.3, 0.85)      # skip the top (sky/background) and the bottom (road)
    percentile: float = 50.0                     # of the inliers; < 50 favours the nearer surface
    mad_k: float = 2.5
    min_pixels: int = 20
    rel_sigma_floor: float = 0.05                # measurement sigma floor as a fraction of Z
    vru_cols: Optional[Tuple[float, float]] = None   # narrower columns for person/bicycle boxes, if set


VRU_CLASSES = ("person", "bicycle", "motorcycle")


def robust_object_depth(depth: np.ndarray, box, params: RobustDepthParams = RobustDepthParams(),
                        class_name: str = "") -> Tuple[float, float, int]:
    """(Z, sigma_Z, n_inliers) of an object from the depth map; Z = nan if not measurable."""
    h, w = depth.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cols = params.vru_cols if params.vru_cols is not None and class_name in VRU_CLASSES else params.cols
    c0, c1 = int(max(0, x1 + cols[0] * bw)), int(min(w, np.ceil(x1 + cols[1] * bw)))
    r0, r1 = int(max(0, y1 + params.rows[0] * bh)), int(min(h, np.ceil(y1 + params.rows[1] * bh)))
    if c1 - c0 < 2 or r1 - r0 < 2:
        return np.nan, np.nan, 0
    v = depth[r0:r1, c0:c1].ravel()
    v = v[np.isfinite(v) & (v > 0.1)]
    if v.size < params.min_pixels:
        return np.nan, np.nan, int(v.size)
    for _ in range(2):
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
        inl = v[np.abs(v - med) <= params.mad_k * mad]
        if inl.size < params.min_pixels:
            break
        v = inl
    z = float(np.percentile(v, params.percentile))
    spread = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return z, max(spread, params.rel_sigma_floor * z), int(v.size)


def lateral_position(u: float, z: float, fx: float, cx: float) -> float:
    return (u - cx) * z / fx


# =============================================================================
# Component 2: temporal consistency of depth
# =============================================================================


@dataclass
class DepthConsistencyGate:
    """Accept a depth sample only if |Z - Z_pred| <= v_max*dt + k*sigma; otherwise return the prediction.

    The prediction extrapolates the last two accepted samples (velocity clipped
    to +-v_max). After ``max_rejects`` consecutive rejections the next sample is
    accepted, so a real jump (e.g. a re-associated track) cannot lock the gate."""
    k: float = 3.0
    rel_sigma: float = 0.05
    v_max: float = MAX_CLOSING_SPEED_M_S
    max_rejects: int = 2
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2))
    rejects: int = 0
    n_rejected: int = 0

    def __call__(self, t: float, z: float) -> Tuple[float, bool]:
        if not self.history:
            self.history.append((t, z))
            return z, True
        t1, z1 = self.history[-1]
        v = 0.0
        if len(self.history) == 2:
            t0, z0 = self.history[0]
            v = float(np.clip((z1 - z0) / max(t1 - t0, 1e-6), -self.v_max, self.v_max))
        dt = max(t - t1, 1e-6)
        pred = z1 + v * dt
        tol = self.v_max * dt + self.k * self.rel_sigma * max(pred, 1.0) * np.sqrt(2.0)
        if abs(z - pred) <= tol or self.rejects >= self.max_rejects:
            self.rejects = 0
            self.history.append((t, z))
            return z, True
        self.rejects += 1
        self.n_rejected += 1
        return pred, False


# =============================================================================
# Component 3: time-aware gated Kalman filter
# =============================================================================


@dataclass(frozen=True)
class KalmanParams:
    q_lon: float = 4.0            # white-noise acceleration spectral density along Z (m^2/s^3)
    q_lat: float = 1.0            # ... along X
    sigma_v0: float = 10.0        # prior sigma of a new track's velocity (m/s)
    gate_chi2: float = 13.8       # 2-dof chi-square, 99.9%
    max_rejects: int = 3          # consecutive gated measurements before re-initialising
    max_coast_s: float = 0.5      # forget a track not measured for this long
    min_updates: int = 2          # same minimum as B0 (two samples)
    pixel_sigma: float = 2.0      # lateral image-position noise (px)


class ObjectKalmanFilter:
    """State [X, Z, VX, VZ] in the camera frame; F and Q depend on the actual dt."""

    def __init__(self, t: float, x: float, z: float, R: np.ndarray, params: KalmanParams) -> None:
        self.p = params
        self.t = t
        self.s = np.array([x, z, 0.0, 0.0])
        self.P = np.diag([R[0, 0], R[1, 1], params.sigma_v0 ** 2, params.sigma_v0 ** 2])
        self.n_updates, self.rejects, self.n_rejected = 1, 0, 0

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = self.p.q_lat * q
        Q[np.ix_([1, 3], [1, 3])] = self.p.q_lon * q
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _update(self, H: np.ndarray, zvec: np.ndarray, R: np.ndarray, gate: bool) -> bool:
        y = zvec - H @ self.s
        S = H @ self.P @ H.T + R
        if gate and float(y @ np.linalg.solve(S, y)) > self.p.gate_chi2:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T          # Joseph form
        return True

    def update_position(self, x: float, z: float, R: np.ndarray) -> bool:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        ok = self._update(H, np.array([x, z]), R, gate=self.n_updates >= 2)
        if ok:
            self.n_updates += 1
            self.rejects = 0
        else:
            self.rejects += 1
            self.n_rejected += 1
        return ok

    def update_closing_speed(self, vz: float, sigma: float) -> bool:
        """A direct measurement of VZ (from the scale cue)."""
        H = np.zeros((1, 4))
        H[0, 3] = 1.0
        return self._update(H, np.array([vz]), np.array([[sigma ** 2]]), gate=self.n_updates >= 2)


def position_noise(z: float, sigma_z: float, u: float, fx: float, cx: float, pixel_sigma: float) -> np.ndarray:
    """R for (X, Z) with X = (u - cx) Z / fx: depth error propagates into X, plus pixel noise."""
    sx2 = ((u - cx) / fx * sigma_z) ** 2 + (pixel_sigma * z / fx) ** 2
    return np.diag([sx2, sigma_z ** 2])


# =============================================================================
# Component 4: KLT scale change with forward-backward check
# =============================================================================


@dataclass(frozen=True)
class KLTParams:
    shrink: float = 0.1            # ignore this fraction of the box on each side (background)
    max_corners: int = 80
    quality: float = 0.01
    min_distance: int = 3
    win: int = 15
    levels: int = 3
    fb_max_px: float = 1.0         # forward-backward error limit; inf disables the check
    min_points: int = 6
    min_box_px: float = 16.0       # skip boxes smaller than this in either dimension
    min_pair_px: float = 6.0
    max_pairs: int = 400
    axis: str = "both"             # "vertical": scale from vertical separations only (less affected by
                                   # side faces and perspective of off-axis objects)


@dataclass
class ScaleMeasurement:
    scale: float                   # new size / old size (> 1 = approaching)
    sigma: float
    n_points: int
    u_old: float                   # reference point (median of tracked points), old and new image x
    u_new: float


_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def klt_scale(prev_gray: np.ndarray, gray: np.ndarray, prev_box, box, params: KLTParams = KLTParams(),
              rng: Optional[np.random.Generator] = None) -> Optional[ScaleMeasurement]:
    x1, y1, x2, y2 = prev_box
    bw, bh = x2 - x1, y2 - y1
    if bw < params.min_box_px or bh < params.min_box_px:
        return None
    h, w = gray.shape
    # Work on a crop around both boxes (with room for the pyramid and the motion) rather than the
    # whole image: corner detection and LK pyramids then cost per object, not per image.
    nx1, ny1, nx2, ny2 = box
    margin = int(0.3 * max(bw, bh, nx2 - nx1, ny2 - ny1)) + params.win * 2 ** params.levels // 2
    cx1, cy1 = int(max(0, min(x1, nx1) - margin)), int(max(0, min(y1, ny1) - margin))
    cx2, cy2 = int(min(w, max(x2, nx2) + margin)), int(min(h, max(y2, ny2) + margin))
    g0, g1 = prev_gray[cy1:cy2, cx1:cx2], gray[cy1:cy2, cx1:cx2]
    off = np.array([cx1, cy1], dtype=np.float32)
    mask = np.zeros_like(g0)
    mx1, mx2 = int(max(0, x1 + params.shrink * bw)) - cx1, int(min(w, x2 - params.shrink * bw)) - cx1
    my1, my2 = int(max(0, y1 + params.shrink * bh)) - cy1, int(min(h, y2 - params.shrink * bh)) - cy1
    if mx2 - mx1 < 4 or my2 - my1 < 4:
        return None
    mask[my1:my2, mx1:mx2] = 255
    p0 = cv2.goodFeaturesToTrack(g0, params.max_corners, params.quality, params.min_distance,
                                 mask=mask, blockSize=5)
    if p0 is None or len(p0) < params.min_points:
        return None
    lk = dict(winSize=(params.win, params.win), maxLevel=params.levels, criteria=_LK_CRITERIA)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    ok = st1.ravel() == 1
    if np.isfinite(params.fb_max_px):
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
        fb = np.linalg.norm((p0r - p0).reshape(-1, 2), axis=1)
        ok &= (st2.ravel() == 1) & (fb <= params.fb_max_px)
    a, b = p0.reshape(-1, 2)[ok] + off, p1.reshape(-1, 2)[ok] + off
    pad = 0.1 * max(nx2 - nx1, ny2 - ny1)
    inside = (b[:, 0] >= nx1 - pad) & (b[:, 0] <= nx2 + pad) & (b[:, 1] >= ny1 - pad) & (b[:, 1] <= ny2 + pad)
    a, b = a[inside], b[inside]
    n = len(a)
    if n < params.min_points:
        return None
    i, j = np.triu_indices(n, k=1)
    if len(i) > params.max_pairs:
        pick = (rng or np.random.default_rng(0)).choice(len(i), params.max_pairs, replace=False)
        i, j = i[pick], j[pick]
    if params.axis == "vertical":
        d0, d1 = np.abs(a[i, 1] - a[j, 1]), np.abs(b[i, 1] - b[j, 1])
    else:
        d0, d1 = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    keep = d0 >= params.min_pair_px
    if keep.sum() < params.min_points:
        return None
    r = d1[keep] / d0[keep]
    s = float(np.median(r))
    mad = 1.4826 * float(np.median(np.abs(r - s)))
    return ScaleMeasurement(scale=s, sigma=max(mad, 1e-4) / np.sqrt(n), n_points=n,
                            u_old=float(np.median(a[:, 0])), u_new=float(np.median(b[:, 0])))


# =============================================================================
# Adapters
# =============================================================================


def _camera(frame: CachedFrame):
    cam = frame.camera
    if cam is None:
        raise RuntimeError("experimental adapters need frame.camera (set by the harness)")
    return cam


def _box_centre_u(box) -> float:
    return 0.5 * (box[0] + box[2])


class _B0Core:
    """B0's own velocity logic (velocity_b0.RelativeVelocityEstimator), fed modified positions --
    so a depth component's effect is measured with everything else held equal to B0."""

    def __init__(self, vcfg: dict) -> None:
        from velocity_b0 import RelativeVelocityEstimator
        self.est = RelativeVelocityEstimator(history_length=vcfg["history_length"],
                                             max_gap_seconds=vcfg["max_gap_seconds"], ema_alpha=vcfg["ema_alpha"])

    def update(self, objs, t):
        return self.est.update(objs, timestamp=t)


@dataclass
class _Pos:
    track_id: int
    x_m: float
    z_m: float


class RobustDepthAdapter:
    """Component D: B0's velocity logic with robust object depth (and X from the box centre)."""
    needs = ("depth",)

    def __init__(self, vcfg: dict, params: RobustDepthParams = RobustDepthParams()) -> None:
        self.vcfg, self.params = vcfg, params

    def reset(self) -> None:
        self.core = _B0Core(self.vcfg)

    def process(self, frame: CachedFrame) -> Dict[int, VelocityEstimate]:
        cam, depth = _camera(frame), frame.depth()
        objs, zs = [], {}
        for o in frame.objects:
            if o.track_id is None:
                continue
            z, _, _ = robust_object_depth(depth, o.box, self.params, o.class_name)
            if np.isfinite(z):
                objs.append(_Pos(o.track_id, lateral_position(_box_centre_u(o.box), z, cam.fx, cam.cx), z))
                zs[o.track_id] = z
        out = self.core.update(objs, frame.timestamp)
        return {tid: VelocityEstimate(v.vx_m_s, v.vz_m_s, v.valid, z_m=zs[tid]) for tid, v in out.items()}


class DepthGateAdapter:
    """Component T: B0's velocity logic and B0's depth, with the temporal consistency gate on Z."""

    def __init__(self, vcfg: dict, k: float = 3.0, rel_sigma: float = 0.05) -> None:
        self.vcfg, self.k, self.rel_sigma = vcfg, k, rel_sigma

    def reset(self) -> None:
        self.core = _B0Core(self.vcfg)
        self.gates: Dict[int, DepthConsistencyGate] = {}
        self.last_seen: Dict[int, float] = {}
        self.n_rejected = 0

    def process(self, frame: CachedFrame) -> Dict[int, VelocityEstimate]:
        t = frame.timestamp
        objs = []
        for o in frame.bev_objects:
            if o.track_id is None:
                objs.append(o)
                continue
            # B0 forgets a track after one missing frame; the gate follows the same rule.
            if self.last_seen.get(o.track_id) is None or t - self.last_seen[o.track_id] > 0.15:
                self.gates[o.track_id] = DepthConsistencyGate(k=self.k, rel_sigma=self.rel_sigma)
            self.last_seen[o.track_id] = t
            z, accepted = self.gates[o.track_id](t, o.z_m)
            self.n_rejected += not accepted
            x = o.x_m if accepted else o.x_m * z / o.z_m          # same bearing, gated range
            objs.append(_Pos(o.track_id, x, z))
        out = self.core.update(objs, t)
        return {tid: VelocityEstimate(v.vx_m_s, v.vz_m_s, v.valid) for tid, v in out.items()}


class ScaleAdapter:
    """Component S: closing speed and TTC from KLT scale change, with B0's depth for Z and B0's
    temporal structure (span of up to history_length-1 frame pairs, then EMA). TTC = Z / closing
    = span / (s_span - 1) does not depend on depth at all."""
    needs = ("gray",)

    def __init__(self, vcfg: dict, params: KLTParams = KLTParams(), use_robust_depth: bool = False,
                 sigma_scale: float = 1.0) -> None:
        self.vcfg, self.params, self.use_robust_depth = vcfg, params, use_robust_depth
        self.sigma_scale = sigma_scale            # calibration multiplier, fitted on dev
        if use_robust_depth:
            self.needs = ("gray", "depth")

    def reset(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None
        self.prev_boxes: Dict[int, tuple] = {}
        self.pairs: Dict[int, Deque] = {}
        self.smoothed: Dict[int, Tuple[float, float]] = {}
        self.rng = np.random.default_rng(0)
        self.n_measured = self.n_failed = 0

    def _depth(self, frame, o: CachedObject) -> float:
        if self.use_robust_depth:
            return robust_object_depth(frame.depth(), o.box, class_name=o.class_name)[0]
        return o.z_m if o.has_bev else np.nan

    def process(self, frame: CachedFrame) -> Dict[int, VelocityEstimate]:
        cam, t, gray = _camera(frame), frame.timestamp, frame.gray()
        span = self.vcfg["history_length"] - 1
        alpha = self.vcfg["ema_alpha"]
        out: Dict[int, VelocityEstimate] = {}
        boxes = {}
        consecutive = self.prev_t is not None and 0 < t - self.prev_t <= 0.15
        for o in frame.objects:
            if o.track_id is None:
                continue
            tid = o.track_id
            boxes[tid] = o.box
            if not consecutive or tid not in self.prev_boxes:
                self.pairs.pop(tid, None)
                self.smoothed.pop(tid, None)
            else:
                m = klt_scale(self.prev_gray, gray, self.prev_boxes[tid], o.box, self.params, self.rng)
                if m is None:
                    self.n_failed += 1
                    self.pairs.pop(tid, None)
                    self.smoothed.pop(tid, None)
                else:
                    self.n_measured += 1
                    self.pairs.setdefault(tid, deque(maxlen=max(span, 1))).append((t - self.prev_t, m))
            z = self._depth(frame, o)
            hist = self.pairs.get(tid)
            if not hist or not np.isfinite(z):
                out[tid] = VelocityEstimate(0.0, 0.0, False)
                continue
            T = sum(dt for dt, _ in hist)
            log_s = sum(np.log(m.scale) for _, m in hist)
            s_span = float(np.exp(log_s))
            closing = z * (s_span - 1.0) / T
            # lateral: X_t - X_{t-1} with the same Z (Z_{t-1} = s Z_t), averaged over the span
            vx_raw = float(np.mean([((m.u_new - cam.cx) - m.scale * (m.u_old - cam.cx)) * z / cam.fx / dt
                                    for dt, m in hist]))
            vz_raw = -closing
            if tid in self.smoothed:
                pvx, pvz = self.smoothed[tid]
                vx, vz = alpha * vx_raw + (1 - alpha) * pvx, alpha * vz_raw + (1 - alpha) * pvz
            else:
                vx, vz = vx_raw, vz_raw
            self.smoothed[tid] = (vx, vz)
            sig_log = float(np.sqrt(sum((m.sigma / m.scale) ** 2 for _, m in hist)))
            sigma = self.sigma_scale * float(np.hypot(z * s_span * sig_log / T, 0.05 * abs(closing)))
            out[tid] = VelocityEstimate(vx, vz, True, sigma_vz=sigma, z_m=z if self.use_robust_depth else np.nan)
        self.prev_gray, self.prev_t, self.prev_boxes = gray, t, boxes
        return out


class KalmanAdapter:
    """Component K (and combinations): time-aware gated Kalman filter per track.

    depth="b0" uses the pipeline's BEV positions; depth="robust" uses robust_object_depth.
    use_scale=True adds the KLT scale cue as a direct VZ measurement. sigma_gate (m/s), when
    set, marks estimates whose sigma_vz exceeds it as invalid (component U)."""

    def __init__(self, vcfg: dict, depth: str = "b0", use_scale: bool = False, sigma_gate: Optional[float] = None,
                 params: KalmanParams = KalmanParams(), rel_sigma_b0: float = 0.08,
                 depth_params: RobustDepthParams = RobustDepthParams(), klt: KLTParams = KLTParams(),
                 sigma_scale: float = 1.0, scale_sigma_mult: float = 1.0) -> None:
        self.vcfg, self.depth_mode, self.use_scale, self.sigma_gate = vcfg, depth, use_scale, sigma_gate
        self.p, self.rel_sigma_b0, self.depth_params, self.klt = params, rel_sigma_b0, depth_params, klt
        self.sigma_scale = sigma_scale            # output calibration multiplier, fitted on dev
        self.scale_sigma_mult = scale_sigma_mult  # inflation of the scale cue's own sigma inside the filter
        needs = []
        if depth == "robust":
            needs.append("depth")
        if use_scale:
            needs.append("gray")
        self.needs = tuple(needs)

    def reset(self) -> None:
        self.filters: Dict[int, ObjectKalmanFilter] = {}
        self.prev_gray, self.prev_t, self.prev_boxes = None, None, {}
        self.rng = np.random.default_rng(0)
        self.n_gated = 0

    def _measure(self, frame, o: CachedObject, cam):
        u = _box_centre_u(o.box)
        if self.depth_mode == "robust":
            z, sz, _ = robust_object_depth(frame.depth(), o.box, self.depth_params, o.class_name)
            if not np.isfinite(z):
                return None
            x = lateral_position(u, z, cam.fx, cam.cx)
        else:
            if not o.has_bev:
                return None
            z, x = o.z_m, o.x_m
            sz = self.rel_sigma_b0 * z
            u = cam.cx + x * cam.fx / z
        return x, z, position_noise(z, sz, u, cam.fx, cam.cx, self.p.pixel_sigma)

    def process(self, frame: CachedFrame) -> Dict[int, VelocityEstimate]:
        cam, t = _camera(frame), frame.timestamp
        gray = frame.gray() if self.use_scale else None
        consecutive = self.prev_t is not None and 0 < t - self.prev_t <= 0.15
        out: Dict[int, VelocityEstimate] = {}
        boxes = {}
        for tid in [k for k, f in self.filters.items() if t - f.t > self.p.max_coast_s]:
            del self.filters[tid]
        for o in frame.objects:
            if o.track_id is None:
                continue
            tid = o.track_id
            boxes[tid] = o.box
            meas = self._measure(frame, o, cam)
            kf = self.filters.get(tid)
            if meas is None:
                if kf is not None:
                    kf.predict(t)
            else:
                x, z, R = meas
                if kf is None or kf.rejects >= self.p.max_rejects:
                    kf = self.filters[tid] = ObjectKalmanFilter(t, x, z, R, self.p)
                else:
                    kf.predict(t)
                    if not kf.update_position(x, z, R):
                        self.n_gated += 1
                if self.use_scale and consecutive and tid in self.prev_boxes:
                    m = klt_scale(self.prev_gray, gray, self.prev_boxes[tid], o.box, self.klt, self.rng)
                    if m is not None:
                        dt = t - self.prev_t
                        zc = max(kf.s[1], 0.5)
                        closing = zc * (m.scale - 1.0) / dt
                        sigma = self.scale_sigma_mult * np.hypot(zc * m.sigma / dt, 0.1 * abs(closing)) + 0.05
                        kf.update_closing_speed(-closing, sigma)
            if kf is None:
                out[tid] = VelocityEstimate(0.0, 0.0, False)
                continue
            sig = self.sigma_scale * float(np.sqrt(max(kf.P[3, 3], 0.0)))
            valid = kf.n_updates >= self.p.min_updates and kf.t == t
            if self.sigma_gate is not None and sig > self.sigma_gate:
                valid = False
            out[tid] = VelocityEstimate(float(kf.s[2]), float(kf.s[3]), valid, sigma_vz=sig,
                                        z_m=float(kf.s[1]) if self.depth_mode == "robust" else np.nan)
        if self.use_scale:
            self.prev_gray, self.prev_t, self.prev_boxes = gray, t, boxes
        else:
            self.prev_t = t
        return out


# Stage 2 configurations, frozen after the dev sweeps (selection on pooled dev; 2-fold cross-validation
# inside dev in the Stage 2 report). The sigma multipliers are set by the calibration pass (CALIBRATION).
TUNED_GATE = dict(k=1.5, rel_sigma=0.03)
TUNED_KALMAN = KalmanParams(q_lon=16.0)
TUNED_REL_SIGMA_B0 = 0.08
TUNED_KLT = KLTParams(axis="vertical", fb_max_px=0.5)
# sigma multipliers = median(|error| / sigma) / 0.6745 on pooled dev (2-fold check in the Stage 2 report);
# the scale cue's multiplier is also applied to its measurement noise inside the combined filters.
CALIBRATION = {"K_kalman": 0.525, "S_klt_scale": 2.59, "KS_kalman_scale": 0.751, "DKS_robust_kalman_scale": 0.798,
               "scale_cue_in_filter": 2.59, "sigma_gate": 2.35}   # m/s, pooled-dev choice: <= 5 points of coverage lost


def build_estimators(vcfg: dict, names, calibration: Optional[dict] = None) -> Dict[str, object]:
    """The named Stage 2 estimators. Single components are measured with everything else as in B0:

    D_robust_depth   robust object depth, B0 velocity logic
    T_depth_gate     B0 depth + temporal consistency gate, B0 velocity logic
    K_kalman         B0 depth + time-aware gated Kalman filter (replaces B0's difference + EMA)
    S_klt_scale      KLT scale change (vertical, FB check) for closing speed, B0 depth for Z, B0-like smoothing
    ablations        S_klt_scale_noFB (no forward-backward check), S_klt_scale_both_axes (isotropic scale),
                     *_untuned (the pre-sweep defaults)
    combinations     KS_kalman_scale (Kalman + scale cue, B0 depth), DKS_robust_kalman_scale (+ robust depth),
                     KS_kalman_scale_gated (+ uncertainty gate)
    """
    from velocity_eval import B0Adapter
    cal = {**CALIBRATION, **(calibration or {})}
    ks = dict(depth="b0", use_scale=True, params=TUNED_KALMAN, rel_sigma_b0=TUNED_REL_SIGMA_B0, klt=TUNED_KLT,
              scale_sigma_mult=cal["scale_cue_in_filter"])
    registry = {
        "B0": lambda: B0Adapter(vcfg["history_length"], vcfg["max_gap_seconds"], vcfg["ema_alpha"]),
        "D_robust_depth": lambda: RobustDepthAdapter(vcfg),
        "T_depth_gate": lambda: DepthGateAdapter(vcfg, **TUNED_GATE),
        "T_depth_gate_untuned": lambda: DepthGateAdapter(vcfg),
        "K_kalman": lambda: KalmanAdapter(vcfg, depth="b0", params=TUNED_KALMAN, rel_sigma_b0=TUNED_REL_SIGMA_B0,
                                          sigma_scale=cal["K_kalman"]),
        "K_kalman_untuned": lambda: KalmanAdapter(vcfg, depth="b0"),
        "S_klt_scale": lambda: ScaleAdapter(vcfg, TUNED_KLT, sigma_scale=cal["S_klt_scale"]),
        "S_klt_scale_noFB": lambda: ScaleAdapter(vcfg, KLTParams(axis="vertical", fb_max_px=float("inf"))),
        "S_klt_scale_both_axes": lambda: ScaleAdapter(vcfg, KLTParams(axis="both", fb_max_px=0.5)),
        "S_klt_scale_untuned": lambda: ScaleAdapter(vcfg),
        "KS_kalman_scale": lambda: KalmanAdapter(vcfg, **ks, sigma_scale=cal["KS_kalman_scale"]),
        "DKS_robust_kalman_scale": lambda: KalmanAdapter(vcfg, **{**ks, "depth": "robust"},
                                                         sigma_scale=cal["DKS_robust_kalman_scale"]),
        "KS_kalman_scale_gated": lambda: KalmanAdapter(vcfg, **ks, sigma_scale=cal["KS_kalman_scale"],
                                                       sigma_gate=cal["sigma_gate"]),
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown estimators {unknown}; known: {sorted(registry)}")
    return {n: registry[n]() for n in names}
