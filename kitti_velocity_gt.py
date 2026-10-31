"""Ground-truth object and ego kinematics for the KITTI tracking training set.

Everything is expressed in the frame the pipeline measures in: cam2 (x right,
y down, z forward, metres) at the frame being evaluated. For a labelled object
at frame t, using the 2*half_window + 1 consecutive frames centred on t (all of
which must carry the same track):

* ``rel_v``  -- the least-squares slope of the object's cam2 position, each
  sample in *its own frame's* camera coordinates. This is the apparent motion
  that a camera-only estimator such as B0 measures (and what TTC is defined on):
  closing speed = -rel_vz.
* ``abs_v``  -- the slope after mapping every sample into cam2 at frame t with
  the OXTS ego poses: the object's velocity over the ground, expressed in the
  current camera axes. Parked cars should have abs_v ~ 0.
* ``ego_v``  -- the slope of the camera centre itself (mapped into cam2 at t).
* ``ego_yaw_rate`` -- rotation rate about the camera's y (down) axis.

For small rotations these satisfy rel_v = abs_v - ego_v - omega x p, which the
tests check on the real data as an internal consistency test of the transform
chain.

Velocities come from differencing hand-placed 3D annotations, not from a
measured object speed, so they carry label noise; the LSQ slope over 5 frames
(0.4 s) has about 2.2x lower noise than a +-1 frame central difference, which
is also reported for sensitivity analysis. Times are the sequence's
reconstructed timestamps (nominal 0.1 s spacing by default; see kitti_tracking.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from kitti_tracking import KittiTrackingSequence, TrackLabel, invert_rigid, transform_points
from ttc_calculator import compute_ttc

GT_HALF_WINDOW = 2
VEHICLE_TYPES = ("Car", "Van", "Truck")
VRU_TYPES = ("Pedestrian", "Cyclist")
CORRIDOR_HALF_WIDTH_M = 2.0


def lsq_slope(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares slope of y (n, d) against t (n,), per column."""
    tc = t - t.mean()
    return (tc[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()


def yaw_about_y(R: np.ndarray) -> float:
    """Rotation angle about the camera y axis of a (near-)y-rotation matrix."""
    return float(np.arctan2(R[0, 2], R[2, 2]))


@dataclass
class EgoKinematics:
    frame: int
    has_window: bool
    vx: float = np.nan              # camera velocity in cam2 at t (m/s)
    vy: float = np.nan
    vz: float = np.nan
    yaw_rate: float = np.nan        # rad/s about cam2 y (down): positive = turning right
    oxts_vf: float = np.nan         # OXTS forward speed (m/s), for comparison
    oxts_yaw_rate: float = np.nan   # -wz (OXTS wz is about the upward z axis)


@dataclass
class ObjectKinematics:
    sequence: str
    frame: int
    track_id: int
    type: str
    truncated: int
    occluded: int
    bbox: tuple
    x: float                        # bottom-face centre in cam2 (m)
    z: float
    z_near: float                   # smallest z of the 3D box corners (front surface for a car ahead)
    x_near_abs: float               # smallest |x| of the 3D box corners (lateral clearance)
    has_window: bool
    rel_vx: float = np.nan
    rel_vz: float = np.nan
    abs_vx: float = np.nan
    abs_vz: float = np.nan
    rel_vx_cd: float = np.nan       # +-1-frame central differences (sensitivity analysis)
    rel_vz_cd: float = np.nan
    fit_rms_m: float = np.nan       # RMS residual of the straight-line fit in (x, z): label-noise diagnostic

    @property
    def closing_speed(self) -> float:
        return -self.rel_vz

    @property
    def abs_speed(self) -> float:
        return float(np.hypot(self.abs_vx, self.abs_vz))

    @property
    def in_corridor(self) -> bool:
        return self.x_near_abs <= CORRIDOR_HALF_WIDTH_M

    @property
    def ttc(self) -> Optional[float]:
        """Ground-truth TTC with the pipeline's own definition (ttc_calculator.compute_ttc)."""
        if not self.has_window:
            return None
        return compute_ttc(self.z_near, self.closing_speed)


def _window(per_track: Dict[int, TrackLabel], frame: int, half: int) -> Optional[List[TrackLabel]]:
    frames = range(frame - half, frame + half + 1)
    if all(f in per_track for f in frames):
        return [per_track[f] for f in frames]
    return None


def ego_kinematics(seq: KittiTrackingSequence, half_window: int = GT_HALF_WINDOW) -> List[EgoKinematics]:
    t_all = seq.timestamps
    out = []
    o = seq.oxts
    for k in range(seq.n_frames):
        ego = EgoKinematics(frame=k, has_window=False, oxts_vf=float(o[k, 8]), oxts_yaw_rate=float(-o[k, 19]))
        if half_window <= k < seq.n_frames - half_window:
            idx = np.arange(k - half_window, k + half_window + 1)
            centres = np.array([seq.T_cam2(k, i)[:3, 3] for i in idx])
            yaws = np.array([yaw_about_y(seq.T_cam2(k, i)[:3, :3]) for i in idx])
            v = lsq_slope(t_all[idx], centres)
            ego.has_window = True
            ego.vx, ego.vy, ego.vz = (float(c) for c in v)
            ego.yaw_rate = float(lsq_slope(t_all[idx], yaws[:, None])[0])
        out.append(ego)
    return out


def object_kinematics(seq: KittiTrackingSequence, half_window: int = GT_HALF_WINDOW) -> List[ObjectKinematics]:
    """Kinematics for every non-DontCare label of the sequence (rows without a full window have NaN velocities)."""
    t_all = seq.timestamps
    offset = seq.calib.cam2_offset
    out: List[ObjectKinematics] = []
    for track_id, per_track in sorted(seq.tracks.items()):
        for frame, label in sorted(per_track.items()):
            p = np.asarray(label.location) + offset
            corners = label.corners_cam0rect() + offset
            row = ObjectKinematics(
                sequence=seq.name, frame=frame, track_id=track_id, type=label.type,
                truncated=label.truncated, occluded=label.occluded, bbox=label.bbox,
                x=float(p[0]), z=float(p[2]), z_near=float(corners[:, 2].min()),
                x_near_abs=0.0 if corners[:, 0].min() <= 0 <= corners[:, 0].max() else float(np.abs(corners[:, 0]).min()),
                has_window=False,
            )
            if frame - 1 in per_track and frame + 1 in per_track:
                d = (np.asarray(per_track[frame + 1].location) - np.asarray(per_track[frame - 1].location)) / (
                    t_all[frame + 1] - t_all[frame - 1])
                row.rel_vx_cd, row.rel_vz_cd = float(d[0]), float(d[2])
            win = _window(per_track, frame, half_window)
            if win is not None:
                idx = np.array([l.frame for l in win])
                rel = np.array([np.asarray(l.location) + offset for l in win])
                absolute = np.array([transform_points(seq.T_cam2(frame, l.frame), np.asarray(l.location) + offset)[0]
                                     for l in win])
                tt = t_all[idx]
                v_rel, v_abs = lsq_slope(tt, rel), lsq_slope(tt, absolute)
                resid = rel[:, [0, 2]] - (rel[:, [0, 2]].mean(axis=0) + np.outer(tt - tt.mean(), v_rel[[0, 2]]))
                row.has_window = True
                row.rel_vx, row.rel_vz = float(v_rel[0]), float(v_rel[2])
                row.abs_vx, row.abs_vz = float(v_abs[0]), float(v_abs[2])
                row.fit_rms_m = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
            out.append(row)
    return out


def rigid_motion_residual(row: ObjectKinematics, ego: EgoKinematics, seq: KittiTrackingSequence) -> np.ndarray:
    """rel_v - (abs_v - ego_v - omega x p) in the x-z plane; ~0 if the transform chain is consistent."""
    omega = np.array([0.0, ego.yaw_rate, 0.0])
    p = np.array([row.x, 0.0, row.z])
    predicted = np.array([row.abs_vx, 0.0, row.abs_vz]) - np.array([ego.vx, 0.0, ego.vz]) - np.cross(omega, p)
    return np.array([row.rel_vx - predicted[0], row.rel_vz - predicted[2]])


def is_primary_eligible(row: ObjectKinematics, min_box_height_px: float = 25.0,
                        z_range=(3.0, 60.0)) -> bool:
    """The pre-registered primary evaluation subset (see task5_evaluation_protocol.md, section 3)."""
    return (row.type in VEHICLE_TYPES and row.truncated == 0 and row.occluded <= 1 and row.has_window
            and z_range[0] <= row.z_near and row.z <= z_range[1]
            and (row.bbox[3] - row.bbox[1]) >= min_box_height_px)


def is_vru_eligible(row: ObjectKinematics, min_box_height_px: float = 25.0, z_range=(3.0, 60.0)) -> bool:
    """Secondary subset: pedestrians and cyclists under the same visibility and window rules."""
    return (row.type in VRU_TYPES and row.truncated == 0 and row.occluded <= 1 and row.has_window
            and z_range[0] <= row.z_near and row.z <= z_range[1]
            and (row.bbox[3] - row.bbox[1]) >= min_box_height_px)


def project_box(seq: KittiTrackingSequence, label: TrackLabel) -> Optional[np.ndarray]:
    """2D bounding rectangle (x1, y1, x2, y2) of the label's 3D box projected with P2, clipped to the image."""
    corners = label.corners_cam0rect()
    if (corners[:, 2] <= 0.1).any():
        return None
    hom = np.hstack([corners, np.ones((8, 1))]) @ seq.calib.P[2].T
    uv = hom[:, :2] / hom[:, 2:3]
    w, h = seq.image_size
    return np.array([max(uv[:, 0].min(), 0), max(uv[:, 1].min(), 0),
                     min(uv[:, 0].max(), w - 1), min(uv[:, 1].max(), h - 1)])


__all__ = [
    "GT_HALF_WINDOW", "VEHICLE_TYPES", "VRU_TYPES", "CORRIDOR_HALF_WIDTH_M", "EgoKinematics", "ObjectKinematics",
    "ego_kinematics", "object_kinematics", "rigid_motion_residual", "is_primary_eligible", "is_vru_eligible",
    "project_box", "lsq_slope", "yaw_about_y", "invert_rigid",
]
