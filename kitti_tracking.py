"""Reader for the KITTI multi-object tracking training set, with ego poses from OXTS.

Layout (data_tracking_{image_2,label_2,oxts,calib}.zip, extracted):

    <root>/image_02/<seq>/<frame:06d>.png   left colour camera, rectified; size differs by recording day
    <root>/label_02/<seq>.txt               17 columns per object per frame (devkit readme)
    <root>/oxts/<seq>.txt                   30 GPS/IMU fields per frame (KITTI raw-data devkit layout)
    <root>/calib/<seq>.txt                  P0-P3 (3x4), R_rect (3x3), Tr_velo_cam, Tr_imu_velo (3x4)

Coordinate frames (all right-handed):

* IMU / OXTS: x forward, y left, z up.
* Velodyne:   x forward, y left, z up (Tr_imu_velo maps IMU -> Velodyne).
* "cam0 rect": the rectified reference camera, x right, y down, z forward
  (R_rect @ Tr_velo_cam maps Velodyne -> cam0 rect). Label locations are in
  this frame.
* "cam2": the rectified left colour camera that took image_02. Same axes as
  cam0 rect, shifted by t2 = K2^-1 P2[:, 3] (about +6 cm in x). Pixel geometry
  of image_02 -- and therefore everything the pipeline measures -- is in this
  frame.

There are no timestamp files in the tracking benchmark. ``timestamps`` are
reconstructed as t = k * frame_period_s, by default the nominal 10 Hz
(0.1 s). The data say the true image spacing is longer: GPS displacement
over straight 20-frame spans against integrated OXTS forward speed gives
0.10355 s (all 15 sequences with such spans within 0.1030-0.1040 s), and the per-frame
displacement of parked cars in the (LiDAR-based) labels is uniform at that
spacing. The OXTS record of each frame is the 100 Hz sample nearest the image,
so consecutive OXTS records are 0.10 or 0.11 s apart. Velocities computed with
the nominal period are therefore ~3.5% too large in m/s, for ground truth and
estimators alike; MEASURED_FRAME_PERIOD_S is available as the alternative.

Missing or malformed files raise immediately; an evaluation must never skip
frames silently. KITTI is licensed CC BY-NC-SA 3.0: nothing read here (and
nothing derived from it) may be committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from camera import CameraModel

KITTI_TRACKING_ROOT = "/workspace/datasets/kitti/tracking/extracted/training"
KITTI_TRACKING_FPS = 10.0                 # nominal capture rate (devkit readme)
MEASURED_FRAME_PERIOD_S = 0.10355         # estimated from OXTS (see module docstring and tests)
EARTH_RADIUS_M = 6378137.0  # as in the KITTI raw devkit's convertOxtsToPose.m

# Field order of an OXTS record (KITTI raw-data devkit, dataformat.txt).
OXTS_FIELDS = (
    "lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au", "wx", "wy", "wz", "wf", "wl", "wu",
    "pos_accuracy", "vel_accuracy", "navstat", "numsats", "posmode", "velmode", "orimode",
)
OXTS_INDEX = {name: i for i, name in enumerate(OXTS_FIELDS)}

# The files use R_rect / Tr_velo_cam; the devkit readme (and the object benchmark) say R0_rect / Tr_velo_to_cam.
_CALIB_ALIASES = {"R0_rect": "R_rect", "Tr_velo_to_cam": "Tr_velo_cam", "Tr_imu_to_velo": "Tr_imu_velo"}
_CALIB_SIZES = {"P0": 12, "P1": 12, "P2": 12, "P3": 12, "R_rect": 9, "Tr_velo_cam": 12, "Tr_imu_velo": 12}

LABEL_COLUMNS = 17
DONT_CARE = "DontCare"


def _homogeneous(m34: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[: m34.shape[0], : m34.shape[1]] = m34
    return out


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform (R, t) -> (R^T, -R^T t)."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


# =============================================================================
# Calibration
# =============================================================================


@dataclass(frozen=True)
class TrackingCalibration:
    P: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]   # P0..P3, 3x4 each
    R_rect: np.ndarray        # 3x3
    Tr_velo_cam: np.ndarray   # 4x4, Velodyne -> cam0 (unrectified)
    Tr_imu_velo: np.ndarray   # 4x4, IMU -> Velodyne

    @property
    def K2(self) -> np.ndarray:
        """Intrinsics of image_02 (the left colour camera)."""
        return self.P[2][:, :3].copy()

    @property
    def cam2_offset(self) -> np.ndarray:
        """t2 with P2 = K2 [I | t2]: cam0-rect coordinates map to cam2 coordinates as x_cam2 = x_cam0rect + t2."""
        return np.linalg.solve(self.K2, self.P[2][:, 3])

    @property
    def T_cam0rect_velo(self) -> np.ndarray:
        return _homogeneous(self.R_rect) @ self.Tr_velo_cam

    @property
    def T_cam0rect_imu(self) -> np.ndarray:
        """IMU -> rectified cam0: R_rect @ Tr_velo_cam @ Tr_imu_velo."""
        return self.T_cam0rect_velo @ self.Tr_imu_velo

    @property
    def T_cam2_cam0rect(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.cam2_offset
        return T

    @property
    def T_cam2_imu(self) -> np.ndarray:
        return self.T_cam2_cam0rect @ self.T_cam0rect_imu


def read_tracking_calib(path: Union[str, Path]) -> TrackingCalibration:
    values: Dict[str, np.ndarray] = {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        key, *nums = line.split()
        key = _CALIB_ALIASES.get(key.rstrip(":"), key.rstrip(":"))
        if key not in _CALIB_SIZES:
            raise ValueError(f"{path}:{line_no}: unknown calibration key {key!r}")
        if len(nums) != _CALIB_SIZES[key]:
            raise ValueError(f"{path}:{line_no}: {key} has {len(nums)} values, expected {_CALIB_SIZES[key]}")
        values[key] = np.array(nums, dtype=np.float64)
    missing = set(_CALIB_SIZES) - set(values)
    if missing:
        raise ValueError(f"{path}: missing calibration entries {sorted(missing)}")
    P = tuple(values[f"P{i}"].reshape(3, 4) for i in range(4))
    calib = TrackingCalibration(
        P=P, R_rect=values["R_rect"].reshape(3, 3),
        Tr_velo_cam=_homogeneous(values["Tr_velo_cam"].reshape(3, 4)),
        Tr_imu_velo=_homogeneous(values["Tr_imu_velo"].reshape(3, 4)),
    )
    for name, R in (("R_rect", calib.R_rect), ("Tr_velo_cam", calib.Tr_velo_cam[:3, :3]),
                    ("Tr_imu_velo", calib.Tr_imu_velo[:3, :3])):
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-4) and np.linalg.det(R) > 0):
            raise ValueError(f"{path}: {name} rotation is not orthonormal")
    K2 = calib.K2
    if not (K2[0, 0] > 0 and K2[1, 1] > 0 and np.allclose(K2[2], [0, 0, 1]) and abs(K2[0, 1]) < 1e-9):
        raise ValueError(f"{path}: P2 does not have a pinhole intrinsic block: {K2.tolist()}")
    return calib


# =============================================================================
# Labels
# =============================================================================


@dataclass(frozen=True)
class TrackLabel:
    frame: int
    track_id: int            # -1 for DontCare regions
    type: str
    truncated: int           # 0 (not), 1 (partly), 2 (heavily) -- integers in the tracking benchmark
    occluded: int            # 0 visible, 1 partly, 2 largely occluded, 3 unknown
    alpha: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in image_02 pixels
    dimensions: Tuple[float, float, float]    # height, width, length (m)
    location: Tuple[float, float, float]      # x, y, z (m) of the bottom-face centre, cam0 rect frame
    rotation_y: float        # yaw about the camera y axis, [-pi, pi]

    @property
    def is_dont_care(self) -> bool:
        return self.type == DONT_CARE

    @property
    def box_height_px(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def corners_cam0rect(self) -> np.ndarray:
        """The 8 corners of the 3D box (8, 3) in cam0 rect coordinates (object benchmark convention)."""
        h, w, l = self.dimensions
        x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2.0
        y = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float64)
        z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2.0
        c, s = np.cos(self.rotation_y), np.sin(self.rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return (R @ np.vstack([x, y, z])).T + np.asarray(self.location)


def read_tracking_labels(path: Union[str, Path]) -> List[TrackLabel]:
    labels: List[TrackLabel] = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        f = line.split()
        if len(f) != LABEL_COLUMNS:
            raise ValueError(f"{path}:{line_no}: {len(f)} columns, expected {LABEL_COLUMNS}")
        labels.append(TrackLabel(
            frame=int(f[0]), track_id=int(f[1]), type=f[2], truncated=int(float(f[3])), occluded=int(float(f[4])),
            alpha=float(f[5]), bbox=tuple(float(v) for v in f[6:10]),
            dimensions=tuple(float(v) for v in f[10:13]), location=tuple(float(v) for v in f[13:16]),
            rotation_y=float(f[16]),
        ))
    return labels


# =============================================================================
# OXTS and poses
# =============================================================================


def read_oxts(path: Union[str, Path]) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if data.shape[1] != len(OXTS_FIELDS):
        raise ValueError(f"{path}: {data.shape[1]} OXTS fields per line, expected {len(OXTS_FIELDS)}")
    return data


def estimate_frame_period(oxts: np.ndarray, gps_poses: np.ndarray, span: int = 20,
                          min_speed: float = 3.0, max_yaw_rate: float = 0.05) -> Tuple[float, float]:
    """(GPS distance, integrated vf * nominal period) summed over moving, near-straight ``span``-frame chunks.

    0.1 * distance / integral estimates the true image spacing from OXTS alone.
    Chunks with any |wz| above ``max_yaw_rate`` are skipped: on a curve the GPS
    chord is shorter than the driven arc, which would bias the estimate low.
    """
    nominal = 1.0 / KITTI_TRACKING_FPS
    vf, wz = oxts[:, OXTS_INDEX["vf"]], oxts[:, OXTS_INDEX["wz"]]
    dist = integral = 0.0
    for s in range(0, len(oxts) - span, span):
        if vf[s:s + span + 1].min() > min_speed and np.abs(wz[s:s + span + 1]).max() <= max_yaw_rate:
            dist += float(np.linalg.norm(gps_poses[s + span, :3, 3] - gps_poses[s, :3, 3]))
            integral += nominal * float(0.5 * vf[s] + vf[s + 1:s + span].sum() + 0.5 * vf[s + span])
    return dist, integral


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def oxts_to_poses(oxts: np.ndarray) -> np.ndarray:
    """IMU poses (N, 4, 4) mapping IMU_k coordinates to a local world frame anchored at frame 0.

    A direct port of the KITTI raw devkit's convertOxtsToPose.m: Mercator
    projection with the scale fixed by the first latitude, R = Rz(yaw) Ry(pitch)
    Rx(roll), then everything expressed relative to the first pose. GPS
    positions are sampled on the OXTS 10 ms grid, so per-frame displacements
    alternate between ~0.10 s and ~0.11 s worth of motion.
    """
    lat, lon, alt = oxts[:, 0], oxts[:, 1], oxts[:, 2]
    scale = np.cos(np.radians(lat[0]))
    tx = scale * np.radians(lon) * EARTH_RADIUS_M
    ty = scale * EARTH_RADIUS_M * np.log(np.tan(np.radians(90.0 + lat) / 2.0))
    poses = np.zeros((len(oxts), 4, 4))
    for k, rec in enumerate(oxts):
        T = np.eye(4)
        T[:3, :3] = _rot_z(rec[5]) @ _rot_y(rec[4]) @ _rot_x(rec[3])
        T[:3, 3] = (tx[k], ty[k], alt[k])
        poses[k] = T
    T0_inv = invert_rigid(poses[0])
    return np.einsum("ij,njk->nik", T0_inv, poses)


def oxts_to_odometry_poses(oxts: np.ndarray, dt: float = MEASURED_FRAME_PERIOD_S) -> np.ndarray:
    """IMU poses (N, 4, 4) like ``oxts_to_poses`` but with the translation integrated from velocity.

    Orientation is the same R = Rz(yaw) Ry(pitch) Rx(roll); the position is the
    trapezoidal integral of the body-frame velocity (vf, vl, vu) rotated into
    the world frame over the true image spacing ``dt``. Unlike the GPS
    positions, this has no 10 ms sampling alternation, so it is the better
    frame-to-frame motion (parked cars come out with a lower and unbiased
    apparent speed); it drifts slowly, so use ``oxts_to_poses`` for long-range
    geometry.

    The vn/ve fields are not used: in the tracking OXTS files they do not agree
    with the GPS track's direction under any axis convention, while vf/vl do.
    """
    R = np.stack([_rot_z(r[5]) @ _rot_y(r[4]) @ _rot_x(r[3]) for r in oxts])
    v_world = np.einsum("nij,nj->ni", R, oxts[:, [OXTS_INDEX["vf"], OXTS_INDEX["vl"], OXTS_INDEX["vu"]]])
    steps = 0.5 * (v_world[1:] + v_world[:-1]) * dt
    positions = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    poses = np.tile(np.eye(4), (len(oxts), 1, 1))
    poses[:, :3, :3] = R
    poses[:, :3, 3] = positions
    T0_inv = invert_rigid(poses[0])
    return np.einsum("ij,njk->nik", T0_inv, poses)


# =============================================================================
# Sequences
# =============================================================================


@dataclass
class KittiTrackingSequence:
    root: Path
    name: str                                   # "0000" .. "0020"
    image_paths: List[Path] = field(repr=False)
    frame_period_s: float = 1.0 / KITTI_TRACKING_FPS

    @property
    def n_frames(self) -> int:
        return len(self.image_paths)

    @property
    def timestamps(self) -> np.ndarray:
        """Reconstructed capture times in seconds: frame * frame_period_s (no timestamp files exist)."""
        return np.arange(self.n_frames, dtype=np.float64) * self.frame_period_s

    @cached_property
    def calib(self) -> TrackingCalibration:
        return read_tracking_calib(self.root / "calib" / f"{self.name}.txt")

    @cached_property
    def labels(self) -> List[TrackLabel]:
        labels = read_tracking_labels(self.root / "label_02" / f"{self.name}.txt")
        bad = [l.frame for l in labels if not 0 <= l.frame < self.n_frames]
        if bad:
            raise ValueError(f"sequence {self.name}: labels refer to frames {sorted(set(bad))[:5]} "
                             f"outside 0..{self.n_frames - 1}")
        return labels

    @cached_property
    def labels_by_frame(self) -> List[List[TrackLabel]]:
        """One (possibly empty) list per frame -- frames without annotations stay present."""
        out: List[List[TrackLabel]] = [[] for _ in range(self.n_frames)]
        for label in self.labels:
            out[label.frame].append(label)
        return out

    @cached_property
    def tracks(self) -> Dict[int, Dict[int, TrackLabel]]:
        """track_id -> {frame: label}, DontCare excluded. Raises on duplicates or a type change."""
        out: Dict[int, Dict[int, TrackLabel]] = {}
        for label in self.labels:
            if label.is_dont_care:
                continue
            per_track = out.setdefault(label.track_id, {})
            if label.frame in per_track:
                raise ValueError(f"sequence {self.name}: track {label.track_id} labelled twice in frame {label.frame}")
            per_track[label.frame] = label
        for tid, per_track in out.items():
            types = {l.type for l in per_track.values()}
            if len(types) != 1:
                raise ValueError(f"sequence {self.name}: track {tid} changes type {sorted(types)}")
        return out

    @cached_property
    def oxts(self) -> np.ndarray:
        data = read_oxts(self.root / "oxts" / f"{self.name}.txt")
        if len(data) != self.n_frames:
            raise ValueError(f"sequence {self.name}: {len(data)} OXTS records for {self.n_frames} images")
        return data

    @cached_property
    def imu_poses_gps(self) -> np.ndarray:
        """(N, 4, 4) IMU_k -> world (world = IMU at frame 0), GPS positions (devkit method)."""
        return oxts_to_poses(self.oxts)

    @cached_property
    def imu_poses(self) -> np.ndarray:
        """(N, 4, 4) IMU_k -> world, positions integrated from OXTS velocity over the measured image
        spacing (see oxts_to_odometry_poses). Independent of frame_period_s, which only sets timestamps."""
        return oxts_to_odometry_poses(self.oxts, MEASURED_FRAME_PERIOD_S)

    def _imu_to_cam2(self, imu_poses: np.ndarray) -> np.ndarray:
        T_c_i = self.calib.T_cam2_imu
        return np.einsum("ij,njk,kl->nil", T_c_i, imu_poses, invert_rigid(T_c_i))

    @cached_property
    def cam2_poses(self) -> np.ndarray:
        """(N, 4, 4) cam2_k -> world, where world is cam2 at frame 0 (x right, y down, z forward)."""
        return self._imu_to_cam2(self.imu_poses)

    @cached_property
    def cam2_poses_gps(self) -> np.ndarray:
        return self._imu_to_cam2(self.imu_poses_gps)

    def T_cam2(self, target: int, source: int) -> np.ndarray:
        """Transform mapping cam2 coordinates at frame ``source`` to cam2 coordinates at frame ``target``."""
        return invert_rigid(self.cam2_poses[target]) @ self.cam2_poses[source]

    @cached_property
    def image_size(self) -> Tuple[int, int]:
        """(width, height) of this sequence's images (checked on the first frame; constant per sequence)."""
        image = cv2.imread(str(self.image_paths[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Could not read image: {self.image_paths[0]}")
        return int(image.shape[1]), int(image.shape[0])

    @cached_property
    def camera(self) -> CameraModel:
        w, h = self.image_size
        return CameraModel.from_matrix(self.calib.K2, w, h, source=f"KITTI tracking calib/{self.name}.txt P2")

    def load_image(self, frame: int) -> np.ndarray:
        image = cv2.imread(str(self.image_paths[frame]), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {self.image_paths[frame]}")
        if (image.shape[1], image.shape[0]) != self.image_size:
            raise ValueError(f"{self.image_paths[frame]}: size {image.shape[1]}x{image.shape[0]} differs from "
                             f"the sequence's {self.image_size}")
        return image  # BGR uint8, as the rest of the pipeline expects

    def label_to_cam2(self, label: TrackLabel) -> np.ndarray:
        """The label's location (bottom-face centre) in cam2 coordinates."""
        return np.asarray(label.location, dtype=np.float64) + self.calib.cam2_offset


class KittiTrackingDataset:
    """All sequences under ``root`` (the extracted ``training`` directory)."""

    def __init__(self, root: Union[str, Path] = KITTI_TRACKING_ROOT,
                 sequences: Optional[Sequence[str]] = None,
                 frame_period_s: float = 1.0 / KITTI_TRACKING_FPS) -> None:
        self.root = Path(root)
        self.frame_period_s = float(frame_period_s)
        image_root = self.root / "image_02"
        if not image_root.is_dir():
            raise FileNotFoundError(f"KITTI tracking images not found at {image_root}")
        names = sorted(p.name for p in image_root.iterdir() if p.is_dir())
        if sequences is not None:
            unknown = set(sequences) - set(names)
            if unknown:
                raise ValueError(f"Unknown sequences {sorted(unknown)} under {image_root}")
            names = [n for n in names if n in set(sequences)]
        self.sequences: List[KittiTrackingSequence] = [self._load_sequence(n) for n in names]

    def _load_sequence(self, name: str) -> KittiTrackingSequence:
        paths = sorted((self.root / "image_02" / name).glob("*.png"))
        if not paths:
            raise ValueError(f"sequence {name}: no images")
        expected = [f"{k:06d}.png" for k in range(len(paths))]
        if [p.name for p in paths] != expected:
            first_bad = next(e for p, e in zip([p.name for p in paths] + [""], expected) if p != e)
            raise ValueError(f"sequence {name}: frame files are not contiguous from 000000 (expected {first_bad})")
        for sub, ext in (("label_02", ".txt"), ("oxts", ".txt"), ("calib", ".txt")):
            if not (self.root / sub / f"{name}{ext}").is_file():
                raise FileNotFoundError(f"sequence {name}: missing {sub}/{name}{ext}")
        return KittiTrackingSequence(root=self.root, name=name, image_paths=paths, frame_period_s=self.frame_period_s)

    def __iter__(self):
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, name: str) -> KittiTrackingSequence:
        for seq in self.sequences:
            if seq.name == name:
                return seq
        raise KeyError(name)
