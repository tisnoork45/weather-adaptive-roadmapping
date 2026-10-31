"""Tests for the frozen B0 estimator, the KITTI tracking loader, ground-truth kinematics and the
offline velocity evaluation (Task 5, Stage 1).

Run with:
    pytest tests_kitti_tracking.py -v

Synthetic tests need nothing but the code. Tests marked "dataset" read the KITTI
tracking training set at kitti_tracking.KITTI_TRACKING_ROOT (or $KITTI_TRACKING_ROOT)
and are skipped when it is absent; they enforce the ground-truth validity gates
G1-G5 of task5_evaluation_protocol.md. Set $KITTI_VELOCITY_CACHE to a cache
directory to also check gate G6 (B0 replay == online production output).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import kitti_tracking as kt
from kitti_tracking import (
    KITTI_TRACKING_ROOT, MEASURED_FRAME_PERIOD_S, KittiTrackingDataset, TrackLabel, estimate_frame_period, invert_rigid,
    oxts_to_odometry_poses, oxts_to_poses, read_oxts, read_tracking_calib, read_tracking_labels, transform_points,
)
from kitti_velocity_gt import (
    VEHICLE_TYPES, ego_kinematics, is_primary_eligible, lsq_slope, object_kinematics, project_box,
    rigid_motion_residual,
)
from velocity_eval import (
    B0Adapter, CachedFrame, CachedObject, VelocityEstimate, cluster_bootstrap_mean, match_frame,
    paired_cluster_bootstrap, replay, summarize_rows, ttc_band,
)

HERE = Path(__file__).parent
B0_SHA256 = "4c7b88236a6676748f710f36aac6a559ed830c28917b37065079e14a0278738b"  # relative_velocity.py @ 1892061
DATA_ROOT = Path(os.environ.get("KITTI_TRACKING_ROOT", KITTI_TRACKING_ROOT))
needs_dataset = pytest.mark.skipif(not (DATA_ROOT / "image_02").is_dir(), reason="KITTI tracking data not present")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============================================================================
# B0: frozen copy of the production velocity estimator
# =============================================================================


def test_b0_file_is_frozen() -> None:
    assert _sha256(HERE / "velocity_b0.py") == B0_SHA256


def test_b0_is_byte_identical_to_production_estimator() -> None:
    # Fails the day relative_velocity.py changes: B0 must then stay as it is, and this
    # test is updated to say so -- it documents that B0 *was* production at freeze time.
    assert (HERE / "velocity_b0.py").read_bytes() == (HERE / "relative_velocity.py").read_bytes()


def _random_scenario(seed: int, n_frames: int = 120):
    """Frames of duck-typed BEV objects exercising every branch of the estimator:
    births, missed frames (history wiped), untracked objects, repeated timestamps,
    gaps longer than max_gap_seconds and history longer than history_length."""
    rng = np.random.default_rng(seed)
    t, frames = 0.0, []
    for _ in range(n_frames):
        step = rng.choice([0.0, 0.04, 0.1, 0.1, 0.1, 2.5], p=[0.05, 0.2, 0.3, 0.3, 0.1, 0.05])
        t += float(step)
        objs = []
        for tid in range(6):
            if rng.random() < 0.8:
                objs.append(SimpleNamespace(track_id=tid, class_name="car", x_m=float(rng.normal(0, 5)),
                                            z_m=float(rng.uniform(3, 60)), confidence=0.9))
        if rng.random() < 0.3:
            objs.append(SimpleNamespace(track_id=None, class_name="car", x_m=1.0, z_m=10.0, confidence=0.4))
        rng.shuffle(objs)
        frames.append((t, objs))
    return frames


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("params", [dict(), dict(history_length=3, max_gap_seconds=0.5, ema_alpha=1.0),
                                    dict(history_length=8, max_gap_seconds=5.0, ema_alpha=0.2)])
def test_b0_reproduces_production_outputs(seed, params) -> None:
    import relative_velocity
    import velocity_b0
    prod, frozen = relative_velocity.RelativeVelocityEstimator(**params), velocity_b0.RelativeVelocityEstimator(**params)
    n_valid = 0
    for t, objs in _random_scenario(seed):
        a, b = prod.update(objs, timestamp=t), frozen.update(objs, timestamp=t)
        assert a.keys() == b.keys()
        for tid in a:
            assert (a[tid].vx_m_s, a[tid].vz_m_s, a[tid].speed_m_s, a[tid].valid) == \
                   (b[tid].vx_m_s, b[tid].vz_m_s, b[tid].speed_m_s, b[tid].valid)   # exact, not approx
            n_valid += a[tid].valid
    assert n_valid > 50                               # the scenario really produced estimates


def test_b0_wall_clock_fallback_matches(monkeypatch) -> None:
    import relative_velocity
    import velocity_b0
    clock = iter(np.repeat(np.arange(0, 10, 0.1), 2))      # each update reads the clock once
    fake = lambda: float(next(clock))                       # noqa: E731
    monkeypatch.setattr(relative_velocity.time, "perf_counter", fake)
    monkeypatch.setattr(velocity_b0.time, "perf_counter", fake)
    prod, frozen = relative_velocity.RelativeVelocityEstimator(), velocity_b0.RelativeVelocityEstimator()
    for k in range(40):
        objs = [SimpleNamespace(track_id=1, x_m=0.5 * k, z_m=30.0 - k)]
        a, b = prod.update(objs), frozen.update(objs)
        assert (a[1].vx_m_s, a[1].vz_m_s, a[1].valid) == (b[1].vx_m_s, b[1].vz_m_s, b[1].valid)


def test_b0_adapter_replays_like_the_estimator() -> None:
    import relative_velocity
    frames, direct = [], relative_velocity.RelativeVelocityEstimator()
    expected = []
    for k, (t, objs) in enumerate(_random_scenario(7, 60)):
        cached = [CachedObject(track_id=o.track_id, class_name="car", box=(0, 0, 1, 1), confidence=o.confidence,
                               has_bev=True, x_m=o.x_m, z_m=o.z_m) for o in objs]
        cached.insert(0, CachedObject(track_id=99, class_name="car", box=(0, 0, 1, 1), confidence=0.9))  # no BEV
        frames.append(CachedFrame(sequence="s", frame=k, timestamp=t, objects=cached))
        expected.append(direct.update(objs, timestamp=t))
    got = replay(B0Adapter(5, 2.0, 0.5), frames)
    for e, g in zip(expected, got):
        assert e.keys() == g.keys() and 99 not in g
        for tid in e:
            assert (e[tid].vx_m_s, e[tid].vz_m_s, e[tid].valid) == (g[tid].vx_m_s, g[tid].vz_m_s, g[tid].valid)
    # reset() really starts over: a second replay gives the same answers
    again = replay(B0Adapter(5, 2.0, 0.5), frames)
    assert [{k: (v.vx_m_s, v.valid) for k, v in d.items()} for d in again] == \
           [{k: (v.vx_m_s, v.valid) for k, v in d.items()} for d in got]


# =============================================================================
# Synthetic KITTI-format sequences
# =============================================================================

K_SYN = np.array([[700.0, 0, 600.0], [0, 700.0, 180.0], [0, 0, 1]])
# Velodyne (x fwd, y left, z up) -> camera (x right, y down, z fwd); IMU == Velodyne here.
VELO_TO_CAM = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float)


def _fmt(values) -> str:
    return " ".join(f"{v:.12e}" for v in np.ravel(values))


def write_calib(path: Path, aliases: bool = False, baseline_px: float = 0.0) -> None:
    P2 = np.hstack([K_SYN, [[baseline_px], [0], [0]]])
    P0 = np.hstack([K_SYN, np.zeros((3, 1))])
    r, v = ("R0_rect", "Tr_velo_to_cam") if aliases else ("R_rect", "Tr_velo_cam")
    path.write_text("\n".join([f"P0: {_fmt(P0)}", f"P1: {_fmt(P0)}", f"P2: {_fmt(P2)}", f"P3: {_fmt(P0)}",
                               f"{r} {_fmt(np.eye(3))}", f"{v} {_fmt(VELO_TO_CAM)}",
                               f"Tr_imu_velo {_fmt(np.eye(4)[:3])}"]) + "\n")


def synthetic_oxts(speed: float, yaw_rate: float, n: int, period: float = MEASURED_FRAME_PERIOD_S,
                   lat0: float = 49.0, lon0: float = 8.4) -> np.ndarray:
    """OXTS records of a vehicle driving at constant speed and yaw rate, sampled every ``period`` s,
    with GPS positions consistent with the devkit's Mercator projection."""
    er = kt.EARTH_RADIUS_M
    scale = np.cos(np.radians(lat0))
    yaw = 0.3 + yaw_rate * period * np.arange(n)
    # exact arc positions in a local east/north frame
    if abs(yaw_rate) < 1e-12:
        east = speed * period * np.arange(n) * np.cos(yaw[0])
        north = speed * period * np.arange(n) * np.sin(yaw[0])
    else:
        east = speed / yaw_rate * (np.sin(yaw) - np.sin(yaw[0]))
        north = -speed / yaw_rate * (np.cos(yaw) - np.cos(yaw[0]))
    mx0 = scale * np.radians(lon0) * er
    my0 = scale * er * np.log(np.tan(np.radians(90 + lat0) / 2))
    lon = np.degrees((mx0 + east) / (scale * er))
    lat = np.degrees(2 * np.arctan(np.exp((my0 + north) / (scale * er))) - np.pi / 2)
    o = np.zeros((n, 30))
    o[:, 0], o[:, 1], o[:, 2], o[:, 5] = lat, lon, 100.0, yaw
    o[:, 8] = speed          # vf
    o[:, 19] = yaw_rate      # wz (about up)
    o[:, 25] = 4
    return o


def write_sequence(root: Path, name: str, n_frames: int, oxts: np.ndarray, labels: list,
                   size=(64, 32), skip_image: int = -1) -> None:
    for sub in ("image_02/" + name, "label_02", "oxts", "calib"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for k in range(n_frames):
        if k != skip_image:
            cv2.imwrite(str(root / "image_02" / name / f"{k:06d}.png"), img)
    (root / "oxts" / f"{name}.txt").write_text("\n".join(_fmt(r) for r in oxts) + "\n")
    write_calib(root / "calib" / f"{name}.txt")
    (root / "label_02" / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))


def label_line(frame, tid, typ, x, z, y=1.6, h=1.5, w=1.6, l=4.0, ry=0.0, trunc=0, occ=0, bbox=(100, 100, 200, 180)):
    return (f"{frame} {tid} {typ} {trunc} {occ} -1.5 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} "
            f"{h} {w} {l} {x} {y} {z} {ry}")


def world_static_labels(oxts: np.ndarray, point_imu0: np.ndarray, tid: int = 1) -> list:
    """Labels of a world-static point seen from the moving vehicle (camera == IMU frame, cam axes)."""
    poses = oxts_to_odometry_poses(oxts, MEASURED_FRAME_PERIOD_S)
    lines = []
    for k, T in enumerate(poses):
        p_imu = transform_points(invert_rigid(T), point_imu0)[0]
        p_cam = VELO_TO_CAM[:, :3] @ p_imu
        lines.append(label_line(k, tid, "Car", p_cam[0], p_cam[2], y=p_cam[1]))
    return lines


# --- parsing ----------------------------------------------------------------------


def test_calib_parses_both_key_spellings(tmp_path) -> None:
    write_calib(tmp_path / "a.txt")
    write_calib(tmp_path / "b.txt", aliases=True, baseline_px=42.0)
    a, b = read_tracking_calib(tmp_path / "a.txt"), read_tracking_calib(tmp_path / "b.txt")
    assert np.allclose(a.K2, K_SYN) and np.allclose(a.cam2_offset, 0)
    assert np.allclose(b.cam2_offset, [42.0 / 700.0, 0, 0])            # t2 = K^-1 P2[:, 3]
    # IMU forward (x) is camera forward (z); IMU left (y) is camera -x; IMU up is camera -y
    assert np.allclose(a.T_cam2_imu[:3, :3] @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(a.T_cam2_imu[:3, :3] @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(a.T_cam2_imu[:3, :3] @ [0, 0, 1], [0, -1, 0])


@pytest.mark.parametrize("mutation, message", [
    (lambda s: s.replace("Tr_imu_velo", "Tr_unknown"), "unknown calibration key"),
    (lambda s: "\n".join(l for l in s.splitlines() if not l.startswith("P3")), "missing calibration"),
    (lambda s: s.replace("P2: ", "P2: 1.0 "), "has 13 values"),
    (lambda s: s.replace("R_rect " + _fmt(np.eye(3)), "R_rect " + _fmt(2 * np.eye(3))), "not orthonormal"),
])
def test_calib_rejects_malformed_files(tmp_path, mutation, message) -> None:
    write_calib(tmp_path / "c.txt")
    (tmp_path / "c.txt").write_text(mutation((tmp_path / "c.txt").read_text()))
    with pytest.raises(ValueError, match=message):
        read_tracking_calib(tmp_path / "c.txt")


def test_labels_parse_and_reject_wrong_column_count(tmp_path) -> None:
    p = tmp_path / "l.txt"
    p.write_text(label_line(3, 7, "Car", 1.0, 20.0) + "\n"
                 "3 -1 DontCare -1 -1 -10 1 2 3 4 -1 -1 -1 -1000 -1000 -1000 -10\n")
    labels = read_tracking_labels(p)
    assert labels[0].frame == 3 and labels[0].track_id == 7 and labels[0].location == (1.0, 1.6, 20.0)
    assert labels[1].is_dont_care and labels[1].track_id == -1
    p.write_text(label_line(3, 7, "Car", 1.0, 20.0) + " 0.9\n")       # 18 columns (results format)
    with pytest.raises(ValueError, match="18 columns"):
        read_tracking_labels(p)


def test_box_corners_follow_the_object_benchmark_convention() -> None:
    l = TrackLabel(
        frame=0, track_id=0, type="Car", truncated=0, occluded=0, alpha=0, bbox=(0, 0, 1, 1),
        dimensions=(1.5, 1.6, 4.0), location=(1.0, 1.6, 20.0), rotation_y=np.pi / 2)
    c = l.corners_cam0rect()
    assert np.allclose(c[:, 1].max(), 1.6) and np.allclose(c[:, 1].min(), 0.1)    # bottom face at y, top at y-h
    # rotation_y = pi/2: length runs along -z ... +z, so the z extent is the length
    assert np.isclose(np.ptp(c[:, 2]), 4.0) and np.isclose(np.ptp(c[:, 0]), 1.6)


# --- sequence alignment ---------------------------------------------------------------------


def test_sequence_alignment_missing_labels_and_timestamps(tmp_path) -> None:
    o = synthetic_oxts(10.0, 0.0, 6)
    write_sequence(tmp_path, "0000", 6, o, [label_line(0, 1, "Car", 1, 20), label_line(4, 1, "Car", 1, 18)])
    ds = KittiTrackingDataset(tmp_path)
    seq = ds["0000"]
    assert seq.n_frames == 6 and len(seq.oxts) == 6
    assert [len(f) for f in seq.labels_by_frame] == [1, 0, 0, 0, 1, 0]     # unlabelled frames stay present
    assert np.allclose(seq.timestamps, 0.1 * np.arange(6))
    seq2 = KittiTrackingDataset(tmp_path, frame_period_s=MEASURED_FRAME_PERIOD_S)["0000"]
    assert np.allclose(seq2.timestamps, MEASURED_FRAME_PERIOD_S * np.arange(6))
    assert seq.camera.calibrated and seq.camera.fx == 700.0 and (seq.camera.width, seq.camera.height) == (64, 32)
    assert seq.load_image(5).shape == (32, 64, 3)


def test_alignment_errors_are_raised(tmp_path) -> None:
    o = synthetic_oxts(10.0, 0.0, 5)
    write_sequence(tmp_path / "a", "0000", 5, o, [], skip_image=2)
    with pytest.raises(ValueError, match="not contiguous"):
        KittiTrackingDataset(tmp_path / "a")
    write_sequence(tmp_path / "b", "0000", 5, o[:4], [])
    with pytest.raises(ValueError, match="4 OXTS records for 5 images"):
        _ = KittiTrackingDataset(tmp_path / "b")["0000"].oxts
    write_sequence(tmp_path / "c", "0000", 5, o, [label_line(5, 1, "Car", 1, 20)])
    with pytest.raises(ValueError, match="outside 0..4"):
        _ = KittiTrackingDataset(tmp_path / "c")["0000"].labels
    write_sequence(tmp_path / "d", "0000", 5, o, [])
    (tmp_path / "d" / "calib" / "0000.txt").unlink()
    with pytest.raises(FileNotFoundError, match="calib"):
        KittiTrackingDataset(tmp_path / "d")
    write_sequence(tmp_path / "e", "0000", 5, o, [label_line(1, 1, "Car", 1, 20), label_line(1, 1, "Car", 1, 21)])
    with pytest.raises(ValueError, match="labelled twice"):
        _ = KittiTrackingDataset(tmp_path / "e")["0000"].tracks
    write_sequence(tmp_path / "f", "0000", 5, o, [label_line(1, 1, "Car", 1, 20), label_line(2, 1, "Van", 1, 20)])
    with pytest.raises(ValueError, match="changes type"):
        _ = KittiTrackingDataset(tmp_path / "f")["0000"].tracks


# --- poses and ground truth -------------------------------------------------------------------


@pytest.mark.parametrize("yaw_rate", [0.0, 0.25])
def test_odometry_and_gps_poses_agree_on_synthetic_drive(yaw_rate) -> None:
    o = synthetic_oxts(12.0, yaw_rate, 40)
    odo, gps = oxts_to_odometry_poses(o, MEASURED_FRAME_PERIOD_S), oxts_to_poses(o)
    step = np.linalg.norm(np.diff(odo[:, :3, 3], axis=0), axis=1)
    assert np.allclose(step, 12.0 * MEASURED_FRAME_PERIOD_S, rtol=2e-3)       # chord ~ arc for small steps
    assert np.abs(odo[:, :3, 3] - gps[:, :3, 3]).max() < 0.05                # same trajectory, two methods
    assert np.allclose(odo[:, :3, :3], gps[:, :3, :3])
    dist, integral = estimate_frame_period(o, gps)
    if yaw_rate == 0.0:
        assert 0.1 * dist / integral == pytest.approx(MEASURED_FRAME_PERIOD_S, rel=1e-4)
    else:
        assert integral == 0.0                                       # curves are excluded (chord < arc)


@pytest.mark.parametrize("yaw_rate", [0.0, 0.2])
def test_world_static_object_ground_truth(tmp_path, yaw_rate) -> None:
    """Ego drives at 10 m/s past a parked car: absolute GT velocity ~ 0, relative GT velocity ~ -ego
    velocity - omega x p, and the rigid-motion identity holds."""
    n = 30
    o = synthetic_oxts(10.0, yaw_rate, n)
    labels = world_static_labels(o, np.array([[35.0, -3.0, -1.6]]))
    write_sequence(tmp_path, "0000", n, o, labels)
    seq = KittiTrackingDataset(tmp_path)["0000"]
    rows = [r for r in object_kinematics(seq) if r.has_window]
    ego = ego_kinematics(seq)
    assert len(rows) == n - 4 and not object_kinematics(seq)[0].has_window
    scale = MEASURED_FRAME_PERIOD_S / 0.1          # nominal timestamps: speeds are per 0.1 s of nominal time
    for r in rows:
        e = ego[r.frame]
        assert abs(r.abs_vx) < 0.02 and abs(r.abs_vz) < 0.02
        assert e.vz == pytest.approx(10.0 * scale, rel=1e-3)
        assert e.yaw_rate == pytest.approx(-yaw_rate * scale, abs=1e-6)     # cam y is down: -wz
        assert np.abs(rigid_motion_residual(r, e, seq)).max() < 0.02
        if yaw_rate == 0.0:
            assert r.rel_vz == pytest.approx(-10.0 * scale, rel=1e-3) and abs(r.rel_vx) < 1e-6
            assert r.rel_vz_cd == pytest.approx(r.rel_vz, rel=1e-6)
            assert r.closing_speed > 0 and r.ttc == pytest.approx(r.z_near / r.closing_speed)


def test_moving_object_ground_truth(tmp_path) -> None:
    """Stationary ego, car moving 5 m/s towards the camera and 1 m/s to the right."""
    n = 12
    o = synthetic_oxts(0.0, 0.0, n)
    labels = [label_line(k, 3, "Car", -2.0 + 0.1 * k, 40.0 - 0.5 * k) for k in range(n)]   # per 0.1 s frame
    write_sequence(tmp_path, "0000", n, o, labels)
    seq = KittiTrackingDataset(tmp_path)["0000"]
    for r in object_kinematics(seq):
        if r.has_window:
            assert (r.rel_vx, r.rel_vz) == (pytest.approx(1.0), pytest.approx(-5.0))
            assert (r.abs_vx, r.abs_vz) == (pytest.approx(1.0), pytest.approx(-5.0))
            assert r.fit_rms_m == pytest.approx(0.0, abs=1e-9)


def test_track_continuity_controls_the_window(tmp_path) -> None:
    n = 12
    o = synthetic_oxts(0.0, 0.0, n)
    frames = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]                    # frame 5 missing: a gap
    write_sequence(tmp_path, "0000", n, o, [label_line(k, 1, "Car", 0, 30 - k) for k in frames])
    rows = {r.frame: r for r in object_kinematics(KittiTrackingDataset(tmp_path)["0000"])}
    assert [k for k, r in sorted(rows.items()) if r.has_window] == [2, 8, 9]
    assert np.isnan(rows[4].rel_vz_cd) and not np.isnan(rows[3].rel_vz_cd)


def test_lsq_slope_is_exact_on_lines_and_reduces_noise() -> None:
    t = np.arange(5) * 0.1
    assert lsq_slope(t, np.stack([3 * t + 1, -2 * t], 1)) == pytest.approx([3, -2])
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, size=(20000, 5))
    slopes = np.array([lsq_slope(t, row[:, None])[0] for row in y[:4000]])
    cd = (y[:4000, 3] - y[:4000, 1]) / 0.2
    assert slopes.std() < cd.std() / 2                              # ~0.316 vs ~0.707 m/s per 0.1 m noise


def test_primary_eligibility_rules() -> None:
    from kitti_velocity_gt import ObjectKinematics
    base = dict(sequence="0", frame=5, track_id=1, type="Car", truncated=0, occluded=1, bbox=(0, 0, 50, 40),
                x=0.0, z=20.0, z_near=18.0, x_near_abs=0.0, has_window=True)
    assert is_primary_eligible(ObjectKinematics(**base))
    for change in (dict(type="Pedestrian"), dict(truncated=1), dict(occluded=2), dict(has_window=False),
                   dict(z_near=2.0), dict(z=61.0), dict(bbox=(0, 0, 50, 20)), dict(type="DontCare")):
        assert not is_primary_eligible(ObjectKinematics(**{**base, **change}))


# =============================================================================
# Matching and metrics
# =============================================================================


def test_matching_respects_iou_and_class_compatibility() -> None:
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=float)
    det = np.array([[0, 0, 10, 9], [20, 0, 30, 10], [40, 0, 46, 10], [1, 0, 11, 10]], dtype=float)
    m = match_frame(det, ["car", "person", "truck", "truck"], gt, ["Car", "Car", "Van"])
    assert sorted((d, g) for d, g, _ in m) == [(0, 0), (2, 2)]     # person never matches a Car; one-to-one
    assert match_frame(det[:0], [], gt, ["Car"] * 3) == []


def test_ttc_bands_and_bootstrap() -> None:
    assert [ttc_band(v) for v in (None, 1.0, 2.0, 4.0, 7.0)] == ["NONE", "CRITICAL", "WARNING", "CAUTION", "NONE"]
    vals, clusters = np.arange(10.0), np.repeat(["a", "b"], 5)
    ci = cluster_bootstrap_mean(vals, clusters, n_resamples=500)
    assert ci["mean"] == 4.5 and ci["ci_low"] <= 4.5 <= ci["ci_high"] and ci["n_clusters"] == 2
    same = paired_cluster_bootstrap(vals + 1, vals + 1, clusters, n_resamples=200)
    assert same["diff"] == 0 and same["diff_ci_low"] == same["diff_ci_high"] == 0
    better = paired_cluster_bootstrap(vals + 1, 0.5 * (vals + 1), clusters, n_resamples=200)
    assert better["rel_change"] == pytest.approx(-0.5) and better["rel_ci_high"] == pytest.approx(-0.5)


def _row(**kw):
    base = dict(sequence="0000", matched=True, B0_valid=True, B0_vx=0.0, B0_vz=-2.0, gt_rel_vx=0.0,
                gt_closing=2.0, gt_z_near=15.0, ego_moving=True, ego_turning=False, occluded=0,
                gt_in_corridor=True, gt_ttc=7.5, pipeline_z_m=15.0)
    base.update(kw)
    return base


def test_summary_counts_coverage_errors_and_false_alarms() -> None:
    rows = [_row(), _row(B0_vz=-4.0), _row(matched=False, B0_valid=False), _row(B0_valid=False),
            _row(B0_vz=-12.0, gt_ttc=None, gt_closing=-1.0)]
    s = summarize_rows(rows, "B0", n_boot=100)
    assert s["n_eligible_gt"] == 5 and s["n_matched"] == 4 and s["n_valid"] == 3
    assert s["coverage_end_to_end"] == pytest.approx(0.6)
    # closing errors: 0, +2, +13
    assert s["closing_speed"]["mae"] == pytest.approx(5.0) and s["closing_speed"]["bias"] == pytest.approx(5.0)
    # GT bands: NONE (7.5 s), NONE, NONE (not closing); estimates: 7.5 s NONE, 3.75 s CAUTION, 1.25 s CRITICAL
    assert s["false_critical_rate"] == pytest.approx(1 / 3) and s["false_urgent_rate"] == pytest.approx(1 / 3)
    assert s["sign_accuracy_gt_ge_1ms"] == pytest.approx(2 / 3)


def test_bev_alignment_and_equivalence_check() -> None:
    ev = importlib.import_module("evaluate_kitti_velocity")
    dets = [SimpleNamespace(track_id=1, class_name="car", confidence=0.9),
            SimpleNamespace(track_id=None, class_name="car", confidence=0.4),
            SimpleNamespace(track_id=3, class_name="truck", confidence=0.8)]
    bev = [SimpleNamespace(track_id=1, class_name="car", confidence=0.9),
           SimpleNamespace(track_id=3, class_name="truck", confidence=0.8)]
    assert ev._align_bev(dets, bev) == [0, -1, 1]
    with pytest.raises(RuntimeError):
        ev._align_bev(dets[:1], bev)
    frame = CachedFrame(sequence="s", frame=0, timestamp=0.0, objects=[], online={"1": [0.5, -1.0, 1.1, True]})
    ok = ev.check_b0_equivalence([frame], [{1: VelocityEstimate(0.5, -1.0, True)}])
    bad = ev.check_b0_equivalence([frame], [{1: VelocityEstimate(0.5, -1.0 + 1e-12, True)}])
    assert ok == {"n_compared": 1, "n_mismatched": 0} and bad["n_mismatched"] == 1


# =============================================================================
# The real dataset (ground-truth validity gates G1-G5)
# =============================================================================


@pytest.fixture(scope="module")
def dataset():
    return KittiTrackingDataset(DATA_ROOT)


@pytest.fixture(scope="module")
def kinematics(dataset):
    return {seq.name: (object_kinematics(seq), ego_kinematics(seq)) for seq in dataset}


@needs_dataset
def test_dataset_structure_and_alignment(dataset) -> None:
    counts = [s.n_frames for s in dataset]
    assert len(dataset) == 21 and sum(counts) == 8008
    assert counts == [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145, 339,
                      1059, 837]
    empty = {s.name: sum(1 for f in s.labels_by_frame if not f) for s in dataset}
    assert {k: v for k, v in empty.items() if v} == {"0006": 1, "0007": 64, "0009": 19}
    for s in dataset:
        assert len(s.oxts) == s.n_frames                       # one OXTS record per image
        assert all(0 <= l.frame < s.n_frames for l in s.labels)
        _ = s.tracks                                           # no duplicates, no type changes


@needs_dataset
def test_dataset_calibration(dataset) -> None:
    sizes = {s.name: s.image_size for s in dataset}
    assert {sizes[n] for n in ("0000", "0013")} == {(1242, 375)} and sizes["0014"] == (1224, 370)
    assert sizes["0018"] == (1238, 374) and sizes["0020"] == (1241, 376)
    assert len({round(s.calib.K2[0, 0], 3) for s in dataset}) == 4
    ref = dataset["0000"].calib.Tr_imu_velo
    for s in dataset:
        assert np.allclose(s.calib.Tr_imu_velo, ref)
        R = s.calib.T_cam2_imu[:3, :3]                         # IMU (fwd, left, up) -> cam (right, down, fwd)
        assert np.abs(R - np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).max() < 0.02
        assert 0.04 < s.calib.cam2_offset[0] < 0.08            # cam2 sits ~6 cm from cam0


@needs_dataset
def test_track_gaps(dataset) -> None:
    gaps = 0
    for s in dataset:
        for per in s.tracks.values():
            frames = sorted(per)
            gaps += sum(1 for a, b in zip(frames, frames[1:]) if b != a + 1)
    assert gaps == 3


@needs_dataset
def test_G1_label_boxes_project_onto_2d_boxes(dataset) -> None:
    from tracking import iou_batch
    ious = []
    for s in dataset:
        for l in s.labels:
            if l.type in VEHICLE_TYPES and l.truncated == 0 and l.occluded == 0:
                b = project_box(s, l)
                if b is not None:
                    ious.append(iou_batch(b[None].astype(np.float32), np.array([l.bbox], np.float32))[0, 0])
    assert len(ious) > 10000 and np.median(ious) >= 0.95


@needs_dataset
def test_G2_rigid_motion_identity(dataset, kinematics) -> None:
    res = [np.linalg.norm(rigid_motion_residual(r, ego[r.frame], s))
           for s in dataset for r, ego in [(r, kinematics[s.name][1]) for r in kinematics[s.name][0]]
           if r.has_window and ego[r.frame].has_window]
    assert len(res) > 40000 and np.median(res) <= 0.05


@needs_dataset
def test_G3_parked_cars_are_static_in_the_world(kinematics) -> None:
    obs = []
    for rows, ego in kinematics.values():
        tracks = defaultdict(list)
        for r in rows:
            if r.has_window and r.type in VEHICLE_TYPES and r.truncated == 0 and r.occluded <= 1 \
                    and abs(ego[r.frame].vz) > 3:
                tracks[r.track_id].append((r.abs_speed, r.abs_vz, ego[r.frame].vz))
        obs += [o for v in tracks.values() if len(v) >= 5 and np.median([x[0] for x in v]) < 0.8 for o in v]
    obs = np.array(obs)
    assert len(obs) > 5000
    assert np.median(obs[:, 0]) <= 0.3
    assert abs(np.polyfit(obs[:, 2], obs[:, 1], 1)[0]) <= 0.01      # no drift proportional to ego speed


@needs_dataset
def test_G4_frame_period_from_oxts(dataset) -> None:
    dist = integral = 0.0
    for s in dataset:
        d, i = estimate_frame_period(s.oxts, s.imu_poses_gps)
        dist, integral = dist + d, integral + i
    period = 0.1 * dist / integral
    assert 0.1030 <= period <= 0.1040 and period == pytest.approx(MEASURED_FRAME_PERIOD_S, abs=2e-4)


@needs_dataset
def test_G5_camera_yaw_rate_matches_gyro(kinematics) -> None:
    scale = MEASURED_FRAME_PERIOD_S / 0.1
    d = [e.yaw_rate - scale * e.oxts_yaw_rate for _, ego in kinematics.values() for e in ego if e.has_window]
    assert np.median(np.abs(d)) <= 0.01
    v = np.array([(e.vz, e.oxts_vf) for _, ego in kinematics.values() for e in ego if e.has_window and e.oxts_vf > 3])
    assert np.median(v[:, 0] / v[:, 1]) == pytest.approx(scale, rel=0.01)


@needs_dataset
def test_primary_subset_size(kinematics) -> None:
    n = sum(is_primary_eligible(r) for rows, _ in kinematics.values() for r in rows)
    assert n == 17863


@pytest.mark.skipif(not os.environ.get("KITTI_VELOCITY_CACHE"), reason="set KITTI_VELOCITY_CACHE to a cache dir")
def test_G6_b0_replay_reproduces_cached_production_output() -> None:
    ev = importlib.import_module("evaluate_kitti_velocity")
    cache = Path(os.environ["KITTI_VELOCITY_CACHE"])
    manifest = json.loads((cache / "manifest.json").read_text())
    ds = KittiTrackingDataset(manifest["dataset_root"], sequences=sorted(manifest["sequences"]),
                              frame_period_s=manifest["frame_period_s"])
    v = manifest["config"]["velocity"]
    total = 0
    for seq in ds:
        frames = ev.load_cached_frames(cache, seq)
        eq = ev.check_b0_equivalence(frames, replay(B0Adapter(v["history_length"], v["max_gap_seconds"],
                                                              v["ema_alpha"]), frames))
        assert eq["n_mismatched"] == 0
        total += eq["n_compared"]
    assert total > 0
