"""Bird's-Eye-View (BEV) mapping module.

Uses the estimated monocular depth map plus pinhole camera intrinsics to
back-project pixels into a 3D point cloud (built with Open3D, if available),
then flattens that point cloud onto a 2D top-down grid image. Detected /
tracked objects are projected onto the same BEV grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import Detection
from utils import get_logger

logger = get_logger(__name__)

try:
    import open3d as o3d  # noqa: F401
    _OPEN3D_AVAILABLE = True
except Exception:  # noqa: BLE001
    _OPEN3D_AVAILABLE = False
    logger.warning(
        "Open3D not available; BEV mapping still works via numpy-based "
        "projection, but 3D point-cloud (.ply) export is disabled."
    )


@dataclass
class BEVObject:
    track_id: Optional[int]
    class_name: str
    x_m: float
    z_m: float
    confidence: float


@dataclass
class BEVResult:
    bev_image: np.ndarray
    objects: List[BEVObject] = field(default_factory=list)
    point_cloud: Optional[object] = None
    inference_time_ms: float = 0.0


class BEVMapper:
    """Projects camera-space detections + depth into a top-down BEV representation."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 bev_range_m: float = 60.0, bev_width_m: float = 40.0,
                 resolution_px_per_m: float = 10.0) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.bev_range_m, self.bev_width_m = bev_range_m, bev_width_m
        self.resolution = resolution_px_per_m
        self.bev_h = int(bev_range_m * resolution_px_per_m)
        self.bev_w = int(bev_width_m * resolution_px_per_m)

    def pixel_to_ground_point(self, u: float, v: float, depth_m: float) -> Tuple[float, float, float]:
        z = float(depth_m)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    def build_point_cloud(self, frame: np.ndarray, depth_map: np.ndarray,
                           stride: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        h, w = depth_map.shape[:2]
        us, vs = np.arange(0, w, stride), np.arange(0, h, stride)
        grid_u, grid_v = np.meshgrid(us, vs)
        depths = depth_map[grid_v, grid_u]
        valid = (depths > 0.5) & (depths < self.bev_range_m)

        xs = (grid_u[valid] - self.cx) * depths[valid] / self.fx
        ys = (grid_v[valid] - self.cy) * depths[valid] / self.fy
        zs = depths[valid]
        points = np.stack([xs, ys, zs], axis=1).astype(np.float32)

        frame_resized = frame if frame.shape[:2] == depth_map.shape[:2] else cv2.resize(frame, (w, h))
        colors_bgr = frame_resized[grid_v, grid_u][valid]
        colors = colors_bgr[:, ::-1].astype(np.float32) / 255.0
        return points, colors

    def to_open3d_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        if not _OPEN3D_AVAILABLE or len(points) == 0:
            return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _sample_object_depth(self, depth_map: np.ndarray, u: int, v: int,
                              x1: float, y1: float, x2: float, y2: float, half_size: int = 2) -> float:
        """Median depth over a small window around (u, v), clipped to both
        the image bounds and the detection box itself.

        A single sampled pixel is noisy in two independent ways: normal
        frame-to-frame depth-map noise, and landing exactly on a foreground/
        background depth discontinuity at the object's silhouette edge
        (common, since (u, v) is deliberately the box's bottom-center --
        right where an object typically meets the ground/background). A
        small median window is a standard, cheap way to suppress both
        without smoothing across real object boundaries (clipped to the
        box, so it never pulls in background pixels beside a thin object).
        Falls back to the original single-pixel read if the window ends up
        empty of valid samples, so behavior degrades gracefully rather than
        silently returning something new for the same case it always
        handled (behind-camera/invalid depth -> caller's existing
        `depth_val <= 0.1` skip is unaffected).
        """
        h, w = depth_map.shape[:2]
        wy1 = max(int(y1), v - half_size, 0)
        wy2 = min(int(y2), v + half_size, h - 1)
        wx1 = max(int(x1), u - half_size, 0)
        wx2 = min(int(x2), u + half_size, w - 1)
        if wy2 < wy1 or wx2 < wx1:
            return float(depth_map[v, u])
        patch = depth_map[wy1:wy2 + 1, wx1:wx2 + 1]
        valid = patch[patch > 0.1]
        if valid.size == 0:
            return float(depth_map[v, u])
        return float(np.median(valid))

    def project_objects_to_bev(self, detections: List[Detection], depth_map: np.ndarray) -> List[BEVObject]:
        h, w = depth_map.shape[:2]
        bev_objects: List[BEVObject] = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            u = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
            v = int(np.clip(y2, 0, h - 1))
            depth_val = self._sample_object_depth(depth_map, u, v, x1, y1, x2, y2)
            if depth_val <= 0.1:
                continue
            x_m, _, z_m = self.pixel_to_ground_point(u, v, depth_val)
            bev_objects.append(BEVObject(det.track_id, det.class_name, x_m, z_m, det.confidence))
        return bev_objects

    def render_bev_image(self, points: np.ndarray, bev_objects: List[BEVObject]) -> np.ndarray:
        canvas = np.full((self.bev_h, self.bev_w, 3), 15, dtype=np.uint8)

        for range_m in range(0, int(self.bev_range_m) + 1, 10):
            y_px = self.bev_h - 1 - int(range_m * self.resolution)
            cv2.line(canvas, (0, y_px), (self.bev_w, y_px), (50, 50, 50), 1)
            cv2.putText(canvas, f"{range_m}m", (5, max(12, y_px - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        if points is not None and len(points):
            xs, zs = points[:, 0], points[:, 2]
            px = (self.bev_w / 2 + xs * self.resolution).astype(np.int32)
            py = (self.bev_h - 1 - zs * self.resolution).astype(np.int32)
            valid = (px >= 0) & (px < self.bev_w) & (py >= 0) & (py < self.bev_h)
            canvas[py[valid], px[valid]] = (90, 90, 90)

        for obj in bev_objects:
            px = int(self.bev_w / 2 + obj.x_m * self.resolution)
            py = int(self.bev_h - 1 - obj.z_m * self.resolution)
            if not (0 <= px < self.bev_w and 0 <= py < self.bev_h):
                continue
            color = (60, 220, 60) if obj.class_name == "car" else (60, 60, 230)
            cv2.circle(canvas, (px, py), 6, color, -1)
            label = f"{obj.class_name}" if obj.track_id is None else f"ID{obj.track_id}:{obj.class_name}"
            cv2.putText(canvas, label, (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        ego_x, ego_y = self.bev_w // 2, self.bev_h - 5
        cv2.rectangle(canvas, (ego_x - 8, ego_y - 14), (ego_x + 8, ego_y), (0, 255, 0), -1)
        return canvas

    def build_bev(self, frame: np.ndarray, depth_map: np.ndarray, detections: List[Detection]) -> BEVResult:
        start = time.perf_counter()
        points, colors = self.build_point_cloud(frame, depth_map, stride=10)
        pcd = self.to_open3d_point_cloud(points, colors)
        bev_objects = self.project_objects_to_bev(detections, depth_map)

        bev_image = self.render_bev_image(points, bev_objects)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return BEVResult(bev_image=bev_image, objects=bev_objects, point_cloud=pcd, inference_time_ms=elapsed_ms)

    @staticmethod
    def save_point_cloud(pcd, path: str) -> bool:
        if not _OPEN3D_AVAILABLE or pcd is None:
            logger.warning("Cannot save point cloud: Open3D unavailable or empty cloud.")
            return False
        o3d.io.write_point_cloud(path, pcd)
        logger.info("Saved point cloud to %s", path)
        return True
