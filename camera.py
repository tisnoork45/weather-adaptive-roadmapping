"""Pinhole camera model passed per frame through the pipeline.

Intrinsics belong to a specific image size. A dataset frame brings its own
calibrated matrix (e.g. KITTI's per-frame K); a video or webcam falls back to
config.yaml's ``camera`` section, which is a placeholder unless
``camera.calibrated`` is true. Every consumer can therefore tell whether its
metric geometry rests on a real calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "unknown"
    calibrated: bool = False

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int, source: str,
                    calibrated: bool = True) -> "CameraModel":
        K = np.asarray(K, dtype=np.float64)
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                   width=int(width), height=int(height), source=source, calibrated=calibrated)

    @classmethod
    def from_config(cls, camera_cfg) -> "CameraModel":
        """Camera from config.yaml's ``camera`` section (intrinsics defined at width x height)."""
        intr = camera_cfg.intrinsics
        return cls(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.cx), cy=float(intr.cy),
                   width=int(camera_cfg.width), height=int(camera_cfg.height), source="config.yaml camera",
                   calibrated=bool(camera_cfg.get("calibrated", False)))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled_to(self, width: int, height: int, aspect_tolerance: float = 0.01,
                  allow_aspect_change: bool = False) -> "CameraModel":
        """The same camera at another resolution (same field of view, no crop).

        Only valid when the aspect ratio is unchanged: a different aspect ratio
        means a crop or a different sensor, and rescaling would silently give
        wrong geometry, so that raises -- unless ``allow_aspect_change`` is set,
        which callers use only for uncalibrated placeholder intrinsics (whose
        geometry is approximate anyway) to keep the old per-axis scaling.
        """
        if (width, height) == (self.width, self.height):
            return self
        src_aspect, dst_aspect = self.width / self.height, width / height
        if abs(dst_aspect / src_aspect - 1.0) > aspect_tolerance and not allow_aspect_change:
            raise ValueError(
                f"Cannot rescale {self.width}x{self.height} intrinsics ({self.source}) to {width}x{height}: "
                f"aspect ratio {src_aspect:.3f} vs {dst_aspect:.3f}. Provide intrinsics for this resolution."
            )
        sx, sy = width / self.width, height / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
                       width=int(width), height=int(height))
