"""Reader for the KITTI depth-prediction "selected validation" set.

Layout (data_depth_selection.zip, 2017 depth benchmark):

    <root>/val_selection_cropped/
        image/              <drive>_image_<frame>_image_<cam>.png              uint8 RGB, 1216x352
        groundtruth_depth/  <drive>_groundtruth_depth_<frame>_image_<cam>.png  uint16, metres*256, 0 = no GT
        velodyne_raw/       <drive>_velodyne_raw_<frame>_image_<cam>.png       uint16, projected raw LiDAR
        intrinsics/         <drive>_image_<frame>_image_<cam>.txt              3x3 K, row-major, for the crop

Companion files differ from the image name only by the type token, so they
are derived from it rather than matched by sort order. Missing or malformed
files raise immediately -- an evaluation must never silently skip frames.
KITTI is licensed CC BY-NC-SA 3.0: nothing read here may be committed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union

import cv2
import numpy as np

KITTI_DEPTH_SCALE = 256.0
KITTI_CAMERA_HEIGHT_M = 1.65  # documented mounting height of the KITTI cameras (Geiger et al., IJRR 2013)

_IMAGE_NAME = re.compile(r"^(?P<drive>\d{4}_\d\d_\d\d_drive_\d{4}_sync)_image_(?P<frame>\d{10})_image_(?P<cam>0[23])\.png$")


def read_kitti_depth_png(path: Union[str, Path]) -> np.ndarray:
    """Read a KITTI depth PNG as float32 metres, with 0 meaning "no depth"."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Could not read depth PNG: {path}")
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel uint16 PNG, got {raw.dtype} with shape {raw.shape}")
    if raw.max() <= 255:
        # Same guard as the devkit's read_depth.py: an 8-bit-range map is not a valid KITTI depth map.
        raise ValueError(f"{path}: maximum value {raw.max()} <= 255 -- not a 16-bit KITTI depth map")
    return raw.astype(np.float32) / KITTI_DEPTH_SCALE


def read_kitti_intrinsics(path: Union[str, Path]) -> np.ndarray:
    """Read a 3x3 camera matrix stored as 9 row-major numbers."""
    values = np.loadtxt(str(path), dtype=np.float64).ravel()
    if values.size != 9:
        raise ValueError(f"{path}: expected 9 values for a 3x3 camera matrix, got {values.size}")
    K = values.reshape(3, 3)
    if not (K[0, 0] > 0 and K[1, 1] > 0 and np.allclose(K[2], [0, 0, 1]) and K[0, 1] == 0 and K[1, 0] == 0):
        raise ValueError(f"{path}: not a pinhole camera matrix: {K.tolist()}")
    return K


@dataclass(frozen=True)
class KittiDepthSample:
    name: str
    drive: str
    frame: str
    camera: str          # "02" = left colour camera, "03" = right
    image_path: Path
    gt_path: Path
    velodyne_path: Path
    intrinsics_path: Path

    def load_image(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {self.image_path}")
        return image  # BGR uint8, as the rest of the pipeline expects

    def load_gt(self) -> np.ndarray:
        return read_kitti_depth_png(self.gt_path)

    def load_velodyne(self) -> np.ndarray:
        return read_kitti_depth_png(self.velodyne_path)

    def load_intrinsics(self) -> np.ndarray:
        return read_kitti_intrinsics(self.intrinsics_path)


class KittiDepthSelection:
    """All samples of one split of the depth-selection set, sorted by file name."""

    def __init__(self, root: Union[str, Path], split: str = "val_selection_cropped") -> None:
        self.root = Path(root)
        self.split_dir = self.root / split
        image_dir = self.split_dir / "image"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"KITTI split not found: {image_dir}")

        samples: List[KittiDepthSample] = []
        problems: List[str] = []
        for image_path in sorted(image_dir.glob("*.png")):
            m = _IMAGE_NAME.match(image_path.name)
            if not m:
                problems.append(f"unexpected file name: {image_path.name}")
                continue
            drive, frame, cam = m.group("drive"), m.group("frame"), m.group("cam")
            sample = KittiDepthSample(
                name=image_path.stem, drive=drive, frame=frame, camera=cam, image_path=image_path,
                gt_path=self.split_dir / "groundtruth_depth" / f"{drive}_groundtruth_depth_{frame}_image_{cam}.png",
                velodyne_path=self.split_dir / "velodyne_raw" / f"{drive}_velodyne_raw_{frame}_image_{cam}.png",
                intrinsics_path=self.split_dir / "intrinsics" / f"{drive}_image_{frame}_image_{cam}.txt",
            )
            for p in (sample.gt_path, sample.velodyne_path, sample.intrinsics_path):
                if not p.is_file():
                    problems.append(f"missing companion file: {p.name}")
            samples.append(sample)

        if problems:
            raise FileNotFoundError(f"{len(problems)} problem(s) in {self.split_dir}: " + "; ".join(problems[:5]))
        if not samples:
            raise FileNotFoundError(f"No images in {image_dir}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> KittiDepthSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[KittiDepthSample]:
        return iter(self.samples)

    @property
    def drives(self) -> List[str]:
        return sorted({s.drive for s in self.samples})
