"""Recompute the headline KITTI depth metrics of the metric Depth Anything V2 model from per-image values.

Source: outputs/experiments/kitti_depth/20261030-194322_1892061/per_image_metrics.csv (1,000 frames of
KITTI depth val_selection_cropped, LiDAR ground truth, standard protocol GT in (0.001, 80] m).
Writes data/depth_metrics_recomputed.csv (mean over images per method). No graph is made from this.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "outputs/experiments/kitti_depth/20261030-194322_1892061/per_image_metrics.csv"
OUT = Path(__file__).resolve().parents[1] / "data/depth_metrics_recomputed.csv"
d = pd.read_csv(SRC)
s = d[d.protocol == "standard"].groupby("method")[["abs_rel", "rmse", "rmse_log", "delta1"]].mean()
s["frames"] = d[d.protocol == "standard"].groupby("method").size()
s.to_csv(OUT, float_format="%.4f")
print(s.round(3))
