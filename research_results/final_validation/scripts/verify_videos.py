"""Decode every frame of the three source videos and both sets of final exports.

Checks frame count, resolution, FPS, FOURCC, decode success, blank frames (std < 2), and colour
preservation: median |export - source| over a top-centre strip with no HUD (y 0-40, x 320-640)
on frames 100/400/800. Writes data/video_verification.csv. Read-only.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data/video_verification.csv"
SOURCES = {"ROAD": "road.mp4", "NIGHT_1": "data/sample_videos/night_1.mp4", "NIGHT_2": "data/sample_videos/night_2.mp4"}
EXPORTS = {"overlay_depth": "output/final_capstone_demo", "no_tint": "output/final_capstone_demo_no_tint"}
STRIP, CHECK_FRAMES = (slice(0, 40), slice(320, 640)), (100, 400, 800)


def scan(path):
    c = cv2.VideoCapture(str(path))
    info = {"meta_frames": int(c.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": c.get(cv2.CAP_PROP_FPS),
            "width": int(c.get(3)), "height": int(c.get(4)),
            "fourcc": int(c.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")}
    n, blank, strips = 0, 0, {}
    while True:
        ok, f = c.read()
        if not ok:
            break
        if f.std() < 2.0:
            blank += 1
        if n in CHECK_FRAMES:
            strips[n] = f[STRIP].astype(np.int16)
        n += 1
    c.release()
    info.update(decoded_frames=n, blank_frames=blank, size_bytes=path.stat().st_size)
    return info, strips


rows = []
for key, src in SOURCES.items():
    s_info, s_strips = scan(REPO / src)
    rows.append({"video": key, "kind": "source", "path": src, **s_info})
    for kind, folder in EXPORTS.items():
        p = REPO / folder / f"CAPSTONE_FINAL_{key}_DEMO.mp4"
        e_info, e_strips = scan(p)
        diffs = [np.abs(e_strips[k] - s_strips[k]) for k in s_strips if k in e_strips]
        d = np.concatenate([x.ravel() for x in diffs])
        rows.append({"video": key, "kind": kind, "path": str(p.relative_to(REPO)), **e_info,
                     "frames_match_source": e_info["decoded_frames"] == s_info["decoded_frames"],
                     "resolution_matches_source": (e_info["width"], e_info["height"]) == (s_info["width"], s_info["height"]),
                     "fps_matches_source": abs(e_info["fps"] - s_info["fps"]) < 0.01,
                     "strip_median_abs_diff": float(np.median(d)), "strip_share_within_12": float((d <= 12).mean())})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(df.drop(columns=["path"]).to_string(index=False))
