"""Recompute FPS / latency of the T6.5 before/after benchmark from the raw per-frame CSVs.

Source: /workspace/experiments/performance_profiling/20261102-111418_T6.5_review/benchmark/
  r1/baseline_<mode>_<before|after>_frames.csv and r2_r3/baseline_<mode>_<before|after>_r{2,3}_frames.csv
Same harness, same machine (H100 MIG 3g.40gb slice), same video (road.mp4, 7,249 frames), before/after
runs interleaved. Steady state = frames 11-7249 (frame 1 includes CUDA warm-up), as in full_video_runs.md.
  frame_wall_ms : whole loop iteration (decode + process() + HUD drawing + video write)
  process_ms    : PerceptionPipeline.process() only
Writes data/fps_latency_runs.csv (one row per run) and data/fps_latency_summary.csv (mean over 3 runs).
"""
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/workspace/experiments/performance_profiling/20261102-111418_T6.5_review/benchmark")
OUT = Path(__file__).resolve().parents[1] / "data"
rows = []
for mode in ("relative", "metric"):
    for code in ("before", "after"):
        for run, f in (("r1", B / f"r1/baseline_{mode}_{code}_frames.csv"),
                       ("r2", B / f"r2_r3/baseline_{mode}_{code}_r2_frames.csv"),
                       ("r3", B / f"r2_r3/baseline_{mode}_{code}_r3_frames.csv")):
            d = pd.read_csv(f)
            s = d[d.frame >= 11]
            rows.append({"depth_mode": mode, "code": code, "run": run, "n_frames_total": len(d),
                         "n_frames_steady": len(s), "source_file": str(f),
                         "frame_wall_mean_ms": s.frame_wall_ms.mean(), "frame_wall_p95_ms": np.percentile(s.frame_wall_ms, 95),
                         "frame_wall_fps": 1000 / s.frame_wall_ms.mean(),
                         "process_mean_ms": s.process_ms.mean(), "process_p95_ms": np.percentile(s.process_ms, 95),
                         "process_fps": 1000 / s.process_ms.mean(),
                         "depth_kind": ",".join(sorted(d.depth_kind.astype(str).unique()))})
r = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
r.to_csv(OUT / "fps_latency_runs.csv", index=False, float_format="%.4f")
num = ["frame_wall_mean_ms", "frame_wall_p95_ms", "frame_wall_fps", "process_mean_ms", "process_p95_ms", "process_fps"]
g = r.groupby(["depth_mode", "code"])[num]
summ = g.mean().join(g.min(), rsuffix="_min").join(g.max(), rsuffix="_max").reset_index()
summ.to_csv(OUT / "fps_latency_summary.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(r.drop(columns=["source_file"]).round(2).to_string())
print(summ[["depth_mode", "code", "frame_wall_mean_ms", "frame_wall_fps", "frame_wall_fps_min", "frame_wall_fps_max", "process_mean_ms", "process_fps"]].round(2))
