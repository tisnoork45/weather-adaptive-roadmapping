"""Re-derive the original '14.4 FPS' figure (research_audit.md) from logs/pipeline.log.

The run is the 2026-10-29 full road.mp4 CLI run (output outputs/videos/road_full.mp4, mtime 09:24:29).
Log timestamps have 1-s resolution; the run is relative-depth (pre-metric code) and the time includes
decode, HUD drawing and video encoding. Not comparable with the T6.5 harness numbers in graph 4.
Writes data/fps_original_runpy_log.csv.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
lines = (REPO / "logs/pipeline.log").read_text(errors="replace").splitlines()
t = lambda s: datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
build = next(l for l in lines if l.startswith("2026-10-29 09:16:08") and "Building perception pipeline" in l)
ready = next(l for l in lines if l.startswith("2026-10-29 09:16:11") and "initialized successfully" in l)
end = next(l for l in lines if l.startswith("2026-10-29 09:24:29") and "End of stream" in l)
last = [l for l in lines if l.startswith("2026-10-29 09:24:28") and "Frame 7230" in l][0]
frames = 7249
rows = [{"interval": "pipeline build -> end of stream", "seconds": (t(end) - t(build)).total_seconds()},
        {"interval": "pipeline ready -> end of stream", "seconds": (t(end) - t(ready)).total_seconds()}]
for r in rows:
    r["frames"] = frames
    r["fps"] = frames / r["seconds"]
    r["timestamp_resolution_s"] = 1
rows.append({"interval": "last HUD FPS reading (Frame 7230, 60-frame window)", "seconds": None, "frames": None,
             "fps": float(last.split("FPS=")[1].split()[0]), "timestamp_resolution_s": None})
df = pd.DataFrame(rows)
df.to_csv(REPO / "research_results/final_validation/data/fps_original_runpy_log.csv", index=False, float_format="%.3f")
print(df.to_string(index=False))
