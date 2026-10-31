"""Dev-only uncertainty calibration with 2-fold CV inside dev. Usage: calib.py <stage> <out.json> [scale_cue_mult]"""
import json, sys
from pathlib import Path
import numpy as np
stage, outp = sys.argv[1], sys.argv[2]
mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
sys.argv = sys.argv[:1]
import evaluate_kitti_velocity as ev
import velocity_experimental as vx
from kitti_tracking import KittiTrackingDataset
from velocity_eval import replay_many
C = Path("/workspace/experiments/kitti_velocity/cache/20261031-190612_1892061-dirty")
m = json.loads((C / "manifest.json").read_text()); V = m["config"]["velocity"]
DEV = [n for n in sorted(m["sequences"]) if int(n) % 2 == 0]
FOLD_A = {"0000", "0004", "0008", "0012", "0016", "0020"}
names = {"1": ["B0", "K_kalman", "S_klt_scale"], "2": ["B0", "KS_kalman_scale", "DKS_robust_kalman_scale"]}[stage]
adapters = vx.build_estimators(V, names, {"scale_cue_in_filter": mult})
rows = []
for seq in KittiTrackingDataset(m["dataset_root"], sequences=DEV, frame_period_s=m["frame_period_s"]):
    frames = ev.load_cached_frames(C, seq)
    est, _ = replay_many(adapters, frames)
    rows += ev.evaluate_sequence(seq, frames, est)
res = {}
for name in names[1:]:
    for fold, sel in (("A", lambda s: s in FOLD_A), ("B", lambda s: s not in FOLD_A), ("all", lambda s: True)):
        v = [r for r in rows if r["subset"] == "primary" and sel(r["sequence"]) and r[f"{name}_valid"]]
        e = np.array([-r[f"{name}_vz"] - r["gt_closing"] for r in v]); s = np.array([r[f"{name}_sigma"] for r in v])
        k = float(np.median(np.abs(e) / s) / 0.6745)
        res[f"{name}|{fold}"] = {"k68": k, "n": len(v)}
    for fit, ev_fold in (("A", "B"), ("B", "A")):
        k = res[f"{name}|{fit}"]["k68"]
        v = [r for r in rows if r["subset"] == "primary" and ((r["sequence"] in FOLD_A) == (ev_fold == "A")) and r[f"{name}_valid"]]
        z = np.array([abs(-r[f"{name}_vz"] - r["gt_closing"]) / (k * r[f"{name}_sigma"]) for r in v])
        res[f"{name}|fit{fit}_on{ev_fold}"] = {"within1": float(np.mean(z <= 1)), "within196": float(np.mean(z <= 1.96)),
                                               "within3": float(np.mean(z <= 3))}
    print(name, {k.split("|")[1]: v for k, v in res.items() if k.startswith(name)})
Path(outp).write_text(json.dumps(res, indent=1))
