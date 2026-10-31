"""Dev-only parameter sweep with 2-fold cross-validation inside dev. Usage: sweep.py <group> <out.json>"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.argv, group, outp = sys.argv[:1], sys.argv[1], sys.argv[2]
import evaluate_kitti_velocity as ev
import velocity_experimental as vx
from kitti_tracking import KittiTrackingDataset
from velocity_eval import B0Adapter, replay_many, paired_cluster_bootstrap
C = Path("/workspace/experiments/kitti_velocity/cache/20261031-190612_1892061-dirty")
m = json.loads((C / "manifest.json").read_text()); V = m["config"]["velocity"]
DEV = [n for n in sorted(m["sequences"]) if int(n) % 2 == 0]
FOLD_A = {"0000", "0004", "0008", "0012", "0016", "0020"}
ds = KittiTrackingDataset(m["dataset_root"], sequences=DEV, frame_period_s=m["frame_period_s"])
P, K, R = vx.RobustDepthParams, vx.KalmanParams, vx.KLTParams
groups = {
 "depth": {f"D_p{p}_{tag}": (lambda p=p, vc=vc: vx.RobustDepthAdapter(V, P(percentile=p, vru_cols=vc)))
           for p in (20, 30, 40, 50) for tag, vc in (("same", None), ("vru035", (0.35, 0.65)))},
 "gate": {f"T_k{k}_r{r}": (lambda k=k, r=r: vx.DepthGateAdapter(V, k=k, rel_sigma=r)) for k in (1.5, 2, 3, 5) for r in (0.03, 0.05, 0.1)},
 "kalman": {f"K_q{q}_r{r}": (lambda q=q, r=r: vx.KalmanAdapter(V, params=K(q_lon=q), rel_sigma_b0=r))
            for q in (1, 4, 16) for r in (0.04, 0.08, 0.15)},
 "scale": {f"S_{ax}_fb{fb}": (lambda ax=ax, fb=fb: vx.ScaleAdapter(V, R(axis=ax, fb_max_px=fb)))
           for ax in ("both", "vertical") for fb in (0.5, 1.0, 2.0)},
}
adapters = {"B0": B0Adapter(V["history_length"], V["max_gap_seconds"], V["ema_alpha"])}
adapters.update({k: f() for k, f in groups[group].items()})
rows, lat = [], {k: [] for k in adapters}
t0 = time.time()
for seq in ds:
    frames = ev.load_cached_frames(C, seq)
    est, times = replay_many(adapters, frames)
    for k, v in times.items(): lat[k] += v
    rows += ev.evaluate_sequence(seq, frames, est)
res = {}
for name in adapters:
    for sub in ("primary", "vru"):
        for fold, sel in (("A", lambda s: s in FOLD_A), ("B", lambda s: s not in FOLD_A), ("all", lambda s: True)):
            rr = [r for r in rows if r["subset"] == sub and sel(r["sequence"])]
            both = [r for r in rr if r["B0_valid"] and r[f"{name}_valid"]]
            e = np.array([-r[f"{name}_vz"] - r["gt_closing"] for r in both]); e0 = np.array([-r["B0_vz"] - r["gt_closing"] for r in both])
            sig = np.array([r[f"{name}_sigma"] for r in both], dtype=float)
            res[f"{name}|{sub}|{fold}"] = {"n": len(both), "cov": sum(r[f"{name}_valid"] for r in rr if r["matched"]) / max(len(rr), 1),
                "mae": float(np.abs(e).mean()) if len(e) else None, "mae_b0": float(np.abs(e0).mean()) if len(e0) else None,
                "bias": float(e.mean()) if len(e) else None, "medae": float(np.median(np.abs(e))) if len(e) else None,
                "k68": float(np.median(np.abs(e) / sig) / 0.6745) if np.isfinite(sig).any() else None}
    res[f"{name}|latency_ms"] = float(np.median(lat[name]))
Path(outp).write_text(json.dumps(res, indent=1))
print(group, "done in %.0fs" % (time.time() - t0))
for name in adapters:
    line = [name]
    for sub in ("primary", "vru"):
        for fold in ("A", "B"):
            r = res[f"{name}|{sub}|{fold}"]
            line.append(f"{sub[:3]}{fold}: {r['mae']:.2f}/{r['mae_b0']:.2f} ({100*(r['mae']/r['mae_b0']-1):+.0f}%) b{r['bias']:+.2f}" if r["mae"] else f"{sub}{fold}: -")
    print(" | ".join(line), f"| {res[name+'|latency_ms']:.2f}ms")
