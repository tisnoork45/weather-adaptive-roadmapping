"""Independent recomputation of the Task 5 Stage 2 KITTI velocity results.

Reads the frozen per-object table written by the single test-split run
(/workspace/experiments/kitti_velocity/stage2/04_test_once/per_object.csv.gz)
with pandas/numpy only (no project code), and recomputes for the primary subset:
  coverage, closing-speed MAE / RMSE / median / bias, sign accuracy (|GT| >= 1 m/s),
  and the paired MAE change of each estimator vs B0 (rows where both are valid).
Closing speed = -vz (camera frame, positive = approaching), as in velocity_eval.py.
Writes:
  data/velocity_metrics_recomputed.csv   one row per (subset, estimator)
  data/velocity_aligned_samples.csv      aligned GT vs B0 vs KS closing speeds (primary, both valid)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/workspace/experiments/kitti_velocity/stage2/04_test_once/per_object.csv.gz")
OUT = Path(__file__).resolve().parents[1] / "data"
EST = ["B0", "K_kalman", "S_klt_scale", "KS_kalman_scale", "KS_kalman_scale_gated"]

df = pd.read_csv(SRC, dtype={"sequence": str})
for c in ["matched"] + [f"{e}_valid" for e in EST]:
    df[c] = df[c].astype(str).str.lower() == "true"

rows = []
for subset, d in df.groupby("subset"):
    n = len(d)
    m = d[d.matched]
    for e in EST:
        v = m[m[f"{e}_valid"]]
        err = (-v[f"{e}_vz"]) - v.gt_closing
        strong = v.gt_closing.abs() >= 1.0
        sign = (np.sign(-v.loc[strong, f"{e}_vz"]) == np.sign(v.loc[strong, "gt_closing"])).mean()
        both = m[m["B0_valid"] & m[f"{e}_valid"]]
        a = ((-both["B0_vz"]) - both.gt_closing).abs().mean()
        b = ((-both[f"{e}_vz"]) - both.gt_closing).abs().mean()
        rows.append({
            "split": "test", "subset": subset, "estimator": e, "n_eligible_gt": n, "n_matched": len(m),
            "n_valid": len(v), "coverage_end_to_end": len(v) / n,
            "closing_mae_m_s": err.abs().mean(), "closing_rmse_m_s": float(np.sqrt((err ** 2).mean())),
            "closing_median_ae_m_s": err.abs().median(), "closing_bias_m_s": err.mean(),
            "sign_accuracy_gt_ge_1ms": sign, "n_sign_rows": int(strong.sum()),
            "paired_n_vs_B0": len(both), "paired_mae_B0_m_s": a, "paired_mae_est_m_s": b,
            "paired_rel_change_vs_B0": (b - a) / a,
            "unpaired_rel_change_vs_B0": None,
            "n_sequences": v.sequence.nunique(), "n_frames_with_rows": d[["sequence", "frame"]].drop_duplicates().shape[0],
        })
res = pd.DataFrame(rows)
for subset in res.subset.unique():
    b0 = res[(res.subset == subset) & (res.estimator == "B0")].closing_mae_m_s.iloc[0]
    k = res.subset == subset
    res.loc[k, "unpaired_rel_change_vs_B0"] = (res.loc[k, "closing_mae_m_s"] - b0) / b0
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "velocity_metrics_recomputed.csv", index=False, float_format="%.6f")

p = df[(df.subset == "primary") & df.matched & df.B0_valid & df.KS_kalman_scale_valid]
al = pd.DataFrame({"sequence": p.sequence, "frame": p.frame, "gt_track": p.gt_track, "type": p.type,
                   "gt_z_near_m": p.gt_z_near, "gt_closing_m_s": p.gt_closing,
                   "B0_closing_m_s": -p.B0_vz, "KS_closing_m_s": -p.KS_kalman_scale_vz})
al.to_csv(OUT / "velocity_aligned_samples.csv", index=False, float_format="%.6f")
pd.set_option("display.width", 250)
print(res[["subset", "estimator", "n_valid", "coverage_end_to_end", "closing_mae_m_s", "closing_rmse_m_s",
           "sign_accuracy_gt_ge_1ms", "paired_n_vs_B0", "paired_rel_change_vs_B0", "unpaired_rel_change_vs_B0"]])
print("aligned primary samples:", len(al), "sequences:", sorted(df.sequence.unique()),
      "frames with any eligible row:", df[["sequence", "frame"]].drop_duplicates().shape[0])
