"""A1-A6 on the single test run, with paired sequence-bootstrap CIs. Reads outputs only; no tuning."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from velocity_eval import ttc_band, estimated_ttc
R = sys.argv[1]
S = json.load(open(f"{R}/summary.json")); sm, lat, pr = S["summaries"], S["latency_ms"], S["paired_vs_B0"]
d = pd.read_csv(f"{R}/per_object.csv.gz")
d["gt_ttc"] = d.gt_ttc.where(d.gt_ttc.notna(), None)
CANDS = ["K_kalman", "S_klt_scale", "KS_kalman_scale", "KS_kalman_scale_gated"]
BINS = [(3, 10), (10, 20), (20, 40), (40, 60)]
rng = np.random.default_rng(0); NB = 2000

def per_seq(sub, est):
    """Per-sequence sums needed for every criterion (ratio-of-sums estimators)."""
    out = {}
    for s, g in sub.groupby("sequence"):
        v = g[g.matched & g[f"{est}_valid"]]
        e = (-v[f"{est}_vz"] - v.gt_closing)
        strong = v.gt_closing.abs() >= 1
        ip = v[v.gt_in_corridor.astype(bool)]
        gtb = ip.gt_ttc.map(lambda t: ttc_band(None if pd.isna(t) else t))
        eb = [ttc_band(estimated_ttc(z, vz)) for z, vz in zip(ip.pipeline_z_m, ip[f"{est}_vz"])]
        none = (gtb == "NONE").values; eb = np.array(eb)
        urg = gtb.isin(["CRITICAL", "WARNING"]).values
        rec = {"n_elig": len(g), "n_valid": len(v),
               "sign_ok": float((np.sign(-v[f"{est}_vz"][strong]) == np.sign(v.gt_closing[strong])).sum()), "n_sign": float(strong.sum()),
               "fu": float((none & np.isin(eb, ["CRITICAL", "WARNING"])).sum()), "n_none": float(none.sum()),
               "mu": float((urg & np.isin(eb, ["CAUTION", "NONE"])).sum()), "n_urg": float(urg.sum())}
        for lo, hi in BINS:
            m = (v.gt_z_near >= lo) & (v.gt_z_near < hi)
            rec[f"ae_{lo}"] = float(e[m].abs().sum()); rec[f"n_{lo}"] = float(m.sum())
        out[s] = rec
    return pd.DataFrame(out).T

def boot(a, b, f):
    """Paired bootstrap over sequences of f(sums_b) - f(sums_a) (or ratio inside f)."""
    seqs = a.index.values; vals = []
    for _ in range(NB):
        pick = rng.choice(seqs, len(seqs))
        vals.append(f(a.loc[pick].sum(), b.loc[pick].sum()))
    vals = np.array(vals, dtype=float); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(f(a.sum(), b.sum())), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

res = {}
for subset in ("primary", "vru"):
    sub = d[d.subset == subset]
    A0 = per_seq(sub, "B0")
    res[subset] = {"B0": {"coverage": A0.n_valid.sum() / A0.n_elig.sum(), "sign": A0.sign_ok.sum() / A0.n_sign.sum(),
                          "false_urgent": A0.fu.sum() / A0.n_none.sum(), "missed_urgent": A0.mu.sum() / max(A0.n_urg.sum(), 1),
                          "n_urgent": A0.n_urg.sum(), "n_none": A0.n_none.sum(),
                          "bins": {f"{lo}-{hi}": (A0[f'ae_{lo}'].sum() / A0[f'n_{lo}'].sum(), A0[f'n_{lo}'].sum()) for lo, hi in BINS}}}
    for c in CANDS:
        B = per_seq(sub, c)
        p = pr[f"{subset} / {c}"]
        r = {"A1_rel_change": (p["rel_change"], p["rel_ci_low"], p["rel_ci_high"]),
             "A2_coverage_diff": boot(A0, B, lambda a, b: b.n_valid / b.n_elig - a.n_valid / a.n_elig),
             "coverage": B.n_valid.sum() / B.n_elig.sum(),
             "A3_false_urgent_diff": boot(A0, B, lambda a, b: b.fu / b.n_none - a.fu / a.n_none),
             "false_urgent": B.fu.sum() / B.n_none.sum(),
             "missed_urgent_diff": boot(A0, B, lambda a, b: (b.mu / b.n_urg - a.mu / a.n_urg) if a.n_urg and b.n_urg else np.nan),
             "missed_urgent": B.mu.sum() / max(B.n_urg.sum(), 1),
             "A5_sign_diff": boot(A0, B, lambda a, b: b.sign_ok / b.n_sign - a.sign_ok / a.n_sign),
             "sign": B.sign_ok.sum() / B.n_sign.sum(),
             "A6_latency_ms": (lat[c]["median_ms"], lat[c]["p95_ms"]), "B0_latency_ms": lat["B0"]["median_ms"]}
        bins = {}
        for lo, hi in BINS:
            n = B[f"n_{lo}"].sum()
            ratio = boot(A0, B, lambda a, b, lo=lo: (b[f"ae_{lo}"] / b[f"n_{lo}"]) / (a[f"ae_{lo}"] / a[f"n_{lo}"]) if a[f"n_{lo}"] and b[f"n_{lo}"] else np.nan)
            bins[f"{lo}-{hi}"] = {"n": n, "mae": B[f"ae_{lo}"].sum() / n if n else np.nan, "ratio_vs_B0": ratio}
        r["A4_bins"] = bins
        if subset == "primary":
            r["pass"] = {
                "A1": p["rel_change"] <= -0.25 and p["rel_ci_high"] <= -0.10,
                "A2": r["coverage"] >= res[subset]["B0"]["coverage"] - 0.05,
                "A3": r["false_urgent"] <= res[subset]["B0"]["false_urgent"] and r["false_urgent"] <= 0.10,
                "A4": all(not (b["n"] >= 200 and b["ratio_vs_B0"][0] > 1.10) for b in bins.values()),
                "A5": r["sign"] >= res[subset]["B0"]["sign"],
                "A6": lat[c]["median_ms"] - lat["B0"]["median_ms"] <= 20.0}
        res[subset][c] = r
json.dump(res, open(f"{R}/acceptance_test.json", "w"), indent=1, default=float)
for subset in ("primary", "vru"):
    b = res[subset]["B0"]
    print(f"\n=== {subset}: B0 coverage {b['coverage']:.3f} sign {b['sign']:.3f} false-urgent {b['false_urgent']:.4f} (GT-NONE n={b['n_none']:.0f}) missed-urgent {b['missed_urgent']:.3f} (GT-urgent n={b['n_urgent']:.0f})")
    print("    B0 bins:", {k: (round(v[0], 2), int(v[1])) for k, v in b["bins"].items()})
    for c in CANDS:
        r = res[subset][c]; f = lambda t, k=1, d=3: f"{k*t[0]:+.{d}f} [{k*t[1]:+.{d}f}, {k*t[2]:+.{d}f}]"
        print(f"  {c}: " + (f"PASS={all(r['pass'].values())} {r['pass']}" if "pass" in r else ""))
        print(f"    A1 MAE change {f(r['A1_rel_change'],100,1)}%  | A2 cov {r['coverage']:.3f}, diff {f(r['A2_coverage_diff'])}")
        print(f"    A3 false-urgent {r['false_urgent']:.4f}, diff {f(r['A3_false_urgent_diff'],1,4)} | missed-urgent {r['missed_urgent']:.3f}, diff {f(r['missed_urgent_diff'])}")
        print(f"    A5 sign {r['sign']:.3f}, diff {f(r['A5_sign_diff'])} | A6 latency median {r['A6_latency_ms'][0]:.2f} ms (p95 {r['A6_latency_ms'][1]:.2f}), B0 {r['B0_latency_ms']:.2f}")
        print("    A4 bins: " + "; ".join(f"{k}: n={v['n']:.0f} MAE {v['mae']:.2f} ratio {v['ratio_vs_B0'][0]:.2f} [{v['ratio_vs_B0'][1]:.2f},{v['ratio_vs_B0'][2]:.2f}]" for k, v in r["A4_bins"].items()))
