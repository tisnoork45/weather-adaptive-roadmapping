"""Graphs 7 and 8 from the production pipeline's own recorded outputs.

Input: research_results/final_validation/runs/<run_id>/{manifest.json,frames.csv,objects.csv}, written by
record_production_outputs.py (run.py's RunRecorder, metric depth, unmodified pipeline) for road.mp4,
night_1.mp4 and night_2.mp4. These videos are UNLABELLED: everything here is a distribution of model
OUTPUTS (predicted weather, predicted risk, estimated TTC), not accuracy against ground truth.

Graph 7: share of frames per overall (worst-object) risk level, grouped by the pipeline's (smoothed)
         predicted weather label. A frame with no assessed object is SAFE by the engine's definition.
Graph 8: per-object TTC estimates, grouped by predicted weather. Inclusion rule (documented in the report):
         velocity_valid == True, ttc_s present (object closing at >= 0.15 m/s), finite, 0 < ttc_s < 60 s
         (60 s is the calculator's cap, so capped values carry no information and are excluded),
         distance_m finite and > 0.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from figstyle import DATA, INK, INK2, MUTED, RISK, RISK_ORDER, SERIES, note, save

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
TTC_CAP = 60.0


def load():
    frames, objects, meta = [], [], []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        m = json.loads((d / "manifest.json").read_text())
        src = Path(str(m["source"]["description"])).stem
        f = pd.read_csv(d / "frames.csv")
        o = pd.read_csv(d / "objects.csv")
        f["video"], o["video"] = src, src
        o = o.merge(f[["frame_index", "weather_label"]], on="frame_index", how="left")
        frames.append(f)
        objects.append(o)
        meta.append({"run_dir": d.name, "video": src, "status": m.get("status"), "frames": len(f),
                     "objects": len(o), "depth_kinds": ",".join(sorted(f.depth_kind.astype(str).unique()))})
    return pd.concat(frames, ignore_index=True), pd.concat(objects, ignore_index=True), pd.DataFrame(meta)


def graph7(frames):
    t = (frames.groupby(["weather_label", "overall_risk_level"]).size().unstack(fill_value=0)
         .reindex(columns=RISK_ORDER, fill_value=0))
    t["n_frames"] = t.sum(axis=1)
    t = t.sort_values("n_frames", ascending=False)
    share = t[RISK_ORDER].div(t.n_frames, axis=0)
    out = t.join(share.add_suffix("_share"))
    out.to_csv(DATA / "risk_distribution_by_weather.csv", float_format="%.6f")
    (frames.groupby(["video", "weather_label", "overall_risk_level"]).size().unstack(fill_value=0)
     .reindex(columns=RISK_ORDER, fill_value=0).to_csv(DATA / "risk_distribution_by_video_and_weather.csv"))
    mult = yaml.safe_load((REPO / "config.yaml").read_text())["risk"]["weather_multipliers"]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for lvl in RISK_ORDER:
        v = share[lvl].values
        ax.barh(y, v, left=left, color=RISK[lvl], edgecolor="#ffffff", linewidth=1.5, label=lvl, height=0.6, zorder=2)
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 0.06:
                ax.text(l + w / 2, y[i], f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if lvl in ("CAUTION", "WARNING") else "#ffffff")
        left += v
    ax.set_yticks(y, [f"{w}\n({int(n):,} frames, ×{mult.get(w, 1.0):.2f})" for w, n in zip(t.index, t.n_frames)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of frames (overall predicted risk = worst object in the frame)")
    ax.set_ylabel("Predicted weather (threshold multiplier)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Predicted risk distribution by predicted weather\n(model outputs on unlabelled demo videos, not ground-truth risk)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(bottom=0.3)
    note(fig, "Videos: road.mp4, night_1.mp4, night_2.mp4 (all frames, metric depth, production pipeline, uncalibrated camera). "
              "Weather and risk are pipeline predictions;\nno accident or risk ground truth exists. "
              "Source: research_results/final_validation/runs/*/frames.csv.")
    save(fig, "07_weather_risk_distribution.png")
    return t


def graph8(objects):
    o = objects.copy()
    o["velocity_valid"] = o.velocity_valid.astype(str).str.lower().isin(["true", "1", "1.0"])  # CSV stores 1.0 / 0.0 / empty
    reasons = pd.Series("included", index=o.index)
    reasons[~o.velocity_valid] = "no valid velocity"
    reasons[o.velocity_valid & o.ttc_s.isna()] = "not closing (no TTC)"
    has = o.velocity_valid & o.ttc_s.notna()
    reasons[has & ~np.isfinite(o.ttc_s.fillna(np.inf))] = "non-finite TTC"
    reasons[has & (o.ttc_s <= 0)] = "TTC <= 0"
    reasons[has & (o.ttc_s >= TTC_CAP)] = "TTC at 60 s cap"
    reasons[has & ~(np.isfinite(o.distance_m) & (o.distance_m > 0))] = "invalid distance"
    o["inclusion"] = reasons
    o.groupby(["weather_label", "inclusion"]).size().unstack(fill_value=0).to_csv(DATA / "ttc_inclusion_counts.csv")
    keep = o[o.inclusion == "included"]
    keep[["video", "frame_index", "timestamp_s", "track_id", "class_name", "weather_label", "distance_m",
          "vz_m_s", "ttc_s", "risk_level"]].to_csv(DATA / "ttc_samples.csv", index=False, float_format="%.4f")
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())["risk"]
    base, mult = cfg["ttc_thresholds_s"], cfg["weather_multipliers"]
    weathers = keep.weather_label.value_counts().index.tolist()
    stats = keep.groupby("weather_label").ttc_s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    stats.to_csv(DATA / "ttc_summary_by_weather.csv", float_format="%.4f")
    bins = np.arange(0, TTC_CAP + 1e-9, 1.0)
    fig, axes = plt.subplots(len(weathers), 1, figsize=(8.4, 2.3 * len(weathers) + 1.2), sharex=True, squeeze=False)
    for ax, w in zip(axes[:, 0], weathers):
        v = keep[keep.weather_label == w].ttc_s
        ax.hist(v, bins=bins, color=SERIES[0], edgecolor="#ffffff", linewidth=0.6, zorder=2,
                weights=np.full(len(v), 100.0 / len(v)))
        m = mult.get(w, 1.0)
        for lvl, ls in (("CRITICAL", "-"), ("WARNING", "--"), ("CAUTION", ":")):
            ax.axvline(base[lvl] * m, color=INK, lw=1, ls=ls, label=f"{lvl} < {base[lvl] * m:.2f} s")
        ax.set_ylabel("% of samples")
        ax.set_title(f"Predicted weather: {w}  (n = {len(v):,} object-frames, median {v.median():.1f} s)",
                     fontsize=10, loc="left")
        ax.legend(fontsize=7.5, loc="upper right", title=f"thresholds ×{m:.2f}", title_fontsize=7.5)
    axes[-1, 0].set_xlabel("Estimated time-to-collision (s), 1-s bins")
    axes[-1, 0].set_xlim(0, TTC_CAP)
    fig.suptitle("Distribution of estimated TTC by predicted weather\n(model outputs on unlabelled demo videos; "
                 "not validated real-world TTC accuracy)", fontsize=11.5)
    fig.subplots_adjust(bottom=0.09, top=0.92, hspace=0.5)
    note(fig, "Included: valid velocity, object closing (>= 0.15 m/s), 0 < TTC < 60 s (60 s = calculator cap),\n"
              "finite positive distance; exclusion counts in data/ttc_inclusion_counts.csv. Vertical lines: the\n"
              "weather-adjusted risk thresholds. Monocular metric depth + finite-difference velocity (B0), uncalibrated\n"
              "camera; on KITTI this velocity estimator's closing-speed MAE is 3.87 m/s. Source: runs/*/objects.csv.")
    save(fig, "08_ttc_distribution.png")
    return o, keep, stats


if __name__ == "__main__":
    frames, objects, meta = load()
    meta.to_csv(DATA / "production_runs_manifest.csv", index=False)
    print(meta.to_string(index=False))
    print(graph7(frames))
    o, keep, stats = graph8(objects)
    print(o.groupby(["weather_label", "inclusion"]).size().unstack(fill_value=0))
    print(stats.round(2))
