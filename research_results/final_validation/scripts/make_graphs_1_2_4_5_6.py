"""Graphs 1, 2, 4, 5, 6 from the CSVs written by the recompute_* scripts (run those first).

Graph 6 also reads the sequence-level bootstrap CIs stored by the frozen test run
(/workspace/experiments/kitti_velocity/stage2/04_test_once/summary.json); they are copied into
data/velocity_error_comparison.csv next to the recomputed point values.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import DATA, INK, INK2, MUTED, SEQ, SERIES, note, save

VEL_SUMMARY = Path("/workspace/experiments/kitti_velocity/stage2/04_test_once/summary.json")


def graph1():
    cm = pd.read_csv(DATA / "weather_confusion_matrix.csv", index_col=0)
    classes = list(cm.columns)
    counts = cm.values
    norm = counts / counts.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f7fbff"] + SEQ)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            dark = norm[i, j] > 0.55
            ax.text(j, i, f"{counts[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)
    labels = [f"{c}\n(n={int(counts[i].sum())})" for i, c in enumerate(classes)]
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class (dataset label)")
    ax.set_title("Weather classification confusion matrix\nEfficientNet-B0, held-out test split (n = 798)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row-normalised share (recall)")
    cb.outline.set_visible(False)
    fig.subplots_adjust(bottom=0.2)
    note(fig, "Cells: image count and share of the actual class. Labels: BDD100K weather attribute + DAWN fog images.\n"
              "Source: outputs/weather_runs/final_test_eval.json (deployed WeatherClassifier.infer path). "
              "Fog has only 48 test images.")
    save(fig, "01_weather_confusion_matrix.png")


def graph2():
    m = pd.read_csv(DATA / "weather_per_class_metrics.csv")
    acc = float(m.loc[m["class"] == "accuracy", "accuracy"].iloc[0])
    m = m[m["class"] != "accuracy"]
    names = [("macro\naverage" if c == "macro_average" else f"{c}\n(n={int(s)})") for c, s in zip(m["class"], m.support)]
    x = np.arange(len(m))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (col, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]):
        bars = ax.bar(x + (k - 1) * w, m[col], w - 0.03, color=SERIES[k], label=lab, zorder=2)
    ax.axvline(len(m) - 1.5, color=MUTED, lw=0.8, ls=":")
    ax.axhline(acc, color=INK, lw=1, ls="--", zorder=3)
    ax.text(len(m) - 0.5, acc, f" accuracy\n {acc:.1%}", fontsize=8.5, color=INK, va="center", ha="left")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Weather classification performance per class (held-out test split, n = 798)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_xlim(-0.6, len(m) - 0.4)
    fig.subplots_adjust(bottom=0.2, right=0.9)
    mac = m[m["class"] == "macro_average"].iloc[0]
    note(fig, f"Macro averages (unweighted over six classes): precision {mac.precision:.3f}, recall {mac.recall:.3f}, "
              f"F1 {mac.f1:.3f}. Per-class values: data/weather_per_class_metrics.csv.\n"
              "Fog n = 48: low-support class, wider uncertainty. Source: outputs/weather_runs/final_test_eval.json.")
    save(fig, "02_weather_classification_performance.png")


def graph4():
    r = pd.read_csv(DATA / "fps_latency_runs.csv")
    s = pd.read_csv(DATA / "fps_latency_summary.csv")
    conds = [("relative", "before"), ("relative", "after"), ("metric", "before"), ("metric", "after")]
    lbl = ["Relative depth\nbefore T6.5", "Relative depth\nafter T6.5", "Metric depth\nbefore T6.5", "Metric depth\nafter T6.5"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    w = 0.36
    panels = [("FPS (frames per second)", [("frame_wall_fps", "Full loop (decode + pipeline + HUD + video write)"),
                                          ("process_fps", "Pipeline only (process())")], "Throughput"),
              ("Latency per frame (ms)", [("frame_wall_mean_ms", "Full loop mean"),
                                         ("frame_wall_p95_ms", "Full loop P95")], "Latency")]
    for ax, (ylab, series, title) in zip(axes, panels):
        for k, (col, lab) in enumerate(series):
            means = [s[(s.depth_mode == a) & (s.code == b)][col].iloc[0] for a, b in conds]
            ax.bar(x + (k - 0.5) * w, means, w - 0.04, color=SERIES[k], label=lab, zorder=2)
            for i, (a, b) in enumerate(conds):
                pts = r[(r.depth_mode == a) & (r.code == b)][col]
                ax.scatter(np.full(len(pts), x[i] + (k - 0.5) * w), pts, s=12, color=INK, zorder=3, lw=0)
                ax.text(x[i] + (k - 0.5) * w, max(pts) + (0.4 if "fps" in col else 1.5), f"{means[i]:.1f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, lbl, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Production pipeline speed before and after the T6.5 optimisation (road.mp4, 7,249 frames)", fontsize=12)
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.22)
    note(fig, "Bars: mean of 3 interleaved full-video runs per condition; dots: each run (steady state, frames 11-7,249). "
              "Same harness, video and GPU (NVIDIA H100 80GB, MIG 3g.40gb slice), FP32.\n"
              "Source: /workspace/experiments/performance_profiling/20261102-111418_T6.5_review/benchmark/. "
              "The earlier '14.4 FPS' figure used a different method (log wall-clock) and is not plotted.")
    save(fig, "04_fps_latency_comparison.png")


def graph5():
    a = pd.read_csv(DATA / "velocity_aligned_samples.csv")
    lim = 30.0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    for ax, (col, name, c) in zip(axes, [("B0_closing_m_s", "Baseline B0 (production finite-difference estimator)", SERIES[0]),
                                         ("KS_closing_m_s", "Experimental KS (Kalman + KLT scale change)", SERIES[1])]):
        out = ((a[col].abs() > lim) | (a.gt_closing_m_s.abs() > lim)).sum()
        mae = (a[col] - a.gt_closing_m_s).abs().mean()
        ax.scatter(a.gt_closing_m_s, a[col].clip(-lim, lim), s=4, alpha=0.25, color=c, lw=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1, ls="--", label="ideal (estimate = reference)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nMAE {mae:.2f} m/s on these {len(a):,} paired samples", fontsize=10)
        ax.set_xlabel("KITTI reference closing speed (m/s)")
        ax.text(0.03, 0.97, f"{out} of {len(a):,} samples beyond ±{lim:.0f} m/s\nare drawn at the frame edge",
                transform=ax.transAxes, fontsize=7.5, va="top", color=INK2)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Estimated closing speed (m/s)")
    fig.suptitle("Experimental velocity estimation on KITTI tracking (test split, offline, not in the production pipeline)",
                 fontsize=11.5)
    fig.subplots_adjust(bottom=0.2, top=0.84)
    note(fig, "Each dot: one ground-truth vehicle in one frame where both estimators returned a velocity (primary subset, "
              "10 test sequences). Closing speed = -dZ/dt, positive = approaching.\nReference: least-squares slope of the KITTI 3D-label position "
              "over frames t-2..t+2 (camera frame). Source: /workspace/experiments/kitti_velocity/stage2/04_test_once/per_object.csv.gz.")
    save(fig, "05_velocity_estimation_comparison.png")


def graph6():
    v = pd.read_csv(DATA / "velocity_metrics_recomputed.csv")
    summ = json.loads(VEL_SUMMARY.read_text())["summaries"]
    rows = []
    for e, lab in [("B0", "Baseline B0"), ("KS_kalman_scale", "Experimental KS")]:
        r = v[(v.subset == "primary") & (v.estimator == e)].iloc[0]
        ci = summ[f"primary / {e}"]["closing_speed_mae_ci"]
        rows.append({"estimator": e, "label": lab, "n_valid": int(r.n_valid), "mae_m_s": r.closing_mae_m_s,
                     "mae_ci_low": ci["ci_low"], "mae_ci_high": ci["ci_high"], "rmse_m_s": r.closing_rmse_m_s,
                     "median_ae_m_s": r.closing_median_ae_m_s, "sign_accuracy": r.sign_accuracy_gt_ge_1ms,
                     "paired_rel_change_vs_B0": r.paired_rel_change_vs_B0, "paired_n": int(r.paired_n_vs_B0)})
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "velocity_error_comparison.csv", index=False, float_format="%.6f")
    metrics = [("mae_m_s", "MAE"), ("rmse_m_s", "RMSE"), ("median_ae_m_s", "Median AE")]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for k, r in t.iterrows():
        vals = [r[m] for m, _ in metrics]
        ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=SERIES[k], label=f"{r.label} (n = {r.n_valid:,})", zorder=2)
        ax.errorbar(x[0] + (k - 0.5) * w, r.mae_m_s, yerr=[[r.mae_m_s - r.mae_ci_low], [r.mae_ci_high - r.mae_m_s]],
                    color=INK, capsize=4, lw=1, zorder=3)
        tops = [r.mae_ci_high, vals[1], vals[2]]
        for xi, v_, top in zip(x, vals, tops):
            ax.text(xi + (k - 0.5) * w, top + 0.2, f"{v_:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ks = t.iloc[1]
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylabel("Closing-speed error (m/s)")
    ax.set_ylim(0, 11)
    ax.set_title("Closing-speed error, KITTI tracking test split (experimental, offline)")
    ax.legend(loc="upper right")
    fig.subplots_adjust(bottom=0.22)
    note(fig, f"Paired MAE change vs B0 on the {ks.paired_n:,} samples where both estimators are valid: "
              f"{ks.paired_rel_change_vs_B0:+.1%}. Error bars: 95% bootstrap CI over sequences (MAE only).\n"
              "KS is not integrated into the production pipeline; production still uses B0. "
              "Source: /workspace/experiments/kitti_velocity/stage2/04_test_once/.")
    save(fig, "06_velocity_error_comparison.png")


if __name__ == "__main__":
    graph1(); graph2(); graph4(); graph5(); graph6()
