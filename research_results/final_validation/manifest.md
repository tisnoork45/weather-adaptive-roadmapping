# Final validation: graph manifest

Generated 2026-11-02 on branch `research-improvements`, commit `93f98efb7789346b3ccc6bb9c6ff8ba03937b6bc` (working tree has uncommitted Task 7 changes in `app.py`, `tests_dashboard.py`, `README.md` and `CAPSTONE_SUMMARY.md`; none of them touch the pipeline).

All paths are relative to `research_results/final_validation/` unless they start with `/` or name a repository file.

To regenerate everything from the repository root, run:

```
python -B research_results/final_validation/scripts/recompute_weather_metrics.py
python -B research_results/final_validation/scripts/recompute_velocity_metrics.py
python -B research_results/final_validation/scripts/recompute_fps_latency.py
python -B research_results/final_validation/scripts/make_graphs_1_2_4_5_6.py
python -B research_results/final_validation/scripts/make_graphs_7_8.py      # needs runs/ (record_production_outputs.py)
```

| # | Graph file | Status | Source dataset | Source file(s) | Metrics shown | Evaluation split | Graph data (CSV) | Scripts |
|---|---|---|---|---|---|---|---|---|
| 1 | `graphs/01_weather_confusion_matrix.png` | **VERIFIED** | BDD100K weather attribute + DAWN fog images (`data/weather_dataset_c3000`) | `outputs/weather_runs/final_test_eval.json` (key `weather_classifier`) | Counts and row-normalised share (recall) | Held-out **test** split, 798 images; evaluated once; checkpoint SHA-256 `0bda562e…` = deployed `weights/efficientnet_weather.pth` | `data/weather_confusion_matrix.csv` | `scripts/recompute_weather_metrics.py`, `scripts/make_graphs_1_2_4_5_6.py` (`graph1`) |
| 2 | `graphs/02_weather_classification_performance.png` | **VERIFIED** | as graph 1 | as graph 1 | Per-class precision, recall, F1; macro averages; overall accuracy | as graph 1 | `data/weather_per_class_metrics.csv`, `data/weather_metric_check.csv` | as graph 1 (`graph2`) |
| 3 | `graphs/03_object_detection_performance.png` | **NOT READY (not generated)** | — | — | — | No precision / recall / mAP evaluation against ground truth exists | — | — |
| 4 | `graphs/04_fps_latency_comparison.png` | **VERIFIED** | `road.mp4`, 7,249 frames | `/workspace/experiments/performance_profiling/20261102-111418_T6.5_review/benchmark/{r1,r2_r3}/baseline_*_frames.csv` (12 runs) | FPS (full loop and `process()` only); mean and P95 full-loop latency (ms) | Not a dataset split: 3 interleaved before/after runs per depth mode, frames 11–7,249 | `data/fps_latency_runs.csv`, `data/fps_latency_summary.csv` | `scripts/recompute_fps_latency.py`, `scripts/make_graphs_1_2_4_5_6.py` (`graph4`) |
| 5 | `graphs/05_velocity_estimation_comparison.png` | **VERIFIED** (experimental, offline) | KITTI tracking (training set, test half) | `/workspace/experiments/kitti_velocity/stage2/04_test_once/per_object.csv.gz` | Reference vs estimated closing speed (m/s), B0 and KS, 8,467 paired samples | KITTI tracking **test** split = odd sequences 0001–0019 (4,784 frames); primary subset | `data/velocity_aligned_samples.csv` | `scripts/recompute_velocity_metrics.py`, `scripts/make_graphs_1_2_4_5_6.py` (`graph5`) |
| 6 | `graphs/06_velocity_error_comparison.png` | **VERIFIED** (experimental, offline) | as graph 5 | as graph 5, plus bootstrap CIs from `/workspace/experiments/kitti_velocity/stage2/04_test_once/summary.json` | Closing-speed MAE (95 % CI), RMSE, median AE (m/s); paired MAE change | as graph 5 | `data/velocity_error_comparison.csv`, `data/velocity_metrics_recomputed.csv` | as graph 5 (`graph6`) |
| 7 | `graphs/07_weather_risk_distribution.png` | **GENERATED: descriptive model outputs only** (no ground truth) | Unlabelled demo videos `road.mp4`, `night_1.mp4`, `night_2.mp4` | `runs/*/frames.csv` (production pipeline, metric depth, recorded 2026-11-02) | Share of frames per predicted overall risk level, by predicted weather | None (not an evaluation) | `data/risk_distribution_by_weather.csv`, `data/risk_distribution_by_video_and_weather.csv` | `scripts/record_production_outputs.py`, `scripts/make_graphs_7_8.py` (`graph7`) |
| 8 | `graphs/08_ttc_distribution.png` | **GENERATED: descriptive model outputs only** (no ground truth) | as graph 7 | `runs/*/objects.csv` | Histogram of estimated TTC (s) per predicted weather, with weather-adjusted thresholds | None (not an evaluation) | `data/ttc_samples.csv`, `data/ttc_summary_by_weather.csv`, `data/ttc_inclusion_counts.csv` | as graph 7 (`graph8`) |

## Other evidence files (not plotted)

| File | Content | Script |
|---|---|---|
| `data/weather_split_leakage.csv` | Train/val/test overlap checks for the weather dataset | `scripts/check_weather_split_leakage.py` |
| `data/depth_metrics_recomputed.csv` | KITTI depth metrics recomputed from per-image values | `scripts/recompute_depth_metrics.py` |
| `data/fps_original_runpy_log.csv` | The original "14.4 FPS" figure re-derived from `logs/pipeline.log` | `scripts/extract_original_fps_log.py` |
| `data/video_verification.csv` | Frame count, resolution, FPS, blank frames and colour check of the sources and both export folders | `scripts/verify_videos.py` |
| `data/production_runs_manifest.csv` | The three recorded production runs (status, frames, objects, depth kind) | `scripts/make_graphs_7_8.py` |
| `runs/<run_id>/` | Production pipeline recordings: `manifest.json`, `frames.csv` and `objects.csv` | `scripts/record_production_outputs.py` |
| `logs/record_*.log` | Console logs of the three recordings | — |
| `logs/tests/*.log` | Complete test and golden-regression logs | `scripts/run_validation_tests.sh` with `scripts/mkcopy.sh` (archived copies; they run the tests in a scratch copy of the tree under the session scratchpad, never in the project) |
| `scripts/figstyle.py` | Shared figure style (fonts, palette, 300 DPI) | — |
