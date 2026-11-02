# Final project validation report

**Project:** Weather Adaptive Road Mapping for Autonomous Vehicles
**Date:** 2026-11-02
**Scope:** validate the existing system and its reported results, then produce research graphs from verified data only. No features were added and no production file was changed.

**Status words used below:**
- **VERIFIED:** raw data found, metric recomputed independently and matching.
- **PARTIALLY VERIFIED:** raw data found but something is missing or differs; the gap is stated.
- **NOT VERIFIED:** no raw data, or the claim does not match it.

---

## 0. Frozen state (Phase 1)

| Item | Value |
|---|---|
| Branch | `research-improvements`, 4 commits ahead of `origin/research-improvements`, nothing pushed |
| HEAD | `93f98efb7789346b3ccc6bb9c6ff8ba03937b6bc` (perf: optimize depth preprocessing, positional cache, and BEV mapping) |
| Uncommitted changes | ` M README.md`, ` M app.py`, ` M tests_dashboard.py`, `?? CAPSTONE_SUMMARY.md`. These are the Task 7 dashboard and documentation changes, still awaiting review. None touches the pipeline, models or config. |
| New in this task | `?? research_results/` (this folder) |
| Configuration | `config.yaml`, SHA-256 `ecd7b59f…`, unchanged. The app default depth mode stays `relative`; metric mode is enabled per run with `--depth-mode metric`. |
| Python / PyTorch / CUDA | Python 3.10.12; torch 2.4.0a0+f70bd71a48.nv24.06; CUDA 12.5 |
| Other libraries | OpenCV 4.10.0; ultralytics 8.4.192; numpy 1.26.4; Streamlit 1.60.0; matplotlib 3.9.0 |
| GPU | NVIDIA H100 80GB HBM3, MIG 3g.40gb slice, driver 535.161.08. All results in FP32. |
| Experiment directories | `/workspace/experiments/{kitti_velocity, performance_profiling, task7_capstone_readiness}` (459 files); `outputs/weather_runs/`; `outputs/experiments/kitti_depth/`; `outputs/regression/` (goldens) |
| Test files | `tests.py`, `tests_depth.py`, `tests_kitti_tracking.py`, `tests_velocity_experimental.py`, `tests_ego_motion.py`, `tests_foundation.py`, `tests_regression.py`, `tests_dashboard.py`; `scripts/regression/golden.py` |

**Final model weights** (SHA-256 recorded before and after; identical):

| Role | File | SHA-256 (prefix) |
|---|---|---|
| Detection, YOLO11x (COCO-pretrained) | `weights/yolo11x.pt` | `7bc158aa…` |
| Depth, relative (Depth Anything V2 ViT-S) | `weights/depth_anything_v2_vits.pth` | `715fade1…` |
| Depth, metric (Depth Anything V2 ViT-S, Virtual KITTI 2) | `weights/depth_anything_v2_metric_vkitti_vits.pth`, a link to `/workspace/checkpoints/…` | `9203e538…` |
| Weather, EfficientNet-B0, 6 classes | `weights/efficientnet_weather.pth` (= `outputs/weather_runs/B_c3000-s42/model.pth`) | `0bda562e…` |

---

## 1. Software validation results (Phase 2)

The production system is **camera-only**. It contains no LiDAR, radar or IMU integration. The fusion module weighs camera-derived cues only. KITTI LiDAR and OXTS data were used only as **offline ground truth** in experiments.

| # | Component | Implementation | Validated by | Result |
|---|---|---|---|---|
| 1 | YOLO11x detection | `models.py` (ultralytics), COCO weights, conf 0.35, 640 px | Unit/integration tests; goldens (0 differences); runs report `backend: neural network, device: cuda` | **Works.** No accuracy evaluation exists (see graph 3). |
| 2 | Depth Anything V2 | `models.py` / `depth_anything_v2/`; relative ViT-S and metric ViT-S (vKITTI) | 67 tests in `tests_depth.py`; goldens; KITTI depth evaluation (section 4) | **Works; metric depth evaluated on KITTI** |
| 3 | EfficientNet-B0 weather | `models.py::WeatherClassifier`, 6 classes, temporal smoothing | Tests; held-out test evaluation (section 4) | **Works; evaluated** |
| 4 | ByteTrack-style tracking | `tracking.py` | Tests in `tests.py` / `tests_kitti_tracking.py`; goldens | **Works.** No MOT metrics (MOTA/IDF1) measured. |
| 5 | BEV mapping | `mapping.py`, pinhole back-projection, metric mode only | `test_metric_pipeline_produces_metric_bev_and_risk`, `test_relative_pipeline_never_builds_bev_or_point_cloud`, and others; all 9,414 recorded demo frames have `bev_available = 1` | **Works.** Positions approximate: camera uncalibrated (`camera.calibrated: false`). |
| 6 | Distance estimation | BEV z of each object from metric depth | KITTI velocity harness: pipeline z vs label `z_near`, mean abs-rel 0.090 (primary vehicles, test split) | **Partially validated.** KITTI only (favourable camera domain); no ground truth for the demo videos. |
| 7 | Relative velocity | `relative_velocity.py` (B0 finite difference on BEV track history) | Tests; KITTI test split (section 5) | **Works; large errors** (MAE 3.87 m/s) |
| 8 | TTC | `ttc_calculator.py` (≥ 0.15 m/s closing, capped at 60 s) | `test_ttc_basic_math`, `test_ttc_none_when_not_closing`, `test_ttc_capped_at_max`, … | **Formula verified.** Real-world TTC accuracy not validated. |
| 9 | Risk assessment | `risk_engine.py` (worst object; TTC bands + distance floor) | `test_risk_engine_ttc_boundaries`, `…distance_floor…`, `test_end_to_end_ttc_risk_escalation_with_pipeline_config` | **Logic verified.** Accuracy against real risk not validated. |
| 10 | Weather-dependent thresholds | `risk.weather_multipliers`: clear 1.00, overcast 1.05, night 1.25, rain 1.30, snow 1.50, fog 1.60 | `test_risk_engine_weather_adaptive_thresholds_genuinely_change_risk`; config validation tests | **Verified as implemented** (a design rule, not a learned or validated quantity) |
| 11 | CLI video processing | `run.py` | Three full metric-mode runs recorded today (status `complete`, 9,414 frames); final exports verified (below) | **Works** |
| 12 | Streamlit dashboard | `app.py` | 17 `tests_dashboard.py` + 49 `tests_foundation.py` tests (66 passed), Streamlit AppTest | **Works.** Live browser launch was tested in Task 7, not repeated here. |

### Sample videos and final exports

All frames of each file were decoded with OpenCV (`data/video_verification.csv`).

| Video | Source frames / res / FPS | Export folder | Frames | Res | FPS | Blank | Colour diff vs source* |
|---|---|---|---|---|---|---|---|
| road | 7,249 / 1024×576 / 23.997 (repo-root `road.mp4`) | `output/final_capstone_demo` (with `--overlay-depth`) | 7,249 ✓ | ✓ | ✓ | 0 | median 44 levels (tinted) |
| | | `output/final_capstone_demo_no_tint` | 7,249 ✓ | ✓ | ✓ | 0 | **median 2 levels; 99.9 % within 12** |
| night_1 | 837 / 1024×576 / 25 | overlay | 837 ✓ | ✓ | ✓ | 0 | median 25 |
| | | no_tint | 837 ✓ | ✓ | ✓ | 0 | **median 2; 100 % within 12** |
| night_2 | 1,328 / 1024×576 / 25 | overlay | 1,328 ✓ | ✓ | ✓ | 0 | median 22 |
| | | no_tint | 1,328 ✓ | ✓ | ✓ | 0 | **median 2; 100 % within 12** |

\* Measured over a top-centre strip with no HUD (rows 0–40, cols 320–640), frames 100/400/800, 8-bit levels.

The **latest exports (`final_capstone_demo_no_tint`) preserve the original colours**, apart from compression noise.

**Metric mode and annotations:**
- Every progress line of the exports and of today's recordings reports `Depth=metric_m`.
- The recording manifests show all three models running as neural networks on CUDA, with no fallback.
- The expected HUD, boxes, track IDs, weather label, risk card and guide were confirmed visually in Task 7.

**Issues and limitations found**
- The exports use the `mp4v` codec, because no H.264 encoder is available in this environment. Play them in VLC.
- The CLI does not draw BEV, weather probabilities or adaptive weights into the saved video.
- On the unlabelled demo videos the weather classifier gives implausible labels for some frames:
  - **road.mp4:** 572 frames "rain" and 552 "overcast";
  - **night clips:** 189 frames "fog" and 20 "rain".
  - These are probable misclassifications. They cannot be scored without labels.
- **Very high predicted risk:** 35 % of the predicted-clear road.mp4 frames are CRITICAL (graph 7). This matches the known B0 velocity noise (README: depth noise produces false CRITICAL frames).

---

## 2. Test results

The tests were run on 2026-11-02, after the recordings, in a scratch copy of the current working tree, so nothing was written into the project. Complete logs are in `logs/tests/`.

| Check | Result |
|---|---|
| Full pytest suite, 8 test files, default device | **295 passed, 2 skipped**, 1 warning (`full_default.log`) |
| Skips | `tests_kitti_tracking.py:571` (needs `KITTI_VELOCITY_CACHE`; run separately below); `tests_ego_motion.py:104` (degenerate case by design) |
| G6 KITTI equivalence (with cache) | 1 passed |
| Golden regression, relative | road.mp4, 240/240 frames, **0 differences** |
| Golden regression, metric | 240/240 frames, **0 differences** |
| G6 replay | KITTI 0000, frames 0–49: **0 differences**, depth identical |
| Pipeline `reset()` / shared-model threads | 0 differences (3 checks) |
| Dashboard + recording lifecycle | **66 passed** |

**Known open issue:** the intermittent pytest segfault seen once on 2026-11-01 did not occur in this run. It remains **UNRESOLVED**; this run does not show it is fixed.

---

## 3. Dataset verification

| Dataset | Use | Split | Leakage check | Verdict |
|---|---|---|---|---|
| **BDD100K weather attribute + DAWN** (`data/weather_dataset_c3000`) | Weather classifier train / val / test | train 15,730 · val 800 · test 798 | See below | **No exact or near-duplicate leakage found** |
| **KITTI tracking, training set** (21 sequences, 8,008 frames, `/workspace/datasets/kitti/tracking`) | Offline velocity / TTC ground truth (Task 5) | dev = even sequences 0000–0020 (11 seq, 3,224 frames); test = odd 0001–0019 (10 seq, **4,784 frames**), fixed before any result | Tuning only on dev (with 2-fold CV inside dev); test scored once (`--score-test`, frozen config `04_test_frozen_config.json`) | **Protocol sound** |
| **KITTI depth, val_selection_cropped** (1,000 frames, 13 drives) | Depth evaluation | Evaluation only | The metric model was trained on Virtual KITTI 2 (synthetic), not on KITTI | **OK** |
| road.mp4, night_1.mp4, night_2.mp4 | Demo and regression only | — | Unlabelled; never used for training or tuning | No accuracy claims made from them |

**How the weather split was made**
- **BDD100K test and val images** come from BDD's official **val** split (`prepare_weather_dataset.py`, `--val-fraction-as-test 0.3`).
- **BDD100K training images** come from BDD's official **train** split.
- **Model selection:** the checkpoint was chosen on **validation only** (4 runs: `outputs/weather_runs/summarize.py` asserts that no test results were read).
- **Single test pass:** the test split was evaluated once (`final_eval.py` refuses to re-run).

**Leakage scan** (`scripts/check_weather_split_leakage.py`, results in `data/weather_split_leakage.csv`):
- **Exact copies:** 0 shared file names and 0 byte-identical files between test and train/val.
- **Same source:** no shared BDD100K video id.
- **Near duplicates:** a 64-bit difference hash (Hamming distance ≤ 4) flagged 14 test images versus train and 1 versus val.
- **Manual check:** 6 representative pairs were inspected visually; all show different scenes. Most flagged pairs also cross classes, which is typical of low-texture fog and night images.
- **Residual risk:** not every flagged pair was inspected. Scene-level similarity inside DAWN (same photographer or location) cannot be excluded by hashing.

**Label quality:** BDD100K weather tags are crowd-sourced and noisy. "Actual class" in graph 1 means the dataset label, not verified weather.

---

## 4. Model evaluation

| Model | Evaluation | Ground truth | Result | Source |
|---|---|---|---|---|
| EfficientNet-B0 weather | Held-out test, 798 images, deployed `WeatherClassifier.infer()` path | Dataset labels | Accuracy **82.33 %**; macro-F1 **83.30 %**; macro-P 84.6 %; macro-R 82.7 % | `outputs/weather_runs/final_test_eval.json` |
| Same weights, batched training-eval path | Same images | Same | 82.21 % / 83.19 %; 99.87 % prediction agreement with the deployed path | same file |
| Depth Anything V2 metric (ViT-S, vKITTI) | KITTI depth val_selection_cropped, 1,000 frames; GT in (0.001, 80] m | LiDAR | AbsRel **0.132** [0.117, 0.147]; RMSE 5.26 m; δ<1.25 0.806. Recomputed from per-image values: identical. | `outputs/experiments/kitti_depth/20261030-194322_1892061/`, `data/depth_metrics_recomputed.csv` |
| Old relative→0–80 m conversion (removed) | Same | LiDAR | AbsRel 2.50; δ<1.25 0.022 | same |
| YOLO11x | **None** | — | — | — |
| Tracker | **None** (no MOTA/IDF1) | — | — | — |

The depth result applies to KITTI's camera only. KITTI's focal length is close to the training camera's, and KITTI scenes are clear daytime. These are pixel-wise depth metrics, not object-distance accuracy.

---

## 5. Verified research metrics (Phase 3)

### A. Weather classifier: **VERIFIED**

| Claim | Raw data | Recomputed | Status |
|---|---|---|---|
| Accuracy 82.3 % | Confusion matrix in `final_test_eval.json` (rows = true, cols = predicted, `models._confusion_from`) | 657 / 798 = **0.823308** | ✅ |
| Macro-F1 83.3 % | same | **0.832990** (unweighted mean of 6 per-class F1) | ✅ |
| Six classes | same | clear, rain, fog, snow, night, overcast | ✅ |

- **Units:** fractions.
- **Split:** held-out test, 798 images.
- **Scripts:**
  - evaluation: `outputs/weather_runs/final_eval.py`;
  - recomputation: `scripts/recompute_weather_metrics.py` (max difference over all 21 stored metrics = 0.0).
- **Model match:** the stored checkpoint hash equals the deployed weights.

**Caveats**
- Per-image predictions were not saved, so recomputation starts from the stored confusion matrix, not from raw predictions.
- Fog has only 48 test images: 44 DAWN and 4 BDD.
- Per source, BDD macro-F1 is 0.737 (BDD fog: 1 of 4 correct).

### B. Experimental Kalman + KLT velocity estimator (KS): **VERIFIED, with two wording corrections**

| Claim | Recomputed from `stage2/04_test_once/per_object.csv.gz` | Status |
|---|---|---|
| Baseline error 3.87 m/s | B0 closing-speed MAE **3.8722** m/s (n = 8,513 valid of 9,387 eligible) | ✅ |
| Improved error 1.14 m/s | KS (`KS_kalman_scale`) MAE **1.1431** m/s (n = 8,506) | ✅ |
| Reduction 69.7 % | **−69.70 %** *paired* change on the 8,467 samples where both are valid (3.7527 → 1.1373 m/s). The unpaired ratio (3.872 → 1.143) is −70.5 %. | ✅ The 69.7 % is the paired figure; say "paired". |
| Sign accuracy 91.1 % → 99.3 % | B0 **0.9109** (n = 7,755) → KS **0.9932** (n = 7,744), for \|GT closing\| ≥ 1 m/s | ✅ |
| "KITTI: 8,008 frames" | 8,008 = all 21 KITTI tracking training sequences, processed into the cache. **The 3.87 / 1.14 figures come from the test half only: 10 sequences, 4,784 frames.** Primary-subset rows fall in 3,597 frames of 9 sequences. | ⚠️ **Partially:** describe as "8,008 frames processed; metrics on the 4,784-frame held-out test split" |

**Definitions**
- **Units:** m/s.
- **Closing speed:** −dZ/dt in the camera frame; positive means approaching.
- **Reference:** least-squares slope of the KITTI 3D-label position over frames t−2…t+2.

**Primary subset**
- Car, Van and Truck;
- not truncated, occlusion ≤ 1;
- matched to a pipeline track at IoU ≥ 0.5.

**The two estimators**
- **B0** is the production finite-difference estimator, replayed offline. The replay is identical to the online estimates (29,860 compared, 0 mismatched).
- **KS** is a Kalman filter on BEV position plus KLT scale-change of the box.

**Leakage, scripts and supporting results**
- **Leakage:** parameters were tuned on dev only, and the frozen configuration was recorded before the single test run. None was found.
- **Scripts:**
  - evaluation: `evaluate_kitti_velocity.py` (protocol `task5_evaluation_protocol.md`, SHA-256 `7a3d47e6…`);
  - recomputation: `scripts/recompute_velocity_metrics.py`.
- **Confidence intervals** (95 %, sequence bootstrap): B0 [3.19, 4.48], KS [0.93, 1.39]; paired change [−72.5 %, −65.2 %].
- **VRU subset** (pedestrians and cyclists): B0 2.01 → KS 1.07 m/s (paired −46.6 %).

**Important:** KS is **experimental and offline. It is not integrated**; the production pipeline still uses B0. Production and experimental velocity results must not be combined. Any production velocity, TTC or risk figure reflects B0.

### C. Production pipeline speed

| Claim | Raw data | Recomputed | Status |
|---|---|---|---|
| Original 14.4 FPS | `logs/pipeline.log`, 2026-10-29 full road.mp4 CLI run (relative depth, earlier code; output `outputs/videos/road_full.mp4`) | 7,249 frames / 501 s (build → end) = **14.47 FPS**; 14.56 FPS excluding the 3 s model build. Timestamps have 1-s resolution. | ⚠️ **PARTIALLY VERIFIED:** consistent with 14.4. Coarse wall-clock, different code and method; not comparable with T6.5. |
| T6.5 ≈ 17–18 FPS | 12 per-frame CSVs, `20261102-111418_T6.5_review/benchmark/` | Full loop after T6.5: **17.24 FPS relative** (runs 17.14–17.31), **17.32 FPS metric** (16.48–17.93); before: 14.10 / 13.36. `process()` only after T6.5: 19.95 / 20.10 FPS. | ✅ **VERIFIED** (same harness, GPU, video and FP32; interleaved before/after runs) |

- **"Full loop"** covers decode, `process()`, HUD drawing and video write.
- **Throughput gain:** about +22 % (relative depth) and +30 % (metric depth).
- **Latency, full-loop mean (metric depth):** 74.9 → 57.8 ms.
- **Hardware:** an H100 MIG slice. Results on other GPUs will differ.

### D. Distance, TTC and risk accuracy: **NOT VERIFIED (no suitable ground truth)**

- **Distance:** the only distance check is on KITTI vehicles: mean abs-rel 0.090 against label `z_near` (primary, test split).
- **No ground truth for TTC or risk:** neither has real-world ground truth.
- **What KITTI does show:** KITTI-derived TTC bands give a missed-urgent rate of 15.5 % for B0 and 6.9 % for KS. This is the TTC-band agreement of the experimental harness, not a crash-risk validation.
- **No accuracy claim:** accurate metric distance, TTC or risk on the demo videos must not be claimed.

---

## 6. Unverified or unsupported claims

| Claim / wording | Problem | Recommended wording |
|---|---|---|
| "KITTI: 8,008 frames" next to 3.87 / 1.14 m/s | The metrics use the 4,784-frame test split only | "8,008 frames cached; metrics on the held-out 4,784-frame test split (10 sequences)" |
| "69.7 % reduction" | It is the paired change | "69.7 % lower MAE on paired samples (95 % CI 65.2–72.5 %)" |
| Any object-detection precision, recall or mAP | No ground-truth evaluation exists | Do not report |
| Accuracy of distance, TTC or risk on road.mp4 / night clips | No labels | Report only as illustrative model outputs |
| "Velocity error 1.14 m/s" attached to the running system | KS is not integrated | Production B0: 3.87 m/s |
| "Fusion" in a multi-sensor sense | Camera-only system | "Camera-only, weather-weighted cue fusion" |
| Original 14.4 FPS as the baseline for T6.5 | Different method and code | Use 14.1 / 13.4 → 17.2 / 17.3 FPS from the paired harness |
| Weather robustness of depth, detection or velocity | All quantitative evaluations are clear daytime (KITTI) or classification only | State explicitly as untested |

---

## 7. Graph status (Phases 4–6)

| # | File | Status | Notes |
|---|---|---|---|
| 1 | `graphs/01_weather_confusion_matrix.png` | ✅ Generated, VERIFIED data | Counts and row-normalised recall |
| 2 | `graphs/02_weather_classification_performance.png` | ✅ Generated, VERIFIED data | P/R/F1 per class, macro averages, accuracy line |
| 3 | `graphs/03_object_detection_performance.png` | ❌ **NOT READY, not generated** | See below |
| 4 | `graphs/04_fps_latency_comparison.png` | ✅ Generated, VERIFIED data | FPS and latency in separate panels (no dual axis); every run shown as a dot |
| 5 | `graphs/05_velocity_estimation_comparison.png` | ✅ Generated, VERIFIED data | 8,467 aligned GT vs B0 vs KS samples. Points beyond ±30 m/s are drawn at the edge, with counts stated (192 B0, 10 KS). |
| 6 | `graphs/06_velocity_error_comparison.png` | ✅ Generated, VERIFIED data | MAE with 95 % CI, RMSE, median AE; labelled experimental/offline |
| 7 | `graphs/07_weather_risk_distribution.png` | ✅ Generated: **predicted outputs only** | 9,414 frames from today's production recordings |
| 8 | `graphs/08_ttc_distribution.png` | ✅ Generated: **estimated outputs only** | 20,080 object-frames after the documented exclusions |

All PNGs are 300 DPI. Every number in them comes from the CSVs in `data/`, which the scripts in `scripts/` produce.

**Graph 3: why it is NOT READY**
- **Not measured:** YOLO11x was never evaluated against ground-truth boxes. Precision, recall, mAP@50 and mAP@50:95 do not exist.
- **What exists instead:** the only related number is the KITTI harness's *GT-to-track match rate* at IoU ≥ 0.5 (0.921 for primary vehicles, 0.725 for VRUs, test split). It is a tracked-detection recall proxy under a velocity protocol, with no precision and no confidence sweep, so it is not a detection metric.
- **Required:**
  - a labelled detection set (e.g. the KITTI 2D object or tracking labels already on disk, BDD100K detection val, or a weather-specific set such as DAWN boxes);
  - a frozen class mapping (COCO → KITTI/BDD) and a protocol for DontCare, truncation and minimum height;
  - a standard evaluator (COCO or KITTI devkit).

**Graphs 7 and 8: how the data was produced**
- **Recording:** `scripts/record_production_outputs.py` runs `run.py`'s own `main()` with `--depth-mode metric --record-run` and the built-in `RunRecorder`. Only the recording directory changes, through an in-memory default argument, to `research_results/final_validation/runs/`.
- **Pipeline:** no code changed, no video was written, and all three models ran on CUDA with no fallback.
- **Weather groups are predictions:** the "fog" and "rain" groups come from the night clips and road.mp4 and are probably misclassifications. The groups describe the system's behaviour, not true weather.

**Graph 8 exclusion rule.** A sample is kept if all of these hold:
- `velocity_valid`;
- a TTC exists (closing ≥ 0.15 m/s);
- the TTC is finite, with 0 < TTC < 60 s (60 s is the calculator's cap, so capped values are excluded);
- distance is finite and > 0.

Of 40,409 recorded object-frames, the exclusion counts are:

| Reason | Object-frames |
|---|---|
| Kept | 20,080 |
| No valid velocity | 7,287 |
| Not closing | 12,721 |
| At the 60 s cap | 321 |

---

## 8. Known limitations

1. **Camera-only.** There is no LiDAR, radar or IMU in the running system.
2. **Uncalibrated camera on the demo videos.** Placeholder intrinsics, so BEV positions, distances, speeds and TTC are approximate.
3. **Production velocity is B0** (KITTI MAE 3.87 m/s, RMSE 9.54 m/s). This produces many false CRITICAL frames (graph 7). The better KS estimator is offline only.
4. **Weather adaptation is a fixed rule.** Threshold multipliers were chosen by design and never fitted or validated against outcomes.
5. **No weather-specific evaluation of detection, depth or velocity.** KITTI is clear daytime.
6. **Weather classifier:**
   - fog has low test support (48 images);
   - BDD weather labels are noisy;
   - implausible labels appear on the demo videos;
   - per-image test predictions were not stored.
7. **No detection or tracking metrics** (mAP, MOTA or IDF1).
8. **Speed numbers are for one H100 MIG slice in FP32.**
9. **Exports are `mp4v`** (no H.264 in this environment). The CLI video lacks BEV, weather probabilities and adaptive weights.
10. **Intermittent pytest segfault** (2026-11-01). Unresolved.

---

## 9. Experiments still required

1. **Detection evaluation:** YOLO11x on a labelled set, with a frozen class mapping and the COCO/KITTI evaluator. Ideally split by weather (e.g. BDD100K val detection labels grouped by its weather tag). This unblocks graph 3.
2. **Weather-stratified evaluation:** depth (where LiDAR exists, e.g. a dataset with adverse-weather depth ground truth) and velocity/TTC, to support any claim of weather *robustness*.
3. **Integration of KS:** if it is to be claimed for the live system, integrate it as a separate, approved task, then rerun the KITTI test protocol through the live pipeline.
4. **Calibrated-camera recording:** distance and TTC validation against a reference (e.g. a calibrated rig or KITTI-style labels).
5. **Weather multipliers:** a sensitivity or ablation study on labelled near-miss data, or at least a documented rationale.
6. **Tracking metrics:** MOTA/HOTA/IDF1 on KITTI tracking.
7. **Optional, for reproducibility:** store per-image weather test predictions in any future evaluation. The existing test pass must not be repeated for tuning.

---

## 10. Recommendations before mentor review

1. Present the **verified** headline results only:
   - weather 82.3 % accuracy, 83.3 % macro-F1 (n = 798);
   - metric depth AbsRel 0.132 on KITTI;
   - B0 3.87 → KS 1.14 m/s (experimental, offline; −69.7 % paired; KITTI test split, 4,784 frames);
   - 13.4–14.1 → 17.2–17.3 FPS full loop (same harness).
2. Fix the two wording issues: **8,008 vs 4,784 frames**, and the **paired** 69.7 %.
3. Label graphs 7 and 8 as *predicted / estimated outputs*. Point out the misclassified weather groups instead of hiding them.
4. State plainly that detection metrics are **not available** (graph 3 NOT READY), rather than substituting the match rate.
5. Keep the camera-only statement and the "KS not integrated" statement in every place where velocity or TTC appears.
6. Review and decide on the pending Task 7 changes (`app.py`, `tests_dashboard.py`, `README.md`, `CAPSTONE_SUMMARY.md`); they are still uncommitted.
7. `research_results/` is **not git-ignored**. Decide whether to commit it:
   - the `runs/` CSVs are about 6 MB and come from the demo videos, which are not KITTI data;
   - the scripts reference absolute `/workspace/experiments` paths.

---

## Integrity statement

Checked at the end of the task against hashes recorded at the start:
- **Unchanged:**
  - all 203 tracked files;
  - the four uncommitted Task 7 files;
  - `config.yaml`;
  - all 5 weight files;
  - all 459 files under `/workspace/experiments`, with none added;
  - Task 5 checksums (29/29);
  - the protected-file set;
  - the goldens (`outputs/regression/`, 0 differences);
  - `outputs/weather_runs/`;
  - both demo export folders (sizes and timestamps unchanged).
- **Not changed:** no dependency was installed or removed, and nothing was committed or pushed.
- **The only side effect outside `research_results/`:** the three recordings appended their console log lines to `logs/pipeline.log`. This is the git-ignored runtime log that `run.py` always appends to.
