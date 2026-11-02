# Capstone summary: Weather Adaptive Road Mapping for Autonomous Vehicles

This is the delivery summary: what the system does, how to run the demonstration, what has been measured, and what has not. [README.md](README.md) has the full details; each section below says where to look.

## 1. Overview and objectives

The system is a camera-only perception pipeline for a forward-facing vehicle camera. For each frame it:
- detects and tracks road users;
- estimates depth;
- classifies the weather and lighting;
- adapts its confidence weighting and risk thresholds to that weather;
- in metric-depth mode, maps objects onto a bird's-eye view (BEV) with distance, relative speed, time-to-collision (TTC) and a risk level.

A Streamlit dashboard shows every stage live.

**Objectives:**
1. One working end-to-end pipeline built from genuine pretrained or fine-tuned models.
2. Weather awareness that visibly changes the system's outputs, not just a label.
3. Honest reporting: every value on screen comes from the pipeline, and estimates are labelled as estimates.

**Scope:**
- **Sensors:** the only sensor is one monocular camera. No LiDAR, radar or IMU is used, so there is no physical sensor fusion.
- **"Adaptive fusion":** this means the weather-dependent weighting of two camera-derived modules (detection and depth).
- **Metric outputs:** distances, speeds and TTC are uncalibrated monocular estimates.

## 2. Architecture and workflow

```
video / camera / synthetic frame
  → WeatherClassifier (EfficientNet-B0) → temporal smoothing (1 s)
  → YOLODetector (YOLO11x, COCO) → ByteTracker (track IDs)
  → DepthEstimator (Depth Anything V2 ViT-S: relative by default, or metric)
  → ConfidenceEstimator + AdaptiveFusionEngine (weather-dependent module weights)
  → [metric depth only] BEVMapper → RelativeVelocityEstimator → TTC → RiskEngine (weather-adjusted thresholds)
  → PipelineOutput → app.py (Streamlit dashboard) or run.py (OpenCV window / annotated video)
```

`PerceptionPipeline.process()` in `pipeline.py` runs these stages in order. The two user interfaces only display what it returns. README "Architecture / Pipeline" has the full diagram.

**How weather changes the outputs.** The smoothed weather label (`clear`, `overcast`, `rain`, `fog`, `snow` or `night`) drives two adjustments, both set in `config.yaml`:
1. **Module weights and confidences.** `fusion.weather_modifiers` lowers the detection and depth confidences in bad conditions. For example, at night: detection ×0.70, depth ×0.65. The dashboard shows the result as ADAPTIVE MODULE WEIGHTS and an overall scene confidence. The weights are displayed; they do not change detections or depth values.
2. **Risk thresholds.** `risk.weather_multipliers` widens the TTC thresholds, so the same TTC becomes a more severe risk level sooner. Examples: night ×1.25, rain ×1.30, fog ×1.60. The RISK LEVEL GUIDE shows the active thresholds and the weather they were adjusted for. Risk levels themselves are computed only in metric-depth mode.

## 3. Technologies and models

| Purpose | Technology | Status |
|---|---|---|
| Object detection | Ultralytics YOLO11x, COCO-pretrained (`weights/yolo11x.pt`) | Official weights, used unchanged |
| Tracking | ByteTrack (Kalman filter + Hungarian matching), `tracking.py` | An algorithm; has no weights |
| Depth | Depth Anything V2 ViT-S: relative (`depth_anything_v2_vits.pth`) or metric outdoor, Virtual KITTI 2 (`depth_anything_v2_metric_vkitti_vits.pth`) | Official weights |
| Weather | EfficientNet-B0 (torchvision), fine-tuned in this project on BDD100K + DAWN (`weights/efficientnet_weather.pth`) | Trained here |
| Mapping | Pinhole back-projection to BEV (NumPy); Open3D point cloud only for `.ply` export | Geometry |
| UI | Streamlit 1.60 dashboard (`app.py`); OpenCV HUD (`run.py`) | — |
| Stack | Python 3.10, PyTorch 2.4 (NGC container), OpenCV 4.10, NumPy 1.26 | See ENVIRONMENT.md and requirements-lock.txt |

## 4. Modules

README "Repository Structure" lists every file. The runtime core is:
- `pipeline.py`: orchestration and dashboard render helpers;
- `models.py`: the model wrappers, with heuristic fallbacks unless in strict mode;
- `tracking.py`, `fusion.py`, `mapping.py`, `camera.py`;
- `relative_velocity.py`, `ttc_calculator.py`, `risk_engine.py`;
- `utils.py`, `config_schema.py`, `run_logging.py`;
- `app.py` and `run.py`.

The offline evaluation code (KITTI depth, Task 5 velocity and ego-motion) is separate and is not part of the live pipeline.

## 5. Datasets and how they are actually used

| Data | Used for | Notes |
|---|---|---|
| BDD100K (Kaggle mirror) + DAWN | Training and testing the weather classifier | README "Training the Weather Classifier" |
| KITTI depth selection set (1,000 frames) | Evaluating depth | Stored outside the repository (CC BY-NC-SA 3.0) |
| KITTI tracking benchmark | Task 5 velocity evaluation (offline) | Stored outside the repository |
| `road.mp4` (daytime, 7,249 frames, 24 fps) | Demonstration, golden regression outputs, runtime benchmarks | No ground truth |
| `data/sample_videos/night_1.mp4` (837 frames), `night_2.mp4` (1,328 frames), both 25 fps | Night-time demonstration only | Never trained or benchmarked on; no ground truth |
| nuScenes, Seeing Through Fog, CARLA | Set-up helpers only | No pipeline code reads them |

## 6. Performance and evaluation results

These are formal evaluations on datasets with ground truth. The demo videos have none (see section 7).

| What | Result | Source |
|---|---|---|
| Weather classifier, held-out test set (n = 798) | Accuracy 0.823, macro-F1 0.833. Dashcam fog is not established (only 4 BDD fog images). | README "Selected weather checkpoint and results" |
| Metric depth on KITTI (1,000 frames) | AbsRel 0.132, RMSE 5.26 m, δ<1.25 0.806. This is pixel-wise depth, not object distance, on a favourable camera domain. | README "KITTI depth evaluation" |
| Production velocity estimator (B0), KITTI tracking test split | Closing-speed MAE 3.87 m/s | `task5_stage2_results.md` |
| Experimental KS velocity estimator, same split | MAE 1.14 m/s. **Offline only, not integrated.** | `task5_stage2_results.md` |
| End-to-end runtime, full `road.mp4`, H100, `run.py`-equivalent loop (read → process → HUD → write) | Relative mode: 57.8–58.4 ms per frame (17.1–17.3 FPS). Metric mode: 55.8–60.7 ms per frame (16.5–17.9 FPS). Three runs per mode; ranges, not pooled. | `/workspace/experiments/performance_profiling/20261102-111418_T6.5_review/` |

The dashboard deliberately caps playback at `pipeline.target_fps` (15) and adds browser rendering on top, so its frame rate is lower than the numbers above. The dashboard frame rate itself has not been benchmarked.

## 7. Demonstration findings: day and night videos (observations, not accuracy)

Each video was processed completely, in both depth modes, through the real pipeline. Dashboard panels were rendered with the dashboard's own helpers. The evidence is in `/workspace/experiments/task7_capstone_readiness/`.

These are counts and visual inspections. **There is no ground truth for these videos, so no accuracy, distance or speed can be claimed from them.**

| | road.mp4 (day) | night_1.mp4 | night_2.mp4 |
|---|---|---|---|
| Frames processed | 7,249 / 7,249 | 837 / 837 | 1,328 / 1,328 |
| Crashes; heuristic fallbacks used | none; none | none; none | none; none |
| Frames with ≥1 detection | 88.6% | 100% | 95.9% |
| Detections per frame (mean) | 3.8 | 8.7 | 4.2 |
| Unique track IDs; median track length | 645; 13 frames | 152; 15.5 frames | 232; 8 frames |
| Tracks whose class label changed | 17% | 20% | 22% |
| Weather (smoothed) | clear 84.5%, rain 7.9%, overcast 7.6% | night 94.5%, fog 5.5% | night 87.7%, fog 10.8%, rain 1.5% |
| CRITICAL TTC threshold applied | 1.5–1.95 s | 1.875–2.4 s | 1.875–2.4 s |
| Metric-mode risk: share of frames CRITICAL | 36% | 55% | 52% |

**Observations:**
- **Darkness is classified as `night`, not as fog or rain.** The `fog` frames at night coincide with visible haze or smog and with strong headlight or streetlight glare. In `night_2` those causes are hard to separate by eye.
- **Daytime false "rain".** On `road.mp4`, about 8% of frames are labelled `rain` on clearly dry, sunny road (e.g. frames 1368–1577 and 2319–2454). This is a classifier error: domain shift from the dashcam training data, plus the windscreen and dashboard in view. It is not a software fault.
- **Night detection.** It works well under street lighting: cars, motorcycles, people and auto-rickshaws are found. Auto-rickshaws are labelled `truck`, because COCO has no such class. Detections are missed in heavy glare and haze; for example, `night_2` frame 1000 has none although vehicles are visible. No obvious glare-induced false positives were seen in the sampled frames.
- **Tracking.** It holds IDs for nearby vehicles. It fragments more at night, especially in `night_2` (shorter tracks), and labels flip between similar classes such as car/truck and motorcycle/bicycle.
- **Depth.** It gives a usable near/far structure in both modes. At night the dark foreground road gets implausible metric values (for example, about 13 m for the road just ahead in `night_1`).
- **Speed, TTC and risk.** Depth noise turns into implausible relative speeds (one frame shows −149 km/h) and many CRITICAL frames. This is the known limitation of the production velocity estimator; see section 8.

## 8. Limitations

- **Sensors:** camera only; no LiDAR, radar or IMU, and no physical multi-sensor fusion. No ego-motion estimate (the HUD shows ego speed N/A), so velocities are relative to the moving camera.
- **Metric outputs:** distance, speed, TTC and risk are monocular estimates from an uncalibrated camera, not validated on these videos. On the demo videos the production velocity estimator gives noisy speeds and frequent false CRITICAL levels.
  - The better experimental estimator (KS) is evaluated offline only and is not integrated.
- **Weather classifier:** false `rain` and `overcast` on some sunny daytime frames, and `fog` at night under glare or haze. Dashcam fog performance has not been established.
- **Adaptive weights:** the module weights are hand-tuned rules and are displayed only; they do not alter detections or depth. Weather changes risk only through the TTC thresholds, in metric mode.
- **Night perception:** detections are missed under strong glare and haze, and tracks are shorter at night.
- **Default mode:** relative depth, which has no unit. BEV, distance, TTC and risk need metric depth (`--depth-mode metric`).
- **Testing:** an intermittent pytest segmentation fault was seen once (2026-11-01) and remains **unresolved**. It has not recurred in later full-suite runs.

## 9. Launch and demonstration instructions

The commands are run from the repository root, in the reference environment (see ENVIRONMENT.md):

```bash
# Dashboard, default relative depth: detection, tracking, depth map, weather, adaptive weights
streamlit run app.py

# Dashboard with metric depth: also the bird's-eye view, distance, relative speed, TTC and risk
streamlit run app.py -- --depth-mode metric
```

Open the URL Streamlit prints (by default http://localhost:8501), then:
1. **Choose the video.** In the sidebar, set **Input source → Video file**. The path is pre-filled with `road.mp4`; for the night clips enter `data/sample_videos/night_1.mp4` or `data/sample_videos/night_2.mp4`. Playback starts; use **Pause**, **Resume**, **Stop** and **Restart**. Changing the path starts the new video from its first frame.
2. **Detection and tracking:** the camera feed shows boxes with track IDs. Expand **DETECTED OBJECTS** for the table.
3. **Depth:** the DEPTH MAP panel (near = dark, far = bright), with the DEPTH & CALIBRATION card explaining the mode.
4. **Weather:** the WEATHER card shows the label, confidence and all six class probabilities.
5. **Weather adaptation:**
   - ADAPTIVE MODULE WEIGHTS changes with the weather, for example between `road.mp4` and a night clip.
   - The RISK LEVEL GUIDE shows TTC thresholds "weather-adjusted for" the current label: CRITICAL < 1.5 s when clear, < 1.9 s at night.
6. **Bird's-eye view:** only with `-- --depth-mode metric`; otherwise the panel explains why it is unavailable.
7. **Status:** MODULE STATUS shows which models are really running (`MODEL INFERENCE` vs `HEURISTIC FALLBACK`).

**Without a browser**, a video can be written as an annotated MP4:

```bash
python run.py --source data/sample_videos/night_1.mp4 --depth-mode metric --output outputs/videos/night_1_annotated.mp4
```

**Before a demo**, run `pytest tests_dashboard.py -q` (under a minute on the reference GPU machine).
