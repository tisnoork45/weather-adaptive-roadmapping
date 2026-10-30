# Research-Readiness Audit — Weather Adaptive Road Mapping

*Date: 2026-10-30. Inspection only: no project source, config, weight, dataset or output file was modified.
This file is the only addition. Tests and profiling ran in a scratchpad copy of the code (weights, `data/` and
`road.mp4` were symlinked, read-only use), so `logs/pipeline.log` and `__pycache__/` were not touched.*

---

## 0. Scope corrections to the task brief

- **Lane detection is not part of the system.** It was removed in commit `15a7dd1` ("Remove lane detection subsystem").
  Leftovers remain: `fusion.py` `estimate_lane_confidence` / `lane_signal_weights` (dead code), the `download.py --only lane`
  option, and many `*.before_lane_removal.bak` files. Any paper must not claim lane detection.
- **The "weather-adaptive sensor weighting" changes no output.** It is computed but only displayed (see B1).
- **No LiDAR, radar, IMU or odometry input exists.** Ego speed is always `N/A`.

---

## 1. Project structure, Git status and backup integrity

| Item | Finding |
|---|---|
| Branch / HEAD | `main` @ `4f0dc0f` (3 commits: `b52e30e`, `15a7dd1`, `4f0dc0f`) |
| Uncommitted | 15 modified tracked files (`.gitignore README.md app.py config.yaml depth_anything_v2/dpt.py mapping.py merge_dawn_weather.py models.py pipeline.py prepare_weather_dataset.py requirements.txt run.py tests.py tracking.py utils.py`) + untracked `outputs/weather_runs/` and `research_audit.md`. **All work since 29 Oct (dataset leakage fixes, retrained classifier, smoothing, intrinsics, tracking) exists only in the working tree and the tarball.** |
| Clutter in repo root | 27 `*.bak` / `*.before_*` / `*_old.py` / `*broken_backup` files, `fix_run_py.py`, `depth_anything_v2_repo.zip` (94 MB), `logs.zip`, `bdd_files.csv`, `Depth-Anything-V2/` *and* `external/depth_anything_v2/` (two upstream copies; only the root `depth_anything_v2/` package is imported), `risk_delivery/` (an older copy of the risk modules). |
| Backup `/workspace/roadmapping_backup/project_code_latest.tar.gz` | 402.6 MB, 29 Oct 09:45. `gzip -t` OK, 293 entries. **All 21 `*.py`, `config.yaml`, `README.md`, `.gitignore` are byte-identical to the current working tree.** Contains the 4 weight files. Does **not** contain `.git/`, `road.mp4`, `outputs/` or datasets. So it restores code + weights, but not commit history or the weather-run reports. `git_commit.txt` records `4f0dc0f`; `git_status.txt` is empty (0 bytes) because it was written for the older `project_code.tar.gz` on 28 Oct. |
| Disk | `/workspace` (NFS): 101 GB, 11 GB used, **90 GB free**. |

---

## 2. Source files by responsibility

| Responsibility | File · symbol |
|---|---|
| Detection | `models.py` `YOLODetector` (YOLO11x, ultralytics 8.4.192, conf 0.35, IoU 0.45, 640 px, classes 0,1,2,3,5,7,9,11) |
| Depth | `models.py` `DepthEstimator._real_infer` (Depth Anything V2 ViT-S, relative); `depth_anything_v2/dpt.py` |
| Weather | `models.py` `WeatherClassifier`, `WeatherTemporalSmoother`, `load_weather_rgb`/`preprocess_weather_rgb`, `train_weather_model`; data: `prepare_weather_dataset.py`, `merge_dawn_weather.py`; eval: `outputs/weather_runs/final_eval.py`, `summarize.py`, `run_round.sh` |
| Tracking | `tracking.py` `ByteTracker`, `KalmanBoxTracker` (2-D image-plane Kalman) |
| Lane detection | **none** (removed) |
| Sensor weighting | `fusion.py` `ConfidenceEstimator`, `AdaptiveFusionEngine`; config `fusion:`, `confidence_estimation:` |
| Distance | `mapping.py` `BEVMapper.project_objects_to_bev` / `_sample_object_depth` / `pixel_to_ground_point` |
| Ego speed | none. `pipeline.py` `draw_ego_speed_panel` always prints `N/A` |
| Relative velocity | `relative_velocity.py` `RelativeVelocityEstimator` |
| TTC | `ttc_calculator.py` `compute_ttc`, `compute_ttc_for_objects` |
| Risk | `risk_engine.py` `RiskEngine`; config `risk:` |
| Orchestration / UI | `pipeline.py` `PerceptionPipeline.process` + render helpers; `run.py` (OpenCV/video writer); `app.py` (Streamlit); `utils.py` (`FPSTracker`, `MetricsTracker`, `VideoStream`, drawing) |
| Unused / legacy | `carla_sim.py`, `test_risk_runtime.py` (points at a non-existent `data/sample_videos/road.mp4`), `models_old.py`, `models_backup_before_weather.py`, `fix_run_py.py`, `setup.py` |

---

## 3. Data flow (input video → dashboard)

```
VideoStream (cv2) ──frame BGR, timestamp = (i-1)/video_fps [video] or perf_counter [live]
 └─ PerceptionPipeline.process(frame, ts)
     1 WeatherClassifier.infer  → raw probs (EfficientNet-B0, 224 squash)
       WeatherTemporalSmoother  → EMA over time, τ = 1.0 s  → weather.label
     2 YOLODetector.infer       → boxes, classes, scores
     3 ByteTracker.update       → track_id per detection (2-D IoU, Kalman in cx,cy,a,h)
     4 DepthEstimator.infer     → raw relative disparity → per-frame EMA min/max → (1-norm)·80 m "depth"
     5 ConfidenceEstimator + AdaptiveFusionEngine(weather.label) → sensor_weights, scene confidence   [display only]
     6 BEVMapper.build_bev      → per object: median depth in 5×5 window at box bottom-centre → (x_m, z_m)
     7 RelativeVelocityEstimator(ts) → v = (p_newest − p_oldest)/Δt over ≤5 samples, EMA α=0.5
     8 compute_ttc_for_objects  → TTC = z_m / (−v_z) if closing > 0.15 m/s, cap 60 s
     9 RiskEngine.assess(weather.label) → per-object level; frame level = worst object
 └─ run.py: draw boxes/HUD/panels → cv2.VideoWriter (mp4v)       app.py: st.image + HTML panels, sleep to target_fps
```

---

## 4. Tests (run in a scratchpad copy)

| Run | Result |
|---|---|
| `pytest tests.py` (CUDA) | **54 passed** in 20.6 s |
| `TEST_DEVICE=cpu pytest tests.py` | **54 passed** in 19.8 s |

The tests cover plumbing and math contracts: TTC arithmetic, threshold boundaries, tracker ID logic, preprocessing
parity, smoother behaviour and intrinsics scaling. **No test compares any distance, speed or TTC with ground truth.**
They show the code runs as written, not that its outputs are correct.

---

## 5. Models, configuration and GPU use

| Model | Path | SHA-256 prefix | Notes |
|---|---|---|---|
| YOLO11x | `weights/yolo11x.pt` | `7bc158aa95c0ebfd` | COCO-pretrained, not fine-tuned |
| Depth Anything V2 ViT-S | `weights/depth_anything_v2_vits.pth` | `715fade13be8f229` | **relative** model (not the metric checkpoints) |
| EfficientNet-B0 weather | `weights/efficientnet_weather.pth` | `0bda562ef4508667` | run B_c3000-s42, 6 classes |
| (backup, read-only) | `weights/efficientnet_weather.backup-20261029-df1bd218.pth` | `df1bd21886d02b41` | previous checkpoint |

GPU: H100 80GB, MIG enabled, one 3g.40gb instance (60 SMs, 40 GB). Idle at audit time (37 MiB, no processes).
`nvidia-smi` utilisation counters are not readable inside MIG ("Insufficient Permissions"), so utilisation was measured from PyTorch.

**Profiling, road.mp4 frames 30–600, CUDA, fp32:**

| Stage | Median ms |
|---|---|
| Weather | 6.0 |
| Detection (YOLO11x) | 8.8 |
| Tracking | 0.4 |
| **Depth (DA-V2 ViT-S @518)** | **43.8** |
| Fusion | 0.8 |
| BEV | 1.9 |
| Velocity/TTC/risk | 0.1 |
| **`process()` total** | **61.6 median / 66.3 p95 → 16.2 FPS** |

Peak GPU memory: 917 MiB allocated, 1130 MiB reserved (about 3 % of the MIG slice). Depth is about 71 % of latency.
fp16/bf16 autocast and a smaller depth input size are untested speed-ups.

Config notes: the intrinsics (`fx=fy=1000, cx=640, cy=360` @1280×720) are **assumed, not calibrated** for the
road.mp4 camera. `tracking.frame_rate: 30`, but the video is 23.997 fps. `pipeline.target_fps: 15` throttles only the Streamlit loop.

---

## 6. Weather classifier: training, evaluation, leakage

**Labels.** BDD100K `attributes.weather` + `timeofday`, mapped by the priority fog > snow > rain > night > overcast > clear.
"partly cloudy" is mapped to clear. DAWN contributes fog/rain/snow (`Sand` is unused).
`night` is a time-of-day label, not weather, so a clear night and a rainy night land in different classes by rule.

**Splits (`data/weather_dataset_c3000`):**

| | clear | rain | fog | snow | night | overcast |
|---|---|---|---|---|---|---|
| train | 3000 | 3199 | 328 (198 DAWN + 130 BDD) | 3203 | 3000 | 3000 |
| val | 150 | 150 | 50 | 150 | 150 | 150 |
| test | 150 | 150 | **48 (44 DAWN + 4 BDD)** | 150 | 150 | 150 |

DAWN rain and snow (199/203) appear in train only. DAWN fog is the only DAWN class in val/test.

**Leakage controls in place:** SHA-256 exact dedupe; 256-bit dHash near-duplicate grouping (threshold 24, one split per
group); BDD val/test split by session prefix, with sessions in train dropped from val/test; DAWN holdout 0.3 assigned by filename hash.
Selection used val macro-F1 only. The test set was evaluated once (`final_eval.py`, which refuses to re-run).

**Results (test, n=798):** accuracy 0.823, macro-F1 0.833. BDD-only: accuracy 0.818; F1 0.820 excluding fog. BDD fog is 1/4 correct.

**Remaining leakage and validity risks (not proven, need experiments; see C):**
1. **Source shortcut for fog.** 92 % of test fog is DAWN, which was collected differently (web images, framing, resolution).
   The 0.91 fog F1 may partly measure "looks like DAWN" rather than fog. Only 4 BDD dashcam fog images exist in test.
2. **BDD train ↔ test** is only session-separated for val/test; the original BDD train/val split is reused for the train pool.
   Geographic and time-of-day correlation remains (same cities).
3. **Label noise.** BDD weather tags are crowd-sourced; overcast ↔ clear is the dominant confusion (22/150 clear → overcast).
4. **Domain shift observed on road.mp4.** About 16 % of the full video was labelled rain or overcast while sunny and dry
   (tree shade, backlight, camera pointing at the dashboard).
5. The dHash threshold of 24 produced one inspected false positive. The threshold was not validated against a labelled near-duplicate set.

---

## 7. Distance, ego speed, relative speed, TTC and risk formulas, and why they are unreliable

**Formulas as implemented:**

- Depth: `d_raw` = DA-V2 relative output (affine-invariant **inverse depth / disparity**).
  Then `m_lo, m_hi` = EMA(α=0.3) of the per-frame min/max of `d_raw`, and
  `depth = (1 − clip((d_raw − m_lo)/(m_hi − m_lo))) · 80 m`.
- Distance: `z_m` = median depth in a 5×5 window at the box bottom-centre. `x_m = (u − cx)·z_m/fx`.
- Ego speed: not computed.
- Relative velocity: `v = (p_t − p_{t−k})/Δt`, k ≤ 4 samples (≈0.17 s at 24 fps), EMA α=0.5.
  History is deleted the first frame a track is missing.
- TTC: `z_m / (−v_z)` if `−v_z > 0.15 m/s`, else None. Capped at 60 s.
- Risk: TTC < {1.5, 3.0, 5.0} s × weather multiplier → CRITICAL/WARNING/CAUTION. Distance floor: < 2.5 m → WARNING, < 5 m → CAUTION.
  Frame level = worst object in view.

**Measured on road.mp4 (frames 30–600):**

| Quantity | Value |
|---|---|
| Per-frame disparity minimum | always 0.0 (sky) |
| Per-frame disparity maximum (EMA) | 8.4 – 15.8, so the whole depth scale moves by up to ~1.9× during the clip |
| \|Δz\| between consecutive frames, same track (tracks ≥ 20 frames) | median 0.35 m, p95 4.0 m |
| Speed implied by median 1-frame z jitter alone | **≈ 30 km/h** |
| Closing-speed distribution (valid estimates) | p5 −9.4, median 1.4, p95 **49.9 m/s (180 km/h)** |
| Valid estimates with \|closing speed\| > 36 km/h | 25.5 % |
| Frame-level risk | SAFE 453, CAUTION 16, WARNING 28, **CRITICAL 73 (13 %)** |
| Detections whose box bottom lies in the lowest 15 % of the image (dashboard zone) | 129 |

**Root causes, in order of impact:**

1. **Wrong functional form.** Disparity is ∝ 1/Z, but it is mapped *linearly* to depth.
   `Z = 80·(1 − d/d_max)` cannot match `Z = 1/(a·d + b)` for any range, so distances are biased, and by different amounts at different ranges.
2. **Per-frame scale normalisation.** `d_max` is the nearest pixel in the frame, usually the dashboard or bonnet, and changes when anything enters
   the bottom of the image. Every object's `z` changes with it, so static objects acquire velocity. This is the main source of the 30 km/h noise floor.
3. **Arbitrary metric anchor.** The 80 m maximum and the 0 m minimum are assumptions. Nothing ties them to the scene: no calibrated intrinsics,
   no ground plane, no LiDAR.
4. **Differentiation over a 0.17 s baseline** amplifies depth noise. A 0.35 m error becomes about 2 m/s. The EMA then smooths a biased signal.
5. **No ego motion.** "Relative velocity" includes ego speed for every static object. Parked cars at the roadside "close" at ego speed,
   and TTC uses only longitudinal `z`, with no lateral or path check. So any roadside object can become CRITICAL.
6. **Risk aggregates over all objects in view.** The worst object sets the frame level, whether or not it is in the ego path.
7. **Velocity history is dropped on a single missed frame**, while the tracker keeps the ID for up to 30 frames. Recovered tracks restart with
   two-sample, very noisy estimates.
8. **Intrinsics are assumed.** Lateral `x_m` scales with the unknown true fx. Longitudinal `z` does not depend on fx in the current method.
9. **Weather misclassification** (6.4) multiplies the thresholds by 1.3 in false "rain" segments.

Conclusion: the distance, speed, TTC and risk values shown are **uncalibrated, unvalidated estimates**. Their error is dominated by
depth scale instability, not by the TTC or risk logic.

---

## 8. FPS: three different quantities

| Quantity | Where | What it measures | Value |
|---|---|---|---|
| **Inference latency / inference FPS** | `PipelineOutput.total_latency_ms` (`process()` wall time) and per-stage `metrics[*_latency_ms]` | Model + post-processing time only. No explicit `cuda.synchronize`, but every stage returns CPU/NumPy results, so the timings are effectively synchronised. | 61.6 ms → **16.2 FPS** |
| **Processing (loop) FPS** | `MetricsTracker.fps_tracker` (ticks inside `process()`, window 60). This is the "FPS" on the HUD and in the logs. | Tick-to-tick interval, so it *includes the caller's* decode, drawing, video encoding (run.py) or Streamlit serialisation + `time.sleep` throttle (app.py) | run.py full video: **≈14.4 FPS** (7249 frames / 505 s, includes about 3 s model loading); app.py: **capped at `target_fps`=15** by `sleep` |
| **Browser rendering FPS** | **Not measured anywhere** | Frames actually painted by the browser. Streamlit sends each `st.image` as a re-encoded image over a websocket and may coalesce or drop updates. | unknown |

The HUD label "FPS" is ambiguous and should state which quantity it is.
A paper needs all three, reported separately, with hardware (MIG 3g.40gb) and precision (fp32) stated.

---

## 9. Experiment scripts, logs, metrics and outputs

| Artefact | Content |
|---|---|
| `outputs/weather_runs/{A_c800,B_c3000}-s{42,43}/` | `train.log`, `report.json` (val metrics per epoch), `model.pth` (gitignored) |
| `outputs/weather_runs/final_test_eval.json` | single test evaluation of the promoted checkpoint (overall, per-class, per-source, confusion, latency, checkpoint SHA) |
| `outputs/weather_runs/{run_round.sh, summarize.py, final_eval.py}` | reproducible training, selection and final evaluation |
| `outputs/videos/` | `road_full.mp4` (281.6 MB, 7249 frames), `road_test_10s{,_v2,_v3}.mp4` |
| `logs/pipeline.log` | 2251 lines; one line every 30 frames (FPS, weather, objects, latency); 14 historical ERROR / crash lines from earlier runs |
| `imported_logs/`, `streamlit_run.log`, `logs.zip` | historical |
| `bdd_files.csv` | file list only |
| `outputs/bev/`, `outputs/frames/` | empty |
| **Missing** | No per-frame or per-object CSV of distance, speed, TTC, risk or weather. No detection or tracking metrics. No depth metrics. No runtime benchmark file. |

---

## 10. Code changes needed for research readiness (summary; ordered list in G)

Strict no-fallback mode; dataset adapters with real intrinsics, timestamps and ground truth; per-frame structured logging; a corrected depth-to-metric
method; ego-motion input; a robust velocity estimator; in-path TTC/risk; FPS separation; evaluation scripts; repository clean-up and commits.

---

# Prioritised report

## A. Working components

1. YOLO11x detection on CUDA (8.8 ms/frame).
2. Depth Anything V2 ViT-S loads and runs on CUDA and CPU (device bug fixed in `dpt.py`). It produces *relative* depth.
3. EfficientNet-B0 weather classifier: test accuracy 0.823 / macro-F1 0.833. Training and inference preprocessing are identical. Reproducible training with seeds. Single held-out test evaluation.
4. Weather dataset pipeline with exact- and near-duplicate dedupe, session-grouped BDD val/test, and fog holdout.
5. Weather temporal smoothing (flicker on the 240-frame clip went from 21 frames to 0).
6. ByteTrack with lost-track recovery: 42 → 17 IDs, median track lifetime 3 → 22 frames on the 240-frame clip.
7. Intrinsics scaling to frame size.
8. Video-timeline timestamps for prerecorded input.
9. End-to-end run of all 7249 frames with no errors. The output decodes completely.
10. 54/54 tests pass on CUDA and on CPU.
11. Backup tarball is valid and matches the working-tree code.

## B. Confirmed software problems

| # | Problem | Evidence |
|---|---|---|
| B1 | **Adaptive sensor weights change no output.** `fusion.sensor_weights` is used only for display and box transparency (`pipeline.py:584`). Detection, depth, distance, TTC and risk never read it. The only weather effect on outputs is the risk-threshold multiplier. | grep of all uses |
| B2 | **Depth mapped linearly from disparity**, with per-frame min/max normalisation | `models.py` `_real_infer`; scale drift up to 1.9× |
| B3 | **Velocity noise floor ≈ 30 km/h**; p95 closing speed 180 km/h | profiling (§7) |
| B4 | No ego speed / ego motion. Static objects get ego-speed closing velocity | `draw_ego_speed_panel` |
| B5 | TTC uses longitudinal `z` only, with no path or lateral relevance. Frame risk = worst object anywhere in view | `ttc_calculator.py`, `risk_engine.py` |
| B6 | Velocity history deleted on one missed frame, which is inconsistent with tracker recovery | `RelativeVelocityEstimator._prune_stale` |
| B7 | Silent fallbacks remain: detector fallback, classical depth fallback, weather colour heuristic. A results run could silently use heuristics. | `models.py` lines 156, 281, 292, 477 |
| B8 | Intrinsics assumed, not calibrated. `tracking.frame_rate` 30 vs 24 fps video | `config.yaml` |
| B9 | The "FPS" label mixes inference and loop rate. Browser FPS not measured. app.py throttled to 15 | §8 |
| B10 | Confidence "signals" are not confidences: depth "model_score" = normalised depth-map std; detection stability = variation in detection count | `fusion.py` |
| B11 | No per-frame result logging; no ground-truth evaluation of any geometric quantity | §9 |
| B12 | Repository hygiene: all recent work uncommitted; 27 backup files; two duplicate upstream DA-V2 trees; dead lane code; stale `test_risk_runtime.py` and `risk_delivery/` | §1 |
| B13 | Weather classifier misreads sunny road.mp4 segments (≈16 % of frames) as rain/overcast | full-run analysis |

## C. Suspected problems requiring experiments

1. Fog performance partly reflects a DAWN-source shortcut. *Test:* evaluate on BDD/other dashcam fog (ACDC fog, Foggy Cityscapes); compare DAWN vs non-DAWN fog recall.
2. Weather → reliability assumption: the hand-set modifiers (e.g. fog depth ×0.45) are not derived from data. *Test:* measure detection and depth error per weather condition on nuScenes (rain, night vs clear) and Virtual KITTI 2 (fog, rain variants of identical scenes).
3. Do the risk-threshold multipliers reduce missed hazards, or only add false alarms? *Test:* precision/recall of risk alerts against ground-truth-derived TTC, per condition.
4. How much distance error comes from scale versus shape? *Test:* oracle per-frame scale/shift alignment of disparity to LiDAR (upper bound) versus the current mapping versus metric DA-V2.
5. Box bottom-centre sampling on occluded or truncated objects and the dashboard zone. *Test:* distance error versus truncation and occlusion labels (KITTI).
6. Tracker quality beyond ID counts. *Test:* HOTA / IDF1 / MOTA on KITTI tracking.
7. Weather domain shift to Indian roads. *Test:* small hand-labelled road.mp4 set (frame-level clear/overcast/rain) to measure real error.
8. fp16 effect on accuracy and speed.

## D. Distance and speed validation requirements

| Quantity | Reference needed | Alignment requirement | Metrics |
|---|---|---|---|
| Per-pixel depth | LiDAR projected into the image (KITTI Velodyne + calib; nuScenes LIDAR_TOP + calibrated_sensor) | Same timestamp (KITTI: hardware-synchronised at 10 Hz; nuScenes: keyframes at 2 Hz) | AbsRel, SqRel, RMSE, RMSE-log, δ<1.25/1.25²/1.25³, by weather |
| Object distance | 3D box centre or nearest box face (KITTI tracking labels; nuScenes annotations), or LiDAR points inside the box | Same frame; matched by 2D IoU between detections and projected 3D boxes | MAE, relative error, signed bias by range bin (0–10/10–20/20–40/40–80 m) |
| Ego speed | KITTI OXTS `vf` (forward velocity); nuScenes CAN bus `vehicle_monitor` speed, or ego_pose differencing | Timestamp interpolation to camera time | MAE (km/h), only if ego speed is estimated (e.g. visual odometry); otherwise used as an *input* |
| Relative velocity | Difference of ground-truth 3D positions of the same track ID over Δt (expressed in ego frame); nuScenes `box_velocity` minus ego velocity; nuScenes radar Doppler (`vx_comp`) as a sparse cross-check | Track identity: KITTI `track_id`, nuScenes `instance_token` | MAE (m/s), closing-sign accuracy, by baseline window |
| Valid reference coverage | Camera and depth reference must overlap: KITTI front camera ↔ Velodyne (full); nuScenes CAM_FRONT ↔ LIDAR_TOP | | % objects with reference |

`road.mp4` has **no** reference measurements and can only be used for qualitative and runtime results.

## E. TTC and risk corrections required

1. Fix distance first (B2). TTC error is roughly proportional to distance error plus relative-speed error.
2. Use ego-frame relative motion with ego speed as an input where available. Show TTC as N/A when the estimate's uncertainty is too high.
3. Velocity: keep history across short gaps (up to `track_buffer`), use a least-squares slope over ≥ 0.5 s or a 3D constant-velocity Kalman filter on (x, z), and gate outliers.
4. **Path relevance:** compute TTC only for objects within an ego corridor (e.g. |x| < half-lane + margin at closest approach), or use the closest-point-of-approach time rather than longitudinal z / v_z.
5. Risk: in-path objects only for the frame level; hysteresis or minimum dwell time for level changes; distance floor in metres only after calibration.
6. Weather multipliers should be justified (from braking-distance literature or measured perception error) and ablated (on/off).
7. Ground-truth TTC = GT distance / GT closing speed (no dataset provides TTC labels directly). Report TTC MAE for TTC < 10 s, and alert precision/recall/false-alarms per minute for each threshold.
8. Keep the "uncalibrated estimate" labels until validated numbers exist. Do not tune the formulas on the test data.

## F. Dataset recommendation

| Dataset | Components needed | Provides aligned references? | Weather coverage | Use |
|---|---|---|---|---|
| **KITTI tracking** (21 training sequences) | left colour images (image_02), velodyne, calib, oxts, label_02 (3D boxes + track IDs, every frame at 10 Hz) | **Yes, per frame**: LiDAR depth, 3D distance, track IDs → relative velocity, OXTS ego velocity | clear daytime only | Primary for distance, relative speed, TTC, tracking (HOTA) |
| **KITTI raw** (selected drives, synced+rectified) | image_02, velodyne_points, oxts, calib, timestamps | Depth and ego speed: yes. Object labels: only via tracklets on some drives | clear only | Ego-speed / visual-odometry experiments if needed |
| **nuScenes** (mini first; then a subset of trainval) | v1.0 metadata (sample, sample_data, ego_pose, calibrated_sensor, sample_annotation, instance, scene), CAM_FRONT, LIDAR_TOP, RADAR_FRONT, **CAN bus expansion** | Yes at **2 Hz keyframes** (camera ↔ LiDAR synchronised); sweeps (12 Hz camera) need interpolation. Ego pose for every sample_data. | **rain and night** scenes (scene descriptions); no fog or snow | Weather-stratified distance, velocity and TTC; the core "does weather adaptation help" experiment |
| Virtual KITTI 2 *(optional)* | RGB per variation (clone/fog/rain/overcast/morning/sunset), depth, camera poses, intrinsics, object poses | Yes, dense and exact | fog and rain on identical geometry | Controlled weather ablation of depth and detection error |
| ACDC / Foggy Cityscapes *(optional)* | images + weather labels | no geometry | fog, rain, snow, night | External weather-classifier test (fog shortcut) |

Recommendation: **KITTI tracking for geometric validation** (dense per-frame ground truth, small, fits on disk) **plus nuScenes for weather-stratified validation**, starting with nuScenes-mini (≈4 GB, 10 scenes; check rain/night coverage from `scene.description` after download) and then selected trainval scenes.
Full nuScenes trainval (hundreds of GB) does **not** fit in the 90 GB free; download metadata plus specific blob parts only.
KITTI tracking velodyne (~35 GB) plus images (~15 GB) fits, but tightly together with nuScenes. Sizes are approximate and must be checked before download.
Fog and snow cannot be validated geometrically with either dataset (Seeing Through Fog / DENSE would be needed; it requires an access request).

## G. Proposed fixes in implementation order

1. **Commit the current working tree** on a branch (code only; the .gitignore already excludes data, weights and videos). Remove `.bak` and legacy files in a separate commit, after you approve.
2. **Strict mode** (`--strict` / config `strict: true`): raise on any model fallback. Write a run manifest (git SHA, weight SHA-256, config, device, precision).
3. **Per-frame structured logger** (CSV/Parquet): frame, ts, track_id, class, box, x/z estimate, v, TTC, risk, raw and smoothed weather, weights, stage latencies.
4. **Dataset adapters** (`datasets/kitti.py`, `datasets/nuscenes.py`): yield image, timestamp, true intrinsics, ego speed and GT objects. The pipeline accepts per-frame intrinsics.
5. **Depth to metric:** remove per-frame min/max normalisation, invert disparity correctly, and evaluate three options: (a) DA-V2 metric checkpoint (outdoor/VKITTI; requires a new weight download, not yet authorised); (b) ground-plane scale recovery from camera height and pitch; (c) LiDAR-aligned oracle (upper bound only). Mask the bonnet/dashboard region.
6. **Ego-motion input** (OXTS / CAN / ego_pose). Optional monocular visual odometry for road.mp4; otherwise keep N/A.
7. **Velocity estimator:** keep history across gaps, regression or Kalman, uncertainty output.
8. **TTC/risk:** path corridor, CPA-based TTC, hysteresis, uncertainty gating. Tests against analytic scenarios.
9. **Make weather adaptation do something measurable**, or describe it honestly as display-only. For example, weather-dependent detection confidence thresholds or depth reliability weighting, then ablate adaptation on and off.
10. **FPS separation:** report inference ms, loop FPS and browser FPS (client-side `requestAnimationFrame` counter or frame sequence numbers). Add `cuda.synchronize` timing in benchmark mode. fp16 ablation.
11. **Evaluation scripts** (`eval/depth.py`, `eval/distance.py`, `eval/velocity_ttc.py`, `eval/tracking.py`, `eval/runtime.py`) with fixed splits and seeds, writing JSON/CSV to `outputs/experiments/<run>/`.
12. Ground-truth regression tests (small fixtures) and documentation (README limitations, a reproducibility section).

## H. Research experiments and metrics

| ID | Question | Data | Metrics |
|---|---|---|---|
| E1 | Depth accuracy: current vs fixed vs metric vs oracle | KITTI tracking, nuScenes keyframes | AbsRel, RMSE, δ thresholds; per range bin |
| E2 | Object distance accuracy | KITTI 3D labels, nuScenes annotations | MAE, relative error, bias by range and class |
| E3 | Relative velocity | KITTI tracks, nuScenes (+ radar cross-check) | MAE m/s, sign accuracy, window-length ablation |
| E4 | TTC | Same, GT-derived TTC | MAE for TTC<10 s; alert precision/recall/F1; false alarms per minute |
| E5 | Risk levels | Same | confusion matrix vs GT levels; alert lead time |
| E6 | Weather effect on perception | nuScenes rain/night vs clear; VKITTI2 variants | detection recall/AP (IoU 0.5 on projected boxes), depth AbsRel per condition |
| E7 | Does weather adaptation help? | Same as E6 | E4/E5 metrics with adaptation on vs off; correlation of fusion weight with measured error (Spearman ρ) |
| E8 | Weather-classifier generalisation | BDD test (done), ACDC/Foggy Cityscapes, hand-labelled road.mp4 sample | accuracy, macro-F1, fog DAWN vs non-DAWN |
| E9 | Tracking | KITTI tracking | HOTA, IDF1, MOTA, ID switches |
| E10 | Runtime | road.mp4, KITTI | inference ms per stage, loop FPS, browser FPS, GPU memory; fp32 vs fp16 |

Report mean ± 95 % CI (bootstrap over sequences), and keep tuning and test sequences separate (as already done for weather).

## I. Commands and dependencies for the next stage (not executed)

**Python packages (not installed now):**
- `nuscenes-devkit` (brings `pyquaternion`, `shapely`)
- `pykitti` (or plain parsers, which need no dependency)
- `TrackEval` (git) or `motmetrics` for HOTA/MOTA
- optional: `pyarrow` for Parquet

`ffmpeg` (apt) is needed for an H.264 browser-playable video. Already present: `open3d`, `pandas`, `scikit-learn`, `scipy`.

**Downloads** (all require registration or licence acceptance; nothing downloaded yet):
- KITTI tracking: <https://www.cvlibs.net/datasets/kitti/eval_tracking.php> → `data_tracking_image_2`, `data_tracking_velodyne`, `data_tracking_calib`, `data_tracking_oxts`, `data_tracking_label_2`
- nuScenes: <https://www.nuscenes.org/nuscenes#download> → `v1.0-mini.tgz`, then `can_bus.zip`; later trainval metadata plus selected blobs. `python download.py dataset nuscenes-mini --nuscenes-url "<signed-url>"` already exists.
- optional DA-V2 metric weights: `depth_anything_v2_metric_vkitti_vits.pth` (Hugging Face `depth-anything/Depth-Anything-V2-Metric-VKITTI-Small`)

**Safe commands to re-run this audit's checks:**
```bash
pytest tests.py -q                        # 54 tests (CUDA)
TEST_DEVICE=cpu pytest tests.py -q        # CPU
gzip -t /workspace/roadmapping_backup/project_code_latest.tar.gz
sha256sum weights/*.pth weights/*.pt
```
Running these inside the project appends to `logs/pipeline.log` and writes `__pycache__`. Use `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider` or a copy, as this audit did.
