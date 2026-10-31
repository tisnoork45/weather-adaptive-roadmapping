# Task 5 evaluation protocol: object velocity, closing speed and TTC on KITTI tracking

Status: **fixed 2026-10-31, before any comparative experiment.** Every evaluation run
records this file's SHA-256 in its manifest. A change after that date is an *amendment*:
append it to section 9 with the date and the reason, and never re-score an earlier run
under the new rules without saying so.

Code: `kitti_tracking.py` (data and poses), `kitti_velocity_gt.py` (ground truth),
`velocity_eval.py` (matching and metrics), `evaluate_kitti_velocity.py` (cache and run),
`velocity_b0.py` (frozen baseline). Tests: `tests_kitti_tracking.py`.

## 1. Question and scope

How accurately does a front-facing monocular pipeline estimate the closing speed, lateral
speed and time-to-collision of the objects ahead? Each new estimator (B1-B4) is
compared with B0, the production estimator as of commit 1892061, frozen byte for byte in
`velocity_b0.py`. All estimators receive identical inputs: the production pipeline's
detections, ByteTrack IDs, metric depth (Depth Anything V2 Metric, Virtual KITTI 2,
ViT-S) and BEV positions, computed once per frame and cached.

Only the front camera (image_02) is used. LiDAR, radar and the OXTS GPS/IMU are used
**only to build ground truth**, never as estimator inputs. Nothing here describes
rear or side awareness.

## 2. Data

- **Data:** the KITTI tracking training set, all 21 sequences and 8,008 frames (CC BY-NC-SA 3.0; nothing derived from it is committed).
- **Split by sequence parity, chosen before looking at any result:**

  | Split | Sequences | Frames | Primary observations | VRU observations |
  |---|---|---|---|---|
  | **dev** | 0000, 0002, …, 0020 (11) | 3,224 | 8,476 | 2,683 |
  | **test** | 0001, 0003, …, 0019 (10) | 4,784 | 9,387 | 8,022 |

  - Every parameter of a new estimator is chosen on **dev**.
  - **test** is scored once per frozen estimator configuration, and any re-scoring after a change is reported as such.
  - The split puts each recording day's camera calibration in both halves, except sequence 0020, which is dev only. It also puts stationary-ego sequences in both: 0012 and 0016 in dev, 0017 in test.
- **Timestamps:**
  - **Primary analysis:** the benchmark has no timestamp files, so frame k is at t = 0.1·k s, the nominal 10 Hz.
  - **Measured spacing:** the true image spacing is 0.10355 s.
    - GPS displacement vs integrated OXTS forward speed, over straight 20-frame spans, puts all 15 sequences that have such spans within 0.1030–0.1040 s.
    - The labels independently confirm it: parked-car displacement per frame is uniform at 1.038 × v·0.1 s.
  - **Consequence:** all speeds in the primary analysis, for ground truth and estimators alike, are about 3.5% too large in m/s and TTCs about 3.5% too short.
    - Comparisons between estimators are unaffected, because every estimator and the ground truth use the same clock.
  - **Sensitivity analysis:** the same runs repeated with `--frame-period 0.10355`.
- **Ego poses:**
  - **Orientation:** OXTS roll/pitch/yaw, as in the devkit.
  - **Translation:** body-frame velocity (vf, vl, vu) integrated over 0.10355 s per frame.
  - **Why not GPS positions:** they are sampled on the OXTS 10 ms grid, which alternates 0.10 and 0.11 s steps.
  - **Why not 0.1 s integration:** it leaves parked cars drifting at −3.05% of ego speed.
  - **Not used:** the vn/ve fields, which agree with no axis convention.
- **Camera frame:** cam2, the frame of image_02. Labels are given in rectified cam0, and their locations are shifted by t2 = K2⁻¹·P2[:, 3] (about +6 cm in x).

## 3. Ground truth and eligibility

**Per labelled object and frame t** (definitions in `kitti_velocity_gt.py`):

| Quantity | Definition |
|---|---|
| Relative velocity (`gt_rel_vx`, `gt_rel_vz`) | Least-squares slope of the label position over frames t−2…t+2, each sample in its own frame's cam2 coordinates. All five frames must carry the same track ID. This is the apparent motion that B0 estimates. |
| Closing speed | −`gt_rel_vz` (positive = approaching) |
| Absolute velocity (`gt_abs_*`) | The same slope after mapping all samples into cam2 at t with the ego poses |
| Distance for TTC | `z_near`, the smallest z of the 3D box corners: the visible front surface of a vehicle ahead |
| GT TTC | `compute_ttc(z_near, closing speed)`, the pipeline's own function (minimum closing speed 0.15 m/s, cap 60 s) |
| Corridor | The 3D box comes within 2.0 m laterally of the camera axis, i.e. smallest abs(x) of its corners ≤ 2.0 m. A crude straight-road proxy; it ignores curvature. |

**Primary subset** (`is_primary_eligible`). Every condition must hold:
- type Car, Van or Truck;
- truncated = 0 and occluded ≤ 1;
- full ±2-frame window;
- z_near ≥ 3 m and z ≤ 60 m;
- 2D box height ≥ 25 px (the KITTI "moderate" height).

**Secondary subset (VRU):** the same rules for Pedestrian and Cyclist.

**Always excluded:** DontCare, Misc, Tram and Person (sitting).

**Ground-truth noise floor:** velocities are differences of hand-placed annotations.
- The median RMS residual of the 5-frame line fit is 0.007 m (P90 0.043 m).
- Parked cars seen from a moving ego vehicle have a median apparent absolute speed of 0.17 m/s.
- So differences between estimators below about 0.2 m/s are not resolvable.

## 4. Matching

Per frame, every pipeline detection that has a track ID is matched one-to-one to the non-DontCare labels:
- by the Hungarian algorithm on 2D IoU;
- with IoU ≥ 0.5;
- only between compatible classes (`velocity_eval.COMPATIBLE_CLASSES`).

An eligible label with no match counts against coverage. An estimator "has" a velocity
for the label when it returned `valid=True` for the matched track in that frame.

## 5. Metrics

Each metric is computed per subset and per split.

1. **Closing-speed error.**
   - e = (−v̂z) − `gt_closing`;
   - MAE (primary metric), RMSE, median, P90 and bias;
   - MAE and bias with 95% CIs.
2. **Lateral-speed error:** v̂x − `gt_rel_vx`, MAE.
3. **Coverage.**
   - *End-to-end:* valid / eligible.
   - *Given matched:* valid / matched.
   - The detection rate is reported alongside.
4. **Sign accuracy:** approaching vs receding, for |GT closing| ≥ 1 m/s.
5. **TTC bands.**
   - Bands use the clear-weather thresholds CRITICAL < 1.5 s, WARNING < 3 s and CAUTION < 5 s; NONE otherwise.
   - Estimated TTC uses the pipeline's distance `z_m` and the estimator's v̂z.
   - Scored for in-corridor objects only:
     - the confusion matrix;
     - **false-urgent rate** = P(estimate CRITICAL or WARNING | GT NONE);
     - **false-critical rate** = P(estimate CRITICAL | GT NONE);
     - **missed-urgent rate** = P(estimate CAUTION or NONE | GT CRITICAL or WARNING);
     - band agreement.
6. **Stratified closing-speed MAE:**
   - range: 3–10, 10–20, 20–40 and 40–60 m;
   - ego moving (|v| > 1 m/s) vs stationary;
   - turning (|yaw rate| > 0.1 rad/s) vs straight;
   - occlusion level.
7. **Distance input:** |z_pipeline − z_near| / z_near, a diagnostic of the shared depth input.
8. **Ego motion** (only for estimators that estimate it):
   - forward-speed and yaw-rate errors against the pose-derived camera velocity;
   - the false "moving" rate on the stationary-ego sequences.
9. **World-static objects** (only for estimators that output absolute velocity):
   - estimated absolute speed of GT-static vehicles (GT absolute speed < 0.5 m/s);
   - false "moving" rate at 1.0 m/s.
10. **Runtime:** added median latency per frame on the cache machine (H100), measured separately from the cache run.

**Statistics:**
- **Bootstrap:** 95% percentile bootstrap, resampling whole sequences (2,000 resamples, seed 0).
- **Comparisons with B0:** a paired bootstrap of the MAE difference and the relative change, on observations where *both* estimators are valid, so that a method can't gain by abstaining.
  - Coverage is judged separately (A2).
- **Note:** with 10–11 sequences per split the CIs are wide by construction.

## 6. Acceptance thresholds (pre-registered)

A candidate is **accepted** as an improvement over B0 only if all of A1–A6 hold on the **test** split, primary subset, nominal timestamps.

| # | Criterion |
|---|---|
| A1 | Paired closing-speed MAE: relative change ≤ −25% (point estimate), and the upper end of the 95% CI ≤ −10% |
| A2 | End-to-end coverage ≥ B0's coverage − 0.05 (absolute) |
| A3 | In-corridor false-urgent rate ≤ B0's, and ≤ 0.10 |
| A4 | No range bin with n ≥ 200 has a closing-speed MAE more than 10% worse than B0's (unpaired, same bin) |
| A5 | Sign accuracy ≥ B0's |
| A6 | Added median latency ≤ 20 ms per frame (H100) |

A candidate that also estimates ego motion must also meet E1–E3 on test.

| # | Criterion |
|---|---|
| E1 | Camera forward-speed MAE ≤ 0.5 m/s |
| E2 | Yaw-rate MAE ≤ 0.02 rad/s |
| E3 | False "moving" rate on stationary-ego frames ≤ 5% |

**Absolute targets** are reported but not gating, because their feasibility is unknown:
- closing-speed MAE ≤ 1.0 m/s for objects at 3–20 m and ≤ 2.0 m/s at 20–40 m;
- in-corridor false-critical rate ≤ 2%.

**Ground-truth validity gates.** All are enforced by `tests_kitti_tracking.py` and must pass before any result is reported.

| # | Gate |
|---|---|
| G1 | 3D label boxes projected with P2 match the 2D boxes: median IoU ≥ 0.95 for unoccluded, untruncated vehicles |
| G2 | Camera-frame identity rel_v = abs_v − ego_v − ω×p: median residual ≤ 0.05 m/s |
| G3 | Parked cars with the ego moving: median apparent absolute speed ≤ 0.3 m/s, and |slope vs ego speed| ≤ 1% |
| G4 | Estimated frame period from OXTS within 0.1030–0.1040 s |
| G5 | The camera yaw rate derived from the poses equals −wz (OXTS): median |difference| ≤ 0.01 rad/s |
| G6 | The B0 replay on the cache reproduces the production estimator's online output exactly: 0 mismatches over every cached frame |

## 7. Reporting rules

- Report dev and test separately, and all stratifications, including those that look bad.
- B0 and every candidate are scored from the same cache, identified by its manifest run ID and commit.
- Results labelled "nominal timestamps" are 3.5% scale-biased in m/s (section 2). The sensitivity run with the measured period is reported next to the primary result.
- Distances and speeds are calibrated-camera monocular estimates on KITTI. They say nothing about road.mp4 or any uncalibrated camera.

## 8. Known limitations

- Velocity ground truth comes from annotations, not measurements, and the timing is reconstructed.
- The corridor is a straight-road proxy.
- The weather is clear daytime only, so the weather multipliers are not exercised.
- The tracker runs with the app's `frame_rate: 30` setting on 10 Hz data. This is kept as production behaviour and is not tuned.
- The metric depth model's training camera (Virtual KITTI 2) is close to KITTI's, so depth quality here is a favourable case.

## 9. Amendments

### Amendment 1 (2026-10-31, Stage 2): additions only

Nothing in sections 1–8 changed: subsets, primary metric, thresholds A1–A6/E1–E3, gates and the split stand. The Stage 2 dev runs (`stage2/01_defaults_dev`, `02_tuned_dev`, `03_final_dev`) record the pre-amendment SHA-256 `6091e584…`. The harness computed these additions in those runs; this amendment documents them.

- **Secondary metrics.**
  - Uncertainty calibration for estimators that state σ(vz): the fractions of errors within 1σ and 1.96σ, Spearman(σ, |e|), and RMSE vs mean σ per σ quintile.
  - Coverage/MAE gating curves.
  - Compute-only latency per estimator: image and depth reads are excluded, because both are already in memory in the live pipeline.
  - TTC bands with the estimator's own distance, where it measures one.
- **Tuning procedure.**
  - Parameters are chosen on pooled dev from small, pre-listed grids.
  - Every choice is also cross-validated inside dev: fold A = 0000, 0004, 0008, 0012, 0016, 0020; fold B = the other dev sequences. Each fold's choice is scored on the other fold.
  - The pre-sweep defaults were scored first (`01_defaults_dev`) and are reported.
- **Calibration of σ.**
  - The multiplier is median(|e|/σ)/0.6745 on pooled dev, checked across the two folds.
  - The σ-gate threshold is the smallest one losing at most 5 points of coverage on pooled dev.
- **Test-split guard.** The harness refuses to score experimental estimators on test or all without `--score-test`.
