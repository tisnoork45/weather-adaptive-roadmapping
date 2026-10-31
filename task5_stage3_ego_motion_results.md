# Task 5 Stage 3 results: experimental monocular ego motion (stopped, not integrated)

Aggregate results from the dev split only. Per-frame records and all KITTI-derived outputs stay outside the repository. The code is in `ego_motion.py` (offline only; nothing in the live pipeline imports it), `evaluate_kitti_egomotion.py` and `tests_ego_motion.py`.

## What was evaluated
**Inputs** (only what the camera pipeline has at run time):
- consecutive frames;
- the camera matrix;
- the pipeline's metric depth map and object boxes (to mask objects that may move);
- timestamps;
- optionally, the camera height as a configuration constant (1.65 m on KITTI).

KITTI OXTS was used only as ground truth.

**Conventions:**
- **Camera frame:** x right, y down, z forward.
- **Forward speed:** the camera's displacement along its own z axis per second.
- **Yaw rate:** about the camera's y axis, which points down; positive = turning right (= −wz of the OXTS).
- **Timing:** nominal 0.1 s.

**Methods:**
- **Baseline:** median flow of distant points gives yaw; median depth decrease gives speed.
- **Main estimator:** KLT background features with a forward-backward check, PnP with RANSAC on depth-backprojected points, a rotation-compensated parallax test for stationarity, and an optional speed/yaw filter.
- **Design options tried on dev:** depth range, near-point translation, ground-height rescale, flow-seeded KLT, and a road-plane homography scale from dense flow.

## Results (dev, 2-fold cross-validation inside dev)
**Folds:**
- **Fold A:** sequences 0000, 0004, 0008, 0012, 0016, 0020 (1,958 frames, 308 stationary).
- **Fold B:** sequences 0002, 0006, 0010, 0014, 0018 (1,222 frames, 244 stationary).
- **Coverage** is 1.00 for every variant shown.

| Variant | E1 speed MAE, A / B (≤ 0.5 m/s) | E2 yaw MAE, A / B (≤ 0.02 rad/s) | E3 false "moving", A / B (≤ 5%) | ms/frame |
|---|---|---|---|---|
| Baseline (all dev) | 4.08 ❌ | 0.021 ❌ | 0.20 ❌ | 9.2 |
| PnP, points ≤ 40 m | 2.57 / 1.48 ❌ | 0.028 / 0.019 | 0 / 0 ✅ | 12.5 |
| **PnP, points ≤ 80 m** | 2.83 / 1.42 ❌ | **0.011 / 0.007 ✅** | **0 / 0 ✅** | 12.0 |
| PnP, points ≤ 15 m | 1.73 / 1.92 ❌ | 0.053 / 0.111 ❌ | 0.003 / 0 ✅ | 12.4 |
| Road-plane scale | 2.72 / 1.31 ❌ | 0.011 / 0.007 ✅ | 0 / 0 ✅ | **104.5** |

- **E1 fails for every variant.** Forward speed is underestimated, increasingly with speed: the median estimated/true ratio is about 0.91 at 1–4 m/s and about 0.71–0.76 above 4 m/s.
- **The road-plane scale only partly corrects it:** the median ratio goes from 0.68 to 0.73 (fold A) and from 0.84 to 0.91 (fold B).
- **Latency:** the road-plane scale takes about 8.7× as long, beyond the 20 ms per frame used for Stage 2.
- **E2 and E3 pass on both dev folds** with the 12 ms PnP variant (points ≤ 80 m). They were **not** confirmed on the test split.

## Failure cases and open questions
- **At high speed:**
  - inlier counts fall from about 470 to about 110;
  - per-frame speed is noisy;
  - speed errors coincide with yaw errors.
- **Ruled out:**
  - near-road depth scale (road height within about 5% of 1.65 m, uncorrelated with the speed error);
  - KLT tracking too short (flow seeding made no difference).
- **Partly explained:** far-depth compression. The remaining cause is not identified.
- **Data not recorded for the road-plane variants:** the fallback rate and the per-speed breakdown.

## Decision
- Ego-motion experimentation was stopped on 2026-10-31.
- No test-split evaluation was run.
- The estimator is not integrated into the pipeline and is not suitable for integration in its current form.
