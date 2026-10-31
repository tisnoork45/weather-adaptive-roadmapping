# Task 5 Stage 2 results: B0 vs the KS velocity estimator (KITTI tracking)

Aggregate results only. Per-object rows, caches and every other KITTI-derived output stay outside the repository (KITTI is CC BY-NC-SA 3.0). The protocol, subsets and acceptance criteria are in `task5_evaluation_protocol.md`; the code is in `velocity_experimental.py` and `evaluate_kitti_velocity.py`.

- **B0:** the production relative-velocity estimator, frozen byte for byte in `velocity_b0.py` (SHA-256 `4c7b8823…`).
- **KS:** an offline time-aware Kalman filter on the pipeline's metric object positions, plus a closing-speed cue from KLT image-scale change (vertical separations, forward-backward check).
  - Frozen in `velocity_experimental.py` (SHA-256 `61d22e18…`).
  - **Not** integrated into the live pipeline.
- **Shared inputs:** all estimators replay the same cached outputs of the unchanged production pipeline: detections, ByteTrack IDs, metric Depth Anything V2 depth, and BEV positions.
- **B0 equivalence:** B0's replay reproduces the production estimator exactly (dev 18,036 values, test 29,860 values, 0 mismatches).
- **Timing:** nominal 0.1 s.

## Procedure
1. **Design and tuning on the dev split only** (even sequence numbers), with 2-fold cross-validation inside dev. The scripts are in `scripts/task5/`.
2. **Freezing:** the configuration and the σ multipliers were frozen and checksummed before the test run.
3. **One test run:** the test split (odd sequence numbers) was scored once. There was no tuning or recalibration afterwards.

## Primary subset (vehicles), closing speed
**Columns:**
- **MAE:** mean absolute closing-speed error in m/s, with a 95% sequence-bootstrap CI.
- **Paired change:** relative MAE change against B0 on observations where both are valid, with a paired sequence-bootstrap CI.
- **Latency:** median compute per frame (H100).

| Split | Estimator | Coverage | MAE [95% CI] | Paired change vs B0 [95% CI] | Sign accuracy | ms/frame |
|---|---|---|---|---|---|---|
| dev | B0 | 0.870 | 4.09 [3.48, 5.24] | — | 0.883 | 0.04 |
| dev | KS | 0.868 | 1.47 [1.13, 2.07] | −63.5% [−70.7, −50.5] | 0.982 | 5.4 |
| **test** | B0 | 0.907 | 3.87 [3.19, 4.48] | — | 0.911 | 0.04 |
| **test** | **KS** | 0.906 | **1.14 [0.93, 1.39]** | **−69.7% [−72.5, −65.2]** (n = 8,467) | 0.993 | 7.1 (p95 16.9) |

**Test, KS vs B0:**
- **Lateral-speed MAE:** 0.90 vs 1.49 m/s.
- **False urgent alerts** (in-corridor TTC band CRITICAL or WARNING when ground truth says neither): 0.0007 vs 0.024.
- **Missed urgent alerts:** 0.069 vs 0.155.
- **MAE by distance:**

  | Distance | KS | B0 | KS/B0 |
  |---|---|---|---|
  | 3–10 m | 0.78 | 1.17 | 0.67 |
  | 10–20 m | 0.70 | 2.19 | 0.32 |
  | 20–40 m | 1.35 | 5.34 | 0.25 |
  | 40–60 m | 2.83 | 9.02 | 0.31 |

## Acceptance on test (protocol section 6, primary subset)
| | A1 MAE −25% (CI ≤ −10%) | A2 coverage | A3 false urgent | A4 range bins | A5 sign | A6 latency | Verdict |
|---|---|---|---|---|---|---|---|
| K (Kalman only) | ✅ −32.1% | ✅ | ✅ | ✅ | ✅ | ✅ | pass |
| S (scale cue only) | ✅ −60.0% | ✅ | ✅ | ❌ 3–10 m ×1.17 | ✅ | ✅ | **fail** |
| **KS** | ✅ −69.7% | ✅ | ✅ | ✅ | ✅ | ✅ | **pass** |
| KS + σ gate (exploratory) | ✅ −69.3% | ✅ | ✅ | ✅ | ✅ | ✅ | pass |

## Pedestrians and cyclists (secondary subset, test)
- **KS vs B0:**
  - MAE 1.07 vs 2.01 m/s; paired change −46.6% [−64.6, −34.2];
  - coverage 0.711 vs 0.704;
  - sign accuracy 0.933 vs 0.913.
- **Open limitation, sign accuracy with a stationary ego vehicle:**
  - The scale cue reverses the sign more often than B0 when the ego vehicle is stationary: S 0.80, KS 0.81, B0 0.89. Slow-walking pedestrians are the main source of these errors.
  - On dev, 235 of the scale cue's 245 wrong-sign cases came from one stationary-ego sequence.

## Limitations
- **Ground truth:** velocity is differenced from annotations, and timing is reconstructed; speeds at nominal timing are about 3.5% high, for estimators and ground truth alike.
- **Corridor:** a straight-road proxy.
- **Urgent alerts:** few cases (624 urgent in-corridor observations on test).
- **Data:** clear daytime only; the camera and depth model are a favourable match for KITTI.
- **Untested elsewhere:** road.mp4 and uncalibrated cameras.
