# Task 5 reproduction scripts (Stage 2)

The scripts behind the Stage 2 tuning, calibration and test acceptance analysis, copied verbatim from the session in which they were run on 2026-10-31. They read the evaluation cache and results, which live **outside** the repository (KITTI is CC BY-NC-SA 3.0, so nothing derived from it is committed).

| Script (original name) | Produced | Data used |
|---|---|---|
| `sweep_stage2.py` (`sweep.py`) | `stage2/sweeps/{depth,gate,kalman,scale}.json`: parameter grids for D, T, K and S, with 2-fold cross-validation inside dev | dev only |
| `calibrate_stage2.py` (`calib.py`) | `stage2/sweeps/calib{1,2}.json`: σ multipliers (median(\|e\|/σ)/0.6745) and the cross-fold check | dev only |
| `acceptance_stage2.py` (`acceptance.py`) | `stage2/04_test_once/acceptance_test.json`: A1–A6 with paired sequence-bootstrap CIs, from the single test run's outputs; no estimator is re-run | test outputs (read only) |

**Running them.** Run from the repository root, so that the project modules import:

```
PYTHONPATH=. python -B scripts/task5/sweep_stage2.py {depth|gate|kalman|scale} <out.json>
PYTHONPATH=. python -B scripts/task5/calibrate_stage2.py 1 <out.json>
PYTHONPATH=. python -B scripts/task5/calibrate_stage2.py 2 <out.json> 2.59
PYTHONPATH=. python -B scripts/task5/acceptance_stage2.py <evaluation run dir>
```

The originals were run from a scratch copy of the repository, because at the time importing `utils` wrote to `logs/pipeline.log` in the working directory.

**Logging since T6.1.** Importing `utils`, or any other project module, no longer creates `logs/` or `logs/pipeline.log`. Only the application entry points (`run.py`, `app.py`, `carla_sim.py`, `download.py`) configure file logging, from the `logging:` section of `config.yaml`.

The frozen Task 5 evaluation scripts (`evaluate_kitti_velocity.py`, `evaluate_kitti_egomotion.py`) and these reproduction scripts never configure logging themselves. They still print their INFO progress messages to the console, in the same format as before, but they no longer create `logs/pipeline.log`. To keep a log file, redirect the console output, e.g. `... 2>&1 | tee run.log`.

The frozen scripts were not modified; only the shared logging set-up in `utils.py` changed. Their console output was checked to be identical, and their cached results were confirmed bit-identical.

**Hard-coded inputs:**
- `sweep_stage2.py` and `calibrate_stage2.py` read the cache at `/workspace/experiments/kitti_velocity/cache/20261031-190612_1892061-dirty`, built by `evaluate_kitti_velocity.py cache`.
- The dev split (even sequence numbers) and the fold A sequences (0000, 0004, 0008, 0012, 0016, 0020) are fixed in the scripts.

**Caveats for exact reproduction:**
- **The `kalman`, `scale` and `depth` sweep groups** build estimators with explicit parameters, so they reproduce as-is.
- **`calibrate_stage2.py` ran before calibration was frozen,** when `velocity_experimental.CALIBRATION` held multipliers of 1.0 and no σ gate.
  - Against the frozen `velocity_experimental.py` (SHA-256 `61d22e18…`), the frozen multipliers are applied first. Its k68 output then comes out near 1 instead of reproducing 0.525 (K), 2.59 (S), 0.751 (KS) and 0.798 (DKS).
  - To reproduce the original numbers, run it with those multipliers temporarily reset to 1.0, in a scratch copy only.
  - Never edit the frozen file.
- **Frozen results:** every value they produced is recorded in `velocity_experimental.py` (`TUNED_*`, `CALIBRATION`) and in `/workspace/experiments/kitti_velocity/stage2/04_test_frozen_config.json`.
