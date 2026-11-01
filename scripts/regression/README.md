# Regression checks

`golden.py` checks that the perception pipeline still produces exactly the outputs it produced before. `tests_regression.py` runs these checks automatically, and skips them with a clear reason when the data they need is absent.

| Check | Compares | Needs |
|---|---|---|
| `compare --mode relative` / `--mode metric` | `road.mp4` frames 0–239, fed as `run.py` feeds a video, against a stored golden | `road.mp4`, a golden in `outputs/regression/`, a CUDA GPU, a matching environment fingerprint |
| `reset --mode metric` | One pipeline before and after `reset()`; two pipelines sharing thread-safe models in parallel threads (`road.mp4`, 60 frames) | `road.mp4`, a CUDA GPU |
| `kitti` | KITTI tracking sequence 0000, frames 0–49, through the frozen Task 5 harness, against the frozen Task 5 cache | The KITTI tracking data and the Task 5 cache (read only), a CUDA GPU |

What each check compares:
- **Video goldens:** every per-frame output:
  - detections and track IDs;
  - smoothed and raw weather;
  - a SHA-256 of the depth map;
  - fusion weights;
  - BEV objects and image hash;
  - velocities, TTC and risk.

  The comparison is exact, because on one machine the pipeline is bit-reproducible.
- **KITTI check:**
  - it writes a new 50-frame cache to a scratch directory and only *reads* the frozen cache;
  - it compares every cached field except latency, plus the float16 depth maps.

**Track IDs in a rebuilt KITTI cache.** Since T6.2, every tracker numbers its tracks from 1. Before, one counter was shared by every tracker in the process.

The frozen harness (`evaluate_kitti_velocity.py cache`) builds a new pipeline for each sequence, so a rebuilt cache can differ from the frozen one in its numeric track IDs:
- The first sequence processed matches exactly. That is sequence 0000, which this check uses.
- Later sequences have each track ID lowered by a constant: the number of tracks the earlier sequences created. In a check of sequences 0000 and 0001, 0001's IDs were 105 lower.
- Every other cached field is unchanged: detections, classes, confidences, boxes, depth and BEV positions. The metrics are unchanged too, because the evaluation uses track IDs only as keys within one sequence.
- A rebuilt cache is therefore not byte-identical to the frozen cache for sequences after the first. The frozen cache and results are not affected.

## Where goldens live

Goldens are stored in `$WARM_REGRESSION_DIR`, by default `outputs/regression/`, which is git-ignored. Each mode has two files:
- `road_<mode>.json.gz`: the golden itself;
- `road_<mode>.manifest.json`: its provenance, a SHA-256 of the data, and an environment fingerprint.

The fingerprint records the SHA-256 of the video, `config.yaml` and the checkpoints, plus the torch version and GPU name. GPU outputs are only bit-identical on the same software and hardware. When the fingerprint differs, `compare` therefore reports **skipped** (exit status 2) rather than a false regression.

## Creating goldens

**After a change that is *meant* to alter outputs:**
- record new goldens and state why: `python -B scripts/regression/golden.py record --mode relative --provenance "..."`;
- never re-record goldens just to make a failing check pass.

**The current goldens** were adopted from the T6.1 baseline:

```
python -B scripts/regression/golden.py adopt --mode relative --source <t6.1 base_relative_r1.json.gz> --provenance "..."
python -B scripts/regression/golden.py adopt --mode metric   --source <t6.1 base_metric_r1.json.gz>   --provenance "..."
```

The T6.1 baseline goldens were produced at commit `f51e868`. The pipeline reproduced them bit for bit at every later step through commit `ab53ad4`.

## Running

```
pytest tests_regression.py -v                                   # all checks; GPU checks run in subprocesses
python -B scripts/regression/golden.py compare --mode metric    # one check directly
python -B scripts/regression/golden.py kitti --out /tmp/kitti_check
```

The GPU checks run in a separate process, so global torch settings changed by other tests cannot affect bit-exact results. One example is `cudnn.deterministic`, which the weather-training smoke test sets.
