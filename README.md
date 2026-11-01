# Weather Adaptive Road Mapping for Autonomous Vehicles

A weather-adaptive perception system for autonomous vehicles, in pure Python, built on a **single forward-facing monocular camera**. Its stages:
- object detection;
- monocular depth estimation;
- weather classification;
- dynamic confidence estimation;
- weather-adaptive weighting of the camera modules;
- multi-object tracking;
- bird's-eye-view (BEV) mapping;
- relative-velocity, time-to-collision (TTC) and risk assessment.

They run as one real-time pipeline, with a live Streamlit dashboard.

> **Final Year Engineering Capstone Project.** No ROS. It runs from a plain Python environment, or in VS Code. The reference results were produced in an NVIDIA NGC PyTorch container; see [ENVIRONMENT.md](ENVIRONMENT.md).

**Scope: what this system is and is not.**
- **Sensors.** The only sensor is one forward-facing camera. LiDAR, radar and IMU are **not** used, and there is no rear or side awareness.
- **"Fusion."** "Adaptive sensor fusion" here means weather-dependent weighting of two camera-derived modules (detection and depth). Those weights and the scene confidence are shown in the UI; they do not feed the velocity, TTC or risk computation.
- **Distances and speeds.** These need metric depth, and they are **uncalibrated monocular estimates** unless `camera.calibrated: true` holds for a real calibration.
- **Default mode.** The application default is relative depth, which has no unit, so it shows no distances.

---

## Repository Structure

The repository has a flat layout: one module per concern, all in the root. Core runtime files:

```
weather-adaptive-roadmapping/
├── README.md, ENVIRONMENT.md      # this file; environments (standard vs NGC reference) and the lock file
├── requirements.txt               # supported version ranges (what the software needs)
├── requirements-lock.txt          # exact versions of the reference environment (generated)
├── setup.py                       # pip install -e . (installs every module listed in PY_MODULES)
├── config.yaml                    # ALL configuration in one file (validated on load, see "Configuration")
│
├── run.py                         # headless / OpenCV-window entry point (--record-run for structured logs)
├── app.py                         # Streamlit dashboard
├── carla_sim.py                   # CARLA simulator bridge (not tested in the reference environment)
├── download.py                    # model weights + dataset setup helpers
│
├── pipeline.py                    # PerceptionPipeline: orchestrates every stage per frame + dashboard render helpers
├── models.py                      # YOLO11x detector, Depth Anything V2 (relative/metric), EfficientNet-B0 weather
│                                  #   classifier + its trainer; heuristic fallbacks unless strict
├── tracking.py                    # ByteTrack (Kalman filter + two-stage Hungarian matching)
├── fusion.py                      # confidence estimation + weather-adaptive module weights (display only)
├── mapping.py                     # BEV projection (metric depth only) + optional Open3D point cloud
├── camera.py                      # pinhole camera model (calibrated, or placeholder intrinsics)
├── relative_velocity.py           # production relative-velocity estimator ("B0", frozen for Task 5)
├── ttc_calculator.py              # time-to-collision per object
├── risk_engine.py                 # weather-adjusted TTC risk levels
├── utils.py                       # config loading, logging, metrics, drawing, video IO, device selection
├── config_schema.py               # config.yaml schema and validation (clear errors with key paths)
├── run_logging.py                 # opt-in run recorder: manifest.json + frames.csv + objects.csv
├── depth_anything_v2/             # vendored Depth Anything V2 model code (used at run time)
│
├── prepare_weather_dataset.py     # BDD100K -> weather dataset (see "Training the Weather Classifier")
├── merge_dawn_weather.py          # adds DAWN fog/rain/snow with de-duplication
│
├── scripts/env/                   # generate_lock.py, check_env.py (no installs)
├── weights/                       # model checkpoints (git-ignored)
├── data/                          # datasets (git-ignored)
├── logs/                          # rotating log file, written by the entry points only
└── outputs/                       # videos, frames, weather runs; outputs/runs/ and outputs/experiments/ git-ignored
```

The offline evaluation files run outside the live pipeline. All KITTI data and derived outputs stay outside Git (CC BY-NC-SA 3.0).

| Files | Purpose |
|---|---|
| `kitti_depth.py`, `depth_metrics.py`, `depth_methods.py`, `evaluate_kitti_depth.py`, `tests_depth.py` | KITTI depth evaluation (see "Depth Modes and KITTI Evaluation") |
| `kitti_tracking.py`, `kitti_velocity_gt.py`, `velocity_eval.py`, `velocity_b0.py`, `evaluate_kitti_velocity.py`, `tests_kitti_tracking.py`, `task5_evaluation_protocol.md` | Task 5: KITTI tracking velocity evaluation of the production estimator. `velocity_b0.py` is a byte-identical frozen copy of `relative_velocity.py`. |
| `velocity_experimental.py`, `tests_velocity_experimental.py`, `scripts/task5/`, `task5_stage2_results.md` | Task 5 Stage 2: the experimental KS velocity estimator. Offline only; **not integrated** into the live pipeline. |
| `ego_motion.py`, `evaluate_kitti_egomotion.py`, `tests_ego_motion.py`, `task5_stage3_ego_motion_results.md` | Task 5 Stage 3: experimental monocular ego-motion. Stopped; **not integrated**. |
| `tests.py`, `tests_foundation.py`, `tests_regression.py` | Pipeline tests; configuration, logging, recorder, packaging and lock tests; golden-output regression and runtime-state tests |
| `scripts/regression/` | `golden.py` (golden `road.mp4` outputs, KITTI check against the Task 5 cache, reset determinism) and its README |
| `research_audit.md` | Earlier research audit |

**Kept, pending a separate clean-up decision (not used by the software):**
- the `*.bak` / `*.before_*` backup copies, `models_backup_before_weather.py`, `fix_run_py.py` and `test_risk_runtime.py`;
- `Depth-Anything-V2/`, the upstream repository clone (the pipeline uses the vendored `depth_anything_v2/`);
- `risk_delivery/`, `bdd_files.csv`, `test_depth_frame.jpg` and `streamlit_run.log`.

## Architecture / Pipeline

```
Camera / Video / CARLA
        │  BGR frame
        ▼
┌────────────────────────────────────────────────────┐
│              PerceptionPipeline.process()            │
│                                                        │
│  1. WeatherClassifier        → weather + probs          │
│  2. YOLODetector             → object boxes               │
│  3. ByteTracker              → track IDs                    │
│  4. DepthEstimator           → dense depth map                 │
│  5. ConfidenceEstimator      → per-module confidence              │
│  6. AdaptiveFusionEngine     → dynamic sensor weights                │
│  7. BEVMapper                → point cloud + top-down map               │
│  8. RelativeVelocityEstimator→ per-track closing/lateral velocity          │
│  9. compute_ttc_for_objects  → time-to-collision per object                   │
│ 10. RiskEngine                → weather-adjusted risk level                      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 ▼
              app.py (Streamlit dashboard) / run.py (OpenCV)
```

**Lane detection was removed from this project.** An earlier revision included a classical Canny+Hough-transform lane detector (`LaneDetector` in `models.py`); it has been fully removed from the backend, the BEV projection, both UIs, and the config/test surface. It is not part of the current architecture.

Every model wrapper in `models.py` follows the same pattern: try to load the real pretrained checkpoint; if unavailable (no internet, weights not yet downloaded, license not yet accepted), fall back to a deterministic classical-CV / heuristic implementation, so **the whole pipeline always runs end-to-end**, which is essential for demoing the adaptive-fusion logic without every large checkpoint being present.

**Strict mode** (`pipeline.strict: true` in config.yaml, or `python run.py --strict`) turns every fallback into an error instead, so a research run can never silently use a heuristic. The KITTI evaluation always runs strict. Heuristic depth, when allowed, is tagged as such and never used as metres.

---

## Model Status — What's Actually Trained vs. Classical CV

Being precise about this matters for a capstone report. Here is the honest state of every component:

| Component | Status | Notes |
|---|---|---|
| **Object Detection** (YOLO11x) | ✅ Genuine pretrained deep model | Official Ultralytics COCO-pretrained weights, loaded and run as-is. |
| **Depth Estimation** (Depth Anything V2) | ✅ Genuine pretrained deep models | Two explicit modes (see "Depth Modes"): **relative** (official ViT-S; unitless inverse depth, so distance/speed/TTC/risk show N/A) and **metric** (official Metric Outdoor ViT-S trained on Virtual KITTI 2; metres). Metric mode is evaluated against KITTI LiDAR — a favourable camera domain; on an uncalibrated camera such as road.mp4 its metres remain estimates. |
| **Weather Classification** (EfficientNet-B0) | ✅ Genuine fine-tuned deep model | Fine-tuned on BDD100K + DAWN (see "Selected weather checkpoint and results" below): held-out test accuracy 0.823, macro-F1 0.833. **Dashcam fog performance is not established.** No generic weather checkpoint is downloaded (see "A note on the previous broken download" below). |
| **Object Tracking** (ByteTrack) | ✅ Genuine, correctly implemented algorithm | ByteTrack is a Kalman-filter + Hungarian-matching *algorithm*, not a neural network — it was never meant to have separate pretrained weights. Fully functional as shipped, no training required. |
| **Adaptive Fusion / Confidence Estimation** | ⚠️ Rule-based, **display only** | Weather-conditioned weighting (with EMA smoothing) of two **camera-derived** modules: detection and depth. Hand-tuned rules, not a trained model. The weights and the scene confidence appear in the UI. They do **not** feed BEV, velocity, TTC or risk. Risk is weather-adjusted separately, through `risk.weather_multipliers`. |
| **BEV Mapping** | ✅ Geometric computation, **metric depth only** | Deterministic pinhole back-projection, not ML. Disabled in relative-depth mode. With placeholder intrinsics (`camera.calibrated: false`) positions are approximate. |
| **Relative Velocity / TTC / Risk** | ⚠️ Implemented; **large errors** | Finite-difference velocity from BEV track history, then time-to-collision, then weather-multiplied risk thresholds (`relative_velocity.py`, `ttc_calculator.py`, `risk_engine.py`). Needs metric depth. <ul><li>**KITTI test split** (Task 5): closing-speed MAE 3.87 m/s.</li><li>**road.mp4:** depth noise produces many false CRITICAL frames.</li></ul> An experimental estimator (KS) did much better offline (`task5_stage2_results.md`) but is **not integrated**. |
| **Multi-sensor input (LiDAR / radar / IMU)** | ❌ Not implemented | Camera only. `download.py` still has dataset set-up helpers for nuScenes and Seeing Through Fog, but no code reads their LiDAR or radar data. |
| **Ego-motion, rear/side awareness** | ❌ Not implemented | Forward camera only. Monocular ego-motion was tried offline and stopped (`task5_stage3_ego_motion_results.md`). Relative velocities are therefore relative to the moving ego vehicle. |
| **CARLA bridge** (`carla_sim.py`) | ⚠️ Present, untested here | The `carla` package and server are not installed in the reference environment. |

### A note on the previous broken weather-classifier download
An earlier version of `download.py` downloaded a checkpoint from a *different* EfficientNet-B0 implementation (lukemelas/EfficientNet-PyTorch, key names like `_conv_stem`, `_fc`) into this project's *torchvision*-based model (key names like `features.0`, `classifier.1`). Since **zero** keys overlapped between the two naming conventions, `load_state_dict(strict=False)` silently loaded nothing — the "pretrained" classifier was actually running on random initialization while appearing to load successfully. This has been fixed: `download.py` no longer downloads a generic checkpoint at all (torchvision's own correctly-matched ImageNet weights are fetched automatically inside training instead), and `WeatherClassifier` now refuses to load any checkpoint that isn't in this project's own verified format, falling back to the heuristic classifier instead of silently running on garbage weights.

### Lane Detection has been removed
An earlier revision of this project included a classical Canny-edge + Hough-transform lane detector (`LaneDetector` in `models.py`), plus a never-loaded architectural sketch (`UFLDv2Net`) for a future learned lane model. Both have been fully removed: there is no lane detector, no lane overlay, no lane confidence signal, and no lane-related configuration anywhere in the current pipeline, BEV projection, dashboards, or test suite. If lane detection is reintroduced in the future, it should be designed fresh rather than restored from history.

---

## Depth Modes and KITTI Evaluation

Depth Anything V2 runs in one of two explicit modes (`depth_estimation.mode` in config.yaml, or `python run.py --depth-mode metric`):

| Mode | Checkpoint | Output | Distance / speed / TTC / risk |
|---|---|---|---|
| `relative` (**default**) | `weights/depth_anything_v2_vits.pth` | unitless inverse depth (near = large) | **N/A** — no metric values are invented |
| `metric` | `weights/depth_anything_v2_metric_vkitti_vits.pth` → symlink to `/workspace/checkpoints/…` (SHA-256 `9203e538…`, HF revision `c725b858`) | depth in metres, (0, 80] | computed; labelled *"metric depth; camera uncalibrated"* unless `camera.calibrated: true` |

**Why the old conversion was removed.** The pipeline used to min/max-normalise the relative output and map it linearly onto 0–80 m. The relative output is *inverse* depth, so that mapping is not a distance: on KITTI it gives AbsRel 2.50 and δ<1.25 of 0.02 (table below). A frozen copy is kept in `depth_methods.py` only to measure this.

### KITTI depth evaluation

```bash
python -B evaluate_kitti_depth.py            # all 1,000 frames of val_selection_cropped, strict models
pytest tests_depth.py -v                     # units, masks, calibration, loader, metric correctness
```

Data: KITTI depth-prediction *selected validation* set (1,000 frames, 13 drives, 1216×352, per-frame intrinsics) from `data_depth_selection.zip` (SHA-256 `11028c5f…`), stored outside the repository under `/workspace/datasets/kitti/`. KITTI is CC BY-NC-SA 3.0, so every derived output goes to `outputs/experiments/` (git-ignored).

Standard protocol (GT in (0.001, 80] m), mean over images, 95% CI from a bootstrap over the 13 drives:

| Method | Uses GT? | AbsRel | RMSE (m) | RMSE-log | δ<1.25 | frames |
|---|---|---|---|---|---|---|
| Metric ViT-S | no | 0.132 [0.117, 0.147] | 5.26 [4.51, 5.81] | 0.210 | 0.806 [0.737, 0.862] | 1000 |
| Metric ViT-S × fx/725 | no | 0.134 | 5.30 | 0.213 | 0.803 | 1000 |
| Old linear 0–80 m conversion | no | 2.502 [2.346, 2.646] | 32.99 | 1.236 | 0.022 | 1000 |
| Ground plane (K + 1.65 m height) | no | 0.170 [0.127, 0.217] | 7.05 [5.78, 8.29] | 0.200 | 0.825 [0.775, 0.875] | 994 (6 fits rejected) |
| Relative aligned to GT per frame (**oracle**) | yes | 0.081 [0.077, 0.088] | 3.63 | 0.123 | 0.939 | 1000 |

The official-devkit protocol (no depth cap, devkit SqRel) and range/camera breakdowns are in each run's `summary.md`. Per-frame, the metric model is robust (worst-frame AbsRel 0.36), while the ground-plane baseline matches it on typical frames (median 0.127 vs 0.125) but fails badly when the road ahead is not a flat plane seen from a level camera (20 frames with AbsRel > 0.5, worst 6.6).

**What these numbers do and do not show.**
- They are *pixel-wise depth* accuracy. They are **not** object-distance accuracy (this subset has no 3D boxes); good depth metrics do not by themselves establish accurate object distances.
- KITTI is a **favourable camera domain** for the metric model (focal length 707–722 px vs its 725 px training camera; clear daytime). They do **not** show that metric depth generalises to road.mp4 or other cameras.
- On road.mp4 (uncalibrated phone/dashcam), metric mode places objects at plausible distances (e.g. a car at 30.9 m that the old conversion put at 75.1 m), but the unchanged velocity estimator turns per-frame depth noise into large speed errors (45 % of frames CRITICAL over the first 240 frames). Velocity/TTC are a separate roadmap task, which is why `relative` remains the application default.

---

## Installation

Two environments are documented in [ENVIRONMENT.md](ENVIRONMENT.md):
- **Standard installation:** any machine; versions within the `requirements.txt` ranges.
- **NGC reference environment:** the NVIDIA PyTorch 24.06 container in which every reported result was produced. Exact versions are in `requirements-lock.txt`.

Its PyTorch and torchvision are NVIDIA container builds, which only that image reproduces.

### Prerequisites
- Python 3.9–3.12 and Git.
- Optional: an NVIDIA GPU with CUDA.
- Optional: VS Code with the Python extension.

### 1. Clone and create a virtual environment

**Linux / macOS:**
```bash
git clone https://github.com/<your-username>/weather-adaptive-road-mapping.git
cd weather-adaptive-road-mapping
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/<your-username>/weather-adaptive-road-mapping.git
cd weather-adaptive-road-mapping
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install PyTorch, then the dependencies and the project

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU (CUDA 12.1 build)
# or: pip install torch torchvision                                                 # CPU, or Apple Silicon (mps)
pip install -r requirements.txt
pip install -e .                                                                     # every module + depth_anything_v2/
```

**In the NGC reference container**, PyTorch is already installed:
- `pip install -r requirements-lock.txt` should report every package as already satisfied;
- then run `pip install -e .`.

### 3. Check the environment

```bash
python scripts/env/check_env.py
```

This reports packages that are missing or outside their `requirements.txt` range, and differences from the reference lock. It installs nothing.

### 4. Model weights

Weights are not in Git; see "Downloading Model Weights".

| Checkpoint | How to get it |
|---|---|
| `weights/yolo11x.pt`, `weights/depth_anything_v2_vits.pth` | `python download.py models` |
| `weights/depth_anything_v2_metric_vkitti_vits.pth` | `python download.py models --only depth_metric` (SHA-256 verified). In this workspace it is a symlink to `/workspace/checkpoints/`. |
| `weights/efficientnet_weather.pth` | Trained locally; see "Training the Weather Classifier". No generic checkpoint is downloaded. |

Without a checkpoint the pipeline uses a heuristic fallback and says so in the log. `--strict` (or `pipeline.strict: true`) turns that into an error.

### 5. Run the tests

```bash
pytest tests.py tests_depth.py tests_kitti_tracking.py tests_velocity_experimental.py tests_ego_motion.py tests_foundation.py tests_regression.py -q
```

## Setup (VS Code)

```bash
code .
```
Select the `venv` interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*). Pre-configured run/debug targets are in `.vscode/launch.json`: headless demo, webcam, Streamlit dashboard, model download, weather-classifier training, and the full test suite.

All required folders (`weights/`, `data/`, `logs/`, `outputs/`) already exist with `.gitkeep` placeholders.

---

## Dataset

The evaluations use **KITTI**, which is stored outside the repository:
- the depth selection set for depth;
- the tracking benchmark for velocity (Task 5).

`download.py` also has set-up helpers for the sources below. **No pipeline or evaluation code reads them yet**, and in particular no LiDAR or radar data is used.

| Source | Status in this project | Access |
|---|---|---|
| **nuScenes** | Set-up helper only (camera frames could be fed as video; no dataset reader) | Free registration at [nuscenes.org](https://www.nuscenes.org) |
| **Seeing Through Fog (STF)** | Set-up helper only (its LiDAR and radar are not used) | Access request required |
| **CARLA** | Simulator bridge in `carla_sim.py` (untested in the reference environment) | Free download, no registration |

```bash
python download.py dataset nuscenes-mini
python download.py dataset seeing-through-fog
python download.py dataset carla
python download.py dataset --all           # print instructions for all three
python download.py dataset --verify-nuscenes
```

**No dataset yet?** Every entry point supports a built-in **synthetic demo stream** (`--synthetic` flag / dashboard checkbox) — run the entire pipeline immediately with zero setup.

*(For the weather classifier specifically, see "Training the Weather Classifier" under "Downloading Model Weights" below — it uses a BDD100K Kaggle mirror rather than nuScenes/STF, since it needs `weather`/`timeofday`-labeled images.)*

### CARLA weather testing
```bash
./CarlaUE4.sh -quality-level=Low         # or CarlaUE4.exe on Windows
python carla_sim.py --weather fog --show --max-frames 500
```

---

## Downloading Model Weights

```bash
python download.py models                 # downloads YOLO11x + Depth Anything V2
python download.py models --only yolo depth
python download.py models --only depth_metric   # metric-depth checkpoint, SHA-256 verified
```

| Model | Path | Source |
|---|---|---|
| YOLO11x | `weights/yolo11x.pt` | Auto-downloaded via `ultralytics` |
| Depth Anything V2 (ViT-S) | `weights/depth_anything_v2_vits.pth` | HuggingFace |
| Weather classification | *(none — train your own, see below)* | See walkthrough below |

`python download.py models --only weather` prints an explanation instead of attempting a broken download (see "Model Status" above for why). `download.py` also still accepts a legacy `lane` target for backward compatibility with old scripts/muscle memory — it is a no-op that only prints an explanatory message; there is no lane detector left in the project for it to download weights for.

### Training the Weather Classifier (real walkthrough)

**Step 1 — get a real, road-scene, weather-labeled dataset via the Kaggle mirror.**
This project uses BDD100K (Berkeley DeepDrive) images, because — unlike generic outdoor-photo weather datasets — it's actual dashcam footage labeled with both `weather` (`clear`/`rainy`/`snowy`/`overcast`/`partly cloudy`/`foggy`) and `timeofday` (`daytime`/`night`/`dawn-dusk`) attributes, which is exactly the information needed to build our six classes.

If the official `bdd100k.com` / `dl.cv.ethz.ch` sites give your browser a certificate error, use the **Kaggle mirror** instead (`solesensei/solesensei_bdd100k`) — same images, same label schema, served from `kaggle.com` which doesn't have that certificate problem, and scriptable end-to-end via the Kaggle CLI:

```powershell
# 1. Install the Kaggle CLI (one-time)
pip install kaggle

# 2. Get a free Kaggle API token (one-time, via kaggle.com, no cert issues there):
#      - Create/log into an account at https://www.kaggle.com
#      - Go to https://www.kaggle.com/settings -> API -> "Create New Token"
#      - This downloads kaggle.json to your Downloads folder. Move it into place:
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"

# 3. Download the dataset (several GB -- this will take a while; let it run)
New-Item -ItemType Directory -Force -Path "bdd100k_download" | Out-Null
cd bdd100k_download
kaggle datasets download -d solesensei/solesensei_bdd100k

# 4. Extract it
Expand-Archive -Path "solesensei_bdd100k.zip" -DestinationPath . -Force
cd ..
```
(No Kaggle CLI/token? You can also just click the plain "Download" button on
https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k in your browser instead — kaggle.com itself isn't affected by the certificate issue on the official mirror.)

The extracted layout is:
```
bdd100k_download\bdd100k\bdd100k\images\100k\train\   (images)
bdd100k_download\bdd100k\bdd100k\images\100k\val\     (images)
bdd100k_download\bdd100k\bdd100k\labels\              (label .json files)
```
(If a future re-upload of the mirror nests things slightly differently, just point the paths in Step 2 at wherever the `train`/`val` image folders and the `labels` folder actually ended up — `prepare_weather_dataset.py` accepts either an exact `.json` file or the containing folder for `--labels-train`/`--labels-val`, and will auto-detect the right file by name.)

**Step 2 — build the class-organized dataset (BDD100K + DAWN).**

The commands below are the exact procedure used for the shipped checkpoint (Linux paths; on Windows use backtick `` ` `` line continuations). In this workspace `data/weather_dataset/bdd100k` and `data/weather_dataset/dawn` are symlinks to the raw datasets, which are only ever read, never modified.

```bash
B=data/weather_dataset/bdd100k
# 1. BDD100K -> train/val/test. Add --dry-run first to preview the class distribution.
python prepare_weather_dataset.py \
    --images-train $B/bdd100k/bdd100k/images/100k/train \
    --images-val   $B/bdd100k/bdd100k/images/100k/val \
    --labels-train $B/bdd100k_labels_release/bdd100k/labels \
    --labels-val   $B/bdd100k_labels_release/bdd100k/labels \
    --output data/weather_dataset_c3000 \
    --max-per-class 3000 --max-per-class-val 150 --val-fraction-as-test 0.3 --seed 42

# 2. DAWN fog/rain/snow -> train, with 30% of DAWN fog held out to val/test (15% each).
python merge_dawn_weather.py --dawn-images data/weather_dataset/dawn --dawn-labels data/weather_dataset/dawn \
    --dataset-root data/weather_dataset_c3000 --holdout-fraction 0.3
```

How leakage between splits is prevented (each was a real problem found in an earlier build):
- **BDD100K recording sessions.** File names are `<session>-<frame>`. `test/` is carved out of the BDD val pool **per session**, so frames from one drive can't sit in both val and test (87 sessions did before), and any session that also appears in BDD's train pool is dropped from val/test.
- **DAWN exact duplicates.** DAWN ships byte-identical images under different names (e.g. `haze-024`, `haze-039` and `foggy-060` are one file). `merge_dawn_weather.py` deduplicates by SHA-256 (19 skipped).
- **DAWN near-duplicates.** DAWN also has bursts of near-identical shots, sometimes filed under two conditions. Images within 24 bits on a 256-bit perceptual hash are grouped and each group is kept in one split (28 groups). The threshold was chosen by inspecting pairs: ≤16 bits were the same scene, ≥26 different scenes.

Resulting dataset (`data/weather_dataset_c3000`, 17,328 images, verified: 0 labels disagreeing with the raw BDD JSON / DAWN filenames, 0 shared filenames, 0 identical files and 0 BDD sessions across splits; the near-duplicate check flags 1 cross-split pair at exactly the 24-bit threshold, inspected and confirmed to be two different low-texture scenes, a BDD night-rain windshield and a DAWN fog road):

| class | train (BDD + DAWN) | val (BDD + DAWN) | test (BDD + DAWN) |
|---|---|---|---|
| clear | 3000 + 0 | 150 + 0 | 150 + 0 |
| rain | 3000 + 199 | 150 + 0 | 150 + 0 |
| fog | 130 + 198 | 9 + 41 | 4 + 44 |
| snow | 3000 + 203 | 150 + 0 | 150 + 0 |
| night | 3000 + 0 | 150 + 0 | 150 + 0 |
| overcast | 3000 + 0 | 150 + 0 | 150 + 0 |

Class mapping from BDD100K attributes (priority order fog > snow > rain > night > overcast > clear): `partly cloudy` counts as **clear**, and rainy/snowy images taken **at night** are labelled **rain/snow**, not night (about 40% of BDD rain images are night-time).

**Step 3 — train on the GPU (validation-only model selection).**
```bash
python models.py --data-dir data/weather_dataset_c3000 --output outputs/weather_runs/B_c3000-s42/model.pth \
    --run-dir outputs/weather_runs/B_c3000-s42 --seed 42 --epochs 20 --num-workers 8 --no-test-eval
```
- Images are decoded and preprocessed by the **same functions** `WeatherClassifier` uses at inference (`load_weather_rgb` + `preprocess_weather_rgb`: OpenCV decode, RGB, squash-resize to 224×224 with `INTER_AREA`, ImageNet normalization). A test asserts the two paths produce identical tensors. Previously training used PIL + torchvision `Resize` while inference used `cv2.resize(INTER_LINEAR)`, which cost the deployed pipeline ~1.6 test-accuracy points.
- Augmentation: aspect-preserving random crop (80–100% per side), horizontal flip, brightness/contrast jitter. Saturation/hue are **not** jittered, because saturation is a cue separating clear from overcast.
- Class imbalance: balanced `WeightedRandomSampler` on train plus sqrt-inverse-frequency loss weights. Val/test are never resampled.
- The best epoch is chosen by **validation macro-F1**. `--no-test-eval` keeps the test split untouched during any run used to compare settings. `--patience N` enables early stopping.
- Seeded (Python/NumPy/torch, sampler and worker RNGs, deterministic cuDNN). GPU float reductions can still vary slightly, so runs are reproducible to within noise rather than bit-exact.
- The checkpoint stores `model`, `classes`, `val_acc`, `val_macro_f1`, `best_epoch`, `preprocessing` and `train_config`. `WeatherClassifier` takes input size and normalization from the checkpoint, and warns if a checkpoint predates the shared preprocessing.

**Step 4 — evaluate on the held-out test set once:** `python outputs/weather_runs/final_eval.py` evaluates `weights/efficientnet_weather.pth` through both the training evaluation code and the real `WeatherClassifier.infer()`. It refuses to run a second time once `outputs/weather_runs/final_test_eval.json` exists.

#### Selected weather checkpoint and results

**How it was chosen.** Two dataset sizes were compared with two seeds each, on validation only:

| training data | val macro-F1 (seeds 42 / 43) | val acc | BDD val macro-F1 excl. fog | train acc at best epoch |
|---|---|---|---|---|
| ≤800 BDD per class | 0.786 / 0.774 | 0.781 | 0.785 | 0.973 |
| **≤3000 BDD per class** | **0.815 / 0.814** | **0.811** | **0.812** | 0.899 |

More data improved every class on validation (rain recall 0.69 → 0.76, overcast 0.79 → 0.84) and reduced overfitting. Validation macro-F1 peaks at epochs 5–6 and then declines, so selecting the best epoch matters. The shipped checkpoint is run `B_c3000-s42` (best epoch 6, val macro-F1 0.815), promoted unchanged to `weights/efficientnet_weather.pth` (SHA-256 `0bda562e…0c28`). The earlier checkpoint is kept read-only as `weights/efficientnet_weather.backup-20261029-df1bd218.pth`.

**Held-out test results** (evaluated once, n = 798; the deployed `WeatherClassifier.infer()` path, which agrees with the batched training evaluation on 797/798 images):

| | accuracy | macro-F1 |
|---|---|---|
| All test images | **0.823** | **0.833** |
| BDD100K dashcam only (n=754) | 0.818 | 0.820 excl. fog (0.739 incl. 4-image fog class) |
| DAWN only (fog, n=44) | 0.909 (fog recall) | — |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| clear | 0.878 | 0.767 | 0.819 | 150 |
| rain | 0.817 | 0.713 | 0.762 | 150 |
| fog | 0.976 | 0.854 | 0.911 | 48 (44 DAWN + 4 BDD) |
| snow | 0.852 | 0.847 | 0.849 | 150 |
| night | 0.820 | 0.880 | 0.849 | 150 |
| overcast | 0.734 | 0.900 | 0.808 | 150 |

```
true\pred  clear  rain  fog  snow  night  overcast
clear        115     2    0     9      3        21
rain           2   107    0     7     17        17
fog            0     4   41     1      1         1
snow           7     2    0   127      6         8
night          0    13    0     3    132         2
overcast       7     3    1     2      2       135
```

**Limitations (state these in any report):**
- **Dashcam fog performance is not established.** The fog score above is almost entirely DAWN (web photos of fog). On BDD100K dashcam fog the model got 1 of 4 test images and 0–1 of 9 validation images right. BDD's own `foggy` labels are unreliable: in a random sample of 20, only about 4 showed visible fog and about half were clear night scenes. Measuring dashcam fog needs a separately hand-verified dashcam fog set (e.g. ACDC fog).
- **Night-time rain and snow are hard by construction.** They are labelled rain/snow but look like the night class. Most rain errors are rain→night (17) and rain→overcast (17).
- **Clear vs overcast** is the other main confusion (21 clear→overcast), partly because `partly cloudy` is mapped to clear.
- Validation and test are small (150 per class; fog 50/48), so differences of about ±0.02 are within noise.
- The model still overfits somewhat (train ≈ 0.90 vs val ≈ 0.81 at the selected epoch).

### A note on the previous class-order bug
The original `train_weather_model()` trained against whatever index order `torchvision.datasets.ImageFolder` happened to assign (alphabetical: `clear, fog, night, overcast, rain, snow`), while `WeatherClassifier` at inference time zipped softmax outputs against `config.yaml`'s declared order (`clear, rain, fog, snow, night, overcast`). Those two orders differ, so every prediction's label would have been silently scrambled (e.g. the model's "index 1" meant `fog` during training but was read as `rain` at inference). This is now fixed by remapping to a single canonical order during training and saving that order into the checkpoint itself, so there is no code path left where the two can drift apart.

---

## Running

**Streamlit dashboard (recommended):**
```bash
streamlit run app.py
```

**Headless / OpenCV window:**
```bash
python run.py --synthetic --show                                  # synthetic demo
python run.py --source 0 --show                                    # webcam
python run.py --source road.mp4 --output outputs/videos/out.mp4    # video file (video-timeline timestamps)
python run.py --synthetic --show --overlay-depth
python run.py --source road.mp4 --depth-mode metric --strict      # metric depth (enables BEV/velocity/TTC/risk)
python run.py --source road.mp4 --max-frames 240 --record-run      # also record a structured run (below)
```
Press **`q`** to quit the OpenCV window. `--depth-mode` and `--strict` override `config.yaml` for that run only, and the overridden configuration is validated again.

**Logging.** The entry points (`run.py`, `app.py`, `carla_sim.py`, `download.py`) configure logging from the `logging:` section of `config.yaml`: console plus a rotating `logs/pipeline.log`. Importing a module never creates files. Scripts that do not configure logging print INFO messages to the console only.

This includes the frozen Task 5 evaluation scripts (`evaluate_kitti_velocity.py`, `evaluate_kitti_egomotion.py`). They print the same console logs as before but no longer create `logs/pipeline.log`; redirect their output (e.g. `| tee run.log`) to keep a copy. See `scripts/task5/README.md`.

### Recorded runs (structured per-frame logging)

`python run.py ... --record-run` (or the dashboard's **Record run** checkbox) writes `outputs/runs/<run_id>/`, under `paths.outputs_dir`. Recording is **off by default**, and without it nothing is written.

| File | Contents |
|---|---|
| `manifest.json` | <ul><li>Run: git commit, branch and changed files; the command; status (`recording`, `complete`, `interrupted` or `failed`).</li><li>Configuration: the resolved config, the config file's SHA-256, command-line overrides.</li><li>Models: checkpoint paths and SHA-256s; whether each model is the network or the heuristic fallback.</li><li>Environment: device and GPU; package versions.</li><li>Pipeline: depth mode, strict mode, camera intrinsics and calibration status, source and fps.</li><li>A note stating that outputs are monocular-camera estimates.</li></ul> |
| `frames.csv` | One row per frame: weather (smoothed and raw), detection and track counts, depth kind, BEV and valid-velocity counts, overall risk, minimum TTC, scene confidence, per-stage latencies, fps |
| `objects.csv` | One row per object per frame: box, class, confidence, track ID, BEV x/z, relative velocity, distance, TTC, risk level. The metric columns are empty in relative-depth mode. |
| `INCOMPLETE` | Present until the run closes as `complete` |

**How recording stays out of the way:**
- It only reads each finished `PipelineOutput`, after `process()` returns.
- An I/O error stops recording (status `failed`) but never interrupts inference.
- The manifest is written atomically.
- A run whose manifest still says `recording` was killed or crashed and is incomplete.
- `outputs/runs/` is git-ignored, because recordings of KITTI frames are KITTI-derived.

---

## Dashboard Contents

Camera feed with boxes/track IDs • detected-objects table (with distance/TTC/risk) • weather probability chart • depth colormap • per-module dynamic confidence • live adaptive sensor-weight chart • risk-level guide • FPS • object count • bird's-eye-view map • per-module latency chart.

---

## Configuration

Everything tunable lives in `config.yaml`: camera, model paths and thresholds, the weather tables, velocity, risk and logging. For example, the **weather → module-weight modifier table** behind the adaptive weighting:

```yaml
fusion:
  weather_modifiers:
    fog:
      camera_detection: 0.55   # lower = trust detection less in fog
      camera_depth: 0.45
```

### Validation

`config.yaml` is checked against `config_schema.py` when loaded. `PerceptionPipeline` checks it again, which catches command-line or in-memory overrides, before any model loads.

**Errors** stop with one message listing every problem and its key path. They cover:
- missing required keys and wrong types;
- out-of-range values: probabilities in [0, 1], positive intrinsics and sizes, `velocity.history_length >= 2`;
- unknown choices: `device.preference`, `depth_estimation.mode`, `depth_estimation.encoder`, `logging.level`;
- consistency: every weather class must have a `fusion.weather_modifiers` entry and a `risk.weather_multipliers` entry, and `risk.ttc_thresholds_s` must satisfy CRITICAL < WARNING < CAUTION.

For example:

```
config_schema.ConfigError: Invalid configuration file config.yaml (2 problem(s)):
  - device.preference: must be one of ['auto', 'cuda', 'mps', 'cpu'], got 'gpu'
  - risk.ttc_thresholds_s.WARNING: must be > CRITICAL (1.5), got 1.0
```

**Warnings** cover unknown keys, which are probably typos, and weather entries that no class uses. Validation never changes the configuration. Optional keys fall back to the defaults the code always used.

| Key | Default if omitted | Meaning |
|---|---|---|
| `pipeline.strict` | `false` | `true` = a model that fails to load or run raises instead of using a heuristic fallback |
| `pipeline.run_weather_classification` | `true` | `false` = no classifier is loaded. Weather is reported as `unknown` (confidence 0), so the fusion modifiers and risk multipliers are neutral (×1.0). |
| `depth_estimation.mode` | `relative` | `metric` enables BEV, velocity, TTC and risk |
| `camera.calibrated` | `false` | `false` labels metric outputs as uncalibrated estimates |
| `logging.*` | INFO, console + `logs/pipeline.log` (5 MiB × 3) | Applied by the entry points |
| `paths.outputs_dir` | `outputs` | Recorded runs go to `<outputs_dir>/runs/` |

**Reserved keys** are accepted and documented but not read by any code yet:
- `camera.mount_height_m`, `camera.pitch_deg`, `camera.extrinsics`: camera mounting geometry, for later work;
- `dashboard.refresh_interval_sec`: the dashboard paces itself with `pipeline.target_fps`;
- `paths.weights_dir`, `paths.data_dir`, `paths.logs_dir`: checkpoint paths are set per model, and the log file by `logging.log_file`.

---

## Testing

```bash
pytest tests.py tests_depth.py tests_kitti_tracking.py tests_velocity_experimental.py tests_ego_motion.py tests_foundation.py tests_regression.py -q
```
By default the tests run on CPU with synthetic frames. Set `TEST_DEVICE=cuda` to run the pipeline tests on the GPU.

| Suite | Covers |
|---|---|
| `tests.py` | Detector contracts, fusion ranges and weather ordering, ByteTrack, the end-to-end pipeline, velocity/TTC/risk |
| `tests_depth.py` | Depth units, masks, calibration, the KITTI depth loader |
| `tests_kitti_tracking.py`, `tests_velocity_experimental.py`, `tests_ego_motion.py` | Task 5 evaluation code |
| `tests_foundation.py` | <ul><li>Configuration validation.</li><li>No import-time logging side effects.</li><li>The run recorder, including failure and interruption handling.</li><li>`run_weather_classification`.</li><li>`setup.py` module coverage and the environment lock.</li></ul> |
| `tests_regression.py` | <ul><li>Exact golden comparison of `road.mp4` outputs (relative and metric).</li><li>KITTI 0000, frames 0–49, against the frozen Task 5 cache.</li><li>Determinism after `PerceptionPipeline.reset()`, and with models shared across threads.</li><li>Independent pipelines and trackers.</li><li>Model backend status.</li><li>`run.py --synthetic`.</li><li>Two Streamlit sessions.</li></ul> See `scripts/regression/README.md`. |

Tests that need the KITTI data skip when it is absent. One test also needs `KITTI_VELOCITY_CACHE` set to a Task 5 cache directory. The golden checks skip, with the reason, when `road.mp4`, the goldens in `outputs/regression/` or a GPU are missing, or when the environment fingerprint differs from the one the goldens were recorded in.

### Final Testing Checklist (using your `road.mp4`)

Work through these in order; each builds on the previous one.

1. **Baseline regression check.** Run the full test command above; every test should pass after any change.
2. **Confirm weight files present.** `weights/yolo11x.pt` and `weights/depth_anything_v2_vits.pth` should exist; `weights/efficientnet_weather.pth` should exist (the selected checkpoint above). If it's missing, the pipeline logs a warning and uses the colour heuristic, which is not a trained model.
3. **Train + evaluate the weather classifier** per the walkthrough above; note the printed overall accuracy and per-class confusion matrix for your report.
4. **Run on `road.mp4` headless, with output saved:**
   ```bash
   python run.py --source road.mp4 --show --output outputs/videos/road_annotated.mp4
   ```
   Manually verify frame-by-frame:
   - Detection boxes track real vehicles/pedestrians in the footage (you already confirmed 4-5 objects detect correctly).
   - Weather label displayed matches what you can see in the footage (if `road.mp4` is all one condition, at minimum confirm it doesn't flicker wildly between unrelated classes).
   - Depth colormap is spatially sensible (nearer objects/road should look visually distinct from the horizon).
   - BEV panel places detected objects at roughly plausible relative positions.
   - Distance/TTC/risk in the detected-objects table and HUD update sensibly as tracked objects move.
5. **Run the Streamlit dashboard against `road.mp4`.** In the sidebar choose **Input source → Video file** and enter the video's path, e.g. `road.mp4` in the repository root. The field is pre-filled from `camera.source` when that is a file path; otherwise it shows `data/sample_videos/road.mp4`, which does not exist in this checkout. Video files use video-timeline timestamps, exactly like `run.py`, so velocity/TTC/risk are correct however slow each frame is. Confirm the adaptive sensor-weight bar chart, FPS, and latency panels update live and stay CPU-reasonable.
6. **Contrast test (optional but strengthens your report):** if you can find or record a second short clip in a visibly different condition (rain/fog/dusk), run it through the same pipeline and confirm the "Adaptive Sensor Weights" chart visibly shifts lower for `camera_depth` compared to the clear clip, and that the risk-level guide's TTC thresholds visibly tighten — this is the actual thesis claim of the project and worth screenshotting for both conditions side-by-side.
7. **Document limitations explicitly in your report:** depth mode (relative depth has no unit; metric depth is validated only on KITTI's favourable camera domain), and the weather classifier's per-class accuracy including the naturally-small fog class — see "Model Status" above for the full honest breakdown.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Could not open video source | Pass `--synthetic`, or check `config.yaml → camera.source` |
| `ConfigError` at start-up | The message lists every problem with its key path; fix `config.yaml` (see "Configuration → Validation") |
| `ModuleNotFoundError` for a project module outside the repository folder | Run `pip install -e .` (or set `PYTHONPATH` to the repository) |
| Slow FPS on CPU | Lower `detection.input_size`, disable stages in `config.yaml → pipeline`, or install CUDA torch |
| Import errors for `ultralytics` / `depth_anything_v2` | Re-run `pip install -r requirements.txt`; fallbacks keep the pipeline running meanwhile |
| Open3D import fails | BEV mapping still works via numpy projection; only `.ply` export is disabled |
| nuScenes/STF access denied | Both require accepting a license on the provider's site first |
| CARLA connection refused | Make sure the CARLA server is running and matches `--host`/`--port` |

---

## Future Work

- Additional sensors (radar, LiDAR, IMU). Today `fusion.py` only weighs the two camera-derived modules.
- Learned (rather than hand-tuned) confidence calibration
- Temporal BEV occupancy grid accumulation
- Multi-camera nuScenes ingestion + full-surround BEV
- ONNX/TensorRT export for embedded deployment (e.g. Jetson)

---

## License

Provided for academic/capstone educational purposes. Third-party pretrained models (YOLO11x, Depth Anything V2, EfficientNet) and datasets (nuScenes, Seeing Through Fog, CARLA) remain subject to their own respective licenses.
