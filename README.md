# Weather Adaptive Road Mapping for Autonomous Vehicles

A weather-adaptive multi-sensor perception system for autonomous vehicles, in pure Python. Combines object detection, monocular depth estimation, weather classification, dynamic confidence estimation, adaptive sensor fusion, multi-object tracking, relative-velocity/time-to-collision risk assessment, and bird's-eye-view (BEV) mapping into one real-time pipeline with a live Streamlit dashboard.

> **Final Year Engineering Capstone Project** — no ROS, no Docker, runs entirely inside VS Code.

This is the **consolidated, low-file-count edition** of the project: the same full functionality as a typical multi-package implementation, but organized into a handful of readable files grouped by concern instead of one class per file.

---

## Repository Structure

```
weather-adaptive-road-mapping/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config.yaml            # ALL configuration in one file (camera, models, fusion, dashboard)
│
├── run.py                 # Headless / OpenCV-window entry point
├── app.py                 # Streamlit real-time dashboard
├── download.py            # Downloads model weights + sets up datasets (nuScenes/STF/CARLA)
├── carla_sim.py           # CARLA simulator bridge (spawn vehicle, cycle weather presets)
│
├── utils.py                # Config loading, logging, metrics, visualization, video IO, device selection
├── models.py                # YOLO11x detector, Depth Anything V2, EfficientNet weather classifier + trainer
├── tracking.py               # Self-contained ByteTrack (Kalman filter + two-stage Hungarian matching)
├── fusion.py                  # Dynamic confidence estimation + weather-adaptive sensor fusion
├── mapping.py                  # Open3D point cloud + bird's-eye-view rasterization
├── pipeline.py                  # Orchestrates all modules per frame + Streamlit render helpers
│
├── tests.py                # Full pytest suite (detection, fusion, tracking, pipeline)
│
├── weights/                # Downloaded model checkpoints (gitignored)
├── data/                    # Datasets (gitignored — see Dataset section)
├── logs/                     # Rotating log files
└── outputs/                   # Saved videos / frames / BEV point clouds
```

**13 Python files total** (down from a one-class-per-file layout), each still fully typed, documented, and under ~500 lines, with the same clean separation of concerns:

| File | Responsibility |
|---|---|
| `utils.py` | Cross-cutting infrastructure: config, logging, metrics, drawing, video IO, device selection |
| `models.py` | All three perception *models* (detection, depth, weather) — each with a real network + graceful fallback |
| `tracking.py` | The tracking *algorithm* — kept separate because ByteTrack is a self-contained, non-trivial component |
| `fusion.py` | The *adaptive* logic that makes this project "weather-adaptive" — confidence scoring + dynamic weighting |
| `mapping.py` | The *spatial* output — BEV point cloud + top-down map |
| `pipeline.py` | *Orchestration* — wires every module together per frame, plus dashboard rendering |

---

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

---

## Model Status — What's Actually Trained vs. Classical CV

Being precise about this matters for a capstone report. Here is the honest state of every component:

| Component | Status | Notes |
|---|---|---|
| **Object Detection** (YOLO11x) | ✅ Genuine pretrained deep model | Official Ultralytics COCO-pretrained weights, loaded and run as-is. |
| **Depth Estimation** (Depth Anything V2) | ✅ Genuine pretrained deep model | Official ViT-S weights. Output is **relative** depth linearly rescaled into an assumed max range for BEV visualization — it is *not* metrically calibrated (no LiDAR ground truth used), so treat BEV distances as approximate, not measured. |
| **Weather Classification** (EfficientNet-B0) | ✅ Genuine fine-tuned deep model | Fine-tuned on BDD100K + DAWN (see "Selected weather checkpoint and results" below): held-out test accuracy 0.823, macro-F1 0.833. **Dashcam fog performance is not established.** No generic weather checkpoint is downloaded (see "A note on the previous broken download" below). |
| **Object Tracking** (ByteTrack) | ✅ Genuine, correctly implemented algorithm | ByteTrack is a Kalman-filter + Hungarian-matching *algorithm*, not a neural network — it was never meant to have separate pretrained weights. Fully functional as shipped, no training required. |
| **Adaptive Sensor Fusion / Confidence Estimation** | ✅ Genuine, by design rule-based | This is the project's engineered "adaptive" logic layer (weather-conditioned weighting + EMA smoothing) — intentionally hand-tuned/heuristic, not a trained model. This is the correct way to build it, not a shortcut. |
| **BEV Mapping** (Open3D) | ✅ Genuine geometric computation | Deterministic pinhole back-projection, not ML. |
| **Relative Velocity / TTC / Risk** | ✅ Genuine geometric + rule-based computation | Finite-difference velocity from BEV track history → time-to-collision → weather-multiplied risk thresholds (`relative_velocity.py`, `ttc_calculator.py`, `risk_engine.py`). |

### A note on the previous broken weather-classifier download
An earlier version of `download.py` downloaded a checkpoint from a *different* EfficientNet-B0 implementation (lukemelas/EfficientNet-PyTorch, key names like `_conv_stem`, `_fc`) into this project's *torchvision*-based model (key names like `features.0`, `classifier.1`). Since **zero** keys overlapped between the two naming conventions, `load_state_dict(strict=False)` silently loaded nothing — the "pretrained" classifier was actually running on random initialization while appearing to load successfully. This has been fixed: `download.py` no longer downloads a generic checkpoint at all (torchvision's own correctly-matched ImageNet weights are fetched automatically inside training instead), and `WeatherClassifier` now refuses to load any checkpoint that isn't in this project's own verified format, falling back to the heuristic classifier instead of silently running on garbage weights.

### Lane Detection has been removed
An earlier revision of this project included a classical Canny-edge + Hough-transform lane detector (`LaneDetector` in `models.py`), plus a never-loaded architectural sketch (`UFLDv2Net`) for a future learned lane model. Both have been fully removed: there is no lane detector, no lane overlay, no lane confidence signal, and no lane-related configuration anywhere in the current pipeline, BEV projection, dashboards, or test suite. If lane detection is reintroduced in the future, it should be designed fresh rather than restored from history.

---

## Installation

### Prerequisites
Python 3.9–3.12, Git, (optional) NVIDIA GPU + CUDA, VS Code with the Python extension.

### 1. Clone & create a virtual environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/<your-username>/weather-adaptive-road-mapping.git
cd weather-adaptive-road-mapping
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
git clone https://github.com/<your-username>/weather-adaptive-road-mapping.git
cd weather-adaptive-road-mapping
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

**CPU-only (all platforms):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU (NVIDIA CUDA 12.1, Windows/Linux)** — install the CUDA build first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Apple Silicon (macOS)** — automatically uses the `mps` backend via `utils.get_device()`:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 3. (Optional) editable install
```bash
pip install -e .
```

---

## Setup (VS Code)

```bash
code .
```
Select the `venv` interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*). Pre-configured run/debug targets are in `.vscode/launch.json`: headless demo, webcam, Streamlit dashboard, model download, weather-classifier training, and the full test suite.

All required folders (`weights/`, `data/`, `logs/`, `outputs/`) already exist with `.gitkeep` placeholders.

---

## Dataset

| Source | Role | Access |
|---|---|---|
| **nuScenes** | Primary — real multi-camera driving dataset | Free registration at [nuscenes.org](https://www.nuscenes.org) |
| **Seeing Through Fog (STF)** | Secondary — adverse-weather camera/lidar/radar dataset | Access request required |
| **CARLA** | Simulator — generates synthetic, weather-controllable footage | Free download, no registration |

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
python run.py --source data/sample_videos/demo.mp4 --output outputs/videos/out.mp4
python run.py --synthetic --show --overlay-depth
```
Press **`q`** to quit the OpenCV window.

---

## Dashboard Contents

Camera feed with boxes/track IDs • detected-objects table (with distance/TTC/risk) • weather probability chart • depth colormap • per-module dynamic confidence • live adaptive sensor-weight chart • risk-level guide • FPS • object count • bird's-eye-view map • per-module latency chart.

---

## Configuration

Everything tunable lives in `config.yaml` — camera, model paths/thresholds, and crucially the **weather → sensor-weight modifier table** that drives the adaptive behavior:

```yaml
fusion:
  weather_modifiers:
    fog:
      camera_detection: 0.55   # lower = trust detection less in fog
      camera_depth: 0.45
```

---

## Testing

```bash
pytest tests.py -v
```
Runs entirely on CPU with synthetic frames by default; `TEST_DEVICE=cuda pytest tests.py -v` runs the pipeline tests on the GPU instead. Covers: detector interface contracts, confidence/fusion ranges and weather-conditioned ordering (fog ≤ clear), ByteTrack ID persistence + low-confidence recovery, and full end-to-end pipeline execution.

### Final Testing Checklist (using your `road.mp4`)

Work through these in order; each builds on the previous one.

1. **Baseline regression check.** `pytest tests.py -v` → should still show every test passing after any changes (the fallback/synthetic-frame paths are unaffected by the weather-classifier fixes).
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
5. **Run the Streamlit dashboard against `road.mp4`:** in the sidebar choose **Input source → Video file** and enter `data/sample_videos/road.mp4` (pre-filled when `camera.source` in `config.yaml` is a file path). Video files use video-timeline timestamps, exactly like `run.py`, so velocity/TTC/risk are correct however slow each frame is. Confirm the adaptive sensor-weight bar chart, FPS, and latency panels update live and stay CPU-reasonable.
6. **Contrast test (optional but strengthens your report):** if you can find or record a second short clip in a visibly different condition (rain/fog/dusk), run it through the same pipeline and confirm the "Adaptive Sensor Weights" chart visibly shifts lower for `camera_depth` compared to the clear clip, and that the risk-level guide's TTC thresholds visibly tighten — this is the actual thesis claim of the project and worth screenshotting for both conditions side-by-side.
7. **Document limitations explicitly in your report:** relative (not metric) depth, and the weather classifier's per-class accuracy including the naturally-small fog class — see "Model Status" above for the full honest breakdown.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Could not open video source | Pass `--synthetic`, or check `config.yaml → camera.source` |
| Slow FPS on CPU | Lower `detection.input_size`, disable stages in `config.yaml → pipeline`, or install CUDA torch |
| Import errors for `ultralytics` / `depth_anything_v2` | Re-run `pip install -r requirements.txt`; fallbacks keep the pipeline running meanwhile |
| Open3D import fails | BEV mapping still works via numpy projection; only `.ply` export is disabled |
| nuScenes/STF access denied | Both require accepting a license on the provider's site first |
| CARLA connection refused | Make sure the CARLA server is running and matches `--host`/`--port` |

---

## Future Work

- Radar/LiDAR fusion as a fourth modality in `fusion.py`
- Learned (rather than hand-tuned) confidence calibration
- Temporal BEV occupancy grid accumulation
- Multi-camera nuScenes ingestion + full-surround BEV
- ONNX/TensorRT export for embedded deployment (e.g. Jetson)

---

## License

Provided for academic/capstone educational purposes. Third-party pretrained models (YOLO11x, Depth Anything V2, EfficientNet) and datasets (nuScenes, Seeing Through Fog, CARLA) remain subject to their own respective licenses.
