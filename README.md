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
| **Weather Classification** (EfficientNet-B0) | ✅ Genuine, once you fine-tune it | See the training walkthrough below. Ships with **no** pretrained weather checkpoint by design (see "A note on the previous broken download" below) — you must train one. |
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

**Step 2 — build the class-organized dataset:**
```powershell
# Preview the class distribution first -- no files copied yet.
python prepare_weather_dataset.py `
    --images-train "bdd100k_download\bdd100k\bdd100k\images\100k\train" `
    --images-val   "bdd100k_download\bdd100k\bdd100k\images\100k\val" `
    --labels-train "bdd100k_download\bdd100k\bdd100k\labels" `
    --labels-val   "bdd100k_download\bdd100k\bdd100k\labels" `
    --dry-run

# Looks reasonable? Now actually copy the files:
python prepare_weather_dataset.py `
    --images-train "bdd100k_download\bdd100k\bdd100k\images\100k\train" `
    --images-val   "bdd100k_download\bdd100k\bdd100k\images\100k\val" `
    --labels-train "bdd100k_download\bdd100k\bdd100k\labels" `
    --labels-val   "bdd100k_download\bdd100k\bdd100k\labels" `
    --output data/weather_dataset --max-per-class 800
```
(The backtick `` ` `` is PowerShell's line-continuation character — make sure there's a trailing space before it on each line, or just put the whole command on one line.)

This writes `data/weather_dataset/{train,val,test}/<class>/*.jpg` with the six canonical classes, and automatically carves a held-out `test/` split out of the val pool (never used for checkpoint selection, so the final reported accuracy is honest).

**Known dataset limitation, stated upfront:** `foggy` images are naturally rare in BDD100K (on the order of a few hundred across the whole dataset). The prep script will warn you about this in its class-distribution printout. Report this limitation explicitly rather than hiding it — e.g. by showing the confusion matrix, where a weak fog row is expected and explainable. If you want to strengthen the fog/snow classes, the [DAWN dataset](https://www.kaggle.com/datasets/shuvoalok/dawn-dataset) (1000 real driving images across fog/rain/snow/sandstorm, freely downloadable from Kaggle, no signup beyond a Kaggle account) can supplement those folders manually — just drop extra images into `data/weather_dataset/train/fog/` etc. before training.

**Step 3 — train (CPU-friendly):**
```powershell
python models.py --data-dir data/weather_dataset --epochs 15 --freeze-backbone
```
`--freeze-backbone` trains only the classifier head + last block, which is dramatically faster on a CPU-only laptop at a modest accuracy cost — a reasonable tradeoff to state in your report. Drop it for a full fine-tune if you have time/GPU.

This will:
- Fine-tune from **torchvision's own** ImageNet-B0 weights (correctly key-matched, auto-downloaded).
- Fix the class-index mapping so the model is always trained against the canonical order `["clear","rain","fog","snow","night","overcast"]`, regardless of how `ImageFolder` alphabetizes the subfolders internally (this was a real bug in the original version — see below).
- Save `weights/efficientnet_weather.pth` as a dict: `{"model": state_dict, "classes": [...], "val_acc": ...}` — `WeatherClassifier` reads the `"classes"` list back out at inference time, so the label order can never silently mismatch again.
- Print a full **per-class accuracy table + confusion matrix** on the validation set, and again on the held-out test set if one exists — this is what you should screenshot/report, not just a single "accuracy" number.

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
Runs entirely on CPU with synthetic frames: detector interface contracts, confidence/fusion ranges and weather-conditioned ordering (fog ≤ clear), ByteTrack ID persistence + low-confidence recovery, and full end-to-end pipeline execution.

### Final Testing Checklist (using your `road.mp4`)

Work through these in order; each builds on the previous one.

1. **Baseline regression check.** `pytest tests.py -v` → should still show every test passing after any changes (the fallback/synthetic-frame paths are unaffected by the weather-classifier fixes).
2. **Confirm weight files present.** `weights/yolo11x.pt` and `weights/depth_anything_v2_vits.pth` should exist; `weights/efficientnet_weather.pth` should exist **only after** you've run the training walkthrough above (its absence is expected and fine — the heuristic fallback handles it).
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
5. **Run the Streamlit dashboard against `road.mp4`:** set `camera.source: "road.mp4"` in `config.yaml` (or point the dashboard's video source), uncheck "Use synthetic demo stream", and confirm the adaptive sensor-weight bar chart, FPS, and latency panels update live and stay CPU-reasonable.
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
