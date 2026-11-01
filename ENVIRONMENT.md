# Software environment

There are two ways to set up the software, and they make different promises.

| | Standard installation | NGC reference environment |
|---|---|---|
| What it is | Any machine with Python 3.9–3.12; packages from PyPI within the ranges in `requirements.txt` | The NVIDIA NGC PyTorch container in which the software was developed and every result was produced |
| Versions | Whatever pip resolves inside the ranges | Exactly those in `requirements-lock.txt` |
| PyTorch | A public build (CPU or CUDA) that you install | The container's own build, `torch 2.4.0a0+f70bd71a48.nv24.6` |
| Reproducibility | Same code and models; numbers can differ slightly between builds and devices | Repeated runs gave bit-identical outputs (checked on road.mp4 and on KITTI against the Task 5 cache) |

## Standard installation

1. **Install PyTorch first**, picking the build for your hardware from https://pytorch.org/get-started/locally/. For example:

   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   `requirements.txt` only needs `torch>=2.1` and `torchvision>=0.16`.
2. **Install the rest and the project** (editable):

   ```
   pip install -r requirements.txt
   pip install -e .
   ```

   `pip install -e .` installs every module listed in `setup.py` (`PY_MODULES`) plus the vendored `depth_anything_v2/` package.
3. **Model weights** are not downloaded automatically. See the README sections "Downloading Model Weights" and "Training the Weather Classifier".
4. **Check the environment:**

   ```
   python scripts/env/check_env.py
   ```

   This reports any package missing or outside its `requirements.txt` range (exit status 1). It also lists where the installed versions differ from the reference lock; those differences are expected outside the container.

To get as close as possible to the reference versions on a standard machine:
- install a public PyTorch 2.4.x and torchvision 0.19.x build, then
- run `pip install -r requirements-lock.txt`.

The `# CONTAINER BUILD` lines are comments and are skipped. The result is close to the reference environment but not identical, because the PyTorch, CUDA and cuDNN builds differ.

## NGC reference environment

| Component | Version |
|---|---|
| Container image | `nvcr.io/nvidia/pytorch:24.06-py3` (`NVIDIA_PYTORCH_VERSION=24.06`, build 96418707) |
| Python | 3.10.12 (Linux x86_64) |
| CUDA / cuDNN / TensorRT | 12.5.0 / 9.1.0 / 10.1.0 |
| PyTorch | `2.4.0a0+f70bd71a48.nv24.6` (NVIDIA build, from a wheel inside the image) |
| torchvision | `0.19.0a0` (built from source inside the image) |
| GPU used for results | NVIDIA H100 80GB HBM3 (MIG 3g.40gb), driver 535.161.08 |

**Container builds:**
- **PyTorch and torchvision** are NVIDIA pre-release builds. They are not on PyPI and **cannot be reproduced outside this image**. `requirements-lock.txt` therefore lists them as `# CONTAINER BUILD` comments, not pins.
  - The nearest public releases are torch 2.4.0 and torchvision 0.19.0. They are close but not the same builds.
- **Other wheels shipped by the image** (pandas, scipy, pyarrow and a few small libraries, installed from `/rapids` wheel files) are pinned to the same version numbers that exist on PyPI. The lock marks each one. Byte-identity with the PyPI wheels has not been verified.

**Inside the container:** `pip install -r requirements-lock.txt` should report every pin as already satisfied. Then run `pip install -e .` (or `PYTHONPATH=.`) and `python scripts/env/check_env.py`.

## The lock file

`requirements-lock.txt` is generated from the installed packages, never edited by hand:

```
python scripts/env/generate_lock.py
```

**What it contains:** the exact installed version of every package in `requirements.txt` and of everything they depend on, read from installed metadata. Extras are not followed.

**What the generator does not do:** install, upgrade or download anything. When dependencies change, it is regenerated from the reference container.

`requirements.txt` stays the source of truth for what the software needs. The lock records what it was run with.

## Data and outputs kept out of Git

| What | Where | Why |
|---|---|---|
| KITTI datasets | `/workspace/datasets/kitti/...` | CC BY-NC-SA 3.0 |
| Evaluation caches and results | `/workspace/experiments/` and `outputs/experiments/` | Derived from KITTI |
| Recorded runs | `outputs/runs/` | See below |

**Recorded runs** come from `python run.py --record-run` or the dashboard's "Record run" checkbox. Each run's `manifest.json` records:
- the git commit;
- the configuration and its SHA-256;
- the checkpoint SHA-256s;
- the package versions and the device.

Two runs can be compared exactly. Recorded runs are git-ignored because they can contain outputs derived from licensed data.
