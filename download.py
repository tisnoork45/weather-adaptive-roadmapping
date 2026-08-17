"""Download pretrained model weights and set up datasets, in one place.

Usage:
    python download.py models                       # download all model weights
    python download.py models --only yolo depth
    python download.py dataset nuscenes-mini
    python download.py dataset seeing-through-fog
    python download.py dataset carla
    python download.py dataset --all                # print instructions for all three
    python download.py dataset nuscenes-mini --nuscenes-url "<signed-url>"
    python download.py dataset --verify-nuscenes
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict

import requests
from tqdm import tqdm

from utils import get_logger, load_config

logger = get_logger(__name__)

DATA_ROOT = Path("data")


# =============================================================================
# Model weight downloads
# =============================================================================


def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Already exists, skipping: %s", destination)
        return True

    logger.info("Downloading %s -> %s", url, destination)
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(dest_path, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                                     desc=dest_path.name) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
                        progress.update(len(chunk))
        logger.info("Downloaded: %s", destination)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download %s (%s). A classical/CV fallback will be used instead.", url, exc)
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return False


def download_yolo(cfg) -> bool:
    dest = cfg.detection.weights_path
    if Path(dest).exists():
        logger.info("Already exists, skipping: %s", dest)
        return True
    try:
        from ultralytics import YOLO
        import shutil

        logger.info("Using ultralytics auto-download for YOLO11x...")
        model = YOLO("yolo11x.pt")
        source_path = getattr(model, "ckpt_path", None) or "yolo11x.pt"
        if Path(source_path).exists():
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, dest)
            logger.info("Copied YOLO11x weights to %s", dest)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ultralytics auto-download failed (%s); trying direct URL.", exc)

    return download_file(cfg.detection.download_url, dest)


def _explain_lane_unavailable(cfg) -> bool:
    """No-op: UFLD-v2 is intentionally not auto-downloaded. See README.md >
    'Lane Detection: Design Decision' for the full rationale. In short: the
    official checkpoint is trained for the upstream `parsingNet` architecture
    (not this project's simplified `UFLDv2Net`), so even a successful
    download would not be state-dict-compatible, and the official weights
    are not published behind a stable scriptable URL anyway. This project
    uses a classical Canny+Hough-transform lane detector with temporal
    smoothing as its documented primary lane-detection method instead.
    """
    logger.warning(
        "Skipping 'lane': UFLD-v2 pretrained weights are intentionally NOT "
        "auto-downloaded by this project (the official checkpoint requires "
        "the exact upstream 'parsingNet' architecture, not the simplified "
        "network sketch included here, and isn't hosted at a stable direct-"
        "download URL). This project uses a classical Canny+Hough-transform "
        "lane detector by design. See README.md > 'Lane Detection: Design "
        "Decision' for details."
    )
    return True


def _explain_weather_unavailable(cfg) -> bool:
    """No-op: a generic ImageNet checkpoint is intentionally not auto-downloaded.

    A prior version of this script downloaded lukemelas/EfficientNet-PyTorch's
    checkpoint (key names like `_conv_stem`, `_fc`) into this project's
    torchvision-based model (key names like `features.0`, `classifier.1`).
    Because NO keys overlap between those two naming conventions,
    `load_state_dict(strict=False)` silently loaded ZERO weights -- the model
    was randomly initialized while appearing to have loaded successfully.
    To avoid repeating that, this project fine-tunes directly from
    torchvision's own (correctly key-matched) ImageNet weights inside
    train_weather_model() -- there is nothing to separately download here.
    """
    logger.warning(
        "Skipping 'weather': no generic/pretrained checkpoint is downloaded "
        "for the weather classifier. torchvision's own ImageNet-B0 weights "
        "are fetched automatically and correctly inside train_weather_model() "
        "when you actually train. Run: python prepare_weather_dataset.py ... "
        "then: python models.py --data-dir data/weather_dataset --epochs 15"
    )
    return True


def download_models(cfg, only=None) -> None:
    downloaders: Dict[str, Callable] = {
        "yolo": download_yolo,
        "depth": lambda c: download_file(c.depth_estimation.download_url, c.depth_estimation.weights_path),
        "lane": _explain_lane_unavailable,
        "weather": _explain_weather_unavailable,
    }
    targets = only if only else list(downloaders.keys())

    results = {}
    for name in targets:
        logger.info("=" * 60)
        logger.info("Downloading model: %s", name)
        results[name] = downloaders[name](cfg)

    logger.info("=" * 60)
    logger.info("Download summary:")
    for name, ok in results.items():
        logger.info("  %-10s %s", name, "OK" if ok else "FAILED (fallback will be used)")


# =============================================================================
# Dataset setup
# =============================================================================


def _download_and_extract(url: str, dest_dir: Path, archive_name: str = "download.zip") -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / archive_name
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(archive_path, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                                        desc=archive_name) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
                        progress.update(len(chunk))
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed (%s).", exc)
        return False

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        archive_path.unlink()
        logger.info("Extracted dataset into %s", dest_dir)
        return True
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip archive: %s", archive_path)
        return False


def setup_nuscenes_mini(signed_url: str = None) -> None:
    dest = DATA_ROOT / "nuscenes"
    dest.mkdir(parents=True, exist_ok=True)

    if signed_url:
        logger.info("Attempting download of nuScenes-mini using provided signed URL...")
        if _download_and_extract(signed_url, dest, archive_name="v1.0-mini.zip"):
            verify_nuscenes_devkit()
            return
        logger.warning("Automatic download failed; falling back to manual instructions.")

    logger.info(
        "\n=========================================================\n"
        " MANUAL DOWNLOAD REQUIRED: nuScenes\n"
        "=========================================================\n"
        " 1. Create an account at https://www.nuscenes.org/sign-up\n"
        " 2. Go to https://www.nuscenes.org/nuscenes#download\n"
        " 3. Accept the license and download 'Mini' (or 'Trainval') split.\n"
        " 4. Extract so the folder layout looks like:\n"
        f"      {dest}/v1.0-mini/, {dest}/samples/, {dest}/sweeps/, {dest}/maps/\n"
        " 5. pip install nuscenes-devkit\n"
        " 6. Verify with: python download.py dataset --verify-nuscenes\n"
        "========================================================="
    )


def verify_nuscenes_devkit() -> bool:
    try:
        from nuscenes.nuscenes import NuScenes  # type: ignore

        nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT / "nuscenes"), verbose=True)
        logger.info("nuScenes-mini loaded successfully: %d scenes found.", len(nusc.scene))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not verify nuScenes dataset (%s).", exc)
        return False


def setup_seeing_through_fog() -> None:
    dest = DATA_ROOT / "seeing_through_fog"
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(
        "\n=========================================================\n"
        " MANUAL DOWNLOAD REQUIRED: Seeing Through Fog (STF / DENSE)\n"
        "=========================================================\n"
        " 1. Visit: https://light.princeton.edu/publication/seeing-through-fog/\n"
        " 2. Submit the dataset access request form.\n"
        " 3. Once granted, download the camera image subset.\n"
        f" 4. Extract into: {dest}/\n"
        " 5. Re-encode an image sequence into video for this project with:\n"
        "      ffmpeg -framerate 10 -pattern_type glob -i '*.png' \\\n"
        "             -c:v libx264 data/sample_videos/stf_demo.mp4\n"
        "========================================================="
    )


def setup_carla() -> None:
    dest = DATA_ROOT / "carla"
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(
        "\n=========================================================\n"
        " SIMULATOR SETUP: CARLA\n"
        "=========================================================\n"
        " 1. Download a prebuilt release: https://github.com/carla-simulator/carla/releases\n"
        f" 2. Extract to: {dest}/CARLA_<version>/\n"
        " 3. pip install carla==<version matching your server>\n"
        " 4. Launch the server:\n"
        "      Windows: CarlaUE4.exe -quality-level=Low\n"
        "      Linux/Mac: ./CarlaUE4.sh -quality-level=Low\n"
        " 5. Stream frames into this pipeline with:\n"
        "      python carla_sim.py --host 127.0.0.1 --port 2000\n"
        "========================================================="
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights and/or set up datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="Download pretrained model weights.")
    models_parser.add_argument("--only", nargs="*", choices=["yolo", "depth", "lane", "weather"])

    dataset_parser = subparsers.add_parser("dataset", help="Set up a dataset or simulator.")
    dataset_parser.add_argument("name", nargs="?", choices=["nuscenes-mini", "seeing-through-fog", "carla"])
    dataset_parser.add_argument("--nuscenes-url", type=str, default=None)
    dataset_parser.add_argument("--verify-nuscenes", action="store_true")
    dataset_parser.add_argument("--all", action="store_true")

    args = parser.parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.command == "models":
        cfg = load_config("config.yaml")
        download_models(cfg, only=args.only)
        return

    if args.command == "dataset":
        if args.verify_nuscenes:
            verify_nuscenes_devkit()
            return
        if args.all:
            setup_nuscenes_mini()
            setup_seeing_through_fog()
            setup_carla()
            return
        if args.name == "nuscenes-mini":
            setup_nuscenes_mini(signed_url=args.nuscenes_url)
        elif args.name == "seeing-through-fog":
            setup_seeing_through_fog()
        elif args.name == "carla":
            setup_carla()
        else:
            dataset_parser.print_help()


if __name__ == "__main__":
    main()
