"""Prepare a weather-classification dataset from BDD100K for train_weather_model().

BDD100K (Berkeley DeepDrive) is a large-scale driving dataset whose images
are labeled with `attributes.weather` and `attributes.timeofday` fields --
exactly the information needed to build our six target classes
(clear, rain, fog, snow, night, overcast), sourced from real dashcam
footage rather than generic/non-driving photos.

Official BDD100K schema (https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md):
    attributes.weather:   "rainy" | "snowy" | "clear" | "overcast" |
                           "partly cloudy" | "foggy" | "undefined"
    attributes.timeofday: "daytime" | "night" | "dawn/dusk" | "undefined"

===========================================================================
STEP 1: Get BDD100K via the Kaggle mirror (avoids the official
dl.cv.ethz.ch / bdd100k.com sites, which some networks/browsers reject with
a certificate error). This mirror (`solesensei/solesensei_bdd100k`) hosts
the exact same images + label JSON schema, just served from kaggle.com.

IMPORTANT: `kaggle datasets files -d solesensei/solesensei_bdd100k` lists
individual images as separate paginated entries (tens of thousands) --
that pagination is JUST Kaggle's browsing index. The actual download
command bundles everything server-side into ONE zip regardless:

    kaggle datasets download -d solesensei/solesensei_bdd100k

You never need to page through the file listing to download the data.

    1. Create a free Kaggle account at https://www.kaggle.com if you don't
       have one, then get an API token: Settings -> API -> "Create New
       Token" (downloads a `kaggle.json` file).
    2. `pip install kaggle`, then place `kaggle.json` at
       `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`
       (Mac/Linux).
    3. `kaggle datasets download -d solesensei/solesensei_bdd100k` (several
       GB -- this will take a while) then extract the zip.
    4. The extracted layout is: <extracted>/bdd100k/bdd100k/images/100k/{train,val}/
       and <extracted>/bdd100k/bdd100k/labels/ (containing the label JSON files).
    (Full exact PowerShell commands are in the project README.)

    KNOWN QUIRK, HANDLED AUTOMATICALLY: on this mirror, images under
    .../100k/train are further split into trainA/ and trainB/ subfolders
    (not sitting flat), and val may be split similarly. This script indexes
    images recursively (see `_index_images_by_filename`), so this nesting
    is handled transparently -- you do NOT need to manually flatten any
    folders before running this script.

    NOTE: `--labels-train` / `--labels-val` below accept EITHER a direct
    path to the label .json file OR a path to a folder containing it --
    this script will find the right file automatically.

===========================================================================
STEP 2: Run this script
===========================================================================
    python prepare_weather_dataset.py `
        --images-train "C:\\path\\to\\bdd100k\\bdd100k\\images\\100k\\train" `
        --images-val   "C:\\path\\to\\bdd100k\\bdd100k\\images\\100k\\val" `
        --labels-train "C:\\path\\to\\bdd100k\\bdd100k\\labels" `
        --labels-val   "C:\\path\\to\\bdd100k\\bdd100k\\labels" `
        --output data/weather_dataset `
        --max-per-class 800

(Backtick ` is PowerShell's line-continuation character; on Mac/Linux use a
backslash instead, or just put the whole command on one line.)

Run with --dry-run first to see the class distribution BEFORE copying any
files (recommended, since BDD100K's "foggy" images are naturally rare --
see the honest note in `_summarize` below).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from utils import get_logger

logger = get_logger(__name__)

CANONICAL_CLASSES = ["clear", "rain", "fog", "snow", "night", "overcast"]

# Priority order used when an image could plausibly map to more than one
# class (e.g. "overcast" weather + "night" timeofday). Rarer / more visually
# dominant conditions take priority over subtler ones. This ordering is a
# deliberate, documented design choice -- report it as such in your capstone
# writeup rather than treating the mapping as ground truth.
_PRIORITY = ["fog", "snow", "rain", "night", "overcast", "clear"]


def map_attributes_to_class(weather: str, timeofday: str) -> Optional[str]:
    """Map a BDD100K (weather, timeofday) attribute pair to one of our 6 classes.

    Returns None for images that shouldn't be used (undefined attributes).
    """
    candidates = set()

    if weather == "foggy":
        candidates.add("fog")
    if weather == "snowy":
        candidates.add("snow")
    if weather == "rainy":
        candidates.add("rain")
    if timeofday == "night":
        candidates.add("night")
    if weather == "overcast":
        candidates.add("overcast")
    if weather in ("clear", "partly cloudy"):
        candidates.add("clear")

    if not candidates:
        return None  # weather == "undefined" and timeofday != "night", etc.

    for cls in _PRIORITY:
        if cls in candidates:
            return cls
    return None  # should not happen given the mapping above


def _resolve_label_file(labels_path: str, split_hint: str) -> Path:
    """Accept either a direct .json file path OR a directory containing one.

    BDD100K label filenames have varied across dataset releases/mirrors
    (e.g. `bdd100k_labels_images_train.json` in the original 2018 release,
    `det_train.json` in the 2020 detection package, and Kaggle mirrors have
    sometimes renamed them again). Rather than hardcode one exact filename
    and break the moment a mirror differs, this resolves a directory to the
    most plausible matching .json file by looking for `split_hint`
    ("train" or "val") in the filename.
    """
    path = Path(labels_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Labels path does not exist: {labels_path}")

    candidates = sorted(path.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No .json files found in labels directory: {labels_path}. "
            f"Point --labels-{split_hint} directly at the label JSON file instead."
        )

    matching = [c for c in candidates if split_hint in c.name.lower()]
    chosen = matching[0] if matching else candidates[0]

    if len(candidates) > 1 and len(matching) != 1:
        logger.warning(
            "Multiple .json files found in %s and none uniquely matched '%s' -- "
            "using %s. Pass the exact file via --labels-%s if this is wrong. "
            "All candidates found: %s",
            labels_path, split_hint, chosen.name, split_hint, [c.name for c in candidates],
        )
    else:
        logger.info("Resolved --labels-%s directory to file: %s", split_hint, chosen.name)

    return chosen


def load_bdd_labels(labels_path: str, split_hint: str = "") -> Dict[str, Dict[str, str]]:
    """Parse a BDD100K label JSON file into {image_filename: {weather, timeofday}}.

    `labels_path` may be a direct file path or a directory (auto-resolved
    via `_resolve_label_file`).
    """
    resolved_path = _resolve_label_file(labels_path, split_hint) if split_hint else Path(labels_path)
    with open(resolved_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected {resolved_path} to contain a JSON list of image label "
            f"entries (the standard BDD100K format), but got {type(raw).__name__}. "
            "If this is a differently-structured mirror, inspect the file and "
            "adjust load_bdd_labels() accordingly."
        )

    parsed: Dict[str, Dict[str, str]] = {}
    for entry in raw:
        name = entry.get("name")
        attrs = entry.get("attributes", {})
        weather = attrs.get("weather", "undefined")
        timeofday = attrs.get("timeofday", "undefined")
        if name is None:
            continue
        parsed[name] = {"weather": weather, "timeofday": timeofday}

    if not parsed:
        raise ValueError(
            f"Parsed 0 labeled images from {resolved_path}. The file exists and "
            "is valid JSON, but no entries had a 'name' field -- double-check "
            "this is really a BDD100K per-image label file, not e.g. a "
            "categories/metadata file from the same download."
        )
    return parsed


def _index_images_by_filename(images_dir_path: Path) -> Dict[str, Path]:
    """Build a {basename: full_path} index via a single recursive walk.

    BDD100K's Kaggle mirror has a documented quirk where images under
    .../100k/train are further split into trainA/ and trainB/ subfolders
    (and similarly for val), rather than sitting flat in the train/
    directory. A naive `images_dir / filename` join silently finds 0
    matches in that case -- confirmed by direct testing -- rather than
    raising an obvious error, because `Path.exists()` just returns False
    for every file. That looks like "no images matched" rather than "wrong
    path shape", which is a confusing failure mode to debug under time
    pressure. Indexing recursively once, up front, makes the lookup work
    correctly regardless of nesting depth, with no manual flattening step.
    """
    index: Dict[str, Path] = {}
    image_exts = {".jpg", ".jpeg", ".png"}
    for path in images_dir_path.rglob("*"):
        if path.suffix.lower() in image_exts:
            index[path.name] = path
    return index


def build_class_index(images_dir: str, labels_path: str, split_hint: str = "") -> Dict[str, List[Path]]:
    """Group image file paths by their mapped weather class.

    Looks images up via a recursive filename index (see
    `_index_images_by_filename`), so it transparently handles BDD100K's
    documented trainA/trainB (etc.) nesting without requiring the images
    to be manually flattened first.
    """
    images_dir_path = Path(images_dir)
    if not images_dir_path.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    labels = load_bdd_labels(labels_path, split_hint=split_hint)
    filename_index = _index_images_by_filename(images_dir_path)
    logger.info("Indexed %d image files under %s (recursive)", len(filename_index), images_dir)

    class_to_files: Dict[str, List[Path]] = defaultdict(list)
    unmatched = 0
    dropped_undefined = 0

    for filename, attrs in labels.items():
        image_path = filename_index.get(filename)
        if image_path is None:
            unmatched += 1
            continue
        cls = map_attributes_to_class(attrs["weather"], attrs["timeofday"])
        if cls is None:
            dropped_undefined += 1
            continue
        class_to_files[cls].append(image_path)

    logger.info(
        "Indexed %s: %d images matched, %d labels had no corresponding image "
        "file, %d dropped as undefined/unmappable.",
        images_dir, sum(len(v) for v in class_to_files.values()), unmatched, dropped_undefined,
    )
    if unmatched > 0 and sum(len(v) for v in class_to_files.values()) == 0:
        raise RuntimeError(
            f"0 images matched between {labels_path} and {images_dir} -- the "
            "label filenames don't correspond to any file found (recursively) "
            "under the images directory. Double-check --images-train/"
            "--images-val point at the correct top-level folder (e.g. "
            ".../images/100k/train), and that it actually contains the .jpg files "
            "somewhere within it."
        )
    return class_to_files


def _summarize(class_to_files: Dict[str, List[Path]], split_name: str) -> None:
    logger.info("--- %s class distribution (before capping) ---", split_name)
    for cls in CANONICAL_CLASSES:
        count = len(class_to_files.get(cls, []))
        flag = "  <-- naturally rare in BDD100K; report this honestly rather than hiding it" \
            if cls == "fog" and count < 300 else ""
        logger.info("  %-10s %6d images%s", cls, count, flag)


def copy_split(
    class_to_files: Dict[str, List[Path]],
    output_dir: Path,
    split_name: str,
    max_per_class: Optional[int],
    seed: int,
    dry_run: bool,
) -> Counter:
    rng = random.Random(seed)
    counts: Counter = Counter()

    for cls in CANONICAL_CLASSES:
        files = list(class_to_files.get(cls, []))
        rng.shuffle(files)
        if max_per_class is not None:
            files = files[:max_per_class]
        counts[cls] = len(files)

        if dry_run:
            continue

        dest_dir = output_dir / split_name / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/weather_dataset/ from BDD100K.")
    parser.add_argument("--images-train", type=str, required=True,
                         help="Path to BDD100K train images directory (e.g. .../images/100k/train). "
                              "Subfolders like trainA/trainB are handled automatically.")
    parser.add_argument("--images-val", type=str, required=True,
                         help="Path to BDD100K val images directory.")
    parser.add_argument("--labels-train", type=str, required=True,
                         help="Path to the BDD100K train labels .json file, OR a "
                              "directory containing it (auto-detected by filename).")
    parser.add_argument("--labels-val", type=str, required=True,
                         help="Path to the BDD100K val labels .json file, OR a "
                              "directory containing it (auto-detected by filename).")
    parser.add_argument("--output", type=str, default="data/weather_dataset",
                         help="Output directory (will contain train/, val/, and test/ subfolders).")
    parser.add_argument("--max-per-class", type=int, default=800,
                         help="Cap images per class in the train split, to keep CPU "
                              "training time reasonable and prevent common classes "
                              "(clear/overcast) from dominating. Use -1 for no cap.")
    parser.add_argument("--max-per-class-val", type=int, default=150,
                         help="Cap images per class in the val split.")
    parser.add_argument("--val-fraction-as-test", type=float, default=0.3,
                         help="Fraction of the (capped) val split to instead route into "
                              "a held-out test/ split, evaluated once at the end of "
                              "training but never used for checkpoint selection. "
                              "Set to 0 to disable and only produce train/ + val/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                         help="Only print the class distribution; don't copy any files.")
    args = parser.parse_args()

    output_dir = Path(args.output)

    logger.info("Indexing BDD100K train split...")
    train_classes = build_class_index(args.images_train, args.labels_train, split_hint="train")
    _summarize(train_classes, "TRAIN")

    logger.info("Indexing BDD100K val split...")
    val_classes = build_class_index(args.images_val, args.labels_val, split_hint="val")
    _summarize(val_classes, "VAL")

    max_train = None if args.max_per_class < 0 else args.max_per_class

    if args.dry_run:
        logger.info("Dry run: no files copied. Re-run without --dry-run to build the dataset.")
        copy_split(train_classes, output_dir, "train", max_train, args.seed, dry_run=True)
        copy_split(val_classes, output_dir, "val", args.max_per_class_val, args.seed, dry_run=True)
        return

    logger.info("Copying TRAIN split (capped at %s per class)...", max_train)
    train_counts = copy_split(train_classes, output_dir, "train", max_train, args.seed, dry_run=False)

    # Split the val pool into val/ (for checkpoint selection) and test/
    # (held out, evaluated once at the end) so the reported accuracy isn't
    # measured on the same split used to pick the best epoch.
    #
    # The split is made per recording session, not per image: BDD100K names
    # are "<session>-<frame>", and frames from one session (sometimes
    # near-identical) used to land in both val/ and test/ (87 sessions in the
    # first build), so test wasn't independent of the split used for
    # checkpoint selection. BDD's own train/val split is already grouped by
    # session; any session that still appears in the train pool is dropped
    # from val/test entirely.
    def session(path: Path) -> str:
        return path.name.split("-")[0]

    train_sessions = {session(p) for files in train_classes.values() for p in files}
    val_sessions = sorted({session(p) for files in val_classes.values() for p in files} - train_sessions)
    rng = random.Random(args.seed + 1)
    rng.shuffle(val_sessions)
    test_sessions = set(val_sessions[: int(len(val_sessions) * args.val_fraction_as_test)])

    val_pool: Dict[str, List[Path]] = {}
    test_pool: Dict[str, List[Path]] = {}
    dropped = 0
    for cls, files in val_classes.items():
        kept = [p for p in files if session(p) not in train_sessions]
        dropped += len(files) - len(kept)
        test_pool[cls] = [p for p in kept if session(p) in test_sessions]
        val_pool[cls] = [p for p in kept if session(p) not in test_sessions]
    logger.info("Val/test split by recording session: %d test sessions of %d; "
                "%d val images dropped because their session also appears in train.",
                len(test_sessions), len(val_sessions), dropped)

    logger.info("Copying VAL split (capped at %s per class)...", args.max_per_class_val)
    val_counts = copy_split(val_pool, output_dir, "val", args.max_per_class_val, args.seed, dry_run=False)

    test_counts: Counter = Counter()
    if args.val_fraction_as_test > 0:
        logger.info("Copying held-out TEST split...")
        test_counts = copy_split(test_pool, output_dir, "test", args.max_per_class_val, args.seed, dry_run=False)

    logger.info("=" * 60)
    logger.info("Dataset written to %s", output_dir.resolve())
    logger.info("%-10s %8s %8s %8s", "class", "train", "val", "test")
    for cls in CANONICAL_CLASSES:
        logger.info("%-10s %8d %8d %8d", cls, train_counts.get(cls, 0),
                     val_counts.get(cls, 0), test_counts.get(cls, 0))
    logger.info("=" * 60)
    logger.info(
        "Next step: python models.py --data-dir %s --epochs 15 --freeze-backbone", args.output
    )


if __name__ == "__main__":
    main()
