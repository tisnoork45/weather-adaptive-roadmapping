"""Merge DAWN dataset images into the existing BDD100K-based weather dataset.

Standalone, self-contained script -- does NOT import from or modify any
other project file (models.py, prepare_weather_dataset.py, utils.py, etc.).
Only ever writes into data/weather_dataset/train/<class>/ -- val/ and test/
are never touched, by construction (there is no code path in this script
that can write to them).

===========================================================================
DAWN's actual directory structure (flat, no per-condition subfolders)
===========================================================================
    dawn_download/images/foggy-001.jpg, haze-014.jpg, rain_storm-027.jpg, ...
    dawn_download/labels/foggy-001.xml, haze-014.xml, rain_storm-027.xml, ...

All images sit directly in one flat directory; the DAWN "condition" is
encoded in the filename itself as `<condition>-<number>` (e.g.
`rain_storm-027.jpg` -> condition `rain_storm`). This is extracted via
`extract_condition()` below by stripping the trailing `-<digits>` suffix,
NOT by looking at subfolder names (there aren't any).

===========================================================================
DAWN class mapping (per your specification)
===========================================================================
    foggy       -> fog
    haze        -> fog
    mist        -> fog
    rain_storm  -> rain
    snow_storm  -> snow

Ignored (not copied, not an error -- just skipped and logged):
    dusttornado, sand_storm, sand_storm_g2, sand_storm_g2_

Any DAWN condition that is neither in the mapping above nor in the ignore
list is also skipped, with a warning (so a typo'd/renamed/unexpected
condition fails loudly instead of silently copying nothing or the wrong
thing).

===========================================================================
Filename uniqueness (a correctness detail worth knowing about)
===========================================================================
Because DAWN's filenames already embed their condition (`foggy-001.jpg`,
`haze-001.jpg` -- note these are DIFFERENT strings despite both mapping to
the `fog` class and both using numeric suffix `001`), simply prefixing the
original filename with `dawn_` (e.g. `dawn_foggy-001.jpg`) is already
collision-safe with no extra disambiguation needed. This also makes
re-running the script safe/idempotent (item 8 below): the destination
filename is fully deterministic from the source filename, so a file that
was already copied is detected via a simple `dest.exists()` check and
skipped, never duplicated or overwritten.

===========================================================================
Label matching
===========================================================================
For each candidate image, this script looks for a label file with the same
filename stem (any extension) anywhere under --dawn-labels (matched via a
one-time recursive index, not a per-image directory walk). Only images
that have a matching label file are copied -- an image with no
corresponding label is treated as an orphan/data-integrity issue and is
skipped with a warning rather than silently included.

===========================================================================
USAGE
===========================================================================
    # See what WOULD happen -- no files are written:
    python merge_dawn_weather.py --dry-run

    # Actually copy:
    python merge_dawn_weather.py

    # Custom paths, if yours differ from the defaults:
    python merge_dawn_weather.py `
        --dawn-images dawn_download/images `
        --dawn-labels dawn_download/labels `
        --dataset-root data/weather_dataset `
        --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# --- DAWN condition (from filename) -> our canonical weather_dataset class --
CLASS_MAP: Dict[str, str] = {
    "foggy": "fog",
    "haze": "fog",
    "mist": "fog",
    "rain_storm": "rain",
    "snow_storm": "snow",
}

# --- DAWN conditions to explicitly skip (not an error) ----------------------
IGNORE_CONDITIONS = {
    "dusttornado",
    "sand_storm",
    "sand_storm_g2",
    "sand_storm_g2_",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Every class this script is capable of writing into. Used as a hard
# allow-list safety check -- see `_assert_safe_target_dir`.
_TARGET_CLASSES = sorted(set(CLASS_MAP.values()))

# DAWN's flat naming convention is '<condition>-<number>', e.g. 'foggy-001',
# 'rain_storm-027', 'sand_storm_g2-014'. This strips a trailing '-<digits>'
# suffix to recover the condition token.
_CONDITION_SUFFIX_RE = re.compile(r"-\d+$")


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_condition(stem: str) -> str:
    """Extract the DAWN condition token from an image/label filename stem.

    E.g. 'rain_storm-027' -> 'rain_storm'. If no trailing '-<digits>'
    suffix is found (unexpected naming), the whole stem is returned as-is,
    which will then correctly fall through to the "unrecognized condition"
    warning path below rather than being silently misclassified.
    """
    match = _CONDITION_SUFFIX_RE.search(stem)
    return stem[:match.start()] if match else stem


def _assert_safe_target_dir(dataset_root: Path) -> Path:
    """Hard-validate --dataset-root itself is not already a split directory
    (train/val/test), then return dataset_root/train.

    Note: because this script's copy logic ALWAYS appends "train" itself
    (see below), there is no code path anywhere in this script capable of
    constructing a "val" or "test" destination path -- that guarantee comes
    from the code structure, not from validating the input. This check
    instead catches the realistic user error of pointing --dataset-root
    directly AT a split (e.g. accidentally passing ".../weather_dataset/val"),
    which would otherwise silently produce a nonsensical nested
    ".../weather_dataset/val/train" directory instead of failing loudly.
    """
    root_name = dataset_root.resolve().name.lower()
    if root_name in ("train", "val", "test"):
        raise RuntimeError(
            f"--dataset-root should be the PARENT of train/val/test (e.g. "
            f"'data/weather_dataset'), not a split directory itself. You "
            f"passed a path whose final component is '{root_name}' "
            f"({dataset_root.resolve()}). This script always appends 'train' "
            f"itself when writing output -- refusing to run to avoid "
            f"creating a confusing nested directory."
        )
    train_dir = dataset_root / "train"
    return train_dir


def index_labels(labels_root: Path) -> Dict[str, Path]:
    """Recursively index every label file under `labels_root` by filename stem.

    Handles arbitrary nesting (flat layout, mirrored per-class subfolders,
    or anything in between) via a single recursive walk, rather than
    assuming one fixed structure.
    """
    if not labels_root.is_dir():
        raise FileNotFoundError(f"--dawn-labels directory does not exist: {labels_root}")

    index: Dict[str, Path] = {}
    for path in labels_root.rglob("*"):
        if path.is_file():
            index[path.stem] = path
    return index


def discover_dawn_images(
    images_root: Path, label_index: Dict[str, Path]
) -> Tuple[Dict[str, List[Path]], Dict[str, int]]:
    """Recursively find every image under `images_root` (this handles both
    DAWN's actual flat layout -- images directly in the directory -- and any
    nested layout, via a single rglob), determine each image's DAWN
    condition from its FILENAME (not a subfolder -- there are none), and
    group matched images by target class.

    Returns:
        images_by_class: {target_class: [image_path, ...]}
            -- only images that have a matching label file are included.
        orphan_counts: {condition: count_of_images_with_no_label}
            -- for the summary report, so missing labels are visible rather
            than silently dropped.
    """
    if not images_root.is_dir():
        raise FileNotFoundError(f"--dawn-images directory does not exist: {images_root}")

    all_images = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        log(f"WARNING: no image files found under {images_root} (searched recursively).")

    by_condition: Dict[str, List[Path]] = defaultdict(list)
    for img_path in all_images:
        condition = extract_condition(img_path.stem).lower()
        by_condition[condition].append(img_path)

    images_by_class: Dict[str, List[Path]] = defaultdict(list)
    orphan_counts: Dict[str, int] = {}

    for condition in sorted(by_condition):
        paths = by_condition[condition]

        if condition in IGNORE_CONDITIONS:
            log(f"Skipping DAWN condition '{condition}' (in ignore list) -- {len(paths)} images.")
            continue
        if condition not in CLASS_MAP:
            log(f"WARNING: unrecognized DAWN condition '{condition}' ({len(paths)} images) "
                f"-- not in the class mapping or the ignore list. Skipping it. If this is "
                f"a real condition you want included, add it to CLASS_MAP at the top of "
                f"this script.")
            continue

        target_class = CLASS_MAP[condition]
        matched, orphaned = 0, 0
        for img_path in paths:
            if img_path.stem in label_index:
                images_by_class[target_class].append(img_path)
                matched += 1
            else:
                orphaned += 1

        orphan_counts[condition] = orphaned
        log(f"DAWN condition '{condition}' -> class '{target_class}': "
            f"{len(paths)} images found, {matched} have a matching label "
            f"({orphaned} orphaned/skipped, no label file).")

    return images_by_class, orphan_counts


def plan_copies(
    images_by_class: Dict[str, List[Path]], train_dir: Path
) -> Dict[str, List[Tuple[Path, Path]]]:
    """For each target class, compute (source, destination) pairs, skipping
    any destination that already exists (idempotency -- see module docstring).
    """
    plan: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for target_class, paths in images_by_class.items():
        dest_dir = train_dir / target_class
        for img_path in paths:
            dest_name = f"dawn_{img_path.name}"
            dest_path = dest_dir / dest_name
            if dest_path.exists():
                continue  # already copied in a previous run -- skip, don't duplicate
            plan[target_class].append((img_path, dest_path))
    return plan


def count_existing_dawn_files(train_dir: Path) -> Dict[str, int]:
    """Count how many dawn_-prefixed files already exist per target class
    (used both for the dry-run 'already present' report and to detect
    prior runs).
    """
    counts: Dict[str, int] = {}
    for cls in _TARGET_CLASSES:
        dest_dir = train_dir / cls
        if dest_dir.is_dir():
            counts[cls] = sum(1 for p in dest_dir.glob("dawn_*") if p.is_file())
        else:
            counts[cls] = 0
    return counts


def count_total_files(train_dir: Path) -> Dict[str, int]:
    """Count the TOTAL number of files (BDD100K + DAWN combined) per target
    class, for the final post-copy summary.
    """
    counts: Dict[str, int] = {}
    for cls in _TARGET_CLASSES:
        dest_dir = train_dir / cls
        counts[cls] = sum(1 for p in dest_dir.glob("*") if p.is_file()) if dest_dir.is_dir() else 0
    return counts


def execute_copies(plan: Dict[str, List[Tuple[Path, Path]]]) -> Counter:
    copied = Counter()
    for target_class, pairs in plan.items():
        dest_dir_created = set()
        for src, dest in pairs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest_dir_created.add(dest.parent)
            shutil.copy2(src, dest)
            copied[target_class] += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge DAWN dataset images into data/weather_dataset/train/ only."
    )
    parser.add_argument("--dawn-images", type=str, default="dawn_download/images",
                         help="Path to DAWN's images directory (flat -- images directly "
                              "in this directory, condition encoded in each filename, "
                              "e.g. 'foggy-001.jpg'). Searched recursively regardless.")
    parser.add_argument("--dawn-labels", type=str, default="dawn_download/labels",
                         help="Path to DAWN's labels directory (searched recursively).")
    parser.add_argument("--dataset-root", type=str, default="data/weather_dataset",
                         help="Root of the existing weather dataset. Only its train/ "
                              "subfolder is ever written to; val/ and test/ under this "
                              "root are never touched.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be copied without writing any files.")
    args = parser.parse_args()

    images_root = Path(args.dawn_images)
    labels_root = Path(args.dawn_labels)
    dataset_root = Path(args.dataset_root)

    train_dir = _assert_safe_target_dir(dataset_root)

    log("=" * 70)
    log("Indexing DAWN label files...")
    label_index = index_labels(labels_root)
    log(f"Indexed {len(label_index)} label files under {labels_root}")

    log("=" * 70)
    log("Discovering DAWN images and matching them to labels...")
    images_by_class, _orphan_counts = discover_dawn_images(images_root, label_index)

    log("=" * 70)
    plan = plan_copies(images_by_class, train_dir)
    already_present = count_existing_dawn_files(train_dir)

    log("Per-class summary (eligible = has a matching label):")
    for cls in _TARGET_CLASSES:
        eligible = len(images_by_class.get(cls, []))
        would_copy = len(plan.get(cls, []))
        already = already_present.get(cls, 0)
        log(f"  {cls:8s}  eligible={eligible:5d}  already_copied={already:5d}  "
            f"WILL_COPY={would_copy:5d}")
    log("=" * 70)

    if args.dry_run:
        total_would_copy = sum(len(v) for v in plan.values())
        log(f"DRY RUN: 0 files written. {total_would_copy} files would be copied "
            f"in total if re-run without --dry-run.")
        return

    log("Copying files into data/weather_dataset/train/ ...")
    copied = execute_copies(plan)
    for cls in _TARGET_CLASSES:
        log(f"  {cls:8s}  copied_this_run={copied.get(cls, 0)}")

    log("=" * 70)
    log("Final class counts in data/weather_dataset/train/ (BDD100K + DAWN combined):")
    final_counts = count_total_files(train_dir)
    for cls in _TARGET_CLASSES:
        log(f"  {cls:8s}  total={final_counts.get(cls, 0)}")
    log("=" * 70)
    log("Done. val/ and test/ under the dataset root were not touched.")


if __name__ == "__main__":
    main()
