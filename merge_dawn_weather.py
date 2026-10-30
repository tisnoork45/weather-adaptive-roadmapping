"""Merge DAWN dataset images into the existing BDD100K-based weather dataset.

Standalone, self-contained script -- does NOT import from or modify any
other project file (models.py, prepare_weather_dataset.py, utils.py, etc.).
By default only ever writes into data/weather_dataset/train/<class>/ --
val/ and test/ are never touched. The one exception is the opt-in
--holdout-fraction flag (see "Holdout" below).

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
Holdout (opt-in: --holdout-fraction)
===========================================================================
BDD100K has very few fog images, so val/test end up with almost no fog
(10 val / 3 test in the logged training run) while train's fog is mostly
DAWN. That makes fog metrics meaningless. --holdout-fraction F sends a
fraction F of each --holdout-classes class (default: fog) to val/ and test/
(F/2 each) instead of train/. Which split an image goes to is a pure
function of its filename hash, so re-runs are stable; and an image already
present in ANY split -- matched by file content, since DAWN contains
byte-identical images under different names -- is skipped, so the same
image can never end up in both train and val/test (no leakage), even when re-running over a dataset
merged earlier without a holdout. Those already-in-train images stay in
train -- move them by hand if you want a larger holdout.

===========================================================================
USAGE
===========================================================================
    # See what WOULD happen -- no files are written:
    python merge_dawn_weather.py --dry-run

    # Actually copy:
    python merge_dawn_weather.py

    # Hold out 30% of DAWN fog for val/test (15% each):
    python merge_dawn_weather.py --holdout-fraction 0.3 --dry-run

    # Custom paths, if yours differ from the defaults:
    python merge_dawn_weather.py `
        --dawn-images dawn_download/images `
        --dawn-labels dawn_download/labels `
        --dataset-root data/weather_dataset `
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
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

SPLITS = ("train", "val", "test")

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

    Note: the copy logic always appends the split name itself ("train", or
    "val"/"test" only for an explicit --holdout-fraction), so this check
    exists to catch the realistic user error of pointing --dataset-root
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


def assign_split(img_name: str, target_class: str, holdout_classes: List[str],
                 holdout_fraction: float) -> str:
    """Deterministically pick train/val/test for one image (see "Holdout")."""
    if holdout_fraction <= 0 or target_class not in holdout_classes:
        return "train"
    bucket = int(hashlib.sha1(img_name.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32
    if bucket < holdout_fraction / 2:
        return "val"
    if bucket < holdout_fraction:
        return "test"
    return "train"


def plan_copies(
    images_by_class: Dict[str, List[Path]], dataset_root: Path,
    holdout_classes: List[str] = (), holdout_fraction: float = 0.0,
    near_dup_threshold: int = 24,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """For each target class, compute (source, destination) pairs, skipping
    any image already present in ANY split (idempotency, and no train/val/test
    leakage when the holdout settings differ from a previous run).

    Duplicates are detected by file CONTENT, not just name: DAWN ships
    byte-identical images under different names (e.g. haze-024, haze-039 and
    foggy-060 are one file), so name-only checks let the same image land in
    train and val/test. Only the first copy (in sorted order) of each content
    hash is used, and anything whose content already exists anywhere in the
    dataset's splits is skipped.

    DAWN also contains bursts of near-identical shots of one scene (sometimes
    filed under two conditions, e.g. a rain_storm and a foggy image of the
    same moment). Images whose perceptual hashes are within
    `near_dup_threshold` bits are grouped, and each group goes to ONE split:
    a split already used by a previously copied member, else train if any
    member is outside `holdout_classes`, else the split of the group's first
    member by name. Threshold 24 (of 256 bits) was chosen by inspection:
    pairs <= 16 were the same scene, pairs >= 26 were different scenes.
    """
    existing_hashes = {
        _sha256(p) for split in SPLITS if (dataset_root / split).is_dir()
        for p in (dataset_root / split).rglob("*") if p.is_file() and not p.name.startswith("dawn_")
    }
    # 1) Exact-content dedupe. Previously copied DAWN files stay in the
    #    candidate list (marked with their split) so new members of their
    #    near-duplicate group follow them.
    candidates: List[Tuple[str, Path, str]] = []   # (class, source, split already on disk or "")
    seen_hashes = set(existing_hashes)
    skipped_duplicates = 0
    for target_class in sorted(images_by_class):
        for img_path in sorted(images_by_class[target_class]):
            dest_name = f"dawn_{img_path.name}"
            on_disk = next((s for s in SPLITS if (dataset_root / s / target_class / dest_name).exists()), "")
            content_hash = _sha256(img_path)
            if content_hash in seen_hashes and not on_disk:
                skipped_duplicates += 1
                continue  # identical image already planned or already in the dataset
            seen_hashes.add(content_hash)
            candidates.append((target_class, img_path, on_disk))

    # 2) Near-duplicate grouping (union-find over perceptual-hash distance).
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [_dhash(path) for _, path, _ in candidates] if near_dup_threshold > 0 else []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= near_dup_threshold:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(candidates)):
        groups[find(i)].append(i)

    # 3) One split per group.
    plan: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    multi_member_groups = 0
    for members in groups.values():
        multi_member_groups += len(members) > 1
        on_disk_splits = sorted({candidates[i][2] for i in members if candidates[i][2]})
        if on_disk_splits:
            split = on_disk_splits[0]
        elif any(candidates[i][0] not in holdout_classes for i in members):
            split = "train"
        else:
            first = min(members, key=lambda i: candidates[i][1].name)
            split = assign_split(candidates[first][1].name, candidates[first][0],
                                 holdout_classes, holdout_fraction)
        for i in members:
            target_class, img_path, on_disk = candidates[i]
            if not on_disk:
                plan[target_class].append((img_path, dataset_root / split / target_class / f"dawn_{img_path.name}"))
    log(f"Skipped {skipped_duplicates} images whose exact content is a duplicate "
        f"(of another DAWN image or of a file already in the dataset).")
    log(f"Grouped near-duplicate shots into {multi_member_groups} multi-image groups "
        f"(perceptual-hash distance <= {near_dup_threshold}); each group is kept in one split.")
    return plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dhash(path: Path) -> int:
    """256-bit difference hash: robust to resizing/re-encoding of the same shot."""
    from PIL import Image  # only needed here; keeps the rest of the script stdlib-only

    with Image.open(path) as im:
        small = im.convert("L").resize((17, 16), Image.BILINEAR)
    px = list(small.getdata())
    bits = 0
    for row in range(16):
        for col in range(16):
            bits = (bits << 1) | (px[row * 17 + col + 1] > px[row * 17 + col])
    return bits


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
        description="Merge DAWN dataset images into data/weather_dataset/train/ (and, with --holdout-fraction, val/ + test/)."
    )
    parser.add_argument("--dawn-images", type=str, default="dawn_download/images",
                         help="Path to DAWN's images directory (flat -- images directly "
                              "in this directory, condition encoded in each filename, "
                              "e.g. 'foggy-001.jpg'). Searched recursively regardless.")
    parser.add_argument("--dawn-labels", type=str, default="dawn_download/labels",
                         help="Path to DAWN's labels directory (searched recursively).")
    parser.add_argument("--dataset-root", type=str, default="data/weather_dataset",
                         help="Root of the existing weather dataset. Only its train/ "
                              "subfolder is written to unless --holdout-fraction is set.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be copied without writing any files.")
    parser.add_argument("--holdout-fraction", type=float, default=0.0,
                         help="Opt-in: fraction of each --holdout-classes class to send to "
                              "val/ and test/ (half each) instead of train/. Default 0 = "
                              "train only.")
    parser.add_argument("--holdout-classes", nargs="+", default=["fog"],
                         help="Classes the holdout applies to (default: fog).")
    parser.add_argument("--near-dup-threshold", type=int, default=24,
                         help="Group DAWN images whose 256-bit perceptual hashes differ by at "
                              "most this many bits into one split (0 disables).")
    args = parser.parse_args()
    if not 0.0 <= args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be in [0, 1).")

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
    plan = plan_copies(images_by_class, dataset_root, args.holdout_classes, args.holdout_fraction,
                       args.near_dup_threshold)
    already_present = count_existing_dawn_files(train_dir)

    log("Per-class summary (eligible = has a matching label):")
    for cls in _TARGET_CLASSES:
        eligible = len(images_by_class.get(cls, []))
        would_copy = len(plan.get(cls, []))
        already = already_present.get(cls, 0)
        log(f"  {cls:8s}  eligible={eligible:5d}  already_copied={already:5d}  "
            f"WILL_COPY={would_copy:5d}")
        if args.holdout_fraction > 0 and cls in args.holdout_classes:
            by_split = Counter(dest.parent.parent.name for _, dest in plan.get(cls, []))
            log(f"  {'':8s}  holdout split of WILL_COPY: train={by_split['train']} "
                f"val={by_split['val']} test={by_split['test']}")
    log("=" * 70)

    if args.dry_run:
        total_would_copy = sum(len(v) for v in plan.values())
        log(f"DRY RUN: 0 files written. {total_would_copy} files would be copied "
            f"in total if re-run without --dry-run.")
        return

    log(f"Copying files into {dataset_root}/{'{train,val,test}' if args.holdout_fraction > 0 else 'train'}/ ...")
    copied = execute_copies(plan)
    for cls in _TARGET_CLASSES:
        log(f"  {cls:8s}  copied_this_run={copied.get(cls, 0)}")

    log("=" * 70)
    log("Final class counts in data/weather_dataset/train/ (BDD100K + DAWN combined):")
    final_counts = count_total_files(train_dir)
    for cls in _TARGET_CLASSES:
        log(f"  {cls:8s}  total={final_counts.get(cls, 0)}")
    log("=" * 70)
    if args.holdout_fraction > 0:
        log(f"Done. Held-out images for {args.holdout_classes} were written to val/ and test/.")
    else:
        log("Done. val/ and test/ under the dataset root were not touched.")


if __name__ == "__main__":
    main()
