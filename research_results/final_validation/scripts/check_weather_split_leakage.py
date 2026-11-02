"""Leakage check for the weather test split (data/weather_dataset_c3000).

Read-only. Checks, across train/val/test:
  1. identical file names,
  2. byte-identical files (SHA-256),
  3. near-duplicate images (64-bit difference hash, Hamming distance <= 4),
  4. shared BDD100K / DAWN name stems (same source video or sequence).
Writes data/weather_split_leakage.csv (one row per check) and prints a summary.
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/weather_dataset_c3000"
OUT = Path(__file__).resolve().parents[1] / "data/weather_split_leakage.csv"
SPLITS = ("train", "val", "test")


def dhash(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def stem_key(name):
    # BDD100K: <video-id>-<frame-id>.jpg ; DAWN merged as dawn_<Condition>-<n>.jpg
    s = Path(name).stem
    return s.rsplit("-", 1)[0] if s.startswith("dawn_") else s.split("-")[0]


files = {s: sorted(p for p in (DATA / s).rglob("*") if p.is_file()) for s in SPLITS}
names = {s: {p.name for p in files[s]} for s in SPLITS}
sha = {s: {} for s in SPLITS}
dh = {s: {} for s in SPLITS}
for s in SPLITS:
    for p in files[s]:
        sha[s][p] = hashlib.sha256(p.read_bytes()).hexdigest()
        dh[s][p] = dhash(p)
    print(f"{s}: {len(files[s])} files hashed", flush=True)

rows = []
test_sha = defaultdict(list)
for p, h in sha["test"].items():
    test_sha[h].append(p)
for other in ("train", "val"):
    rows.append({"check": "same_filename", "pair": f"test-{other}", "count": len(names["test"] & names[other])})
    rows.append({"check": "sha256_identical", "pair": f"test-{other}",
                 "count": sum(1 for h in sha[other].values() if h in test_sha)})
    stems_t = {stem_key(n) for n in names["test"]}
    rows.append({"check": "shared_name_stem", "pair": f"test-{other}",
                 "count": len(stems_t & {stem_key(n) for n in names[other]})})
    # near duplicates: exact Hamming search with numpy popcount
    t = np.array([v for v in dh["test"].values() if v is not None], dtype=np.uint64)
    o = np.array([v for v in dh[other].values() if v is not None], dtype=np.uint64)
    near = 0
    for v in t:
        x = np.bitwise_xor(o, v)
        bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        near += int((bits <= 4).any())
    rows.append({"check": "near_duplicate_dhash_le4", "pair": f"test-{other}", "count": near})
unread = sum(1 for s in SPLITS for v in dh[s].values() if v is None)
rows.append({"check": "unreadable_images", "pair": "all", "count": unread})
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "pair", "count"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
