"""Report how the current Python environment differs from requirements.txt and requirements-lock.txt.

    python scripts/env/check_env.py           # exit status 1 if a requirements.txt range is not met

Nothing is installed or changed. Three checks:

  1. requirements.txt: every listed package is installed within its version range (the software's
     actual requirement; a failure here sets the exit status).
  2. requirements-lock.txt pins: installed version == the reference environment's version
     (informational: other versions inside the requirements.txt ranges are allowed).
  3. CONTAINER BUILD lines (NGC-specific torch/torchvision builds): reported for information only;
     outside the reference container a different, public build is expected.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Optional, Tuple

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
CONTAINER_LINE = re.compile(r"^#\s*CONTAINER BUILD:\s*([A-Za-z0-9._-]+)==(\S+)")


def installed_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def read_requirements(path: Path) -> List[Requirement]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(Requirement(line))
    return out


def read_lock(path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    pins, container = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CONTAINER_LINE.match(line)
        if m:
            container.append((m.group(1), m.group(2)))
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins.append((name.strip(), version.strip()))
    return pins, container


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    ap.add_argument("--lock", default=str(ROOT / "requirements-lock.txt"))
    ap.add_argument("--verbose", action="store_true", help="also list packages that match")
    args = ap.parse_args()

    failures = 0
    print(f"Python {sys.version.split()[0]} ({sys.executable})")

    print(f"\n[1] {args.requirements} (version ranges)")
    for req in read_requirements(Path(args.requirements)):
        have = installed_version(req.name)
        if have is None:
            print(f"  MISSING   {req}")
            failures += 1
        elif req.specifier and not req.specifier.contains(Version(have), prereleases=True):
            print(f"  OUT OF RANGE  {req.name} {have} (needs {req.specifier})")
            failures += 1
        elif args.verbose:
            print(f"  ok        {req.name} {have}")

    lock = Path(args.lock)
    if not lock.exists():
        print(f"\n[2] {lock} not found; skipped")
    else:
        pins, container = read_lock(lock)
        differ = [(n, v, installed_version(n)) for n, v in pins if installed_version(n) != v]
        print(f"\n[2] {lock.name}: {len(pins) - len(differ)} of {len(pins)} pinned packages match the reference")
        for name, want, have in differ:
            print(f"  differs   {name}: installed {have or 'not installed'}, reference {want}")
        print(f"\n[3] container builds in the reference environment (informational)")
        for name, want in container:
            have = installed_version(name)
            state = "same build" if have == want else f"installed {have or 'not installed'}"
            print(f"  {name}: reference {want} -> {state}")

    print("\nResult:", "requirements.txt satisfied" if failures == 0 else f"{failures} requirements.txt problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
