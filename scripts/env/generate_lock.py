"""Write requirements-lock.txt from the packages installed in the current environment.

    python scripts/env/generate_lock.py              # writes requirements-lock.txt in the repository root
    python scripts/env/generate_lock.py --stdout

The lock lists the exact installed version of every package in
requirements.txt and everything they depend on (the transitive closure,
read from installed package metadata). Nothing is installed, upgraded or
downloaded.

Packages whose installed build cannot be fetched from PyPI -- a local version
label (e.g. ``+f70bd71a48.nv24.6``) or a pre-release built from a local
file or directory, as in the NVIDIA NGC PyTorch container -- are written as
comments, not pins: they are reproducible only in that container image (see
ENVIRONMENT.md). Packages installed from a local wheel whose version number
also exists as a regular release are pinned, with a note.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def direct_requirements(path: Path) -> List[Requirement]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(Requirement(line))
    return reqs


def _distribution(name: str) -> Optional[metadata.Distribution]:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def closure(direct: List[Requirement]) -> Tuple[Dict[str, metadata.Distribution], List[str]]:
    """Installed distributions reachable from ``direct`` (extras are not followed)."""
    env = default_environment()
    env["extra"] = ""
    found: Dict[str, metadata.Distribution] = {}
    missing: List[str] = []
    todo = [r.name for r in direct]
    seen: Set[str] = set()
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        dist = _distribution(name)
        if dist is None:
            missing.append(name)
            continue
        found[name] = dist
        for spec in dist.requires or []:
            try:
                req = Requirement(spec)
            except InvalidRequirement:
                continue
            if req.marker is None or req.marker.evaluate(env):
                todo.append(req.name)
    return found, sorted(missing)


def provenance(dist: metadata.Distribution) -> Tuple[str, str]:
    """('pypi' | 'local-version' | 'local-file', detail)."""
    url = ""
    raw = dist.read_text("direct_url.json")
    if raw:
        try:
            url = json.loads(raw).get("url", "")
        except ValueError:
            url = ""
    version = Version(dist.version)
    if version.local is not None or (url.startswith("file:") and version.is_prerelease):
        return "local-version", url
    if url.startswith("file:"):
        return "local-file", url
    return "pypi", url


def render(direct: List[Requirement]) -> str:
    dists, missing = closure(direct)
    direct_names = {canonicalize_name(r.name) for r in direct}
    lines = [
        "# requirements-lock.txt -- exact versions of the reference environment (generated; do not edit by hand)",
        f"# Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')} by scripts/env/generate_lock.py from "
        "requirements.txt and the installed packages.",
        f"# Reference environment: Python {platform.python_version()} on {platform.system()} {platform.machine()}"
        + (", NVIDIA NGC PyTorch container" if _ngc_version() else "")
        + (f" {_ngc_version()}" if _ngc_version() else "") + ". See ENVIRONMENT.md.",
        "#",
        "# Lines starting with '# CONTAINER BUILD' are packages built inside that container image. They are not",
        "# on PyPI and cannot be reproduced from this file; outside the container install a public build instead",
        "# (ENVIRONMENT.md, 'Standard installation'). Every other line is an exact pin.",
        "",
    ]

    def block(title: str, names: List[str]) -> None:
        lines.append(f"# --- {title}")
        for name in names:
            dist = dists[name]
            kind, url = provenance(dist)
            label = dist.metadata["Name"]
            if kind == "local-version":
                lines.append(f"# CONTAINER BUILD: {label}=={dist.version}  (installed from {url or 'the image'})")
            elif kind == "local-file":
                lines.append(f"{label}=={dist.version}  # installed from a container wheel ({url}); "
                             "pinned to the same version number")
            else:
                lines.append(f"{label}=={dist.version}")
        lines.append("")

    block("direct dependencies (requirements.txt)", sorted(n for n in dists if n in direct_names))
    block("transitive dependencies", sorted(n for n in dists if n not in direct_names))
    if missing:
        lines.append("# Required by an installed package's metadata but not installed here: " + ", ".join(missing))
        lines.append("")
    return "\n".join(lines)


def _ngc_version() -> Optional[str]:
    import os

    v = os.environ.get("NVIDIA_PYTORCH_VERSION")
    return f"{v} (nvcr.io/nvidia/pytorch:{v}-py3)" if v and re.fullmatch(r"\d+\.\d+", v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    ap.add_argument("--out", default=str(ROOT / "requirements-lock.txt"))
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    text = render(direct_requirements(Path(args.requirements)))
    if args.stdout:
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
