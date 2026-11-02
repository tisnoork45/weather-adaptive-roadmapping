#!/usr/bin/env bash
# Copy the project working tree (tracked + new non-ignored files) to $1, with weights/video/data symlinked.
set -euo pipefail
PROJ=/workspace/weather-adaptive-roadmapping
SB=${1:?dest}
case "$SB" in /tmp/claude-0/*) ;; *) echo "refusing: $SB"; exit 1;; esac
rm -rf -- "$SB"; mkdir -p "$SB"
cd "$PROJ"
{ git ls-files; git ls-files --others --exclude-standard; } | grep -v '^Depth-Anything-V2/' | sort -u \
  | while read -r f; do [ -f "$f" ] && printf '%s\0' "$f"; done | xargs -0 cp --parents -t "$SB"
mkdir -p "$SB/weights"
for w in "$PROJ"/weights/*.pt "$PROJ"/weights/*.pth; do ln -sf "$(readlink -f "$w")" "$SB/weights/$(basename "$w")"; done
ln -sfn "$PROJ/road.mp4" "$SB/road.mp4"
rm -rf -- "${SB:?}/data"; ln -sfn "$PROJ/data" "$SB/data"
