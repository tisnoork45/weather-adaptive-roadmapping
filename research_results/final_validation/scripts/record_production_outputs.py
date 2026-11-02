"""Record the unmodified production pipeline's per-frame / per-object outputs.

Runs run.py's own main() (same loop, same timestamps, same RunRecorder used by
`python run.py --record-run`), but sends the recording to
research_results/final_validation/runs/ instead of outputs/runs/, so nothing is
written inside the production output folders. No video is written (no --output).
No production file is modified: the redirect is an in-memory default argument.

Usage (from the repository root):
  python -B research_results/final_validation/scripts/record_production_outputs.py \
      --source road.mp4 --depth-mode metric --record-run
"""
import functools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = Path(__file__).resolve().parents[1] / "runs"
sys.path.insert(0, str(REPO))

import run  # noqa: E402  (production CLI module, unchanged)

if "--record-run" not in sys.argv:
    sys.exit("pass --record-run; this wrapper exists only to redirect the recording directory")
RUNS.mkdir(parents=True, exist_ok=True)
run.RunRecorder.start = functools.partial(run.RunRecorder.start, root=str(RUNS))
run.main()
