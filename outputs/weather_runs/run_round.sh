#!/usr/bin/env bash
# Usage: run_round.sh NAME DATA_DIR SEED [extra models.py args...]
# Validation-only experiment: never evaluates the test split.
set -euo pipefail
name=$1; data=$2; seed=$3; shift 3
dir=outputs/weather_runs/$name-s$seed
mkdir -p "$dir"
python models.py --data-dir "$data" --output "$dir/model.pth" --run-dir "$dir" --seed "$seed" \
  --epochs 20 --num-workers 8 --no-test-eval "$@" > "$dir/train.log" 2>&1
echo "$name-s$seed done (exit $?)"
