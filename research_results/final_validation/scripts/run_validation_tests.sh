#!/usr/bin/env bash
# T6.5 verification in a scratch copy of the working tree (no __pycache__/logs written into the project).
# Goldens are read from the project (read-only use). Every pytest run keeps its COMPLETE output.
set -uo pipefail
S=/tmp/claude-0/-workspace-weather-adaptive-roadmapping/38a226b4-d2ac-4a06-80af-84bdc163140e/scratchpad
PROJ=/workspace/weather-adaptive-roadmapping; C=$S/fv/tests/copy; L=/workspace/weather-adaptive-roadmapping/research_results/final_validation/logs/tests
mkdir -p $L
bash $S/t61/mkcopy.sh $C >/dev/null
export WARM_REGRESSION_DIR=$PROJ/outputs/regression PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
cd $C
K="preprocess or pos_embed or point_cloud or depth_model_output_identical or never_builds_bev"
T="tests.py tests_depth.py tests_kitti_tracking.py tests_velocity_experimental.py tests_ego_motion.py tests_foundation.py tests_regression.py tests_dashboard.py"
echo "== goldens"
for m in relative metric; do python -B scripts/regression/golden.py compare --mode $m > $L/golden_$m.log 2>&1; echo "golden $m exit=$?"; tail -2 $L/golden_$m.log; done
rm -rf $S/fv/tests/kitti_g6
python -B scripts/regression/golden.py kitti --out $S/fv/tests/kitti_g6 > $L/g6.log 2>&1; echo "g6 exit=$?"; tail -2 $L/g6.log
python -B scripts/regression/golden.py reset --mode metric --frames 60 > $L/reset.log 2>&1; echo "reset exit=$?"; tail -3 $L/reset.log
echo "== full suite, default device"
python -B -m pytest -v -p no:cacheprovider -rs $T > $L/full_default.log 2>&1; echo "exit=$?"; tail -3 $L/full_default.log
echo "== G6 pytest with KITTI_VELOCITY_CACHE"
KITTI_VELOCITY_CACHE=/workspace/experiments/kitti_velocity/cache/20261031-190612_1892061-dirty python -B -m pytest -v -p no:cacheprovider tests_kitti_tracking.py -k G6 > $L/g6_pytest.log 2>&1; echo "exit=$?"; tail -1 $L/g6_pytest.log
echo "== dashboard + recording lifecycle tests"
python -B -m pytest -v -p no:cacheprovider -rs tests_dashboard.py tests_foundation.py > $L/dashboard_lifecycle.log 2>&1; echo "exit=$?"; tail -1 $L/dashboard_lifecycle.log
