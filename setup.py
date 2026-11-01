"""Setuptools packaging script for Weather Adaptive Road Mapping.

Install in editable/development mode with:
    pip install -e .
"""

from pathlib import Path

from setuptools import find_namespace_packages, setup

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8") \
    if (this_dir / "README.md").exists() else ""


def parse_requirements(path: str):
    req_file = this_dir / path
    if not req_file.exists():
        return []
    reqs = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


# Every top-level module the software imports. Backup copies (*.bak,
# models_backup_before_weather.py), one-off patch scripts (fix_run_py.py) and
# test files are deliberately left out. tests_foundation.py checks that every
# local import made by these modules is itself listed here.
PY_MODULES = [
    # live pipeline
    "utils", "config_schema", "camera", "models", "tracking", "fusion", "mapping",
    "relative_velocity", "ttc_calculator", "risk_engine", "pipeline", "run_logging",
    # entry points
    "run", "app", "carla_sim", "download",
    # dataset preparation
    "prepare_weather_dataset", "merge_dawn_weather",
    # offline evaluation (KITTI depth, Task 5 velocity and ego-motion)
    "kitti_depth", "depth_methods", "depth_metrics", "evaluate_kitti_depth",
    "kitti_tracking", "kitti_velocity_gt", "velocity_b0", "velocity_eval", "velocity_experimental",
    "evaluate_kitti_velocity", "ego_motion", "evaluate_kitti_egomotion",
]


setup(
    name="weather-adaptive-road-mapping",
    version="1.0.0",
    description="Weather Adaptive Road Mapping for Autonomous Vehicles - "
                 "a weather-adaptive monocular-camera perception system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Capstone Project Team",
    python_requires=">=3.9,<3.13",
    py_modules=PY_MODULES,
    # depth_anything_v2/ is the vendored Depth Anything V2 model code (no
    # __init__.py at the top level, hence namespace-package discovery).
    packages=find_namespace_packages(include=["depth_anything_v2", "depth_anything_v2.*"]),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
)
