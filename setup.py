"""Setuptools packaging script for Weather Adaptive Road Mapping.

Install in editable/development mode with:
    pip install -e .
"""

from pathlib import Path

from setuptools import setup

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


setup(
    name="weather-adaptive-road-mapping",
    version="1.0.0",
    description="Weather Adaptive Road Mapping for Autonomous Vehicles - "
                 "a weather-adaptive multi-sensor perception system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Capstone Project Team",
    python_requires=">=3.9,<3.13",
    py_modules=["utils", "models", "tracking", "fusion", "mapping", "pipeline", "download", "carla_sim"],
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
)
