"""Tests for the T6.1 software foundation: configuration validation, logging
set-up, the run recorder, packaging, and pipeline.run_weather_classification.

Run with:
    pytest tests_foundation.py -v

Like tests.py, everything runs on synthetic frames; TEST_DEVICE selects the
device for the tests that build a full pipeline (default cpu).
"""

from __future__ import annotations

import ast
import copy
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from config_schema import ConfigError, ensure_valid, reserved_keys, validate_config
from utils import ConfigDict, generate_synthetic_frame, load_config

TEST_DEVICE = os.environ.get("TEST_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parent


@pytest.fixture()
def raw_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# =============================================================================
# Configuration validation
# =============================================================================


def test_shipped_config_is_valid_without_warnings(raw_config) -> None:
    report = validate_config(raw_config)
    assert report.errors == [] and report.warnings == []
    assert validate_config(load_config(str(ROOT / "config.yaml"))).ok


def test_validation_never_modifies_the_config(raw_config) -> None:
    before = copy.deepcopy(raw_config)
    cfg = ConfigDict(raw_config)
    validate_config(raw_config)
    validate_config(cfg)
    assert raw_config == before and cfg == before


def test_reserved_keys_are_documented_not_flagged() -> None:
    assert set(reserved_keys()) == {
        "paths.weights_dir", "paths.data_dir", "paths.logs_dir", "camera.mount_height_m", "camera.pitch_deg",
        "camera.extrinsics", "dashboard.refresh_interval_sec",
    }


def _set(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if value is _DELETE:
        del node[last]
    else:
        node[last] = value


_DELETE = object()


@pytest.mark.parametrize("key, value, expected", [
    ("device.preference", "gpu", "device.preference: must be one of ['auto', 'cuda', 'mps', 'cpu'], got 'gpu'"),
    ("camera.intrinsics.fx", _DELETE, "camera.intrinsics.fx: required key is missing"),
    ("camera.intrinsics.fx", -5.0, "camera.intrinsics.fx: must be > 0, got -5.0"),
    ("camera.width", 1280.5, "camera.width: expected integer, got float 1280.5"),
    ("pipeline.strict", "yes", "pipeline.strict: expected boolean (true/false), got str 'yes'"),
    ("detection.confidence_threshold", 1.5, "detection.confidence_threshold: must be <= 1.0, got 1.5"),
    ("detection.classes_of_interest", [0, "car"], "detection.classes_of_interest[1]: expected integer"),
    ("depth_estimation.mode", "absolute", "depth_estimation.mode: must be one of ['relative', 'metric']"),
    ("depth_estimation.encoder", "vitg", "depth_estimation.encoder: must be one of ['vits', 'vitb', 'vitl']"),
    ("depth_estimation.metric_sha256", "abc", "depth_estimation.metric_sha256: must be a 64-character"),
    ("depth_estimation.ignore_bottom_rows_fraction", 1.0,
     "depth_estimation.ignore_bottom_rows_fraction: must be < 1.0"),
    ("weather_classification.normalize_std", [0.2, 0.2],
     "weather_classification.normalize_std: must have exactly 3 entries, got 2"),
    ("weather_classification.classes", ["clear", "clear"], "weather_classification.classes: must not contain"),
    ("velocity.history_length", 1, "velocity.history_length: must be >= 2, got 1"),
    ("velocity.ema_alpha", 0.0, "velocity.ema_alpha: must be > 0, got 0.0"),
    ("risk.ttc_thresholds_s.WARNING", 1.0, "risk.ttc_thresholds_s.WARNING: must be > CRITICAL (1.5), got 1.0"),
    ("risk.ttc_thresholds_s.CAUTION", 3.0, "risk.ttc_thresholds_s.CAUTION: must be > WARNING (3.0), got 3.0"),
    ("risk.weather_multipliers.fog", _DELETE, "risk.weather_multipliers: no entry for weather class(es) ['fog']"),
    ("fusion.weather_modifiers.night", _DELETE,
     "fusion.weather_modifiers: no entry for weather class(es) ['night']"),
    ("fusion.weather_modifiers.rain.camera_depth", -0.1,
     "fusion.weather_modifiers.rain.camera_depth: must be >= 0, got -0.1"),
    ("logging.level", "LOUD", "logging.level: must be one of"),
    ("tracking", [1, 2], "tracking: expected a mapping of keys, got list"),
])
def test_each_kind_of_invalid_value_is_reported_with_its_key_path(raw_config, key, value, expected) -> None:
    _set(raw_config, key, value)
    errors = validate_config(raw_config).errors
    assert any(e.startswith(expected) for e in errors), errors


def test_unknown_keys_and_unused_weather_entries_only_warn(raw_config) -> None:
    raw_config["camera"]["typo_key"] = 1
    raw_config["new_section"] = {}
    raw_config["risk"]["weather_multipliers"]["sandstorm"] = 2.0
    raw_config["fusion"]["weather_modifiers"]["clear"]["lidar"] = 1.0
    report = validate_config(raw_config)
    assert report.ok
    assert "camera.typo_key: unknown key (ignored by the software)" in report.warnings
    assert "new_section: unknown section (ignored by the software)" in report.warnings
    assert any(w.startswith("risk.weather_multipliers: entries ['sandstorm']") for w in report.warnings)
    assert any(w.startswith("fusion.weather_modifiers.clear: ['lidar']") for w in report.warnings)


def test_optional_keys_may_be_omitted(raw_config) -> None:
    for dotted in ("pipeline.strict", "pipeline.run_weather_classification", "depth_estimation.mode",
                   "camera.calibrated", "camera.extrinsics", "logging", "paths", "project", "datasets"):
        _set(raw_config, dotted, _DELETE)
    assert validate_config(raw_config).errors == []


def test_load_config_raises_one_error_listing_every_problem(tmp_path, raw_config) -> None:
    raw_config["device"]["preference"] = "gpu"
    raw_config["velocity"]["history_length"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    with pytest.raises(ConfigError) as info:
        load_config(str(path))
    assert len(info.value.errors) == 2
    assert str(path) in str(info.value) and "device.preference" in str(info.value)
    assert isinstance(load_config(str(path), validate=False), ConfigDict)


def test_load_config_logs_warnings(tmp_path, raw_config, caplog) -> None:
    raw_config["camera"]["typo_key"] = 1
    path = tmp_path / "warn.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    with caplog.at_level(logging.WARNING):
        load_config(str(path))
    assert "camera.typo_key: unknown key" in caplog.text


def test_missing_attribute_error_names_the_full_key_path() -> None:
    cfg = load_config(str(ROOT / "config.yaml"))
    with pytest.raises(AttributeError, match=r"'camera\.intrinsics\.fz' not found"):
        _ = cfg.camera.intrinsics.fz
    cfg.camera.new_block = {"a": 1}
    with pytest.raises(AttributeError, match=r"'camera\.new_block\.b' not found"):
        _ = cfg.camera.new_block.b
    assert copy.deepcopy(cfg) == cfg


def test_pipeline_rejects_invalid_overrides_before_loading_models() -> None:
    from pipeline import PerceptionPipeline

    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.depth_estimation["mode"] = "metrc"
    with pytest.raises(ConfigError, match="depth_estimation.mode"):
        PerceptionPipeline(cfg)


def test_ensure_valid_returns_the_report(raw_config) -> None:
    assert ensure_valid(raw_config).ok


# =============================================================================
# Logging: no import-time side effects
# =============================================================================


def _run_python(code: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=300)


def test_importing_project_modules_creates_no_files_or_handlers(tmp_path) -> None:
    code = ("import logging, os, utils, pipeline, evaluate_kitti_velocity\n"
            "print(len(logging.getLogger().handlers), sorted(os.listdir('.')))\n")
    res = _run_python(code, tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == "0 []"
    assert list(tmp_path.iterdir()) == []


def test_unconfigured_scripts_still_print_info_to_stdout(tmp_path) -> None:
    code = ("from utils import get_logger\n"
            "get_logger('demo').info('progress message')\n"
            "get_logger('demo').debug('hidden debug')\n")
    res = _run_python(code, tmp_path)
    assert res.returncode == 0, res.stderr
    assert "| INFO     | demo | progress message" in res.stdout
    assert "hidden debug" not in res.stdout and list(tmp_path.iterdir()) == []


def test_configure_logging_from_config_is_idempotent_and_writes_the_configured_file(tmp_path) -> None:
    code = ("import logging\n"
            "from utils import configure_logging_from_config, get_logger, load_config\n"
            f"cfg = load_config({str(ROOT / 'config.yaml')!r})\n"
            "cfg.logging['log_file'] = 'out/sub/run.log'\n"
            "cfg.logging['level'] = 'debug'\n"
            "log = get_logger('demo')\n"
            "for _ in range(3):\n"
            "    configure_logging_from_config(cfg)\n"
            "log.debug('one debug line')\n"
            "print('handlers', len(logging.getLogger().handlers))\n")
    res = _run_python(code, tmp_path)
    assert res.returncode == 0, res.stderr
    assert "handlers 2" in res.stdout
    assert res.stdout.count("one debug line") == 1
    log_text = (tmp_path / "out" / "sub" / "run.log").read_text()
    assert log_text.count("one debug line") == 1


def test_configure_logging_without_file(tmp_path) -> None:
    code = ("from utils import configure_logging, get_logger\n"
            "configure_logging(log_to_file=False)\n"
            "get_logger('demo').info('console only')\n")
    res = _run_python(code, tmp_path)
    assert res.returncode == 0 and res.stdout.count("console only") == 1
    assert list(tmp_path.iterdir()) == []


# =============================================================================
# pipeline.run_weather_classification
# =============================================================================


@pytest.fixture(scope="module")
def no_weather_pipeline():
    from pipeline import PerceptionPipeline

    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.device.preference = TEST_DEVICE
    cfg.pipeline["run_weather_classification"] = False
    return PerceptionPipeline(cfg)


def test_disabled_weather_classification_reports_unknown_with_neutral_adjustments(no_weather_pipeline) -> None:
    from pipeline import WEATHER_UNKNOWN, render_weather_card_html

    assert no_weather_pipeline.weather_classifier is None
    for seed in range(3):
        out = no_weather_pipeline.process(generate_synthetic_frame(640, 360, seed=seed), timestamp=seed / 30)
        assert out.weather.label == out.weather_raw.label == WEATHER_UNKNOWN
        assert out.weather.confidence == 0.0 and out.weather.probabilities == {}
        assert out.fusion.weather_label == WEATHER_UNKNOWN
    assert no_weather_pipeline.fusion_engine.get_weather_modifier(WEATHER_UNKNOWN, "camera_detection") == 1.0
    assert no_weather_pipeline.fusion_engine.get_weather_modifier(WEATHER_UNKNOWN, "camera_depth") == 1.0
    base = dict(no_weather_pipeline.config.risk.ttc_thresholds_s)
    assert no_weather_pipeline.risk_engine.get_weather_adjusted_thresholds(WEATHER_UNKNOWN) == base
    html = render_weather_card_html(out)
    assert "UNKNOWN" in html and "Classification disabled" in html


# =============================================================================
# Packaging (setup.py)
# =============================================================================


def _setup_py_modules() -> list:
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PY_MODULES" for t in node.targets) \
                and isinstance(node.value, ast.List):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("setup.py has no literal PY_MODULES list")


# Root-level .py files that are deliberately not installed.
NOT_PACKAGED = {"setup", "fix_run_py", "models_backup_before_weather", "test_risk_runtime"}


def test_setup_py_lists_every_module_and_its_local_imports() -> None:
    listed = set(_setup_py_modules())
    root_modules = {p.stem for p in ROOT.glob("*.py")}
    assert all((ROOT / f"{m}.py").exists() for m in listed), sorted(m for m in listed if not (ROOT / f"{m}.py").exists())
    unlisted = sorted(m for m in root_modules - listed - NOT_PACKAGED if not m.startswith("tests"))
    assert unlisted == [], f"add to setup.py PY_MODULES (or NOT_PACKAGED): {unlisted}"
    for module in listed:
        tree = ast.parse((ROOT / f"{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for name in names:
                top = name.split(".")[0]
                if top in root_modules:
                    assert top in listed, f"{module}.py imports {top}, which setup.py does not install"


def test_setup_py_installs_the_vendored_depth_anything_package() -> None:
    text = (ROOT / "setup.py").read_text()
    assert 'find_namespace_packages(include=["depth_anything_v2", "depth_anything_v2.*"])' in text
    assert (ROOT / "depth_anything_v2" / "dpt.py").exists() and (ROOT / "depth_anything_v2" / "dpt_metric.py").exists()


# =============================================================================
# Run recorder (run_logging.py)
# =============================================================================


def _fake_output(frame_index: int, metric: bool = True):
    """A PipelineOutput with one tracked car (BEV + velocity + risk) and one untracked person."""
    import numpy as np

    from camera import CameraModel
    from fusion import FusionResult
    from mapping import BEVObject, BEVResult
    from models import Detection, DetectionResult, DepthResult, WeatherResult
    from pipeline import PipelineOutput
    from relative_velocity import RelativeVelocity
    from risk_engine import ObjectRisk, RiskAssessment

    car = Detection([10.0, 20.0, 110.0, 90.0], 2, "car", 0.9, track_id=7)
    person = Detection([200.0, 30.0, 230.0, 120.0], 0, "person", 0.6, track_id=None)
    weather = WeatherResult("rain", 0.8, {"rain": 0.8, "clear": 0.2})
    bev = BEVResult(np.zeros((4, 4, 3), np.uint8), [BEVObject(7, "car", 1.5, 12.0, 0.9)]) if metric else None
    return PipelineOutput(
        frame=np.zeros((360, 640, 3), np.uint8), weather=weather,
        detections=DetectionResult([car, person]),
        depth=DepthResult(np.ones((360, 640), np.float32), mean_depth=1.0, kind="metric" if metric else "relative"),
        fusion=FusionResult({"camera_detection": 0.7}, {"camera_detection": 0.8}, 0.75, "rain"),
        bev=bev,
        relative_velocities={7: RelativeVelocity(7, 0.1, -2.0, 2.0025, True)} if metric else {},
        risk=RiskAssessment([ObjectRisk(7, "car", 12.09, 6.0, "SAFE")], "SAFE", "rain", {}) if metric else None,
        metrics={"fps": 20.0, "detection_latency_ms": 12.5}, total_latency_ms=50.0, weather_raw=weather,
        camera=CameraModel(500.0, 500.0, 320.0, 180.0, 640, 360, "test", False),
    )


@pytest.fixture()
def recorder_setup(tmp_path):
    from types import SimpleNamespace

    cfg = load_config(str(ROOT / "config.yaml"))
    for section, key in (("detection", "weights_path"), ("depth_estimation", "weights_path"),
                         ("depth_estimation", "metric_weights_path"), ("weather_classification", "weights_path")):
        f = tmp_path / f"{section}_{key}.bin"
        f.write_bytes(section.encode())
        cfg[section][key] = str(f)
    model = SimpleNamespace(_use_fallback=False, model=object(), device="cpu")
    pipeline = SimpleNamespace(depth_mode="metric", detector=model, depth_estimator=model,
                               weather_classifier=SimpleNamespace(_use_fallback=True, model=None, device="cpu"))
    return cfg, pipeline, tmp_path / "runs"


def _read_csv(path: Path) -> list:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_recorder_writes_manifest_and_csvs_and_marks_completion(recorder_setup) -> None:
    import hashlib
    import json

    from run_logging import FRAME_COLUMNS, INCOMPLETE_MARKER, OBJECT_COLUMNS, RunRecorder

    cfg, pipeline, root = recorder_setup
    rec = RunRecorder.start(cfg, pipeline, source="clip.mp4", fps=24.0, root=str(root),
                            config_path=str(ROOT / "config.yaml"), cli_overrides={"pipeline.strict": True},
                            command=["run.py", "--record-run"], run_id="test-run")
    run_dir = root / "test-run"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "recording" and (run_dir / INCOMPLETE_MARKER).exists()
    assert manifest["config"]["file_sha256"] == hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()
    assert manifest["config"]["resolved"]["depth_estimation"]["mode"] == cfg.depth_estimation.mode
    assert manifest["config"]["cli_overrides_in_memory"] == {"pipeline.strict": True}
    depth_ckpt = Path(cfg.depth_estimation.metric_weights_path)
    assert manifest["checkpoints"]["depth"]["sha256"] == hashlib.sha256(depth_ckpt.read_bytes()).hexdigest()
    assert manifest["models"]["weather"]["backend"] == "heuristic fallback"
    assert manifest["pipeline"]["depth_mode"] == "metric" and manifest["camera"]["calibrated"] is False
    assert "UNCALIBRATED" in manifest["outputs_note"]
    assert manifest["source"] == {"description": "clip.mp4", "fps": 24.0,
                                  "timestamps": "video timeline (frame_index / fps)"}
    for key in ("git", "versions", "device", "command", "started_utc"):
        assert key in manifest

    rec.record(_fake_output(0), 0, 0.0)
    rec.record(_fake_output(1), 1, 1 / 24)
    rec.close("complete")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["frames_recorded"] == 2
    assert manifest["objects_recorded"] == 4 and not (run_dir / INCOMPLETE_MARKER).exists()
    assert manifest["camera"]["first_frame"]["fx"] == 500.0

    frames, objects = _read_csv(run_dir / "frames.csv"), _read_csv(run_dir / "objects.csv")
    assert list(frames[0]) == FRAME_COLUMNS and list(objects[0]) == OBJECT_COLUMNS
    assert [f["frame_index"] for f in frames] == ["0", "1"]
    assert frames[0]["weather_label"] == "rain" and frames[0]["n_detections"] == "2"
    assert frames[0]["n_tracked"] == "1" and frames[0]["overall_risk_level"] == "SAFE"
    car, person = objects[0], objects[1]
    assert (car["track_id"], car["bev_z_m"], car["vz_m_s"], car["ttc_s"], car["risk_level"]) == \
        ("7", "12", "-2", "6", "SAFE")
    assert person["track_id"] == "" and person["bev_z_m"] == "" and person["risk_level"] == ""


def test_relative_depth_recording_leaves_metric_columns_empty(recorder_setup) -> None:
    import json

    from run_logging import RunRecorder

    cfg, pipeline, root = recorder_setup
    pipeline.depth_mode = "relative"
    rec = RunRecorder.start(cfg, pipeline, source=0, root=str(root), run_id="rel")
    rec.record(_fake_output(0, metric=False), 0, None)
    rec.close("complete")
    manifest = json.loads((root / "rel" / "manifest.json").read_text())
    assert "relative" in manifest["outputs_note"] and manifest["source"]["timestamps"] == "wall clock"
    objects = _read_csv(root / "rel" / "objects.csv")
    assert all(o["bev_x_m"] == o["distance_m"] == o["ttc_s"] == "" for o in objects)


def test_interrupted_recording_stays_marked_incomplete(recorder_setup) -> None:
    import json

    from run_logging import INCOMPLETE_MARKER, RunRecorder

    cfg, pipeline, root = recorder_setup
    rec = RunRecorder.start(cfg, pipeline, source="x", root=str(root), run_id="intr")
    rec.record(_fake_output(0), 0, 0.0)
    rec.close("interrupted")
    rec.close("complete")                                    # a second close is ignored
    manifest = json.loads((root / "intr" / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and (root / "intr" / INCOMPLETE_MARKER).exists()


def test_recording_failure_never_raises_and_disables_the_recorder(recorder_setup, caplog) -> None:
    import json

    from run_logging import INCOMPLETE_MARKER, RunRecorder

    cfg, pipeline, root = recorder_setup
    rec = RunRecorder.start(cfg, pipeline, source="x", root=str(root), run_id="fail")
    rec.record(_fake_output(0), 0, 0.0)
    rec._frames_fh.close()                                   # simulate e.g. a disk error on the next write
    with caplog.at_level(logging.ERROR):
        rec.record(_fake_output(1), 1, 0.1)                  # must not raise
        rec.record(_fake_output(2), 2, 0.2)                  # no-op once disabled
    assert not rec.active and "Run recording failed at frame 1" in caplog.text
    manifest = json.loads((root / "fail" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["frames_recorded"] == 1
    assert "frame 1" in manifest["error"] and (root / "fail" / INCOMPLETE_MARKER).exists()
    rec.close("complete")                                    # cannot turn a failed run into a complete one
    assert json.loads((root / "fail" / "manifest.json").read_text())["status"] == "failed"


def test_recorder_start_failure_returns_none(recorder_setup, caplog) -> None:
    from run_logging import RunRecorder

    cfg, pipeline, root = recorder_setup
    root.parent.mkdir(parents=True, exist_ok=True)
    blocker = root.parent / "not_a_dir"
    blocker.write_text("x")
    with caplog.at_level(logging.ERROR):
        assert RunRecorder.start(cfg, pipeline, source="x", root=str(blocker / "runs")) is None
    assert "continuing without recording" in caplog.text


def test_recorder_never_reuses_a_run_directory(recorder_setup) -> None:
    from run_logging import RunRecorder

    cfg, pipeline, root = recorder_setup
    first = RunRecorder.start(cfg, pipeline, source="x", root=str(root), run_id="same")
    second = RunRecorder.start(cfg, pipeline, source="x", root=str(root), run_id="same")
    assert first.run_dir.name == "same" and second.run_dir.name == "same-1"
    first.close()
    second.close()


def test_recorder_default_root_is_outputs_dir_runs(recorder_setup, tmp_path, monkeypatch) -> None:
    from run_logging import RunRecorder

    cfg, pipeline, _ = recorder_setup
    monkeypatch.chdir(tmp_path)
    rec = RunRecorder.start(cfg, pipeline, source="x", run_id="default-root",
                            config_path=str(ROOT / "config.yaml"))
    assert rec.run_dir == Path(cfg.paths.outputs_dir) / "runs" / "default-root"
    rec.close()


def test_recording_is_off_by_default(monkeypatch) -> None:
    import run

    monkeypatch.setattr(sys, "argv", ["run.py", "--synthetic"])
    assert run.parse_args().record_run is False


# =============================================================================
# Environment lock (requirements-lock.txt, scripts/env/)
# =============================================================================


def test_lock_covers_every_requirement_and_never_pins_a_container_build() -> None:
    import re

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lock = (ROOT / "requirements-lock.txt").read_text()
    pinned = {canonicalize_name(m.group(1)): m.group(2)
              for m in re.finditer(r"^([A-Za-z0-9._-]+)==(\S+)", lock, re.MULTILINE)}
    container = {canonicalize_name(m.group(1)): m.group(2)
                 for m in re.finditer(r"^# CONTAINER BUILD: ([A-Za-z0-9._-]+)==(\S+)", lock, re.MULTILINE)}
    assert not any("+" in v for v in pinned.values()), "local-version builds must not be pinned"
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        assert name in pinned or name in container, f"{req.name} missing from requirements-lock.txt"
        version = pinned.get(name) or container[name]
        assert req.specifier.contains(version, prereleases=True), f"lock {req.name}=={version} outside {req.specifier}"


def test_check_env_script_runs_without_installing_anything() -> None:
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "env" / "check_env.py")], capture_output=True,
                         text=True, timeout=120)
    assert res.returncode in (0, 1) and "[1]" in res.stdout and "Result:" in res.stdout
