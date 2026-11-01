"""Dashboard (app.py and the render helpers in pipeline.py) correctness tests.

Run with:
    pytest tests_dashboard.py -v

* Render helpers with stub models (fast, any device): backend labels, relative vs
  metric depth and calibration wording, module weights, status badge, and the
  objects table's Track ID column.
* The Streamlit app itself (AppTest, real models), from a temporary working
  directory: the default video path, a missing video, pause/resume/stop/restart,
  the end of a video, recording status, and strict metric depth.

The videos are used only to exercise the user interface: no accuracy is measured.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from camera import CameraModel
from models import DEPTH_KIND_RELATIVE, DepthResult
from pipeline import (
    PerceptionModels, PerceptionPipeline, model_config_key, render_depth_calibration_html, render_metric_row_html,
    render_module_status_html, render_module_weights_html, render_object_table_df, render_risk_guide_html,
    render_status_badge_html,
)
from tests_regression import StubDepth, StubDetector, StubWeather, _frame
from utils import load_config

ROOT = Path(__file__).resolve().parent
ROAD = ROOT / "road.mp4"


def _load_app_module():
    if "dashboard_app" not in sys.modules:
        spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["dashboard_app"] = module            # dataclasses look their module up here
        spec.loader.exec_module(module)
    return sys.modules["dashboard_app"]


# =============================================================================
# Render helpers, stub models
# =============================================================================


class StubRelativeDepth(StubDepth):
    """Relative (unitless inverse) depth, as Depth Anything V2's relative model returns."""

    def _infer(self, k, frame):
        return DepthResult.from_map(np.full(frame.shape[:2], 1.0 + 0.01 * k, dtype=np.float32), 1.0,
                                    kind=DEPTH_KIND_RELATIVE)


@pytest.fixture()
def cfg():
    return load_config(str(ROOT / "config.yaml"))


def _pipeline(cfg, depth=None, fallback=False, weather=True) -> PerceptionPipeline:
    models = PerceptionModels(StubDetector(fallback=fallback), depth or StubDepth(fallback=fallback),
                              StubWeather(fallback=fallback) if weather else None, model_config_key(cfg))
    return PerceptionPipeline(cfg, models=models)


def _outputs(p: PerceptionPipeline, n: int = 12, camera=None):
    return [p.process(_frame(k), timestamp=k / 10.0, camera=camera) for k in range(n)]


def _tag(html: str, label: str) -> str:
    """The status tag shown for a module row."""
    m = re.search(re.escape(label) + r'.*?class="status-tag"[^>]*>([^<]+)<', html, re.S)
    assert m, f"no status row for {label!r}"
    return m.group(1)


def test_object_table_track_ids_are_one_nullable_integer_column(cfg) -> None:
    out = _outputs(_pipeline(cfg), n=4)[-1]                     # frame 3: one low-confidence, untracked car
    ids = [d.track_id for d in out.detections.detections]
    assert None in ids and any(i is not None for i in ids)
    table = render_object_table_df(out)
    assert str(table["Track ID"].dtype) == "Int64"
    assert [None if pd.isna(v) else int(v) for v in table["Track ID"]] == ids     # IDs preserved, untracked = <NA>
    pa.Table.from_pandas(table)                                                     # Arrow-serialisable as is
    with pytest.raises(pa.ArrowInvalid):                                            # the old int/"-" mix was not
        pa.Table.from_pandas(pd.DataFrame({"Track ID": [1, "-"]}))


def test_module_status_shows_each_models_real_backend(cfg) -> None:
    out = _outputs(_pipeline(cfg))[-1]
    html = render_module_status_html(out, cfg.pipeline)
    for label in ("Object Detection", "Depth Estimation", "Weather Classification"):
        assert _tag(html, label) == "MODEL INFERENCE"
    assert _tag(html, "Object Tracking") == "RUNNING"
    assert _tag(html, "BEV Mapping") == "RUNNING" and _tag(html, "Velocity / TTC / Risk") == "RUNNING"
    assert "configured backend (as loaded), not a guarantee that every frame's inference succeeded" in html

    out = _outputs(_pipeline(cfg, fallback=True))[-1]
    html = render_module_status_html(out, cfg.pipeline)
    for label in ("Object Detection", "Depth Estimation", "Weather Classification"):
        assert _tag(html, label) == "HEURISTIC FALLBACK"                  # never shown as a trained model
    assert "MODEL INFERENCE" not in html and "instead of the trained model" in html

    cfg.pipeline["run_weather_classification"] = False
    cfg.pipeline["run_depth_estimation"] = False
    cfg.pipeline["run_tracking"] = False
    out = _outputs(_pipeline(cfg, weather=False), n=2)[-1]
    html = render_module_status_html(out, cfg.pipeline)
    assert _tag(html, "Weather Classification") == "DISABLED" and _tag(html, "Depth Estimation") == "DISABLED"
    assert _tag(html, "Object Tracking") == "DISABLED"
    assert _tag(html, "BEV Mapping") == "INACTIVE"                        # enabled, but no depth to map
    assert "INACTIVE: enabled in config.yaml but not running (depth estimation is disabled" in html
    assert render_metric_row_html(out).count("DISABLED") == 1            # weather, not a fabricated label


def test_relative_depth_is_never_presented_as_metric(cfg) -> None:
    out = _outputs(_pipeline(cfg, depth=StubRelativeDepth()))[-1]
    assert out.bev is None and out.risk is None
    card = render_depth_calibration_html(out, cfg.pipeline)
    assert "RELATIVE" in card and "no metric scale" in card and "METRIC" not in card
    assert "Distance, relative speed and TTC: unavailable (relative depth has no metric scale)" in card
    status = render_module_status_html(out, cfg.pipeline)
    assert _tag(status, "BEV Mapping") == "INACTIVE" and _tag(status, "Velocity / TTC / Risk") == "INACTIVE"
    badge = render_status_badge_html(out, "PROCESSING", risk_note="relative depth has no metric scale")
    assert "ROAD STATUS: N/A" in badge and "risk not computed: relative depth has no metric scale" in badge
    guide = render_risk_guide_html(out, None, risk_note="relative depth has no metric scale")
    assert "Thresholds only: risk is not being computed" in guide


def test_metric_depth_labels_strictness_calibration_and_validation(cfg) -> None:
    out = _outputs(_pipeline(cfg))[-1]                                    # config camera: uncalibrated
    assert out.risk is not None and not out.camera.calibrated
    card = render_depth_calibration_html(out, cfg.pipeline)
    assert "METRIC (NON-STRICT)" in card and "not a range measurement" in card
    assert "UNCALIBRATED" in card and "not validated against ground truth" in card and "approximate" in card

    cfg.pipeline["strict"] = True
    K = np.array([[700.0, 0, 320.0], [0, 700.0, 180.0], [0, 0, 1]])
    camera = CameraModel.from_matrix(K, 640, 360, source="test calibration")
    out = _outputs(_pipeline(cfg), camera=camera)[-1]
    card = render_depth_calibration_html(out, cfg.pipeline)
    assert "METRIC (STRICT)" in card and "calibrated intrinsics (test calibration)" in card
    assert "UNCALIBRATED" not in card and "approximate" not in card
    assert "not validated against ground truth" in card                  # calibrated is still not validated


def test_module_weights_and_status_wording(cfg) -> None:
    out = _outputs(_pipeline(cfg))[-1]
    weights = render_module_weights_html(out)
    assert "ADAPTIVE MODULE WEIGHTS" in weights and "SENSOR" not in weights.upper().replace("SENSOR FUSION", "")
    assert "not physical sensor fusion" in weights and "no LiDAR, radar or IMU" in weights
    status = render_module_status_html(out, cfg.pipeline)
    assert "MODULE STATUS" in status and "SENSOR" not in status
    badge = render_status_badge_html(out, "PAUSED")
    assert "PAUSED" in badge and not re.search(r"(?<!IN)ACTIVE", badge)  # no fixed "ACTIVE" label


def test_playback_commands() -> None:
    app = _load_app_module()
    P, S, E, X, R = app.PLAYING, app.PAUSED, app.STOPPED, app.ENDED, app.ERROR
    assert app.apply_command(P, None) == (P, False)                       # ordinary rerun: continue
    assert app.apply_command(P, app.CMD_PAUSE) == (S, False)
    assert app.apply_command(S, app.CMD_PLAY) == (P, False)               # resume where it paused
    assert app.apply_command(P, app.CMD_STOP) == (E, True)
    for state in (E, X, R):
        assert app.apply_command(state, app.CMD_PLAY) == (P, True)         # play after stop/end/error: from frame 0
    assert app.apply_command(S, app.CMD_RESTART) == (P, True)
    assert app.apply_command(E, app.CMD_PAUSE) == (E, False)              # nothing to pause


def test_default_video_path_finds_the_existing_road_mp4(tmp_path, monkeypatch) -> None:
    if not ROAD.is_file():
        pytest.skip("road.mp4 not found")
    app = _load_app_module()
    cfg = load_config(str(ROOT / "config.yaml"))
    assert not isinstance(cfg.camera.source, str)                        # so the sample-video default applies
    monkeypatch.chdir(ROOT)
    assert app.default_video_path(cfg) == "road.mp4"
    monkeypatch.chdir(tmp_path)                                           # run from elsewhere: absolute path
    assert app.default_video_path(cfg) == str(ROAD)
    assert Path(app.default_video_path(cfg)).is_file()


# =============================================================================
# The Streamlit app (AppTest, real models)
# =============================================================================


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """A working directory with config.yaml and the weights, so the app writes only here."""
    (tmp_path / "config.yaml").write_bytes((ROOT / "config.yaml").read_bytes())
    (tmp_path / "weights").symlink_to(ROOT / "weights", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))
    return tmp_path


def _session(max_frames: int = 3):
    testing = pytest.importorskip("streamlit.testing.v1", reason="Streamlit AppTest is not available")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["max_frames"] = max_frames
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


def _video_frames(path: Path, n: int) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    return frames


def test_app_pause_resume_stop_and_restart(workdir) -> None:
    at = _ok(_session().run())
    p = at.session_state["pipeline"]
    assert at.session_state["playback"] == "playing" and at.session_state["position"] == 3
    assert at.button(key="btn_play").disabled and not at.button(key="btn_pause").disabled
    assert "PLAYING · source: synthetic demo stream · frame 3" in _texts(at.caption)

    _ok(at.run())                                                         # ordinary rerun: continues
    assert at.session_state["position"] == 6 and p.metrics.frame_count == 6

    _ok(at.button(key="btn_pause").click().run())                        # pause: nothing processed
    assert at.session_state["playback"] == "paused" and p.metrics.frame_count == 6
    assert "Paused after frame 6" in _texts(at.info) and at.button(key="btn_play").label == "▶ Resume"
    assert len(at.get("image")) == 2                                      # the last frame stays on screen
    _ok(at.run())                                                         # reruns while paused stay paused
    assert at.session_state["playback"] == "paused" and p.metrics.frame_count == 6

    _ok(at.button(key="btn_play").click().run())                         # resume: same stream, no reset
    assert at.session_state["position"] == 9 and p.metrics.frame_count == 9

    _ok(at.button(key="btn_stop").click().run())                         # stop: nothing shown or processed
    assert at.session_state["playback"] == "stopped" and at.session_state["position"] == 0
    assert "Stopped. ▶ Play starts" in _texts(at.info) and len(at.get("image")) == 0
    assert p.metrics.frame_count == 9
    _ok(at.button(key="btn_play").click().run())                         # play after stop: from the start
    assert at.session_state["position"] == 3 and p.metrics.frame_count == 3

    _ok(at.run())
    _ok(at.button(key="btn_restart").click().run())                      # restart: from the start
    assert at.session_state["position"] == 3 and p.metrics.frame_count == 3
    assert at.session_state["pipeline"] is p                              # one pipeline throughout


def test_app_video_file_plays_the_real_video_and_resumes_at_the_exact_frame(workdir) -> None:
    if not ROAD.is_file():
        pytest.skip("road.mp4 not found")
    at = _ok(_session().run())
    _ok(at.radio(key="source_kind").set_value("Video file").run())
    assert at.text_input(key="video_path").value == str(ROAD)             # the default finds the real file
    assert not at.error and not at.warning
    assert f"PLAYING · source: video file `{ROAD}` · frame 3 / " in _texts(at.caption)
    p = at.session_state["pipeline"]
    assert p.metrics.frame_count == 3                                     # source change: fresh pipeline state

    expected = _video_frames(ROAD, 6)
    assert np.array_equal(at.session_state["last_output"].frame, expected[2])
    _ok(at.button(key="btn_pause").click().run())
    _ok(at.button(key="btn_play").click().run())                         # resumes with frames 3, 4, 5
    assert at.session_state["position"] == 6 and p.metrics.frame_count == 6
    assert np.array_equal(at.session_state["last_output"].frame, expected[5])
    fps = cv2.VideoCapture(str(ROAD)).get(cv2.CAP_PROP_FPS)
    assert p.weather_smoother._last_t == pytest.approx(5 / fps)          # video timeline, not restarted

    table = at.dataframe[0].value                                          # real detections
    assert str(table["Track ID"].dtype) == "Int64" and table["Track ID"].notna().any()


def test_app_missing_video_shows_an_error_instead_of_synthetic_input(workdir) -> None:
    at = _ok(_session().run())
    p = at.session_state["pipeline"]
    _ok(at.radio(key="source_kind").set_value("Video file").run())
    frames_before = p.metrics.frame_count
    missing = str(workdir / "no_such_video.mp4")
    _ok(at.text_input(key="video_path").set_value(missing).run())
    assert at.session_state["playback"] == "error"
    assert "Video file not found" in _texts(at.error) and "Nothing is being processed" in _texts(at.error)
    assert p.metrics.frame_count == frames_before and len(at.get("image")) == 0   # no substitute stream
    if ROAD.is_file():
        _ok(at.text_input(key="video_path").set_value(str(ROAD)).run())  # fixing the path plays it
        assert at.session_state["playback"] == "playing" and at.session_state["position"] == 3


def test_app_end_of_video_then_play_starts_again(workdir) -> None:
    clip = workdir / "five_frames.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for k in range(5):
        writer.write(np.full((180, 320, 3), 40 * k, dtype=np.uint8))
    writer.release()
    at = _ok(_session(max_frames=10).run())
    _ok(at.radio(key="source_kind").set_value("Video file").run())
    _ok(at.text_input(key="video_path").set_value(str(clip)).run())
    assert at.session_state["playback"] == "ended" and at.session_state["position"] == 5
    assert "End of the video after 5 frames" in _texts(at.info)
    _ok(at.run())                                                         # stays ended on a rerun
    assert at.session_state["playback"] == "ended" and at.session_state["pipeline"].metrics.frame_count == 5
    at.session_state["max_frames"] = 3
    _ok(at.button(key="btn_play").click().run())                         # play: from the first frame again
    assert at.session_state["playback"] == "playing" and at.session_state["position"] == 3
    assert at.session_state["pipeline"].metrics.frame_count == 3


def test_app_recording_status_complete_and_failed(workdir, monkeypatch) -> None:
    at = _ok(_session().run())
    assert "Not recording." in _texts(at.sidebar.markdown)
    _ok(at.checkbox(key="record_run").check().run())
    recorder = at.session_state["recorder"]
    run_dir = Path(recorder.run_dir).resolve()
    assert run_dir.parent == workdir.resolve() / "outputs" / "runs"            # the working directory's outputs/
    assert f"● RECORDING** — 3 frames  \n`{run_dir}`" in _texts(at.sidebar.markdown)
    _ok(at.run())                                                         # a rerun keeps the same recording
    assert at.session_state["recorder"] is recorder and recorder.frames_recorded == 6
    _ok(at.button(key="btn_stop").click().run())                         # stop: closed as complete
    assert "recorder" not in at.session_state
    assert f"Last recording **COMPLETE** — 6 frames  \n`{run_dir}`" in _texts(at.sidebar.markdown)
    assert "Recording starts when playback runs." in _texts(at.sidebar.markdown)   # still ticked, stopped
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "complete"
    assert not (run_dir / "INCOMPLETE").exists()

    import run_logging

    def _broken(self, *args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(run_logging.RunRecorder, "_record", _broken)
    _ok(at.button(key="btn_play").click().run())                         # a new recording, which fails
    failed_dir = Path(at.session_state["recorder"].run_dir).resolve()
    assert failed_dir != run_dir and "RECORDING FAILED" in _texts(at.sidebar.markdown)
    _ok(at.checkbox(key="record_run").uncheck().run())
    assert "Last recording **FAILED**" in _texts(at.sidebar.markdown) and "marked INCOMPLETE" in _texts(
        at.sidebar.markdown)
    assert json.loads((failed_dir / "manifest.json").read_text())["status"] == "failed"
    assert (failed_dir / "INCOMPLETE").exists()


def test_app_strict_metric_depth_is_labelled_as_such(workdir) -> None:
    if not ROAD.is_file():
        pytest.skip("road.mp4 not found")
    text = (workdir / "config.yaml").read_text()
    text = re.sub(r'(?m)^(\s*mode:\s*)"relative"', r'\1"metric"', text, count=1)
    text = re.sub(r"(?m)^(\s*strict:\s*)false", r"\1true", text, count=1)
    (workdir / "config.yaml").write_text(text)                            # the temporary copy only
    at = _ok(_session().run())
    _ok(at.radio(key="source_kind").set_value("Video file").run())
    p = at.session_state["pipeline"]
    assert p.depth_mode == "metric" and p.pipeline_cfg.strict
    html = _texts(at.markdown)
    assert "METRIC (STRICT)" in html and "UNCALIBRATED" in html and "not validated against ground truth" in html
    assert "Object Detection" in html and "BIRD'S-EYE VIEW" not in html  # the BEV map itself is shown
    assert len(at.get("image")) == 3
    assert "depth mode metric (strict)" in _texts(at.caption)
