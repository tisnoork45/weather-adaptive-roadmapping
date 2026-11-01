"""Regression and runtime-state tests.

Run with:
    pytest tests_regression.py -v

* Golden checks (road.mp4 against stored goldens; KITTI 0000 against the frozen
  Task 5 cache; reset/shared-model determinism with the real models). These run
  scripts/regression/golden.py in a subprocess, so global torch settings changed
  by other tests cannot affect bit-exact results, and they skip with the reason
  when the video, goldens, KITTI data or a GPU is unavailable.
* Runtime state with stub models (fast, any device): PerceptionPipeline.reset(),
  independent pipelines and trackers, models shared across threads, backend status.
* Entry points: `run.py --synthetic` and the Streamlit app (two sessions), each run
  from a temporary working directory so nothing is written into the repository.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from models import (
    DEPTH_KIND_HEURISTIC, DEPTH_KIND_METRIC, Detection, DetectionResult, DepthResult, WeatherResult,
)
from pipeline import (
    BACKEND_DISABLED, BACKEND_FALLBACK, BACKEND_NETWORK, PerceptionModels, PerceptionPipeline, model_config_key,
)
from tracking import ByteTracker
from utils import load_config

ROOT = Path(__file__).resolve().parent
GOLDEN_SCRIPT = ROOT / "scripts" / "regression" / "golden.py"

_spec = importlib.util.spec_from_file_location("regression_golden", GOLDEN_SCRIPT)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def _run_golden(*args: str, timeout: int = 900) -> str:
    """Run golden.py in a fresh process; skip on exit status 2, fail on 1."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-B", str(GOLDEN_SCRIPT), *args], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=timeout)
    report = [line for line in res.stdout.splitlines() if line.strip()]
    if res.returncode == golden.EXIT_SKIPPED:
        pytest.skip(report[-1] if report else "skipped by golden.py")
    assert res.returncode == golden.EXIT_OK, "\n".join(report[-25:]) + "\n" + res.stderr[-3000:]
    return res.stdout


# =============================================================================
# Golden-output regression (real models, GPU)
# =============================================================================


@pytest.mark.parametrize("mode", ["relative", "metric"])
def test_road_mp4_outputs_match_the_golden(mode) -> None:
    out = _run_golden("compare", "--mode", mode)
    assert ", 0 differences" in out


def test_kitti_first_50_frames_match_the_frozen_task5_cache(tmp_path) -> None:
    out = _run_golden("kitti", "--out", str(tmp_path / "kitti_cache"))
    assert "0 differences, depth identical: True" in out


def test_real_models_are_deterministic_after_reset_and_when_shared_across_threads() -> None:
    out = _run_golden("reset", "--mode", "metric", "--frames", "60")
    assert out.count(": 0 differences") == 3


# =============================================================================
# Runtime state with stub models
# =============================================================================

W, H = 640, 360


def _frame(k: int) -> np.ndarray:
    return np.full((H, W, 3), k % 256, dtype=np.uint8)      # the stubs read the frame index back


class _StubModel:
    """Counts infer() calls that overlap in time (two threads inside the same model at once)."""

    def __init__(self, fallback: bool = False, check_overlap: bool = False) -> None:
        self._use_fallback, self.model, self.device = fallback, object(), "cpu"
        self.check_overlap, self.active, self.overlaps = check_overlap, 0, 0
        self._count_lock = threading.Lock()                # protects the counters, not infer()
        self.rendezvous = None                             # see test_each_model_has_its_own_lock
        self._rendezvous_left = 0

    def infer(self, frame: np.ndarray):
        with self._count_lock:
            self.active += 1
            if self.active > 1:
                self.overlaps += 1
            wait = self._rendezvous_left > 0
            self._rendezvous_left -= wait
        try:
            if wait:                                       # the first two calls meet inside infer()
                self.rendezvous.wait(timeout=10)
            if self.check_overlap:
                time.sleep(0.002)                          # widen the window for overlapping calls
            return self._infer(int(frame[0, 0, 0]), frame)
        finally:
            with self._count_lock:
                self.active -= 1


class StubDetector(_StubModel):
    """A car approaching and growing, a static pedestrian, and an intermittent low-confidence object."""

    def _infer(self, k: int, frame: np.ndarray) -> DetectionResult:
        dets = [Detection([280.0 - k, 150.0 - k / 2, 360.0 + k, 230.0 + k], 2, "car", 0.9),
                Detection([500.0, 160.0, 530.0, 240.0], 0, "person", 0.7)]
        if k % 3 == 0:
            dets.append(Detection([100.0, 200.0, 140.0, 230.0], 2, "car", 0.4))
        return DetectionResult(dets, inference_time_ms=1.0)


class StubDepth(_StubModel):
    """Metric depth; everything gets 0.4 m closer per frame."""

    def _infer(self, k: int, frame: np.ndarray) -> DepthResult:
        depth = np.full(frame.shape[:2], 40.0 - 0.4 * k, dtype=np.float32)
        return DepthResult.from_map(depth, 1.0, kind=DEPTH_KIND_HEURISTIC if self._use_fallback else DEPTH_KIND_METRIC)


class StubWeather(_StubModel):
    """Clear for 10 frames, then rain (exercises the temporal smoother)."""

    def _infer(self, k: int, frame: np.ndarray) -> WeatherResult:
        p_rain = 0.8 if k >= 10 else 0.1
        probs = {"clear": 1.0 - p_rain, "rain": p_rain, "fog": 0.0, "snow": 0.0, "night": 0.0, "overcast": 0.0}
        label = max(probs, key=probs.get)
        return WeatherResult(label, probs[label], probs, inference_time_ms=1.0)


@pytest.fixture()
def cfg():
    return load_config(str(ROOT / "config.yaml"))


def _stub_models(cfg, **kw) -> PerceptionModels:
    return PerceptionModels(StubDetector(**kw), StubDepth(**kw), StubWeather(**kw), model_config_key(cfg))


def _run(pipeline: PerceptionPipeline, frames=range(30), fps: float = 10.0) -> list:
    return [golden.dump_output(pipeline.process(_frame(k), timestamp=i / fps)) for i, k in enumerate(frames)]


def test_stub_stream_exercises_every_temporal_component(cfg) -> None:
    p = PerceptionPipeline(cfg, models=_stub_models(cfg))
    out = _run(p)
    assert {d[4] for d in out[-1]["dets"] if d[4] is not None} == {1, 2}
    assert any(v[3] for v in out[-1]["vel"].values())                       # valid velocities
    assert out[-1]["risk"] is not None and out[-1]["risk"][3]                # TTC / risk computed
    assert out[0]["weather"][0] == "clear" and out[-1]["weather"][0] == "rain"


def test_reset_restores_a_fresh_pipeline_exactly_without_reloading_models(cfg) -> None:
    models = _stub_models(cfg)
    p = PerceptionPipeline(cfg, models=models)
    first = _run(p)
    detector, depth, weather, risk_engine = p.detector, p.depth_estimator, p.weather_classifier, p.risk_engine
    p.reset()
    assert p.tracker.tracked == [] and p.tracker.lost == [] and p.tracker._next_id == 1
    assert p.velocity_estimator._history == {} and p.velocity_estimator._smoothed == {}
    assert p.weather_smoother._probs is None and p.fusion_engine._smoothed_weights is None
    assert p.confidence_estimator.stability_tracker._history == {}
    assert p.metrics.frame_count == 0 and p._frame_index == 0
    assert (p.detector, p.depth_estimator, p.weather_classifier, p.risk_engine) == \
        (detector, depth, weather, risk_engine) and p.models is models
    assert _run(p) == first
    assert _run(PerceptionPipeline(cfg, models=models)) == first             # == a brand-new pipeline


def test_two_pipelines_have_independent_ids_and_temporal_state(cfg) -> None:
    models = _stub_models(cfg)
    solo = _run(PerceptionPipeline(cfg, models=models))
    a, b = PerceptionPipeline(cfg, models=models), PerceptionPipeline(cfg, models=models)
    out_a, out_b = [], []
    for i in range(30):                                                       # interleaved frame by frame
        out_a.append(golden.dump_output(a.process(_frame(i), timestamp=i / 10.0)))
        out_b.append(golden.dump_output(b.process(_frame(29 - i), timestamp=i / 10.0)))
    assert out_a == solo
    assert {d[4] for d in out_b[0]["dets"] if d[4] is not None} == {1, 2}      # b numbers its own tracks
    a.reset()
    assert b.tracker._next_id > 1 and b.metrics.frame_count == 30             # resetting a leaves b alone
    assert a.tracker is not b.tracker and a.velocity_estimator is not b.velocity_estimator


def test_trackers_number_ids_independently() -> None:
    def dets():
        return [Detection([10.0, 10.0, 50.0, 50.0], 2, "car", 0.9), Detection([100.0, 10.0, 150.0, 60.0], 2, "car", 0.9)]

    t1, t2 = ByteTracker(), ByteTracker()
    assert [d.track_id for d in t1.update(dets())] == [1, 2]
    assert [d.track_id for d in t2.update(dets())] == [1, 2]
    t3 = ByteTracker()
    t3.update([Detection([300.0, 10.0, 350.0, 60.0], 2, "car", 0.9)])
    assert [d.track_id for d in t1.update(dets()[:1] + [Detection([400.0, 10.0, 450.0, 60.0], 2, "car", 0.9)])] \
        == [1, 3]


MODEL_ROLES = ("detector", "depth_estimator", "weather_classifier")


def _run_threads(pipelines) -> dict:
    results = {}
    threads = [threading.Thread(target=lambda i, p: results.__setitem__(i, _run(p)), args=(i, p))
               for i, p in enumerate(pipelines)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_shared_models_serialise_inference_and_threads_keep_their_own_state(cfg) -> None:
    models = _stub_models(cfg, check_overlap=True)
    solo = _run(PerceptionPipeline(cfg, models=models))
    shared = models.thread_safe()
    assert all(getattr(shared.thread_safe(), r) is getattr(shared, r) for r in MODEL_ROLES)   # idempotent
    results = _run_threads([PerceptionPipeline(cfg, models=shared) for _ in range(4)])
    assert all(results[i] == solo for i in range(4))
    for role in MODEL_ROLES:                                                  # never two infer() calls at once
        assert getattr(models, role).overlaps == 0, role


@pytest.mark.parametrize("unlocked", MODEL_ROLES)
def test_each_model_has_its_own_lock(cfg, unlocked) -> None:
    """Leave one model unwrapped: only that model sees overlapping calls. This shows the
    overlap check can fail, and that each model is serialised by its own lock.

    The unwrapped model makes its first two calls wait for each other inside infer(),
    so they overlap whatever the thread timing (otherwise the locked models before it
    can serialise the threads by chance and hide the overlap)."""
    models = _stub_models(cfg, check_overlap=True)
    shared = models.thread_safe()
    bare = getattr(models, unlocked)
    bare.rendezvous, bare._rendezvous_left = threading.Barrier(2), 2
    setattr(shared, unlocked, bare)                                           # this one without its lock
    _run_threads([PerceptionPipeline(cfg, models=shared) for _ in range(4)])
    assert getattr(models, unlocked).overlaps > 0
    for role in MODEL_ROLES:
        if role != unlocked:
            assert getattr(models, role).overlaps == 0, role


def test_pipeline_rejects_models_built_for_other_settings(cfg) -> None:
    models = _stub_models(cfg)
    cfg.depth_estimation["mode"] = "metric"
    with pytest.raises(ValueError, match="different model settings"):
        PerceptionPipeline(cfg, models=models)


def test_model_status_reports_each_backend(cfg) -> None:
    p = PerceptionPipeline(cfg, models=_stub_models(cfg))
    out = p.process(_frame(1), timestamp=0.0)
    assert out.model_status == {"detection": BACKEND_NETWORK, "depth": BACKEND_NETWORK, "weather": BACKEND_NETWORK}

    fallback = PerceptionModels(StubDetector(fallback=True), StubDepth(fallback=True), StubWeather(fallback=True),
                                model_config_key(cfg)).thread_safe()
    out = PerceptionPipeline(cfg, models=fallback).process(_frame(1), timestamp=0.0)
    assert out.model_status == {"detection": BACKEND_FALLBACK, "depth": BACKEND_FALLBACK, "weather": BACKEND_FALLBACK}

    cfg.pipeline["run_weather_classification"] = False
    cfg.pipeline["run_depth_estimation"] = False
    no_weather = PerceptionModels(StubDetector(), StubDepth(), None, model_config_key(cfg))
    p = PerceptionPipeline(cfg, models=no_weather)
    assert p.process(_frame(1), timestamp=0.0).model_status == \
        {"detection": BACKEND_NETWORK, "depth": BACKEND_DISABLED, "weather": BACKEND_DISABLED}
    assert p.model_status()["depth"] == BACKEND_DISABLED


# =============================================================================
# Entry points, from a temporary working directory
# =============================================================================


@pytest.fixture()
def workdir(tmp_path):
    """A working directory with config.yaml and the weights, so entry points write only here."""
    (tmp_path / "config.yaml").write_bytes((ROOT / "config.yaml").read_bytes())
    (tmp_path / "weights").symlink_to(ROOT / "weights", target_is_directory=True)
    return tmp_path


def test_run_py_synthetic_cli(workdir) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), "--synthetic", "--max-frames", "3"],
                         cwd=workdir, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "Using synthetic demo stream" in res.stdout and "Reached max-frames limit (3)." in res.stdout
    assert "Reached max-frames limit (3)." in (workdir / "logs" / "pipeline.log").read_text()
    assert not (workdir / "outputs").exists()                                 # no recording unless asked


def test_streamlit_sessions_have_independent_pipelines_on_shared_models(workdir, monkeypatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1", reason="Streamlit AppTest is not available")
    monkeypatch.chdir(workdir)
    monkeypatch.syspath_prepend(str(ROOT))

    def session(max_frames: int):
        at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.session_state["max_frames"] = max_frames
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    a, b = session(4), session(2)
    pa, pb = a.session_state["pipeline"], b.session_state["pipeline"]
    assert pa is not pb and pa.tracker is not pb.tracker and pa.velocity_estimator is not pb.velocity_estimator
    assert pa.models is pb.models and pa.detector is pb.detector              # one copy of the models
    assert (pa.metrics.frame_count, pb.metrics.frame_count) == (4, 2)
    assert len(a.get("image")) == 2                                           # camera and depth (BEV needs metric depth)
    assert any("ROAD STATUS" in m.value for m in a.markdown)
    lock_a = a.session_state["pipeline_lock"]
    assert lock_a is not b.session_state["pipeline_lock"]                    # one lock per session
    a.run()                                                                   # an ordinary rerun of A ...
    assert a.session_state["pipeline"] is pa and a.session_state["pipeline_lock"] is lock_a   # ... same objects
    assert pa.metrics.frame_count == 8 and a.session_state["position"] == 8  # ... continues A's stream
    assert pb.metrics.frame_count == 2                                        # B untouched
    assert not lock_a.locked()                                                # released when the run ends

    # A run that starts while the session lock is held (the previous run still finishing)
    # waits for it: no frame is processed until the lock is released.
    lock_a.acquire()
    rerun = threading.Thread(target=a.run)
    try:
        rerun.start()
        time.sleep(1.0)
        assert rerun.is_alive() and pa.metrics.frame_count == 8               # still waiting, state untouched
    finally:
        lock_a.release()
    rerun.join(timeout=300)
    assert not rerun.is_alive() and not a.exception
    assert pa.metrics.frame_count == 12 and pb.metrics.frame_count == 2       # then continued with 4 frames
