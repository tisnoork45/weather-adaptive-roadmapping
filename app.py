"""Streamlit real-time dashboard for the Weather Adaptive Road Mapping system.

Autonomous-vehicle perception HUD: presentation only. Every value shown here
comes directly from PerceptionPipeline.process()'s PipelineOutput (or, for
the weather-adjusted risk thresholds, from the pipeline's own already-built
RiskEngine instance via its existing get_weather_adjusted_thresholds()
method) -- no detection, tracking, depth, weather, fusion, velocity, TTC, or
risk computation happens in this file.

Playback (sidebar): Play/Resume, Pause, Stop and Restart. Any other rerun --
changing a widget, ticking "Record run" -- continues the same video or
synthetic stream from the next frame with the pipeline's state intact. The
stream starts again from its first frame (and the pipeline is reset) only on
Restart, on Play after Stop or the end of the video, or when the source
changes. A live camera cannot be resumed: every run opens it afresh.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from pipeline import (
    PerceptionModels,
    PerceptionPipeline,
    PipelineOutput,
    metric_unavailable_reason,
    model_config_key,
    render_bev_panel,
    render_camera_feed,
    render_dashboard_css,
    render_depth_calibration_html,
    render_depth_panel,
    render_header,
    render_latency_html,
    render_metric_row_html,
    render_module_status_html,
    render_module_weights_html,
    render_object_table_df,
    render_risk_guide_html,
    render_status_badge_html,
    render_unavailable_card_html,
    render_weather_card_html,
)
from run_logging import STATUS_COMPLETE, STATUS_FAILED, RunRecorder
from utils import VideoStream, configure_logging_from_config, generate_synthetic_frame, get_logger, load_config

logger = get_logger(__name__)

st.set_page_config(page_title="Weather Adaptive Road Mapping", page_icon="🚗", layout="wide")

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading perception models...")
def load_shared_models(model_key: str, _config) -> PerceptionModels:
    """The trained models, loaded once per process and shared by every browser
    session (inference is serialised per model). They hold no per-stream state."""
    return PerceptionModels.load(_config).thread_safe()


def session_pipeline() -> PerceptionPipeline:
    """This browser session's own pipeline: its own tracks, velocity histories,
    smoothing and metrics, on top of the shared models. Kept in st.session_state,
    so other sessions never see or change it."""
    if "pipeline" not in st.session_state:
        config = load_config("config.yaml")
        configure_logging_from_config(config)
        models = load_shared_models(model_config_key(config), config)
        st.session_state["pipeline"] = PerceptionPipeline(config, models=models)
        # Held by whichever script run is using this session's pipeline (see main()).
        st.session_state["pipeline_lock"] = threading.Lock()
    return st.session_state["pipeline"]


SOURCE_SYNTHETIC = "Synthetic demo stream"
SOURCE_VIDEO = "Video file"
SOURCE_CAMERA = "Camera (config.yaml)"

# The sample video ships at the repository root; older checkouts kept it under data/sample_videos/.
DEFAULT_VIDEO_CANDIDATES = ("road.mp4", "data/sample_videos/road.mp4")

# Playback states (st.session_state["playback"]) and the button commands that change them.
PLAYING, PAUSED, STOPPED, ENDED, ERROR = "playing", "paused", "stopped", "ended", "error"
CMD_PLAY, CMD_PAUSE, CMD_STOP, CMD_RESTART = "play", "pause", "stop", "restart"


def apply_command(state: str, command: Optional[str]) -> Tuple[str, bool]:
    """The playback state after a button press, and whether the stream must start
    again from its first frame (with the pipeline reset)."""
    if command == CMD_PLAY:
        return PLAYING, state in (STOPPED, ENDED, ERROR)
    if command == CMD_PAUSE:
        return (PAUSED if state == PLAYING else state), False
    if command == CMD_STOP:
        return STOPPED, True
    if command == CMD_RESTART:
        return PLAYING, True
    return state, False


def _queue_command(command: str) -> None:
    # Button callbacks run before the script and outside the session lock, so they
    # only queue the command; main() applies it while holding the lock.
    st.session_state["playback_command"] = command


class SourceUnavailable(Exception):
    """The selected source cannot be opened. The dashboard shows why; it never
    substitutes another source."""


@dataclass
class FrameSource:
    next_frame: Callable[[], Optional[np.ndarray]]
    stream: Optional[VideoStream]
    fps: Optional[float]          # video-timeline fps (prerecorded files only)
    frame_count: Optional[int]    # frames in a video file, if known
    start_frame: int              # index of the first frame next_frame() returns
    live: bool                    # a camera: cannot be paused and resumed


def _synthetic_source(cfg, start_frame: int = 0) -> FrameSource:
    counter = {"i": start_frame}

    def _next_frame():
        counter["i"] += 1
        return generate_synthetic_frame(cfg.camera.width, cfg.camera.height, seed=counter["i"])

    return FrameSource(_next_frame, None, None, None, start_frame, live=False)


def get_frame_source(cfg, source_kind: str, video_path: str = "", start_frame: int = 0) -> FrameSource:
    """Open the selected source, positioned at ``start_frame`` (video files and the
    synthetic stream; a camera always starts live). Raises SourceUnavailable.

    ``fps`` is set only for prerecorded files: the caller must then use
    video-timeline timestamps (frame_index / fps) instead of wall-clock time,
    exactly like run.py, or relative velocity / TTC / risk are computed
    against processing time rather than the video's own timeline.
    """
    if source_kind == SOURCE_SYNTHETIC:
        return _synthetic_source(cfg, start_frame)

    if source_kind == SOURCE_VIDEO:
        if not video_path.strip():
            raise SourceUnavailable("No video path entered.")
        if not Path(video_path).is_file():
            raise SourceUnavailable(f"Video file not found: `{video_path}` "
                                    f"(relative paths are resolved from `{Path.cwd()}`).")
        source = video_path
    else:
        source = cfg.camera.source
    stream = VideoStream(source=source, width=cfg.camera.width, height=cfg.camera.height)
    try:
        stream.open()
    except IOError as exc:
        stream.release()
        raise SourceUnavailable(f"Could not open source `{source}`.") from exc

    fps = frame_count = None
    if isinstance(source, str):
        reported_fps = stream.cap.get(cv2.CAP_PROP_FPS)
        fps = reported_fps if reported_fps and reported_fps > 1e-3 else (cfg.camera.fps or 30.0)
        frame_count = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        logger.info("Prerecorded video source; using video-timeline timestamps at %.3f fps.", fps)

    live = source_kind == SOURCE_CAMERA
    position = 0
    if start_frame and not live:
        stream.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        position = int(round(stream.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if position != start_frame:       # this file cannot seek exactly: start again from frame 0
            logger.warning("Could not seek %r to frame %d (got %d); restarting from frame 0.",
                           source, start_frame, position)
            stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            position = 0

    def _next_frame():
        ok, frame = stream.read()
        return frame if ok and frame is not None else None

    return FrameSource(_next_frame, stream, fps, frame_count, position, live)


def default_video_path(cfg) -> str:
    """config.yaml's camera.source when it is a file path; otherwise the first sample
    video that exists (relative to the working directory, else to this file)."""
    if isinstance(cfg.camera.source, str):
        return cfg.camera.source
    for rel in DEFAULT_VIDEO_CANDIDATES:
        if Path(rel).is_file():
            return rel
        if (APP_DIR / rel).is_file():
            return str(APP_DIR / rel)
    return DEFAULT_VIDEO_CANDIDATES[0]


def _source_label(source_kind: str, video_path: str, cfg) -> str:
    if source_kind == SOURCE_VIDEO:
        return f"video file `{video_path}`"
    if source_kind == SOURCE_CAMERA:
        return f"camera `{cfg.camera.source}`"
    return "synthetic demo stream"


# --- Recording -----------------------------------------------------------------
# One recording per continuous stream: kept open across pause/resume and reruns, and
# closed when the stream ends (Stop, end of video, source change, Restart, unticking
# "Record run"), or as failed if the dashboard loop crashes.


def _close_recording(status: str, error: Optional[str] = None) -> None:
    recorder = st.session_state.pop("recorder", None)
    if recorder is None:
        return
    recorder.close(status, error)          # no-op if it already stopped itself (status 'failed')
    st.session_state["last_recording"] = {
        "path": str(Path(recorder.run_dir).resolve()), "status": recorder.manifest.get("status", status),
        "frames": recorder.frames_recorded, "error": recorder.manifest.get("error"),
    }


def render_recording_status_md(requested: bool = False) -> str:
    """Sidebar recording status: where it is written, and how the last recording ended.
    ``requested``: "Record run" is ticked (a recording starts when frames are processed)."""
    recorder = st.session_state.get("recorder")
    if recorder is not None and recorder.active:
        return f"**● RECORDING** — {recorder.frames_recorded} frames  \n`{Path(recorder.run_dir).resolve()}`"
    if recorder is not None:                # stopped itself after an I/O error
        return (f"**RECORDING FAILED** — {recorder.manifest.get('error') or 'see the log'}  \n"
                f"`{Path(recorder.run_dir).resolve()}` (marked INCOMPLETE)")
    last = st.session_state.get("last_recording")
    waiting = "Recording starts when playback runs." if requested else "Not recording."
    if last is None:
        return waiting
    status = last["status"]
    if status == STATUS_COMPLETE:
        head = "Last recording **COMPLETE**"
    elif status == STATUS_FAILED:
        head = f"Last recording **FAILED** ({last.get('error') or 'see the log'}); marked INCOMPLETE"
    else:
        head = f"Last recording **{status.upper()}**; marked INCOMPLETE"
    return f"{waiting}  \n{head} — {last['frames']} frames  \n`{last['path']}`"


# --- Rendering ----------------------------------------------------------------


@dataclass
class Placeholders:
    status_badge: object
    status_line: object
    metric_row: object
    camera: object
    module_status: object
    bev: object
    weather: object
    depth: object
    depth_info: object
    risk_guide: object
    module_weights: object
    diagnostics: object
    objects: object


def render_output(ph: Placeholders, output: PipelineOutput, pipeline: PerceptionPipeline,
                  activity: str) -> None:
    """Fill every panel from one PipelineOutput. One atomic update per placeholder,
    so nothing accumulates across frames."""
    cfg = pipeline.config
    reason = metric_unavailable_reason(output, cfg.pipeline)

    ph.status_badge.markdown(render_status_badge_html(output, activity, risk_note=reason), unsafe_allow_html=True)
    ph.metric_row.markdown(render_metric_row_html(output), unsafe_allow_html=True)
    ph.camera.image(cv2.cvtColor(render_camera_feed(output), cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
    ph.module_status.markdown(render_module_status_html(output, cfg.pipeline), unsafe_allow_html=True)

    if output.bev is not None:
        ph.bev.image(cv2.cvtColor(render_bev_panel(output), cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
    else:
        ph.bev.markdown(render_unavailable_card_html(
            "BIRD'S-EYE VIEW", f"Not available: {reason}. The BEV map needs metric depth "
                               "(depth_estimation.mode: metric in config.yaml)."), unsafe_allow_html=True)

    ph.weather.markdown(render_weather_card_html(output), unsafe_allow_html=True)
    if output.model_status.get("depth") == "disabled":
        ph.depth.markdown(render_unavailable_card_html("DEPTH MAP", "Depth estimation is disabled in config.yaml."),
                          unsafe_allow_html=True)
    else:
        caption = ("Metric depth estimate (near = warm, far = cool)" if output.depth.is_metric
                   else "Relative depth: near/far order only, no metric scale")
        ph.depth.image(cv2.cvtColor(render_depth_panel(output), cv2.COLOR_BGR2RGB), channels="RGB",
                       width="stretch", caption=caption)
    ph.depth_info.markdown(render_depth_calibration_html(output, cfg.pipeline), unsafe_allow_html=True)

    ph.risk_guide.markdown(render_risk_guide_html(output, pipeline.risk_engine, risk_note=reason),
                           unsafe_allow_html=True)
    ph.module_weights.markdown(render_module_weights_html(output), unsafe_allow_html=True)
    ph.diagnostics.markdown(render_latency_html(output), unsafe_allow_html=True)

    objects_df = render_object_table_df(output)
    if objects_df is None:
        ph.objects.info("No objects currently detected.")
        return
    columns = list(objects_df.columns)
    if reason:      # distance / speed / TTC / risk do not exist this frame: hide those columns
        columns = ["Track ID", "Class", "Confidence"]
    ph.objects.dataframe(
        objects_df, width="stretch", hide_index=True, column_order=columns,
        column_config={"Track ID": st.column_config.NumberColumn(
            "Track ID", format="%d", help="Empty: detected but not (yet) tracked.")},
    )


def main() -> None:
    pipeline = session_pipeline()
    # With Streamlit's fast reruns, a widget change starts the new script run while the
    # previous one is still finishing its current frame (it stops at its next st.* call).
    # Both would use this session's pipeline and playback state, so each whole run --
    # including the reset, the frame loop and the cleanup -- holds the session's lock:
    # the new run waits (about one frame) until the previous run has stopped and
    # released it. The lock is per session; other sessions and the shared models are
    # not blocked by it.
    with st.session_state["pipeline_lock"]:
        _run_dashboard(pipeline)


def _run_dashboard(pipeline: PerceptionPipeline) -> None:
    cfg = pipeline.config
    ss = st.session_state
    state, from_start = apply_command(ss.get("playback", PLAYING), ss.pop("playback_command", None))
    if from_start:
        ss["position"] = 0
        _close_recording(STATUS_COMPLETE)

    # Injected exactly once per script run -- never re-injected inside the
    # per-frame loop, so it cannot itself accumulate.
    st.markdown(render_dashboard_css(), unsafe_allow_html=True)

    header_placeholder = st.empty()
    with header_placeholder.container():
        render_header(st, cfg.dashboard.title)
        status_badge_placeholder = st.empty()
    status_line_placeholder = st.empty()
    message_placeholder = st.empty()

    with st.sidebar:
        st.header("Controls")
        source_kind = st.radio("Input source", [SOURCE_SYNTHETIC, SOURCE_VIDEO, SOURCE_CAMERA], index=0,
                               key="source_kind")
        video_path = ""
        if source_kind == SOURCE_VIDEO:
            video_path = st.text_input("Video path", value=default_video_path(cfg), key="video_path")

        # A new source is a new stream: it starts from its first frame.
        stream_key = (source_kind, video_path if source_kind == SOURCE_VIDEO else "")
        if ss.get("stream_key") != stream_key:
            if "stream_key" in ss:
                _close_recording(STATUS_COMPLETE)
                if state in (ENDED, ERROR):
                    state = PLAYING
            ss["stream_key"], ss["position"] = stream_key, 0
            ss.pop("last_output", None)

        st.subheader("Playback")
        col1, col2 = st.columns(2)
        col1.button("▶ Resume" if state == PAUSED else "▶ Play", key="btn_play", width="stretch",
                    disabled=state == PLAYING, on_click=_queue_command, args=(CMD_PLAY,))
        col2.button("⏸ Pause", key="btn_pause", width="stretch", disabled=state != PLAYING,
                    on_click=_queue_command, args=(CMD_PAUSE,))
        col1.button("⏹ Stop", key="btn_stop", width="stretch", disabled=state == STOPPED,
                    on_click=_queue_command, args=(CMD_STOP,))
        col2.button("⟲ Restart", key="btn_restart", width="stretch",
                    on_click=_queue_command, args=(CMD_RESTART,))
        st.caption("Pause and Resume continue from the same frame. Restart, or Play after Stop or "
                   "the end of the video, starts again from the first frame with a fresh pipeline "
                   "state. A live camera always starts afresh.")

        st.markdown("---")
        record_run = st.checkbox("Record run", value=False, key="record_run",
                                 help="Write a manifest and per-frame/per-object CSVs to outputs/runs/ "
                                      "(paths.outputs_dir in config.yaml). Closing the browser or "
                                      "stopping the server while recording leaves the run marked "
                                      "INCOMPLETE. With a live camera, every rerun starts a new "
                                      "recording.")
        recording_placeholder = st.empty()

        st.markdown("---")
        st.subheader("Configuration")
        pipe = cfg.pipeline
        depth_mode = pipeline.depth_mode + (" (strict)" if pipe.get("strict", False) else " (non-strict)")

        def _on(flag: bool) -> str:
            return "on" if flag else "off"

        st.caption(f"From config.yaml (read-only): depth mode {depth_mode}; depth estimation "
                   f"{_on(pipe.run_depth_estimation)}, weather classification "
                   f"{_on(pipe.run_weather_classification)}, tracking {_on(pipe.run_tracking)}, "
                   f"BEV mapping {_on(pipe.run_bev_mapping)}. What actually runs each frame is shown "
                   "in MODULE STATUS.")
        st.markdown("---")
        st.caption("Weather Adaptive Road Mapping for Autonomous Vehicles — Capstone Project")

    if not record_run:
        _close_recording(STATUS_COMPLETE)
    if state in (STOPPED, ERROR):
        _close_recording(STATUS_COMPLETE)
    ss["playback"] = state
    recording_placeholder.markdown(render_recording_status_md(record_run))

    # --- Stable placeholders, created exactly ONCE, before the loop -------
    # Every one of these gets exactly one atomic update call per frame
    # (.markdown / .image / .dataframe), never several sequential st.*
    # calls into the same container -- this is what guarantees no
    # per-frame accumulation, by construction.
    metric_row_placeholder = st.empty()

    st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
    camera_placeholder = st.empty()

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        module_status_placeholder = st.empty()
    with row1_col2:
        bev_placeholder = st.empty()

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        weather_placeholder = st.empty()
    with row2_col2:
        depth_placeholder = st.empty()
        depth_info_placeholder = st.empty()

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        risk_guide_placeholder = st.empty()
    with row3_col2:
        module_weights_placeholder = st.empty()

    with st.expander("SYSTEM DIAGNOSTICS", expanded=False):
        diagnostics_placeholder = st.empty()
    with st.expander("DETECTED OBJECTS", expanded=False):
        objects_placeholder = st.empty()

    ph = Placeholders(status_badge_placeholder, status_line_placeholder, metric_row_placeholder,
                      camera_placeholder, module_status_placeholder, bev_placeholder, weather_placeholder,
                      depth_placeholder, depth_info_placeholder, risk_guide_placeholder,
                      module_weights_placeholder, diagnostics_placeholder, objects_placeholder)
    label = _source_label(source_kind, video_path, cfg)
    position = ss.get("position", 0)

    if state != PLAYING:
        last = ss.get("last_output")
        if state == PAUSED:
            message_placeholder.info(f"Paused after frame {position} of the {label}. ▶ Resume continues "
                                     "from the next frame (a live camera starts afresh).")
        elif state == STOPPED:
            message_placeholder.info(f"Stopped. ▶ Play starts the {label} from the first frame.")
        elif state == ENDED:
            message_placeholder.info(f"End of the video after {position} frames. ▶ Play starts it again "
                                     "from the first frame.")
        else:
            message_placeholder.error(ss.get("source_error", "The source could not be opened."))
        status_line_placeholder.caption(f"{state.upper()} · source: {label} · frames processed: {position}")
        if last is not None and state in (PAUSED, ENDED):
            render_output(ph, last, pipeline, state.upper())
        return

    try:
        source = get_frame_source(cfg, source_kind, video_path, start_frame=position)
    except SourceUnavailable as exc:
        ss["playback"], ss["source_error"] = ERROR, f"{exc} Nothing is being processed."
        _close_recording(STATUS_COMPLETE)
        message_placeholder.error(ss["source_error"])
        status_line_placeholder.caption(f"ERROR · source: {label}")
        return

    stream = source.stream
    if source.start_frame == 0:
        # A stream that starts from its first frame (new source, Restart, Play after
        # Stop/end, a live camera, or a file that could not seek) starts with fresh
        # temporal state: no tracks or velocities carried over from before.
        if position != 0:
            _close_recording(STATUS_COMPLETE)
            if not source.live:
                message_placeholder.warning(f"Could not resume the {label} at frame {position}; "
                                            "started again from the first frame.")
        pipeline.reset()
        ss["position"] = 0
        ss.pop("last_output", None)

    if record_run and ss.get("recorder") is None:
        source_desc = {SOURCE_SYNTHETIC: "synthetic demo stream", SOURCE_VIDEO: video_path}.get(
            source_kind, cfg.camera.source)
        recorder = RunRecorder.start(cfg, pipeline, source=source_desc, fps=source.fps,
                                     command=["streamlit", "run", "app.py"])
        if recorder is None:
            message_placeholder.warning("Recording could not start (see the log); processing continues "
                                        "without recording.")
        else:
            ss["recorder"] = recorder
    recording_placeholder.markdown(render_recording_status_md(record_run))
    total = f" / {source.frame_count}" if source.frame_count else ""

    try:
        frame_budget = 1.0 / max(cfg.pipeline.target_fps, 1)
        # Effectively unbounded; Streamlit reruns handle looping. A "max_frames" entry in
        # session state (set only by headless tests) stops this run early.
        max_frames = int(ss.get("max_frames", 0)) or 100000
        for _ in range(max_frames):
            loop_start = time.perf_counter()
            frame = source.next_frame()
            if frame is None:
                ss["playback"] = ENDED
                _close_recording(STATUS_COMPLETE)
                recording_placeholder.markdown(render_recording_status_md(record_run))
                message_placeholder.info(f"End of the video after {ss['position']} frames. ▶ Play starts "
                                         "it again from the first frame.")
                status_line_placeholder.caption(f"ENDED · source: {label} · frames processed: {ss['position']}")
                break

            # Video files: video-timeline timestamp (see run.py). Live camera /
            # synthetic: None, so the pipeline's wall-clock fallback is used,
            # which is correct when "now" is the frame.
            frame_index = ss["position"]
            timestamp = frame_index / source.fps if source.fps else None
            output = pipeline.process(frame, timestamp=timestamp)
            recorder = ss.get("recorder")
            if recorder is not None:
                recorder.record(output, frame_index, timestamp)   # reads the output only; never raises
            # Saved before any st.* call (where a rerun can stop this run), so the
            # position always matches the frames the pipeline has processed.
            ss["position"], ss["last_output"] = frame_index + 1, output

            status_line_placeholder.caption(f"PLAYING · source: {label} · frame {frame_index + 1}{total}")
            render_output(ph, output, pipeline, "PROCESSING")
            if recorder is not None:
                recording_placeholder.markdown(render_recording_status_md(record_run))

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, frame_budget - elapsed))
    except Exception as exc:
        # Streamlit signals reruns/stops with exceptions (plain Exception subclasses in
        # some versions) -- those are control flow, not crashes: the stream and any
        # recording continue in the next run. A real error ends the recording as failed.
        if type(exc).__name__ not in ("RerunException", "StopException"):
            logger.exception("Dashboard pipeline loop crashed.")
            _close_recording(STATUS_FAILED, f"dashboard loop error: {exc!r}")
            ss["playback"], ss["source_error"] = ERROR, f"The dashboard loop crashed: {exc!r}"
        raise
    finally:
        if stream is not None:
            stream.release()


if __name__ == "__main__":
    main()
