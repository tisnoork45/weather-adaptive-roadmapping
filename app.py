"""Streamlit real-time dashboard for the Weather Adaptive Road Mapping system.

Autonomous-vehicle perception HUD: presentation only. Every value shown here
comes directly from PerceptionPipeline.process()'s PipelineOutput (or, for
the weather-adjusted risk thresholds, from the pipeline's own already-built
RiskEngine instance via its existing get_weather_adjusted_thresholds()
method) -- no detection, tracking, depth, weather, fusion, velocity, TTC, or
risk computation happens in this file.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time

import cv2
import streamlit as st

from pipeline import (
    PerceptionPipeline,
    render_bev_panel,
    render_camera_feed,
    render_dashboard_css,
    render_depth_panel,
    render_header,
    render_latency_html,
    render_metric_row_html,
    render_object_table_df,
    render_risk_guide_html,
    render_sensor_status_html,
    render_sensor_weights_html,
    render_status_badge_html,
    render_weather_card_html,
)
from utils import VideoStream, generate_synthetic_frame, get_logger, load_config

logger = get_logger(__name__)

st.set_page_config(page_title="Weather Adaptive Road Mapping", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading perception pipeline (models)...")
def load_pipeline() -> PerceptionPipeline:
    config = load_config("config.yaml")
    return PerceptionPipeline(config)


def get_frame_source(cfg, use_synthetic: bool):
    if use_synthetic:
        counter = {"i": 0}

        def _next_frame():
            counter["i"] += 1
            return generate_synthetic_frame(cfg.camera.width, cfg.camera.height, seed=counter["i"])

        return _next_frame, None

    stream = VideoStream(source=cfg.camera.source, width=cfg.camera.width, height=cfg.camera.height)
    try:
        stream.open()
    except IOError:
        logger.warning("Could not open configured camera source; falling back to synthetic demo.")
        return get_frame_source(cfg, use_synthetic=True)[0], None

    def _next_frame():
        ok, frame = stream.read()
        return frame if ok and frame is not None else None

    return _next_frame, stream


def main() -> None:
    pipeline = load_pipeline()
    cfg = pipeline.config

    # Injected exactly once per script run -- never re-injected inside the
    # per-frame loop, so it cannot itself accumulate.
    st.markdown(render_dashboard_css(), unsafe_allow_html=True)

    header_placeholder = st.empty()
    with header_placeholder.container():
        render_header(st, cfg.dashboard.title)
        status_badge_placeholder = st.empty()

    with st.sidebar:
        st.header("Controls")
        run_live = st.toggle("Run live pipeline", value=True)
        use_synthetic = st.checkbox("Use synthetic demo stream (no camera/dataset required)", value=True)
        st.markdown("---")
        st.subheader("Pipeline Stages")
        st.checkbox("Depth Estimation", value=cfg.pipeline.run_depth_estimation, disabled=True)
        st.checkbox("Weather Classification", value=cfg.pipeline.run_weather_classification, disabled=True)
        st.checkbox("Object Tracking", value=cfg.pipeline.run_tracking, disabled=True)
        st.checkbox("BEV Mapping", value=cfg.pipeline.run_bev_mapping, disabled=True)
        st.markdown("---")
        st.caption("Weather Adaptive Road Mapping for Autonomous Vehicles — Capstone Project")

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
        sensor_status_placeholder = st.empty()
    with row1_col2:
        bev_placeholder = st.empty()

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        weather_placeholder = st.empty()
    with row2_col2:
        depth_placeholder = st.empty()

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        risk_guide_placeholder = st.empty()
    with row3_col2:
        sensor_weights_placeholder = st.empty()

    with st.expander("SYSTEM DIAGNOSTICS", expanded=False):
        diagnostics_placeholder = st.empty()
    with st.expander("DETECTED OBJECTS", expanded=False):
        objects_placeholder = st.empty()

    if not run_live:
        st.info("Toggle 'Run live pipeline' in the sidebar to start streaming.")
        return

    next_frame_fn, stream = get_frame_source(cfg, use_synthetic)

    try:
        frame_budget = 1.0 / max(cfg.pipeline.target_fps, 1)
        for _ in range(100000):  # effectively unbounded; Streamlit reruns handle looping
            loop_start = time.perf_counter()
            frame = next_frame_fn()
            if frame is None:
                st.warning("End of video stream reached.")
                break

            # No timestamp passed: this source is always live (webcam) or
            # synthetic, never a prerecorded file, so PerceptionPipeline's
            # wall-clock fallback for relative-velocity timing is exactly
            # correct here (see run.py for the prerecorded-video case,
            # which needs an explicit video-timeline timestamp instead).
            output = pipeline.process(frame)

            status_badge_placeholder.markdown(render_status_badge_html(output), unsafe_allow_html=True)
            metric_row_placeholder.markdown(render_metric_row_html(output), unsafe_allow_html=True)

            annotated = render_camera_feed(output)
            camera_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True,
            )

            sensor_status_placeholder.markdown(
                render_sensor_status_html(output, cfg.pipeline), unsafe_allow_html=True,
            )
            bev_placeholder.image(
                cv2.cvtColor(render_bev_panel(output), cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True,
            )

            weather_placeholder.markdown(render_weather_card_html(output), unsafe_allow_html=True)
            depth_placeholder.image(
                cv2.cvtColor(render_depth_panel(output), cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True,
            )

            risk_guide_placeholder.markdown(
                render_risk_guide_html(output, pipeline.risk_engine), unsafe_allow_html=True,
            )
            sensor_weights_placeholder.markdown(render_sensor_weights_html(output), unsafe_allow_html=True)

            diagnostics_placeholder.markdown(render_latency_html(output), unsafe_allow_html=True)

            objects_df = render_object_table_df(output)
            if objects_df is not None:
                objects_placeholder.dataframe(objects_df, use_container_width=True, hide_index=True)
            else:
                objects_placeholder.info("No objects currently detected.")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, frame_budget - elapsed))
    finally:
        if stream is not None:
            stream.release()


if __name__ == "__main__":
    main()
