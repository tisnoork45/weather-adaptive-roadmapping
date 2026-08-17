"""Headless / OpenCV-window entry point for the perception pipeline.

Examples:
    python run.py --source 0 --show
    python run.py --source data/sample_videos/demo.mp4 --output outputs/videos/out.mp4
    python run.py --synthetic --show --max-frames 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from pipeline import PerceptionPipeline
from utils import (
    VideoStream, draw_detections, draw_hud, draw_lanes, generate_synthetic_frame,
    get_logger, load_config, overlay_depth_map,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Weather Adaptive Road Mapping pipeline.")
    parser.add_argument("--source", type=str, default=None,
                         help="Camera index, video file path, or RTSP URL. Defaults to config.yaml.")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic generated road scene.")
    parser.add_argument("--show", action="store_true", help="Show a live OpenCV preview window.")
    parser.add_argument("--output", type=str, default=None, help="Path to save an annotated output video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--overlay-depth", action="store_true", help="Overlay depth colormap on the feed.")
    return parser.parse_args()


def resolve_source(args, cfg):
    if args.synthetic:
        return None
    if args.source is not None:
        try:
            return int(args.source)
        except ValueError:
            return args.source
    return cfg.camera.source


def main() -> None:
    args = parse_args()
    cfg = load_config("config.yaml")

    logger.info("Building perception pipeline...")
    pipeline = PerceptionPipeline(cfg)

    source = resolve_source(args, cfg)
    stream, writer = None, None
    frame_index, synthetic_seed = 0, 0

    # A prerecorded video file (source is a path string) has its own
    # timeline, independent of how long this CPU pipeline takes to process
    # each frame (~3-4s/frame is common without a GPU). A live camera or the
    # synthetic demo stream has no separate timeline -- "now" IS the frame.
    # This distinction matters for relative-velocity estimation: using
    # wall-clock time as the inter-frame timestamp for a slow-processed
    # video would report ~3-4s between consecutive observations of the same
    # object, permanently exceeding config.yaml's velocity.max_gap_seconds
    # (2.0s) and making every velocity estimate invalid forever, even
    # though the video itself only advances ~0.04s per frame.
    is_prerecorded_video = isinstance(source, str)
    video_fps = None

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        if source is None:
            logger.info("Using synthetic demo stream (no real camera/video required).")
        else:
            stream = VideoStream(source=source, width=cfg.camera.width, height=cfg.camera.height).open()
            if is_prerecorded_video:
                reported_fps = stream.cap.get(cv2.CAP_PROP_FPS)
                video_fps = reported_fps if reported_fps and reported_fps > 1e-3 else (cfg.camera.fps or 30.0)
                logger.info("Prerecorded video source; using video-timeline timestamps at %.3f fps.", video_fps)

        while True:
            if source is None:
                synthetic_seed += 1
                frame = generate_synthetic_frame(cfg.camera.width, cfg.camera.height, seed=synthetic_seed)
                ok = True
            else:
                ok, frame = stream.read()

            if not ok or frame is None:
                logger.info("End of stream.")
                break

            frame_index += 1

            if is_prerecorded_video:
                # Video-timeline position, not wall-clock: frame N of a
                # 23.997fps video occurs at (N-1)/23.997 seconds into the
                # video, regardless of how long this process actually took
                # to analyze it.
                timestamp = (frame_index - 1) / video_fps
            else:
                # Live camera / synthetic stream: no separate timeline --
                # real elapsed time between calls IS the correct signal.
                timestamp = time.perf_counter()

            output = pipeline.process(frame, timestamp=timestamp)

            annotated = frame.copy()
            dets = output.detections
            track_ids = [d.track_id if d.track_id is not None else i for i, d in enumerate(dets.detections)]
            annotated = draw_detections(annotated, dets.boxes(), dets.labels(), dets.scores(), track_ids)
            # NEW: pass side identity so left/right get consistent colors
            # instead of the previous list-index-based coloring (where a
            # single detected line was always yellow regardless of side).
            annotated = draw_lanes(annotated, output.lanes.lanes, sides=output.lanes.sides)

            if args.overlay_depth:
                annotated = overlay_depth_map(annotated, output.depth.depth_map)

            annotated = draw_hud(
                annotated, weather=output.weather.label, fps=output.metrics.get("fps", 0.0),
                object_count=len(output.detections), latency_ms=output.total_latency_ms,
                weights=output.fusion.sensor_weights,
            )

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, cfg.camera.fps or 30, (w, h))
                writer.write(annotated)

            if args.show:
                cv2.imshow("Weather Adaptive Road Mapping", annotated)
                if output.bev is not None:
                    cv2.imshow("Bird's-Eye-View", output.bev.bev_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested by user.")
                    break

            if frame_index % 30 == 0:
                logger.info("Frame %d | FPS=%.1f | Weather=%s | Objects=%d | Latency=%.1fms",
                            frame_index, output.metrics.get("fps", 0.0), output.weather.label,
                            len(output.detections), output.total_latency_ms)

            if args.max_frames and frame_index >= args.max_frames:
                logger.info("Reached max-frames limit (%d).", args.max_frames)
                break
    finally:
        if stream is not None:
            stream.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
