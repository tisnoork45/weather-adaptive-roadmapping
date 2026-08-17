from pathlib import Path
import shutil

RUN = Path("run.py")

if not RUN.exists():
    raise SystemExit("run.py was not found. Run this script from the project folder.")

shutil.copy2(RUN, Path("run.py.before_auto_fix"))

s = RUN.read_text(encoding="utf-8")
main_pos = s.find("def main")
if main_pos < 0:
    raise SystemExit("Could not find def main in run.py.")

try_pos = s.find("    try:", main_pos)
finally_pos = s.find("    finally:", try_pos)
if try_pos < 0 or finally_pos < 0:
    raise SystemExit("Could not locate the try/finally block in run.py.")

new_block = '''    try:
        if source is None:
            logger.info("Using synthetic demo stream (no real camera/video required).")
        else:
            stream = VideoStream(
                source=source,
                width=cfg.camera.width,
                height=cfg.camera.height,
            ).open()

        while True:
            if source is None:
                synthetic_seed += 1
                frame = generate_synthetic_frame(
                    cfg.camera.width,
                    cfg.camera.height,
                    seed=synthetic_seed,
                )
                ok = True
            else:
                ok, frame = stream.read()

            if not ok or frame is None:
                logger.info("End of stream.")
                break

            # Wall-clock time is intentional for relative velocity.
            video_timestamp = time.perf_counter()

            frame_index += 1

            # Support pipeline versions with or without timestamp.
            try:
                output = pipeline.process(frame, timestamp=video_timestamp)
            except TypeError as exc:
                if "timestamp" not in str(exc):
                    raise
                output = pipeline.process(frame)

            annotated = frame.copy()
            dets = output.detections

            track_ids = [
                d.track_id if d.track_id is not None else i
                for i, d in enumerate(dets.detections)
            ]

            annotated = draw_detections(
                annotated,
                dets.boxes(),
                dets.labels(),
                dets.scores(),
                track_ids,
            )

            annotated = draw_lanes(
                annotated,
                output.lanes.lanes,
            )

            if args.overlay_depth:
                annotated = overlay_depth_map(
                    annotated,
                    output.depth.depth_map,
                )

            annotated = draw_hud(
                annotated,
                weather=output.weather.label,
                fps=output.metrics.get("fps", 0.0),
                object_count=len(output.detections),
                latency_ms=output.total_latency_ms,
                weights=output.fusion.sensor_weights,
                risk_level=(
                    output.risk.overall_risk_level
                    if output.risk
                    else None
                ),
            )

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        args.output,
                        fourcc,
                        cfg.camera.fps or 30,
                        (w, h),
                    )
                writer.write(annotated)

            if args.show:
                cv2.imshow(
                    "Weather Adaptive Road Mapping",
                    annotated,
                )

                if output.bev is not None:
                    cv2.imshow(
                        "Bird's-Eye-View",
                        output.bev.bev_image,
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested by user.")
                    break

            if frame_index % 30 == 0:
                logger.info(
                    "Frame %d | FPS=%.1f | Weather=%s | Objects=%d | Latency=%.1fms",
                    frame_index,
                    output.metrics.get("fps", 0.0),
                    output.weather.label,
                    len(output.detections),
                    output.total_latency_ms,
                )

            if args.max_frames and frame_index >= args.max_frames:
                logger.info(
                    "Reached max-frames limit (%d).",
                    args.max_frames,
                )
                break

'''

RUN.write_text(s[:try_pos] + new_block + s[finally_pos:], encoding="utf-8")
print("run.py repaired successfully.")
print("Backup created: run.py.before_auto_fix")
