"""CARLA simulator bridge for the Weather Adaptive Road Mapping pipeline.

Spawns an ego vehicle with an attached RGB camera in a running CARLA server,
sets a weather preset, and feeds the resulting frames directly into the
`PerceptionPipeline` for live testing of the weather-adaptive fusion logic.

Requires a running CARLA server and `pip install carla==<matching version>`.

Usage:
    python carla_sim.py --host 127.0.0.1 --port 2000 --weather rain --show --max-frames 500
"""

from __future__ import annotations

import argparse
import queue
import sys

import cv2
import numpy as np

from pipeline import PerceptionPipeline
from utils import draw_detections, draw_hud, draw_lanes, get_logger, load_config

logger = get_logger(__name__)

WEATHER_PRESETS = {
    "clear": "ClearNoon", "rain": "HardRainNoon", "fog": "CloudyNoon",
    "snow": "ClearNoon", "night": "ClearNight", "overcast": "CloudyNoon",
}


def build_carla_weather(carla_module, condition: str):
    weather = getattr(carla_module.WeatherParameters, WEATHER_PRESETS.get(condition, "ClearNoon"))
    if condition == "fog":
        weather.fog_density, weather.fog_distance, weather.fog_falloff = 80.0, 10.0, 1.0
    elif condition == "snow":
        weather.precipitation = weather.precipitation_deposits = 60.0
        weather.wetness = 40.0
    return weather


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream CARLA simulator frames into the pipeline.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--weather", type=str, default="clear", choices=list(WEATHER_PRESETS.keys()))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=500)
    args = parser.parse_args()

    try:
        import carla  # type: ignore
    except ImportError:
        logger.error("The 'carla' package is not installed. Install the version matching your server, "
                      "e.g.: pip install carla==0.9.15")
        sys.exit(1)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(build_carla_weather(carla, args.weather))
    logger.info("Set CARLA weather preset for condition: %s", args.weather)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("No spawn points available in the current CARLA map.")
        sys.exit(1)

    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", "90")
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=1.55)), attach_to=vehicle)

    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _on_image(image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        frame_queue.put(array[:, :, :3].copy())

    camera.listen(_on_image)

    cfg = load_config("config.yaml")
    pipeline = PerceptionPipeline(cfg)

    logger.info("Streaming CARLA frames into the perception pipeline. Press 'q' to quit.")
    frame_count = 0
    try:
        while frame_count < args.max_frames:
            try:
                frame = frame_queue.get(timeout=5.0)
            except queue.Empty:
                logger.warning("No frame received from CARLA within timeout; stopping.")
                break

            frame_count += 1
            output = pipeline.process(frame)

            annotated = frame.copy()
            dets = output.detections
            track_ids = [d.track_id if d.track_id is not None else i for i, d in enumerate(dets.detections)]
            annotated = draw_detections(annotated, dets.boxes(), dets.labels(), dets.scores(), track_ids)
            annotated = draw_lanes(annotated, output.lanes.lanes)
            annotated = draw_hud(annotated, weather=output.weather.label, fps=output.metrics.get("fps", 0.0),
                                  object_count=len(output.detections), latency_ms=output.total_latency_ms,
                                  weights=output.fusion.sensor_weights)

            if args.show:
                cv2.imshow("CARLA - Weather Adaptive Road Mapping", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 30 == 0:
                logger.info("CARLA frame %d | weather=%s | objects=%d",
                            frame_count, output.weather.label, len(output.detections))
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        if args.show:
            cv2.destroyAllWindows()
        logger.info("CARLA session ended after %d frames.", frame_count)


if __name__ == "__main__":
    main()
