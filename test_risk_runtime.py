import cv2

from pipeline import PerceptionPipeline
from utils import load_config


cfg = load_config("config.yaml")
pipeline = PerceptionPipeline(cfg)

cap = cv2.VideoCapture(r"data/sample_videos/road.mp4")

print("\n--- 5 FRAME RISK TEST ---\n")

for i in range(5):
    ok, frame = cap.read()

    if not ok:
        print("Could not read frame", i + 1)
        break

    output = pipeline.process(frame)

    print(f"FRAME {i + 1}")
    print("  objects:", len(output.detections))
    print("  track IDs:", list(output.relative_velocities.keys()))
    print(
        "  velocities:",
        {
            k: {
                "valid": v.valid,
                "vz": round(v.vz_m_s, 3),
            }
            for k, v in output.relative_velocities.items()
        },
    )
    print("  risk:", output.risk.overall_risk_level if output.risk else None)
    print(
        "  risk objects:",
        len(output.risk.per_object) if output.risk else 0,
    )
    print()

cap.release()