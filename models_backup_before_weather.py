"""Perception models: object detection, lane detection, depth estimation, and
weather classification.

Merged into one file since each individual wrapper is a few hundred lines
and this project's scale doesn't warrant a dedicated file/package per model.
Every model wrapper follows the same pattern: try to load the real
pretrained network; if unavailable, fall back to a deterministic classical
CV / heuristic implementation so the pipeline always runs end-to-end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import get_device, get_logger

logger = get_logger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

DEFAULT_WEATHER_CLASSES = ["clear", "rain", "fog", "snow", "night", "overcast"]


# =============================================================================
# Object Detection - YOLO11x
# =============================================================================


@dataclass
class Detection:
    box: List[float]           # [x1, y1, x2, y2] in pixel coordinates
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> List[float]:
        x1, y1, x2, y2 = self.box
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.detections)

    def boxes(self) -> List[List[float]]:
        return [d.box for d in self.detections]

    def labels(self) -> List[str]:
        return [d.class_name for d in self.detections]

    def scores(self) -> List[float]:
        return [d.confidence for d in self.detections]


class YOLODetector:
    """Object detector built on Ultralytics YOLO11x with a classical CV fallback."""

    def __init__(
        self,
        weights_path: str = "weights/yolo11x.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        classes_of_interest: Optional[Sequence[int]] = None,
        device_preference: str = "auto",
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.classes_of_interest = list(classes_of_interest) if classes_of_interest else None
        self.device = get_device(device_preference)
        self.model = None
        self._use_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            if not Path(self.weights_path).exists():
                logger.warning(
                    "YOLO weights not found at %s; attempting ultralytics auto-download. "
                    "Run download_models.py beforehand for offline use.", self.weights_path,
                )
                self.model = YOLO("yolo11x.pt")
            else:
                self.model = YOLO(self.weights_path)
            self.model.to(self.device)
            logger.info("Loaded YOLO11x model from %s on device %s", self.weights_path, self.device)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load YOLO11x (%s). Falling back to synthetic detector.", exc)
            self._use_fallback = True

    def infer(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        detections = self._fallback_infer(frame) if (self._use_fallback or self.model is None) \
            else self._real_infer(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(detections=detections, inference_time_ms=elapsed_ms)

    def _real_infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame, imgsz=self.input_size, conf=self.confidence_threshold,
            iou=self.iou_threshold, device=self.device, verbose=False,
        )
        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names if hasattr(result, "names") else dict(enumerate(COCO_CLASSES))

        for box, score, cls_id in zip(boxes, scores, class_ids):
            if self.classes_of_interest is not None and cls_id not in self.classes_of_interest:
                continue
            detections.append(Detection(
                box=[float(v) for v in box], class_id=int(cls_id),
                class_name=str(names.get(int(cls_id), f"class_{cls_id}")),
                confidence=float(score),
            ))
        return detections

    def _fallback_infer(self, frame: np.ndarray) -> List[Detection]:
        """Contour-based synthetic detector used only when YOLO weights are unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        min_area = (w * h) * 0.0015
        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            cls_name = "car" if aspect > 1.2 else "object"
            detections.append(Detection(
                box=[float(x), float(y), float(x + bw), float(y + bh)],
                class_id=2 if cls_name == "car" else 0, class_name=cls_name,
                confidence=float(min(0.95, 0.4 + area / (w * h))),
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:10]


# =============================================================================
# Lane Detection - Ultra-Fast Lane Detection V2 (row-anchor classification)
# =============================================================================


@dataclass
class LaneResult:
    lanes: List[List[Tuple[int, int]]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.lanes)


class UFLDv2Net(nn.Module):
    """Row-anchor-based lane detection network, loosely inspired by UFLD-v2's
    row-anchor classification formulation.

    IMPORTANT / HONEST DISCLOSURE: this is NOT the official UFLD-v2 network.
    The real Ultra-Fast-Lane-Detection-v2 (cfzd/Ultra-Fast-Lane-Detection-v2)
    architecture is called `parsingNet` and has a materially different
    structure (multi-stage feature extraction with an auxiliary segmentation
    head, hybrid row+column anchor classification, etc.). Its official
    checkpoints are only valid for that exact `parsingNet` class -- they are
    NOT loadable into this simplified sketch, even if you obtain a correct
    checkpoint file. This class is kept in the codebase as an architectural
    reference / stretch-goal for anyone who wants to vendor the official
    `parsingNet` implementation later (see README.md > "Lane Detection:
    Design Decision"). It is never loaded with pretrained weights by this
    project; `LaneDetector` always uses the classical CV fallback below.
    """

    def __init__(self, backbone: str = "resnet18", num_lanes: int = 4,
                 num_row_anchors: int = 72, num_col_anchors: int = 81) -> None:
        super().__init__()
        self.num_lanes = num_lanes
        self.num_row_anchors = num_row_anchors
        self.num_col_anchors = num_col_anchors

        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights=None)
            feat_dim = 512
        elif backbone == "resnet34":
            net = torchvision.models.resnet34(weights=None)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d((10, 25))
        out_dim = num_lanes * num_row_anchors * (num_col_anchors + 1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 10 * 25, 2048), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits.view(-1, self.num_lanes, self.num_row_anchors, self.num_col_anchors + 1)


class LaneDetector:
    """Lane detector. Uses a classical Canny + Hough-transform pipeline with
    temporal (frame-to-frame) smoothing as its documented, primary method.

    DESIGN DECISION (see README.md > "Lane Detection: Design Decision" for
    the full writeup): this project deliberately does NOT attempt to load a
    UFLD-v2 checkpoint into `UFLDv2Net`, because (a) `UFLDv2Net` is a
    simplified architecture sketch, not the official `parsingNet` the
    published checkpoints were trained for, so the weights would not be
    state-dict-compatible even if downloaded correctly, and (b) the official
    checkpoints are not published behind a stable, scriptable direct-download
    URL. Rather than silently loading a mismatched/garbage checkpoint (which
    would look like it "worked" while producing meaningless output), this
    class always uses the classical CV method below, which is fully
    deterministic, requires no training data or GPU, and is explainable --
    all reasonable properties for a capstone project to claim honestly.
    """

    def __init__(
        self, weights_path: str = "weights/ufld_v2_culane_res18.pth", backbone: str = "resnet18",
        input_width: int = 800, input_height: int = 320, num_lanes: int = 4,
        num_row_anchors: int = 72, num_col_anchors: int = 81,
        confidence_threshold: float = 0.4, device_preference: str = "auto",
        smoothing_alpha: float = 0.6,
    ) -> None:
        self.input_width, self.input_height = input_width, input_height
        self.num_lanes, self.num_row_anchors, self.num_col_anchors = num_lanes, num_row_anchors, num_col_anchors
        self.confidence_threshold = confidence_threshold
        self.device = get_device(device_preference)
        self.weights_path = weights_path
        self.model: UFLDv2Net = None
        # smoothing_alpha: weight given to the NEW frame's fitted line when
        # blending with the previous frame's line (EMA). Lower = smoother /
        # less jittery but slower to react to real lane changes.
        self.smoothing_alpha = smoothing_alpha
        self._prev_coeffs: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._use_fallback = True
        logger.info(
            "LaneDetector using classical Canny+Hough-transform CV pipeline "
            "with temporal smoothing (by design -- see README.md > 'Lane "
            "Detection: Design Decision'). No UFLD-v2 checkpoint is loaded."
        )

    def infer(self, frame: np.ndarray) -> LaneResult:
        start = time.perf_counter()
        lanes, confs = self._cv_fallback(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return LaneResult(lanes=lanes, confidences=confs, inference_time_ms=elapsed_ms)

    def _cv_fallback(self, frame: np.ndarray) -> Tuple[List[List[Tuple[int, int]]], List[float]]:
        """Classical Canny-edge + Hough-transform lane line estimator with
        frame-to-frame EMA smoothing of the fitted line coefficients, so a
        single noisy frame (shadow, glare, momentary occlusion) doesn't cause
        the drawn lane to jump or disappear -- a real, deterministic
        improvement over a naive per-frame-only Hough fit.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        mask = np.zeros_like(edges)
        roi = np.array([[
            (int(w * 0.05), h), (int(w * 0.45), int(h * 0.55)),
            (int(w * 0.55), int(h * 0.55)), (int(w * 0.95), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi / 180, threshold=50,
                                 minLineLength=40, maxLineGap=150)

        left_pts, right_pts = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue
                (left_pts if slope < 0 else right_pts).extend([(x1, y1), (x2, y2)])

        lanes, confidences = [], []
        for side, pts in (("left", left_pts), ("right", right_pts)):
            fitted, raw_count = self._fit_lane_line_smoothed(side, pts, h)
            if fitted:
                lanes.append(fitted)
                confidences.append(min(0.9, 0.3 + 0.05 * raw_count))
        return lanes, confidences

    def _fit_lane_line_smoothed(
        self, side: str, points: List[Tuple[int, int]], height: int
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Fit a line to `points` and EMA-blend its coefficients with the
        previous frame's fit for the same side, to reduce frame-to-frame jitter.
        """
        new_coeffs: Optional[np.ndarray] = None
        if len(points) >= 2:
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            try:
                new_coeffs = np.polyfit(ys, xs, 1)
            except (np.linalg.LinAlgError, ValueError):
                new_coeffs = None

        prev_coeffs = self._prev_coeffs.get(side)
        if new_coeffs is not None and prev_coeffs is not None:
            blended = self.smoothing_alpha * new_coeffs + (1 - self.smoothing_alpha) * prev_coeffs
        elif new_coeffs is not None:
            blended = new_coeffs
        elif prev_coeffs is not None:
            # No detection this frame; briefly coast on the previous fit
            # rather than dropping the lane entirely (handles a couple of
            # frames of occlusion/glare gracefully).
            blended = prev_coeffs
        else:
            self._prev_coeffs[side] = None
            return [], 0

        self._prev_coeffs[side] = blended
        poly = np.poly1d(blended)
        y_samples = np.linspace(int(height * 0.55), height - 1, 10)
        line_points = [(int(poly(y)), int(y)) for y in y_samples]
        return line_points, len(points)


# =============================================================================
# Depth Estimation - Depth Anything V2
# =============================================================================


@dataclass
class DepthResult:
    depth_map: np.ndarray
    inference_time_ms: float = 0.0
    mean_depth: float = 0.0

    @classmethod
    def from_map(cls, depth_map: np.ndarray, elapsed_ms: float) -> "DepthResult":
        return cls(depth_map=depth_map, inference_time_ms=elapsed_ms, mean_depth=float(np.mean(depth_map)))


class DepthEstimator:
    """High-level monocular depth estimator wrapping Depth Anything V2."""

    def __init__(
        self, weights_path: str = "weights/depth_anything_v2_vits.pth", encoder: str = "vits",
        input_size: int = 518, max_depth_m: float = 80.0, device_preference: str = "auto",
    ) -> None:
        self.weights_path, self.encoder, self.input_size = weights_path, encoder, input_size
        self.max_depth_m = max_depth_m
        self.device = get_device(device_preference)
        self.model = None
        self._use_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            cfg = model_configs.get(self.encoder, model_configs["vits"])

            if not Path(self.weights_path).exists():
                raise FileNotFoundError(
                    f"Depth Anything V2 weights not found at {self.weights_path}. "
                    "Run download_models.py to fetch them."
                )

            self.model = DepthAnythingV2(**cfg)
            self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
            self.model.to(self.device).eval()
            logger.info("Loaded Depth Anything V2 (%s) from %s", self.encoder, self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load Depth Anything V2 (%s). Using classical fallback.", exc)
            self._use_fallback = True

    def infer(self, frame: np.ndarray) -> DepthResult:
        start = time.perf_counter()
        if self._use_fallback or self.model is None:
            depth_map = self._fallback_depth(frame)
        else:
            try:
                depth_map = self._real_infer(frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("Depth inference failed (%s); using fallback.", exc)
                depth_map = self._fallback_depth(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DepthResult.from_map(depth_map, elapsed_ms)

    @torch.no_grad()
    def _real_infer(self, frame: np.ndarray) -> np.ndarray:
        raw_depth = np.asarray(self.model.infer_image(frame, self.input_size), dtype=np.float32)
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(raw_depth)
        normalized = (raw_depth - d_min) / (d_max - d_min)
        return (1.0 - normalized) * self.max_depth_m

    def _fallback_depth(self, frame: np.ndarray) -> np.ndarray:
        """Vertical-position + local-sharpness heuristic pseudo-depth estimator."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vertical_prior = np.tile(np.linspace(0.0, 1.0, h).reshape(h, 1), (1, w))
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpness = cv2.GaussianBlur(np.abs(laplacian), (9, 9), 0)
        sharpness = sharpness / (sharpness.max() + 1e-6)
        closeness = np.clip(0.7 * vertical_prior + 0.3 * sharpness, 0.0, 1.0)
        return ((1.0 - closeness) * self.max_depth_m).astype(np.float32)


# =============================================================================
# Weather Classification - EfficientNet-B0
# =============================================================================


@dataclass
class WeatherResult:
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = torchvision.models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class WeatherClassifier:
    """High-level weather classifier with a classical color-statistics fallback.

    Only ever loads checkpoints produced by this project's own
    `train_weather_model()` (a dict with 'model' + 'classes' keys). This
    project deliberately does NOT attempt to load a generic/third-party
    ImageNet checkpoint here: a prior version of this script downloaded the
    lukemelas/EfficientNet-PyTorch checkpoint (key names like `_conv_stem`,
    `_fc`) into this torchvision-based model (key names like `features.0`,
    `classifier.1`) with `strict=False` -- since NO keys overlap between the
    two naming schemes, that call silently loaded ZERO weights, leaving the
    "pretrained" model randomly initialized while looking like it succeeded.
    To avoid repeating that failure mode, an untrained/incompatible
    checkpoint is now rejected loudly and the classifier falls back to the
    classical heuristic instead of running on secretly-random weights.
    """

    def __init__(
        self, weights_path: str = "weights/efficientnet_weather.pth",
        classes: List[str] = None, input_size: int = 224,
        normalize_mean: List[float] = None, normalize_std: List[float] = None,
        device_preference: str = "auto",
    ) -> None:
        self.classes = classes or DEFAULT_WEATHER_CLASSES
        self.input_size = input_size
        self.mean = np.array(normalize_mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(normalize_std or [0.229, 0.224, 0.225], dtype=np.float32)
        self.device = get_device(device_preference)
        self.weights_path = weights_path
        self.model: nn.Module = None
        self._use_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        if not Path(self.weights_path).exists():
            logger.warning(
                "Weather classifier checkpoint not found at %s; using the classical "
                "color-statistics heuristic. Train a real checkpoint with: "
                "python models.py --data-dir data/weather_dataset --epochs 15", self.weights_path,
            )
            self._use_fallback = True
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu")

            if not (isinstance(checkpoint, dict) and "model" in checkpoint and "classes" in checkpoint):
                logger.error(
                    "Checkpoint at %s is not in this project's expected format "
                    "(dict with 'model' + 'classes' keys, produced by "
                    "train_weather_model()). Refusing to load it as if it were a "
                    "trained weather classifier -- using the classical heuristic "
                    "instead. Re-run: python models.py --data-dir data/weather_dataset",
                    self.weights_path,
                )
                self._use_fallback = True
                return

            checkpoint_classes = checkpoint["classes"]
            if checkpoint_classes != self.classes:
                logger.warning(
                    "Checkpoint class order %s differs from configured class order %s; "
                    "using the checkpoint's own order to guarantee correct label "
                    "mapping (this is what the model was actually trained on).",
                    checkpoint_classes, self.classes,
                )
                self.classes = checkpoint_classes

            self.model = build_efficientnet_b0(num_classes=len(self.classes))
            missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
            if missing or unexpected:
                logger.error(
                    "Weather checkpoint state_dict did not fully match the model "
                    "architecture (missing=%d, unexpected=%d); using the classical "
                    "heuristic instead of a partially-loaded model.", len(missing), len(unexpected),
                )
                self._use_fallback = True
                return

            self.model.to(self.device).eval()
            val_acc = checkpoint.get("val_acc")
            if val_acc is not None:
                logger.info(
                    "Loaded trained weather classifier from %s (reported val_acc=%.4f, classes=%s)",
                    self.weights_path, val_acc, self.classes,
                )
            else:
                logger.info("Loaded weather classifier weights from %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load weather classifier (%s). Using heuristic fallback.", exc)
            self._use_fallback = True

    def infer(self, frame: np.ndarray) -> WeatherResult:
        start = time.perf_counter()
        if self._use_fallback:
            label, probs = self._heuristic_classify(frame)
        else:
            try:
                label, probs = self._neural_infer(frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("Weather neural inference failed (%s); using heuristic.", exc)
                label, probs = self._heuristic_classify(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return WeatherResult(label=label, confidence=probs[label], probabilities=probs,
                              inference_time_ms=elapsed_ms)

    @torch.no_grad()
    def _neural_infer(self, frame: np.ndarray) -> tuple:
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        probs_tensor = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        probs = {cls: float(p) for cls, p in zip(self.classes, probs_tensor)}
        return max(probs, key=probs.get), probs

    def _heuristic_classify(self, frame: np.ndarray) -> tuple:
        """Classical brightness/saturation/contrast/edge-density weather heuristic."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        _, s_chan, v_chan = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mean_brightness = float(np.mean(v_chan)) / 255.0
        mean_saturation = float(np.mean(s_chan)) / 255.0
        contrast = float(np.std(v_chan)) / 255.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 60, 150) > 0))

        scores: Dict[str, float] = {cls: 0.0 for cls in self.classes}
        scores["night"] = max(0.0, 1.0 - mean_brightness * 3.0)
        scores["fog"] = max(0.0, min(1.0, (1.0 - mean_saturation) * (1.0 - contrast) * (1.0 - edge_density) * 1.5))
        scores["snow"] = max(0.0, min(1.0, mean_brightness * (1.0 - mean_saturation) * 1.2))
        scores["rain"] = max(0.0, min(1.0, (1.0 - abs(mean_brightness - 0.45) * 2) * (1.0 - mean_saturation) * 0.9))
        scores["overcast"] = max(0.0, min(1.0, (1.0 - abs(mean_brightness - 0.55) * 2) * (1.0 - mean_saturation) * 0.7))
        scores["clear"] = max(0.0, min(1.0, mean_brightness * mean_saturation * contrast * edge_density * 4.0))

        total = sum(scores.values()) + 1e-6
        probs = {cls: score / total for cls, score in scores.items()}
        return max(probs, key=probs.get), probs


def _build_class_remap(dataset, canonical_classes: List[str]) -> Dict[int, int]:
    """Map an ImageFolder's alphabetical class indices to our canonical order.

    ImageFolder assigns class indices by SORTING folder names alphabetically
    (e.g. clear=0, fog=1, night=2, overcast=3, rain=4, snow=5), which does
    NOT match DEFAULT_WEATHER_CLASSES / config.yaml's declared order
    (clear, rain, fog, snow, night, overcast). Training a model on
    ImageFolder's raw indices and then reading its logits with the
    config-declared order at inference time silently scrambles every label.
    This function builds a remap so the model is always trained against
    `canonical_classes`'s index order, and that same order is saved into the
    checkpoint so `WeatherClassifier` can never mismatch it again.
    """
    remap: Dict[int, int] = {}
    for folder_name, alpha_idx in dataset.class_to_idx.items():
        if folder_name not in canonical_classes:
            raise ValueError(
                f"Found dataset folder '{folder_name}' which is not one of the "
                f"expected weather classes {canonical_classes}. Check "
                f"{dataset.root} for a typo'd or extra subfolder."
            )
        remap[alpha_idx] = canonical_classes.index(folder_name)
    missing = set(canonical_classes) - set(dataset.class_to_idx.keys())
    if missing:
        raise ValueError(
            f"Dataset at {dataset.root} is missing folder(s) for class(es): "
            f"{sorted(missing)}. Every one of {canonical_classes} needs at least "
            f"a few images, or training/evaluation indices will be wrong."
        )
    return remap


def _confusion_matrix(model: nn.Module, loader, device: torch.device, num_classes: int) -> np.ndarray:
    """Build an num_classes x num_classes confusion matrix: rows=true, cols=predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            for true_label, pred_label in zip(labels.numpy(), preds):
                cm[true_label, pred_label] += 1
    return cm


def _log_confusion_matrix(cm: np.ndarray, classes: List[str], split_name: str) -> float:
    """Log a readable per-class accuracy table + confusion matrix, return overall accuracy."""
    total = cm.sum()
    overall_acc = float(np.trace(cm) / total) if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("%s set evaluation (n=%d)", split_name, total)
    logger.info("Overall accuracy: %.4f", overall_acc)
    logger.info("Per-class accuracy (recall):")
    for i, cls in enumerate(classes):
        row_total = cm[i].sum()
        row_acc = cm[i, i] / row_total if row_total > 0 else float("nan")
        logger.info("  %-10s %5.1f%%  (n=%d)", cls, row_acc * 100 if row_total > 0 else 0.0, row_total)

    header = "true\\pred".ljust(10) + "".join(f"{c[:6]:>8}" for c in classes)
    logger.info(header)
    for i, cls in enumerate(classes):
        row = cls[:9].ljust(10) + "".join(f"{cm[i, j]:>8d}" for j in range(len(classes)))
        logger.info(row)
    logger.info("=" * 60)
    return overall_acc


def train_weather_model(
    data_dir: str = "data/weather_dataset",
    output_path: str = "weights/efficientnet_weather.pth",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    input_size: int = 224,
    freeze_backbone: bool = False,
) -> Dict[str, float]:
    """Fine-tune EfficientNet-B0 on a labeled weather dataset (ImageFolder layout).

    Expects: data_dir/train/<class>/*.jpg and data_dir/val/<class>/*.jpg,
    where <class> is one of DEFAULT_WEATHER_CLASSES. If data_dir/test/
    also exists, it is evaluated once at the end for a final, held-out
    accuracy figure (kept separate from the val set used for checkpoint
    selection, so the reported number isn't the same split used to tune).

    Args:
        freeze_backbone: if True, freezes all EfficientNet-B0 feature layers
            and only trains the final classifier head (+ a partially-unfrozen
            last block). Trains MUCH faster on CPU (no GPU needed) at some
            accuracy cost -- a reasonable tradeoff for a laptop-only capstone.

    Returns:
        A dict summary: {"best_val_acc": ..., "test_acc": ... (if test/ exists)}.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    canonical_classes = DEFAULT_WEATHER_CLASSES
    device = get_device("auto")
    logger.info("Training on device: %s | freeze_backbone=%s", device, freeze_backbone)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(Path(data_dir) / "val"), transform=eval_tf)

    train_remap = _build_class_remap(train_ds, canonical_classes)
    val_remap = _build_class_remap(val_ds, canonical_classes)
    train_ds.target_transform = lambda y: train_remap[y]
    val_ds.target_transform = lambda y: val_remap[y]

    logger.info("Canonical class order (index 0..%d): %s", len(canonical_classes) - 1, canonical_classes)
    for cls in canonical_classes:
        n_train = sum(1 for _, idx in train_ds.samples if train_remap[idx] == canonical_classes.index(cls))
        n_val = sum(1 for _, idx in val_ds.samples if val_remap[idx] == canonical_classes.index(cls))
        logger.info("  class '%s': %d train images, %d val images", cls, n_train, n_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_features, len(canonical_classes))
    model = base.to(device)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        # Leave the last feature block trainable for a bit of adaptation
        # while keeping the bulk of the network (and CPU cost) frozen.
        for param in model.features[-1].parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Backbone frozen: training %d / %d parameters (%.1f%%)",
                    trainable, total_params, 100 * trainable / total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_acc = 0.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_cm = _confusion_matrix(model, val_loader, device, len(canonical_classes))
        val_acc = float(np.trace(val_cm) / max(val_cm.sum(), 1))
        train_acc = correct / max(total, 1)
        logger.info("Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_acc=%.4f",
                    epoch, epochs, running_loss / max(total, 1), train_acc, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": canonical_classes, "val_acc": val_acc},
                output_path,
            )
            logger.info("Saved new best checkpoint (val_acc=%.4f) to %s", val_acc, output_path)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

    # Final, honest evaluation report on the best checkpoint.
    best_checkpoint = torch.load(output_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    val_cm = _confusion_matrix(model, val_loader, device, len(canonical_classes))
    _log_confusion_matrix(val_cm, canonical_classes, "Validation (best checkpoint)")

    summary = {"best_val_acc": best_val_acc}

    test_dir = Path(data_dir) / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        test_remap = _build_class_remap(test_ds, canonical_classes)
        test_ds.target_transform = lambda y: test_remap[y]
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_cm = _confusion_matrix(model, test_loader, device, len(canonical_classes))
        test_acc = _log_confusion_matrix(test_cm, canonical_classes, "Held-out TEST")
        summary["test_acc"] = test_acc
    else:
        logger.info(
            "No data/weather_dataset/test/ split found -- reporting validation "
            "accuracy only. For a stronger, more defensible report, create a "
            "held-out test/ split (see prepare_weather_dataset.py) that was "
            "never used for checkpoint selection."
        )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the EfficientNet-B0 weather classifier.")
    parser.add_argument("--data-dir", type=str, default="data/weather_dataset")
    parser.add_argument("--output", type=str, default="weights/efficientnet_weather.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Freeze most of EfficientNet-B0 for much faster CPU-only training.")
    args = parser.parse_args()
    train_weather_model(
        data_dir=args.data_dir, output_path=args.output, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, input_size=args.input_size,
        freeze_backbone=args.freeze_backbone,
    )
