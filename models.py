"""Perception models: object detection, depth estimation, and weather
classification.

Merged into one file since each individual wrapper is a few hundred lines
and this project's scale doesn't warrant a dedicated file/package per model.
Every model wrapper follows the same pattern: try to load the real
pretrained network; if unavailable, fall back to a deterministic classical
CV / heuristic implementation so the pipeline always runs end-to-end.
"""

from __future__ import annotations

import logging
import math
import sys
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

# The vendored `depth_anything_v2` package lives next to this file. Put this
# directory on sys.path so it imports no matter which working directory the
# entry point was launched from (previously it only worked from the repo root).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DINOv2 logs "xFormers not available" twice per model load. xFormers is an
# optional speed-up; the standard attention path is used without it, so the
# warning is noise rather than a problem.
logging.getLogger("dinov2").setLevel(logging.ERROR)

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
        allow_download: bool = True,
    ) -> None:
        self.weights_path = weights_path
        self.allow_download = allow_download
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

            if Path(self.weights_path).exists():
                source = self.weights_path
            elif self.allow_download:
                source = "yolo11x.pt"
                logger.warning(
                    "YOLO weights not found at %s; attempting ultralytics auto-download of %s. "
                    "Run `python download.py models --only yolo` beforehand for offline use.",
                    self.weights_path, source,
                )
            else:
                raise FileNotFoundError(
                    f"YOLO weights not found at {self.weights_path} and auto-download is disabled"
                )
            self.model = YOLO(source)
            self.model.to(self.device)
            # Report what was actually loaded: after an auto-download this is
            # NOT self.weights_path, and logging that path made old logs say
            # "not found at X" immediately followed by "loaded from X".
            loaded_from = getattr(self.model, "ckpt_path", None) or source
            logger.info("Loaded YOLO11x model from %s on device %s", loaded_from, self.device)
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
        normalization_smoothing_alpha: float = 0.3,
    ) -> None:
        self.weights_path, self.encoder, self.input_size = weights_path, encoder, input_size
        self.max_depth_m = max_depth_m
        self.device = get_device(device_preference)
        self.model = None
        self._use_fallback = False
        # EMA-smoothed (d_min, d_max) normalization bounds -- see _real_infer
        # for why: Depth Anything V2's raw output is scene-relative, so
        # recomputing min/max fresh every frame (the original behavior)
        # rescales the ENTIRE depth map differently frame to frame even when
        # nothing in the scene actually moved, which shows up downstream as
        # spurious relative-velocity/TTC noise. Smoothing the bounds over
        # time removes that artificial jitter while still tracking real
        # scene changes (just not instantaneously). 1.0 = no smoothing
        # (identical to the original per-frame behavior).
        self.normalization_smoothing_alpha = normalization_smoothing_alpha
        self._ema_d_min: Optional[float] = None
        self._ema_d_max: Optional[float] = None
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
                    "Run `python download.py models --only depth` to fetch them."
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
        d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(raw_depth)

        # Temporally smooth the normalization bounds (see __init__ docstring
        # comment) rather than rescaling against this frame's raw min/max in
        # isolation. First frame has no prior estimate, so it seeds the EMA
        # with the raw bounds (identical to the original one-frame behavior).
        alpha = self.normalization_smoothing_alpha
        if self._ema_d_min is None:
            self._ema_d_min, self._ema_d_max = d_min, d_max
        else:
            self._ema_d_min = alpha * d_min + (1 - alpha) * self._ema_d_min
            self._ema_d_max = alpha * d_max + (1 - alpha) * self._ema_d_max

        norm_min, norm_max = self._ema_d_min, self._ema_d_max
        if norm_max - norm_min < 1e-6:
            norm_min, norm_max = d_min, d_max  # degenerate smoothed range -- fall back to this frame's own

        # This frame's raw values can fall slightly outside the smoothed
        # bounds (e.g. a genuinely new near/far object just entered frame) --
        # clip rather than let normalized go negative or past 1.0.
        normalized = np.clip((raw_depth - norm_min) / (norm_max - norm_min), 0.0, 1.0)
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


# --- Weather preprocessing: ONE implementation for training AND inference ----
#
# Training used to decode with PIL and resize with torchvision's antialiased
# Resize, while WeatherClassifier decoded with OpenCV and used cv2.resize's
# non-antialiased INTER_LINEAR -- the deployed pipeline scored 1.6 points
# lower than the training report on the same test images. Both paths now go
# through `load_weather_rgb` + `preprocess_weather_rgb`, and checkpoints record
# WEATHER_PREPROCESSING_VERSION so a mismatch is detectable at load time.

WEATHER_PREPROCESSING_VERSION = "rgb-cv2area-squash-v1"


def load_weather_rgb(path) -> np.ndarray:
    """Decode an image file exactly like video frames are decoded (OpenCV), as RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_weather_rgb(rgb: np.ndarray, input_size: int, mean: Sequence[float],
                           std: Sequence[float]) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalized float32 3xSxS tensor.

    Squash-resizes the whole frame (no crop, aspect not preserved) with
    INTER_AREA, which is antialiased when downscaling camera frames.
    """
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class WeatherEvalTransform:
    """Validation/test transform: exactly the inference preprocessing."""

    def __init__(self, input_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.input_size, self.mean, self.std = input_size, list(mean), list(std)

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        return preprocess_weather_rgb(rgb, self.input_size, self.mean, self.std)


class WeatherTrainTransform(WeatherEvalTransform):
    """Training transform: light augmentation in RGB uint8 space, then the
    same preprocessing as inference.

    - Crops keep the frame's own aspect ratio (scale 0.8-1.0 per side) so the
      final squash-resize distorts training crops the same way it distorts
      full frames at inference. torchvision's RandomResizedCrop picked
      roughly square crops out of 16:9 frames instead.
    - Brightness/contrast jitter models camera exposure. Saturation and hue
      are deliberately NOT jittered: colour saturation is one of the few cues
      separating clear from overcast, the most-confused pair.
    """

    def __init__(self, input_size: int, mean: Sequence[float], std: Sequence[float],
                 crop_scale: Tuple[float, float] = (0.8, 1.0), exposure_jitter: float = 0.1) -> None:
        super().__init__(input_size, mean, std)
        self.crop_scale, self.exposure_jitter = crop_scale, exposure_jitter

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        import random

        h, w = rgb.shape[:2]
        scale = random.uniform(*self.crop_scale)
        ch, cw = max(1, round(h * scale)), max(1, round(w * scale))
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        rgb = rgb[y:y + ch, x:x + cw]
        if random.random() < 0.5:
            rgb = rgb[:, ::-1]
        if self.exposure_jitter > 0:
            j = self.exposure_jitter
            brightness, contrast = random.uniform(1 - j, 1 + j), random.uniform(1 - j, 1 + j)
            f = rgb.astype(np.float32)
            f = (f - f.mean()) * contrast + f.mean()
            rgb = np.clip(f * brightness, 0, 255).astype(np.uint8)
        return super().__call__(np.ascontiguousarray(rgb))


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

            preprocessing = checkpoint.get("preprocessing")
            if preprocessing is None:
                logger.warning(
                    "Weather checkpoint %s predates the shared preprocessing (it was trained "
                    "with PIL decoding + torchvision Resize); inference preprocessing differs "
                    "slightly from training. Retrain to remove this mismatch.", self.weights_path,
                )
            else:
                if preprocessing.get("version") != WEATHER_PREPROCESSING_VERSION:
                    logger.warning(
                        "Weather checkpoint preprocessing version %r != this code's %r; "
                        "predictions may be degraded.", preprocessing.get("version"),
                        WEATHER_PREPROCESSING_VERSION,
                    )
                # The checkpoint, not config.yaml, defines what the model was trained on.
                self.input_size = int(preprocessing["input_size"])
                self.mean = np.array(preprocessing["mean"], dtype=np.float32)
                self.std = np.array(preprocessing["std"], dtype=np.float32)

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess_weather_rgb(rgb, self.input_size, self.mean, self.std).unsqueeze(0).to(self.device)
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


class WeatherTemporalSmoother:
    """Exponential moving average of weather class probabilities over time.

    Per-frame predictions flicker on footage unlike the training data: on
    road.mp4, 21 of the first 240 sunny frames (mostly a 0.8 s burst of glare
    on white buildings) came out rain/snow with confidence up to 0.82, which
    then switched fusion weights and risk thresholds frame by frame. Weather
    changes over seconds to minutes, so averaging over ~1 s removes that
    without hiding real changes.

    Time-based rather than per-frame: the averaging weight comes from the
    elapsed time between frames, so behaviour is the same at 14 fps (live
    processing) and 24 fps (video timeline). A backwards step in time (a new
    video starting at 0) or a gap longer than `reset_gap_s` restarts it from
    the current frame. time_constant_s <= 0 disables smoothing.
    """

    def __init__(self, time_constant_s: float = 1.0, reset_gap_s: float = 5.0) -> None:
        self.time_constant_s, self.reset_gap_s = time_constant_s, reset_gap_s
        self.reset()

    def reset(self) -> None:
        self._probs: Optional[Dict[str, float]] = None
        self._last_t: Optional[float] = None

    def update(self, result: WeatherResult, timestamp: float) -> WeatherResult:
        if self.time_constant_s <= 0:
            return result
        dt = None if self._last_t is None else timestamp - self._last_t
        if (self._probs is None or dt is None or dt < 0 or dt > self.reset_gap_s
                or set(self._probs) != set(result.probabilities)):
            self._probs = dict(result.probabilities)
        else:
            alpha = 1.0 - math.exp(-dt / self.time_constant_s)
            self._probs = {c: alpha * p + (1.0 - alpha) * self._probs[c] for c, p in result.probabilities.items()}
        self._last_t = timestamp
        label = max(self._probs, key=self._probs.get)
        return WeatherResult(label=label, confidence=self._probs[label], probabilities=dict(self._probs),
                             inference_time_ms=result.inference_time_ms)


def _build_class_remap(dataset, canonical_classes: List[str]) -> Dict[int, int]:
    """Map an ImageFolder's alphabetical class indices to our canonical order.

    ImageFolder assigns class indices by SORTING folder names alphabetically,
    which does NOT match DEFAULT_WEATHER_CLASSES / config.yaml's declared
    order. This function builds a remap so the model is always trained
    against `canonical_classes`'s index order, and that same order is saved
    into the checkpoint so WeatherClassifier can never mismatch it. UNCHANGED
    from the previous version.
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


def _predict(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, targets) for every sample, in loader order (no shuffling)."""
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _confusion_from(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """num_classes x num_classes confusion matrix: rows=true, cols=predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def _sample_sources(dataset) -> np.ndarray:
    """'dawn' for files merged from DAWN (dawn_ prefix), else 'bdd'."""
    return np.array(["dawn" if Path(path).name.startswith("dawn_") else "bdd" for path, _ in dataset.samples])


def _class_counts_from_dataset(dataset, remap: Dict[int, int], num_classes: int) -> np.ndarray:
    """Count how many samples fall into each canonical class index, using the
    dataset's raw (path, alphabetical_idx) sample list + the remap -- a
    single pass, reused for both the weighted sampler and the loss weights
    (and for the existing per-class train/val count logging).
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, raw_idx in dataset.samples:
        counts[remap[raw_idx]] += 1
    return counts


def _build_weighted_sampler(dataset, remap: Dict[int, int], class_counts: np.ndarray, generator=None):
    """Build a WeightedRandomSampler that gives each TRAINING sample a weight
    of 1 / (count of its class), so rare classes (fog especially) are drawn
    roughly as often as common ones over an epoch, rather than the model
    mostly seeing 'rain'/'snow'/etc. This addresses class imbalance at the
    sampling level; see `_moderate_class_weights` for the separate,
    complementary loss-weighting approach (used together, each is individually
    moderate rather than either alone being extreme).

    NEVER used for val/test loaders -- only train_weather_model()'s train_loader.
    """
    from torch.utils.data import WeightedRandomSampler

    sample_weights = np.array(
        [1.0 / max(int(class_counts[remap[raw_idx]]), 1) for _, raw_idx in dataset.samples],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights), num_samples=len(dataset), replacement=True,
        generator=generator,
    )


def _moderate_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    """Compute conservative (NOT extreme inverse-frequency) per-class loss
    weights: square-root inverse frequency, normalized so the weights have
    mean 1.0. Square-rooting compresses the range compared to plain inverse
    frequency (which would make BDD100K's naturally-scarce 'fog' dominate
    the loss disproportionately), and mean-normalizing keeps the overall
    loss magnitude comparable to unweighted CrossEntropyLoss.
    """
    total = max(int(class_counts.sum()), 1)
    freq = class_counts.astype(np.float64) / total
    inv_sqrt = 1.0 / np.sqrt(freq + 1e-6)
    normalized = inv_sqrt / inv_sqrt.mean()
    return torch.tensor(normalized, dtype=torch.float32)


def _precision_recall_f1_from_cm(cm: np.ndarray) -> List[Dict[str, float]]:
    """Per-class precision/recall/F1/support computed directly from a
    confusion matrix (rows=true, cols=predicted) -- no sklearn dependency.
    """
    num_classes = cm.shape[0]
    results: List[Dict[str, float]] = []
    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({"precision": precision, "recall": recall, "f1": f1, "support": support})
    return results


def _log_classification_report(
    cm: np.ndarray, classes: List[str], split_name: str, low_support_threshold: int = 20,
    macro_over_supported: bool = False,
) -> Dict[str, float]:
    """Log overall accuracy, macro-F1, per-class precision/recall/F1, and the
    confusion matrix for one evaluation split. Explicitly flags any class
    (fog especially, called out by name) whose support in THIS split is too
    small to trust the reported metric -- rather than presenting every
    number with equal implied confidence.

    Replaces the previous `_log_confusion_matrix` (which only reported
    per-class accuracy); kept as a single function since both the
    post-training validation report and the held-out test report need
    identical formatting.
    """
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    per_class = _precision_recall_f1_from_cm(cm)
    # Per-source subsets lack some classes entirely (DAWN has no clear/night/
    # overcast); averaging those zero-support F1s in would be meaningless.
    scored = [m for m in per_class if m["support"] > 0] if macro_over_supported else per_class
    macro_f1 = float(np.mean([m["f1"] for m in scored])) if scored else 0.0

    logger.info("=" * 70)
    logger.info("%s classification report (n=%d)", split_name, total)
    logger.info("Overall accuracy: %.4f | Macro-F1: %.4f", accuracy, macro_f1)
    logger.info("%-10s %10s %10s %10s %8s", "class", "precision", "recall", "f1", "support")
    for cls, m in zip(classes, per_class):
        logger.info("%-10s %10.3f %10.3f %10.3f %8d", cls, m["precision"], m["recall"], m["f1"], m["support"])
        if cls == "fog" and m["support"] < low_support_threshold:
            logger.warning(
                "  \u26a0 FOG METRICS ABOVE ARE LOW-CONFIDENCE: only %d fog samples in "
                "the '%s' split. Report this caveat explicitly in your capstone "
                "writeup -- do not present fog precision/recall/F1 as a reliable "
                "estimate on this few samples.",
                m["support"], split_name,
            )
        elif m["support"] < low_support_threshold:
            logger.info(
                "  (note: only %d samples for '%s' in this split -- treat with caution)",
                m["support"], cls,
            )

    header = "true\\pred".ljust(10) + "".join(f"{c[:6]:>8}" for c in classes)
    logger.info(header)
    for i, cls in enumerate(classes):
        row = cls[:9].ljust(10) + "".join(f"{cm[i, j]:>8d}" for j in range(len(classes)))
        logger.info(row)
    logger.info("=" * 70)

    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {c: m for c, m in zip(classes, per_class)},
        "confusion_matrix": cm.tolist(),
    }


def _evaluate_split(model: nn.Module, loader, dataset, device: torch.device,
                    classes: List[str], split_name: str) -> Dict[str, object]:
    """Overall report plus one report per data source (BDD100K dashcam vs DAWN),
    so DAWN results are never mistaken for dashcam results."""
    preds, targets = _predict(model, loader, device)
    sources = _sample_sources(dataset)
    report = _log_classification_report(_confusion_from(preds, targets, len(classes)), classes, split_name)
    report["by_source"] = {}
    for source in ("bdd", "dawn"):
        mask = sources == source
        if mask.any():
            report["by_source"][source] = _log_classification_report(
                _confusion_from(preds[mask], targets[mask], len(classes)), classes,
                f"{split_name} [{source.upper()} only; macro-F1 over classes present]",
                macro_over_supported=True,
            )
    return report


# =============================================================================
# train_weather_model() -- the function this task asked to modify
# =============================================================================


def train_weather_model(
    data_dir: str = "data/weather_dataset",
    output_path: str = "weights/efficientnet_weather.pth",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    input_size: int = 224,
    freeze_backbone: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    class_weighted_loss: bool = True,
    exposure_jitter: float = 0.1,
    patience: int = 0,
    evaluate_test: bool = True,
    run_dir: Optional[str] = None,
    pretrained: bool = True,
) -> Dict[str, object]:
    """Fine-tune EfficientNet-B0 on a labeled weather dataset (ImageFolder layout).

    Expects data_dir/{train,val}/<class>/*.jpg and optionally data_dir/test/,
    where <class> is one of DEFAULT_WEATHER_CLASSES. Files prefixed `dawn_`
    are treated as DAWN; everything else as BDD100K (for per-source metrics).

    Procedure:
      - Images are decoded and preprocessed by exactly the functions
        WeatherClassifier uses at inference (load_weather_rgb +
        preprocess_weather_rgb); training adds only WeatherTrainTransform's
        light augmentation in front.
      - Class imbalance: WeightedRandomSampler on train (val/test are never
        resampled), plus optional sqrt-inverse-frequency loss weights.
      - The best epoch is selected on VALIDATION macro-F1 only. `patience`
        > 0 stops after that many epochs without improvement.
      - The test split is evaluated exactly once, after training, and only if
        evaluate_test=True. Pass evaluate_test=False for any run used to
        compare settings, so the test set never influences choices.
      - Seeded (python/numpy/torch, sampler and worker RNGs, cuDNN
        deterministic). GPU float reductions can still differ slightly between
        runs/hardware, so results are reproducible to within noise, not bit-exact.

    The checkpoint stays backward compatible ({"model", "classes", "val_acc", ...})
    and additionally records "preprocessing", "train_config" and "best_epoch".
    If run_dir is given, a JSON report (config, data counts, per-epoch history,
    val/test reports) is written to run_dir/report.json.
    """
    import json
    import random

    from torch.utils.data import DataLoader
    from torchvision import datasets

    config = {k: v for k, v in locals().items() if k not in ("json", "random", "DataLoader", "datasets")}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    canonical_classes = DEFAULT_WEATHER_CLASSES
    num_classes = len(canonical_classes)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    device = get_device("auto")
    logger.info("Training on device: %s | config=%s", device, config)

    train_tf = WeatherTrainTransform(input_size, mean, std, exposure_jitter=exposure_jitter)
    eval_tf = WeatherEvalTransform(input_size, mean, std)

    def image_folder(split: str, transform):
        ds = datasets.ImageFolder(str(Path(data_dir) / split), transform=transform, loader=load_weather_rgb)
        remap = _build_class_remap(ds, canonical_classes)
        ds.target_transform = remap.__getitem__  # picklable for DataLoader workers (lambdas fail on Windows)
        return ds, remap

    train_ds, train_remap = image_folder("train", train_tf)
    val_ds, val_remap = image_folder("val", eval_tf)

    train_class_counts = _class_counts_from_dataset(train_ds, train_remap, num_classes)
    val_class_counts = _class_counts_from_dataset(val_ds, val_remap, num_classes)
    logger.info("Canonical class order (index 0..%d): %s", num_classes - 1, canonical_classes)
    for i, cls in enumerate(canonical_classes):
        logger.info("  class '%s': %d train images, %d val images",
                    cls, int(train_class_counts[i]), int(val_class_counts[i]))

    generator = torch.Generator().manual_seed(seed)
    sampler = _build_weighted_sampler(train_ds, train_remap, train_class_counts, generator=generator)
    class_weights = _moderate_class_weights(train_class_counts).to(device) if class_weighted_loss else None
    if class_weights is not None:
        logger.info("Loss class weights (sqrt-inverse-frequency, mean-normalized): %s",
                    {c: round(float(w), 3) for c, w in zip(canonical_classes, class_weights.cpu())})

    loader_kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda",
                     "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    preprocessing = {"version": WEATHER_PREPROCESSING_VERSION, "input_size": input_size,
                     "mean": mean, "std": std}
    best_val_macro_f1, best_epoch, history = -1.0, 0, []
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_preds, val_targets = _predict(model, val_loader, device)
        val_cm = _confusion_from(val_preds, val_targets, num_classes)
        val_acc = float(np.trace(val_cm) / max(int(val_cm.sum()), 1))
        val_macro_f1 = float(np.mean([m["f1"] for m in _precision_recall_f1_from_cm(val_cm)]))
        train_acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": running_loss / max(total, 1), "train_acc": train_acc,
                        "val_acc": val_acc, "val_macro_f1": val_macro_f1})
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_acc=%.4f val_macro_f1=%.4f",
            epoch, epochs, running_loss / max(total, 1), train_acc, val_acc, val_macro_f1,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1, best_epoch = val_macro_f1, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": canonical_classes,
                    "val_acc": val_acc,            # kept for backward compatibility
                    "val_macro_f1": val_macro_f1,  # the actual selection criterion
                    "best_epoch": epoch,
                    "preprocessing": preprocessing,
                    "train_config": config,
                },
                output_path,
            )
            logger.info("Saved new best checkpoint (epoch %d, val_macro_f1=%.4f, val_acc=%.4f) to %s",
                        epoch, val_macro_f1, val_acc, output_path)
        elif patience and epoch - best_epoch >= patience:
            logger.info("Early stopping: no val macro-F1 improvement for %d epochs.", patience)
            break

    logger.info("Training complete. Best epoch %d, val_macro_f1=%.4f", best_epoch, best_val_macro_f1)

    # Reload the best checkpoint (by validation macro-F1) and produce the full reports.
    model.load_state_dict(torch.load(output_path, map_location="cpu")["model"])
    model.to(device)
    val_report = _evaluate_split(model, val_loader, val_ds, device, canonical_classes, "Validation (best checkpoint)")
    summary: Dict[str, object] = {
        "best_epoch": best_epoch, "best_val_acc": val_report["accuracy"],
        "best_val_macro_f1": val_report["macro_f1"],
    }
    report = {"config": config, "preprocessing": preprocessing, "history": history, "best_epoch": best_epoch,
              "counts": {"train": dict(zip(canonical_classes, train_class_counts.tolist())),
                         "val": dict(zip(canonical_classes, val_class_counts.tolist()))},
              "validation": val_report}

    test_dir = Path(data_dir) / "test"
    if not evaluate_test:
        logger.info("evaluate_test=False: the test split was not evaluated (use this for any run that "
                    "compares settings).")
    elif test_dir.exists() and any(test_dir.iterdir()):
        test_ds, _ = image_folder("test", eval_tf)
        # Never sampled -- plain evaluation pass, exactly once, over the held-out test set.
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        test_report = _evaluate_split(model, test_loader, test_ds, device, canonical_classes, "Held-out TEST")
        summary["test_accuracy"] = test_report["accuracy"]
        summary["test_macro_f1"] = test_report["macro_f1"]
        report["test"] = test_report
        report["counts"]["test"] = dict(zip(
            canonical_classes, _class_counts_from_dataset(test_ds, _build_class_remap(test_ds, canonical_classes),
                                                          num_classes).tolist()))
    else:
        logger.info("No %s split found -- reporting validation metrics only.", test_dir)

    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(run_dir) / "report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Wrote run report to %s", Path(run_dir) / "report.json")
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
    parser.add_argument("--num-workers", type=int, default=4,
                         help="DataLoader worker processes (0 = load in the main process).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-class-weighted-loss", action="store_true",
                         help="Rely on the balanced sampler alone instead of also reweighting the loss.")
    parser.add_argument("--exposure-jitter", type=float, default=0.1,
                         help="Brightness/contrast jitter strength (0 disables).")
    parser.add_argument("--patience", type=int, default=0,
                         help="Early-stop after N epochs without val macro-F1 improvement (0 = off).")
    parser.add_argument("--no-test-eval", action="store_true",
                         help="Do not evaluate the test split (use for every run that compares settings).")
    parser.add_argument("--run-dir", type=str, default=None, help="Write report.json here.")
    args = parser.parse_args()
    train_weather_model(
        data_dir=args.data_dir, output_path=args.output, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, input_size=args.input_size,
        freeze_backbone=args.freeze_backbone, num_workers=args.num_workers, seed=args.seed,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        class_weighted_loss=not args.no_class_weighted_loss, exposure_jitter=args.exposure_jitter,
        patience=args.patience, evaluate_test=not args.no_test_eval, run_dir=args.run_dir,
    )
