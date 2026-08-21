"""Perception models: object detection, depth estimation, and weather
classification.

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


def _confusion_matrix(model: nn.Module, loader, device: torch.device, num_classes: int) -> np.ndarray:
    """Build a num_classes x num_classes confusion matrix: rows=true, cols=predicted.
    UNCHANGED from the previous version.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            for true_label, pred_label in zip(labels.numpy(), preds):
                cm[true_label, pred_label] += 1
    return cm


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


def _build_weighted_sampler(dataset, remap: Dict[int, int], class_counts: np.ndarray):
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
    cm: np.ndarray, classes: List[str], split_name: str, low_support_threshold: int = 20
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
    macro_f1 = float(np.mean([m["f1"] for m in per_class])) if per_class else 0.0

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
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {c: m for c, m in zip(classes, per_class)},
    }


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
) -> Dict[str, float]:
    """Fine-tune EfficientNet-B0 on a labeled weather dataset (ImageFolder layout).

    Expects: data_dir/train/<class>/*.jpg, data_dir/val/<class>/*.jpg, and
    optionally data_dir/test/<class>/*.jpg, where <class> is one of
    DEFAULT_WEATHER_CLASSES.

    CHANGES from the previous version (class-imbalance handling + richer
    metrics -- see the task's requirements list for the full rationale):
      - The training DataLoader now uses a WeightedRandomSampler (per-sample
        weight = 1 / class count) so rare classes, especially 'fog', are not
        drastically under-represented in each epoch. Val/test loaders are
        UNCHANGED (shuffle=False, no sampler, never touched by this).
      - CrossEntropyLoss now uses moderate (sqrt-inverse-frequency,
        mean-normalized) class weights -- not extreme inverse-frequency.
      - Best-checkpoint selection is now based on validation MACRO-F1
        (fairer across classes of very different size) instead of only
        overall accuracy. `val_acc` is still recorded in the checkpoint too.
      - Validation and the held-out test evaluation now report per-class
        precision/recall/F1, macro-F1, and the confusion matrix (previously:
        only per-class accuracy + confusion matrix). Fog's metrics are
        explicitly flagged as low-confidence in both reports whenever its
        support in that split is small.
      - Checkpoint format is unchanged/backward compatible: still a dict
        with "model" and "classes" keys, which is all WeatherClassifier
        requires. "val_acc" is kept; "val_macro_f1" is added (purely
        additive -- WeatherClassifier is not modified and does not require
        this key).

    Returns:
        A summary dict: {"best_val_acc", "best_val_macro_f1",
        "test_accuracy", "test_macro_f1" (only if a test/ split exists)}.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    canonical_classes = DEFAULT_WEATHER_CLASSES
    num_classes = len(canonical_classes)
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

    train_class_counts = _class_counts_from_dataset(train_ds, train_remap, num_classes)
    val_class_counts = _class_counts_from_dataset(val_ds, val_remap, num_classes)

    logger.info("Canonical class order (index 0..%d): %s", num_classes - 1, canonical_classes)
    for i, cls in enumerate(canonical_classes):
        logger.info("  class '%s': %d train images, %d val images",
                     cls, int(train_class_counts[i]), int(val_class_counts[i]))

    # --- Class-imbalance handling: sampler (train only) + moderate loss weights ---
    sampler = _build_weighted_sampler(train_ds, train_remap, train_class_counts)
    class_weights = _moderate_class_weights(train_class_counts).to(device)
    logger.info(
        "Moderated (sqrt-inverse-frequency, mean-normalized) class weights for loss: %s",
        {cls: round(float(w), 3) for cls, w in zip(canonical_classes, class_weights.cpu())},
    )

    # WeightedRandomSampler and shuffle=True are mutually exclusive in
    # DataLoader; the sampler already handles the random-order+reweighting.
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    # Val/test loaders NEVER use the sampler -- plain, unweighted, unshuffled
    # iteration over every sample exactly once, so reported metrics reflect
    # the true class distribution of that split.
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_features, num_classes)
    model = base.to(device)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Backbone frozen: training %d / %d parameters (%.1f%%)",
                    trainable, total_params, 100 * trainable / total_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_macro_f1 = -1.0
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

        val_cm = _confusion_matrix(model, val_loader, device, num_classes)
        val_total = int(val_cm.sum())
        val_acc = float(np.trace(val_cm) / val_total) if val_total > 0 else 0.0
        val_per_class = _precision_recall_f1_from_cm(val_cm)
        val_macro_f1 = float(np.mean([m["f1"] for m in val_per_class])) if val_per_class else 0.0
        train_acc = correct / max(total, 1)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_acc=%.4f val_macro_f1=%.4f",
            epoch, epochs, running_loss / max(total, 1), train_acc, val_acc, val_macro_f1,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": canonical_classes,
                    "val_acc": val_acc,            # kept for backward compatibility
                    "val_macro_f1": val_macro_f1,  # new: actual selection criterion
                },
                output_path,
            )
            logger.info("Saved new best checkpoint (val_macro_f1=%.4f, val_acc=%.4f) to %s",
                        val_macro_f1, val_acc, output_path)

    logger.info("Training complete. Best val_macro_f1=%.4f", best_val_macro_f1)

    # Reload the best checkpoint (by macro-F1) and produce the full report.
    best_checkpoint = torch.load(output_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    val_cm = _confusion_matrix(model, val_loader, device, num_classes)
    val_report = _log_classification_report(val_cm, canonical_classes, "Validation (best checkpoint)")

    summary = {"best_val_acc": val_report["accuracy"], "best_val_macro_f1": val_report["macro_f1"]}

    test_dir = Path(data_dir) / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        test_remap = _build_class_remap(test_ds, canonical_classes)
        test_ds.target_transform = lambda y: test_remap[y]
        # Never sampled -- plain evaluation pass, exactly once, over the held-out test set.
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_cm = _confusion_matrix(model, test_loader, device, num_classes)
        test_report = _log_classification_report(test_cm, canonical_classes, "Held-out TEST")
        summary["test_accuracy"] = test_report["accuracy"]
        summary["test_macro_f1"] = test_report["macro_f1"]
    else:
        logger.info(
            "No data/weather_dataset/test/ split found -- reporting validation "
            "metrics only."
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
