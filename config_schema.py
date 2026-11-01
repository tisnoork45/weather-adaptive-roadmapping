"""Schema and validation for config.yaml.

Validation reports problems; it never modifies the configuration. Defaults
listed here are the values the code already uses when an optional key is
missing (they are applied where the key is read, e.g.
``config.pipeline.get("strict", False)``), so validating a configuration can
never change what the pipeline does with it.

    errors   -> ConfigError (a ValueError) listing every problem with its full key path
    warnings -> unknown keys, weather labels with no classifier class, etc.

Keys marked ``reserved`` are accepted and documented but not read by any code
yet (e.g. the camera mounting geometry, kept for later work).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_MISSING = object()
NUMBER = (int, float)


class ConfigError(ValueError):
    """config.yaml failed validation; ``errors`` holds one message per problem."""

    def __init__(self, errors: Sequence[str], source: str = "configuration") -> None:
        self.errors = list(errors)
        self.source = source
        super().__init__(f"Invalid {source} ({len(self.errors)} problem(s)):\n"
                         + "\n".join(f"  - {e}" for e in self.errors))


@dataclass
class Key:
    """One configuration value."""
    types: Tuple[type, ...]
    doc: str = ""
    required: bool = True
    default: Any = _MISSING              # documentation only: the default the code applies
    choices: Optional[Sequence[Any]] = None
    min: Optional[float] = None          # inclusive bounds for numbers
    max: Optional[float] = None
    gt: Optional[float] = None           # exclusive bounds for numbers
    lt: Optional[float] = None
    nullable: bool = False
    item: Optional["Key"] = None         # for lists: every element
    length: Optional[int] = None         # for lists: exact length
    non_empty: bool = False
    reserved: bool = False
    check: Optional[Callable[[Any], Optional[str]]] = None


@dataclass
class Section:
    """A mapping with known keys (``keys``) or arbitrary keys whose values all match ``values``."""
    keys: Dict[str, Any] = field(default_factory=dict)
    doc: str = ""
    required: bool = True
    values: Any = None                   # Key or Section for every entry of an open mapping
    reserved: bool = False


def _num(doc: str = "", **kw) -> Key:
    return Key(NUMBER, doc, **kw)


def _int(doc: str = "", **kw) -> Key:
    return Key((int,), doc, **kw)


def _str(doc: str = "", **kw) -> Key:
    return Key((str,), doc, **kw)


def _bool(doc: str = "", **kw) -> Key:
    return Key((bool,), doc, **kw)


def _probability(doc: str = "", **kw) -> Key:
    return _num(doc, min=0.0, max=1.0, **kw)


def _sha256(value: str) -> Optional[str]:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        return "must be a 64-character hexadecimal SHA-256"
    return None


TTC_LEVELS = ("CRITICAL", "WARNING", "CAUTION")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

SCHEMA = Section(keys={
    "project": Section(required=False, doc="informational; recorded in run manifests", keys={
        "name": _str(required=False),
        "version": _str(required=False),
    }),
    "paths": Section(required=False, keys={
        "outputs_dir": _str("root directory for recorded runs (<outputs_dir>/runs/)", required=False,
                            default="outputs"),
        "weights_dir": _str(required=False, reserved=True),
        "data_dir": _str(required=False, reserved=True),
        "logs_dir": _str("the log file location is logging.log_file", required=False, reserved=True),
    }),
    "device": Section(keys={
        "preference": Key((str,), choices=("auto", "cuda", "mps", "cpu")),
    }),
    "camera": Section(keys={
        "source": Key((int, str), "webcam index, video path or stream URL"),
        "width": _int(gt=0),
        "height": _int(gt=0),
        "fps": _num("fallback frame rate when a video does not report one", gt=0),
        "calibrated": _bool("false = placeholder intrinsics; metric outputs are labelled uncalibrated",
                            required=False, default=False),
        "intrinsics": Section(doc="pinhole intrinsics at width x height", keys={
            "fx": _num(gt=0), "fy": _num(gt=0), "cx": _num(min=0), "cy": _num(min=0),
        }),
        "mount_height_m": _num(required=False, reserved=True),
        "pitch_deg": _num(required=False, reserved=True),
        "extrinsics": Section(required=False, reserved=True, keys={
            "translation": Section(required=False, values=_num()),
            "rotation": Section(required=False, values=_num()),
        }),
    }),
    "logging": Section(required=False, doc="applied by the entry points (run.py, app.py, carla_sim.py, download.py)",
                       keys={
        "level": Key((str,), required=False, default="INFO",
                     check=lambda v: None if v.upper() in LOG_LEVELS else f"must be one of {list(LOG_LEVELS)}"),
        "log_to_file": _bool(required=False, default=True),
        "log_file": _str(required=False, default="logs/pipeline.log"),
        "max_bytes": _int(required=False, default=5 * 1024 * 1024, min=0),
        "backup_count": _int(required=False, default=3, min=0),
    }),
    "pipeline": Section(keys={
        "run_depth_estimation": _bool(),
        "run_weather_classification": _bool("false = no classifier; weather is 'unknown' (neutral x1.0 adjustments)",
                                            required=False, default=True),
        "run_tracking": _bool(),
        "run_bev_mapping": _bool("BEV, velocity, TTC and risk; needs metric depth"),
        "strict": _bool("true = a model that fails to load/run raises instead of using a heuristic fallback",
                        required=False, default=False),
        "target_fps": _num(gt=0),
    }),
    "dashboard": Section(keys={
        "title": _str(),
        "refresh_interval_sec": _num(required=False, reserved=True, min=0),
    }),
    "detection": Section(keys={
        "weights_path": _str(),
        "download_url": _str(required=False),
        "confidence_threshold": _probability(),
        "iou_threshold": _probability(),
        "input_size": _int(gt=0),
        "classes_of_interest": Key((list,), "COCO class ids to keep (null = all)", nullable=True,
                                   item=_int(min=0)),
    }),
    "depth_estimation": Section(keys={
        "mode": Key((str,), required=False, default="relative", choices=("relative", "metric")),
        "weights_path": _str("relative-depth checkpoint"),
        "download_url": _str(required=False),
        "metric_weights_path": _str("metric-depth checkpoint (used when mode is metric)"),
        "metric_download_url": _str(required=False),
        "metric_sha256": _str(required=False, check=_sha256),
        "encoder": Key((str,), choices=("vits", "vitb", "vitl")),
        "input_size": _int(gt=0),
        "max_depth_m": _num(gt=0),
        "ignore_bottom_rows_fraction": _num(required=False, default=0.0, min=0.0, lt=1.0),
    }),
    "weather_classification": Section(keys={
        "weights_path": _str(),
        "input_size": _int(gt=0),
        "classes": Key((list,), item=_str(), non_empty=True,
                       check=lambda v: "must not contain duplicates" if len(set(v)) != len(v) else None),
        "temporal_smoothing_s": _num(required=False, default=1.0, min=0),
        "normalize_mean": Key((list,), item=_num(), length=3),
        "normalize_std": Key((list,), item=_num(gt=0), length=3),
    }),
    "tracking": Section(keys={
        "track_thresh": _probability(),
        "track_buffer": _int(min=1),
        "match_thresh": _probability(),
        "frame_rate": _num(gt=0),
        "min_box_area": _num(min=0),
    }),
    "mapping": Section(keys={
        "bev_range_m": _num(gt=0),
        "bev_width_m": _num(gt=0),
        "bev_resolution_px_per_m": _num(gt=0),
    }),
    "fusion": Section(keys={
        "base_weights": Section(values=_num(min=0)),
        "weather_modifiers": Section(values=Section(values=_num(min=0))),
        "ema_alpha": _num(gt=0, max=1),
    }),
    "confidence_estimation": Section(keys={
        "detection_signal_weights": Section(values=_num(min=0)),
        "depth_signal_weights": Section(values=_num(min=0)),
        "history_length": _int(min=1),
    }),
    "velocity": Section(keys={
        "history_length": _int(min=2),
        "max_gap_seconds": _num(gt=0),
        "ema_alpha": _num(gt=0, max=1),
    }),
    "risk": Section(keys={
        "min_safe_distance_m": _num(min=0),
        "ttc_thresholds_s": Section(keys={level: _num(gt=0) for level in TTC_LEVELS}),
        "weather_multipliers": Section(values=_num(gt=0)),
    }),
    "datasets": Section(required=False, keys={
        "kitti_depth_selection_root": _str(required=False),
    }),
})


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _type_name(types: Tuple[type, ...]) -> str:
    names = {int: "integer", float: "number", str: "string", bool: "boolean (true/false)", list: "list"}
    if types == NUMBER:
        return "number"
    return " or ".join(names.get(t, t.__name__) for t in types)


def _is_type(value: Any, types: Tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in types:
        return False                     # YAML true/false is not a number here
    return isinstance(value, types)


def _check_key(spec: Key, value: Any, path: str, report: ValidationReport) -> None:
    if value is None:
        if not spec.nullable:
            report.errors.append(f"{path}: expected {_type_name(spec.types)}, got null")
        return
    if not _is_type(value, spec.types):
        report.errors.append(f"{path}: expected {_type_name(spec.types)}, got {type(value).__name__} {value!r}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        report.errors.append(f"{path}: must be finite, got {value!r}")
        return
    if spec.choices is not None and value not in spec.choices:
        report.errors.append(f"{path}: must be one of {list(spec.choices)}, got {value!r}")
        return
    if isinstance(value, NUMBER):
        for bound, ok, text in ((spec.min, lambda b: value >= b, ">="), (spec.max, lambda b: value <= b, "<="),
                                (spec.gt, lambda b: value > b, ">"), (spec.lt, lambda b: value < b, "<")):
            if bound is not None and not ok(bound):
                report.errors.append(f"{path}: must be {text} {bound}, got {value!r}")
    if isinstance(value, list):
        if spec.non_empty and not value:
            report.errors.append(f"{path}: must not be empty")
        if spec.length is not None and len(value) != spec.length:
            report.errors.append(f"{path}: must have exactly {spec.length} entries, got {len(value)}")
        if spec.item is not None:
            for i, v in enumerate(value):
                _check_key(spec.item, v, f"{path}[{i}]", report)
    if spec.check is not None:
        msg = spec.check(value)
        if msg:
            report.errors.append(f"{path}: {msg}, got {value!r}")


def _check(spec: Any, value: Any, path: str, report: ValidationReport) -> None:
    if isinstance(spec, Key):
        _check_key(spec, value, path, report)
        return
    if not isinstance(value, Mapping):
        report.errors.append(f"{path or 'configuration'}: expected a mapping of keys, got {type(value).__name__} "
                             f"{value!r}")
        return
    if spec.values is not None:
        for k, v in value.items():
            _check(spec.values, v, f"{path}.{k}" if path else str(k), report)
        return
    for name, sub in spec.keys.items():
        sub_path = f"{path}.{name}" if path else name
        if name not in value:
            if sub.required:
                report.errors.append(f"{sub_path}: required key is missing")
            continue
        _check(sub, value[name], sub_path, report)
    for name in value:
        if name not in spec.keys:
            report.warnings.append(f"{path}.{name}: unknown key (ignored by the software)" if path
                                   else f"{name}: unknown section (ignored by the software)")


def _cross_checks(cfg: Mapping, report: ValidationReport) -> None:
    def get(*keys):
        node = cfg
        for k in keys:
            if not isinstance(node, Mapping) or k not in node:
                return None
            node = node[k]
        return node

    classes = get("weather_classification", "classes")
    if isinstance(classes, list) and all(isinstance(c, str) for c in classes):
        for keys in (("fusion", "weather_modifiers"), ("risk", "weather_multipliers")):
            table = get(*keys)
            if not isinstance(table, Mapping):
                continue
            name = ".".join(keys)
            missing = [c for c in classes if c not in table]
            if missing:
                report.errors.append(f"{name}: no entry for weather class(es) {missing} "
                                     f"(weather_classification.classes = {classes})")
            extra = [k for k in table if k not in classes]
            if extra:
                report.warnings.append(f"{name}: entries {extra} are not weather_classification.classes "
                                       f"and are never used")

    base, modifiers = get("fusion", "base_weights"), get("fusion", "weather_modifiers")
    if isinstance(base, Mapping) and isinstance(modifiers, Mapping):
        for label, mods in modifiers.items():
            if isinstance(mods, Mapping):
                unknown = [m for m in mods if m not in base]
                if unknown:
                    report.warnings.append(f"fusion.weather_modifiers.{label}: {unknown} are not keys of "
                                           f"fusion.base_weights and are never used")

    ttc = get("risk", "ttc_thresholds_s")
    if isinstance(ttc, Mapping) and all(_is_type(ttc.get(level), NUMBER) for level in TTC_LEVELS):
        for lower, higher in zip(TTC_LEVELS, TTC_LEVELS[1:]):
            if not ttc[higher] > ttc[lower]:
                report.errors.append(f"risk.ttc_thresholds_s.{higher}: must be > {lower} ({ttc[lower]}), "
                                     f"got {ttc[higher]}")


def validate_config(config: Mapping) -> ValidationReport:
    """Every error and warning for ``config`` (a ConfigDict or plain dict); never raises."""
    report = ValidationReport()
    _check(SCHEMA, config, "", report)
    _cross_checks(config, report)
    return report


def ensure_valid(config: Mapping, source: str = "configuration") -> ValidationReport:
    """validate_config(), raising ConfigError if there is any error."""
    report = validate_config(config)
    if report.errors:
        raise ConfigError(report.errors, source)
    return report


def reserved_keys(spec: Any = SCHEMA, path: str = "") -> List[str]:
    """Dotted paths of the keys the schema accepts but no code reads yet."""
    out: List[str] = []
    if isinstance(spec, Section):
        if spec.reserved and path:
            return [path]
        for name, sub in spec.keys.items():
            out += reserved_keys(sub, f"{path}.{name}" if path else name)
    elif isinstance(spec, Key) and spec.reserved:
        out.append(path)
    return out
