"""Defaults and validation shared by the camera and command line."""

import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

DATASET_ROOT = Path("datasets/raw")
REGISTRY_PATH = Path("datasets/registry.db")
EXPORT_DIRECTORY_NAME = "exports"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must contain 1–64 ASCII letters, digits, underscores or hyphens "
            "and start with a letter or digit."
        )


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30
    duration: float = 3
    countdown: float = 3
    repetitions: int = 1
    codec: str = "mp4v"
    source: str | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("camera index", self.camera_index, 0),
            ("width", self.width, 2),
            ("height", self.height, 2),
            ("repetitions", self.repetitions, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name.capitalize()} must be an integer of at least {minimum}.")
            if name != "repetitions" and value > 2**31 - 1:
                raise ValueError(f"{name.capitalize()} exceeds the OpenCV integer limit.")

        for name, value, minimum in (
            ("FPS", self.fps, "positive"),
            ("Duration", self.duration, "positive"),
            ("Countdown", self.countdown, "non-negative"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if value < 0 or (value == 0 and minimum == "positive"):
                raise ValueError(f"{name} must be {minimum}.")

        if not isinstance(self.codec, str) or not re.fullmatch(r"[A-Za-z0-9]{4}", self.codec):
            raise ValueError("Codec must be a four-character ASCII code.")

        if self.source is not None:
            try:
                url = urlsplit(self.source)
                port = url.port
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("Provide a valid remote camera URL.") from exc
            if (
                url.scheme not in {"http", "https", "rtsp", "rtsps"} or not url.hostname
                or port == 0 or any(c.isspace() or ord(c) < 32 for c in self.source)
                or url.username is not None or url.password is not None
            ):
                raise ValueError("Provide an HTTP(S) or RTSP(S) camera URL without embedded credentials.")


@dataclass(frozen=True)
class QualityConfig:
    """Technical requirements and visual sampling for isolated signs."""

    min_width: int = 640
    min_height: int = 480
    min_fps: float = 20
    min_frame_count: int = 20
    min_duration_seconds: float = 1
    max_duration_seconds: float = 10
    max_file_size_mb: float = 100
    visual_samples: int = 8
    probe_timeout: float = 20
    decode_timeout: float = 60
    quarantine_retention_days: int = 7

    @classmethod
    def from_environment(cls) -> "QualityConfig":
        settings = {}
        for field in fields(cls):
            name = f"UGSL_QUALITY_{field.name.upper()}"
            if (value := os.environ.get(name)) is not None:
                try:
                    settings[field.name] = field.type(value)
                except ValueError as exc:
                    raise ValueError(f"{name} must be a valid number.") from exc
        return cls(**settings)

    def __post_init__(self) -> None:
        for name in ("min_width", "min_height", "min_frame_count", "visual_samples", "quarantine_retention_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("min_fps", "min_duration_seconds", "max_duration_seconds",
                     "max_file_size_mb", "probe_timeout", "decode_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError("Maximum duration must be at least the minimum duration.")

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024**2)


@dataclass(frozen=True)
class ReviewConfig:
    policy: str = "SINGLE"
    confidence_min: float = 0.0
    confidence_max: float = 1.0

    def __post_init__(self) -> None:
        policy = self.policy.strip().upper()
        if policy not in {"SINGLE", "DUAL"}:
            raise ValueError("Review policy must be SINGLE or DUAL.")
        object.__setattr__(self, "policy", policy)
        if not (math.isfinite(self.confidence_min) and math.isfinite(self.confidence_max)
                and self.confidence_min < self.confidence_max):
            raise ValueError("Annotation confidence requires a finite increasing range.")

    @classmethod
    def from_environment(cls) -> "ReviewConfig":
        try:
            minimum = float(os.environ.get("UGSL_ANNOTATION_CONFIDENCE_MIN", "0"))
            maximum = float(os.environ.get("UGSL_ANNOTATION_CONFIDENCE_MAX", "1"))
        except ValueError as exc:
            raise ValueError("Annotation confidence limits must be numbers.") from exc
        return cls(os.environ.get("UGSL_REVIEW_POLICY", "SINGLE"), minimum, maximum)


HAND_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandExtractionConfig:
    """Settings for one MediaPipe tracking session per video."""

    model_path: Path = HAND_MODEL_PATH
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    mirrored_input: bool = False
    no_hand_warning_ms: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        _validate_landmark_thresholds(self)
        if type(self.mirrored_input) is not bool:
            raise ValueError("mirrored_input must be Boolean.")
        if type(self.no_hand_warning_ms) is not int or self.no_hand_warning_ms < 1:
            raise ValueError("no_hand_warning_ms must be a positive integer.")


def _validate_landmark_thresholds(config, *extra: str) -> None:
    for name in ("min_detection_confidence", "min_presence_confidence", "min_tracking_confidence", *extra):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number from 0 to 1.")


POSE_MODEL_VARIANTS = ("lite", "full", "heavy")
POSE_MODEL_PATHS = {
    variant: HAND_MODEL_PATH.parent / f"pose_landmarker_{variant}.task" for variant in POSE_MODEL_VARIANTS
}


@dataclass(frozen=True)
class PoseExtractionConfig:
    """MediaPipe inference settings and pose diagnostic thresholds."""

    model_path: Path | None = None
    model_variant: str | None = None
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    visibility_threshold: float = 0.5
    presence_threshold: float = 0.5
    timestamp_gap_ms: int = 250
    displacement_threshold: float = 0.25

    def __post_init__(self) -> None:
        requested = self.model_variant
        if requested is not None and requested not in POSE_MODEL_VARIANTS:
            raise ValueError("Pose model variant must be lite, full, or heavy.")
        if self.model_path is None:
            variant = requested or "full"
            path = POSE_MODEL_PATHS[variant]
        else:
            path = Path(self.model_path).expanduser()
            inferred = next((key for key, asset in POSE_MODEL_PATHS.items() if path.name == asset.name), None)
            if requested is not None and inferred is not None and requested != inferred:
                raise ValueError("The pose model path and selected model variant disagree.")
            variant = requested or inferred or "custom"
        object.__setattr__(self, "model_variant", variant)
        object.__setattr__(self, "model_path", path)
        _validate_landmark_thresholds(self, "visibility_threshold", "presence_threshold")
        if type(self.timestamp_gap_ms) is not int or self.timestamp_gap_ms < 1:
            raise ValueError("timestamp_gap_ms must be a positive integer.")
        if (
            type(self.displacement_threshold) not in (int, float)
            or not math.isfinite(self.displacement_threshold) or self.displacement_threshold <= 0
        ):
            raise ValueError("displacement_threshold must be a finite positive number.")
