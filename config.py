"""Defaults and validation shared by the camera and command line."""

import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

DATASET_ROOT = Path("datasets/raw")
REGISTRY_PATH = Path("datasets/registry.db")
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
