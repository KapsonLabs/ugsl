"""Defaults and validation shared by the camera and command line."""

import math
import re
from urllib.parse import urlsplit
from dataclasses import dataclass
from pathlib import Path

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
