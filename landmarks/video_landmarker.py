"""Common MediaPipe VIDEO model loading, frame validation and tracking lifecycle."""

import hashlib
from fractions import Fraction
from io import BytesIO
from numbers import Integral
from time import perf_counter
from zipfile import BadZipFile, ZipFile
from zlib import error as ZipDataError

import numpy as np

from landmarks.exceptions import InvalidTimestamp, UnsupportedImageFormat
from landmarks.schemas import DecodedFrame

MAX_INFERENCE_MS = (2**63 - 1) // 1000


class VideoLandmarker:
    kind: str
    task: str
    model_members: set[str]
    model_error: type[Exception]
    inference_error: type[Exception]
    adjust_inference_clock = False

    def __init__(self, config):
        self.config = config
        self.model_sha256: str | None = None
        self.mediapipe_version: str | None = None
        self.initialization_seconds = 0.0
        self._landmarker = None

    def __enter__(self):
        if self._landmarker is not None:
            raise self.model_error(f"This {self.kind} tracker is already open.")
        self._last_source: Fraction | None = None
        self._last_inference: int | None = None
        self._clock_offset = 0
        self.model_sha256 = None
        self.mediapipe_version = None
        started = perf_counter()
        try:
            bundle = self._read_model()
            try:
                import mediapipe as mp

                self.mediapipe_version = mp.__version__
                vision = mp.tasks.vision
                options = getattr(vision, f"{self.task}Options")(
                    base_options=mp.tasks.BaseOptions(model_asset_buffer=bundle),
                    running_mode=vision.RunningMode.VIDEO,
                    **self._task_options(),
                )
                self._landmarker = getattr(vision, self.task).create_from_options(options)
                self._mp = mp
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                raise self.model_error(
                    f"Could not initialize MediaPipe VIDEO {self.kind} tracking with "
                    f"{self.config.model_path}: {exc}"
                ) from exc
        finally:
            self.initialization_seconds = perf_counter() - started
        return self

    def _read_model(self) -> bytes:
        path = self.config.model_path
        try:
            bundle = path.read_bytes()
        except OSError as exc:
            raise self.model_error(f"Cannot read {self.kind} model at {path}.") from exc
        if not bundle:
            raise self.model_error(f"The {self.kind} model at {path} is empty.")
        self.model_sha256 = hashlib.sha256(bundle).hexdigest()
        try:
            with ZipFile(BytesIO(bundle)) as archive:
                if not self.model_members.issubset(archive.namelist()) or archive.testzip() is not None:
                    raise BadZipFile(f"Required {self.kind} detector models are missing or corrupt.")
        except (BadZipFile, NotImplementedError, RuntimeError, ZipDataError) as exc:
            raise self.model_error(f"Incompatible {self.kind} model bundle at {path}: {exc}") from exc
        return bundle

    def __exit__(self, exc_type, *_):
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except (OSError, RuntimeError, ValueError) as exc:
                if exc_type is None:
                    raise self.inference_error(f"Could not close the {self.kind} tracker: {exc}") from exc

    def _timestamps(self, frame: DecodedFrame) -> tuple[Fraction, int, int]:
        source = frame.timestamp_ms
        if isinstance(source, bool) or not isinstance(source, (Integral, Fraction)):
            raise InvalidTimestamp(f"Frame {frame.frame_index} has an invalid source timestamp.")
        source = Fraction(int(source)) if isinstance(source, Integral) else source
        timestamp = source // 1
        if not -(2**63) <= timestamp < 2**63 or (timestamp < 0 and not self.adjust_inference_clock):
            raise InvalidTimestamp(f"Frame {frame.frame_index} has an invalid source timestamp.")
        if self.adjust_inference_clock and self._last_source is not None and source <= self._last_source:
            raise InvalidTimestamp(f"Frame {frame.frame_index} source presentation timestamps are not strictly increasing.")
        if self.adjust_inference_clock and self._last_inference is None and not 0 <= timestamp <= MAX_INFERENCE_MS:
            self._clock_offset = -timestamp
        inference = timestamp + self._clock_offset
        if self._last_inference is not None and inference <= self._last_inference:
            if not self.adjust_inference_clock:
                raise InvalidTimestamp(
                    f"Frame {frame.frame_index} source time {timestamp} ms does not follow "
                    f"{self._last_inference} ms after flooring to whole milliseconds."
                )
            inference = self._last_inference + 1
        if not 0 <= inference <= MAX_INFERENCE_MS:
            raise InvalidTimestamp(f"Frame {frame.frame_index} exceeds the MediaPipe timestamp range.")
        return source, timestamp, inference

    def _infer(self, frame: DecodedFrame):
        if self._landmarker is None:
            raise self.inference_error(f"Open a {self.kind} extractor for the video before extracting frames.")
        pixels = frame.image
        if (
            not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.shape != (frame.height, frame.width, 3) or min(frame.height, frame.width) < 1
        ):
            raise UnsupportedImageFormat(f"{self.kind.capitalize()} extraction requires a nonempty H × W × 3 uint8 RGB frame.")
        source, timestamp, inference = self._timestamps(frame)
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))
            raw = self._landmarker.detect_for_video(image, inference)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise self.inference_error(
                f"MediaPipe failed on frame {frame.frame_index} at {timestamp} ms: {exc}"
            ) from exc
        self._last_source, self._last_inference = source, inference
        return raw, source, timestamp, inference
