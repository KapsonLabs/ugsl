"""Synchronous MediaPipe Tasks VIDEO inference on Phase 2.1 RGB frames."""

import hashlib
from fractions import Fraction
from io import BytesIO
from numbers import Integral
from zipfile import BadZipFile, ZipFile

import numpy as np

from config import HandExtractionConfig
from landmarks.exceptions import HandInferenceError, HandModelError, InvalidTimestamp, UnsupportedImageFormat
from landmarks.hand_parser import parse_hand_result
from landmarks.schemas import DecodedFrame, HandFrame


class HandExtractor:
    """Own one tracking context, closed on completion or interruption."""

    def __init__(self, config: HandExtractionConfig):
        self.config = config
        self.model_sha256: str | None = None
        self.mediapipe_version: str | None = None
        self._landmarker = None
        self._last_timestamp: int | None = None

    def __enter__(self):
        if self._landmarker is not None:
            raise HandModelError("This hand tracker is already open.")
        self._last_timestamp = None
        try:
            bundle = self.config.model_path.read_bytes()
        except OSError as exc:
            raise HandModelError(f"Cannot read hand model at {self.config.model_path}.") from exc
        if not bundle:
            raise HandModelError(f"The hand model at {self.config.model_path} is empty.")
        self.model_sha256 = hashlib.sha256(bundle).hexdigest()
        try:
            with ZipFile(BytesIO(bundle)) as archive:
                required = {"hand_detector.tflite", "hand_landmarks_detector.tflite"}
                if not required.issubset(archive.namelist()) or archive.testzip() is not None:
                    raise BadZipFile("Required hand detector models are missing or corrupt.")
        except (BadZipFile, NotImplementedError, RuntimeError) as exc:
            raise HandModelError(f"Incompatible hand model bundle at {self.config.model_path}: {exc}") from exc

        try:
            import mediapipe as mp

            self.mediapipe_version = mp.__version__
            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=bundle),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._mp = mp
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HandModelError(
                f"Could not initialize MediaPipe VIDEO hand tracking with {self.config.model_path}: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, *_):
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except (OSError, RuntimeError, ValueError) as exc:
                if exc_type is None:
                    raise HandInferenceError(f"Could not close the hand tracker: {exc}") from exc

    def extract(self, frame: DecodedFrame) -> HandFrame:
        if self._landmarker is None:
            raise HandInferenceError("Open a hand extractor for the video before extracting frames.")
        pixels = frame.image
        if (
            not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.shape != (frame.height, frame.width, 3) or min(frame.height, frame.width) < 1
        ):
            raise UnsupportedImageFormat("Hand extraction requires a nonempty H × W × 3 uint8 RGB frame.")
        source_ts = frame.timestamp_ms
        if isinstance(source_ts, bool) or not isinstance(source_ts, (Integral, Fraction)) or source_ts < 0:
            raise InvalidTimestamp(f"Frame {frame.frame_index} has an invalid source timestamp.")
        timestamp = int(source_ts // 1)
        # The native Tasks graph converts integer milliseconds to signed microseconds.
        if timestamp > (2**63 - 1) // 1000:
            raise InvalidTimestamp(f"Frame {frame.frame_index} exceeds the MediaPipe timestamp range.")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise InvalidTimestamp(
                f"Frame {frame.frame_index} source time {timestamp} ms does not follow "
                f"{self._last_timestamp} ms after flooring to whole milliseconds."
            )
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))
            raw = self._landmarker.detect_for_video(image, timestamp)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HandInferenceError(f"MediaPipe failed on frame {frame.frame_index} at {timestamp} ms: {exc}") from exc
        self._last_timestamp = timestamp
        return parse_hand_result(raw, frame.frame_index, timestamp, mirrored_input=self.config.mirrored_input)
