"""Native binding options, RGB and time contracts, and deterministic cleanup."""

import hashlib
import os
import sys
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pytest

from config import HAND_MODEL_PATH, HandExtractionConfig
from landmarks.exceptions import HandInferenceError, HandModelError, InvalidTimestamp, UnsupportedImageFormat
from landmarks.hands import HandExtractor
from landmarks.schemas import DecodedFrame
from tests.test_hand_parser import raw


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "mock.task"
    with ZipFile(path, "w") as bundle:
        for name in ("hand_detector.tflite", "hand_landmarks_detector.tflite"):
            bundle.writestr(name, b"mock model")
    return path


@pytest.fixture
def fake_mediapipe(monkeypatch):
    trackers = []
    options = []

    def create(configuration):
        options.append(configuration)
        tracker = Mock()
        tracker.detect_for_video.return_value = raw()
        trackers.append(tracker)
        return tracker

    binding = SimpleNamespace(
        __version__="test-version",
        tasks=SimpleNamespace(BaseOptions=SimpleNamespace, vision=SimpleNamespace(
            HandLandmarkerOptions=SimpleNamespace, RunningMode=SimpleNamespace(VIDEO="VIDEO"),
            HandLandmarker=SimpleNamespace(create_from_options=Mock(side_effect=create)),
        )),
        Image=Mock(side_effect=SimpleNamespace),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        trackers=trackers,
        options=options,
    )
    monkeypatch.setitem(sys.modules, "mediapipe", binding)
    return binding


def frame(timestamp=0, pixels=None, index=0):
    if pixels is None:
        pixels = np.zeros((180, 240, 3), dtype=np.uint8)
    return DecodedFrame("fixture", index, Fraction(timestamp), Fraction(0), pixels, 240, 180)


@pytest.mark.parametrize("kwargs", [
    {"min_detection_confidence": -0.5}, {"min_presence_confidence": float("nan")},
    {"min_tracking_confidence": 1.1}, {"no_hand_warning_ms": 0}, {"mirrored_input": "yes"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HandExtractionConfig(**kwargs)


def test_video_options_rgb_conversion_and_source_timestamps(fake_mediapipe, model_path):
    config = HandExtractionConfig(model_path, 0.6, 0.7, 0.8)
    rgba = np.zeros((180, 240, 4), dtype=np.uint8)
    pixels = rgba[..., :3]
    pixels[0, 0] = (10, 30, 70)
    with HandExtractor(config) as hand:
        first = hand.extract(frame(Fraction(401, 3), pixels))
        second = hand.extract(frame(167))
    options, = fake_mediapipe.options
    assert options.running_mode == "VIDEO" and options.num_hands == 2
    assert options.min_hand_detection_confidence == 0.6
    assert options.min_hand_presence_confidence == 0.7
    assert options.min_tracking_confidence == 0.8
    assert options.base_options.model_asset_buffer == model_path.read_bytes()
    assert hand.model_sha256 == hashlib.sha256(model_path.read_bytes()).hexdigest()
    tracker, = fake_mediapipe.trackers
    assert [call.args[1] for call in tracker.detect_for_video.call_args_list] == [133, 167]
    image = tracker.detect_for_video.call_args_list[0].args[0]
    assert image.image_format == "SRGB"
    assert image.data.flags.c_contiguous
    assert image.data[0, 0].tolist() == [10, 30, 70]
    assert (first.timestamp_ms, second.timestamp_ms) == (133, 167)
    tracker.close.assert_called_once()


def test_missing_and_incompatible_model(fake_mediapipe, model_path, tmp_path):
    with pytest.raises(HandModelError, match="Cannot read"):
        with HandExtractor(HandExtractionConfig(tmp_path / "missing.task")):
            pytest.fail("Should not open an absent model")
    broken = tmp_path / "broken.task"
    broken.write_bytes(b"not a bundle")
    with pytest.raises(HandModelError, match="Incompatible hand model"):
        with HandExtractor(HandExtractionConfig(broken)):
            pytest.fail("Should not load a corrupt model")
    fake_mediapipe.tasks.vision.HandLandmarker.create_from_options.side_effect = ValueError("invalid task format")
    with pytest.raises(HandModelError, match="invalid task format"):
        with HandExtractor(HandExtractionConfig(model_path)):
            pytest.fail("Should not open an incompatible model")


def test_invalid_rgb_timestamp_and_cleanup_on_error(fake_mediapipe, model_path):
    with pytest.raises(InvalidTimestamp, match="flooring"):
        with HandExtractor(HandExtractionConfig(model_path)) as hand:
            with pytest.raises(UnsupportedImageFormat, match="RGB"):
                hand.extract(frame(0, np.zeros((180, 240, 3), dtype=np.float32)))
            with pytest.raises(InvalidTimestamp, match="invalid source"):
                hand.extract(frame(-1))
            hand.extract(frame(Fraction(21, 2)))
            hand.extract(frame(Fraction(109, 10)))
    first, = fake_mediapipe.trackers
    first.detect_for_video.assert_called_once()
    first.close.assert_called_once()


def test_task_errors_and_independent_tracking_contexts(fake_mediapipe, model_path):
    extractor = HandExtractor(HandExtractionConfig(model_path))
    with pytest.raises(HandInferenceError, match="native failure"):
        with extractor as hand:
            fake_mediapipe.trackers[-1].detect_for_video.side_effect = RuntimeError("native failure")
            hand.extract(frame(0))
    with extractor as hand:
        hand.extract(frame(0))
    assert len(fake_mediapipe.trackers) == 2
    for tracker in fake_mediapipe.trackers:
        tracker.close.assert_called_once()


@pytest.mark.skipif(os.getenv("UGSL_RUN_NATIVE_HAND_TEST") != "1", reason="requires a native MediaPipe session")
def test_official_bundle_loads_with_installed_tasks_api():
    mp = pytest.importorskip("mediapipe")
    with HandExtractor(HandExtractionConfig()) as hand:
        result = hand.extract(frame())
        assert hand.mediapipe_version == mp.__version__
        assert hand.model_sha256 == HAND_MODEL_PATH.with_suffix(".task.sha256").read_text().split()[0]
        assert result.timestamp_ms == 0 and not result.presence_mask.any()
