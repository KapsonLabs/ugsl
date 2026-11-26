import hashlib
import os
from fractions import Fraction
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from config import POSE_MODEL_PATHS, PoseExtractionConfig
from landmarks.exceptions import InvalidTimestamp, PoseInferenceError, PoseModelError, UnsupportedImageFormat
from landmarks.pose import PoseExtractor
from tests.test_hand_extractor import fake_mediapipe, frame
from tests.test_pose_parser import raw


@pytest.fixture
def pose_model(tmp_path):
    path = tmp_path / "pose_landmarker_lite.task"
    with ZipFile(path, "w") as bundle:
        for name in PoseExtractor.model_members:
            bundle.writestr(name, b"mock model")
    return path


@pytest.fixture
def fake_pose(fake_mediapipe):
    vision = fake_mediapipe.tasks.vision
    vision.PoseLandmarkerOptions = vision.HandLandmarkerOptions
    vision.PoseLandmarker = SimpleNamespace(create_from_options=vision.HandLandmarker.create_from_options)
    create_original = vision.PoseLandmarker.create_from_options.side_effect

    def create(options):
        tracker = create_original(options)
        if hasattr(options, "num_poses"):
            tracker.detect_for_video.return_value = raw()
        return tracker

    vision.PoseLandmarker.create_from_options.side_effect = create
    return fake_mediapipe


@pytest.mark.parametrize("settings", [
    {"model_variant": "unknown"}, {"min_detection_confidence": -1}, {"min_presence_confidence": 1.2},
    {"min_tracking_confidence": float("nan")}, {"visibility_threshold": 2},
    {"presence_threshold": float("inf")}, {"timestamp_gap_ms": 0}, {"displacement_threshold": 0},
])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        PoseExtractionConfig(**settings)


def test_model_variant_selection_and_explicit_path(pose_model):
    assert PoseExtractionConfig().model_path == POSE_MODEL_PATHS["full"]
    assert PoseExtractionConfig(model_variant="heavy").model_path == POSE_MODEL_PATHS["heavy"]
    assert PoseExtractionConfig(model_path=pose_model).model_variant == "lite"
    with pytest.raises(ValueError, match="disagree"):
        PoseExtractionConfig(model_path=pose_model, model_variant="full")


def test_missing_and_incompatible_model_errors(fake_pose, tmp_path):
    path = tmp_path / "invalid.task"
    with pytest.raises(PoseModelError, match="Cannot read"):
        with PoseExtractor(PoseExtractionConfig(path)):
            pytest.fail("Missing model opened")
    path.write_bytes(b"Not a Task")
    with pytest.raises(PoseModelError, match="Incompatible pose model"):
        with PoseExtractor(PoseExtractionConfig(path)):
            pytest.fail("Invalid model opened")


def test_initialization_failure_is_reported(fake_pose, pose_model):
    create = fake_pose.tasks.vision.PoseLandmarker.create_from_options
    create.side_effect = ValueError("model API mismatch")
    extractor = PoseExtractor(PoseExtractionConfig(pose_model))
    with pytest.raises(PoseModelError, match="model API mismatch"):
        with extractor:
            pytest.fail("Invalid task opened")
    assert extractor.initialization_seconds > 0
    del fake_pose.tasks.vision.PoseLandmarker
    with pytest.raises(PoseModelError, match="PoseLandmarker"):
        with extractor:
            pytest.fail("Unsupported MediaPipe opened")


def test_video_options_rgb_passthrough_and_independent_cleanup(fake_pose, pose_model):
    config = PoseExtractionConfig(pose_model, min_detection_confidence=0.6, min_presence_confidence=0.7)
    pixels = np.broadcast_to(np.array([1, 2, 3], dtype=np.uint8), (180, 240, 3))
    for _ in range(2):
        with PoseExtractor(config) as pose:
            result = pose.extract(frame(125, pixels, index=3))
            assert (result.frame_index, result.timestamp_ms, result.inference_timestamp_ms) == (3, 125, 125)
        assert pose.model_sha256 == hashlib.sha256(pose_model.read_bytes()).hexdigest()
        assert pose.mediapipe_version == "test-version"
    assert len(fake_pose.trackers) == 2
    for tracker, options in zip(fake_pose.trackers, fake_pose.options, strict=True):
        assert options.running_mode == "VIDEO" and options.num_poses == 1
        assert not options.output_segmentation_masks
        assert (options.min_pose_detection_confidence, options.min_pose_presence_confidence, options.min_tracking_confidence) == (
            0.6, 0.7, 0.5,
        )
        image, time_ms = tracker.detect_for_video.call_args.args
        assert time_ms == 125 and image.image_format == "SRGB"
        assert image.data.flags.c_contiguous and image.data[0, 0].tolist() == [1, 2, 3]
        tracker.close.assert_called_once()


def test_source_pts_are_retained_when_inference_clock_needs_adjustment(fake_pose, pose_model):
    with PoseExtractor(PoseExtractionConfig(pose_model)) as pose:
        frames = [
            pose.extract(frame(time, index=index))
            for index, time in enumerate([Fraction(-3, 2), Fraction(-5, 4), Fraction(3, 2)])
        ]
        assert [f.source_timestamp for f in frames] == [Fraction(-3, 2), Fraction(-5, 4), Fraction(3, 2)]
        assert [f.timestamp_ms for f in frames] == [-2, -2, 1]
        assert [f.inference_timestamp_ms for f in frames] == [0, 1, 3]
        with pytest.raises(InvalidTimestamp, match="strictly increasing"):
            pose.extract(frame(0, index=3))
    calls = fake_pose.trackers[0].detect_for_video.call_args_list
    assert [call.args[1] for call in calls] == [0, 1, 3]


def test_bad_frame_and_inference_exception_close_tracker(fake_pose, pose_model):
    config = PoseExtractionConfig(pose_model)
    with pytest.raises(PoseInferenceError, match="native error"):
        with PoseExtractor(config) as pose:
            with pytest.raises(UnsupportedImageFormat, match="RGB"):
                pose.extract(frame(pixels=np.zeros((180, 240, 3), dtype=np.float32)))
            with pytest.raises(InvalidTimestamp, match="frame index"):
                pose.extract(frame(0, index=-1))
            fake_pose.trackers[0].detect_for_video.side_effect = RuntimeError("native error")
            pose.extract(frame())
    fake_pose.trackers[0].close.assert_called_once()


@pytest.mark.skipif(os.getenv("UGSL_RUN_NATIVE_POSE_TEST") != "1", reason="requires native MediaPipe")
def test_default_model_loads_with_installed_tasks_api():
    mp = pytest.importorskip("mediapipe")
    config = PoseExtractionConfig()
    with PoseExtractor(config) as pose:
        result = pose.extract(frame())
        assert pose.mediapipe_version == mp.__version__
        assert pose.model_sha256 == config.model_path.with_suffix(".task.sha256").read_text().split()[0]
        assert result.timestamp_ms == 0 and not result.pose_detected
