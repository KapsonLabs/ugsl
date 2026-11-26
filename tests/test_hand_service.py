"""Approved manifests and real synthetic video with mocked MediaPipe results."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from config import HandExtractionConfig
from landmarks.exceptions import HandOutputError
from landmarks.hand_debug import export_debug
from landmarks.hand_metrics import hand_metrics, no_hand_warnings
from landmarks.hand_parser import parse_hand_result
from landmarks.service import DecodingService, HandExtractionService
from registry import signers
from tests.test_annotation import setup
from tests.test_dataset_export import system
from tests.test_decoding_service import export
from tests.test_hand_extractor import fake_mediapipe, model_path
from tests.test_hand_parser import points, prediction, raw
from tests.test_ingestion import media
from tests.test_video_decoder import clips


@pytest.mark.parametrize("mode", ["REFERENCE", "PORTABLE"])
def test_approved_interval_timestamps_and_debug_exports(system, mode, tmp_path, model_path, fake_mediapipe):
    service, entry, (sample_id,) = export(system, mode=mode)
    registry_before = service.db.path.read_bytes()
    decoding = DecodingService.from_registry(service.db, entry.name, entry.version)
    hands = HandExtractionService(decoding, HandExtractionConfig(model_path))
    fake_mediapipe.tasks.vision.HandLandmarker.create_from_options.reset_mock()

    def detect(_image, timestamp):
        if timestamp == 125:
            return raw([points()], [[prediction("Left")]])
        return raw()

    original_factory = fake_mediapipe.tasks.vision.HandLandmarker.create_from_options.side_effect

    def create(options):
        tracker = original_factory(options)
        tracker.detect_for_video.side_effect = detect
        return tracker

    fake_mediapipe.tasks.vision.HandLandmarker.create_from_options.side_effect = create
    result = hands.process(sample_id)
    report, arrays = result.report, result.arrays
    assert report["status"] == "PROCESSED" and report["mediapipe_version"] == "test-version"
    assert (report["decoding"]["start_ms"], report["decoding"]["end_ms"]) == (100, 900)
    assert arrays.timestamps_ms[[0, -1]].tolist() == [125, 875]
    assert arrays.timestamps_ms.tolist() == [entry["timestamp_ms"] for entry in report["frames"]]
    assert arrays.image_landmarks.shape == (19, 2, 21, 3)
    assert arrays.presence_mask[0].tolist() == [True, False]
    assert report["metrics"]["frames_with_no_hands"] == 18
    assert report["frames"][0]["detections"][0]["predictions"][0]["label"] == "Left"
    assert service.db.path.read_bytes() == registry_before
    fake_mediapipe.trackers[0].close.assert_called_once()

    root = tmp_path / "debug"
    directory = export_debug(result, root, check_permission=decoding.reader.check_permission)
    with np.load(directory / "hand_landmarks.npz", allow_pickle=False) as archive:
        assert archive["schema_version"].item() == report["schema_version"]
        for name in ("image_landmarks", "world_landmarks", "timestamps_ms", "presence_mask", "handedness_score"):
            np.testing.assert_array_equal(archive[name], getattr(arrays, name))
    assert json.loads((directory / "extraction_report.json").read_text()) == report
    with pytest.raises(HandOutputError, match="already exists"):
        export_debug(result, root, check_permission=decoding.reader.check_permission)


def test_coverage_and_quality_metrics():
    left = raw([points()], [[prediction("Left", 0.8)]])
    right = raw([points()], [[prediction("Right", 0.6)]])
    both = raw([points(), points()], [[prediction("Right", 0.8)], [prediction("Left", 0.6)]])
    duplicate = raw([points(), points()], [[prediction("Left")], [prediction("Left")]])
    invalid = raw([points(float("nan"))], [[prediction("Right")]])
    frames = [
        parse_hand_result(value, index, index * 300)
        for index, value in enumerate((left, right, both, duplicate, invalid, raw(), raw(), raw()))
    ]
    stats = hand_metrics(frames, 2)
    assert stats["total_processed_frames"] == 8
    assert stats["frames_with_both_hands"] == stats["frames_with_only_left_hand"] == stats["frames_with_only_right_hand"] == 1
    assert stats["frames_with_no_hands"] == 5 and stats["frames_with_no_raw_detections"] == 3
    assert stats["ambiguous_handedness_assignments"] == stats["frames_with_invalid_landmark_values"] == 1
    assert stats["overall_frame_coverage"] == 3 / 8
    assert stats["left_hand_coverage"] == stats["right_hand_coverage"] == 2 / 8
    assert stats["average_handedness_score"] == pytest.approx(0.7)
    assert stats["average_processing_time_per_frame_ms"] == 250
    assert stats["effective_throughput_fps"] == 4
    warnings = no_hand_warnings(frames, 500, completed=True)
    assert [warning["code"] for warning in warnings] == ["LONG_NO_HAND_SEQUENCE"]
    assert warnings[0]["start_ms"] == 1500 and warnings[0]["duration_ms"] == 600


def test_direct_file_zero_detections_and_failure_are_distinct(clips, tmp_path, model_path, fake_mediapipe):
    hands = HandExtractionService(None, HandExtractionConfig(model_path, no_hand_warning_ms=500))
    result = hands.inspect_file(clips["vfr"], all_frames=True)
    assert result.report["source"] == "diagnostic_file" and result.report["status"] == "PROCESSED"
    assert result.arrays.timestamps_ms.tolist() == [1000, 1040, 1120, 1160, 1240]
    assert not result.arrays.presence_mask.any()
    assert "NO_HANDS_DETECTED" in {item["code"] for item in result.report["warnings"]}
    with pytest.raises(HandOutputError, match="approved"):
        export_debug(result, tmp_path / "debug", check_permission=lambda: None)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"invalid container")
    broken = hands.inspect_file(corrupt)
    assert broken.report["status"] == "FAILED" and broken.arrays is None
    assert broken.report["error_code"] == "DECODE_FAILED"
    for tracker in fake_mediapipe.trackers:
        tracker.close.assert_called_once()


def test_batch_failure_does_not_stop_remaining_samples(system, model_path, fake_mediapipe):
    service, entry, sample_ids = export(system, count=2)
    decoding = DecodingService.from_registry(service.db, entry.name, entry.version)
    missing = next(decoding.reader.iter_samples())
    missing.video_path.unlink()
    batch = HandExtractionService(decoding, HandExtractionConfig(model_path)).inspect_dataset()
    assert batch["processed"] == batch["failed"] == 1 and batch["dataset_error"] is None
    assert batch["samples"][0]["sample_id"] == missing.sample_id
    assert batch["samples"][0]["error_code"] == "INVALID_SOURCE_VIDEO"
    assert batch["samples"][1]["sample_id"] in sample_ids
    assert batch["samples"][1]["metrics"]["total_processed_frames"] == batch["metrics"]["total_processed_frames"]
    assert len(fake_mediapipe.trackers) == 2
    for tracker in fake_mediapipe.trackers:
        tracker.close.assert_called_once()


def test_revocation_during_extraction_blocks_returning_landmark_arrays(system, setup, model_path, fake_mediapipe):
    service, entry, (sample_id,) = export(system, mode="PORTABLE")
    _, _, _, _, signer, _ = setup
    decoding = DecodingService.from_registry(service.db, entry.name, entry.version)
    hands = HandExtractionService(decoding, HandExtractionConfig(model_path))

    def withdraw(_decoded, _result):
        signers.consent(service.db, signer.code, "WITHDRAWN")
        return True

    result = hands.process(sample_id, on_frame=withdraw)
    assert result.arrays is None and result.report["status"] == "FAILED"
    assert result.report["error_code"] == "DATASET_ACCESS_DENIED"
    assert "REVOKED" in result.report["error"]
    fake_mediapipe.trackers[0].close.assert_called_once()
    assert hands.process(sample_id).report["error_code"] == "DATASET_ACCESS_DENIED"


def test_interrupt_does_not_return_partial_arrays(system, model_path, fake_mediapipe):
    service, entry, (sample_id,) = export(system)
    decoding = DecodingService.from_registry(service.db, entry.name, entry.version)
    hands = HandExtractionService(decoding, HandExtractionConfig(model_path))

    def interrupt(_decoded, _result):
        raise KeyboardInterrupt

    result = hands.process(sample_id, on_frame=interrupt)
    assert result.arrays is None and result.report["status"] == "CANCELLED"
    assert result.report["error_code"] == "INTERRUPTED"
    assert result.report["metrics"]["total_processed_frames"] == 1
    fake_mediapipe.trackers[0].close.assert_called_once()


def test_cli_approved_sample_batch_direct_file_and_skeleton_mode(system, clips, tmp_path, model_path, fake_mediapipe):
    service, entry, (sample_id,) = export(system)
    runner = CliRunner()
    base = ["--db", str(service.db.path), "hands", "--model", str(model_path), "--fps", "12"]
    selector = ["--dataset", entry.name, "--version", entry.version]

    def invoke(*args):
        result = runner.invoke(app, [*base, *map(str, args)])
        assert result.exit_code == 0, (result.output, result.exception)
        return json.loads(result.output)

    assert invoke("extract", sample_id, *selector, "--output", tmp_path / "debug")["status"] == "PROCESSED"
    assert (tmp_path / "debug" / sample_id / "hand_landmarks.npz").is_file()
    assert invoke("inspect", sample_id, *selector)["metrics"]["total_processed_frames"] == 10
    assert invoke("inspect-dataset", *selector)["processed"] == 1
    assert invoke("inspect-file", clips[30])["source"] == "diagnostic_file"
    with patch("landmarks.hand_visualizer.HandLandmarkVisualizer") as visualizer:
        visualizer.return_value.__enter__.return_value.show.return_value = False
        assert invoke("visualize", sample_id, *selector, "--skeleton-only")["status"] == "CANCELLED"
    visualizer.assert_called_once_with(skeleton_only=True)
