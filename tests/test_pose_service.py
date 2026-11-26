import json
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from config import HandExtractionConfig, PoseExtractionConfig
from landmarks.exceptions import DatasetAccessError, PoseOutputError
from landmarks.hand_debug import export_debug as export_hands
from landmarks.pose_debug import export_debug
from landmarks.service import DecodingService, HandExtractionService, PoseExtractionService
from registry import signers
from tests.test_annotation import setup
from tests.test_dataset_export import system
from tests.test_decoding_service import export
from tests.test_hand_extractor import fake_mediapipe, model_path
from tests.test_ingestion import media
from tests.test_pose_extractor import fake_pose, pose_model
from tests.test_pose_parser import points, raw
from tests.test_video_decoder import clips


@pytest.mark.parametrize("mode", ["REFERENCE", "PORTABLE"])
def test_approved_samples_match_decoding_and_report_provenance(system, mode, pose_model, fake_pose):
    system_service, entry, (sample_id,) = export(system, mode=mode)
    before = system_service.db.path.read_bytes()
    decoder = DecodingService.from_registry(system_service.db, entry.name, entry.version, fps=12)
    annotation = next(decoder.reader.iter_records()).annotation
    with decoder.process(sample_id) as run:
        frames = list(run)
    create = fake_pose.tasks.vision.PoseLandmarker.create_from_options
    make_tracker = create.side_effect

    def track(options):
        tracker = make_tracker(options)
        tracker.detect_for_video.side_effect = [
            raw(points(), points()) if index % 2 == 0 else raw() for index in range(len(frames))
        ]
        return tracker

    create.side_effect = track
    service = PoseExtractionService(decoder, PoseExtractionConfig(pose_model), multiple_participant_samples=(sample_id,))
    result = service.process(sample_id)
    arrays, report = result.arrays, result.report
    assert report["status"] == "PROCESSED" and report["model_variant"] == "lite"
    assert report["dataset"] == {"id": decoder.reader.metadata["dataset_id"], "name": entry.name, "version": entry.version}
    assert report["annotation_id"] == annotation.id
    assert report["configuration"]["num_poses"] == 1
    assert report["decoding"]["target_fps"] == 12
    assert len(frames) == 10 and arrays.image_landmarks.shape == (10, 33, 3)
    assert arrays.world_available.tolist() == arrays.pose_detected.tolist() == [True, False] * 5
    assert arrays.frame_indices.tolist() == [frame.frame_index for frame in frames]
    assert arrays.timestamps_ms.tolist() == [frame.timestamp_ms // 1 for frame in frames]
    assert arrays.inference_timestamps_ms.tolist() == arrays.timestamps_ms.tolist()
    assert [Fraction(r["source_pts_ms"]["numerator"], r["source_pts_ms"]["denominator"]) for r in report["frames"]] == [
        frame.timestamp_ms for frame in frames
    ]
    assert report["warnings"] == [{"code": "POSSIBLE_MULTIPLE_PARTICIPANTS", "source": "human_review"}]
    assert np.isnan(arrays.image_landmarks[1]).all()
    assert system_service.db.path.read_bytes() == before
    fake_pose.trackers[0].close.assert_called_once()


def test_debug_exports_coexist_and_recheck_permission(system, pose_model, fake_pose, model_path, tmp_path):
    system_service, entry, (sample_id,) = export(system)
    decoder = DecodingService.from_registry(system_service.db, entry.name, entry.version)
    pose = PoseExtractionService(decoder, PoseExtractionConfig(pose_model)).process(sample_id)
    hands = HandExtractionService(decoder, HandExtractionConfig(model_path)).process(sample_id)
    root = tmp_path / "debug"
    check = decoder.reader.check_permission
    directory = export_hands(hands, root, check_permission=check)
    assert export_debug(pose, root, check_permission=check) == directory
    assert {path.name for path in directory.iterdir()} == {
        "pose_landmarks.npz", "pose_extraction_report.json", "hand_landmarks.npz", "extraction_report.json",
    }
    with np.load(directory / "pose_landmarks.npz", allow_pickle=False) as archive:
        assert archive["schema_version"].item() == "1.0"
        for name in set(archive.files) - {"schema_version"}:
            np.testing.assert_array_equal(archive[name], getattr(pose.arrays, name))
    assert json.loads((directory / "pose_extraction_report.json").read_text()) == pose.report
    with pytest.raises(PoseOutputError, match="already exists"):
        export_debug(pose, root, check_permission=check)

    root2 = tmp_path / "revocation"
    calls = 0

    def revoke_before_publish():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise DatasetAccessError("Revoked before debug publication.")

    with pytest.raises(DatasetAccessError, match="Revoked"):
        export_debug(pose, root2, check_permission=revoke_before_publish)
    assert not (root2 / sample_id).exists()


def test_diagnostic_file_preserves_source_frame_indices_and_is_not_exportable(clips, pose_model, fake_pose, tmp_path):
    service = PoseExtractionService(None, PoseExtractionConfig(pose_model))
    diagnostic = service.inspect_file(clips["vfr"], all_frames=True)
    assert diagnostic.report["source"] == "diagnostic_file" and diagnostic.report["status"] == "PROCESSED"
    assert diagnostic.arrays.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert diagnostic.arrays.timestamps_ms.tolist() == [1000, 1040, 1120, 1160, 1240]
    assert "NO_POSES_DETECTED" in {row["code"] for row in diagnostic.report["warnings"]}
    with pytest.raises(PoseOutputError, match="approved"):
        export_debug(diagnostic, tmp_path / "out", check_permission=lambda: None)
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"not a video")
    result = service.inspect_file(corrupt)
    assert result.report["status"] == "FAILED" and result.report["error_code"] == "DECODE_FAILED"
    assert result.arrays is None
    for tracker in fake_pose.trackers:
        tracker.close.assert_called_once()


def test_batch_reports_individual_failure_and_continues(system, pose_model, fake_pose):
    system_service, entry, ids = export(system, count=2)
    decoder = DecodingService.from_registry(system_service.db, entry.name, entry.version)
    missing = next(decoder.reader.iter_samples())
    missing.video_path.unlink()
    batch = PoseExtractionService(decoder, PoseExtractionConfig(pose_model)).inspect_dataset()
    assert batch["failed"] == batch["processed"] == 1 and batch["dataset_error"] is None
    assert batch["samples"][0]["sample_id"] == missing.sample_id
    assert batch["samples"][0]["error_code"] == "INVALID_SOURCE_VIDEO"
    assert batch["samples"][1]["sample_id"] in ids
    assert batch["metrics"]["total_processed_frames"] == batch["samples"][1]["metrics"]["total_processed_frames"]
    assert len(fake_pose.trackers) == 2
    for tracker in fake_pose.trackers:
        tracker.close.assert_called_once()


def test_revocation_and_interrupt_return_no_arrays(system, setup, pose_model, fake_pose):
    system_service, entry, (sample_id,) = export(system, mode="PORTABLE")
    _, _, _, _, signer, _ = setup
    decoder = DecodingService.from_registry(system_service.db, entry.name, entry.version)
    service = PoseExtractionService(decoder, PoseExtractionConfig(pose_model))

    def interrupt(*_):
        raise KeyboardInterrupt

    interrupted = service.process(sample_id, on_frame=interrupt)
    assert interrupted.report["status"] == "CANCELLED" and interrupted.arrays is None

    def revoke(*_):
        signers.consent(system_service.db, signer.code, "WITHDRAWN")
        return True

    denied = service.process(sample_id, on_frame=revoke)
    assert denied.arrays is None and not denied.report["frames"]
    assert denied.report["error_code"] == "DATASET_ACCESS_DENIED"
    with pytest.raises(DatasetAccessError, match="REVOKED"):
        DecodingService.from_registry(system_service.db, entry.name, entry.version)
    for tracker in fake_pose.trackers:
        tracker.close.assert_called_once()


def test_cli_options_report_and_output(system, clips, pose_model, fake_pose, tmp_path):
    system_service, entry, (sample_id,) = export(system)
    runner = CliRunner()
    base = ["--db", str(system_service.db.path), "pose", "--fps", "12"]
    selector = ["--dataset", entry.name, "--version", entry.version]

    def invoke(*command):
        result = runner.invoke(app, [*base, *map(str, command), "--model-path", str(pose_model)])
        assert result.exit_code == 0, (result.output, result.exception)
        return json.loads(result.output)

    result = invoke("extract", sample_id, *selector, "--output", tmp_path / "debug", "--multiple-participants")
    assert result["model_variant"] == "lite" and result["warnings"][0]["code"] == "POSSIBLE_MULTIPLE_PARTICIPANTS"
    assert (tmp_path / "debug" / sample_id / "pose_landmarks.npz").is_file()
    assert invoke("inspect", sample_id, *selector)["metrics"]["total_processed_frames"] == 10
    assert invoke("inspect-dataset", *selector, "--multiple-participant-sample", sample_id)["processed"] == 1
    assert invoke("inspect-file", clips[24])["source"] == "diagnostic_file"
    with patch("landmarks.pose_visualizer.PoseLandmarkVisualizer") as preview:
        preview.return_value.__enter__.return_value.show.return_value = False
        output = invoke("visualize", sample_id, *selector, "--full-body", "--skeleton-only", "--show-indices")
        assert output["status"] == "CANCELLED"
    assert preview.call_args.kwargs["full_body"] and preview.call_args.kwargs["show_indices"]
