"""Phase 1.5 export integration and read-only decoding workflows."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from dataset_export.models import ExportConfig
from dataset_export.reader import DatasetReader
from landmarks.exceptions import DatasetAccessError, SampleNotFound, SourceVideoError
from landmarks.preview import _caption
from landmarks.service import DecodingService, inspect_file
from registry import datasets, signers
from tests.test_annotation import setup
from tests.test_dataset_export import system
from tests.test_ingestion import media
from tests.test_video_decoder import clips


def export(system, *, mode="REFERENCE", count=1):
    service, approve = system
    ids = [approve()]
    if count > 1:
        ids.append(approve("portrait.mov"))
    entry = service.export(ExportConfig("frames-test", "1.0.0", mode=mode))
    return service, entry, ids


@pytest.mark.parametrize("mode", ["REFERENCE", "PORTABLE"])
def test_approved_intervals_and_read_only_processing(system, mode):
    service, entry, (sample_id,) = export(system, mode=mode)
    before = service.db.path.read_bytes()
    decoder = DecodingService.from_registry(service.db, entry.name, entry.version)
    with decoder.process(sample_id) as job:
        frames = list(job)
    report = job.summary
    assert report.status == "PROCESSED" and report.sample_id == sample_id
    assert (report.dataset_name, report.dataset_version) == (entry.name, entry.version)
    assert (report.start_ms, report.end_ms, report.source_fps, report.target_fps) == (100, 900, 24, 30)
    assert report.selected_frames == report.decoded_frames == len(frames) == 19
    assert report.skipped_frames == 0 and report.source_frames_decoded > report.decoded_frames
    assert report.effective_output_fps == 23.75 and report.sampling_gaps
    assert frames[0].frame_index == 3 and frames[0].timestamp_ms == 125
    assert frames[0].relative_timestamp_ms == 25 and frames[-1].timestamp_ms == 875
    assert report.first_selected_timestamp_ms == 125 and report.last_selected_timestamp_ms == 875
    assert report.decoder_configuration["pixel_format"] == "rgb24"
    assert service.db.path.read_bytes() == before
    assert len(datasets.list_versions(service.db)) == 1


def test_moved_portable_dataset_uses_its_own_media(system, tmp_path, monkeypatch):
    _, entry, (sample_id,) = export(system, mode="PORTABLE")
    path = tmp_path / "moved-export"
    shutil.copytree(entry.output_path, path)
    monkeypatch.delenv("UGSL_REGISTRY_DB", raising=False)
    service = DecodingService(DatasetReader(path), all_frames=True)
    with service.process(sample_id) as job:
        frames = list(job)
    assert job.summary.target_fps is None and len(frames) == job.summary.decoded_frames
    assert Path(job.summary.source_video_path).is_relative_to(path)


def test_batch_continues_after_missing_video(system):
    service, entry, ids = export(system, count=2)
    reader = DatasetReader(entry.output_path, storage_root=service.raw, registry_path=service.db.path)
    # Remove the first manifest video so the next row must be processed after the failure.
    missing = next(reader.iter_samples())
    missing.video_path.unlink()
    job = DecodingService(reader).validate_dataset()
    assert job.dataset_error is None and job.processed == job.failed == 1
    assert job.samples[0].sample_id == missing.sample_id
    assert job.samples[0].status == "FAILED" and job.samples[0].error_code == "INVALID_SOURCE_VIDEO"
    assert job.samples[1].status == "PROCESSED" and job.samples[1].sample_id in ids


def test_tampered_portable_video_fails_manifest_checksum(system):
    service, entry, (sample_id,) = export(system, mode="PORTABLE")
    reader = DatasetReader(entry.output_path, registry_path=service.db.path)
    sample = next(reader.iter_samples())
    sample.video_path.write_bytes(b"x" * sample.file_size_bytes)
    batch = DecodingService(reader).validate_dataset()
    assert batch.failed == 1 and batch.processed == 0
    assert "checksum" in batch.samples[0].error
    assert batch.samples[0].source_frames_decoded == 0
    with DecodingService(reader).process(sample_id) as job:
        with pytest.raises(SourceVideoError):
            next(job.frames)
    assert job.summary.selected_frames == 0


def test_consent_withdrawal_blocks_frames_during_an_open_run(system, setup):
    service, entry, (sample_id,) = export(system, mode="PORTABLE")
    _, _, _, _, signer, _ = setup
    decoder = DecodingService.from_registry(service.db, entry.name, entry.version)
    with decoder.process(sample_id) as job:
        assert next(job.frames).sample_id == sample_id
        signers.consent(service.db, signer.code, "WITHDRAWN")
        with pytest.raises(DatasetAccessError, match="REVOKED"):
            next(job.frames)
    assert job.summary.status == "FAILED" and job.summary.selected_frames == 1
    with pytest.raises(DatasetAccessError, match="REVOKED"):
        DecodingService.from_registry(service.db, entry.name, entry.version)


def test_unknown_dataset_and_sample_are_actionable(system):
    service, entry, _ = export(system)
    with pytest.raises(DatasetAccessError, match="Unknown dataset"):
        DecodingService.from_registry(service.db, "missing", "1.0.0")
    decoder = DecodingService.from_registry(service.db, entry.name, entry.version)
    with pytest.raises(SampleNotFound, match="absent"):
        decoder.process("missing")


def test_early_stop_is_not_successful_processing(clips):
    with inspect_file(clips[30]) as job:
        next(job.frames)
    assert job.summary.selected_frames == 1
    assert job.summary.status == "CANCELLED"
    assert job.summary.effective_output_fps is None


def test_cli_diagnostic_sampling_and_contact_sheet(system, clips, tmp_path):
    service, entry, (sample_id,) = export(system)
    runner = CliRunner()
    base = ["--db", str(service.db.path), "frames"]
    selector = ["--dataset", entry.name, "--version", entry.version]

    def invoke(*args):
        result = runner.invoke(app, [*base, *map(str, args)])
        assert result.exit_code == 0, (result.output, result.exception)
        return json.loads(result.output)

    assert invoke("inspect-file", clips[60])["selected_frames"] == 60
    assert invoke("inspect", sample_id, *selector)["target_fps"] is None
    assert invoke("sample", sample_id, *selector, "--fps", 12)["selected_frames"] == 10
    assert invoke("sample", sample_id, *selector, "--all-frames")["selected_frames"] == 19
    assert invoke("validate-dataset", *selector)["processed"] == 1
    sheet = tmp_path / "sample.png"
    assert invoke("contact-sheet", sample_id, *selector, "--output", sheet, "--limit", 2)["status"] == "PROCESSED"
    assert cv2.imread(str(sheet)).shape == (210, 640, 3)
    previous = sheet.read_bytes()
    failure = runner.invoke(app, [*base, "contact-sheet", sample_id, *selector, "--output", str(sheet)])
    assert failure.exit_code == 1 and sheet.read_bytes() == previous


def test_preview_overlay_does_not_modify_decoded_rgb(clips):
    with inspect_file(clips[30]) as job:
        frame = next(job.frames)
        original = frame.image.copy()
        preview = _caption(frame, job.summary)
    assert np.array_equal(frame.image, original)
    assert not np.shares_memory(frame.image, preview)
    assert preview[-1, -1, 2] == original[-1, -1, 0]
