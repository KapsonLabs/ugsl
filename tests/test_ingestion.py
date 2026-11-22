"""Phase 1.3 validation, ingestion and recovery using synthetic media."""

import hashlib
import json
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from cli import app
from config import QualityConfig
from ingestion import pipeline
from ingestion.capture import CameraCapture, State
from ingestion.hashing import sha256_file
from ingestion.importer import directory_entries, import_batch, manifest_entries
from ingestion.metadata import VideoMetadataExtractor
from ingestion.schemas import IngestionOutcome as Outcome
from ingestion.schemas import ValidationStatus as Status
from ingestion.validator import VideoValidator
from registry import database, media_maintenance, samples, sessions, signers, statistics, vocabulary
from registry.database import Database, DatabaseError
from services.recording_service import capture_registered
from tests.test_capture import Rig, SETTINGS
from tests.test_registry import mock_camera_media


@pytest.fixture
def catalog(tmp_path):
    db = Database(tmp_path / "registry.db")
    root = tmp_path / "dataset" / "raw"
    sign = vocabulary.add(db, gloss="HELLO", target_samples=2)
    signer = signers.add(db, code="TEST001")
    signers.consent(db, signer.code, "granted", "TEST")
    session = sessions.create(db, signer_code=signer.code)
    return pipeline.IngestionService(db, root), sign, signer, session


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], check=True, capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for synthetic media tests.")
    root = tmp_path_factory.mktemp("media")
    for filename, source, duration, codec in (
        ("landscape.mp4", "testsrc2=size=640x480:rate=24", "1.25", "mpeg4"),
        ("portrait.mov", "testsrc2=size=480x640:rate=24", "1.25", "mpeg4"),
        ("landscape.avi", "testsrc2=size=640x480:rate=24", "1.25", "mpeg4"),
        ("landscape.mkv", "testsrc2=size=640x480:rate=24", "1.25", "mpeg4"),
        ("dark.mp4", "color=c=black:s=640x480:r=24", "1.25", "mpeg4"),
        ("small.mp4", "testsrc2=size=320x240:rate=15", "0.6", "mpeg4"),
    ):
        ffmpeg("-f", "lavfi", "-i", source, "-t", duration, "-c:v", codec, str(root / filename))
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=24", "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "1.25", "-c:v", "mpeg4", "-c:a", "aac", "-movflags", "+faststart",
        "-metadata", "creation_time=2026-01-02T09:00:00Z", str(root / "audio.mp4"),
    )
    return root


def ingest(catalog, source, **options):
    service, sign, _, session = catalog
    return service.import_file(source, sign_id=options.pop("sign_id", sign.id),
                               session_id=options.pop("session_id", session.id), **options)


def count(db, table):
    with db.connect() as connection:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_incremental_checksum_independent_of_name(tmp_path):
    data = b"ugsl" * 530_000
    first = tmp_path / "first.mov"
    first.write_bytes(data)
    other = tmp_path / "renamed.mp4"
    shutil.copyfile(first, other)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(first) == sha256_file(other) == expected


@pytest.mark.parametrize("filename, extension, orientation", [
    ("landscape.mp4", ".mp4", "LANDSCAPE"),
    ("portrait.mov", ".mov", "PORTRAIT"),
    ("landscape.avi", ".avi", "LANDSCAPE"),
    ("landscape.mkv", ".mkv", "LANDSCAPE"),
])
def test_extract_inspects_container_and_validates_orientation(media, filename, extension, orientation):
    path = media / filename
    inspected = VideoMetadataExtractor().extract(path)
    assert inspected.extension == extension
    assert inspected.display_orientation == orientation
    assert inspected.nominal_fps == inspected.average_fps == 24
    assert inspected.video_codec == "mpeg4" and inspected.pixel_format == "yuv420p"
    assert inspected.creation_time is None and not inspected.has_audio
    if extension == ".mkv":
        assert VideoMetadataExtractor().extract(path, count_frames=False).frame_count is None
    assert inspected.frame_count == 30
    video, report = VideoValidator().validate(path)
    assert video.frame_count == 30 and video.duration_seconds == pytest.approx(1.25, abs=0.05)
    assert report.validation_status in {Status.PASSED, Status.PASSED_WITH_WARNINGS}
    assert {item.name: item.status for item in report.checks}["minimum_resolution"] == "PASS"


def test_webm_container_under_disguised_extension(tmp_path):
    listing = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    if "libvpx-vp9" not in listing.stdout:
        pytest.skip("This FFmpeg build lacks the optional VP9 encoder for the WebM fixture.")
    path = tmp_path / "disguised.avi"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x480:rate=24", "-t", "1.25", "-c:v",
           "libvpx-vp9", "-deadline", "realtime", "-f", "webm", str(path))
    meta = VideoMetadataExtractor().extract(path)
    assert meta.extension == ".webm" and meta.video_codec == "vp9"
    _, report = VideoValidator().validate(path)
    assert report.validation_status in {Status.PASSED, Status.PASSED_WITH_WARNINGS}


def test_probe_creation_time_audio_rotation_and_unknown_fields(media, monkeypatch):
    audio = VideoMetadataExtractor().extract(media / "audio.mp4")
    assert audio.has_audio
    assert audio.creation_time == "2026-01-02T09:00:00+00:00"
    assert audio.bitrate and audio.file_size_bytes > 0

    def fake_probe(_, path, options):
        if "stream=index" in options:
            return {"streams": []}
        return {"streams": [{"width": 1280, "height": 720, "avg_frame_rate": "0/0",
                             "r_frame_rate": "24/1", "nb_frames": "N/A", "side_data_list": [{"rotation": -90}]}],
                "format": {"format_name": "mov,mp4", "tags": {"major_brand": "isom"}}}

    monkeypatch.setattr(VideoMetadataExtractor, "_probe", fake_probe)
    rotated = VideoMetadataExtractor().extract(media / "landscape.mp4")
    assert rotated.rotation == -90 and rotated.display_orientation == "PORTRAIT"
    assert rotated.average_fps is None and rotated.nominal_fps == 24
    assert rotated.duration_seconds is rotated.frame_count is rotated.bitrate is None
    _, report = VideoValidator().validate(media / "landscape.mp4")
    assert report.validation_status == Status.FAILED
    assert "Frame rate is missing or unknown." in report.errors


def test_full_decode_rejects_corrupt_and_truncated_files(media, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a container")
    truncated = tmp_path / "truncated.mp4"
    data = (media / "audio.mp4").read_bytes()
    truncated.write_bytes(data[:len(data) // 2])
    for path in (corrupt, truncated, tmp_path / "missing.mp4"):
        _, report = VideoValidator().validate(path)
        assert report.validation_status == Status.FAILED
        assert report.checks[0].name == "video_integrity"
    oversize = QualityConfig(max_file_size_mb=0.001)
    _, report = VideoValidator(oversize).validate(media / "landscape.mp4")
    assert "file size limit" in report.errors[0]


def test_ffprobe_or_ffmpeg_timeout_is_operational(media, monkeypatch):
    def timeout(args, **_):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    _, report = VideoValidator().validate(media / "landscape.mp4")
    assert report.validation_status == Status.ERROR
    assert report.errors == ["ffprobe exceeded its time limit."]


def test_thresholds_and_visual_heuristics(media):
    _, low = VideoValidator().validate(media / "small.mp4")
    assert low.validation_status == Status.FAILED
    assert {"minimum_resolution", "frame_rate", "duration", "frame_count"} <= {
        item.name for item in low.checks if item.status == "FAIL"
    }
    _, dark = VideoValidator(QualityConfig(visual_samples=3)).validate(media / "dark.mp4")
    assert dark.validation_status == Status.PASSED_WITH_WARNINGS
    visual = dark.metrics["visual"]
    assert visual["sampled_frames"] == 3 and visual["underexposed_percent"] > 99
    assert visual["mean_brightness"] < 5 and visual["brightness_variability"] == 0
    assert visual["mean_sharpness"] == visual["mean_motion"] == visual["peak_motion"] == 0


def test_import_preserves_original_and_leaves_unknown_capture_fields_null(catalog, media, tmp_path):
    source = tmp_path / "phone-video.avi"
    shutil.copyfile(media / "audio.mp4", source)
    expected = source.read_bytes()
    result = ingest(catalog, source, recorded_at="2026-01-02T12:00:00+03:00",
                    source_reference="Field phone", notes="Front view")
    assert result.outcome == Outcome.INGESTED
    service, sign, _, session = catalog
    sample = samples.show(service.db, result.sample_id)
    assert sample.source_type == "IMPORT" and sample.status == "CAPTURED"
    assert sample.technical_validation_status == Status.PASSED
    assert sample.recorded_at == "2026-01-02T09:00:00+00:00"
    assert (sample.sign_id, sample.session_id) == (sign.id, session.id)
    assert sample.measured_fps is sample.requested_fps is sample.actual_duration_seconds is None
    assert sample.encoded_fps is sample.repetition is None
    assert sample.average_fps == 24 and sample.duration_seconds == pytest.approx(1.25)
    video = service.storage.raw / sample.file_path
    assert video.suffix == ".mp4" and video.read_bytes() == source.read_bytes() == expected
    assert sample.sha256 == hashlib.sha256(expected).hexdigest()
    sidecar = json.loads((service.storage.raw / sample.metadata_path).read_text())
    assert sidecar["source_reference"] == "Field phone" and sidecar["notes"] == "Front view"
    assert sidecar["video_metadata"]["has_audio"] and sidecar["validation"]["validator_version"] == "1.0"
    assert count(service.db, "ingestion_attempts") == count(service.db, "video_validations") == 1
    assert statistics.summary(service.db).footage_seconds == pytest.approx(1.25)
    assert not list(service.storage.staging.iterdir())


def test_duplicate_filename_sign_signer_and_session_conflicts(catalog, media, tmp_path):
    service, sign, signer, _ = catalog
    initial = ingest(catalog, media / "landscape.mp4")
    alias = tmp_path / "different.mov"
    shutil.copyfile(media / "landscape.mp4", alias)
    another = sessions.create(service.db, signer_code=signer.code)
    duplicate = ingest(catalog, alias, session_id=another.id)
    assert duplicate.outcome == Outcome.DUPLICATE and duplicate.sample_id == initial.sample_id
    different_sign = vocabulary.add(service.db, gloss="WATER")
    conflict = ingest(catalog, alias, sign_id=different_sign.id)
    assert conflict.outcome == Outcome.CONFLICT and conflict.sample_id == initial.sample_id
    other = signers.add(service.db, code="TEST002")
    signers.consent(service.db, other.code, "GRANTED")
    other_session = sessions.create(service.db, signer_code=other.code)
    assert ingest(catalog, alias, session_id=other_session.id).error_code == "DUPLICATE_METADATA_CONFLICT"
    assert count(service.db, "video_samples") == 1 and count(service.db, "ingestion_attempts") == 4


def test_concurrent_import_registers_only_one_sample(catalog, media):
    def attempt(_):
        return ingest(catalog, media / "landscape.mp4")

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, range(2)))
    assert {result.outcome for result in results} == {Outcome.INGESTED, Outcome.DUPLICATE}
    assert len({result.sample_id for result in results}) == 1
    assert count(catalog[0].db, "video_samples") == 1


def test_directory_discovery_symlinks_and_batch_summary(catalog, media, tmp_path):
    service, sign, _, session = catalog
    inputs = tmp_path / "inputs"
    nested = inputs / "more"
    nested.mkdir(parents=True)
    shutil.copyfile(media / "landscape.mp4", inputs / "good.mp4")
    shutil.copyfile(media / "dark.mp4", nested / "dark.mp4")
    (inputs / "link").symlink_to(media, target_is_directory=True)
    (inputs / "link.mp4").symlink_to(media / "audio.mp4")
    (inputs / "bad.avi").write_bytes(b"corrupt")
    assert len(directory_entries(inputs, sign.id, session.id)) == 2
    entries = directory_entries(inputs, sign.id, session.id, recursive=True)
    report = import_batch(service, entries)
    summary = report.summary()
    assert summary["discovered"] == 3
    assert summary["ingested"] == 2 and summary["ingested_with_warnings"] == 1 and summary["rejected"] == 1
    assert summary["storage_bytes"] > 0 and summary["duration_seconds"] > 0
    assert not list(service.storage.staging.iterdir())


def test_manifest_dry_run_batch_duplicates_conflicts_and_fail_fast(catalog, media, tmp_path):
    service, sign, _, session = catalog
    other = vocabulary.add(service.db, gloss="WATER")
    manifest = tmp_path / "manifest.csv"
    for filename in ("landscape.mp4", "portrait.mov"):
        shutil.copyfile(media / filename, tmp_path / filename)
    manifest.write_text(
        "file_path,sign_id,session_id,recorded_at,notes\n"
        f"landscape.mp4,{sign.id},{session.id},2026-01-02T09:00:00Z,Front\n"
        f"landscape.mp4,{sign.id},{session.id},,Repeat\n"
        f"landscape.mp4,{other.id},{session.id},,Different gloss\n"
        f"portrait.mov,{sign.id},{session.id},,Portrait\n"
        ",,,invalid,\n",
        encoding="utf-8",
    )
    entries = manifest_entries(manifest)
    before = service.db.path.read_bytes()
    report = import_batch(service, entries, dry_run=True)
    assert [item.outcome for item in report.results] == [
        Outcome.INGESTED, Outcome.DUPLICATE, Outcome.CONFLICT, Outcome.INGESTED, Outcome.REJECTED,
    ]
    assert report.summary()["storage_bytes"] == 0 and report.summary()["projected_storage_bytes"] > 0
    assert service.db.path.read_bytes() == before and not service.storage.raw.parent.exists()
    persisted = import_batch(service, entries, fail_fast=True)
    assert persisted.summary()["discovered"] == 5 and persisted.summary()["processed"] == 3
    assert count(service.db, "ingestion_attempts") == 3
    assert count(service.db, "video_samples") == 1


def test_consent_enforcement_and_opt_in_quarantine(catalog, media):
    service, _, signer, _ = catalog
    invalid = media / "small.mp4"
    result = ingest(catalog, invalid, quarantine=True)
    assert result.outcome == Outcome.REJECTED
    quarantined = service.storage.quarantine / result.attempt_id / "video"
    assert quarantined.read_bytes() == invalid.read_bytes()
    assert count(service.db, "video_samples") == 0
    signers.consent(service.db, signer.code, "withdrawn")
    second = ingest(catalog, invalid, quarantine=True)
    assert second.outcome == Outcome.CONFLICT and "WITHDRAWN" in second.message
    assert len(list(service.storage.quarantine.iterdir())) == 1
    dry = media_maintenance.reconcile(service, dry_run=True, purge_quarantine=True)
    assert dry.actions and quarantined.exists()
    done = media_maintenance.reconcile(service, purge_quarantine=True)
    assert done.actions and not quarantined.exists() and invalid.exists()


def test_container_creation_time_is_not_used_to_infer_consent(catalog, media):
    service, _, signer, _ = catalog
    signers.consent(service.db, signer.code, "WITHDRAWN")
    signers.consent(service.db, signer.code, "GRANTED")
    result = ingest(catalog, media / "audio.mp4")
    assert result.outcome == Outcome.CONFLICT
    assert "verified recording timestamp is required" in result.message
    assert count(service.db, "video_samples") == 0
    assert not list(service.storage.staging.iterdir())


def test_registration_failure_reconcile_preserves_checksum_uuid_and_audit(catalog, media, monkeypatch):
    service, _, _, _ = catalog
    original = pipeline.save_validation

    def fail_after_publish(*_):
        raise DatabaseError("Test rollback")

    with monkeypatch.context() as patcher:
        patcher.setattr(pipeline, "save_validation", fail_after_publish)
        failed = ingest(catalog, media / "landscape.mp4")
    assert failed.outcome == Outcome.ERROR and "registry reconcile" in failed.message
    pending = service.storage.staging / failed.attempt_id
    request = json.loads((pending / "request.json").read_text())
    assert count(service.db, "video_samples") == 0
    before = service.db.path.read_bytes()
    preview = media_maintenance.reconcile(service, dry_run=True)
    assert len(preview.actions) == 1 and not preview.issues
    assert service.db.path.read_bytes() == before and pending.exists()
    assert pipeline.save_validation is original
    recovered = media_maintenance.reconcile(service)
    assert len(recovered.actions) == 1 and not recovered.issues and not pending.exists()
    sample = samples.show(service.db, request["sample_id"])
    assert sample.sha256 == failed.sha256
    with service.db.connect() as connection:
        attempt = connection.execute("SELECT * FROM ingestion_attempts WHERE id = ?", (failed.attempt_id,)).fetchone()
        assert attempt["sample_id"] == sample.id and attempt["result"] == Outcome.INGESTED
    assert not media_maintenance.reconcile(service).actions


def test_revalidation_history_checks_hash_and_keeps_linguistic_status(catalog, media):
    service, _, _, _ = catalog
    result = ingest(catalog, media / "landscape.mp4")
    sample = samples.show(service.db, result.sample_id)
    (service.storage.raw / sample.file_path).write_bytes(b"altered")
    assert media_maintenance.validate_sample(service, sample.id, dry_run=True).validation_status == Status.FAILED
    assert count(service.db, "video_validations") == 1
    assert media_maintenance.validate_sample(service, sample.id).validation_status == Status.FAILED
    updated, = samples.list_samples(service.db, validation_status="failed")
    assert updated.status == "CAPTURED" and updated.id == sample.id
    assert count(service.db, "video_validations") == 2


def test_old_schema_dry_run_and_backfill_are_safe_and_idempotent(tmp_path, media):
    db = Database(tmp_path / "registry.db")
    root = tmp_path / "raw"
    root.mkdir()
    migrations = tmp_path / "original-migrations"
    migrations.mkdir()
    shutil.copyfile(database.MIGRATIONS / "001_initial.sql", migrations / "001_initial.sql")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(database, "MIGRATIONS", migrations)
        sign = vocabulary.add(db, gloss="WATER")
        signer = signers.add(db, code="TEST001")
        signers.consent(db, signer.code, "GRANTED")
        session = sessions.create(db, signer_code=signer.code)
        with db.connect(write=True) as connection:
            for name, fixture in (("unique", "portrait.mov"), ("copy-a", "landscape.mp4"), ("copy-b", "landscape.mp4")):
                path = root / f"{name}.mp4"
                shutil.copyfile(media / fixture, path)
                connection.execute(
                    """INSERT INTO video_samples (id, sign_id, session_id, repetition, file_path, metadata_path,
                       width, height, requested_fps, measured_fps, encoded_fps, frame_count, failed_frame_reads,
                       requested_duration_seconds, actual_duration_seconds, file_size_bytes, recorded_at, registered_at)
                       VALUES (?, ?, ?, 1, ?, ?, 640, 480, 24, 24, 24, 30, 0, 1.25, 1.25, ?, ?, ?)""",
                    (str(uuid4()), sign.id, session.id, path.name, name + ".json", path.stat().st_size,
                     "2026-01-02T09:00:00+00:00", "2026-01-02T09:00:00+00:00"),
                )
    service = pipeline.IngestionService(db, root)
    previous = db.path.read_bytes()
    dry = media_maintenance.backfill_hashes(service, dry_run=True)
    assert len(dry.actions) == 1 and any("Duplicate" in line for line in dry.issues)
    assert db.path.read_bytes() == previous
    with sqlite3.connect(db.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    done = media_maintenance.backfill_hashes(service)
    assert len(done.actions) == 1
    assert not media_maintenance.backfill_hashes(service).actions
    rows = samples.list_samples(db)
    assert len(rows) == 3 and all(row.technical_validation_status == Status.PENDING for row in rows)
    assert sum(row.sha256 is not None for row in rows) == 1
    assert all((root / row.file_path).exists() for row in rows)
    with db.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    duplicate = service.import_file(media / "landscape.mp4", sign_id=sign.id, session_id=session.id)
    assert duplicate.outcome == Outcome.DUPLICATE


def test_camera_technical_rejection_allows_retry_without_incrementing(catalog, monkeypatch, mock_camera_media):
    from ingestion.schemas import ValidationReport

    service, sign, _, session = catalog
    validate = VideoValidator.validate
    calls = []

    def reject_first(self, path):
        calls.append(path)
        if len(calls) == 1:
            return None, ValidationReport(Status.FAILED, [], ["Camera quality test."], [], {})
        return validate(self, path)

    rig = Rig(monkeypatch)
    rig.keys[State.READY].extend([" ", " "])
    rig.keys[State.REVIEW].extend(["a", "a"])
    monkeypatch.setattr(VideoValidator, "validate", reject_first)
    sample, = capture_registered(service.db, CameraCapture(SETTINGS), service.storage.raw,
                                 sign_id=sign.id, session_id=session.id)
    assert sample.repetition == 1 and len(calls) == 2
    stored, = samples.list_samples(service.db)
    assert stored.id == sample.sample_id and stored.source_type == "CAMERA"
    assert count(service.db, "ingestion_attempts") == 2
    assert not list(service.storage.staging.iterdir())


def test_cli_inspect_ingest_batch_validate_and_filter(catalog, media, tmp_path):
    service, sign, _, session = catalog
    base = ["--db", str(service.db.path), "--dataset-root", str(service.storage.raw)]
    cli = CliRunner()

    def invoke(*args):
        result = cli.invoke(app, [*base, *map(str, args)])
        assert result.exit_code == 0, (result.output, result.exception)
        return result.output

    assert json.loads(invoke("video", "inspect", media / "audio.mp4"))["has_audio"]
    assert json.loads(invoke("video", "validate", media / "dark.mp4"))["warnings"]
    output = json.loads(invoke("ingest", "file", media / "landscape.mp4", "--sign", sign.id, "--session", session.id))
    assert output["outcome"] == Outcome.INGESTED
    sid = output["sample_id"]
    assert sid in invoke("sample", "list", "--validation-status", "passed")
    assert json.loads(invoke("sample", "validate", sid))["validation_status"] == Status.PASSED
    all_report = json.loads(invoke("sample", "validate-all", "--dry-run", "--json"))
    assert all_report["summary"]["passed"] == 1
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"file_path,sign_id,session_id\n{media / 'landscape.mp4'},{sign.id},{session.id}\n")
    batch = json.loads(invoke("ingest", "manifest", manifest, "--dry-run", "--json"))
    assert batch["summary"]["duplicates"] == 1
    assert "0 actions" in invoke("registry", "reconcile", "--dry-run")


def test_cli_quality_thresholds_can_be_overridden(media, monkeypatch):
    monkeypatch.setenv("UGSL_QUALITY_MIN_FPS", "30")
    result = CliRunner().invoke(app, ["video", "validate", str(media / "landscape.mp4")])
    assert result.exit_code == 1
    assert json.loads(result.output)["errors"] == ["Frame rate is outside the isolated-sign limits."]
