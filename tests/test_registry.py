"""Registry and camera integration tests backed by temporary databases."""

import json
import shutil
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from typer.testing import CliRunner

from cli import app
from ingestion import pipeline
from ingestion.capture import CameraCapture, State
from ingestion.schemas import ValidationReport, ValidationStatus, VideoMetadata
from ingestion.validator import VideoValidator
from registry import database, media_maintenance, samples, sessions, signers, statistics, vocabulary
from registry.database import Database, DatabaseError, RegistryError
from services.recording_service import capture_registered
from tests.test_capture import FakeWriter, Rig, SETTINGS


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "registry.db")


@pytest.fixture
def root(tmp_path):
    return tmp_path / "datasets" / "raw"


@pytest.fixture
def catalog(db):
    sign = vocabulary.add(db, gloss=" water ", category="Food and drinks", sign_type="dynamic", target_samples=3)
    signer = signers.add(db, code="signer001", proficiency="fluent")
    signers.consent(db, signer.code, "granted", "CONSENT001")
    session = sessions.create(db, signer_code=signer.code)
    return sign, signer, session


def pair(root, *, sample_id=None, directory=("WATER", "SIGNER001"), **changes):
    sample_id = sample_id or str(uuid4())
    path = root.joinpath(*directory, f"{sample_id}.mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(UUID(sample_id).bytes[:8])
    metadata = {
        "sample_id": sample_id, "gloss": "WATER", "signer_id": "SIGNER001",
        "repetition": 1, "camera_index": 0, "width": 1280, "height": 720,
        "requested_fps": 30, "measured_fps": 29.8, "encoded_fps": 30,
        "frame_count": 89, "failed_frame_reads": 0, "requested_duration_seconds": 3,
        "actual_duration_seconds": 3.02, "file_size_bytes": 8,
        "file_path": str(Path("datasets/raw") / path.relative_to(root)),
        "recorded_at": "2026-01-02T12:00:00+03:00",
        **changes,
    }
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return sidecar


def test_lazy_database_initialization_and_foreign_keys(db):
    assert not db.path.exists()
    with db.connect(write=True) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """INSERT INTO recording_sessions
                   (id, signer_id, camera_source_type, camera_angle, lighting_condition, background, started_at)
                   VALUES (?, ?, 'LOCAL', 'FRONT', 'OTHER', 'OTHER', ?)""",
                ("test-session", "missing-signer", "2026-01-02T09:00:00+00:00"),
            )
    with Database(db.path).connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"signs", "signers", "signer_consent_events", "recording_sessions", "video_samples"} <= tables


def test_migrations_are_atomic_idempotent_and_preserve_existing_rows(db, tmp_path, monkeypatch):
    sign = vocabulary.add(db, gloss="HELLO")
    migrations = tmp_path / "migrations"
    shutil.copytree(database.MIGRATIONS, migrations)
    monkeypatch.setattr(database, "MIGRATIONS", migrations)
    script = migrations / "005_test.sql"
    script.write_text("CREATE TABLE migration_test (name TEXT); INVALID SQL;")
    with pytest.raises(RegistryError, match="migration failed"):
        vocabulary.show(db, sign.id)
    with sqlite3.connect(db.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'migration_test'").fetchone() is None
    script.write_text("CREATE TABLE migration_test (name TEXT);")
    for _ in range(2):
        assert vocabulary.show(Database(db.path), sign.id) == sign
        with db.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5


def test_write_transaction_rolls_back_and_database_error_is_sanitized(db, tmp_path):
    with pytest.raises(RuntimeError, match="rollback test"):
        with db.connect(write=True) as connection:
            connection.execute("UPDATE sign_sequence SET last_id = ?", (99,))
            raise RuntimeError("rollback test")
    assert vocabulary.add(db, gloss="WATER").id == "UGSL0001"
    blocked = tmp_path / "private-location"
    blocked.touch()
    with pytest.raises(RegistryError) as error:
        vocabulary.list_signs(Database(blocked / "registry.db"))
    assert "database operation failed" in str(error.value)
    assert "private-location" not in str(error.value)


def test_vocabulary_normalization_variants_filters_and_permanent_id(db, catalog):
    first, _, _ = catalog
    assert first.id == "UGSL0001"
    assert first.gloss == "WATER"
    with pytest.raises(RegistryError, match="already exist"):
        vocabulary.add(db, gloss="ｗａｔｅｒ")
    variant = vocabulary.add(db, gloss="Water", variant_label=" central ", sign_type="static")
    assert variant.id == "UGSL0002"
    with pytest.raises(RegistryError, match="already exist"):
        vocabulary.update(db, variant.id, variant_label="")
    changed = vocabulary.update(db, first.id, gloss="  drinking   water ", target_samples=10, is_active=False)
    assert changed.id == first.id and changed.created_at == first.created_at
    assert changed.gloss == "DRINKING WATER"
    assert changed.target_samples == 10
    assert vocabulary.list_signs(db, category="food and drinks", sign_type="dynamic", active=False) == [changed]
    assert vocabulary.list_signs(db, active=True) == [variant]
    with pytest.raises(RegistryError, match="Unknown sign"):
        vocabulary.show(db, "UNKNOWN")


def test_signer_consent_and_session_lifecycle(db):
    signer = signers.add(db, code="person01", dominant_hand="left", proficiency="native", age_group="adult")
    assert UUID(signer.id).version == 4 and signer.consent_status == "UNKNOWN"
    with pytest.raises(RegistryError, match="already registered"):
        signers.add(db, code="PERSON01")
    updated = signers.update(db, signer.code, region="Central", age_group="")
    assert updated.age_group is None and updated.region == "Central"
    for status in ("UNKNOWN", "DENIED", "WITHDRAWN"):
        signers.consent(db, signer.code, status)
        with pytest.raises(RegistryError, match=status):
            sessions.create(db, signer_code=signer.code)
    granted, _ = signers.consent(db, signer.code, "granted", "REF001")
    assert granted.consent_reference == "REF001"
    assert datetime.fromisoformat(granted.consent_recorded_at).utcoffset() == timedelta(0)
    session = sessions.create(db, signer_code=signer.code, camera_source_type="remote", operator="OP1",
                              camera_angle="left", lighting_condition="natural", background="plain")
    assert session.signer_id == signer.id and session.status == "ACTIVE"
    assert (session.camera_source_type, session.camera_angle, session.lighting_condition) == ("REMOTE", "LEFT", "NATURAL")
    completed = sessions.finish(db, session.id, "completed")
    assert completed.status == "COMPLETED" and completed.ended_at is not None
    assert sessions.finish(db, session.id, "completed") == completed
    with pytest.raises(RegistryError, match="already COMPLETED"):
        sessions.finish(db, session.id, "cancelled")
    signers.update(db, signer.code, is_active=False)
    with pytest.raises(RegistryError, match="inactive"):
        sessions.create(db, signer_code=signer.code)
    assert sessions.list_sessions(db) == [completed]


def test_register_legacy_pair_preserves_metadata_and_path(db, root, catalog):
    sign, _, session = catalog
    sidecar = pair(root)
    meta = json.loads(sidecar.read_text())
    accepted = samples.register(db, root, sidecar)
    assert accepted.created
    sample = accepted.sample
    assert sample.id == meta["sample_id"]
    assert sample.sign_id == sign.id and sample.session_id == session.id
    assert sample.sample_type == "ISOLATED" and sample.status == "CAPTURED"
    assert sample.recorded_at == "2026-01-02T09:00:00+00:00"
    assert datetime.fromisoformat(sample.registered_at).utcoffset() == timedelta(0)
    assert sample.file_path == sidecar.with_suffix(".mp4").relative_to(root).as_posix()
    assert sample.metadata_path == sidecar.relative_to(root).as_posix()
    assert samples.show(db, sample.id) == sample
    assert samples.list_samples(db, sign_id=sign.id) == [sample]
    assert not samples.register(db, root, sidecar).created

    duplicate = pair(root, sample_id=sample.id, directory=("duplicate",))
    with pytest.raises(RegistryError, match="conflicts"):
        samples.register(db, root, duplicate)
    with db.connect(write=True) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="video_samples.file_path"):
            connection.execute(
                """INSERT INTO video_samples (id, sign_id, session_id, sample_type, repetition,
                       file_path, metadata_path, width, height, requested_fps, measured_fps, encoded_fps,
                       frame_count, failed_frame_reads, requested_duration_seconds, actual_duration_seconds,
                       file_size_bytes, status, recorded_at, registered_at)
                       SELECT ?, sign_id, session_id, sample_type, repetition,
                   file_path, metadata_path, width, height, requested_fps, measured_fps, encoded_fps,
                   frame_count, failed_frame_reads, requested_duration_seconds, actual_duration_seconds,
                   file_size_bytes, status, recorded_at, registered_at FROM video_samples WHERE id = ?""",
                (str(uuid4()), sample.id),
            )
    assert len(samples.list_samples(db)) == 1


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"sample_id": "invalid"}, "Sample ID"),
        ({"width": 0}, "width"),
        ({"frame_count": True}, "frame_count"),
        ({"encoded_fps": float("nan")}, "encoded_fps"),
        ({"requested_fps": "30"}, "requested_fps"),
        ({"measured_fps": 0}, "measured_fps"),
        ({"failed_frame_reads": -1}, "failed_frame_reads"),
        ({"recorded_at": "2026-01-02T12:00:00"}, "UTC offset"),
        ({"file_path": "../other.mp4"}, "does not match"),
        ({"file_path": "/etc/unsafe.mp4"}, "does not match"),
        ({"file_path": "folder\\unsafe.mp4"}, "unsafe"),
        ({"file_size_bytes": 999}, "file size"),
    ],
)
def test_invalid_metadata_is_not_registered(db, root, catalog, changes, expected):
    sidecar = pair(root)
    meta = json.loads(sidecar.read_text())
    meta.update(changes)
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(RegistryError, match=expected):
        samples.register(db, root, sidecar)
    assert not samples.list_samples(db)


def test_missing_and_symlinked_files_are_reported(db, root, catalog, tmp_path):
    sidecar = pair(root)
    video = sidecar.with_suffix(".mp4")
    video.unlink()
    with pytest.raises(RegistryError, match="MP4 video is missing"):
        samples.register(db, root, sidecar)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"mock mp4")
    video.symlink_to(outside)
    with pytest.raises(RegistryError, match="Symlinked"):
        samples.register(db, root, sidecar)
    sidecar.unlink()
    video.unlink()
    with pytest.raises(RegistryError, match="sidecar is missing"):
        samples.register(db, root, sidecar)


def test_sync_dry_run_reports_then_registers_and_skips_on_revisit(db, root, catalog):
    sign, _, session = catalog
    valid = pair(root, directory=("legacy", "nested"))
    unknown = pair(root, signer_id="UNKNOWN")
    missing = pair(root)
    missing.with_suffix(".mp4").unlink()
    broken = pair(root)
    broken.write_text("{")
    before = db.path.read_bytes()
    proposed = samples.synchronize(db, root, dry_run=True)
    assert proposed.registered == [valid.relative_to(root).as_posix()]
    assert len(proposed.issues) == 3 and proposed.skipped == 0
    assert any("Unknown signer" in message for path, message in proposed.issues if path.endswith(unknown.name))
    assert db.path.read_bytes() == before
    assert not samples.list_samples(db)
    result = samples.synchronize(db, root, sign_id=sign.id, session_id=session.id)
    assert result.registered == proposed.registered and len(result.issues) == 3
    repeated = samples.synchronize(db, root)
    assert not repeated.registered and repeated.skipped == 1
    assert len(signers.list_signers(db)) == 1


def test_dry_run_does_not_create_an_absent_database(db, root):
    pair(root)
    report = samples.synchronize(db, root, dry_run=True)
    assert len(report.issues) == 1
    assert not db.path.exists()


def test_legacy_sync_requires_unambiguous_sign_and_session(db, root, catalog):
    original, signer, first = catalog
    variant = vocabulary.add(db, gloss=original.gloss, variant_label="Central")
    second = sessions.create(db, signer_code=signer.code)
    sidecar = pair(root)
    with pytest.raises(RegistryError, match="multiple variants"):
        samples.register(db, root, sidecar)
    with pytest.raises(RegistryError, match="No unique active session"):
        samples.register(db, root, sidecar, sign_id=variant.id)
    registered = samples.register(db, root, sidecar, sign_id=variant.id, session_id=second.id).sample
    assert registered.sign_id == variant.id and registered.session_id != first.id


@pytest.mark.parametrize("status", ["COMPLETED", "CANCELLED"])
def test_closed_sessions_reject_new_samples_but_skip_existing(db, root, catalog, status):
    _, _, session = catalog
    existing = pair(root)
    samples.register(db, root, existing)
    sessions.finish(db, session.id, status)
    assert not samples.register(db, root, existing).created
    with pytest.raises(RegistryError, match=status):
        samples.register(db, root, pair(root), session_id=session.id)


def test_registration_rejects_wrong_signer_and_inactive_sign(db, root, catalog):
    sign, _, session = catalog
    other = signers.add(db, code="SIGNER002")
    signers.consent(db, other.code, "GRANTED")
    with pytest.raises(RegistryError, match="does not match the session signer"):
        samples.register(db, root, pair(root, signer_id=other.code), session_id=session.id)
    vocabulary.update(db, sign.id, is_active=False)
    with pytest.raises(RegistryError, match="sign is inactive"):
        samples.register(db, root, pair(root), session_id=session.id)


def test_withdrawal_reports_and_excludes_samples_without_deleting_files(db, root, catalog):
    _, signer, session = catalog
    sidecar = pair(root)
    sample = samples.register(db, root, sidecar).sample
    pending = pair(root)
    withdrawn, affected = signers.consent(db, signer.code, "WITHDRAWN", "W001")
    assert withdrawn.consent_status == "WITHDRAWN" and withdrawn.consent_reference == "W001"
    assert [row.id for row in affected] == [sample.id]
    assert all(row.status == "EXCLUDED" for row in affected)
    assert sidecar.is_file() and sidecar.with_suffix(".mp4").is_file()
    assert not samples.register(db, root, sidecar).created
    with pytest.raises(RegistryError, match="WITHDRAWN"):
        samples.register(db, root, pending, session_id=session.id)
    signers.consent(db, signer.code, "granted", "FRESH")
    assert samples.show(db, sample.id).status == "EXCLUDED"
    with pytest.raises(RegistryError, match="predates the fresh consent"):
        samples.register(db, root, pending, session_id=session.id)
    future = pair(root, recorded_at=datetime.now().astimezone().isoformat())
    assert samples.register(db, root, future, session_id=session.id).created


def test_statistics_totals_groups_and_consent_aware_coverage(db, root, catalog):
    sign, signer, session = catalog
    empty = vocabulary.add(db, gloss="HELLO", target_samples=1)
    second = signers.add(db, code="SIGNER002")
    signers.consent(db, second.code, "GRANTED")
    samples.register(db, root, pair(root))
    samples.register(db, root, pair(root))
    assert statistics.summary(db) == statistics.Summary(2, 2, 2, 1, 2, 6.04, 16)
    assert statistics.grouped(db, "sign")[-1]["unique_signers"] == 1
    assert statistics.grouped(db, "signer")[0]["samples"] == 2
    assert statistics.grouped(db, "session")[0]["id"] == session.id
    assert dict((row["status"], row["samples"]) for row in statistics.grouped(db, "status"))["CAPTURED"] == 2
    coverage = statistics.coverage(db, min_signer_samples=2)
    assert [row["id"] for row in coverage.zero_recording_signs] == [empty.id]
    assert {row["id"] for row in coverage.underrepresented_signs} == {empty.id, sign.id}
    assert coverage.underrepresented_signers == [{"code": second.code, "collected": 0}]
    signers.consent(db, signer.code, "withdrawn")
    water = next(row for row in statistics.coverage(db).signs if row["id"] == sign.id)
    assert water["registered"] == 2 and water["collected"] == water["unique_signers"] == 0


@pytest.fixture
def mock_camera_media(monkeypatch):
    release = FakeWriter.release

    def finish(writer):
        release(writer)
        if writer.path.is_file():
            writer.path.write_bytes(uuid4().bytes)

    def validate(_, path):
        video = VideoMetadata(
            "mov,mp4", ".mp4", "mpeg4", "yuv420p", 160, 120, None, "LANDSCAPE", 8.0, 8.0,
            4, 0.5, 0.5, None, path.stat().st_size, False, None,
        )
        return video, ValidationReport(ValidationStatus.PASSED, [], [], [], {})

    monkeypatch.setattr(FakeWriter, "release", finish)
    monkeypatch.setattr(VideoValidator, "validate", validate)


def test_camera_acceptance_registers_each_take_and_keeps_one_session_for_multiple_signs(db, root, catalog, monkeypatch, mock_camera_media):
    sign, _, session = catalog
    second = vocabulary.add(db, gloss="GOOD MORNING")
    for selection in (sign, second):
        with monkeypatch.context() as patcher:
            rig = Rig(patcher)
            rig.keys[State.READY].extend([" ", " "])
            rig.keys[State.REVIEW].extend(["r", "a"])
            accepted, = capture_registered(db, CameraCapture(SETTINGS), root, sign_id=selection.id, session_id=session.id)
            rig.assert_clean(root)
        assert accepted.repetition == 1
        assert accepted.gloss == selection.gloss and accepted.sign_id == selection.id
        assert accepted.session_id == session.id
        assert Path(accepted.file_path).parent.parent.name == selection.id
    assert {sample.sign_id for sample in samples.list_samples(db, session_id=session.id)} == {sign.id, second.id}
    assert len(list(root.rglob("*.mp4"))) == 2


def test_registration_failure_preserves_files_for_idempotent_reconciliation(db, root, catalog, monkeypatch, mock_camera_media):
    sign, _, session = catalog
    save = pipeline.save_validation
    calls = []

    def fail_second(connection, sample_id, *args):
        calls.append(sample_id)
        if len(calls) == 2:
            assert list(root.rglob(f"{sample_id}.mp4")) and list(root.rglob(f"{sample_id}.json"))
            raise DatabaseError("Simulated database failure")
        return save(connection, sample_id, *args)

    with monkeypatch.context() as patcher:
        rig = Rig(patcher)
        patcher.setattr(pipeline, "save_validation", fail_second)
        rig.keys[State.READY].extend([" ", " "])
        rig.keys[State.REVIEW].extend(["a", "a"])
        with pytest.raises(RegistryError, match="Run registry reconcile"):
            capture_registered(db, CameraCapture(replace(SETTINGS, repetitions=2)), root,
                               sign_id=sign.id, session_id=session.id)
        rig.assert_clean(root)
    assert len(samples.list_samples(db)) == 1
    assert len(list(root.rglob("*.mp4"))) == 2
    vocabulary.update(db, sign.id, gloss="RENAMED WATER")
    service = pipeline.IngestionService(db, root)
    recovered = media_maintenance.reconcile(service)
    assert len(recovered.actions) == 1 and not recovered.issues
    assert sorted(row.repetition for row in samples.list_samples(db)) == [1, 2]
    assert samples.synchronize(db, root).skipped == 2


def test_consent_is_rechecked_after_camera_acceptance(db, root, catalog, monkeypatch, mock_camera_media):
    sign, signer, session = catalog
    validate = VideoValidator.validate

    def withdraw_before_register(*args, **kwargs):
        signers.consent(db, signer.code, "WITHDRAWN")
        return validate(*args, **kwargs)

    rig = Rig(monkeypatch)
    rig.keys[State.READY].append(" ")
    rig.keys[State.REVIEW].append("a")
    monkeypatch.setattr(VideoValidator, "validate", withdraw_before_register)
    with pytest.raises(RegistryError, match="WITHDRAWN"):
        capture_registered(db, CameraCapture(SETTINGS), root, sign_id=sign.id, session_id=session.id)
    assert not samples.list_samples(db)
    assert not list(root.rglob("*.mp4"))
    rig.assert_clean(root)


def test_withdrawal_between_repetitions_blocks_the_next_take(db, root, catalog, monkeypatch, mock_camera_media):
    sign, signer, session = catalog
    ingest = pipeline.IngestionService.ingest_camera

    def withdraw_after_register(*args, **kwargs):
        result = ingest(*args, **kwargs)
        signers.consent(db, signer.code, "WITHDRAWN")
        return result

    rig = Rig(monkeypatch)
    rig.keys[State.READY].extend([" ", " "])
    rig.keys[State.REVIEW].append("a")
    monkeypatch.setattr(pipeline.IngestionService, "ingest_camera", withdraw_after_register)
    with pytest.raises(RegistryError, match="WITHDRAWN"):
        capture_registered(db, CameraCapture(replace(SETTINGS, repetitions=2)), root,
                           sign_id=sign.id, session_id=session.id)
    assert len(list(root.rglob("*.mp4"))) == 1
    assert [row.status for row in samples.list_samples(db)] == ["EXCLUDED"]
    rig.assert_clean(root)


def test_remote_camera_uses_same_review_and_registration(db, root, catalog, monkeypatch, mock_camera_media):
    sign, signer, _ = catalog
    session = sessions.create(db, signer_code=signer.code, camera_source_type="REMOTE")
    rig = Rig(monkeypatch)
    remote = "http://192.168.1.105:8080/stream"
    opened = []

    def open_video(source, *args):
        if source == remote:
            opened.append(args)
            return rig.camera
        return rig.open_capture(source)

    monkeypatch.setattr("ingestion.capture.cv2.VideoCapture", open_video)
    rig.keys[State.READY].append(" ")
    rig.keys[State.REVIEW].extend(["p", "a"])
    with pytest.raises(RegistryError, match="requires a remote camera URL"):
        capture_registered(db, CameraCapture(SETTINGS), root, sign_id=sign.id, session_id=session.id)
    accepted, = capture_registered(db, CameraCapture(replace(SETTINGS, source=remote)), root,
                                    sign_id=sign.id, session_id=session.id)
    assert opened and len(opened[0]) == 2
    assert accepted.camera_index is None and not rig.camera.settings
    assert samples.show(db, accepted.sample_id).session_id == session.id
    rig.assert_clean(root)


def test_cli_routes_registry_and_standalone_capture(db, root, monkeypatch, mock_camera_media):
    runner = CliRunner()
    base = ["--db", str(db.path), "--dataset-root", str(root)]

    def cli(*args):
        result = runner.invoke(app, base + list(args))
        assert result.exit_code == 0, (result.output, result.exception)
        return result.output

    assert "UGSL0001" in cli("sign", "add", "--gloss", "WATER", "--type", "dynamic", "--target-samples", "5")
    assert "WATER" in cli("sign", "list", "--type", "dynamic", "--active")
    assert "UNKNOWN" in cli("signer", "add", "--code", "SIGNER001")
    assert "GRANTED" in cli("signer", "consent", "SIGNER001", "--status", "granted", "--reference", "REF")
    cli("session", "create", "--signer", "SIGNER001")
    session, = sessions.list_sessions(db)
    with monkeypatch.context() as patcher:
        rig = Rig(patcher)
        rig.keys[State.READY].append(" ")
        rig.keys[State.REVIEW].append("a")
        assert "Accepted and registered 1" in cli("capture", "--sign", "UGSL0001", "--session", session.id,
                                                 "--duration", "0.5", "--countdown", "0")
    assert "CAPTURED" in cli("sample", "list", "--sign", "UGSL0001")
    assert "Underrepresented active signs" in cli("stats", "--coverage")
    assert "SIGNER001" in cli("stats", "--by", "signer")
    assert "skipped 1" in cli("registry", "sync", "--dry-run")
    assert "COMPLETED" in cli("session", "complete", session.id)
    with monkeypatch.context() as patcher:
        rig = Rig(patcher)
        rig.keys[State.READY].append(" ")
        rig.keys[State.REVIEW].append("a")
        assert "Accepted standalone test 1" in cli("capture", "--sign", "TEST", "--signer", "SIGNER001",
                                                  "--duration", "0.5", "--countdown", "0")
    assert len(list((root.parent / "standalone").rglob("*.mp4"))) == 1
    assert len(samples.list_samples(db)) == 1
    assert "1 registered samples excluded" in cli("signer", "withdraw", "SIGNER001")


@pytest.mark.parametrize("session_id, message", [
    ("SESSION_ID", "Replace SESSION_ID with a session UUID"),
    ("00000000-0000-4000-8000-000000000000", "Unknown recording session"),
])
def test_capture_explains_how_to_find_the_session_id(db, root, session_id, message):
    result = CliRunner().invoke(app, [
        "--db", str(db.path), "--dataset-root", str(root),
        "capture", "--sign", "UGSL0001", "--session", session_id,
    ])
    assert result.exit_code == 1
    assert message in result.output
    assert "python cli.py session list" in result.output
