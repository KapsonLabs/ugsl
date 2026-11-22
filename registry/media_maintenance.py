"""Backfill, revalidation and recovery for registered and staged videos."""

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_file
from ingestion.importer import media_files
from ingestion.pipeline import IngestionRequest, IngestionService, save_validation
from ingestion.schemas import IngestionOutcome, ValidationReport
from ingestion.storage import Stage, canonical_uuid, safe_path
from ingestion.validator import failed_report
from registry import samples, sessions, signers
from registry.database import Database, RegistryError
from registry.repository import utc_now


@dataclass
class MaintenanceReport:
    actions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _store(db, dry_run):
    return Database(db.path, read_only=True) if dry_run else db


def backfill_hashes(service: IngestionService, *, dry_run=False) -> MaintenanceReport:
    db = _store(service.db, dry_run)
    raw = service.storage.raw
    report = MaintenanceReport()
    groups = defaultdict(list)
    rows = samples.list_samples(db)
    known = {sample.file_path for sample in rows}
    for sample in rows:
        try:
            path = safe_path(raw, sample.file_path)
            if path.stat().st_size != sample.file_size_bytes:
                report.issues.append(f"{sample.id}: file size differs from the registry.")
                continue
            digest = sha256_file(path)
            if sample.sha256 and sample.sha256 != digest:
                report.issues.append(f"{sample.id}: checksum differs from the registry.")
            groups[digest].append((sample.id, sample.sha256, sample.file_path))
        except (MediaError, OSError, RuntimeError):
            report.issues.append(f"{sample.id}: video is missing, unsafe or unreadable.")
    for path in media_files(raw):
        relative = path.relative_to(raw).as_posix()
        if relative not in known:
            try:
                sidecar = safe_path(raw, path.with_suffix(".json").relative_to(raw))
                if sidecar.is_file():
                    digest = sha256_file(path)
                    groups[digest].append((None, None, relative))
                    report.issues.append(f"{relative}: unregistered pair (SHA-256 {digest}). Check registry sync or reconcile.")
            except (MediaError, OSError, RuntimeError):
                report.issues.append(f"{relative}: unregistered pair is unsafe or unreadable.")

    for digest, entries in sorted(groups.items()):
        registered = [entry for entry in entries if entry[0] is not None]
        if len(entries) > 1:
            report.issues.append(f"Duplicate {digest}: " + ", ".join(f"{sid or 'unregistered'} ({path})" for sid, _, path in entries))
        if len(registered) != 1 or registered[0][1] is not None:
            continue
        sid = registered[0][0]
        if not dry_run:
            with db.connect(write=True) as connection:
                existing = connection.execute("SELECT id FROM video_samples WHERE sha256 = ?", (digest,)).fetchone()
                if existing and existing["id"] != sid:
                    report.issues.append(f"{sid}: checksum already registered by {existing['id']}.")
                    continue
                cursor = connection.execute("UPDATE video_samples SET sha256 = ? WHERE id = ? AND sha256 IS NULL", (digest, sid))
                if not cursor.rowcount:
                    continue
        report.actions.append(f"{'Would backfill' if dry_run else 'Backfilled'} {sid} {digest}")
    return report


def validate_sample(service: IngestionService, sample_id: str, *, dry_run=False) -> ValidationReport:
    db = _store(service.db, dry_run)
    sample = samples.show(db, sample_id)
    try:
        with db.connect() as connection:
            session = sessions.find(connection, sample.session_id)
            signer = signers.find_by_id(connection, session.signer_id)
            if signer.consent_status != "GRANTED" or sample.status == "EXCLUDED":
                raise MediaError("CONSENT_RESTRICTED", "This sample is excluded or lacks current granted consent.", operational=True)
        path = safe_path(service.storage.raw, sample.file_path)
        if sample.sha256 is not None and sha256_file(path) != sample.sha256:
            raise MediaError("CHECKSUM_MISMATCH", "The file no longer matches its registered SHA-256.")
        _, report = service.validator.validate(path)
    except MediaError as exc:
        report = failed_report(exc, "registered_video")
    except (OSError, RuntimeError):
        report = failed_report(MediaError("UNREADABLE_FILE", "The registered video is missing or unreadable."), "registered_video")
    if not dry_run:
        now = utc_now()
        with db.connect(write=True) as connection:
            save_validation(connection, sample.id, report, now)
            connection.execute(
                """UPDATE video_samples SET technical_validation_status = ?, technical_validated_at = ?,
                   validator_version = ? WHERE id = ?""",
                (report.validation_status, now, report.validator_version, sample.id),
            )
    return report


def _reconcile_stage(service, db, stage, report, dry_run):
    request = IngestionRequest.from_stage(stage)
    with db.connect() as connection:
        current = connection.execute("SELECT * FROM video_samples WHERE id = ?", (request.sample_id,)).fetchone()
    if current is not None:
        raw = safe_path(service.storage.raw, current["file_path"])
        metadata = safe_path(service.storage.raw, current["metadata_path"])
        if (
            current["sign_id"] != request.sign_id.strip().upper()
            or current["session_id"] != request.session_id.strip().lower()
            or current["sha256"] is None or not raw.samefile(stage.video) or not metadata.samefile(stage.sidecar)
            or sha256_file(raw) != current["sha256"]
        ):
            report.issues.append(f"{request.attempt_id}: staged identity does not match the committed sample.")
            return
        if not dry_run:
            stage.remove()
        report.actions.append(f"{'Would clean' if dry_run else 'Cleaned'} committed staging ID {request.attempt_id}.")
        return

    result = service.ingest(stage.video, request, dry_run=dry_run, recovery=stage)
    if result.outcome in {IngestionOutcome.INGESTED, IngestionOutcome.INGESTED_WITH_WARNINGS}:
        report.actions.append(f"{'Would register' if dry_run else 'Recovered'} {result.sample_id}.")
    else:
        report.issues.append(f"{request.attempt_id}: {result.outcome} {result.message}")


def _quarantine(service, db, report, dry_run, purge):
    root = service.storage.quarantine
    if not root.is_dir():
        return
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            canonical_uuid(directory.name)
            marker = safe_path(root, Path(directory.name) / "report.json")
            info = json.loads(marker.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(info["expires_at"])
            with db.connect(write=purge and not dry_run) as connection:
                signer = signers.find_by_id(connection, info["signer_id"])
                expired = expires <= datetime.now(UTC) or signer.consent_status != "GRANTED"
                if not expired:
                    try:
                        signers.require_recording_consent(connection, signer, info["recorded_at"])
                    except RegistryError:
                        expired = True
                if expired and purge and not dry_run:
                    shutil.rmtree(directory)
            if expired:
                if purge:
                    report.actions.append(f"{'Would purge' if dry_run else 'Purged'} quarantine {directory.name}.")
                else:
                    report.issues.append(f"Quarantine {directory.name} has expired or lacks current consent.")
        except (MediaError, RegistryError, OSError, TypeError, ValueError, KeyError):
            report.issues.append(f"{directory.name}: could not verify quarantine retention.")


def reconcile(service: IngestionService, *, dry_run=False, purge_quarantine=False) -> MaintenanceReport:
    report = MaintenanceReport()
    db = _store(service.db, dry_run)
    staging = service.storage.staging
    staged_ids = set()
    pending_paths = set()
    if staging.is_dir():
        for directory in sorted(staging.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                attempt_id = canonical_uuid(directory.name)
                staged_ids.add(attempt_id)
                stage = Stage(directory)
                with stage.lock():
                    if dry_run and stage.sidecar.is_file() and not stage.sidecar.is_symlink():
                        pending_paths.add(json.loads(stage.sidecar.read_text(encoding="utf-8"))["file_path"])
                    _reconcile_stage(service, db, stage, report, dry_run)
            except BlockingIOError:
                report.issues.append(f"{directory.name}: staging is in use.")
            except (MediaError, RegistryError, OSError, TypeError, ValueError, RuntimeError, KeyError):
                report.issues.append(f"{directory.name}: staging is incomplete or unsafe. Inspect it manually.")

    raw = service.storage.raw
    registered = samples.list_samples(db)
    known_videos = {sample.file_path for sample in registered} | pending_paths
    known_sidecars = {sample.metadata_path for sample in registered} | {
        Path(path).with_suffix(".json").as_posix() for path in pending_paths
    }
    for sample in registered:
        try:
            if not all(safe_path(raw, path).is_file() for path in (sample.file_path, sample.metadata_path)):
                report.issues.append(f"{sample.id}: registered video or sidecar is missing.")
        except (MediaError, RuntimeError):
            report.issues.append(f"{sample.id}: unsafe registered video or sidecar path.")
    for path in media_files(raw):
        if path.relative_to(raw).as_posix() not in known_videos:
            report.issues.append(f"{path.relative_to(raw)}: unregistered video. Legacy pairs can use registry sync.")
        if path.with_suffix(".json").is_file():
            known_sidecars.add(path.with_suffix(".json").relative_to(raw).as_posix())
    for path in sorted(raw.rglob("*.json")):
        if path.relative_to(raw).as_posix() not in known_sidecars:
            report.issues.append(f"{path.relative_to(raw)}: sidecar without a corresponding known video.")
    with db.connect() as connection:
        for row in connection.execute("SELECT id FROM ingestion_attempts WHERE result = 'IN_PROGRESS'"):
            if row["id"] not in staged_ids:
                report.issues.append(f"{row['id']}: interrupted attempt has no staging journal. Retry from the original source.")
    _quarantine(service, db, report, dry_run, purge_quarantine)
    return report
