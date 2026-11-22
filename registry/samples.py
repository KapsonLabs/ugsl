"""Validate, register and reconcile accepted video/JSON pairs."""

import json
import math
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from config import DATASET_ROOT
from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_file
from ingestion.schemas import RecordingMetadata, ValidationStatus
from ingestion.storage import safe_path
from registry import sessions, signers, vocabulary
from registry.database import Database, RegistryError
from registry.models import RecordingSession, VideoSample
from registry.repository import choice, get, integer, normalize, record, utc_now

STATUSES = ("CAPTURED", "APPROVED", "REJECTED", "EXCLUDED")
INTEGER_FIELDS = ("repetition", "width", "height", "frame_count", "failed_frame_reads", "file_size_bytes")
REAL_FIELDS = (
    "requested_fps", "measured_fps", "encoded_fps", "requested_duration_seconds", "actual_duration_seconds",
)


@dataclass(frozen=True)
class Registration:
    sample: VideoSample
    created: bool


@dataclass
class SyncReport:
    registered: list[str]
    skipped: int
    issues: list[tuple[str, str]]


def _uuid(value, name):
    try:
        result = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RegistryError(f"{name} must be a UUID.") from exc
    if result != value:
        raise RegistryError(f"{name} must use canonical UUID format.")
    return result


def _pair(dataset_root: Path, sidecar: Path):
    root = dataset_root.resolve()
    path = sidecar.absolute()
    if not path.is_relative_to(root) or path.suffix != ".json" or ".." in path.parts:
        raise RegistryError("Metadata must be a JSON file inside the raw dataset directory.")
    video = path.with_suffix(".mp4")
    if path.resolve() != path or video.resolve() != video:
        raise RegistryError("Symlinked dataset paths are unsupported.")
    if not path.is_file():
        raise RegistryError("The JSON sidecar is missing.")
    if not video.is_file():
        raise RegistryError("The corresponding MP4 video is missing.")
    return root, path, video


def _load(dataset_root: Path, sidecar: Path):
    try:
        root, path, video = _pair(dataset_root, sidecar)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        if "validation" in payload or "sha256" in payload:
            raise RegistryError("This is an ingestion sidecar. Use registry reconcile for incomplete registrations.")
        known_fields = {field.name for field in fields(RecordingMetadata)}
        meta = RecordingMetadata(**{key: value for key, value in payload.items() if key in known_fields})
        if _uuid(meta.sample_id, "Sample ID") != path.stem:
            raise RegistryError("The sample UUID does not match the file names.")
        if not isinstance(meta.gloss, str) or not normalize(meta.gloss):
            raise RegistryError("Metadata must contain a sign gloss.")
        if not isinstance(meta.signer_id, str):
            raise RegistryError("Metadata must contain a signer code.")
        signers.normalize_code(meta.signer_id)
        if meta.sign_id is not None and (not isinstance(meta.sign_id, str) or not meta.sign_id.strip()):
            raise RegistryError("Metadata sign ID must be a nonempty string.")
        if meta.session_id is not None:
            _uuid(meta.session_id, "Metadata session ID")
        if meta.camera_index is not None:
            integer(meta.camera_index, "Camera index", 0)
        for name in INTEGER_FIELDS:
            integer(getattr(meta, name), name, 0 if name == "failed_frame_reads" else 1)
        for name in REAL_FIELDS:
            value = getattr(meta, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise RegistryError(f"{name} must be finite and positive.")
        timestamp = datetime.fromisoformat(meta.recorded_at)
        if timestamp.tzinfo is None:
            raise RegistryError("The recording timestamp must include a UTC offset.")
        relative = video.relative_to(root)
        if not isinstance(meta.file_path, str) or "\\" in meta.file_path or "\x00" in meta.file_path:
            raise RegistryError("Metadata file path is unsafe.")
        declared = Path(meta.file_path)
        if ".." in declared.parts or not (
            declared == relative or declared == DATASET_ROOT / relative or declared.absolute() == video
        ):
            raise RegistryError("Metadata file path does not match the video inside the dataset.")
        if meta.file_size_bytes != video.stat().st_size:
            raise RegistryError("The video file size does not match the sidecar.")
        return meta, relative.as_posix(), path.relative_to(root).as_posix(), timestamp.astimezone(UTC).isoformat()
    except (OSError, UnicodeError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise RegistryError("Could not read a valid recording sidecar and its MP4 file.") from exc


def _resolve(connection, meta, sign_id, session_id):
    requested_sign = sign_id.strip().upper() if sign_id else None
    embedded_sign = meta.sign_id.strip().upper() if meta.sign_id else None
    if embedded_sign and requested_sign and embedded_sign != requested_sign:
        raise RegistryError("Metadata sign ID differs from the selected sign.")
    if embedded_sign or requested_sign:
        sign = vocabulary.find(connection, embedded_sign or requested_sign)
        if not embedded_sign and sign.gloss != normalize(meta.gloss):
            raise RegistryError("The selected sign does not match the legacy recording gloss.")
    else:
        matches = vocabulary.by_gloss(connection, meta.gloss)
        if len(matches) != 1:
            raise RegistryError("Unknown recording gloss." if not matches else
                                "Recording gloss matches multiple variants. Select a sign ID.")
        sign = matches[0]
    if not sign.is_active:
        raise RegistryError("The sign is inactive.")

    signer = signers.find(connection, meta.signer_id)
    signers.require_eligible(signer)
    requested_session = session_id.strip().lower() if session_id else None
    if meta.session_id and requested_session and meta.session_id != requested_session:
        raise RegistryError("Metadata session ID differs from the selected session.")
    if meta.session_id or requested_session:
        session = sessions.find(connection, meta.session_id or requested_session)
    else:
        matches = connection.execute(
            "SELECT * FROM recording_sessions WHERE signer_id = ? AND status = 'ACTIVE' LIMIT 2", (signer.id,),
        ).fetchall()
        if len(matches) != 1:
            raise RegistryError("No unique active session for this signer. Select a session explicitly.")
        session = record(RecordingSession, matches[0])
    if session.signer_id != signer.id:
        raise RegistryError("Recording signer does not match the session signer.")
    sessions.require_open(session)
    if meta.session_id and (meta.camera_index is None) != (session.camera_source_type == "REMOTE"):
        raise RegistryError("Recording camera source does not match the session camera type.")
    return sign, session, signer


def lookup_hash(connection, root: Path, digest: str, size: int):
    query = """SELECT v.id, v.sign_id, v.file_path, r.signer_id FROM video_samples v
               JOIN recording_sessions r ON r.id = v.session_id WHERE """
    existing = connection.execute(query + "v.sha256 = ?", (digest,)).fetchone()
    if existing is not None:
        return existing
    # Cover older records before backfill by checking plausible matches.
    for row in connection.execute(query + "v.sha256 IS NULL AND v.file_size_bytes = ? ORDER BY v.id", (size,)):
        try:
            path = safe_path(root, row["file_path"])
            if path.is_file() and sha256_file(path) == digest:
                return row
        except (OSError, MediaError):
            continue
    return None


def register(
    db: Database, dataset_root: Path, sidecar: Path, *, sign_id: str | None = None,
    session_id: str | None = None, dry_run: bool = False,
) -> Registration:
    """Register a published pair once, or return the matching existing record."""
    if dry_run and not db.read_only:
        db = Database(db.path, read_only=True)
    meta, file_path, metadata_path, recorded_at = _load(Path(dataset_root), Path(sidecar))
    root = Path(dataset_root).resolve()
    try:
        digest = sha256_file(root / file_path)
    except OSError as exc:
        raise RegistryError("Could not hash the recording.") from exc
    metrics = {name: getattr(meta, name) for name in (*INTEGER_FIELDS, *REAL_FIELDS)}
    with db.connect(write=not dry_run) as connection:
        existing = [record(VideoSample, row) for row in connection.execute(
            "SELECT * FROM video_samples WHERE id = ? OR file_path = ?", (meta.sample_id, file_path),
        )]
        if existing:
            sample = existing[0]
            if (
                len(existing) != 1 or sample.id != meta.sample_id or sample.file_path != file_path
                or sample.metadata_path != metadata_path or sample.recorded_at != recorded_at
                or any(getattr(sample, name) != value for name, value in metrics.items())
                or (meta.sign_id and meta.sign_id.strip().upper() != sample.sign_id)
                or (meta.session_id and meta.session_id != sample.session_id)
                or (sample.sha256 is not None and sample.sha256 != digest)
            ):
                raise RegistryError("The sample UUID or video path conflicts with a registered recording.")
            session = sessions.find(connection, sample.session_id)
            if signers.find_by_id(connection, session.signer_id).code != signers.normalize_code(meta.signer_id):
                raise RegistryError("The sidecar signer differs from the registered sample signer.")
            return Registration(sample, created=False)

        sign, session, signer = _resolve(connection, meta, sign_id, session_id)
        signers.require_recording_consent(connection, signer, recorded_at)
        duplicate = lookup_hash(connection, root, digest, meta.file_size_bytes)
        if duplicate is not None:
            if duplicate["sign_id"] != sign.id or duplicate["signer_id"] != signer.id:
                raise RegistryError(f"Identical content is registered for a different sign or signer ({duplicate['id']}).")
            previous = connection.execute("SELECT * FROM video_samples WHERE id = ?", (duplicate["id"],)).fetchone()
            return Registration(record(VideoSample, previous), created=False)
        sample = VideoSample(
            id=meta.sample_id, sign_id=sign.id, session_id=session.id, sample_type="ISOLATED",
            file_path=file_path, metadata_path=metadata_path, status="CAPTURED",
            recorded_at=recorded_at, registered_at=utc_now(), sha256=digest, **metrics,
        )
        if not dry_run:
            connection.execute(
                """INSERT INTO video_samples (id, sign_id, session_id, sample_type, repetition, file_path,
                   metadata_path, width, height, requested_fps, measured_fps, encoded_fps, frame_count,
                   failed_frame_reads, requested_duration_seconds, actual_duration_seconds, file_size_bytes,
                   status, recorded_at, registered_at, sha256)
                   VALUES (:id, :sign_id, :session_id, :sample_type, :repetition, :file_path, :metadata_path,
                     :width, :height, :requested_fps, :measured_fps, :encoded_fps, :frame_count,
                     :failed_frame_reads, :requested_duration_seconds, :actual_duration_seconds,
                     :file_size_bytes, :status, :recorded_at, :registered_at, :sha256)""",
                vars(sample),
            )
        return Registration(sample, created=True)


def synchronize(db: Database, dataset_root: Path, *, sign_id=None, session_id=None, dry_run=False) -> SyncReport:
    if dry_run:
        db = Database(db.path, read_only=True)
    with db.connect() as connection:
        if sign_id is not None:
            vocabulary.find(connection, sign_id)
        if session_id is not None:
            sessions.find(connection, session_id)
        ingested_paths = {row[0] for row in connection.execute(
            "SELECT metadata_path FROM video_samples WHERE technical_validation_status <> 'PENDING' OR source_type = 'IMPORT'"
        )}
    root = Path(dataset_root).resolve()
    report = SyncReport(registered=[], skipped=0, issues=[])
    proposed_ids, proposed_hashes = set(), {}
    for sidecar in sorted(root.rglob("*.json")):
        relative = sidecar.relative_to(root).as_posix()
        if relative in ingested_paths:
            report.skipped += 1
            continue
        try:
            result = register(db, root, sidecar, sign_id=sign_id, session_id=session_id, dry_run=dry_run)
            if result.created:
                if result.sample.id in proposed_ids:
                    raise RegistryError("The sample UUID conflicts with an earlier recording in this scan.")
                previous = proposed_hashes.get(result.sample.sha256)
                if previous is not None:
                    if previous.sign_id != result.sample.sign_id:
                        raise RegistryError("Identical content conflicts with an earlier recording in this scan.")
                    with db.connect() as connection:
                        first = sessions.find(connection, previous.session_id)
                        second = sessions.find(connection, result.sample.session_id)
                    if first.signer_id != second.signer_id:
                        raise RegistryError("Identical content conflicts with an earlier recording in this scan.")
                    report.skipped += 1
                    continue
                proposed_ids.add(result.sample.id)
                proposed_hashes[result.sample.sha256] = result.sample
                report.registered.append(relative)
            else:
                report.skipped += 1
        except RegistryError as exc:
            report.issues.append((relative, str(exc)))
    return report


def show(db: Database, sample_id: str) -> VideoSample:
    with db.connect() as connection:
        return get(connection, VideoSample, "SELECT * FROM video_samples WHERE id = ?",
                   (_uuid(sample_id.strip().lower(), "Sample ID"),), "Unknown video sample.")


def list_samples(db: Database, *, sign_id=None, session_id=None, status=None, validation_status=None) -> list[VideoSample]:
    if status is not None:
        status = choice(status, "Sample status", STATUSES)
    if validation_status is not None:
        validation_status = choice(validation_status, "Technical validation status", tuple(ValidationStatus))
    with db.connect() as connection:
        sign_id = vocabulary.find(connection, sign_id).id if sign_id is not None else None
        session_id = sessions.find(connection, session_id).id if session_id is not None else None
        return [record(VideoSample, row) for row in connection.execute(
            """SELECT * FROM video_samples WHERE (? IS NULL OR sign_id = ?) AND
               (? IS NULL OR session_id = ?) AND (? IS NULL OR status = ?)
               AND (? IS NULL OR technical_validation_status = ?) ORDER BY recorded_at, id""",
            (sign_id, sign_id, session_id, session_id, status, status, validation_status, validation_status),
        )]
