"""Unified validation, publication and registration for camera and imported video."""

import json
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from config import QualityConfig
from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_file
from ingestion.metadata import timestamp_utc
from ingestion.schemas import IngestionOutcome as Outcome
from ingestion.schemas import IngestionResult, RecordingMetadata, ValidationCheck, ValidationReport, ValidationStatus as Status
from ingestion.storage import DatasetStorage, Stage, canonical_uuid
from ingestion.validator import VideoValidator, failed_report
from registry import samples, sessions, signers, vocabulary
from registry.database import Database, DatabaseError, RegistryError
from registry.models import VideoSample
from registry.repository import utc_now


def _new_id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class IngestionRequest:
    sign_id: str
    session_id: str
    source_type: str = "IMPORT"
    source_reference: str | None = None
    recorded_at: str | None = None
    notes: str | None = None
    camera: RecordingMetadata | None = None
    quarantine: bool = False
    batch_id: str | None = None
    sample_id: str = field(default_factory=_new_id)
    attempt_id: str = field(default_factory=_new_id)

    @classmethod
    def from_stage(cls, stage: Stage):
        values = stage.read_request()
        if not isinstance(values, dict):
            raise ValueError("Invalid staging journal.")
        if values.get("camera") is not None:
            values["camera"] = RecordingMetadata(**values["camera"])
        request = cls(**values)
        if canonical_uuid(request.attempt_id) != stage.directory.name:
            raise ValueError("Staging journal ID mismatch.")
        canonical_uuid(request.sample_id)
        return request


def save_validation(connection, sample_id: str, report: ValidationReport, validated_at: str) -> None:
    connection.execute(
        """INSERT INTO video_validations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (_new_id(), sample_id, report.validation_status, report.validation_profile, report.validator_version,
         json.dumps({"checks": [asdict(check) for check in report.checks], **report.metrics}, allow_nan=False),
         json.dumps(report.errors), json.dumps(report.warnings), validated_at),
    )


class IngestionService:
    def __init__(self, db: Database, dataset_root: Path, config: QualityConfig | None = None):
        self.db = db
        self.storage = DatasetStorage(dataset_root)
        self.config = config or QualityConfig()
        self.validator = VideoValidator(self.config)

    def import_file(self, source: Path, *, sign_id: str, session_id: str, recorded_at=None,
                    source_reference=None, notes=None, quarantine=False, batch_id=None, dry_run=False) -> IngestionResult:
        request = IngestionRequest(
            sign_id, session_id, source_reference=source_reference or Path(source).name, recorded_at=recorded_at,
            notes=notes, quarantine=quarantine, batch_id=batch_id,
        )
        return self.ingest(Path(source), request, dry_run=dry_run)

    def ingest_camera(self, source: Path, camera: RecordingMetadata, *, sign_id: str, session_id: str) -> IngestionResult:
        request = IngestionRequest(
            sign_id, session_id, source_type="CAMERA", source_reference="LOCAL" if camera.camera_index is not None else "REMOTE",
            recorded_at=camera.recorded_at, camera=camera, sample_id=camera.sample_id,
        )
        return self.ingest(source, request)

    def _selection(self, connection, request):
        sign = vocabulary.find(connection, request.sign_id)
        if not sign.is_active:
            raise RegistryError("The sign is inactive.")
        session = sessions.find(connection, request.session_id)
        signer = sessions.require_active(connection, session)
        if request.source_type == "CAMERA":
            camera = request.camera
            if (
                camera is None or camera.sample_id != request.sample_id or camera.sign_id != sign.id
                or camera.session_id != session.id or signers.normalize_code(camera.signer_id) != signer.code
                or (camera.camera_index is None) != (session.camera_source_type == "REMOTE")
            ):
                raise RegistryError("Camera metadata does not match the selected sign, signer or session.")
        elif request.source_type != "IMPORT" or request.camera is not None:
            raise RegistryError("Unexpected ingestion source type or camera metadata.")
        return sign, session, signer

    def _start(self, connection, request):
        connection.execute(
            """INSERT INTO ingestion_attempts (id, batch_id, source_reference, result, started_at)
               VALUES (?, ?, ?, 'IN_PROGRESS', ?) ON CONFLICT(id) DO NOTHING""",
            (request.attempt_id, request.batch_id, request.source_reference, utc_now()),
        )

    def _finish(self, connection, result):
        connection.execute(
            """UPDATE ingestion_attempts SET sample_id = ?, sha256 = ?, result = ?, error_code = ?,
               error_message = ?, report_json = ?, completed_at = ?
               WHERE id = ? AND result NOT IN ('INGESTED', 'INGESTED_WITH_WARNINGS')""",
            (result.sample_id, result.sha256, result.outcome, result.error_code, result.message,
             json.dumps(asdict(result.report), allow_nan=False), utc_now(), result.attempt_id),
        )

    def _failure(self, request, exc, *, digest=None, dry_run=False):
        if isinstance(exc, MediaError):
            error = exc
            outcome = Outcome.ERROR if error.operational else Outcome.REJECTED
        elif isinstance(exc, RegistryError) and not isinstance(exc, DatabaseError):
            error = MediaError("INVALID_ASSOCIATION", str(exc))
            outcome = Outcome.CONFLICT
        else:
            error = MediaError("INGESTION_OPERATION_FAILED", "Ingestion could not complete. Check registry access and storage.",
                               operational=True)
            outcome = Outcome.ERROR
        return IngestionResult(request.attempt_id, outcome, failed_report(error, "ingestion"), sha256=digest,
                               error_code=error.code, message=str(error), dry_run=dry_run)

    def _duplicate(self, connection, digest, size, sign, signer, request, dry_run=False, simulated=None):
        existing = simulated.get(digest) if simulated is not None else None
        if existing is None:
            existing = samples.lookup_hash(connection, self.storage.raw, digest, size)
        if existing is None:
            return None
        conflict = existing["sign_id"] != sign.id or existing["signer_id"] != signer.id
        if conflict:
            error = MediaError("DUPLICATE_METADATA_CONFLICT", "Identical content is registered for a different sign or signer.")
            report = failed_report(error, "duplicate_content")
        else:
            error = MediaError("DUPLICATE_CONTENT", "Identical video content is already registered.")
            report = ValidationReport(Status.PENDING, [ValidationCheck("duplicate_content", "SKIPPED", existing["id"])],
                                      [], [], {})
        return IngestionResult(
            request.attempt_id, Outcome.CONFLICT if conflict else Outcome.DUPLICATE, report,
            sample_id=existing["id"], sha256=digest, error_code=error.code, message=str(error), dry_run=dry_run,
        )

    def ingest(self, source: Path, request: IngestionRequest, *, dry_run=False, recovery: Stage | None = None,
               input_error: str | None = None, simulated: dict | None = None) -> IngestionResult:
        db = Database(self.db.path, read_only=True) if dry_run else self.db
        stage = recovery
        prepared = stage is not None
        result = None
        with ExitStack() as resources:
            try:
                if not dry_run:
                    with db.connect(write=True) as connection:
                        self._start(connection, request)
                if input_error:
                    raise MediaError("INVALID_MANIFEST_ENTRY", input_error)
                with db.connect() as connection:
                    self._selection(connection, request)
                if request.recorded_at is not None:
                    timestamp_utc(request.recorded_at)
                candidate = self.validator.integrity.check_file(source)
                if not dry_run and stage is None:
                    stage = resources.enter_context(self.storage.stage(request.attempt_id))
                    stage.prepare(candidate, asdict(request), camera=request.source_type == "CAMERA",
                                  max_bytes=self.config.max_file_size_bytes)
                    prepared = True
                    candidate = stage.video
                result = self._process(db, candidate, request, stage, dry_run, simulated)
            except (RegistryError, MediaError, OSError, ValueError, RuntimeError) as exc:
                result = self._failure(request, exc, dry_run=dry_run)
                # A take reaching this callback was checked before recording. If the DB went down,
                # hold its private bytes for reconciliation, which will check current consent again.
                if request.camera is not None and isinstance(exc, DatabaseError) and stage is None and not dry_run:
                    try:
                        stage = resources.enter_context(self.storage.stage(request.attempt_id))
                        stage.prepare(source, asdict(request), camera=True, max_bytes=self.config.max_file_size_bytes)
                        prepared = True
                    except (MediaError, OSError, ValueError):
                        pass

            if not dry_run and prepared and result.outcome == Outcome.ERROR:
                try:
                    with db.connect() as connection:
                        session = sessions.find(connection, request.session_id)
                        signer = signers.find_by_id(connection, session.signer_id)
                        signers.require_eligible(signer)
                        signers.require_recording_consent(
                            connection, signer, timestamp_utc(request.recorded_at) if request.recorded_at else None,
                        )
                except DatabaseError:
                    pass
                except RegistryError as exc:
                    result = self._failure(request, exc, digest=result.sha256)

            if not dry_run and result.outcome not in {Outcome.INGESTED, Outcome.INGESTED_WITH_WARNINGS}:
                try:
                    with db.connect(write=True) as connection:
                        self._start(connection, request)
                        self._finish(connection, result)
                except RegistryError as exc:
                    result = self._failure(request, exc, digest=result.sha256)

            if not dry_run and stage is not None:
                if result.outcome == Outcome.ERROR and prepared:
                    result = replace(result, message=f"{result.message} Run registry reconcile. Staging ID {request.attempt_id}.")
                else:
                    try:
                        if result.outcome not in {Outcome.INGESTED, Outcome.INGESTED_WITH_WARNINGS}:
                            self.storage.remove_owned_links(stage)
                        stage.remove()
                    except (MediaError, OSError, ValueError):
                        if result.outcome not in {Outcome.INGESTED, Outcome.INGESTED_WITH_WARNINGS}:
                            result = replace(result, message=f"{result.message} Check staging ID {request.attempt_id}.")
        return result

    def _process(self, db, path, request, stage, dry_run, simulated):
        digest = None
        try:
            digest = sha256_file(path)
            if stage is not None:
                stage.verify_checksum(digest, dry_run=dry_run)
            size = path.stat().st_size
            with db.connect() as connection:
                sign, _, signer = self._selection(connection, request)
                duplicate = self._duplicate(connection, digest, size, sign, signer, request, dry_run, simulated)
            if duplicate:
                return duplicate

            video, report = self.validator.validate(path)
            if request.camera and video and report.validation_status in {Status.PASSED, Status.PASSED_WITH_WARNINGS}:
                fields = ("width", "height", "frame_count", "file_size_bytes")
                if any(getattr(video, name) != getattr(request.camera, name) for name in fields):
                    raise MediaError("CAMERA_METADATA_MISMATCH", "The saved video does not match camera capture metadata.")
            recorded_at = timestamp_utc(request.recorded_at) if request.recorded_at is not None else None
            if report.validation_status in {Status.FAILED, Status.ERROR}:
                failed = report.validation_status == Status.FAILED
                if failed and request.quarantine and not dry_run:
                    with db.connect(write=True) as connection:
                        _, _, signer = self._selection(connection, request)
                        signers.require_recording_consent(connection, signer, recorded_at)
                        self.storage.retain(
                            stage, request.attempt_id, {**asdict(report), "signer_id": signer.id, "recorded_at": recorded_at},
                            self.config.quarantine_retention_days,
                        )
                return IngestionResult(
                    request.attempt_id, Outcome.REJECTED if failed else Outcome.ERROR, report, sha256=digest,
                    error_code="TECHNICAL_VALIDATION_FAILED" if failed else "VALIDATION_ERROR",
                    message=" ".join(report.errors), dry_run=dry_run,
                )

            outcome = Outcome.INGESTED_WITH_WARNINGS if report.warnings else Outcome.INGESTED
            result = IngestionResult(request.attempt_id, outcome, report, request.sample_id, digest,
                                     storage_bytes=size, dry_run=dry_run)
            with db.connect(write=not dry_run) as connection:
                sign, session, signer = self._selection(connection, request)
                signers.require_recording_consent(connection, signer, recorded_at)
                duplicate = self._duplicate(connection, digest, size, sign, signer, request, dry_run)
                if duplicate:
                    return duplicate
                if not dry_run:
                    self._publish(connection, stage, request, sign, session, signer, video, recorded_at, result)
                elif simulated is not None:
                    simulated[digest] = {"id": request.sample_id, "sign_id": sign.id, "signer_id": signer.id}
            return result
        except (RegistryError, MediaError, OSError, ValueError, RuntimeError) as exc:
            return self._failure(request, exc, digest=digest, dry_run=dry_run)

    def _publish(self, connection, stage, request, sign, session, signer, video, recorded_at, result):
        relative = Path(sign.id) / signer.code / f"{request.sample_id}{video.extension}"
        now = utc_now()
        if stage.sidecar.is_file() and not stage.sidecar.is_symlink():
            # Reuse the time recorded with an already-published sidecar.
            now = json.loads(stage.sidecar.read_text(encoding="utf-8"))["registered_at"]
        camera = request.camera
        sample = VideoSample(
            id=request.sample_id, sign_id=sign.id, session_id=session.id, sample_type="ISOLATED",
            repetition=camera.repetition if camera else None, file_path=relative.as_posix(),
            metadata_path=relative.with_suffix(".json").as_posix(), width=video.width, height=video.height,
            requested_fps=camera.requested_fps if camera else None, measured_fps=camera.measured_fps if camera else None,
            encoded_fps=camera.encoded_fps if camera else None, frame_count=video.frame_count,
            failed_frame_reads=camera.failed_frame_reads if camera else None,
            requested_duration_seconds=camera.requested_duration_seconds if camera else None,
            actual_duration_seconds=camera.actual_duration_seconds if camera else None,
            file_size_bytes=video.file_size_bytes, status="CAPTURED", recorded_at=recorded_at, registered_at=now,
            sha256=result.sha256, source_type=request.source_type, source_reference=request.source_reference,
            notes=request.notes, container_format=video.container_format, video_codec=video.video_codec,
            pixel_format=video.pixel_format, average_fps=video.average_fps, nominal_fps=video.nominal_fps,
            duration_seconds=video.duration_seconds, rotation=video.rotation, bitrate=video.bitrate,
            technical_validation_status=result.report.validation_status, technical_validated_at=now,
            validator_version=result.report.validator_version,
        )
        if connection.execute("SELECT 1 FROM video_samples WHERE id = ?", (sample.id,)).fetchone():
            raise MediaError("SAMPLE_ID_CONFLICT", "The sample UUID is already registered to different content.")
        sidecar = {
            **(asdict(camera) if camera else {}), **asdict(sample), "sample_id": sample.id,
            "gloss": camera.gloss if camera else sign.gloss, "signer_id": signer.code,
            "video_metadata": asdict(video), "validation": asdict(result.report),
        }
        self.storage.publish(stage, relative, sidecar, result.sha256)
        values = asdict(sample)
        connection.execute(
            f"INSERT INTO video_samples ({', '.join(values)}) VALUES ({', '.join(':' + key for key in values)})",
            values,
        )
        save_validation(connection, sample.id, result.report, now)
        self._finish(connection, result)
