"""Connect camera acceptance to durable sample registration."""

from pathlib import Path

from config import QualityConfig
from ingestion.capture import CameraCapture
from ingestion.pipeline import IngestionService
from ingestion.schemas import IngestionOutcome, RecordingContext, RecordingMetadata
from registry import sessions, vocabulary
from registry.database import Database, RegistryError


def capture_registered(
    db: Database, camera: CameraCapture, dataset_root: Path, *, sign_id: str, session_id: str,
    quality: QualityConfig | None = None,
) -> list[RecordingMetadata]:
    def validate_selection():
        with db.connect() as connection:
            session = sessions.find(connection, session_id)
            sign = vocabulary.find(connection, sign_id)
            if not sign.is_active:
                raise RegistryError("The sign is inactive.")
            return sign, session, sessions.require_active(connection, session)

    sign, session, signer = validate_selection()
    source_type = "REMOTE" if camera.config.source is not None else "LOCAL"
    if session.camera_source_type != source_type:
        if session.camera_source_type == "REMOTE":
            raise RegistryError("This session requires a remote camera URL (--source).")
        raise RegistryError("This session requires a local camera (--camera).")
    root = dataset_root.resolve()
    context = RecordingContext(sign_id=sign.id, session_id=session.id, gloss=sign.gloss)
    ingestion = IngestionService(db, root, quality)

    def before_record() -> None:
        validate_selection()

    def register_accepted(path: Path, recording: RecordingMetadata) -> str | None:
        result = ingestion.ingest_camera(path, recording, sign_id=sign.id, session_id=session.id)
        if result.outcome == IngestionOutcome.ERROR or result.error_code == "INVALID_ASSOCIATION":
            raise RegistryError(f"Sample {recording.sample_id}: {result.message}")
        if result.outcome not in {IngestionOutcome.INGESTED, IngestionOutcome.INGESTED_WITH_WARNINGS}:
            if result.sample_id:
                return f"{result.message} Existing sample {result.sample_id}."
            return result.message
        return None

    return camera.capture(
        sign.id, signer.code, root, context=context, before_record=before_record, ingest_accepted=register_accepted,
    )
