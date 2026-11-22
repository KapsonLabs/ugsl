"""Connect camera acceptance to durable sample registration."""

from pathlib import Path

from ingestion.capture import CameraCapture
from ingestion.schemas import RecordingContext, RecordingMetadata
from registry import samples, sessions, vocabulary
from registry.database import Database, RegistryError


def capture_registered(
    db: Database, camera: CameraCapture, dataset_root: Path, *, sign_id: str, session_id: str,
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

    def before_record() -> None:
        validate_selection()

    def register_accepted(recording: RecordingMetadata) -> None:
        try:
            pair = (root / recording.file_path).with_suffix(".json")
            samples.register(db, root, pair, sign_id=sign.id, session_id=session.id)
        except RegistryError as exc:
            raise RegistryError(
                f"Sample {recording.sample_id} was saved but registration failed: {exc} "
                "Resolve the reported issue and run registry sync."
            ) from exc

    return camera.capture(
        sign.id, signer.code, root, context=context, before_record=before_record, on_accept=register_accepted,
    )
