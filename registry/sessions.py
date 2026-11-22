"""Recording sessions and the checks shared by capture and registration."""

from uuid import uuid4

from registry import signers
from registry.database import Database, RegistryError
from registry.models import RecordingSession, Signer
from registry.repository import choice, get, optional_text, record, utc_now

SOURCE_TYPES = ("LOCAL", "REMOTE")
ANGLES = ("FRONT", "LEFT", "RIGHT", "OTHER")
LIGHTING = ("NATURAL", "ARTIFICIAL", "LOW_LIGHT", "OTHER")
BACKGROUNDS = ("PLAIN", "COMPLEX", "OTHER")


def find(connection, session_id: str) -> RecordingSession:
    session_id = session_id.strip().lower()
    if session_id == "session_id":
        raise RegistryError("Replace SESSION_ID with a session UUID from `python cli.py session list`.")
    return get(connection, RecordingSession, "SELECT * FROM recording_sessions WHERE id = ?",
               (session_id,), "Unknown recording session. Run `python cli.py session list` using the same database.")


def require_open(session: RecordingSession) -> None:
    if session.status != "ACTIVE":
        raise RegistryError(f"The recording session is {session.status}. Select an active session.")


def require_active(connection, session: RecordingSession) -> Signer:
    require_open(session)
    signer = signers.find_by_id(connection, session.signer_id)
    signers.require_eligible(signer)
    return signer


def create(
    db: Database, *, signer_code: str, operator=None, camera_source_type="LOCAL", camera_angle="FRONT",
    lighting_condition="OTHER", background="OTHER", notes=None,
) -> RecordingSession:
    fields = (
        optional_text(operator), choice(camera_source_type, "Camera source type", SOURCE_TYPES),
        choice(camera_angle, "Camera angle", ANGLES), choice(lighting_condition, "Lighting condition", LIGHTING),
        choice(background, "Background", BACKGROUNDS), optional_text(notes),
    )
    with db.connect(write=True) as connection:
        signer = signers.find(connection, signer_code)
        signers.require_eligible(signer)
        session_id = str(uuid4())
        connection.execute(
            """INSERT INTO recording_sessions (id, signer_id, operator, camera_source_type, camera_angle,
               lighting_condition, background, notes, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, signer.id, *fields, utc_now()),
        )
        return find(connection, session_id)


def show(db: Database, session_id: str) -> RecordingSession:
    with db.connect() as connection:
        return find(connection, session_id)


def list_sessions(db: Database) -> list[RecordingSession]:
    with db.connect() as connection:
        return [record(RecordingSession, row) for row in connection.execute(
            "SELECT * FROM recording_sessions ORDER BY started_at DESC, id",
        )]


def finish(db: Database, session_id: str, status: str) -> RecordingSession:
    status = choice(status, "Final session status", ("COMPLETED", "CANCELLED"))
    with db.connect(write=True) as connection:
        session = find(connection, session_id)
        if session.status == status:
            return session
        if session.status != "ACTIVE":
            raise RegistryError(f"The recording session is already {session.status}.")
        connection.execute(
            "UPDATE recording_sessions SET status = ?, ended_at = ? WHERE id = ?",
            (status, utc_now(), session.id),
        )
        return find(connection, session.id)
