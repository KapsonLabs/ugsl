"""Signer registration and explicit consent changes."""

from datetime import datetime
from uuid import uuid4

from config import validate_identifier
from registry.database import Database, RegistryError
from registry.models import Signer, VideoSample
from registry.repository import choice, get, optional_text, record, utc_now

HANDS = ("LEFT", "RIGHT", "BOTH", "UNKNOWN")
PROFICIENCIES = ("BEGINNER", "INTERMEDIATE", "FLUENT", "NATIVE", "UNKNOWN")
CONSENT_STATUSES = ("UNKNOWN", "GRANTED", "DENIED", "WITHDRAWN")


def normalize_code(code: str) -> str:
    code = code.strip()
    try:
        validate_identifier(code, "Signer code")
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    return code.upper()


def find(connection, code: str) -> Signer:
    return get(connection, Signer, "SELECT * FROM signers WHERE code = ?", (normalize_code(code),), "Unknown signer code.")


def find_by_id(connection, signer_id: str) -> Signer:
    return get(connection, Signer, "SELECT * FROM signers WHERE id = ?", (signer_id,), "Unknown signer ID.")


def require_eligible(signer: Signer) -> None:
    if not signer.is_active:
        raise RegistryError("The signer is inactive.")
    if signer.consent_status != "GRANTED":
        raise RegistryError(f"Training recordings require granted consent. Current status is {signer.consent_status}.")


def require_recording_consent(connection, signer: Signer, recorded_at: str) -> None:
    """Keep pre-withdrawal and withdrawal-period footage out after a fresh grant."""
    grant = connection.execute(
        """SELECT recorded_at FROM signer_consent_events
           WHERE signer_id = ? AND status = 'GRANTED' AND id > (
               SELECT max(id) FROM signer_consent_events WHERE signer_id = ? AND status = 'WITHDRAWN'
           ) ORDER BY id LIMIT 1""",
        (signer.id, signer.id),
    ).fetchone()
    if grant and datetime.fromisoformat(recorded_at) <= datetime.fromisoformat(grant["recorded_at"]):
        raise RegistryError("This recording predates the fresh consent granted after withdrawal.")


def _fields(dominant_hand, age_group, proficiency, region, is_active):
    return (
        choice(dominant_hand, "Dominant hand", HANDS), optional_text(age_group),
        choice(proficiency, "Proficiency", PROFICIENCIES), optional_text(region), int(is_active),
    )


def add(db: Database, *, code: str, dominant_hand="UNKNOWN", age_group=None, proficiency="UNKNOWN", region=None) -> Signer:
    code = normalize_code(code)
    fields = _fields(dominant_hand, age_group, proficiency, region, True)
    with db.connect(write=True) as connection:
        if connection.execute("SELECT 1 FROM signers WHERE code = ?", (code,)).fetchone():
            raise RegistryError("That signer code is already registered.")
        now = utc_now()
        connection.execute(
            """INSERT INTO signers (id, code, dominant_hand, age_group, proficiency, region, is_active,
               created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid4()), code, *fields, now, now),
        )
        return find(connection, code)


def show(db: Database, code: str) -> Signer:
    with db.connect() as connection:
        return find(connection, code)


def list_signers(db: Database) -> list[Signer]:
    with db.connect() as connection:
        return [record(Signer, row) for row in connection.execute("SELECT * FROM signers ORDER BY code")]


def update(db: Database, code: str, **changes) -> Signer:
    columns = ("dominant_hand", "age_group", "proficiency", "region", "is_active")
    if unknown := changes.keys() - set(columns):
        raise RegistryError(f"Unsupported signer fields: {', '.join(sorted(unknown))}.")
    with db.connect(write=True) as connection:
        previous = find(connection, code)
        fields = _fields(*(changes.get(column, getattr(previous, column)) for column in columns))
        connection.execute(
            """UPDATE signers SET dominant_hand = ?, age_group = ?, proficiency = ?, region = ?,
               is_active = ?, updated_at = ? WHERE id = ?""",
            (*fields, utc_now(), previous.id),
        )
        return find(connection, code)


def consent(db: Database, code: str, status: str, reference: str | None = None) -> tuple[Signer, list[VideoSample]]:
    status = choice(status, "Consent status", CONSENT_STATUSES)
    with db.connect(write=True) as connection:
        signer = find(connection, code)
        if signer.consent_status == "WITHDRAWN" and status not in {"GRANTED", "WITHDRAWN"}:
            raise RegistryError("Consent is withdrawn. Record a new grant only after obtaining fresh consent.")
        now = utc_now()
        reference = optional_text(reference)
        connection.execute(
            """UPDATE signers SET consent_status = ?, consent_recorded_at = ?, consent_reference = ?,
               updated_at = ? WHERE id = ?""",
            (status, now, reference, now, signer.id),
        )
        connection.execute(
            "INSERT INTO signer_consent_events (signer_id, status, recorded_at, reference) VALUES (?, ?, ?, ?)",
            (signer.id, status, now, reference),
        )
        affected = []
        if status == "WITHDRAWN":
            connection.execute(
                """UPDATE video_samples SET status = 'EXCLUDED' WHERE session_id IN
                   (SELECT id FROM recording_sessions WHERE signer_id = ?)""",
                (signer.id,),
            )
            affected = [record(VideoSample, row) for row in connection.execute(
                """SELECT v.* FROM video_samples AS v JOIN recording_sessions AS s ON s.id = v.session_id
                   WHERE s.signer_id = ? ORDER BY v.recorded_at, v.id""",
                (signer.id,),
            )]
        return find(connection, code), affected
