"""Vocabulary management and normalized gloss lookup."""

from registry.database import Database, RegistryError
from registry.models import Sign
from registry.repository import choice, get, integer, normalize, optional_text, record, utc_now

SIGN_TYPES = ("STATIC", "DYNAMIC", "UNKNOWN")


def _fields(gloss, description, category, sign_type, variant_label, target_samples, is_active):
    gloss = normalize(gloss)
    if not gloss:
        raise RegistryError("Gloss must contain a label.")
    return (
        gloss, optional_text(description), optional_text(category), choice(sign_type, "Sign type", SIGN_TYPES),
        normalize(variant_label) or None if variant_label is not None else None,
        integer(target_samples, "Target samples", 0), int(is_active),
    )


def find(connection, sign_id: str) -> Sign:
    return get(connection, Sign, "SELECT * FROM signs WHERE id = ?", (sign_id.strip().upper(),), "Unknown sign ID.")


def by_gloss(connection, gloss: str) -> list[Sign]:
    return [record(Sign, row) for row in connection.execute(
        "SELECT * FROM signs WHERE gloss = ? ORDER BY id", (normalize(gloss),),
    )]


def _check_duplicate(connection, gloss, variant, sign_id=""):
    duplicate = connection.execute(
        "SELECT 1 FROM signs WHERE gloss = ? AND variant_label IS ? AND id <> ?",
        (gloss, variant, sign_id),
    ).fetchone()
    if duplicate:
        raise RegistryError("That gloss and variant already exist. Use the existing sign or choose a distinct variant.")


def add(
    db: Database, *, gloss: str, description=None, category=None, sign_type="UNKNOWN",
    variant_label=None, target_samples=0,
) -> Sign:
    fields = _fields(gloss, description, category, sign_type, variant_label, target_samples, True)
    with db.connect(write=True) as connection:
        _check_duplicate(connection, fields[0], fields[4])
        connection.execute("UPDATE sign_sequence SET last_id = last_id + 1")
        number = connection.execute("SELECT last_id FROM sign_sequence").fetchone()[0]
        sign_id = f"UGSL{number:04d}"
        now = utc_now()
        connection.execute(
            """INSERT INTO signs (id, gloss, description, category, sign_type, variant_label,
               target_samples, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sign_id, *fields, now, now),
        )
        return find(connection, sign_id)


def show(db: Database, sign_id: str) -> Sign:
    with db.connect() as connection:
        return find(connection, sign_id)


def update(db: Database, sign_id: str, **changes) -> Sign:
    columns = ("gloss", "description", "category", "sign_type", "variant_label", "target_samples", "is_active")
    if unknown := changes.keys() - set(columns):
        raise RegistryError(f"Unsupported vocabulary fields: {', '.join(sorted(unknown))}.")
    with db.connect(write=True) as connection:
        previous = find(connection, sign_id)
        fields = _fields(*(changes.get(column, getattr(previous, column)) for column in columns))
        _check_duplicate(connection, fields[0], fields[4], previous.id)
        connection.execute(
            """UPDATE signs SET gloss = ?, description = ?, category = ?, sign_type = ?, variant_label = ?,
               target_samples = ?, is_active = ?, updated_at = ? WHERE id = ?""",
            (*fields, utc_now(), previous.id),
        )
        return find(connection, previous.id)


def list_signs(db: Database, *, category=None, sign_type=None, active=None) -> list[Sign]:
    if sign_type is not None:
        sign_type = choice(sign_type, "Sign type", SIGN_TYPES)
    with db.connect() as connection:
        return [record(Sign, row) for row in connection.execute(
            """SELECT * FROM signs WHERE (? IS NULL OR category = ?) AND (? IS NULL OR sign_type = ?)
               AND (? IS NULL OR is_active = ?) ORDER BY gloss, variant_label, id""",
            (category, category, sign_type, sign_type, active, active),
        )]
