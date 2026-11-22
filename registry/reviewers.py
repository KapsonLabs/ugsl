"""Local reviewer records and workflow qualification checks."""

from dataclasses import dataclass
from uuid import uuid4

from config import validate_identifier
from registry.database import Database, RegistryError
from registry.repository import choice, get, optional_text, record, utc_now

PROFICIENCIES = ("FLUENT", "NATIVE", "QUALIFIED_INTERPRETER", "OTHER")
ROLES = ("ANNOTATOR", "REVIEWER", "ADJUDICATOR")


@dataclass(frozen=True)
class Reviewer:
    id: str
    code: str
    display_name: str | None
    proficiency: str
    role: str
    qualification_notes: str | None
    is_active: bool
    created_at: str
    updated_at: str


def workflow_choice(value: str, label: str, choices: tuple[str, ...]) -> str:
    return choice(value.replace("-", "_").replace(" ", "_"), label, choices)


def normalize_code(code: str) -> str:
    code = code.strip()
    try:
        validate_identifier(code, "Reviewer code")
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    return code.upper()


def find(connection, code: str) -> Reviewer:
    return get(connection, Reviewer, "SELECT * FROM reviewers WHERE code = ?", (normalize_code(code),),
               "Unknown reviewer code.")


def require_active(connection, code: str, *, decision=False) -> Reviewer:
    reviewer = find(connection, code)
    if not reviewer.is_active:
        raise RegistryError("The reviewer is inactive.")
    if decision and (reviewer.role not in {"REVIEWER", "ADJUDICATOR"} or reviewer.proficiency == "OTHER"):
        raise RegistryError("Review decisions require a reviewer or adjudicator with declared linguistic proficiency.")
    return reviewer


def add(db: Database, *, code: str, proficiency: str, role: str, display_name=None, qualification_notes=None) -> Reviewer:
    code = normalize_code(code)
    proficiency = workflow_choice(proficiency.strip(), "Reviewer proficiency", PROFICIENCIES)
    role = workflow_choice(role.strip(), "Reviewer role", ROLES)
    with db.connect(write=True) as connection:
        if connection.execute("SELECT 1 FROM reviewers WHERE code = ?", (code,)).fetchone():
            raise RegistryError("That reviewer code is already registered.")
        now = utc_now()
        connection.execute(
            """INSERT INTO reviewers (id, code, display_name, proficiency, role, qualification_notes,
               created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid4()), code, optional_text(display_name), proficiency, role, optional_text(qualification_notes), now, now),
        )
        return find(connection, code)


def show(db: Database, code: str) -> Reviewer:
    with db.connect() as connection:
        return find(connection, code)


def list_reviewers(db: Database) -> list[Reviewer]:
    with db.connect() as connection:
        return [record(Reviewer, row) for row in connection.execute("SELECT * FROM reviewers ORDER BY code")]


def deactivate(db: Database, code: str) -> Reviewer:
    with db.connect(write=True) as connection:
        reviewer = find(connection, code)
        connection.execute("UPDATE reviewers SET is_active = 0, updated_at = ? WHERE id = ?", (utc_now(), reviewer.id))
        return find(connection, code)
