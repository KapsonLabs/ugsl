"""Small shared helpers for registry queries and input validation."""

import unicodedata
from datetime import UTC, datetime

from registry.database import RegistryError


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def record(model, row):
    if row is None:
        return None
    values = dict(row)
    if "is_active" in values:
        values["is_active"] = bool(values["is_active"])
    return model(**values)


def get(connection, model, query, parameters, missing):
    result = record(model, connection.execute(query, parameters).fetchone())
    if result is None:
        raise RegistryError(missing)
    return result


def optional_text(value: str | None) -> str | None:
    return (value.strip() or None) if value is not None else None


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).upper()


def choice(value: str, name: str, allowed: tuple[str, ...]) -> str:
    normalized = value.strip().upper()
    if normalized not in allowed:
        raise RegistryError(f"{name} must be one of {', '.join(allowed)}.")
    return normalized


def integer(value: int, name: str, minimum: int) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise RegistryError(f"{name} must be a SQLite integer of at least {minimum}.")
    return value
