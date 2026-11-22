"""SQLite connections, transactional migrations and operator-safe errors."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS = Path(__file__).with_name("migrations")


class RegistryError(Exception):
    """An error that can be shown to the operator."""


class DatabaseError(RegistryError):
    """A registry connection, migration, or transaction failed."""


def _scripts() -> list[Path]:
    migrations = sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations or [int(path.name[:3]) for path in migrations] != list(range(1, len(migrations) + 1)):
        raise DatabaseError("Registry migrations are missing or out of order.")
    return migrations


def _migrate(connection: sqlite3.Connection, migrations: list[Path]) -> None:
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current > len(migrations):
                raise DatabaseError("The registry schema is newer than this application.")
            for version, script in enumerate(migrations, start=1):
                if version <= current:
                    continue
                # executescript commits first. Execute complete statements inside our transaction.
                statement = ""
                for character in script.read_text(encoding="utf-8"):
                    statement += character
                    if character == ";" and sqlite3.complete_statement(statement):
                        connection.execute(statement)
                        statement = ""
                if statement.strip():
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {version}")
    except (sqlite3.Error, OSError, UnicodeError) as exc:
        raise DatabaseError("Registry migration failed. Check the SQL migrations and database access.") from exc


class Database:
    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path).expanduser()
        self.read_only = read_only

    @contextmanager
    def connect(self, *, write: bool = False):
        """Open a connection and commit or roll back one unit of work."""
        if write and self.read_only:
            raise RegistryError("The registry was opened read-only.")
        connection = None
        try:
            migrations = _scripts()
            in_memory = self.read_only and not self.path.exists()
            if in_memory:
                connection = sqlite3.connect(":memory:")
            elif self.read_only:
                connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > len(migrations):
                raise DatabaseError("The registry schema is newer than this application.")
            if version < len(migrations):
                if self.read_only and not in_memory:
                    snapshot = sqlite3.connect(":memory:")
                    try:
                        connection.backup(snapshot)
                    except BaseException:
                        snapshot.close()
                        raise
                    connection.close()
                    connection = snapshot
                    connection.row_factory = sqlite3.Row
                    connection.execute("PRAGMA foreign_keys = ON")
                _migrate(connection, migrations)
            if not write:
                connection.execute("PRAGMA query_only = ON")

            with connection:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
        except (sqlite3.Error, OSError, OverflowError) as exc:
            raise DatabaseError("Registry database operation failed. Check database access, locks and disk space.") from exc
        finally:
            if connection is not None:
                connection.close()
