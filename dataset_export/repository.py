"""Transactional reads and writes for export snapshots."""

from dataclasses import asdict

from dataset_export.models import DatasetMember, DatasetVersion
from registry.database import RegistryError
from registry.repository import get, record, utc_now


def version(connection, name: str, number: str) -> DatasetVersion:
    return get(connection, DatasetVersion, "SELECT * FROM dataset_versions WHERE name = ? AND version = ?",
               (name.strip(), number), "Unknown dataset version.")


def by_id(connection, dataset_id: str) -> DatasetVersion | None:
    return record(DatasetVersion, connection.execute("SELECT * FROM dataset_versions WHERE id = ?", (dataset_id,)).fetchone())


def members(connection, dataset_id: str) -> list[DatasetMember]:
    return [record(DatasetMember, row) for row in connection.execute(
        "SELECT * FROM dataset_members WHERE dataset_version_id = ? ORDER BY sample_id", (dataset_id,),
    )]


def reserve(connection, dataset: DatasetVersion, membership: list[DatasetMember]) -> None:
    if connection.execute("SELECT 1 FROM dataset_versions WHERE name = ? AND version = ?",
                          (dataset.name, dataset.version)).fetchone():
        raise RegistryError("That dataset name and version are already registered.")
    if connection.execute("SELECT 1 FROM dataset_versions WHERE output_path = ?", (dataset.output_path,)).fetchone():
        raise RegistryError("That export directory is already registered.")
    values = asdict(dataset)
    connection.execute(
        f"INSERT INTO dataset_versions ({', '.join(values)}) VALUES ({', '.join(':' + key for key in values)})",
        values,
    )
    connection.executemany(
        """INSERT INTO dataset_members (id, dataset_version_id, sample_id, annotation_id, sign_id, signer_id,
           approval_round, source_checksum, manifest_entry_checksum, created_at)
           VALUES (:id, :dataset_version_id, :sample_id, :annotation_id, :sign_id, :signer_id,
           :approval_round, :source_checksum, :manifest_entry_checksum, :created_at)""",
        [asdict(member) for member in membership],
    )


def activate(connection, dataset_id: str) -> None:
    cursor = connection.execute(
        "UPDATE dataset_versions SET status = 'ACTIVE', finalized_at = ? WHERE id = ? AND status = 'BUILDING'",
        (utc_now(), dataset_id),
    )
    if cursor.rowcount != 1:
        raise RegistryError("Only a building dataset can become active.")


def fail(connection, dataset_id: str, reason: str) -> None:
    cursor = connection.execute(
        "UPDATE dataset_versions SET status = 'FAILED' WHERE id = ? AND status = 'BUILDING'", (dataset_id,),
    )
    if cursor.rowcount:
        connection.execute(
            """UPDATE dataset_events SET reason = ? WHERE id = (
                SELECT max(id) FROM dataset_events WHERE dataset_version_id = ?
            )""", (reason, dataset_id),
        )
