"""Dataset lifecycle, audit history and impact queries."""

from dataset_export import repository
from dataset_export.models import DatasetVersion
from dataset_export.versioning import version_key
from registry import signers
from registry.database import Database, RegistryError
from registry.repository import record, optional_text, utc_now


def show(db: Database, name: str, number: str) -> DatasetVersion:
    with db.connect() as connection:
        return repository.version(connection, name, number)


def list_versions(db: Database, *, name=None, statuses=()) -> list[DatasetVersion]:
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM dataset_versions WHERE ? IS NULL OR name = ?", (name, name))
        versions = [record(DatasetVersion, row) for row in rows if not statuses or row["status"] in statuses]
    return sorted(versions, key=lambda item: (item.name, version_key(item.version)))


def affected(db: Database, *, signer=None, sample_id=None, annotation_id=None) -> list[DatasetVersion]:
    if signer is None and sample_id is None and annotation_id is None:
        raise RegistryError("Provide a signer, sample or annotation identifier.")
    with db.connect() as connection:
        signer_id = None
        if signer is not None:
            row = connection.execute("SELECT id FROM signers WHERE id = ?", (signer.lower(),)).fetchone()
            signer_id = row["id"] if row else signers.find(connection, signer).id
        return [record(DatasetVersion, row) for row in connection.execute(
            """SELECT DISTINCT v.* FROM dataset_versions v JOIN dataset_members m ON m.dataset_version_id = v.id
               WHERE (? IS NULL OR m.signer_id = ?) AND (? IS NULL OR m.sample_id = ?)
               AND (? IS NULL OR m.annotation_id = ?) ORDER BY v.created_at, v.id""",
            (signer_id, signer_id, sample_id, sample_id, annotation_id, annotation_id),
        )]


def transition(db: Database, name: str, number: str, status: str, reason: str | None = None) -> DatasetVersion:
    with db.connect(write=True) as connection:
        dataset = repository.version(connection, name, number)
        if status == "REVOKED":
            reason = optional_text(reason)
            if not reason:
                raise RegistryError("Dataset revocation requires a reason.")
            if dataset.status == status:
                return dataset
            if dataset.status not in {"ACTIVE", "SUPERSEDED", "BUILDING"}:
                raise RegistryError(f"A {dataset.status} dataset cannot be revoked.")
            connection.execute(
                "UPDATE dataset_versions SET status = ?, revoked_at = ?, revocation_reason = ? WHERE id = ?",
                (status, utc_now(), reason, dataset.id),
            )
        elif status == "SUPERSEDED":
            if dataset.status == status:
                return dataset
            if dataset.status != "ACTIVE":
                raise RegistryError("Only an active dataset can be superseded.")
            connection.execute("UPDATE dataset_versions SET status = ? WHERE id = ?", (status, dataset.id))
        else:
            raise RegistryError("Choose a supported dataset lifecycle transition.")
        return repository.version(connection, name, number)


def history(db: Database, dataset_id: str) -> list[dict]:
    with db.connect() as connection:
        return [dict(row) for row in connection.execute(
            """SELECT previous_status, status, reason, created_at FROM dataset_events
               WHERE dataset_version_id = ? ORDER BY id""",
            (dataset_id,),
        )]
