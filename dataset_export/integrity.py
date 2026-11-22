"""Artifact checks and lifecycle permission, usable with or without SQLite."""

import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset_export import manifest, vocabulary
from dataset_export.models import ANNOTATION_SCHEMA_VERSION, SCHEMA_VERSION, USABLE_STATUSES, DatasetMember, DatasetVersion
from dataset_export.statistics import overall
from dataset_export.storage import BUILD_MARKER, REVOCATION_FILE, relative_path, safe_join, sha256_file
from registry.database import RegistryError

REQUIRED_FILES = {"dataset.json", "manifest.jsonl", "vocabulary.json", "statistics.json", "README.md"}


@dataclass(frozen=True)
class Permission:
    status: str
    permitted: bool | None
    reason: str | None = None


@dataclass(frozen=True)
class Verification:
    integrity_ok: bool
    permission: Permission
    errors: list[str]
    total_samples: int | None = None


def load_metadata(root: Path) -> dict:
    data = manifest.read_json(safe_join(root, "dataset.json"))
    if (
        not isinstance(data, dict) or not manifest.is_uuid(data.get("dataset_id"))
        or data.get("language") != "UGSL" or data.get("schema_version") != SCHEMA_VERSION
        or data.get("annotation_schema_version") != ANNOTATION_SCHEMA_VERSION
        or data.get("export_mode") not in ("PORTABLE", "REFERENCE")
        or not manifest.is_digest(data.get("manifest_checksum")) or not manifest.is_digest(data.get("content_fingerprint"))
        or not isinstance(data.get("configuration"), dict)
        or any(key not in data["configuration"] for key in manifest.CONTENT_OPTIONS)
        or any(not isinstance(data.get(key), str) or not data[key] for key in ("name", "version", "created_at"))
        or any(type(data.get(key)) is not int or data[key] <= 0 for key in (
            "total_samples", "total_signs", "total_signers", "total_recording_sessions", "total_duration_ms", "total_video_bytes",
        ))
        or data.get("manifest") != "manifest.jsonl" or data.get("vocabulary") != "vocabulary.json"
        or data.get("statistics") != "statistics.json" or data.get("manifest_csv") not in (None, "manifest.csv")
        or data.get("sample_types") != ["ISOLATED"]
    ):
        raise RegistryError("Invalid or unsupported dataset metadata.")
    manifest.utc_timestamp(data["created_at"])
    return data


def load_vocabulary(root: Path) -> tuple[dict, dict]:
    payload = manifest.read_json(safe_join(root, "vocabulary.json"))
    return payload, vocabulary.validate_vocabulary(payload)


def resolve_video(root: Path, mode: str, record, storage_root: Path | None) -> Path:
    if mode == "REFERENCE":
        if storage_root is None:
            raise RegistryError("Reference datasets require a configured storage root.")
        return safe_join(storage_root, record.video_path)
    path = relative_path(record.video_path)
    if path.parts != ("videos", record.sign_id, record.sample_id + path.suffix) or not path.suffix:
        raise RegistryError("Invalid portable video path.")
    return safe_join(root, record.video_path)


def find_registry(root: Path, registry_path: Path | None) -> tuple[Path | None, bool]:
    if registry_path is not None:
        return Path(registry_path).expanduser().resolve(), True
    if env := os.environ.get("UGSL_REGISTRY_DB"):
        return Path(env).expanduser().resolve(), True
    for parent in (root.parent, root.parent.parent):
        candidate = parent / "registry.db"
        if candidate.is_file():
            return candidate, False
    return None, False


def registry_snapshot(root: Path, dataset_id: str, registry_path: Path | None, *, include_members=True):
    path, required = find_registry(root, registry_path)
    if path is None:
        return None, None
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            has_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dataset_versions'"
            ).fetchone()
            row = connection.execute("SELECT * FROM dataset_versions WHERE id = ?", (dataset_id,)).fetchone() if has_table else None
            if row:
                membership = [DatasetMember(**dict(member)) for member in connection.execute(
                    "SELECT * FROM dataset_members WHERE dataset_version_id = ? ORDER BY sample_id", (dataset_id,),
                )] if include_members else None
                return DatasetVersion(**dict(row)), membership
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise RegistryError("Could not check the dataset lifecycle in the registry.") from exc
    if required:
        raise RegistryError("The configured registry has no record for this dataset.")
    return None, None


def permission(root: Path, dataset_id: str, version: DatasetVersion | None, revocation_path: Path | None = None) -> Permission:
    receipt = Path(revocation_path) if revocation_path is not None else root / REVOCATION_FILE
    if receipt.exists() or receipt.is_symlink() or revocation_path is not None:
        try:
            if revocation_path is None:
                receipt = safe_join(root, REVOCATION_FILE)
            data = manifest.read_json(receipt)
            if (
                not isinstance(data, dict) or data.get("dataset_id") != dataset_id
                or data.get("status") != "REVOKED" or not isinstance(data.get("reason"), str)
                or not data["reason"].strip()
            ):
                raise RegistryError("The revocation record does not match this dataset.")
            return Permission("REVOKED", False, data["reason"])
        except RegistryError as exc:
            return Permission("UNKNOWN", False, str(exc))
    if version is not None and version.status == "REVOKED":
        return Permission("REVOKED", False, version.revocation_reason)
    marker = root / BUILD_MARKER
    if marker.exists() or marker.is_symlink() or root.name.startswith(".building-"):
        return Permission("BUILDING", False, "The export has not completed publication.")
    if version is None:
        return Permission("UNKNOWN", None, "Current registry lifecycle information is unavailable.")
    return Permission(version.status, version.status in USABLE_STATUSES, version.revocation_reason)


def read_checksums(root: Path) -> dict[str, str]:
    path = safe_join(root, "checksums.sha256")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        raise RegistryError("Cannot read checksums.sha256.") from exc
    checksums = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or not manifest.is_digest(digest) or name in checksums:
            raise RegistryError("Malformed or duplicate checksum entry.")
        relative_path(name)
        checksums[name] = digest
    return checksums


def verify_snapshot(data: dict, records: list, version: DatasetVersion, membership: list[DatasetMember]) -> None:
    if (
        (data["dataset_id"], data["name"], data["version"], data["schema_version"], data["export_mode"])
        != (version.id, version.name, version.version, version.schema_version, version.export_mode)
        or (data["manifest_checksum"], data["content_fingerprint"]) != (version.manifest_checksum, version.content_fingerprint)
        or (data["total_samples"], data["total_signs"], data["total_signers"])
        != (version.total_samples, version.total_signs, version.total_signers)
    ):
        raise RegistryError("Dataset metadata differs from the registry snapshot.")
    by_id = {row.sample_id: row for row in membership}
    if len(records) != len(by_id):
        raise RegistryError("Manifest membership count differs from the registry snapshot.")
    for record in records:
        member = by_id.get(record.sample_id)
        if (
            member is None or member.annotation_id != record.annotation.id or member.sign_id != record.sign_id
            or member.source_checksum != record.sha256
            or member.manifest_entry_checksum != manifest.sha256_text(manifest.canonical(asdict(record)))
        ):
            raise RegistryError("A manifest entry differs from the registry membership snapshot.")


class DatasetIntegrityVerifier:
    def verify(
        self, directory: Path, *, storage_root: Path | None = None, registry_path: Path | None = None,
        revocation_path: Path | None = None, version: DatasetVersion | None = None, membership=None,
    ) -> Verification:
        root = Path(directory).expanduser().resolve()
        storage_root = Path(storage_root).expanduser().resolve() if storage_root is not None else None
        access = Permission("UNKNOWN", None, "Dataset lifecycle could not be determined.")
        errors, records = [], None
        refresh_lifecycle = version is None or registry_path is not None
        try:
            data = load_metadata(root)
        except (RegistryError, OSError, ValueError) as exc:
            return Verification(False, access, [str(exc)])
        try:
            if version is None:
                version, membership = registry_snapshot(root, data["dataset_id"], registry_path)
            access = permission(root, data["dataset_id"], version, revocation_path)
        except RegistryError as exc:
            access = Permission("UNKNOWN", False, str(exc))
        try:
            words, classes = load_vocabulary(root)
            records = list(manifest.read_records(safe_join(root, data["manifest"]), classes))
            totals = overall(records)
            if any(data[key] != value for key, value in totals.items()):
                raise RegistryError("Manifest counts, duration or storage differ from dataset metadata.")
            stats = manifest.read_json(safe_join(root, data["statistics"]))
            if not isinstance(stats, dict) or stats.get("overall") != totals:
                raise RegistryError("Statistics totals differ from the manifest.")
            if manifest.fingerprint(records, words, data["configuration"]) != data["content_fingerprint"]:
                raise RegistryError("Dataset content fingerprint does not match the manifest and vocabulary.")
            checksums = read_checksums(root)
            paths = REQUIRED_FILES | ({"manifest.csv"} if data.get("manifest_csv") else set())
            if data["export_mode"] == "PORTABLE":
                video_paths = {record.video_path for record in records}
                if len(video_paths) != len(records):
                    raise RegistryError("Duplicate portable video path.")
                video_root = safe_join(root, "videos")
                actual = {path.relative_to(root).as_posix() for path in video_root.rglob("*") if path.is_file() or path.is_symlink()}
                if actual != video_paths:
                    raise RegistryError("The portable video directory differs from the manifest.")
                paths |= video_paths
            if paths != set(checksums):
                raise RegistryError("The checksum file does not cover exactly the expected artifact files.")
            for name in sorted(paths):
                actual = sha256_file(safe_join(root, name))
                if actual != checksums[name] or (name == "manifest.jsonl" and actual != data["manifest_checksum"]):
                    errors.append(f"Checksum mismatch: {name}")
            for record in records:
                path = resolve_video(root, data["export_mode"], record, storage_root)
                if not path.is_file() or path.stat().st_size != record.file_size_bytes:
                    errors.append(f"Missing, unreadable or wrong-sized video: {record.sample_id}")
                elif data["export_mode"] == "REFERENCE":
                    if sha256_file(path) != record.sha256:
                        errors.append(f"Source checksum mismatch: {record.sample_id}")
                elif checksums[record.video_path] != record.sha256:
                    errors.append(f"Portable checksum differs from the record: {record.sample_id}")
            if version is not None:
                verify_snapshot(data, records, version, membership)
        except (RegistryError, OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
        if refresh_lifecycle:
            try:
                latest, _ = registry_snapshot(root, data["dataset_id"], registry_path, include_members=False)
                access = permission(root, data["dataset_id"], latest, revocation_path)
            except RegistryError as exc:
                access = Permission("UNKNOWN", False, str(exc))
        return Verification(not errors, access, errors, len(records) if records is not None else None)
