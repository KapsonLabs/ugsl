"""Standard-library dataset reader for isolated-sign manifests."""

import os
from dataclasses import replace
from pathlib import Path

from dataset_export import manifest
from dataset_export.integrity import load_metadata, load_vocabulary, permission, registry_snapshot, resolve_video
from dataset_export.storage import safe_join, sha256_file
from registry.database import RegistryError


class DatasetReader:
    def __init__(self, directory, *, storage_root=None, registry_path=None, revocation_path=None):
        self.directory = Path(directory).expanduser().resolve()
        root = storage_root or os.environ.get("UGSL_REFERENCE_ROOT")
        self.storage_root = Path(root).expanduser().resolve() if root is not None else None
        self.registry_path = registry_path
        self.revocation_path = revocation_path
        self.metadata = load_metadata(self.directory)
        self.schema_version = self.metadata["schema_version"]
        self.vocabulary, self.classes = load_vocabulary(self.directory)
        self._manifest_path = safe_join(self.directory, self.metadata["manifest"])
        self._require_permission()
        if sha256_file(self._manifest_path) != self.metadata["manifest_checksum"]:
            raise RegistryError("The manifest checksum does not match dataset metadata.")
        if self.metadata["export_mode"] == "REFERENCE" and self.storage_root is None:
            raise RegistryError("Reference datasets require a configured storage root.")

    def _require_permission(self):
        version, _ = registry_snapshot(self.directory, self.metadata["dataset_id"], self.registry_path, include_members=False)
        access = permission(self.directory, self.metadata["dataset_id"], version, self.revocation_path)
        if access.permitted is False:
            raise RegistryError(f"Dataset use is blocked ({access.status}). {access.reason or ''}".strip())
        if version and version.manifest_checksum != self.metadata["manifest_checksum"]:
            raise RegistryError("The manifest checksum differs from the registered dataset snapshot.")

    def __iter__(self):
        return self.iter_samples()

    def iter_samples(self, sign_id: str | None = None):
        self._require_permission()
        count = 0
        for record in manifest.read_records(self._manifest_path, self.classes):
            count += 1
            if sign_id is not None and record.sign_id != sign_id.strip().upper():
                continue
            self._require_permission()
            path = resolve_video(self.directory, self.metadata["export_mode"], record, self.storage_root)
            if not path.is_file():
                raise RegistryError(f"Missing dataset video for {record.sample_id}.")
            try:
                with path.open("rb"):
                    pass
            except OSError as exc:
                raise RegistryError(f"Unreadable dataset video for {record.sample_id}.") from exc
            yield replace(record, video_path=path)
        if count != self.metadata["total_samples"]:
            raise RegistryError("Manifest sample count differs from dataset metadata.")
