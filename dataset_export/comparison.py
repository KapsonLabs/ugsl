"""Compare immutable manifest snapshots by sample and annotation identity."""

from collections import Counter
from pathlib import Path

from dataset_export import manifest, repository
from dataset_export.integrity import load_metadata, load_vocabulary, read_checksums, verify_snapshot
from dataset_export.statistics import overall
from dataset_export.storage import safe_join, sha256_file
from registry.database import RegistryError


def load_registered_manifest(db, version):
    root = Path(version.output_path)
    data = load_metadata(root)
    words, classes = load_vocabulary(root)
    path = safe_join(root, data["manifest"])
    if sha256_file(path) != version.manifest_checksum:
        raise RegistryError("The manifest checksum differs from the registry snapshot.")
    records = list(manifest.read_records(path, classes))
    if manifest.fingerprint(records, words, data["configuration"]) != version.content_fingerprint:
        raise RegistryError("The manifest and vocabulary do not match the registered content fingerprint.")
    with db.connect() as connection:
        verify_snapshot(data, records, version, repository.members(connection, version.id))
    return data, records, classes


def load_statistics(db, version):
    data, records, _ = load_registered_manifest(db, version)
    root = Path(version.output_path)
    path = safe_join(root, data["statistics"])
    if sha256_file(path) != read_checksums(root).get("statistics.json"):
        raise RegistryError("The statistics checksum does not match the checksum list.")
    result = manifest.read_json(path)
    if not isinstance(result, dict) or result.get("overall") != overall(records):
        raise RegistryError("The statistics totals differ from the manifest.")
    return result


def compare(db, before, after) -> dict:
    _, old_records, old_classes = load_registered_manifest(db, before)
    _, new_records, new_classes = load_registered_manifest(db, after)
    old = {row.sample_id: row for row in old_records}
    new = {row.sample_id: row for row in new_records}
    changed, unchanged = [], []
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key].annotation, new[key].annotation
        if (a.id, a.version) != (b.id, b.version):
            changed.append({"sample_id": key, "from_annotation": a.id, "from_version": a.version,
                            "to_annotation": b.id, "to_version": b.version})
        else:
            unchanged.append(key)
    old_signers = {row.signer_id for row in old_records}
    new_signers = {row.signer_id for row in new_records}
    old_distribution = Counter(row.sign_id for row in old_records)
    new_distribution = Counter(row.sign_id for row in new_records)
    return {
        "name": before.name, "from_version": before.version, "to_version": after.version,
        "equivalent_content": before.content_fingerprint == after.content_fingerprint,
        "added_samples": sorted(new.keys() - old.keys()), "removed_samples": sorted(old.keys() - new.keys()),
        "unchanged_samples": unchanged, "changed_annotations": changed,
        "added_signs": sorted(new_classes.keys() - old_classes.keys()),
        "removed_signs": sorted(old_classes.keys() - new_classes.keys()),
        "class_mapping_changes": [
            {"sign_id": key, "from_index": old_classes.get(key, {}).get("class_index"),
             "to_index": new_classes.get(key, {}).get("class_index")}
            for key in sorted(old_classes.keys() | new_classes.keys())
            if old_classes.get(key, {}).get("class_index") != new_classes.get(key, {}).get("class_index")
        ],
        "added_signers": sorted(new_signers - old_signers), "removed_signers": sorted(old_signers - new_signers),
        "class_distribution_changes": [
            {"sign_id": key, "from_samples": old_distribution[key], "to_samples": new_distribution[key],
             "change": new_distribution[key] - old_distribution[key]}
            for key in sorted(old_classes.keys() | new_classes.keys()) if old_distribution[key] != new_distribution[key]
        ],
        "duration_change_ms": sum(row.duration_ms for row in new_records) - sum(row.duration_ms for row in old_records),
        "storage_change_bytes": sum(row.file_size_bytes for row in new_records) - sum(row.file_size_bytes for row in old_records),
    }
