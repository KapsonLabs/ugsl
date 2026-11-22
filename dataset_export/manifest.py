"""Stable JSONL records, optional CSV and content equivalence."""

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from dataset_export.models import AnnotationRecord, ManifestRecord, SCHEMA_VERSION
from dataset_export.storage import relative_path
from registry.database import RegistryError

DIGEST = re.compile(r"[0-9a-f]{64}")
CONTENT_OPTIONS = (
    "sign_ids", "categories", "signer_ids", "session_ids", "min_samples_per_sign", "min_signers_per_sign",
    "include_warnings", "max_samples_per_sign", "allow_incomplete_coverage", "review_policy",
)


def canonical(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(canonical(payload) + "\n")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RegistryError(f"Cannot read valid JSON from {path.name}.") from exc


def utc_timestamp(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.utcoffset() is not None:
            return timestamp.astimezone(UTC).isoformat()
    except (ValueError, TypeError, OverflowError):
        pass
    raise RegistryError("Invalid recording timestamp. Use an ISO 8601 value with an explicit offset.")


def is_uuid(value) -> bool:
    try:
        return str(UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def is_digest(value) -> bool:
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def _finite(value) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def orientation(width: int, height: int, rotation: float | None) -> str:
    if rotation is not None and rotation % 180 == 90:
        width, height = height, width
    return "LANDSCAPE" if width > height else "PORTRAIT" if height > width else "SQUARE"


def validate_record(value: dict, classes: dict[str, dict]) -> ManifestRecord:
    try:
        if not isinstance(value, dict):
            raise ValueError
        data = dict(value)
        annotation = AnnotationRecord(**data.pop("annotation"))
        record = ManifestRecord(**data, annotation=annotation)
        if not all(is_uuid(item) for item in (record.sample_id, record.session_id, annotation.id)):
            raise ValueError
        if not isinstance(record.sign_id, str) or record.sign_id not in classes or not is_digest(record.sha256):
            raise ValueError
        sign = classes[record.sign_id]
        if type(record.class_index) is not int or record.class_index != sign["class_index"] or record.gloss != sign["gloss"]:
            raise ValueError
        if not isinstance(record.signer_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", record.signer_id):
            raise ValueError
        if any(type(item) is not int or item <= 0 for item in (
            record.width, record.height, record.frame_count, record.duration_ms, record.file_size_bytes, annotation.version,
        )):
            raise ValueError
        if type(annotation.start_ms) is not int or type(annotation.end_ms) is not int:
            raise ValueError
        if not 0 <= annotation.start_ms < annotation.end_ms <= record.duration_ms:
            raise ValueError
        if record.average_fps is not None and (not _finite(record.average_fps) or record.average_fps <= 0):
            raise ValueError
        if record.rotation is not None and not _finite(record.rotation):
            raise ValueError
        if record.orientation != orientation(record.width, record.height, record.rotation):
            raise ValueError
        if annotation.confidence is not None and not _finite(annotation.confidence):
            raise ValueError
        for label, allowed in (
            (annotation.hand_usage, ("ONE_HANDED", "TWO_HANDED", "UNKNOWN")),
            (annotation.dominant_hand, ("LEFT", "RIGHT", "BOTH", "UNKNOWN")),
            (annotation.facial_expression, ("PRESENT", "ABSENT", "UNKNOWN")),
            (annotation.movement_type, ("STATIC", "DYNAMIC", "UNKNOWN")),
            (annotation.completeness, ("COMPLETE", "INCOMPLETE", "UNKNOWN")),
            (record.source_type, ("CAMERA", "IMPORT")),
            (record.technical_validation_status, ("PASSED", "PASSED_WITH_WARNINGS")),
        ):
            if label not in allowed:
                raise ValueError
        if record.recorded_at is not None and utc_timestamp(record.recorded_at) != record.recorded_at:
            raise ValueError
        if any(item is not None and not isinstance(item, str) for item in (record.container_format, record.video_codec)):
            raise ValueError
        relative_path(record.video_path)
        return record
    except (RegistryError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RegistryError("Invalid manifest record, vocabulary reference, or annotation interval.") from exc


def read_records(path: Path, classes: dict[str, dict]):
    previous = None
    seen = set()
    class_ids = set()
    try:
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                try:
                    record = validate_record(json.loads(line), classes)
                    key = (record.class_index, record.sample_id)
                    if record.sample_id in seen or (previous is not None and previous >= key):
                        raise RegistryError("Duplicate or out-of-order sample.")
                    seen.add(record.sample_id)
                    class_ids.add(record.sign_id)
                    previous = key
                    yield record
                except (RegistryError, ValueError) as exc:
                    raise RegistryError(f"Invalid manifest line {number}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise RegistryError("Cannot read the UTF-8 JSONL manifest.") from exc
    if class_ids != set(classes):
        raise RegistryError("The manifest and vocabulary class sets differ.")


def jsonl(records: list[ManifestRecord]) -> str:
    return "".join(canonical(asdict(record)) + "\n" for record in records)


def fingerprint(records: list[ManifestRecord], vocabulary: dict, configuration: dict) -> str:
    logical = []
    for record in records:
        fields = asdict(record)
        del fields["video_path"]
        logical.append(fields)
    options = {key: configuration[key] for key in CONTENT_OPTIONS}
    return sha256_text(canonical({
        "schema_version": SCHEMA_VERSION, "records": logical, "vocabulary": vocabulary, "selection": options,
    }))


def write_csv(path: Path, records: list[ManifestRecord]) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        annotation = row.pop("annotation")
        row.update({f"annotation_{key}": value for key, value in annotation.items()})
        rows.append({key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                     for key, value in row.items()})
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
