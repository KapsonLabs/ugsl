"""Export configuration, registry records and manifest records."""

from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"
USABLE_STATUSES = {"ACTIVE", "SUPERSEDED"}


@dataclass(frozen=True)
class ExportConfig:
    name: str
    version: str | None = None
    description: str | None = None
    mode: str = "REFERENCE"
    output_directory: Path | None = None
    storage_root: Path | None = None
    sign_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    signer_ids: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    min_samples_per_sign: int = 1
    min_signers_per_sign: int = 1
    include_warnings: bool = True
    max_samples_per_sign: int | None = None
    csv: bool = False
    verify_all_source_checksums: bool = True
    dry_run: bool = False
    allow_incomplete_coverage: bool = False
    bump: str | None = None


@dataclass(frozen=True)
class DatasetVersion:
    id: str
    name: str
    version: str
    schema_version: str
    description: str | None
    export_mode: str
    status: str
    total_samples: int
    total_signs: int
    total_signers: int
    manifest_checksum: str
    content_fingerprint: str
    configuration_json: str
    output_path: str
    created_at: str
    finalized_at: str | None = None
    revoked_at: str | None = None
    revocation_reason: str | None = None


@dataclass(frozen=True)
class DatasetMember:
    id: str
    dataset_version_id: str
    sample_id: str
    annotation_id: str
    sign_id: str
    signer_id: str
    approval_round: int
    source_checksum: str
    manifest_entry_checksum: str
    created_at: str


@dataclass(frozen=True)
class AnnotationRecord:
    id: str
    version: int
    start_ms: int
    end_ms: int
    hand_usage: str
    dominant_hand: str
    movement_type: str
    facial_expression: str
    completeness: str
    confidence: float | None


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    sign_id: str
    gloss: str
    class_index: int
    signer_id: str
    session_id: str
    video_path: str | Path
    sha256: str
    container_format: str | None
    video_codec: str | None
    width: int
    height: int
    average_fps: float | None
    frame_count: int
    duration_ms: int
    orientation: str
    annotation: AnnotationRecord
    technical_validation_status: str
    source_type: str
    recorded_at: str | None
    file_size_bytes: int
    rotation: float | None
