"""Recording states and the accepted sample sidecar schema."""

from dataclasses import dataclass
from enum import StrEnum


class RecordingState(StrEnum):
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    RECORDING = "RECORDING"
    REVIEW = "REVIEW"
    PLAYBACK = "PLAYBACK"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class RecordingContext:
    """Optional labels carried by an accepted take for later reconciliation."""

    sign_id: str
    session_id: str
    gloss: str


@dataclass(frozen=True)
class RecordingMetadata:
    sample_id: str
    gloss: str
    signer_id: str
    repetition: int
    camera_index: int | None
    width: int
    height: int
    requested_fps: float
    measured_fps: float
    encoded_fps: float
    frame_count: int
    failed_frame_reads: int
    requested_duration_seconds: float
    actual_duration_seconds: float
    file_size_bytes: int
    file_path: str
    recorded_at: str
    sign_id: str | None = None
    session_id: str | None = None


class ValidationStatus(StrEnum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
    FAILED = "FAILED"
    ERROR = "ERROR"


class IngestionOutcome(StrEnum):
    INGESTED = "INGESTED"
    INGESTED_WITH_WARNINGS = "INGESTED_WITH_WARNINGS"
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class VideoMetadata:
    container_format: str
    extension: str
    video_codec: str | None
    pixel_format: str | None
    width: int | None
    height: int | None
    rotation: float | None
    display_orientation: str | None
    average_fps: float | None
    nominal_fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    stream_duration_seconds: float | None
    bitrate: int | None
    file_size_bytes: int
    has_audio: bool
    creation_time: str | None


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: str
    actual: object = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ValidationReport:
    validation_status: ValidationStatus
    checks: list[ValidationCheck]
    errors: list[str]
    warnings: list[str]
    metrics: dict
    validation_profile: str = "ISOLATED"
    validator_version: str = "1.0"


@dataclass(frozen=True)
class IngestionResult:
    attempt_id: str
    outcome: IngestionOutcome
    report: ValidationReport
    sample_id: str | None = None
    sha256: str | None = None
    error_code: str | None = None
    message: str | None = None
    storage_bytes: int = 0
    dry_run: bool = False
