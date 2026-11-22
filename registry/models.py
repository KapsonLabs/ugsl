"""Registry records returned by SQLite queries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sign:
    id: str
    gloss: str
    description: str | None
    category: str | None
    sign_type: str
    variant_label: str | None
    target_samples: int
    is_active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Signer:
    id: str
    code: str
    dominant_hand: str
    age_group: str | None
    proficiency: str
    region: str | None
    consent_status: str
    consent_recorded_at: str | None
    consent_reference: str | None
    is_active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RecordingSession:
    id: str
    signer_id: str
    operator: str | None
    camera_source_type: str
    camera_angle: str
    lighting_condition: str
    background: str
    status: str
    notes: str | None
    started_at: str
    ended_at: str | None


@dataclass(frozen=True)
class VideoSample:
    id: str
    sign_id: str
    session_id: str
    sample_type: str
    repetition: int | None
    file_path: str
    metadata_path: str
    width: int
    height: int
    requested_fps: float | None
    measured_fps: float | None
    encoded_fps: float | None
    frame_count: int
    failed_frame_reads: int | None
    requested_duration_seconds: float | None
    actual_duration_seconds: float | None
    file_size_bytes: int
    status: str
    recorded_at: str | None
    registered_at: str
    sha256: str | None = None
    source_type: str = "CAMERA"
    source_reference: str | None = None
    notes: str | None = None
    container_format: str | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    average_fps: float | None = None
    nominal_fps: float | None = None
    duration_seconds: float | None = None
    rotation: float | None = None
    bitrate: int | None = None
    technical_validation_status: str = "PENDING"
    technical_validated_at: str | None = None
    validator_version: str | None = None
    review_state: str = "PENDING"
    current_annotation_id: str | None = None
    review_round: int = 1
