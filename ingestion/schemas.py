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
