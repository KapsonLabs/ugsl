"""Exact internal timing and serializable processing reports."""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass(frozen=True, eq=False)
class DecodedFrame:
    sample_id: str
    frame_index: int
    timestamp_ms: Fraction
    relative_timestamp_ms: Fraction
    image: np.ndarray
    width: int
    height: int
    rotation_degrees: float | None = None


@dataclass(frozen=True)
class SamplingGap:
    first_grid_index: int
    missing_slots: int
    first_grid_timestamp_ms: int
    last_grid_timestamp_ms: int


@dataclass(frozen=True)
class TimestampIssue:
    kind: str
    frame_index: int
    timestamp_ms: int | None
    delta_ms: int | None = None


@dataclass
class ProcessingSummary:
    sample_id: str
    dataset_name: str | None = None
    dataset_version: str | None = None
    source_video_path: str | None = None
    width: int | None = None
    height: int | None = None
    rotation_degrees: float | None = None
    source_fps: float | None = None
    target_fps: float | None = None
    source_frames_decoded: int = 0
    decoded_frames: int = 0
    selected_frames: int = 0
    skipped_frames: int = 0
    start_ms: int = 0
    end_ms: int | None = None
    source_origin_pts_ms: int | None = None
    first_selected_timestamp_ms: int | None = None
    last_selected_timestamp_ms: int | None = None
    effective_output_fps: float | None = None
    variable_timing: bool = False
    timestamp_irregularities: list[TimestampIssue] = field(default_factory=list)
    sampling_gaps: list[SamplingGap] = field(default_factory=list)
    decoder_warnings: list[str] = field(default_factory=list)
    status: str = "PENDING"
    error_code: str | None = None
    error: str | None = None
    decoder_configuration: dict = field(default_factory=dict)
    implementation_version: str = "1.0"


@dataclass
class BatchSummary:
    dataset_name: str
    dataset_version: str
    total_samples: int
    samples: list[ProcessingSummary] = field(default_factory=list)
    dataset_error: str | None = None

    @property
    def processed(self) -> int:
        return sum(row.status == "PROCESSED" for row in self.samples)

    @property
    def failed(self) -> int:
        return sum(row.status == "FAILED" for row in self.samples)


HAND_SCHEMA_VERSION = "1.0"
HAND_SLOTS = ("LEFT", "RIGHT")
HAND_LANDMARK_SHAPE = (2, 21, 3)


@dataclass(eq=False)
class HandFrame:
    frame_index: int
    timestamp_ms: int
    image_landmarks: np.ndarray = field(default_factory=lambda: np.full(HAND_LANDMARK_SHAPE, np.nan, dtype=np.float32))
    world_landmarks: np.ndarray = field(default_factory=lambda: np.full(HAND_LANDMARK_SHAPE, np.nan, dtype=np.float32))
    presence_mask: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.bool_))
    handedness_score: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    raw_image_detections: int = 0
    detections: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    ambiguous_handedness: bool = False
    invalid_landmark_values: bool = False


@dataclass(frozen=True, eq=False)
class HandArrays:
    """Aligned frames with anatomical LEFT and RIGHT slots."""

    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    presence_mask: np.ndarray
    handedness_score: np.ndarray
    timestamps_ms: np.ndarray
    schema_version: str = HAND_SCHEMA_VERSION

    @classmethod
    def from_frames(cls, frames: list[HandFrame]) -> "HandArrays":
        if not frames:
            raise ValueError("Cannot build hand arrays without processed frames.")
        return cls(
            np.stack([frame.image_landmarks for frame in frames]),
            np.stack([frame.world_landmarks for frame in frames]),
            np.stack([frame.presence_mask for frame in frames]),
            np.stack([frame.handedness_score for frame in frames]),
            np.array([frame.timestamp_ms for frame in frames], dtype=np.int64),
        )

    def __post_init__(self):
        if self.timestamps_ms.ndim != 1 or self.timestamps_ms.dtype != np.int64:
            raise ValueError("Hand timestamps must be a one-dimensional int64 array.")
        count = len(self.timestamps_ms)
        if count and (self.timestamps_ms[0] < 0 or np.any(self.timestamps_ms[1:] <= self.timestamps_ms[:-1])):
            raise ValueError("Hand timestamps must be nonnegative and strictly increasing.")
        for array, shape, dtype in (
            (self.image_landmarks, (count, *HAND_LANDMARK_SHAPE), np.float32),
            (self.world_landmarks, (count, *HAND_LANDMARK_SHAPE), np.float32),
            (self.presence_mask, (count, 2), np.bool_),
            (self.handedness_score, (count, 2), np.float32),
        ):
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Hand array must have shape {shape} and dtype {np.dtype(dtype)}.")
        for array in (self.image_landmarks, self.world_landmarks, self.handedness_score):
            if not np.isnan(array[~self.presence_mask]).all():
                raise ValueError("Absent hand slots must contain NaN.")
        if not np.isfinite(self.image_landmarks[self.presence_mask]).all():
            raise ValueError("Present hand slots must have finite image landmarks.")
        scores = self.handedness_score[self.presence_mask]
        if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
            raise ValueError("Present hand slots must have a handedness score from 0 to 1.")
        world = self.world_landmarks[self.presence_mask]
        if not np.all(np.isfinite(world).all(axis=(1, 2)) | np.isnan(world).all(axis=(1, 2))):
            raise ValueError("World landmarks must be finite or entirely missing for each hand.")


@dataclass(frozen=True, eq=False)
class HandExtractionResult:
    report: dict
    arrays: HandArrays | None = None
