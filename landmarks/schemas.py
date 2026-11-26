"""Exact internal timing and serializable processing reports."""

from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from itertools import pairwise

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
    dataset_id: str | None = None
    annotation_id: str | None = None


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


POSE_SCHEMA_VERSION = "1.0"


class PoseLandmark(IntEnum):
    """MediaPipe Pose Landmarker landmark order."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_LANDMARK_COUNT = len(PoseLandmark)
POSE_GROUPS = {
    "upper_body": tuple(range(25)),
    "head": tuple(range(11)),
    "shoulders": (11, 12),
    "arms": (11, 12, 13, 14, 15, 16),
    "elbows": (13, 14),
    "wrists": (15, 16),
    "hand_reference": (15, 16, 17, 18, 19, 20, 21, 22),
    "torso": (11, 12, 23, 24),
    "lower_body": tuple(range(23, POSE_LANDMARK_COUNT)),
}


@dataclass(eq=False)
class PoseFrame:
    frame_index: int
    source_timestamp: Fraction
    timestamp_ms: int
    inference_timestamp_ms: int
    image_landmarks: np.ndarray = field(default_factory=lambda: np.full((POSE_LANDMARK_COUNT, 3), np.nan, dtype=np.float32))
    world_landmarks: np.ndarray = field(default_factory=lambda: np.full((POSE_LANDMARK_COUNT, 3), np.nan, dtype=np.float32))
    visibility: np.ndarray = field(default_factory=lambda: np.full(POSE_LANDMARK_COUNT, np.nan, dtype=np.float32))
    presence_scores: np.ndarray = field(default_factory=lambda: np.full(POSE_LANDMARK_COUNT, np.nan, dtype=np.float32))
    pose_detected: bool = False
    world_available: bool = False
    raw_pose_count: int = 0
    invalid: bool = False
    warnings: list[dict] = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class PoseArrays:
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence_scores: np.ndarray
    pose_detected: np.ndarray
    world_available: np.ndarray
    timestamps_ms: np.ndarray
    frame_indices: np.ndarray
    inference_timestamps_ms: np.ndarray
    schema_version: str = POSE_SCHEMA_VERSION

    @classmethod
    def from_frames(cls, frames: list[PoseFrame]) -> "PoseArrays":
        if not frames:
            raise ValueError("Cannot build pose arrays without processed frames.")
        if any(frame.source_timestamp // 1 != frame.timestamp_ms for frame in frames):
            raise ValueError("Pose milliseconds must align with the exact source timestamps.")
        if any(previous.source_timestamp >= current.source_timestamp for previous, current in pairwise(frames)):
            raise ValueError("Exact source timestamps must be strictly increasing.")
        return cls(
            np.stack([frame.image_landmarks for frame in frames]),
            np.stack([frame.world_landmarks for frame in frames]),
            np.stack([frame.visibility for frame in frames]),
            np.stack([frame.presence_scores for frame in frames]),
            np.array([frame.pose_detected for frame in frames], dtype=np.bool_),
            np.array([frame.world_available for frame in frames], dtype=np.bool_),
            np.array([frame.timestamp_ms for frame in frames], dtype=np.int64),
            np.array([frame.frame_index for frame in frames], dtype=np.int64),
            np.array([frame.inference_timestamp_ms for frame in frames], dtype=np.int64),
        )

    def __post_init__(self):
        count = self.timestamps_ms.size
        points = (count, POSE_LANDMARK_COUNT)
        for name, shape, dtype in (
            ("image_landmarks", (*points, 3), np.float32),
            ("world_landmarks", (*points, 3), np.float32),
            ("visibility", points, np.float32),
            ("presence_scores", points, np.float32),
            ("pose_detected", (count,), np.bool_),
            ("world_available", (count,), np.bool_),
            ("timestamps_ms", (count,), np.int64),
            ("frame_indices", (count,), np.int64),
            ("inference_timestamps_ms", (count,), np.int64),
        ):
            values = getattr(self, name)
            if values.shape != shape or values.dtype != dtype:
                raise ValueError(f"Pose array {name} must have shape {shape} and dtype {np.dtype(dtype)}.")
        if np.any(self.timestamps_ms[1:] < self.timestamps_ms[:-1]):
            raise ValueError("Source timestamps must be nondecreasing.")
        for values in (self.frame_indices, self.inference_timestamps_ms):
            if count and (values[0] < 0 or np.any(values[1:] <= values[:-1])):
                raise ValueError("Source indices and inference timestamps must be nonnegative and strictly increasing.")
        if np.any(self.world_available & ~self.pose_detected):
            raise ValueError("World landmarks require a valid image pose.")
        for values, mask in (
            (self.image_landmarks, self.pose_detected), (self.world_landmarks, self.world_available),
        ):
            if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
                raise ValueError("Pose coordinates must be finite when available and NaN otherwise.")
        for values in (self.visibility, self.presence_scores):
            valid = values[~np.isnan(values)]
            if not np.all((valid >= 0) & (valid <= 1)) or not np.isnan(values[~self.pose_detected]).all():
                raise ValueError("Pose scores must be between 0 and 1 or NaN where unavailable.")


@dataclass(frozen=True, eq=False)
class PoseExtractionResult:
    report: dict
    arrays: PoseArrays | None = None
