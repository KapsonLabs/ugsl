"""Annotation records, workflow states and immutable review history."""

from dataclasses import dataclass
from enum import StrEnum


class ReviewState(StrEnum):
    PENDING = "PENDING"
    ANNOTATED = "ANNOTATED"
    UNDER_REVIEW = "UNDER_REVIEW"
    CORRECTION_REQUIRED = "CORRECTION_REQUIRED"
    SECOND_REVIEW_REQUIRED = "SECOND_REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CORRECTION = "REQUEST_CORRECTION"
    REQUEST_SECOND_REVIEW = "REQUEST_SECOND_REVIEW"


REASONS = (
    "INCORRECT_SIGN", "INCOMPLETE_SIGN", "AMBIGUOUS_GESTURE", "INCORRECT_LABEL", "HANDS_NOT_VISIBLE",
    "FACE_NOT_VISIBLE", "BODY_NOT_VISIBLE", "POOR_VISUAL_QUALITY", "INVALID_TEMPORAL_BOUNDARIES",
    "MULTIPLE_SIGNS", "OTHER",
)
TECHNICAL_STATUSES = ("PASSED", "PASSED_WITH_WARNINGS")
VERDICTS = {Decision.APPROVE, Decision.REJECT}
OPEN_REVIEW_STATES = {ReviewState.ANNOTATED, ReviewState.UNDER_REVIEW, ReviewState.SECOND_REVIEW_REQUIRED}


@dataclass(frozen=True)
class AnnotationInput:
    sign_id: str
    start_ms: int
    end_ms: int
    hand_usage: str = "UNKNOWN"
    dominant_hand: str = "UNKNOWN"
    movement_type: str = "UNKNOWN"
    facial_expression: str = "UNKNOWN"
    completeness: str = "UNKNOWN"
    confidence: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Annotation:
    id: str
    sample_id: str
    sign_id: str
    reviewer_id: str
    start_ms: int
    end_ms: int
    hand_usage: str
    dominant_hand: str
    movement_type: str
    facial_expression: str
    completeness: str
    confidence: float | None
    notes: str | None
    correction_reason: str | None
    version: int
    supersedes_annotation_id: str | None
    created_at: str


@dataclass(frozen=True)
class ReviewDecision:
    id: str
    annotation_id: str
    reviewer_id: str
    review_round: int
    review_policy: str
    decision: str
    reason_code: str | None
    notes: str | None
    is_final: bool
    is_adjudication: bool
    reviewed_at: str


@dataclass(frozen=True)
class ReviewEvent:
    id: str
    sample_id: str
    annotation_id: str | None
    reviewer_id: str
    review_round: int
    event: str
    notes: str | None
    created_at: str
