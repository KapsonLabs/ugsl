"""Queries shared by annotation, review and eligibility."""

from dataclasses import asdict

from annotation.models import Annotation, ReviewDecision, ReviewEvent, VERDICTS
from registry.database import RegistryError
from registry.models import VideoSample
from registry.repository import get, record


def sample(connection, sample_id: str) -> VideoSample:
    return get(connection, VideoSample, "SELECT * FROM video_samples WHERE id = ?", (sample_id.strip().lower(),),
               "Unknown video sample.")


def current(connection, sample: VideoSample) -> Annotation | None:
    if sample.current_annotation_id is None:
        return None
    return get(connection, Annotation, "SELECT * FROM annotations WHERE sample_id = ? AND id = ?",
               (sample.id, sample.current_annotation_id), "The current annotation is missing from the registry.")


def insert_annotation(connection, annotation: Annotation) -> None:
    connection.execute(
        """INSERT INTO annotations (id, sample_id, sign_id, reviewer_id, start_ms, end_ms, hand_usage,
           dominant_hand, movement_type, facial_expression, completeness, confidence, notes, correction_reason,
           version, supersedes_annotation_id, created_at)
           VALUES (:id, :sample_id, :sign_id, :reviewer_id, :start_ms, :end_ms, :hand_usage, :dominant_hand,
           :movement_type, :facial_expression, :completeness, :confidence, :notes, :correction_reason,
           :version, :supersedes_annotation_id, :created_at)""",
        asdict(annotation),
    )


def _decision(row) -> ReviewDecision:
    return ReviewDecision(**{**dict(row), "is_final": bool(row["is_final"]), "is_adjudication": bool(row["is_adjudication"])})


def decisions(connection, annotation_id: str, review_round: int) -> list[ReviewDecision]:
    return [_decision(row) for row in connection.execute(
        "SELECT * FROM review_decisions WHERE annotation_id = ? AND review_round = ? ORDER BY reviewed_at, id",
        (annotation_id, review_round),
    )]


def conflicting(decisions: list[ReviewDecision]) -> bool:
    return len({row.decision for row in decisions if row.decision in VERDICTS and not row.is_adjudication}) == 2


def insert_decision(connection, decision: ReviewDecision) -> None:
    connection.execute(
        """INSERT INTO review_decisions (id, annotation_id, reviewer_id, review_round, review_policy, decision,
           reason_code, notes, is_final, is_adjudication, reviewed_at)
           VALUES (:id, :annotation_id, :reviewer_id, :review_round, :review_policy, :decision, :reason_code,
           :notes, :is_final, :is_adjudication, :reviewed_at)""",
        asdict(decision),
    )


def insert_event(connection, event: ReviewEvent) -> None:
    connection.execute(
        """INSERT INTO review_events (id, sample_id, annotation_id, reviewer_id, review_round, event, notes, created_at)
           VALUES (:id, :sample_id, :annotation_id, :reviewer_id, :review_round, :event, :notes, :created_at)""",
        asdict(event),
    )


def annotation_history(connection, sample_id: str) -> list[Annotation]:
    return [record(Annotation, row) for row in connection.execute(
        "SELECT * FROM annotations WHERE sample_id = ? ORDER BY version", (sample_id,),
    )]


def review_history(connection, sample_id: str) -> tuple[list[ReviewDecision], list[ReviewEvent]]:
    votes = [_decision(row) for row in connection.execute(
        """SELECT d.* FROM review_decisions d JOIN annotations a ON a.id = d.annotation_id
           WHERE a.sample_id = ? ORDER BY a.version, d.review_round, d.reviewed_at, d.id""",
        (sample_id,),
    )]
    events = [record(ReviewEvent, row) for row in connection.execute(
        """SELECT * FROM review_events WHERE sample_id = ? ORDER BY created_at, id""",
        (sample_id,),
    )]
    return votes, events


def require_current(connection, sample: VideoSample, expected_annotation_id=None, expected_round=None) -> Annotation:
    annotation = current(connection, sample)
    if annotation is None:
        raise RegistryError("Save an annotation before starting linguistic review.")
    if (
        (expected_annotation_id is not None and annotation.id != expected_annotation_id)
        or (expected_round is not None and sample.review_round != expected_round)
    ):
        raise RegistryError("This annotation or review round is outdated. Reload the sample before submitting.")
    return annotation
