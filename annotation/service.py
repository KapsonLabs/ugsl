"""Transactional annotation versions and linguistic review transitions."""

import json
from dataclasses import dataclass, fields
from pathlib import Path
from uuid import uuid4

from annotation import repository
from annotation.models import (
    OPEN_REVIEW_STATES, REASONS, VERDICTS, Annotation, AnnotationInput, Decision, ReviewDecision, ReviewEvent, ReviewState,
)
from annotation.validator import AnnotationValidator
from config import ReviewConfig
from registry import reviewers, vocabulary
from registry.database import Database, RegistryError
from registry.models import Sign, VideoSample
from registry.repository import optional_text, utc_now


@dataclass(frozen=True)
class ReviewView:
    sample: VideoSample
    sign: Sign
    annotated_sign: Sign | None
    signer_code: str
    path: Path
    duration_ms: int
    current: Annotation | None
    warnings: tuple[str, ...]
    needs_adjudication: bool


@dataclass(frozen=True)
class ReviewResult:
    decision: ReviewDecision
    state: ReviewState
    needs_adjudication: bool


def _id() -> str:
    return str(uuid4())


def _reason(text: str | None, action: str) -> str:
    value = optional_text(text)
    if value is None:
        raise RegistryError(f"{action} requires a reason.")
    return value


def _update_state(connection, sample: VideoSample, state: ReviewState, *, annotation_id=None, review_round=None) -> None:
    # Every linguistic status update goes through this transaction helper.
    status = {
        ReviewState.APPROVED: "APPROVED",
        ReviewState.REJECTED: "REJECTED",
    }.get(state, "CAPTURED")
    cursor = connection.execute(
        """UPDATE video_samples SET review_state = ?, status = ?, current_annotation_id = ?, review_round = ?
           WHERE id = ? AND status <> 'EXCLUDED'""",
        (state, status, annotation_id or sample.current_annotation_id, review_round or sample.review_round, sample.id),
    )
    if cursor.rowcount != 1:
        raise RegistryError("This sample is excluded from linguistic review.")


def _event(connection, sample, actor, name, notes=None):
    event = ReviewEvent(
        _id(), sample.id, sample.current_annotation_id, actor.id, sample.review_round, name, notes, utc_now(),
    )
    repository.insert_event(connection, event)
    return event


class AnnotationService:
    def __init__(self, db: Database, root: Path, config: ReviewConfig | None = None):
        self.db = db
        self.config = config or ReviewConfig()
        self.validator = AnnotationValidator(Path(root), self.config)

    def view(self, sample_id: str, reviewer_code: str) -> ReviewView:
        with self.db.connect() as connection:
            reviewers.require_active(connection, reviewer_code)
            sample = repository.sample(connection, sample_id)
            path, signer = self.validator.media(connection, sample)
            annotation = repository.current(connection, sample)
            sign = vocabulary.find(connection, sample.sign_id)
            annotated_sign = vocabulary.find(connection, annotation.sign_id) if annotation else None
            votes = repository.decisions(connection, annotation.id, sample.review_round) if annotation else []
            report = connection.execute(
                "SELECT warnings_json FROM video_validations WHERE sample_id = ? ORDER BY validated_at DESC, id DESC LIMIT 1",
                (sample.id,),
            ).fetchone()
        warnings = tuple(json.loads(report["warnings_json"])) if report else ()
        return ReviewView(
            sample, sign, annotated_sign, signer.code, path, self.validator.duration_ms(sample, path), annotation,
            warnings, sample.review_state == ReviewState.SECOND_REVIEW_REQUIRED and repository.conflicting(votes),
        )

    def history(self, sample_id: str):
        with self.db.connect() as connection:
            sample = repository.sample(connection, sample_id)
            return repository.annotation_history(connection, sample.id)

    def review_history(self, sample_id: str):
        with self.db.connect() as connection:
            sample = repository.sample(connection, sample_id)
            return repository.review_history(connection, sample.id)

    def create(self, sample_id: str, reviewer_code: str, inputs: AnnotationInput, *, reason=None) -> Annotation:
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code)
            sample = repository.sample(connection, sample_id)
            if repository.current(connection, sample) is not None:
                raise RegistryError("This sample already has an annotation. Use annotation correct to create a new version.")
            selected = vocabulary.find(connection, inputs.sign_id)
            reason = _reason(reason, "Changing the assigned sign") if selected.id != sample.sign_id else optional_text(reason)
            return self._save(connection, sample, actor, inputs, reason=reason)

    def correct(
        self, sample_id: str, reviewer_code: str, *, reason: str, expected_annotation_id=None,
        expected_round=None, **changes,
    ) -> Annotation:
        reason = _reason(reason, "A correction")
        input_fields = {field.name for field in fields(AnnotationInput)}
        if changes.keys() - input_fields:
            raise RegistryError("Unsupported annotation fields.")
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code)
            sample = repository.sample(connection, sample_id)
            previous = repository.require_current(connection, sample, expected_annotation_id, expected_round)
            if sample.review_state == ReviewState.SECOND_REVIEW_REQUIRED and repository.conflicting(
                repository.decisions(connection, previous.id, sample.review_round)
            ):
                raise RegistryError("An independent adjudicator must resolve the conflicting decisions or request a correction.")
            inputs = AnnotationInput(**{name: changes.get(name, getattr(previous, name)) for name in input_fields})
            return self._save(connection, sample, actor, inputs, previous, reason)

    def _save(self, connection, sample, actor, inputs, previous=None, reason=None) -> Annotation:
        path, _ = self.validator.media(connection, sample, verify=True)
        inputs = self.validator.annotation(connection, inputs, self.validator.duration_ms(sample, path))
        if previous and all(getattr(inputs, field.name) == getattr(previous, field.name) for field in fields(AnnotationInput)):
            raise RegistryError("The correction must change an annotation field.")
        annotation = Annotation(
            _id(), sample.id, inputs.sign_id, actor.id, inputs.start_ms, inputs.end_ms,
            inputs.hand_usage, inputs.dominant_hand, inputs.movement_type, inputs.facial_expression, inputs.completeness,
            inputs.confidence, inputs.notes, reason, previous.version + 1 if previous else 1,
            previous.id if previous else None, utc_now(),
        )
        repository.insert_annotation(connection, annotation)
        _update_state(connection, sample, ReviewState.ANNOTATED, annotation_id=annotation.id, review_round=1)
        return annotation

    def start_review(self, sample_id: str, reviewer_code: str) -> None:
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code, decision=True)
            sample = repository.sample(connection, sample_id)
            repository.require_current(connection, sample)
            if sample.review_state != ReviewState.ANNOTATED:
                raise RegistryError("Only an annotated sample can enter UNDER_REVIEW.")
            self.validator.media(connection, sample, verify=True)
            _event(connection, sample, actor, "REVIEW_STARTED")
            _update_state(connection, sample, ReviewState.UNDER_REVIEW)

    def decide(
        self, sample_id: str, reviewer_code: str, action: Decision | str, *,
        reason_code: str | None = None, notes: str | None = None,
        expected_annotation_id: str | None = None, expected_round: int | None = None,
    ) -> ReviewResult:
        action = Decision(reviewers.workflow_choice(action.strip(), "Decision", tuple(Decision)))
        if action in {Decision.REJECT, Decision.REQUEST_CORRECTION}:
            reason_code = _reason(reason_code, "This decision")
        if reason_code:
            reason_code = reviewers.workflow_choice(reason_code.strip(), "Reason code", REASONS)
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code, decision=True)
            sample = repository.sample(connection, sample_id)
            annotation = repository.require_current(connection, sample, expected_annotation_id, expected_round)
            votes = repository.decisions(connection, annotation.id, sample.review_round)
            if any(vote.reviewer_id == actor.id for vote in votes):
                raise RegistryError("This reviewer already submitted a decision for this annotation and review round.")
            if sample.review_state not in OPEN_REVIEW_STATES:
                raise RegistryError(f"Review decisions are unavailable in state {sample.review_state}.")
            path, _ = self.validator.media(connection, sample, verify=True)
            if action == Decision.APPROVE:
                inputs = AnnotationInput(**{field.name: getattr(annotation, field.name) for field in fields(AnnotationInput)})
                self.validator.annotation(connection, inputs, self.validator.duration_ms(sample, path))
            policy = "DUAL" if self.config.policy == "DUAL" or any(vote.review_policy == "DUAL" for vote in votes) else "SINGLE"
            state, adjudication = self._next_state(sample, annotation, actor, action, policy, votes)
            decision = ReviewDecision(
                _id(), annotation.id, actor.id, sample.review_round, policy, action, reason_code, optional_text(notes),
                state in {ReviewState.APPROVED, ReviewState.REJECTED}, adjudication, utc_now(),
            )
            repository.insert_decision(connection, decision)
            _update_state(connection, sample, state)
            needs_adjudication = state == ReviewState.SECOND_REVIEW_REQUIRED and repository.conflicting([*votes, decision])
            return ReviewResult(decision, state, needs_adjudication)

    @staticmethod
    def _next_state(sample, annotation, actor, action, policy, votes) -> tuple[ReviewState, bool]:
        second = sample.review_state == ReviewState.SECOND_REVIEW_REQUIRED
        adjudication = second and repository.conflicting(votes)
        if adjudication and (actor.role != "ADJUDICATOR" or actor.id == annotation.reviewer_id):
            raise RegistryError("Conflicting decisions require an independent adjudicator.")
        if action == Decision.REQUEST_CORRECTION:
            return ReviewState.CORRECTION_REQUIRED, adjudication
        if action == Decision.REQUEST_SECOND_REVIEW:
            if second:
                raise RegistryError("This sample is already awaiting second review or adjudication.")
            return ReviewState.SECOND_REVIEW_REQUIRED, False

        if actor.id == annotation.reviewer_id and (policy == "DUAL" or second):
            raise RegistryError("Independent review requires a reviewer other than the original annotator.")
        final_state = ReviewState.APPROVED if action == Decision.APPROVE else ReviewState.REJECTED
        if adjudication:
            return final_state, True
        earlier = [vote for vote in votes if vote.decision in VERDICTS]
        if second and earlier and earlier[0].decision != action:
            return ReviewState.SECOND_REVIEW_REQUIRED, False
        if not second and policy == "DUAL" and action == Decision.REJECT:
            return ReviewState.SECOND_REVIEW_REQUIRED, False
        return final_state, False

    def reopen(self, sample_id: str, reviewer_code: str, *, reason: str, expected_annotation_id=None,
               expected_round=None) -> ReviewEvent:
        reason = _reason(reason, "Reopening a sample")
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code, decision=True)
            sample = repository.sample(connection, sample_id)
            repository.require_current(connection, sample, expected_annotation_id, expected_round)
            if sample.review_state != ReviewState.APPROVED or sample.status != "APPROVED":
                raise RegistryError("Only a currently approved sample can be reopened.")
            event = _event(connection, sample, actor, "REOPENED", reason)
            _update_state(connection, sample, ReviewState.ANNOTATED, review_round=sample.review_round + 1)
            return event

    def request_variant(self, sample_id: str, reviewer_code: str, *, notes: str) -> ReviewEvent:
        notes = _reason(notes, "A variant request")
        with self.db.connect(write=True) as connection:
            actor = reviewers.require_active(connection, reviewer_code)
            sample = repository.sample(connection, sample_id)
            self.validator.media(connection, sample)
            return _event(connection, sample, actor, "VARIANT_REQUESTED", notes)
