"""Authoritative eligibility for future exports, without exporting data."""

from dataclasses import dataclass
from pathlib import Path

from annotation import repository
from annotation.models import Decision, ReviewState
from annotation.validator import AnnotationValidator
from config import ReviewConfig
from registry import vocabulary
from registry.database import Database, RegistryError


@dataclass(frozen=True)
class TrainingEligibility:
    sample_id: str
    eligible: bool
    verified_sign_id: str | None = None
    annotation_id: str | None = None
    reason: str | None = None


class TrainingEligibilityService:
    def __init__(self, db: Database, root: Path, config: ReviewConfig | None = None):
        self.db = db
        self.config = config or ReviewConfig()
        self.validator = AnnotationValidator(Path(root), self.config)

    def check(self, sample_id: str) -> TrainingEligibility:
        with self.db.connect() as connection:
            return self.evaluate(connection, repository.sample(connection, sample_id))

    def evaluate(self, connection, sample, *, verify_media: bool = True) -> TrainingEligibility:
        try:
            if sample.status != "APPROVED" or sample.review_state != ReviewState.APPROVED:
                raise RegistryError("The sample has no current linguistic approval.")
            annotation = repository.require_current(connection, sample)
            votes = repository.decisions(connection, annotation.id, sample.review_round)
            final = next((vote for vote in votes if vote.is_final and vote.decision == Decision.APPROVE), None)
            if final is None:
                raise RegistryError("No final approval is attached to the current annotation and review round.")
            dual = self.config.policy == "DUAL" or any(vote.review_policy == "DUAL" for vote in votes)
            if dual and final.reviewer_id == annotation.reviewer_id:
                raise RegistryError("The dual-review policy requires independent approval.")
            if repository.conflicting(votes) and not final.is_adjudication:
                raise RegistryError("Conflicting decisions require a final adjudication.")
            self.validator.media(connection, sample, verify=verify_media)
            sign = vocabulary.find(connection, annotation.sign_id)
            if not sign.is_active:
                raise RegistryError("The verified sign or variant is inactive.")
            return TrainingEligibility(sample.id, True, sign.id, annotation.id)
        except RegistryError as exc:
            return TrainingEligibility(sample.id, False, reason=str(exc))
