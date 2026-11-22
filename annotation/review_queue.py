"""Filter registered clips that are available for annotation or review."""

from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

from annotation.models import TECHNICAL_STATUSES, ReviewState
from annotation.validator import AnnotationValidator
from config import ReviewConfig
from ingestion.schemas import ValidationStatus
from registry import reviewers, sessions, signers, vocabulary
from registry.database import Database, RegistryError
from registry.models import VideoSample
from registry.repository import choice

ORDER = {
    "oldest": "v.registered_at ASC, v.id ASC",
    "newest": "v.registered_at DESC, v.id DESC",
    "sign": "v.sign_id ASC, v.registered_at ASC, v.id ASC",
}
PRIORITY = """CASE v.review_state WHEN 'SECOND_REVIEW_REQUIRED' THEN 0
              WHEN 'CORRECTION_REQUIRED' THEN 1 ELSE 2 END, """
SAMPLE_COLUMNS = tuple(field.name for field in fields(VideoSample))


@dataclass(frozen=True)
class QueueEntry:
    sample_id: str
    sign_id: str
    gloss: str
    signer_code: str
    session_id: str
    source: str
    technical_validation_status: str
    review_state: str
    reviewer_code: str | None
    annotated_sign_id: str | None
    version: int | None
    registered_at: str
    needs_adjudication: bool


def _date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise RegistryError("Queue dates must use YYYY-MM-DD.") from exc


class ReviewQueue:
    def __init__(self, db: Database, root: Path):
        self.db = db
        self.validator = AnnotationValidator(Path(root), ReviewConfig())

    def list(
        self, *, sign_id=None, signer=None, session_id=None, technical_status=None, state=None, reviewer=None,
        source=None, since=None, until=None, order="oldest", prioritize=False,
    ) -> list[QueueEntry]:
        order = order.strip().lower()
        if order not in ORDER:
            raise RegistryError("Queue order must be oldest, newest or sign.")
        if technical_status is not None:
            technical_status = reviewers.workflow_choice(technical_status.strip(), "Technical status", tuple(ValidationStatus))
        if state is not None:
            state = reviewers.workflow_choice(state.strip(), "Review state", tuple(ReviewState))
        if source is not None:
            source = choice(source, "Recording source", ("CAMERA", "IMPORT", "LOCAL", "REMOTE"))
        since, until = _date(since), _date(until)
        if since and until and since > until:
            raise RegistryError("Queue start date must be on or before its end date.")
        ordering = (PRIORITY if prioritize else "") + ORDER[order]

        with self.db.connect() as connection:
            sign_id = vocabulary.find(connection, sign_id).id if sign_id is not None else None
            session_id = sessions.find(connection, session_id).id if session_id is not None else None
            if signer is not None:
                row = connection.execute("SELECT id FROM signers WHERE id = ?", (signer.strip().lower(),)).fetchone()
                signer = row["id"] if row else signers.find(connection, signer).id
            reviewer_id = reviewers.find(connection, reviewer).id if reviewer is not None else None
            rows = connection.execute(
                """SELECT v.*, s.gloss, p.code AS signer_code,
                       CASE v.source_type WHEN 'CAMERA' THEN r.camera_source_type ELSE v.source_type END AS source,
                       a.sign_id AS annotated_sign_id, a.version AS annotation_version, u.code AS reviewer_code,
                       (v.review_state = 'SECOND_REVIEW_REQUIRED' AND 2 = (
                           SELECT count(DISTINCT d.decision) FROM review_decisions d
                           WHERE d.annotation_id = a.id AND d.review_round = v.review_round
                           AND d.decision IN ('APPROVE', 'REJECT') AND NOT d.is_adjudication
                       )) AS needs_adjudication
                   FROM video_samples v
                   JOIN signs s ON s.id = v.sign_id
                   JOIN recording_sessions r ON r.id = v.session_id
                   JOIN signers p ON p.id = r.signer_id
                   LEFT JOIN annotations a ON a.id = v.current_annotation_id
                   LEFT JOIN reviewers u ON u.id = a.reviewer_id
                   WHERE v.status <> 'EXCLUDED' AND v.sample_type = 'ISOLATED'
                   AND p.consent_status = 'GRANTED' AND v.technical_validation_status IN (:pass, :warn)
                   AND (:technical IS NULL OR v.technical_validation_status = :technical)
                   AND (:sign IS NULL OR v.sign_id = :sign OR a.sign_id = :sign)
                   AND (:signer IS NULL OR r.signer_id = :signer)
                   AND (:session IS NULL OR r.id = :session)
                   AND ((:state IS NULL AND v.review_state NOT IN ('APPROVED', 'REJECTED')) OR v.review_state = :state)
                   AND (:reviewer IS NULL OR a.reviewer_id = :reviewer OR EXISTS (
                       SELECT 1 FROM review_decisions d WHERE d.annotation_id = a.id
                       AND d.review_round = v.review_round AND d.reviewer_id = :reviewer
                   ))
                   AND (:source IS NULL OR v.source_type = :source OR
                        (v.source_type = 'CAMERA' AND r.camera_source_type = :source))
                   AND (:since IS NULL OR substr(v.registered_at, 1, 10) >= :since)
                   AND (:until IS NULL OR substr(v.registered_at, 1, 10) <= :until)
                   ORDER BY """ + ordering,
                {
                    "pass": TECHNICAL_STATUSES[0], "warn": TECHNICAL_STATUSES[1],
                    "technical": technical_status, "sign": sign_id, "signer": signer, "session": session_id,
                    "state": state, "reviewer": reviewer_id, "source": source, "since": since, "until": until,
                },
            )
            entries = []
            for row in rows:
                sample = VideoSample(**{key: row[key] for key in SAMPLE_COLUMNS})
                try:
                    self.validator.media(connection, sample)
                except RegistryError:
                    continue
                entries.append(QueueEntry(
                    sample.id, sample.sign_id, row["gloss"], row["signer_code"], sample.session_id, row["source"],
                    sample.technical_validation_status, sample.review_state, row["reviewer_code"], row["annotated_sign_id"],
                    row["annotation_version"], sample.registered_at, bool(row["needs_adjudication"]),
                ))
            return entries
