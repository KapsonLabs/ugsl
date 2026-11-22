"""Counts, footage totals and collection coverage from SQLite."""

from dataclasses import dataclass

from registry.database import Database, RegistryError
from registry.repository import integer

GROUPS = {
    "sign": """SELECT s.id, s.gloss, coalesce(s.variant_label, '') AS variant,
                      count(v.id) AS samples, count(DISTINCT r.signer_id) AS unique_signers
               FROM signs s LEFT JOIN video_samples v ON v.sign_id = s.id
               LEFT JOIN recording_sessions r ON r.id = v.session_id
               GROUP BY s.id ORDER BY s.gloss, s.variant_label, s.id""",
    "signer": """SELECT p.code, count(v.id) AS samples FROM signers p
                 LEFT JOIN recording_sessions r ON r.signer_id = p.id
                 LEFT JOIN video_samples v ON v.session_id = r.id GROUP BY p.id ORDER BY p.code""",
    "session": """SELECT r.id, p.code, r.status, count(v.id) AS samples FROM recording_sessions r
                  JOIN signers p ON p.id = r.signer_id
                  LEFT JOIN video_samples v ON v.session_id = r.id GROUP BY r.id ORDER BY r.started_at, r.id""",
    "status": """WITH states(status) AS (VALUES ('CAPTURED'), ('APPROVED'), ('REJECTED'), ('EXCLUDED'))
                 SELECT states.status, count(v.id) AS samples FROM states
                 LEFT JOIN video_samples v ON v.status = states.status GROUP BY states.status ORDER BY states.status""",
}

ELIGIBLE = """WITH eligible AS (
    SELECT v.id, v.sign_id, r.signer_id FROM video_samples v
    JOIN recording_sessions r ON r.id = v.session_id
    JOIN signers p ON p.id = r.signer_id
    WHERE v.status IN ('CAPTURED', 'APPROVED') AND p.consent_status = 'GRANTED'
)"""


@dataclass(frozen=True)
class Summary:
    total_signs: int
    active_signs: int
    total_signers: int
    total_sessions: int
    total_samples: int
    footage_seconds: float
    storage_bytes: int


@dataclass(frozen=True)
class Coverage:
    signs: list[dict]
    zero_recording_signs: list[dict]
    underrepresented_signs: list[dict]
    underrepresented_signers: list[dict]


def summary(db: Database) -> Summary:
    with db.connect() as connection:
        row = connection.execute(
            """SELECT (SELECT count(*) FROM signs) AS total_signs,
                      (SELECT count(*) FROM signs WHERE is_active) AS active_signs,
                      (SELECT count(*) FROM signers) AS total_signers,
                      (SELECT count(*) FROM recording_sessions) AS total_sessions,
                      count(*) AS total_samples, coalesce(sum(coalesce(duration_seconds, actual_duration_seconds)), 0.0) AS footage_seconds,
                      coalesce(sum(file_size_bytes), 0) AS storage_bytes FROM video_samples""",
        ).fetchone()
    return Summary(**dict(row))


def grouped(db: Database, by: str) -> list[dict]:
    if by not in GROUPS:
        raise RegistryError("Group statistics by sign, signer, session or status.")
    with db.connect() as connection:
        return [dict(row) for row in connection.execute(GROUPS[by])]


def coverage(db: Database, *, min_signer_samples: int = 1) -> Coverage:
    minimum = integer(min_signer_samples, "Minimum signer samples", 0)
    with db.connect() as connection:
        signs = [dict(row) for row in connection.execute(
            ELIGIBLE + """
            SELECT s.id, s.gloss, coalesce(s.variant_label, '') AS variant, s.is_active,
                   s.target_samples, coalesce(t.total, 0) AS registered, count(e.id) AS collected,
                   count(DISTINCT e.signer_id) AS unique_signers
            FROM signs s LEFT JOIN eligible e ON e.sign_id = s.id
            LEFT JOIN (SELECT sign_id, count(*) AS total FROM video_samples GROUP BY sign_id) t ON t.sign_id = s.id
            GROUP BY s.id ORDER BY s.gloss, s.variant_label, s.id""",
        )]
        underrepresented_signers = [dict(row) for row in connection.execute(
            ELIGIBLE + """
            SELECT p.code, count(e.id) AS collected FROM signers p
            LEFT JOIN eligible e ON e.signer_id = p.id
            WHERE p.is_active AND p.consent_status = 'GRANTED'
            GROUP BY p.id HAVING count(e.id) < ? ORDER BY collected, p.code""",
            (minimum,),
        )]
    return Coverage(
        signs=signs,
        zero_recording_signs=[row for row in signs if row["registered"] == 0],
        underrepresented_signs=[row for row in signs if row["is_active"] and row["collected"] < row["target_samples"]],
        underrepresented_signers=underrepresented_signers,
    )


@dataclass(frozen=True)
class AnnotationStatistics:
    counts: dict
    rejection_reasons: list[dict]
    approved_by_sign: list[dict]
    approved_by_signer: list[dict]
    reviewer_activity: list[dict]
    agreement: dict


def annotation_statistics(db: Database, dataset_root, config=None) -> AnnotationStatistics:
    from collections import Counter

    from annotation.eligibility import TrainingEligibilityService
    from annotation.models import ReviewState
    from annotation.review_queue import ReviewQueue
    from registry.models import VideoSample
    from registry.repository import record

    eligible = TrainingEligibilityService(db, dataset_root, config)
    available = Counter(row.review_state for row in ReviewQueue(db, dataset_root).list())
    with db.connect() as connection:
        records = [record(VideoSample, row) for row in connection.execute("SELECT * FROM video_samples")]
        sign_counts = Counter(sample.sign_id for sample in records)
        finalized = [dict(row) for row in connection.execute(
            """SELECT v.id, v.sign_id AS assigned_sign_id, a.sign_id AS verified_sign_id,
                      p.code AS signer_code, d.decision, d.reason_code
               FROM video_samples v JOIN annotations a ON a.id = v.current_annotation_id
               JOIN recording_sessions r ON r.id = v.session_id JOIN signers p ON p.id = r.signer_id
               JOIN review_decisions d ON d.annotation_id = a.id AND d.review_round = v.review_round AND d.is_final
               WHERE (v.status = 'APPROVED' AND v.review_state = 'APPROVED' AND d.decision = 'APPROVE')
                  OR (v.status = 'REJECTED' AND v.review_state = 'REJECTED' AND d.decision = 'REJECT')""",
        )]
        voters = [dict(row) for row in connection.execute(
            """SELECT u.code, u.role, coalesce(a.versions, 0) AS annotation_versions,
                      coalesce(d.decisions, 0) AS review_decisions, coalesce(d.adjudications, 0) AS adjudications
               FROM reviewers u LEFT JOIN (SELECT reviewer_id, count(*) AS versions FROM annotations GROUP BY reviewer_id) a
                 ON a.reviewer_id = u.id
               LEFT JOIN (SELECT reviewer_id, count(*) AS decisions, sum(is_adjudication) AS adjudications
                 FROM review_decisions GROUP BY reviewer_id) d ON d.reviewer_id = u.id ORDER BY u.code""",
        )]
        agreement = dict(connection.execute(
            """WITH independent AS (
                 SELECT d.annotation_id, d.review_round, d.decision,
                        row_number() OVER (PARTITION BY d.annotation_id, d.review_round ORDER BY d.reviewed_at, d.id) AS n
                 FROM review_decisions d JOIN annotations a ON a.id = d.annotation_id
                 WHERE d.decision IN ('APPROVE', 'REJECT') AND NOT d.is_adjudication AND d.reviewer_id <> a.reviewer_id
               )
               SELECT count(*) AS compared_rounds, coalesce(sum(f.decision = s.decision), 0) AS agreed_rounds
               FROM independent f JOIN independent s USING (annotation_id, review_round)
               WHERE f.n = 1 AND s.n = 2""",
        ).fetchone())
        corrected = connection.execute(
            """SELECT count(*) FROM annotations a JOIN video_samples v ON v.id = a.sample_id
               LEFT JOIN annotations prev ON prev.id = a.supersedes_annotation_id
               WHERE a.sign_id <> coalesce(prev.sign_id, v.sign_id)""",
        ).fetchone()[0]

        approved = [row for row in finalized if row["decision"] == "APPROVE"]
        rejected = [row for row in finalized if row["decision"] == "REJECT"]
        sample_ids = {row["id"] for row in approved}
        eligible_ids = {sample.id for sample in records if sample.id in sample_ids and eligible.evaluate(connection, sample).eligible}
        verified = Counter(row["verified_sign_id"] for row in approved)
        assigned = Counter(row["assigned_sign_id"] for row in approved)
        eligible_signs = Counter(row["verified_sign_id"] for row in approved if row["id"] in eligible_ids)
        eligible_signers = Counter(row["signer_code"] for row in approved if row["id"] in eligible_ids)
        by_sign = [
            {**dict(row), "assigned_samples": sign_counts[row["sign_id"]], "approved_by_original_label": assigned[row["sign_id"]],
             "verified_approved": verified[row["sign_id"]], "training_eligible": eligible_signs[row["sign_id"]]}
            for row in connection.execute(
                "SELECT id AS sign_id, gloss, variant_label, target_samples FROM signs ORDER BY id"
            )
        ]
        approved_signers = Counter(row["signer_code"] for row in approved)
        by_signer = [
            {"signer": code, "approved_samples": approved_signers[code], "training_eligible": eligible_signers[code]}
            for (code,) in connection.execute("SELECT code FROM signers ORDER BY code")
        ]

    annotated = sum(sample.current_annotation_id is not None for sample in records)

    def percentage(numerator, denominator):
        return round(100 * numerator / denominator, 2) if denominator else None

    counts = {
        "awaiting_annotation": available[ReviewState.PENDING],
        "annotated_samples": annotated,
        "awaiting_review": sum(available[state] for state in (
            ReviewState.ANNOTATED, ReviewState.UNDER_REVIEW, ReviewState.SECOND_REVIEW_REQUIRED,
        )),
        "approved_samples": len(approved),
        "rejected_samples": len(rejected),
        "correction_required": available[ReviewState.CORRECTION_REQUIRED],
        "second_review_required": available[ReviewState.SECOND_REVIEW_REQUIRED],
        "review_completion_percent": percentage(len(finalized), annotated),
        "approval_rate_percent": percentage(len(approved), len(finalized)),
        "rejection_rate_percent": percentage(len(rejected), len(finalized)),
        "label_corrections": corrected,
        "training_eligible_samples": len(eligible_ids),
    }
    agreement["agreement_percent"] = percentage(agreement["agreed_rounds"], agreement["compared_rounds"])
    reasons = Counter(row["reason_code"] for row in rejected)
    return AnnotationStatistics(
        counts, [{"reason": code, "samples": total} for code, total in sorted(reasons.items())],
        by_sign, by_signer, voters, agreement,
    )
