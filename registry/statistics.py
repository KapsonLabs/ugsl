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
