CREATE TABLE reviewers (
    id TEXT PRIMARY KEY NOT NULL,
    code TEXT NOT NULL UNIQUE,
    display_name TEXT,
    proficiency TEXT NOT NULL CHECK (proficiency IN ('FLUENT', 'NATIVE', 'QUALIFIED_INTERPRETER', 'OTHER')),
    role TEXT NOT NULL CHECK (role IN ('ANNOTATOR', 'REVIEWER', 'ADJUDICATOR')),
    qualification_notes TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE annotations (
    id TEXT PRIMARY KEY NOT NULL,
    sample_id TEXT NOT NULL REFERENCES video_samples(id),
    sign_id TEXT NOT NULL REFERENCES signs(id),
    reviewer_id TEXT NOT NULL REFERENCES reviewers(id),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    hand_usage TEXT NOT NULL CHECK (hand_usage IN ('ONE_HANDED', 'TWO_HANDED', 'UNKNOWN')),
    dominant_hand TEXT NOT NULL CHECK (dominant_hand IN ('LEFT', 'RIGHT', 'BOTH', 'UNKNOWN')),
    movement_type TEXT NOT NULL CHECK (movement_type IN ('STATIC', 'DYNAMIC', 'UNKNOWN')),
    facial_expression TEXT NOT NULL CHECK (facial_expression IN ('PRESENT', 'ABSENT', 'UNKNOWN')),
    completeness TEXT NOT NULL CHECK (completeness IN ('COMPLETE', 'INCOMPLETE', 'UNKNOWN')),
    confidence REAL,
    notes TEXT,
    correction_reason TEXT,
    version INTEGER NOT NULL CHECK (version >= 1),
    supersedes_annotation_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (sample_id, version),
    UNIQUE (sample_id, id),
    FOREIGN KEY (sample_id, supersedes_annotation_id) REFERENCES annotations(sample_id, id)
);
CREATE INDEX annotations_reviewer ON annotations (reviewer_id);

ALTER TABLE video_samples ADD COLUMN review_state TEXT NOT NULL DEFAULT 'PENDING'
    CHECK (review_state IN ('PENDING', 'ANNOTATED', 'UNDER_REVIEW', 'CORRECTION_REQUIRED',
                           'SECOND_REVIEW_REQUIRED', 'APPROVED', 'REJECTED'));
ALTER TABLE video_samples ADD COLUMN current_annotation_id TEXT REFERENCES annotations(id);
ALTER TABLE video_samples ADD COLUMN review_round INTEGER NOT NULL DEFAULT 1 CHECK (review_round >= 1);
CREATE INDEX samples_review_state ON video_samples (review_state, technical_validation_status);

CREATE TABLE review_decisions (
    id TEXT PRIMARY KEY NOT NULL,
    annotation_id TEXT NOT NULL REFERENCES annotations(id),
    reviewer_id TEXT NOT NULL REFERENCES reviewers(id),
    review_round INTEGER NOT NULL CHECK (review_round >= 1),
    review_policy TEXT NOT NULL CHECK (review_policy IN ('SINGLE', 'DUAL')),
    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REJECT', 'REQUEST_CORRECTION', 'REQUEST_SECOND_REVIEW')),
    reason_code TEXT,
    notes TEXT,
    is_final INTEGER NOT NULL CHECK (is_final IN (0, 1)),
    is_adjudication INTEGER NOT NULL CHECK (is_adjudication IN (0, 1)),
    reviewed_at TEXT NOT NULL,
    CHECK (decision NOT IN ('REJECT', 'REQUEST_CORRECTION') OR reason_code IS NOT NULL),
    CHECK (is_final = 0 OR decision IN ('APPROVE', 'REJECT')),
    UNIQUE (annotation_id, review_round, reviewer_id)
);
CREATE UNIQUE INDEX review_final_decision ON review_decisions (annotation_id, review_round) WHERE is_final = 1;
CREATE INDEX review_decisions_reviewer ON review_decisions (reviewer_id);

CREATE TABLE review_events (
    id TEXT PRIMARY KEY NOT NULL,
    sample_id TEXT NOT NULL REFERENCES video_samples(id),
    annotation_id TEXT,
    reviewer_id TEXT NOT NULL REFERENCES reviewers(id),
    review_round INTEGER NOT NULL CHECK (review_round >= 1),
    event TEXT NOT NULL CHECK (event IN ('REVIEW_STARTED', 'REOPENED', 'VARIANT_REQUESTED')),
    notes TEXT,
    created_at TEXT NOT NULL,
    CHECK (annotation_id IS NOT NULL OR event = 'VARIANT_REQUESTED'),
    FOREIGN KEY (sample_id, annotation_id) REFERENCES annotations(sample_id, id)
);
CREATE INDEX review_events_sample ON review_events (sample_id, created_at);

CREATE TRIGGER samples_check_current_annotation
BEFORE UPDATE OF current_annotation_id ON video_samples
WHEN NEW.current_annotation_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM annotations WHERE id = NEW.current_annotation_id AND sample_id = NEW.id
)
BEGIN
    SELECT RAISE(ABORT, 'Current annotation belongs to a different sample');
END;

CREATE TRIGGER annotations_no_update BEFORE UPDATE ON annotations
BEGIN
    SELECT RAISE(ABORT, 'Annotations are immutable');
END;
CREATE TRIGGER annotations_no_delete BEFORE DELETE ON annotations
BEGIN
    SELECT RAISE(ABORT, 'Annotation history is retained');
END;
CREATE TRIGGER review_decisions_no_update BEFORE UPDATE ON review_decisions
BEGIN
    SELECT RAISE(ABORT, 'Review decisions are append-only');
END;
CREATE TRIGGER review_decisions_no_delete BEFORE DELETE ON review_decisions
BEGIN
    SELECT RAISE(ABORT, 'Review decision history is retained');
END;
