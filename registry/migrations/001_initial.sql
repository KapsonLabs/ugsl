CREATE TABLE sign_sequence (last_id INTEGER NOT NULL);
INSERT INTO sign_sequence VALUES (0);

CREATE TABLE signs (
    id TEXT PRIMARY KEY NOT NULL,
    gloss TEXT NOT NULL,
    description TEXT,
    category TEXT COLLATE NOCASE,
    sign_type TEXT NOT NULL DEFAULT 'UNKNOWN' CHECK (sign_type IN ('STATIC', 'DYNAMIC', 'UNKNOWN')),
    variant_label TEXT,
    target_samples INTEGER NOT NULL CHECK (target_samples >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX signs_gloss_variant ON signs (gloss, coalesce(variant_label, ''));

CREATE TABLE signers (
    id TEXT PRIMARY KEY NOT NULL,
    code TEXT NOT NULL UNIQUE,
    dominant_hand TEXT NOT NULL CHECK (dominant_hand IN ('LEFT', 'RIGHT', 'BOTH', 'UNKNOWN')),
    age_group TEXT,
    proficiency TEXT NOT NULL CHECK (proficiency IN ('BEGINNER', 'INTERMEDIATE', 'FLUENT', 'NATIVE', 'UNKNOWN')),
    region TEXT,
    consent_status TEXT NOT NULL DEFAULT 'UNKNOWN'
        CHECK (consent_status IN ('UNKNOWN', 'GRANTED', 'DENIED', 'WITHDRAWN')),
    consent_recorded_at TEXT,
    consent_reference TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE recording_sessions (
    id TEXT PRIMARY KEY NOT NULL,
    signer_id TEXT NOT NULL REFERENCES signers(id),
    operator TEXT,
    camera_source_type TEXT NOT NULL CHECK (camera_source_type IN ('LOCAL', 'REMOTE')),
    camera_angle TEXT NOT NULL CHECK (camera_angle IN ('FRONT', 'LEFT', 'RIGHT', 'OTHER')),
    lighting_condition TEXT NOT NULL CHECK (lighting_condition IN ('NATURAL', 'ARTIFICIAL', 'LOW_LIGHT', 'OTHER')),
    background TEXT NOT NULL CHECK (background IN ('PLAIN', 'COMPLEX', 'OTHER')),
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')),
    notes TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE signer_consent_events (
    id INTEGER PRIMARY KEY,
    signer_id TEXT NOT NULL REFERENCES signers(id),
    status TEXT NOT NULL CHECK (status IN ('UNKNOWN', 'GRANTED', 'DENIED', 'WITHDRAWN')),
    recorded_at TEXT NOT NULL,
    reference TEXT
);
CREATE INDEX consent_events_signer ON signer_consent_events (signer_id, status, id);
CREATE INDEX sessions_signer ON recording_sessions (signer_id, status);

CREATE TABLE video_samples (
    id TEXT PRIMARY KEY NOT NULL,
    sign_id TEXT NOT NULL REFERENCES signs(id),
    session_id TEXT NOT NULL REFERENCES recording_sessions(id),
    sample_type TEXT NOT NULL DEFAULT 'ISOLATED',
    repetition INTEGER NOT NULL CHECK (repetition > 0),
    file_path TEXT NOT NULL UNIQUE,
    metadata_path TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    requested_fps REAL NOT NULL CHECK (requested_fps > 0),
    measured_fps REAL NOT NULL CHECK (measured_fps > 0),
    encoded_fps REAL NOT NULL CHECK (encoded_fps > 0),
    frame_count INTEGER NOT NULL CHECK (frame_count > 0),
    failed_frame_reads INTEGER NOT NULL CHECK (failed_frame_reads >= 0),
    requested_duration_seconds REAL NOT NULL CHECK (requested_duration_seconds > 0),
    actual_duration_seconds REAL NOT NULL CHECK (actual_duration_seconds > 0),
    file_size_bytes INTEGER NOT NULL CHECK (file_size_bytes > 0),
    status TEXT NOT NULL DEFAULT 'CAPTURED' CHECK (status IN ('CAPTURED', 'APPROVED', 'REJECTED', 'EXCLUDED')),
    recorded_at TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE INDEX samples_sign ON video_samples (sign_id);
CREATE INDEX samples_session ON video_samples (session_id);
