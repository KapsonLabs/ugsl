CREATE TABLE video_samples_next (
    id TEXT PRIMARY KEY NOT NULL,
    sign_id TEXT NOT NULL REFERENCES signs(id),
    session_id TEXT NOT NULL REFERENCES recording_sessions(id),
    sample_type TEXT NOT NULL DEFAULT 'ISOLATED',
    repetition INTEGER CHECK (repetition > 0),
    file_path TEXT NOT NULL UNIQUE,
    metadata_path TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    requested_fps REAL CHECK (requested_fps > 0),
    measured_fps REAL CHECK (measured_fps > 0),
    encoded_fps REAL CHECK (encoded_fps > 0),
    frame_count INTEGER NOT NULL CHECK (frame_count > 0),
    failed_frame_reads INTEGER CHECK (failed_frame_reads >= 0),
    requested_duration_seconds REAL CHECK (requested_duration_seconds > 0),
    actual_duration_seconds REAL CHECK (actual_duration_seconds > 0),
    file_size_bytes INTEGER NOT NULL CHECK (file_size_bytes > 0),
    status TEXT NOT NULL DEFAULT 'CAPTURED' CHECK (status IN ('CAPTURED', 'APPROVED', 'REJECTED', 'EXCLUDED')),
    recorded_at TEXT,
    registered_at TEXT NOT NULL,
    sha256 TEXT UNIQUE CHECK (sha256 IS NULL OR length(sha256) = 64),
    source_type TEXT NOT NULL DEFAULT 'CAMERA' CHECK (source_type IN ('CAMERA', 'IMPORT')),
    source_reference TEXT,
    notes TEXT,
    container_format TEXT,
    video_codec TEXT,
    pixel_format TEXT,
    average_fps REAL,
    nominal_fps REAL,
    duration_seconds REAL,
    rotation REAL,
    bitrate INTEGER,
    technical_validation_status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (technical_validation_status IN ('PENDING', 'PASSED', 'PASSED_WITH_WARNINGS', 'FAILED', 'ERROR')),
    technical_validated_at TEXT,
    validator_version TEXT
);
INSERT INTO video_samples_next (id, sign_id, session_id, sample_type, repetition, file_path, metadata_path,
    width, height, requested_fps, measured_fps, encoded_fps, frame_count, failed_frame_reads,
    requested_duration_seconds, actual_duration_seconds, file_size_bytes, status, recorded_at, registered_at)
SELECT id, sign_id, session_id, sample_type, repetition, file_path, metadata_path,
    width, height, requested_fps, measured_fps, encoded_fps, frame_count, failed_frame_reads,
    requested_duration_seconds, actual_duration_seconds, file_size_bytes, status, recorded_at, registered_at
FROM video_samples;
DROP TABLE video_samples;
ALTER TABLE video_samples_next RENAME TO video_samples;
CREATE INDEX samples_sign ON video_samples (sign_id);
CREATE INDEX samples_session ON video_samples (session_id);
CREATE INDEX samples_size_unhashed ON video_samples (file_size_bytes) WHERE sha256 IS NULL;
CREATE INDEX samples_validation_status ON video_samples (technical_validation_status);

CREATE TABLE video_validations (
    id TEXT PRIMARY KEY NOT NULL,
    sample_id TEXT NOT NULL REFERENCES video_samples(id),
    validation_status TEXT NOT NULL CHECK (validation_status IN ('PENDING', 'PASSED', 'PASSED_WITH_WARNINGS', 'FAILED', 'ERROR')),
    validation_profile TEXT NOT NULL,
    validator_version TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    validated_at TEXT NOT NULL
);
CREATE INDEX validations_sample ON video_validations (sample_id, validated_at);

CREATE TABLE ingestion_attempts (
    id TEXT PRIMARY KEY NOT NULL,
    batch_id TEXT,
    sample_id TEXT REFERENCES video_samples(id),
    source_reference TEXT,
    sha256 TEXT,
    result TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    report_json TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX attempts_batch ON ingestion_attempts (batch_id);
