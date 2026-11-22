CREATE TABLE dataset_versions (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT COLLATE NOCASE NOT NULL,
    version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    description TEXT,
    export_mode TEXT NOT NULL CHECK (export_mode IN ('REFERENCE', 'PORTABLE')),
    status TEXT NOT NULL CHECK (status IN ('BUILDING', 'ACTIVE', 'SUPERSEDED', 'REVOKED', 'FAILED')),
    total_samples INTEGER NOT NULL CHECK (total_samples > 0),
    total_signs INTEGER NOT NULL CHECK (total_signs > 0),
    total_signers INTEGER NOT NULL CHECK (total_signers > 0),
    manifest_checksum TEXT NOT NULL CHECK (length(manifest_checksum) = 64),
    content_fingerprint TEXT NOT NULL CHECK (length(content_fingerprint) = 64),
    configuration_json TEXT NOT NULL,
    output_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    finalized_at TEXT,
    revoked_at TEXT,
    revocation_reason TEXT,
    UNIQUE (name, version)
);

CREATE TABLE dataset_members (
    id TEXT PRIMARY KEY NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    sample_id TEXT NOT NULL REFERENCES video_samples(id),
    annotation_id TEXT NOT NULL REFERENCES annotations(id),
    sign_id TEXT NOT NULL REFERENCES signs(id),
    signer_id TEXT NOT NULL REFERENCES signers(id),
    approval_round INTEGER NOT NULL CHECK (approval_round >= 1),
    source_checksum TEXT NOT NULL CHECK (length(source_checksum) = 64),
    manifest_entry_checksum TEXT NOT NULL CHECK (length(manifest_entry_checksum) = 64),
    created_at TEXT NOT NULL,
    UNIQUE (dataset_version_id, sample_id),
    FOREIGN KEY (sample_id, annotation_id) REFERENCES annotations(sample_id, id)
);
CREATE INDEX dataset_member_sample ON dataset_members (sample_id);
CREATE INDEX dataset_member_signer ON dataset_members (signer_id);
CREATE INDEX dataset_member_sign ON dataset_members (sign_id);

CREATE TABLE dataset_events (
    id INTEGER PRIMARY KEY,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    previous_status TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX dataset_event_version ON dataset_events (dataset_version_id, id);

CREATE TRIGGER dataset_version_created AFTER INSERT ON dataset_versions
BEGIN
    INSERT INTO dataset_events (dataset_version_id, previous_status, status, created_at)
    VALUES (NEW.id, NULL, NEW.status, NEW.created_at);
END;

CREATE TRIGGER dataset_status_changed AFTER UPDATE OF status ON dataset_versions
WHEN NEW.status <> OLD.status
BEGIN
    INSERT INTO dataset_events (dataset_version_id, previous_status, status, reason, created_at)
    VALUES (NEW.id, OLD.status, NEW.status, NEW.revocation_reason, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
END;

CREATE TRIGGER dataset_consent_invalidated AFTER UPDATE OF consent_status ON signers
WHEN OLD.consent_status = 'GRANTED' AND NEW.consent_status <> 'GRANTED'
BEGIN
    UPDATE dataset_versions
    SET status = 'REVOKED', revoked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
        revocation_reason = 'Signer consent is no longer granted'
    WHERE status IN ('BUILDING', 'ACTIVE', 'SUPERSEDED') AND id IN (
        SELECT dataset_version_id FROM dataset_members WHERE signer_id = NEW.id
    );
END;

CREATE TRIGGER dataset_approval_invalidated
AFTER UPDATE OF current_annotation_id, review_round, review_state, status, sha256, technical_validation_status ON video_samples
WHEN (OLD.current_annotation_id IS NOT NEW.current_annotation_id OR OLD.review_round <> NEW.review_round
      OR NEW.status <> 'APPROVED' OR NEW.review_state <> 'APPROVED'
      OR OLD.sha256 IS NOT NEW.sha256 OR NEW.technical_validation_status NOT IN ('PASSED', 'PASSED_WITH_WARNINGS'))
BEGIN
    UPDATE dataset_versions
    SET status = 'REVOKED', revoked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
        revocation_reason = 'A member approval, source checksum or technical validation was invalidated'
    WHERE status IN ('BUILDING', 'ACTIVE', 'SUPERSEDED') AND id IN (
        SELECT dataset_version_id FROM dataset_members WHERE sample_id = NEW.id
    );
END;

CREATE TRIGGER dataset_sign_deactivated AFTER UPDATE OF is_active ON signs
WHEN OLD.is_active = 1 AND NEW.is_active = 0
BEGIN
    UPDATE dataset_versions
    SET status = 'REVOKED', revoked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
        revocation_reason = 'A verified vocabulary entry was deactivated'
    WHERE status IN ('BUILDING', 'ACTIVE', 'SUPERSEDED') AND id IN (
        SELECT dataset_version_id FROM dataset_members WHERE sign_id = NEW.id
    );
END;
