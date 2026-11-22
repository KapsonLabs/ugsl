"""Select a deterministic approved snapshot using the Phase 1.4 service."""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from annotation import repository
from annotation.eligibility import TrainingEligibilityService
from config import EXPORT_DIRECTORY_NAME, ReviewConfig, validate_identifier
from dataset_export.manifest import is_digest, utc_timestamp
from dataset_export.models import ExportConfig
from dataset_export.storage import safe_join
from dataset_export.versioning import next_version, version_key
from ingestion.exceptions import MediaError
from registry import sessions, signers, vocabulary
from registry.database import Database, RegistryError
from registry.models import Sign, Signer, VideoSample
from registry.repository import integer, record


@dataclass(frozen=True)
class Selection:
    sample: VideoSample
    sign: Sign
    signer: Signer
    annotation: object
    source: Path
    duration_ms: int
    recorded_at: str | None
    validation_profile: str | None


@dataclass(frozen=True)
class SelectionReport:
    selected: list[Selection]
    excluded: list[dict]
    threshold_failures: list[dict]


def validate_config(config: ExportConfig, raw: Path) -> ExportConfig:
    name = config.name.strip().lower()
    try:
        validate_identifier(name, "Dataset name")
        if config.version is not None:
            version_key(config.version)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    mode = config.mode.strip().upper()
    if mode not in {"REFERENCE", "PORTABLE"}:
        raise RegistryError("Export mode must be reference or portable.")
    if config.version and config.bump:
        raise RegistryError("Choose an explicit version or a version increment.")
    if config.bump and config.bump not in {"patch", "minor", "major"}:
        raise RegistryError("Version increment must be patch, minor or major.")
    integer(config.min_samples_per_sign, "Minimum samples per sign", 0)
    integer(config.min_signers_per_sign, "Minimum signers per sign", 0)
    if config.max_samples_per_sign is not None:
        integer(config.max_samples_per_sign, "Maximum samples per sign", 1)
    output = Path(config.output_directory or raw.parent / EXPORT_DIRECTORY_NAME).expanduser().resolve()
    root = Path(config.storage_root or raw).expanduser().resolve()
    if any(output.is_relative_to(folder) for folder in (raw, raw.parent / "staging", raw.parent / "quarantine")):
        raise RegistryError("Set the export directory outside raw, ingestion staging and quarantine.")
    if mode == "REFERENCE" and not raw.is_relative_to(root):
        raise RegistryError("The reference storage root must contain the raw dataset directory.")
    normalized = lambda values: tuple(sorted({value.strip() for value in values if value.strip()}))
    return replace(
        config, name=name, mode=mode, output_directory=output, storage_root=root,
        sign_ids=normalized(config.sign_ids), signer_ids=normalized(config.signer_ids), session_ids=normalized(config.session_ids),
        categories=normalized(tuple(category.casefold() for category in config.categories)),
    )


def allocate_version(connection, config: ExportConfig) -> str:
    versions = [row[0] for row in connection.execute("SELECT version FROM dataset_versions WHERE name = ?", (config.name,))]
    number = config.version or next_version(versions, config.bump or "patch")
    if number in versions:
        raise RegistryError("That dataset name and version are already registered.")
    return number


def resolve_filters(connection, config: ExportConfig) -> ExportConfig:
    resolved_signers = []
    for value in config.signer_ids:
        row = connection.execute("SELECT id FROM signers WHERE id = ?", (value.lower(),)).fetchone()
        resolved_signers.append(row["id"] if row else signers.find(connection, value).id)
    return replace(
        config, sign_ids=tuple(sorted({vocabulary.find(connection, key).id for key in config.sign_ids})),
        signer_ids=tuple(sorted(set(resolved_signers))),
        session_ids=tuple(sorted({sessions.find(connection, key).id for key in config.session_ids})),
    )


def select(connection, db: Database, raw: Path, config: ExportConfig, review: ReviewConfig) -> SelectionReport:
    eligibility = TrainingEligibilityService(db, raw, review)
    verified = {row["id"]: record(Sign, row) for row in connection.execute("SELECT * FROM signs")}
    requested = {
        sign.id for sign in verified.values()
        if (config.sign_ids or config.categories)
        and (not config.sign_ids or sign.id in config.sign_ids)
        and (not config.categories or (sign.category or "").strip().casefold() in config.categories)
    }
    if (config.sign_ids or config.categories) and not requested:
        raise RegistryError("The requested signs and categories match no vocabulary entries.")
    selected, excluded = [], []
    rows = connection.execute("SELECT * FROM video_samples ORDER BY id")
    for row in rows:
        sample = record(VideoSample, row)
        session = sessions.find(connection, sample.session_id)
        if (config.session_ids and session.id not in config.session_ids) or (
            config.signer_ids and session.signer_id not in config.signer_ids
        ):
            excluded.append({"sample_id": sample.id, "reason": "Outside the requested sessions or signers."})
            continue
        try:
            annotation = repository.current(connection, sample)
            if requested and (annotation.sign_id if annotation else sample.sign_id) not in requested:
                raise RegistryError("Outside the requested verified signs or categories.")
            result = eligibility.evaluate(
                connection, sample, verify_media=(config.verify_all_source_checksums or config.dry_run),
            )
            if not result.eligible:
                raise RegistryError(result.reason)
            if not is_digest(sample.sha256):
                raise RegistryError("A registered SHA-256 is required. Run registry backfill-hashes.")
            if not config.include_warnings and sample.technical_validation_status == "PASSED_WITH_WARNINGS":
                raise RegistryError("Technical warnings are excluded by the export configuration.")
            source = safe_join(raw, sample.file_path)
            duration = eligibility.validator.duration_ms(sample, source)
            eligibility.validator.interval(annotation.start_ms, annotation.end_ms, duration)
            validation = connection.execute(
                """SELECT validation_profile FROM video_validations WHERE sample_id = ?
                   ORDER BY validated_at DESC, id DESC LIMIT 1""", (sample.id,),
            ).fetchone()
            selected.append(Selection(
                sample, verified[result.verified_sign_id], signers.find_by_id(connection, session.signer_id),
                annotation, source, duration, utc_timestamp(sample.recorded_at) if sample.recorded_at else None,
                validation["validation_profile"] if validation else None,
            ))
        except (RegistryError, MediaError) as exc:
            excluded.append({"sample_id": sample.id, "reason": str(exc)})
    selected.sort(key=lambda item: (item.sign.id, item.sample.id))
    counts, unique = Counter(), {}
    capped = []
    for item in selected:
        sign_id = item.sign.id
        if config.max_samples_per_sign is not None and counts[sign_id] >= config.max_samples_per_sign:
            excluded.append({"sample_id": item.sample.id, "reason": "Deterministic per-sign cap reached."})
            continue
        counts[sign_id] += 1
        unique.setdefault(sign_id, set()).add(item.signer.id)
        capped.append(item)
    failures = []
    for sign_id in sorted(requested or set(counts)):
        signer_count = len(unique.get(sign_id, ()))
        if counts[sign_id] < config.min_samples_per_sign or signer_count < config.min_signers_per_sign or not counts[sign_id]:
            failures.append({
                "sign_id": sign_id, "gloss": verified[sign_id].gloss, "samples": counts[sign_id], "unique_signers": signer_count,
                "min_samples": config.min_samples_per_sign, "min_signers": config.min_signers_per_sign,
            })
    return SelectionReport(capped, sorted(excluded, key=lambda item: item["sample_id"]), failures)
