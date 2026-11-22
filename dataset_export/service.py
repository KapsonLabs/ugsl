"""Build, verify, publish and recover immutable dataset versions."""

import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from annotation import repository as annotations
from annotation.eligibility import TrainingEligibilityService
from config import ReviewConfig
from dataset_export import manifest, repository, statistics, vocabulary
from dataset_export.eligibility import allocate_version, resolve_filters, select, validate_config
from dataset_export.integrity import REQUIRED_FILES, DatasetIntegrityVerifier, load_metadata, load_vocabulary
from dataset_export.models import ANNOTATION_SCHEMA_VERSION, SCHEMA_VERSION, AnnotationRecord, DatasetMember, DatasetVersion, ExportConfig, ManifestRecord
from dataset_export.storage import (
    BUILD_MARKER, REVOCATION_FILE, ExportStorageUnavailable, check_space, copy_video, export_lock, publish, safe_join,
    sha256_file,
)
from registry import datasets, sessions
from registry.database import Database, DatabaseError, RegistryError
from registry.repository import utc_now


@dataclass(frozen=True)
class ExportResult:
    status: str
    name: str
    version: str
    output_path: str
    total_samples: int
    excluded: list[dict]
    threshold_failures: list[dict]
    dataset_id: str | None = None
    manifest_checksum: str | None = None
    content_fingerprint: str | None = None
    error: str | None = None


def _folder(name: str, version: str) -> str:
    return f"{name}-v{version}"


def _records(selection, words, config) -> list[ManifestRecord]:
    indices = {row["sign_id"]: row["class_index"] for row in words["classes"]}
    records = []
    for item in selection.selected:
        sample, sign, ann = item.sample, item.sign, item.annotation
        video_path = (
            f"videos/{sign.id}/{sample.id}{item.source.suffix}" if config.mode == "PORTABLE"
            else item.source.relative_to(config.storage_root).as_posix()
        )
        records.append(ManifestRecord(
            sample.id, sign.id, sign.gloss, indices[sign.id], item.signer.code, sample.session_id, video_path,
            sample.sha256, sample.container_format, sample.video_codec, sample.width, sample.height, sample.average_fps,
            sample.frame_count, item.duration_ms, manifest.orientation(sample.width, sample.height, sample.rotation),
            AnnotationRecord(
                ann.id, ann.version, ann.start_ms, ann.end_ms, ann.hand_usage, ann.dominant_hand,
                ann.movement_type, ann.facial_expression, ann.completeness, ann.confidence,
            ),
            sample.technical_validation_status, sample.source_type, item.recorded_at, sample.file_size_bytes, sample.rotation,
        ))
    return records


def _configuration(config: ExportConfig, version: str, raw: Path, review: ReviewConfig) -> tuple[dict, dict]:
    public = asdict(config)
    # The registry retains local paths. The artifact records portable resolution rules.
    public.update(version=version, version_strategy=config.bump or ("explicit" if config.version else "patch"),
                  output_directory=".", storage_root="external" if config.mode == "REFERENCE" else None,
                  review_policy=review.policy, purpose="isolated_sign_training")
    private = {**public, "output_directory": str(config.output_directory), "storage_root": str(config.storage_root),
               "raw_root": str(raw)}
    return public, private


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=2, check=True,
        )
        revision = result.stdout.strip()
        if len(revision) in {40, 64} and all(char in "0123456789abcdef" for char in revision):
            return revision
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _write_readme(stage, data, stats):
    rows = []
    for entry in stats["classes"]:
        gloss = entry["gloss"].replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {entry['sign_id']} | {gloss} | {entry['samples']} | {entry['unique_signers']} |")
    resolution = (
        "Configure the source storage root before reading video references. The original storage must remain accessible."
        if data["export_mode"] == "REFERENCE" else "Video paths are relative to this directory. Keep the videos with the manifest."
    )
    text = f"""# {data['name']} {data['version']}

This UGSL dataset contains isolated sign recordings for the approved training purpose. It includes
{data['total_samples']} samples from {data['total_signers']} pseudonymous signers. The clips originated in camera capture or
local video imports. The source video encoding is preserved. Sign intervals are stored as annotation timestamps.

## Classes

| Sign ID | Verified gloss | Samples | Unique signers |
| --- | --- | --- | --- |
{chr(10).join(rows)}

## Validation and review

The pipeline decodes the source video and checks configured isolated-sign limits for resolution, frame rate,
frame count, duration and size. Visual warnings are recorded where the export configuration allows them.
Human annotations record a verified sign and interval. The saved annotation version had current approval
under the applicable review policy at export time.

## Files and reading

The export format is {data['export_mode']}. {resolution}

- `dataset.json` describes this dataset version, configuration and checksums.
- `manifest.jsonl` is authoritative. Each row contains a sample and session UUID, verified sign and class index,
  pseudonymous signer, relative video path and SHA-256, media properties, source, and optional UTC recording time.
  The nested annotation contains its ID, version, interval in milliseconds, and linguistic attributes.
- `vocabulary.json` freezes the class mapping. Keep this mapping with models trained on the dataset.
- `statistics.json` reports sample distributions and coverage.
- `checksums.sha256` lists the checksums of the generated files and portable video copies where applicable.

Use `python cli.py dataset verify --path PATH_TO_THIS_DATASET` with the UGSL toolkit. For reference exports also supply
`--storage-root`. Python consumers can use `dataset_export.reader.DatasetReader` with the same configured storage root.

## Limitations and consent

These clips represent isolated signs. Continuous signing recognition requires a separate dataset and evaluation.
Coverage and signer representation can vary across classes. Class indices may change between exports.

Check the current registry or a revocation record before use or redistribution. Offline checks verify bytes while
current permission may be unknown. If consent is withdrawn, stop processing the affected version and remove or
restrict distributed copies and downstream artifacts as directed by the dataset custodian.
"""
    with (stage / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_revocation_notice(version: DatasetVersion) -> bool:
    folder = Path(version.output_path)
    if version.status != "REVOKED" or folder.is_symlink() or not folder.is_dir():
        return False
    if load_metadata(folder)["dataset_id"] != version.id:
        raise RegistryError("The export directory belongs to a different dataset.")
    receipt = safe_join(folder, REVOCATION_FILE)
    if receipt.exists():
        return False
    data = {"dataset_id": version.id, "status": version.status, "revoked_at": version.revoked_at,
            "reason": version.revocation_reason}
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, prefix=".revocation-") as stream:
        stream.write(manifest.canonical(data) + "\n")
        stream.flush()
        try:
            os.link(stream.name, receipt)
        except FileExistsError:
            return False
    return True


class DatasetExportService:
    def __init__(self, db: Database, raw: Path, review: ReviewConfig | None = None):
        self.db = db
        self.raw = Path(raw).expanduser().resolve()
        self.review = review or ReviewConfig()
        self.verifier = DatasetIntegrityVerifier()

    def export(self, config: ExportConfig) -> ExportResult:
        config = validate_config(config, self.raw)
        db = Database(self.db.path, read_only=True) if config.dry_run else self.db
        with db.connect() as connection:
            config = resolve_filters(connection, config)
            number = allocate_version(connection, config)
            report = select(connection, db, self.raw, config, self.review)
        outcome = ExportResult(
            "DRY_RUN" if config.dry_run else "BUILDING", config.name, number,
            str(config.output_directory / _folder(config.name, number)), len(report.selected), report.excluded,
            report.threshold_failures,
        )
        if not report.selected:
            return replace(outcome, status="BLOCKED", error="No eligible samples matched the export configuration.")
        if report.threshold_failures and not config.allow_incomplete_coverage:
            return replace(outcome, status="BLOCKED", error="Coverage requirements are not satisfied.")
        words = vocabulary.snapshot(item.sign for item in report.selected)
        records = _records(report, words, config)
        lines = manifest.jsonl(records)
        stats = statistics.summarize(records, words, report)
        if config.dry_run:
            effective, _ = _configuration(config, number, self.raw, self.review)
            return replace(
                outcome, manifest_checksum=manifest.sha256_text(lines),
                content_fingerprint=manifest.fingerprint(records, words, effective),
            )

        with export_lock(config.output_directory):
            with self.db.connect(write=True) as connection:
                number = allocate_version(connection, config)
                effective, private = _configuration(config, number, self.raw, self.review)
                destination = safe_join(config.output_directory, _folder(config.name, number))
                if destination.exists() or destination.is_symlink():
                    raise RegistryError("The export destination already exists.")
                created = utc_now()
                version = DatasetVersion(
                    str(uuid4()), config.name, number, SCHEMA_VERSION, config.description, config.mode, "BUILDING",
                    len(records), len(words["classes"]), stats["overall"]["total_signers"], manifest.sha256_text(lines),
                    manifest.fingerprint(records, words, effective), manifest.canonical(private), str(destination), created,
                )
                membership = [
                    DatasetMember(
                        str(uuid4()), version.id, row.sample_id, row.annotation.id, row.sign_id, item.signer.id,
                        item.sample.review_round, row.sha256, manifest.sha256_text(manifest.canonical(asdict(row))), created,
                    )
                    for item, row in zip(report.selected, records)
                ]
                self._recheck(connection, version, membership, verify_media=False)
                repository.reserve(connection, version, membership)

            stage = config.output_directory / f".building-{version.id}"
            try:
                self._write(stage, version, config, effective, report, records, lines, words, stats)
                self._verify_build(stage, version, membership, config.storage_root)
            except (OSError, RegistryError, ValueError) as exc:
                self._mark_failed(version.id, str(exc))
                raise RegistryError(f"Export preparation failed for {version.name} {version.version}: {exc}") from exc

            try:
                self._finalize(version, stage, destination, membership)
            except DatabaseError as exc:
                raise RegistryError("Export publication was interrupted. Run dataset reconcile.") from exc
            except (RegistryError, OSError) as exc:
                self._mark_failed(version.id, str(exc))
                raise RegistryError(f"Export finalization failed: {exc}. Run dataset reconcile.") from exc

            version = datasets.show(self.db, version.name, version.version)
            self._finish_marker(version, destination)
            return replace(
                outcome, status=version.status, version=number, output_path=str(destination), dataset_id=version.id,
                manifest_checksum=version.manifest_checksum, content_fingerprint=version.content_fingerprint,
            )

    def _write(self, stage, version, config, effective, selection, records, lines, words, stats):
        if config.mode == "PORTABLE":
            check_space(stage.parent, stats["overall"]["total_video_bytes"] + len(lines.encode("utf-8")))
        stage.mkdir(mode=0o700)
        (stage / BUILD_MARKER).touch(exist_ok=False)
        if config.mode == "PORTABLE":
            for item, row in zip(selection.selected, records):
                copy_video(item.source, safe_join(stage, row.video_path))
        with (stage / "manifest.jsonl").open("x", encoding="utf-8", newline="\n") as file:
            file.write(lines)
        manifest.document(stage / "vocabulary.json", words)
        manifest.document(stage / "statistics.json", stats)
        if config.csv:
            manifest.write_csv(stage / "manifest.csv", records)
        data = {
            "dataset_id": version.id, "name": version.name, "version": version.version, "schema_version": version.schema_version,
            "language": "UGSL", "description": version.description, "created_at": version.created_at,
            "export_mode": version.export_mode, **stats["overall"], "manifest": "manifest.jsonl", "vocabulary": "vocabulary.json",
            "statistics": "statistics.json", "manifest_csv": "manifest.csv" if config.csv else None,
            "manifest_checksum": version.manifest_checksum, "content_fingerprint": version.content_fingerprint,
            "configuration": effective, "sample_types": ["ISOLATED"], "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "technical_validation_profiles": sorted({item.validation_profile for item in selection.selected if item.validation_profile}),
            "validator_versions": sorted({item.sample.validator_version for item in selection.selected if item.sample.validator_version}),
            "git_commit": _git_revision(),
        }
        manifest.document(stage / "dataset.json", data)
        _write_readme(stage, data, stats)
        names = REQUIRED_FILES | ({"manifest.csv"} if config.csv else set())
        checksums = {name: sha256_file(stage / name) for name in names}
        if config.mode == "PORTABLE":
            checksums.update({row.video_path: row.sha256 for row in records})
        with (stage / "checksums.sha256").open("x", encoding="utf-8", newline="\n") as file:
            for name, checksum in sorted(checksums.items()):
                file.write(f"{checksum}  {name}\n")

    def _verify_build(self, folder, version, membership, storage_root):
        result = self.verifier.verify(folder, storage_root=storage_root, version=version, membership=membership)
        if not result.integrity_ok:
            raise RegistryError("Export integrity failed. " + " ".join(result.errors))

    def _recheck(self, connection, version, membership, *, verify_media=True, reference_paths=None):
        private = json.loads(version.configuration_json)
        raw = Path(private["raw_root"])
        review = replace(self.review, policy="DUAL") if private["review_policy"] == "DUAL" else self.review
        service = TrainingEligibilityService(self.db, raw, review)
        for member in membership:
            sample = annotations.sample(connection, member.sample_id)
            current = service.evaluate(connection, sample, verify_media=verify_media)
            session = sessions.find(connection, sample.session_id)
            if (
                not current.eligible or current.annotation_id != member.annotation_id
                or current.verified_sign_id != member.sign_id or sample.sha256 != member.source_checksum
                or sample.review_round != member.approval_round or session.signer_id != member.signer_id
                or (not private["include_warnings"] and sample.technical_validation_status != "PASSED")
            ):
                raise RegistryError(f"Snapshot member {member.sample_id} is no longer eligible: "
                                    f"{current.reason or 'Registry state changed.'}")
            if reference_paths is not None:
                root = Path(private["storage_root"])
                path = safe_join(raw, sample.file_path).relative_to(root).as_posix()
                if reference_paths.get(member.sample_id) != path:
                    raise RegistryError(f"Snapshot member {member.sample_id} changed its reference path.")

    @staticmethod
    def _reference_paths(folder, version):
        if version.export_mode != "REFERENCE":
            return None
        _, classes = load_vocabulary(folder)
        return {row.sample_id: row.video_path for row in manifest.read_records(safe_join(folder, "manifest.jsonl"), classes)}

    def _finalize(self, version, folder, destination, membership):
        reference_paths = self._reference_paths(folder, version)
        with self.db.connect(write=True) as connection:
            current = repository.by_id(connection, version.id)
            if current is None or current.status != "BUILDING":
                raise RegistryError("The dataset is no longer in BUILDING state.")
            self._recheck(connection, current, membership, reference_paths=reference_paths)
            if folder != destination:
                publish(folder, destination)
            repository.activate(connection, version.id)

    def _mark_failed(self, dataset_id, reason):
        with self.db.connect(write=True) as connection:
            repository.fail(connection, dataset_id, reason)

    @staticmethod
    def _finish_marker(version, folder):
        try:
            if version.status in {"ACTIVE", "SUPERSEDED"}:
                (folder / BUILD_MARKER).unlink(missing_ok=True)
            elif version.status == "REVOKED":
                write_revocation_notice(version)
        except OSError as exc:
            raise RegistryError("Could not finish the local publication marker. Run dataset reconcile.") from exc

    def reconcile(self, *, dry_run=False) -> list[dict]:
        db = Database(self.db.path, read_only=True) if dry_run else self.db
        report = []
        for version in datasets.list_versions(db):
            destination = Path(version.output_path)
            stage = destination.parent / f".building-{version.id}"
            marker = destination / BUILD_MARKER
            if version.status not in {"BUILDING", "REVOKED", "FAILED"} and not (marker.exists() or marker.is_symlink()):
                continue
            try:
                if version.status == "REVOKED":
                    if destination.is_dir() and not (destination / REVOCATION_FILE).exists():
                        if not dry_run:
                            write_revocation_notice(version)
                        report.append({"version": version.version, "name": version.name, "action": "Write revocation notice"})
                    if stage.is_dir():
                        report.append({"version": version.version, "name": version.name,
                                       "action": f"Revoked staging retained for custodian review: {stage}"})
                    continue
                if version.status == "FAILED":
                    if stage.is_dir() or destination.is_dir():
                        report.append({"version": version.version, "name": version.name, "action": "Failed build retained for inspection"})
                    continue
                lock = nullcontext() if dry_run or not destination.parent.is_dir() else export_lock(destination.parent)
                with lock:
                    with db.connect() as connection:
                        current = repository.by_id(connection, version.id)
                        membership = repository.members(connection, version.id)
                    if current.status != version.status:
                        continue
                    private = json.loads(current.configuration_json)
                    folder = destination if destination.exists() else stage
                    if folder.is_symlink() or (folder == stage and not folder.is_dir()):
                        raise RegistryError("No complete and safe export directory is available.")
                    self._verify_build(folder, current, membership, Path(private["storage_root"]))
                    if current.status == "BUILDING":
                        if dry_run:
                            with db.connect() as connection:
                                self._recheck(connection, current, membership, reference_paths=self._reference_paths(folder, current))
                        else:
                            self._finalize(current, folder, destination, membership)
                        action = "Would finalize" if dry_run else "Finalized"
                    else:
                        action = "Would clear build marker" if dry_run else "Cleared build marker"
                    if not dry_run:
                        current = datasets.show(db, current.name, current.version)
                        self._finish_marker(current, destination)
                    report.append({"name": current.name, "version": current.version, "action": action})
            except (RegistryError, OSError, ValueError) as exc:
                # An unavailable lock or database can be retried. Incomplete bytes require a new build.
                if not dry_run and version.status == "BUILDING" and not isinstance(exc, (DatabaseError, ExportStorageUnavailable)):
                    self._mark_failed(version.id, str(exc))
                report.append({"name": version.name, "version": version.version, "error": str(exc)})
        return report
