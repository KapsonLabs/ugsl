"""CLI entry points for media inspection, ingestion and maintenance."""

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from config import QualityConfig
from ingestion.importer import directory_entries, import_batch, manifest_entries
from ingestion.pipeline import IngestionService
from ingestion.schemas import IngestionOutcome, ValidationStatus
from ingestion.validator import VideoValidator
from registry import media_maintenance, samples
from registry.commands import database, registry_app, report_errors, sample_app
from registry.database import Database

video_app = typer.Typer(help="Inspect and validate local videos.", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingest external videos.", no_args_is_help=True)
DRY_RUN = typer.Option(help="Preview decisions without saving files or registry records.")
QUARANTINE = typer.Option(help="Retain rejected footage in restricted quarantine when consent allows.")
JSON_REPORT = typer.Option("--json", help="Write the structured report to stdout.")
BATCH_FAILURES = {IngestionOutcome.CONFLICT, IngestionOutcome.REJECTED, IngestionOutcome.ERROR}
VALIDATION_FAILURES = {ValidationStatus.FAILED, ValidationStatus.ERROR}


def configured_quality() -> QualityConfig:
    try:
        return QualityConfig.from_environment()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _service(ctx):
    return IngestionService(database(ctx), ctx.obj.dataset_root, configured_quality())


def _json(payload):
    typer.echo(json.dumps(payload, indent=2, allow_nan=False))


def _validation(report):
    _json(asdict(report))
    if report.validation_status in VALIDATION_FAILURES:
        raise typer.Exit(1)


@video_app.command("inspect")
@report_errors
def video_inspect(path: Path):
    validator = VideoValidator(configured_quality())
    _json(asdict(validator.metadata.extract(path)))


@video_app.command("validate")
@report_errors
def video_validate(path: Path):
    _, report = VideoValidator(configured_quality()).validate(path)
    _validation(report)


@ingest_app.command("file")
@report_errors
def ingest_file(
    ctx: typer.Context, path: Path,
    sign: Annotated[str, typer.Option()],
    session: Annotated[str, typer.Option()],
    recorded_at: Annotated[str | None, typer.Option(help="ISO 8601 timestamp with a UTC offset.")] = None,
    source_reference: Annotated[str | None, typer.Option("--source-description", "--source-reference")] = None,
    notes: Annotated[str | None, typer.Option()] = None,
    dry_run: Annotated[bool, DRY_RUN] = False,
    quarantine: Annotated[bool, QUARANTINE] = False,
):
    result = _service(ctx).import_file(path, sign_id=sign, session_id=session, recorded_at=recorded_at,
                                       source_reference=source_reference, notes=notes, dry_run=dry_run, quarantine=quarantine)
    _json(asdict(result))
    if result.outcome in BATCH_FAILURES:
        raise typer.Exit(1)


def _batch(ctx, entries, dry_run, fail_fast, quarantine, json_report):
    report = import_batch(_service(ctx), entries, dry_run=dry_run, fail_fast=fail_fast, quarantine=quarantine)
    summary = report.summary()
    if json_report:
        _json({"summary": summary, "results": [
            {"source_reference": entry.source_reference or entry.file_path.name, **asdict(result)}
            for entry, result in zip(entries, report.results)
        ]})
    else:
        for entry, result in zip(entries, report.results):
            sample = f" [{result.sample_id}]" if result.sample_id else ""
            message = f" {result.message}" if result.message else ""
            typer.echo(f"{entry.file_path.name}: {result.outcome}{sample}{message}")
        if dry_run:
            typer.echo("Dry run")
        for key, value in summary.items():
            typer.echo(f"{key.replace('_', ' ').capitalize()}: {value:.2f}" if isinstance(value, float) else
                       f"{key.replace('_', ' ').capitalize()}: {value}")
    if any(result.outcome in BATCH_FAILURES for result in report.results):
        raise typer.Exit(1)


@ingest_app.command("directory")
@report_errors
def ingest_directory(
    ctx: typer.Context, path: Path, sign: Annotated[str, typer.Option()], session: Annotated[str, typer.Option()],
    recursive: Annotated[bool, typer.Option()] = False, dry_run: Annotated[bool, DRY_RUN] = False,
    fail_fast: Annotated[bool, typer.Option()] = False,
    quarantine: Annotated[bool, QUARANTINE] = False, json_report: Annotated[bool, JSON_REPORT] = False,
):
    _batch(ctx, directory_entries(path, sign, session, recursive=recursive), dry_run, fail_fast, quarantine, json_report)


@ingest_app.command("manifest")
@report_errors
def ingest_manifest(
    ctx: typer.Context, path: Path, dry_run: Annotated[bool, DRY_RUN] = False,
    fail_fast: Annotated[bool, typer.Option()] = False,
    quarantine: Annotated[bool, QUARANTINE] = False, json_report: Annotated[bool, JSON_REPORT] = False,
):
    _batch(ctx, manifest_entries(path), dry_run, fail_fast, quarantine, json_report)


@sample_app.command("validate")
@report_errors
def sample_validate(ctx: typer.Context, sample_id: str, dry_run: Annotated[bool, DRY_RUN] = False):
    _validation(media_maintenance.validate_sample(_service(ctx), sample_id, dry_run=dry_run))


@sample_app.command("validate-all")
@report_errors
def sample_validate_all(
    ctx: typer.Context, dry_run: Annotated[bool, DRY_RUN] = False, json_report: Annotated[bool, JSON_REPORT] = False,
):
    service = _service(ctx)
    db = Database(service.db.path, read_only=True) if dry_run else service.db
    started = time.monotonic()
    validation = {}
    for sample in samples.list_samples(db):
        validation[sample.id] = media_maintenance.validate_sample(service, sample.id, dry_run=dry_run)
    counts = Counter(report.validation_status for report in validation.values())
    summary = {
        "discovered": len(validation), "duration_seconds": time.monotonic() - started, "storage_bytes": 0,
        **{status.lower(): counts[status] for status in ValidationStatus},
    }
    if json_report:
        _json({"summary": summary, "reports": {key: asdict(report) for key, report in validation.items()}})
    else:
        for sample_id, report in validation.items():
            typer.echo(f"{sample_id}: {report.validation_status}")
        typer.echo(f"{len(validation)} checked, {counts[ValidationStatus.PASSED]} passed, "
                   f"{counts[ValidationStatus.PASSED_WITH_WARNINGS]} passed with warnings, "
                   f"{counts[ValidationStatus.FAILED]} failed, {counts[ValidationStatus.ERROR]} errors "
                   f"in {summary['duration_seconds']:.2f}s.")
    if any(counts[status] for status in VALIDATION_FAILURES):
        raise typer.Exit(1)


def _maintenance(report):
    for action in report.actions:
        typer.echo(action)
    for issue in report.issues:
        typer.echo(issue, err=True)
    typer.echo(f"{len(report.actions)} actions, {len(report.issues)} issues.")
    if report.issues:
        raise typer.Exit(1)


@registry_app.command("backfill-hashes")
@report_errors
def registry_backfill_hashes(ctx: typer.Context, dry_run: Annotated[bool, DRY_RUN] = False):
    _maintenance(media_maintenance.backfill_hashes(_service(ctx), dry_run=dry_run))


@registry_app.command("reconcile")
@report_errors
def registry_reconcile(
    ctx: typer.Context, dry_run: Annotated[bool, DRY_RUN] = False,
    purge_quarantine: Annotated[bool, typer.Option(help="Remove expired or consent-restricted quarantine copies.")] = False,
):
    _maintenance(media_maintenance.reconcile(_service(ctx), dry_run=dry_run, purge_quarantine=purge_quarantine))


def mount(app: typer.Typer) -> None:
    app.add_typer(video_app, name="video")
    app.add_typer(ingest_app, name="ingest")
