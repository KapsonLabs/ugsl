"""Dataset export, inspection and lifecycle CLI commands."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from config import ReviewConfig
from dataset_export import comparison, repository
from dataset_export.integrity import DatasetIntegrityVerifier
from dataset_export.models import ExportConfig
from dataset_export.service import DatasetExportService, write_revocation_notice
from registry import datasets
from registry.commands import _details, _table, database, report_errors
from registry.database import RegistryError

dataset_app = typer.Typer(help="Export and manage immutable UGSL dataset versions.", no_args_is_help=True)
JSON_OUTPUT = typer.Option("--json", help="Write machine-readable results to stdout.")
VERSION = typer.Option("--version", help="Dataset semantic version.")


def _json(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def _service(ctx):
    try:
        review = ReviewConfig.from_environment()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return DatasetExportService(database(ctx), ctx.obj.dataset_root, review)


@dataset_app.command("export")
@report_errors
def dataset_export(
    ctx: typer.Context,
    name: Annotated[str, typer.Option()],
    version: Annotated[str | None, VERSION] = None,
    bump: Annotated[str | None, typer.Option(help="Automatically increment patch, minor or major.")] = None,
    description: Annotated[str | None, typer.Option()] = None,
    mode: Annotated[str, typer.Option(help="reference or portable.")] = "reference",
    output_directory: Annotated[Path | None, typer.Option("--output-directory", "--output-dir")] = None,
    storage_root: Annotated[Path | None, typer.Option(help="Root used to resolve reference videos.")] = None,
    sign: Annotated[list[str] | None, typer.Option()] = None,
    category: Annotated[list[str] | None, typer.Option()] = None,
    signer: Annotated[list[str] | None, typer.Option()] = None,
    session: Annotated[list[str] | None, typer.Option()] = None,
    min_samples_per_sign: Annotated[int, typer.Option()] = 1,
    min_signers_per_sign: Annotated[int, typer.Option()] = 1,
    max_samples_per_sign: Annotated[int | None, typer.Option()] = None,
    include_warnings: Annotated[bool, typer.Option("--include-warnings/--exclude-warnings")] = True,
    csv: Annotated[bool, typer.Option()] = False,
    verify_all_source_checksums: Annotated[bool, typer.Option(
        "--verify-all-source-checksums/--no-verify-all-source-checksums",
        help="Hash all candidates during selection. Final source checks are always required.",
    )] = True,
    allow_incomplete_coverage: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
    json_output: Annotated[bool, JSON_OUTPUT] = False,
):
    result = _service(ctx).export(ExportConfig(
        name=name, version=version, bump=bump, description=description, mode=mode, output_directory=output_directory,
        storage_root=storage_root, sign_ids=tuple(sign or ()), categories=tuple(category or ()),
        signer_ids=tuple(signer or ()), session_ids=tuple(session or ()), min_samples_per_sign=min_samples_per_sign,
        min_signers_per_sign=min_signers_per_sign, max_samples_per_sign=max_samples_per_sign,
        include_warnings=include_warnings, csv=csv, verify_all_source_checksums=verify_all_source_checksums,
        allow_incomplete_coverage=allow_incomplete_coverage, dry_run=dry_run,
    ))
    if json_output:
        _json(asdict(result))
    else:
        typer.echo(f"{result.status} {result.name} {result.version} ({result.total_samples} samples)")
        typer.echo(result.output_path)
        if result.excluded:
            typer.echo("\nExcluded samples")
            _table(result.excluded)
        if result.threshold_failures:
            typer.echo("\nCoverage gaps")
            _table(result.threshold_failures)
        if result.error:
            typer.echo(result.error, err=True)
    if result.status not in {"ACTIVE", "DRY_RUN"}:
        raise typer.Exit(1)


@dataset_app.command("list")
@report_errors
def dataset_list(ctx: typer.Context, name: Annotated[str | None, typer.Option()] = None):
    rows = datasets.list_versions(database(ctx), name=name)
    _table([asdict(row) for row in rows], ("name", "version", "status", "export_mode", "total_samples", "output_path"))


@dataset_app.command("show")
@report_errors
def dataset_show(ctx: typer.Context, name: str, version: Annotated[str, VERSION]):
    db = database(ctx)
    row = datasets.show(db, name, version)
    _details(row)
    typer.echo("\nLifecycle history")
    _table(datasets.history(db, row.id))


@dataset_app.command("verify")
@report_errors
def dataset_verify(
    ctx: typer.Context, name: Annotated[str | None, typer.Argument()] = None,
    version: Annotated[str | None, VERSION] = None, path: Annotated[Path | None, typer.Option()] = None,
    storage_root: Annotated[Path | None, typer.Option()] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry to consult for a standalone path.")] = None,
    revocation_record: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, JSON_OUTPUT] = False,
):
    if path is not None and (name is not None or version is not None):
        raise typer.BadParameter("Choose either a dataset name and version or --path.")
    if path is None and (name is None or version is None):
        raise typer.BadParameter("Provide a dataset name and --version, or a standalone --path.")
    snapshot, membership = None, None
    if path is None:
        db = database(ctx)
        with db.connect() as connection:
            snapshot = repository.version(connection, name, version)
            membership = repository.members(connection, snapshot.id)
        path = Path(snapshot.output_path)
        registry = db.path
        if storage_root is None:
            storage_root = Path(json.loads(snapshot.configuration_json)["storage_root"])
    elif registry is None:
        source = ctx.find_root().get_parameter_source("db")
        if source is not None and source.name != "DEFAULT":
            registry = ctx.obj.db_path
    if storage_root is None:
        storage_root = ctx.obj.dataset_root
    result = DatasetIntegrityVerifier().verify(
        path, storage_root=storage_root, registry_path=registry, version=snapshot, membership=membership,
        revocation_path=revocation_record,
    )
    if json_output:
        _json(asdict(result))
    else:
        typer.echo(f"Content integrity {'passed' if result.integrity_ok else 'failed'}")
        label = "unknown" if result.permission.permitted is None else "permitted" if result.permission.permitted else "blocked"
        typer.echo(f"Lifecycle {result.permission.status}. Use {label}.")
        if result.permission.reason:
            typer.echo(result.permission.reason)
        for error in result.errors:
            typer.echo(error, err=True)
    if not result.integrity_ok or result.permission.permitted is False:
        raise typer.Exit(1)


@dataset_app.command("stats")
@report_errors
def dataset_stats(ctx: typer.Context, name: str, version: Annotated[str, VERSION],
                  json_output: Annotated[bool, JSON_OUTPUT] = False):
    db = database(ctx)
    stats = comparison.load_statistics(db, datasets.show(db, name, version))
    if json_output:
        _json(stats)
    else:
        _table([{"metric": key.replace("_", " "), "value": value} for key, value in stats["overall"].items()])
        for label in ("classes", "signers"):
            typer.echo(f"\n{label.title()}")
            _table(stats[label])
        for label in ("quality", "annotations", "coverage"):
            typer.echo(f"\n{label.title()}")
            _json(stats[label])


@dataset_app.command("compare")
@report_errors
def dataset_compare(
    ctx: typer.Context, name: str, from_version: Annotated[str, typer.Option()], to_version: Annotated[str, typer.Option()],
    json_output: Annotated[bool, JSON_OUTPUT] = False,
):
    db = database(ctx)
    result = comparison.compare(db, datasets.show(db, name, from_version), datasets.show(db, name, to_version))
    if json_output:
        _json(result)
        return
    typer.echo(f"{name} {from_version} → {to_version}")
    for key in ("added_samples", "removed_samples", "unchanged_samples", "changed_annotations",
                "added_signs", "removed_signs", "added_signers", "removed_signers"):
        typer.echo(f"{key.replace('_', ' ').capitalize()} {len(result[key])}")
        if result[key] and key != "unchanged_samples":
            if isinstance(result[key][0], dict):
                _table(result[key])
            else:
                typer.echo(", ".join(result[key]))
    for key in ("class_mapping_changes", "class_distribution_changes"):
        typer.echo(f"\n{key.replace('_', ' ').capitalize()}")
        _table(result[key])
    typer.echo(f"Duration change {result['duration_change_ms']} ms")
    typer.echo(f"Storage change {result['storage_change_bytes']} bytes")
    typer.echo(f"Equivalent content {result['equivalent_content']}")


@dataset_app.command("affected")
@report_errors
def dataset_affected(
    ctx: typer.Context,
    signer: Annotated[str | None, typer.Option()] = None,
    sample: Annotated[str | None, typer.Option()] = None,
    annotation: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, JSON_OUTPUT] = False,
):
    rows = datasets.affected(database(ctx), signer=signer, sample_id=sample, annotation_id=annotation)
    if json_output:
        _json([{key: value for key, value in asdict(row).items() if key != "configuration_json"} for row in rows])
    else:
        _table([asdict(row) for row in rows], ("name", "version", "status", "export_mode", "output_path", "revocation_reason"))
        if any(row.status == "REVOKED" for row in rows):
            typer.echo("Stop using revoked versions. Coordinate removal of portable copies and dependent artifacts with the custodian.")


@dataset_app.command("revoke")
@report_errors
def dataset_revoke(ctx: typer.Context, name: str, version: Annotated[str, VERSION],
                   reason: Annotated[str, typer.Option()]):
    record = datasets.transition(database(ctx), name, version, "REVOKED", reason)
    try:
        write_revocation_notice(record)
    except (RegistryError, OSError) as exc:
        typer.echo(f"Registry revocation recorded. Local notice pending reconciliation. {exc}", err=True)
    _details(record)


@dataset_app.command("supersede")
@report_errors
def dataset_supersede(ctx: typer.Context, name: str, version: Annotated[str, VERSION]):
    _details(datasets.transition(database(ctx), name, version, "SUPERSEDED"))


@dataset_app.command("reconcile")
@report_errors
def dataset_reconcile(
    ctx: typer.Context, dry_run: Annotated[bool, typer.Option()] = False,
    json_output: Annotated[bool, JSON_OUTPUT] = False,
):
    report = _service(ctx).reconcile(dry_run=dry_run)
    if json_output:
        _json(report)
    else:
        _table([{**row, "action": row.get("action", ""), "error": row.get("error", "")} for row in report],
               ("name", "version", "action", "error"))
    if any(row.get("error") for row in report):
        raise typer.Exit(1)


def mount(app: typer.Typer) -> None:
    app.add_typer(dataset_app, name="dataset")
