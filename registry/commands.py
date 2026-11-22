"""Typer commands for the dataset registry."""

from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from registry import samples, sessions, signers, statistics, vocabulary
from registry.database import Database, RegistryError

sign_app = typer.Typer(help="Manage the UGSL vocabulary.", no_args_is_help=True)
signer_app = typer.Typer(help="Manage signers and consent.", no_args_is_help=True)
session_app = typer.Typer(help="Manage recording sessions.", no_args_is_help=True)
sample_app = typer.Typer(help="Inspect registered recordings.", no_args_is_help=True)
registry_app = typer.Typer(help="Reconcile recorded files with SQLite.", no_args_is_help=True)


@dataclass(frozen=True)
class RegistryOptions:
    db_path: Path
    dataset_root: Path


def database(ctx: typer.Context) -> Database:
    return Database(ctx.obj.db_path)


def report_errors(function):
    @wraps(function)
    def command(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RegistryError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

    return command


def _details(item):
    for key, value in asdict(item).items():
        typer.echo(f"{key.replace('_', ' ').capitalize()}: {'' if value is None else value}")


def _table(rows, columns=()):
    if not rows:
        typer.echo("No records.")
        return
    columns = columns or tuple(rows[0])
    headings = [key.replace("_", " ").upper() for key in columns]
    values = [[" ".join(str(row[key]).splitlines()) if row[key] is not None else "" for key in columns]
              for row in rows]
    widths = [max(map(len, column)) for column in zip(headings, *values)]
    for row in (headings, *values):
        typer.echo("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def _changes(**fields):
    return {key: value for key, value in fields.items() if value is not None}


@sign_app.command("add")
@report_errors
def sign_add(
    ctx: typer.Context,
    gloss: Annotated[str, typer.Option()],
    category: Annotated[str | None, typer.Option()] = None,
    description: Annotated[str | None, typer.Option()] = None,
    sign_type: Annotated[str, typer.Option("--type")] = "unknown",
    variant: Annotated[str | None, typer.Option("--variant", "--variant-label")] = None,
    target_samples: Annotated[int, typer.Option()] = 0,
):
    _details(vocabulary.add(database(ctx), gloss=gloss, category=category, description=description,
                            sign_type=sign_type, variant_label=variant, target_samples=target_samples))


@sign_app.command("list")
@report_errors
def sign_list(
    ctx: typer.Context,
    category: Annotated[str | None, typer.Option()] = None,
    sign_type: Annotated[str | None, typer.Option("--type")] = None,
    active: Annotated[bool | None, typer.Option("--active/--inactive")] = None,
):
    rows = vocabulary.list_signs(database(ctx), category=category, sign_type=sign_type, active=active)
    _table([asdict(row) for row in rows], ("id", "gloss", "variant_label", "category", "sign_type", "target_samples", "is_active"))


@sign_app.command("show")
@report_errors
def sign_show(ctx: typer.Context, sign_id: str):
    _details(vocabulary.show(database(ctx), sign_id))


@sign_app.command("update")
@report_errors
def sign_update(
    ctx: typer.Context,
    sign_id: str,
    gloss: Annotated[str | None, typer.Option()] = None,
    category: Annotated[str | None, typer.Option()] = None,
    description: Annotated[str | None, typer.Option()] = None,
    sign_type: Annotated[str | None, typer.Option("--type")] = None,
    variant: Annotated[str | None, typer.Option("--variant", "--variant-label")] = None,
    target_samples: Annotated[int | None, typer.Option()] = None,
    active: Annotated[bool | None, typer.Option("--active/--inactive")] = None,
):
    _details(vocabulary.update(database(ctx), sign_id, **_changes(
        gloss=gloss, category=category, description=description, sign_type=sign_type, variant_label=variant,
        target_samples=target_samples, is_active=active,
    )))


@sign_app.command("deactivate")
@report_errors
def sign_deactivate(ctx: typer.Context, sign_id: str):
    _details(vocabulary.update(database(ctx), sign_id, is_active=False))


@signer_app.command("add")
@report_errors
def signer_add(
    ctx: typer.Context,
    code: Annotated[str, typer.Option()],
    dominant_hand: Annotated[str, typer.Option()] = "unknown",
    age_group: Annotated[str | None, typer.Option()] = None,
    proficiency: Annotated[str, typer.Option()] = "unknown",
    region: Annotated[str | None, typer.Option()] = None,
):
    _details(signers.add(database(ctx), code=code, dominant_hand=dominant_hand,
                         age_group=age_group, proficiency=proficiency, region=region))


@signer_app.command("list")
@report_errors
def signer_list(ctx: typer.Context):
    _table([asdict(row) for row in signers.list_signers(database(ctx))],
           ("code", "dominant_hand", "proficiency", "consent_status", "is_active"))


@signer_app.command("show")
@report_errors
def signer_show(ctx: typer.Context, code: str):
    _details(signers.show(database(ctx), code))


@signer_app.command("update")
@report_errors
def signer_update(
    ctx: typer.Context,
    code: str,
    dominant_hand: Annotated[str | None, typer.Option()] = None,
    age_group: Annotated[str | None, typer.Option()] = None,
    proficiency: Annotated[str | None, typer.Option()] = None,
    region: Annotated[str | None, typer.Option()] = None,
    active: Annotated[bool | None, typer.Option("--active/--inactive")] = None,
):
    _details(signers.update(database(ctx), code, **_changes(
        dominant_hand=dominant_hand, age_group=age_group, proficiency=proficiency, region=region, is_active=active,
    )))


@signer_app.command("deactivate")
@report_errors
def signer_deactivate(ctx: typer.Context, code: str):
    _details(signers.update(database(ctx), code, is_active=False))


def _show_consent(ctx, code, status, reference):
    signer, affected = signers.consent(database(ctx), code, status, reference)
    typer.echo(f"{signer.code}: {signer.consent_status} ({signer.consent_recorded_at})")
    if signer.consent_status == "WITHDRAWN":
        typer.echo(f"{len(affected)} registered samples excluded. Review these file pairs for retention.")
        _table([asdict(row) for row in affected], ("id", "file_path", "metadata_path"))


@signer_app.command("consent")
@report_errors
def signer_consent(
    ctx: typer.Context,
    code: str,
    status: Annotated[str, typer.Option()],
    reference: Annotated[str | None, typer.Option()] = None,
):
    _show_consent(ctx, code, status, reference)


@signer_app.command("withdraw")
@report_errors
def signer_withdraw(
    ctx: typer.Context,
    code: str,
    reference: Annotated[str | None, typer.Option()] = None,
):
    _show_consent(ctx, code, "WITHDRAWN", reference)


@session_app.command("create")
@report_errors
def session_create(
    ctx: typer.Context,
    signer: Annotated[str, typer.Option()],
    operator: Annotated[str | None, typer.Option()] = None,
    camera_type: Annotated[str, typer.Option()] = "local",
    angle: Annotated[str, typer.Option()] = "front",
    lighting: Annotated[str, typer.Option()] = "other",
    background: Annotated[str, typer.Option()] = "other",
    notes: Annotated[str | None, typer.Option()] = None,
):
    _details(sessions.create(database(ctx), signer_code=signer, operator=operator, camera_source_type=camera_type,
                             camera_angle=angle, lighting_condition=lighting, background=background, notes=notes))


@session_app.command("list")
@report_errors
def session_list(ctx: typer.Context):
    db = database(ctx)
    codes = {signer.id: signer.code for signer in signers.list_signers(db)}
    rows = [{**asdict(row), "signer": codes.get(row.signer_id, row.signer_id)} for row in sessions.list_sessions(db)]
    _table(rows, ("id", "signer", "camera_source_type", "status", "started_at", "ended_at"))


@session_app.command("show")
@report_errors
def session_show(ctx: typer.Context, session_id: str):
    _details(sessions.show(database(ctx), session_id))


@session_app.command("complete")
@report_errors
def session_complete(ctx: typer.Context, session_id: str):
    _details(sessions.finish(database(ctx), session_id, "COMPLETED"))


@session_app.command("cancel")
@report_errors
def session_cancel(ctx: typer.Context, session_id: str):
    _details(sessions.finish(database(ctx), session_id, "CANCELLED"))


@sample_app.command("list")
@report_errors
def sample_list(
    ctx: typer.Context,
    sign: Annotated[str | None, typer.Option()] = None,
    session: Annotated[str | None, typer.Option()] = None,
    status: Annotated[str | None, typer.Option()] = None,
):
    rows = samples.list_samples(database(ctx), sign_id=sign, session_id=session, status=status)
    _table([asdict(row) for row in rows], ("id", "sign_id", "session_id", "status", "file_path"))


@sample_app.command("show")
@report_errors
def sample_show(ctx: typer.Context, sample_id: str):
    _details(samples.show(database(ctx), sample_id))


@registry_app.command("sync")
@report_errors
def registry_sync(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option(help="Report proposed registrations without writing.")] = False,
    session: Annotated[str | None, typer.Option(help="Assign legacy recordings to this session.")] = None,
    sign: Annotated[str | None, typer.Option(help="Select a variant for matching legacy recordings.")] = None,
):
    report = samples.synchronize(database(ctx), ctx.obj.dataset_root, sign_id=sign, session_id=session, dry_run=dry_run)
    for path in report.registered:
        typer.echo(f"{'Would register' if dry_run else 'Registered'} {path}")
    for path, issue in report.issues:
        typer.echo(f"{path}: {issue}", err=True)
    typer.echo(f"{'Proposed' if dry_run else 'Registered'} {len(report.registered)}, "
               f"skipped {report.skipped}, issues {len(report.issues)}.")
    if report.issues:
        raise typer.Exit(1)


@report_errors
def stats(
    ctx: typer.Context,
    by: Annotated[str | None, typer.Option(help="Group by sign, signer, session or status.")] = None,
    coverage: Annotated[bool, typer.Option(help="Show vocabulary collection progress.")] = False,
    min_signer_samples: Annotated[int, typer.Option(help="Coverage threshold for eligible signers.")] = 1,
):
    db = database(ctx)
    if by:
        _table(statistics.grouped(db, by.strip().lower()))
    if coverage:
        report = statistics.coverage(db, min_signer_samples=min_signer_samples)
        typer.echo("Sign coverage")
        _table(report.signs)
        typer.echo("\nSigns with zero recordings")
        _table(report.zero_recording_signs, ("id", "gloss", "variant"))
        typer.echo("\nUnderrepresented active signs")
        _table(report.underrepresented_signs, ("id", "gloss", "variant", "collected", "target_samples"))
        typer.echo(f"\nEligible signers below {min_signer_samples} samples")
        _table(report.underrepresented_signers)
    if not by and not coverage:
        summary = asdict(statistics.summary(db))
        summary["footage_seconds"] = f"{summary['footage_seconds']:.2f}"
        _table([{"metric": key.replace("_", " "), "value": value} for key, value in summary.items()])
        typer.echo("\nSamples by status")
        _table(statistics.grouped(db, "status"))


def mount(app: typer.Typer) -> None:
    app.add_typer(sign_app, name="sign")
    app.add_typer(signer_app, name="signer")
    app.add_typer(session_app, name="session")
    app.add_typer(sample_app, name="sample")
    app.add_typer(registry_app, name="registry")
    app.command()(stats)
