"""Read-only dataset frame inspection and validation commands."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from dataset_export.reader import DatasetReader
from landmarks.exceptions import DatasetAccessError, FrameProcessingError
from registry.commands import database, report_errors
from registry.database import RegistryError

frames_app = typer.Typer(help="Decode and sample approved dataset videos.", no_args_is_help=True)
DATASET = typer.Option(help="Registered dataset name.")
VERSION = typer.Option(help="Registered dataset version.")
PATH = typer.Option(help="Path to a reference or portable export folder.")
STORAGE = typer.Option(help="Override the root for reference video paths.")
FPS = typer.Option(help="Target sample rate in frames per second.")
ALL = typer.Option(help="Retain all source frames in the approved interval.")


def _json(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _service(ctx, dataset, version, path, storage_root, fps, all_frames):
    from landmarks.service import DecodingService

    if path is None:
        if not dataset or not version:
            raise typer.BadParameter("Provide --dataset and --version, or --path.")
        return DecodingService.from_registry(
            database(ctx), dataset, version, storage_root=storage_root, fps=fps, all_frames=all_frames,
        )
    if dataset or version:
        raise typer.BadParameter("Choose a registered dataset or --path.")
    source = ctx.find_root().get_parameter_source("db")
    explicit_db = ctx.obj.db_path if source is not None and source.name != "DEFAULT" else None
    try:
        reader = DatasetReader(path, storage_root=storage_root, registry_path=explicit_db)
    except (RegistryError, OSError, ValueError) as exc:
        raise DatasetAccessError(str(exc)) from exc
    return DecodingService(reader, fps=fps, all_frames=all_frames)


def _consume(run, action=None):
    with run:
        try:
            if action is not None:
                action(run)
            else:
                for _ in run:
                    pass
        except FrameProcessingError as exc:
            if run.summary.error_code is None:
                run.summary.status, run.summary.error_code, run.summary.error = "FAILED", exc.code, str(exc)
    _json(asdict(run.summary))
    if run.summary.status == "FAILED":
        raise typer.Exit(1)


@frames_app.command("inspect")
@report_errors
def inspect_sample(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str | None, DATASET] = None,
    version: Annotated[str | None, VERSION] = None, path: Annotated[Path | None, PATH] = None,
    storage_root: Annotated[Path | None, STORAGE] = None,
):
    """Decode and report every frame in the approved interval."""
    _consume(_service(ctx, dataset, version, path, storage_root, 30, True).process(sample_id))


@frames_app.command("sample")
@report_errors
def sample_frames(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str | None, DATASET] = None,
    version: Annotated[str | None, VERSION] = None, path: Annotated[Path | None, PATH] = None,
    storage_root: Annotated[Path | None, STORAGE] = None, fps: Annotated[float, FPS] = 30,
    all_frames: Annotated[bool, ALL] = False,
):
    """Apply time-domain sampling and report the result."""
    _consume(_service(ctx, dataset, version, path, storage_root, fps, all_frames).process(sample_id))


@frames_app.command("inspect-file")
@report_errors
def inspect_direct_file(path: Path):
    """Decode a local diagnostic file without treating it as an approved sample."""
    from landmarks.service import inspect_file

    _consume(inspect_file(path))


@frames_app.command("preview")
@report_errors
def preview_sample(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str | None, DATASET] = None,
    version: Annotated[str | None, VERSION] = None, path: Annotated[Path | None, PATH] = None,
    storage_root: Annotated[Path | None, STORAGE] = None, fps: Annotated[float, FPS] = 30,
    all_frames: Annotated[bool, ALL] = False,
):
    """Step forward with Space, Right, or N. Press Q to exit."""
    from landmarks.preview import preview_frames

    _consume(_service(ctx, dataset, version, path, storage_root, fps, all_frames).process(sample_id), preview_frames)


@frames_app.command("contact-sheet")
@report_errors
def contact_sheet(
    ctx: typer.Context, sample_id: str, output: Annotated[Path, typer.Option()],
    dataset: Annotated[str | None, DATASET] = None, version: Annotated[str | None, VERSION] = None,
    path: Annotated[Path | None, PATH] = None, storage_root: Annotated[Path | None, STORAGE] = None,
    fps: Annotated[float, FPS] = 30, all_frames: Annotated[bool, ALL] = False,
    limit: Annotated[int, typer.Option(min=1, max=12)] = 9,
):
    """Write an optional contact sheet of the first selected frames."""
    from landmarks.preview import save_contact_sheet

    run = _service(ctx, dataset, version, path, storage_root, fps, all_frames).process(sample_id)
    _consume(run, lambda processing: save_contact_sheet(processing, output, limit))


@frames_app.command("validate-dataset")
@report_errors
def validate_dataset(
    ctx: typer.Context, dataset: Annotated[str | None, DATASET] = None,
    version: Annotated[str | None, VERSION] = None, path: Annotated[Path | None, PATH] = None,
    storage_root: Annotated[Path | None, STORAGE] = None, fps: Annotated[float, FPS] = 30,
    all_frames: Annotated[bool, ALL] = False,
):
    """Process samples sequentially and report faults without saving arrays."""
    report = _service(ctx, dataset, version, path, storage_root, fps, all_frames).validate_dataset()
    _json({**asdict(report), "processed": report.processed, "failed": report.failed})
    if report.failed or report.dataset_error:
        raise typer.Exit(1)


def mount(app: typer.Typer) -> None:
    app.add_typer(frames_app, name="frames")
