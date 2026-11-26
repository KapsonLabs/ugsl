"""Typer commands for independent pose extraction and diagnostics."""

from pathlib import Path
from typing import Annotated

import typer

from config import PoseExtractionConfig
from landmarks.commands import DATASET, VERSION, _json, finish_extraction
from landmarks.exceptions import FrameProcessingError
from landmarks.service import DecodingService, PoseExtractionService
from registry.commands import database, report_errors

pose_app = typer.Typer(help="Extract and inspect MediaPipe body pose landmarks.", no_args_is_help=True)
MODEL_PATH = typer.Option(envvar="UGSL_POSE_MODEL", help="Local Pose Landmarker .task bundle.")
MULTIPLE = typer.Option(help="Flag possible multiple participants based on human review.")


@pose_app.callback()
def options(
    ctx: typer.Context,
    model_variant: Annotated[str | None, typer.Option(help="Select lite, full, or heavy.")] = None,
    detection_confidence: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
    presence_confidence: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
    tracking_confidence: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
    visibility_threshold: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
    presence_threshold: Annotated[float, typer.Option(min=0, max=1)] = 0.5,
    timestamp_gap_ms: Annotated[int, typer.Option(min=1)] = 250,
    displacement_threshold: Annotated[float, typer.Option(help="Diagnostic jump threshold in normalized XY.")] = 0.25,
    fps: Annotated[float, typer.Option(help="Target frame sample rate, as in the hands group.")] = 30,
    all_frames: Annotated[bool, typer.Option(help="Use all decoded frames in the selected interval.")] = False,
):
    ctx.meta["ugsl_pose"] = ({
        "model_variant": model_variant,
        "min_detection_confidence": detection_confidence,
        "min_presence_confidence": presence_confidence,
        "min_tracking_confidence": tracking_confidence,
        "visibility_threshold": visibility_threshold,
        "presence_threshold": presence_threshold,
        "timestamp_gap_ms": timestamp_gap_ms,
        "displacement_threshold": displacement_threshold,
    }, fps, all_frames)


def _config(ctx, model_path):
    settings, fps, all_frames = ctx.meta["ugsl_pose"]
    try:
        return PoseExtractionConfig(model_path=model_path, **settings), fps, all_frames
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _service(ctx, dataset, version, model_path, flagged_samples=()):
    config, fps, all_frames = _config(ctx, model_path)
    decoding = DecodingService.from_registry(database(ctx), dataset, version, fps=fps, all_frames=all_frames)
    return PoseExtractionService(decoding, config, multiple_participant_samples=flagged_samples)


@pose_app.command("extract")
@report_errors
def extract(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    model_path: Annotated[Path | None, MODEL_PATH] = None,
    output: Annotated[Path | None, typer.Option(help="Write optional NPZ and JSON output.")] = None,
    multiple_participants: Annotated[bool, MULTIPLE] = False,
):
    """Extract an approved sample and optionally write debug artifacts."""
    service = _service(ctx, dataset, version, model_path, (sample_id,) if multiple_participants else ())
    result = service.process(sample_id)
    if output is not None and result.arrays is not None:
        from landmarks.pose_debug import export_debug

        try:
            export_debug(result, output, check_permission=service.decoding.reader.check_permission)
        except FrameProcessingError as exc:
            service._record_failure(result.report, exc)
    finish_extraction(result)


@pose_app.command("inspect")
@report_errors
def inspect_sample(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    model_path: Annotated[Path | None, MODEL_PATH] = None,
    multiple_participants: Annotated[bool, MULTIPLE] = False,
):
    """Print extraction metrics and diagnostics for an approved sample."""
    service = _service(ctx, dataset, version, model_path, (sample_id,) if multiple_participants else ())
    finish_extraction(service.process(sample_id))


@pose_app.command("visualize")
@report_errors
def visualize(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    model_path: Annotated[Path | None, MODEL_PATH] = None,
    full_body: Annotated[bool, typer.Option(help="Show landmarks below the hips as well.")] = False,
    skeleton_only: Annotated[bool, typer.Option(help="Use a black canvas.")] = False,
    show_indices: Annotated[bool, typer.Option(help="Draw MediaPipe landmark indices.")] = False,
    show_visibility: Annotated[bool, typer.Option(help="Draw per-landmark visibility scores.")] = False,
    mirror_preview: Annotated[bool, typer.Option(help="Mirror playback pixels only.")] = False,
    multiple_participants: Annotated[bool, MULTIPLE] = False,
):
    """Play an approved recording with a pose overlay. Q or Escape exits."""
    from landmarks.pose_visualizer import PoseLandmarkVisualizer

    service = _service(ctx, dataset, version, model_path, (sample_id,) if multiple_participants else ())
    with PoseLandmarkVisualizer(
        model_variant=service.config.model_variant, full_body=full_body, skeleton_only=skeleton_only,
        show_indices=show_indices, show_visibility=show_visibility,
        visibility_threshold=service.config.visibility_threshold, mirror_preview=mirror_preview,
    ) as preview:
        result = service.process(sample_id, on_frame=preview.show)
    finish_extraction(result, allow_early_exit=True)


@pose_app.command("inspect-file")
@report_errors
def inspect_direct_file(
    ctx: typer.Context, path: Path, model_path: Annotated[Path | None, MODEL_PATH] = None,
    multiple_participants: Annotated[bool, MULTIPLE] = False,
):
    """Inspect a diagnostic video outside the approval workflow."""
    config, fps, all_frames = _config(ctx, model_path)
    service = PoseExtractionService(None, config, multiple_participant_samples=(path.stem,) if multiple_participants else ())
    finish_extraction(service.inspect_file(path, fps=fps, all_frames=all_frames))


@pose_app.command("inspect-dataset")
@report_errors
def inspect_dataset(
    ctx: typer.Context, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    model_path: Annotated[Path | None, MODEL_PATH] = None,
    multiple_participant_sample: Annotated[list[str] | None, typer.Option(
        help="Flag a sample from metadata or human review. Repeat for multiple sample IDs.",
    )] = None,
):
    """Print a batch report with an individual outcome for each sample."""
    report = _service(ctx, dataset, version, model_path, multiple_participant_sample or ()).inspect_dataset()
    _json(report)
    if report["cancelled"]:
        raise typer.Exit(130)
    if report["failed"] or report["dataset_error"]:
        raise typer.Exit(1)


def mount(app: typer.Typer) -> None:
    app.add_typer(pose_app, name="pose")
