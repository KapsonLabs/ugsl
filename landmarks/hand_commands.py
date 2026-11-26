"""UGSL hand extraction, inspection, and playback commands."""

from pathlib import Path
from typing import Annotated

import typer

from config import HAND_MODEL_PATH, HandExtractionConfig
from landmarks.commands import _json, finish_extraction
from landmarks.exceptions import FrameProcessingError
from landmarks.service import DecodingService, HandExtractionService
from registry.commands import database, report_errors

hands_app = typer.Typer(help="Extract and inspect MediaPipe hand landmarks.", no_args_is_help=True)
DATASET = typer.Option(help="Registered dataset name.")
VERSION = typer.Option(help="Registered dataset version.")


@hands_app.callback()
def options(
    ctx: typer.Context,
    model: Annotated[Path, typer.Option(envvar="UGSL_HAND_MODEL", help="MediaPipe .task bundle.")] = HAND_MODEL_PATH,
    detection_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    presence_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    tracking_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    mirrored_input: Annotated[bool, typer.Option(help="Swap anatomical slots for stored mirrored pixels.")] = False,
    no_hand_warning_ms: Annotated[int, typer.Option(min=1)] = 1000,
    fps: Annotated[float, typer.Option(help="Target frame sampling rate.")] = 30,
    all_frames: Annotated[bool, typer.Option(help="Use all decoded frames in the interval.")] = False,
):
    try:
        config = HandExtractionConfig(
            model_path=model,
            min_detection_confidence=detection_confidence,
            min_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            mirrored_input=mirrored_input,
            no_hand_warning_ms=no_hand_warning_ms,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Leave the parent registry context in place for the existing database helper.
    ctx.meta["ugsl_hands"] = (config, fps, all_frames)


def _registered(ctx, dataset, version) -> HandExtractionService:
    config, fps, all_frames = ctx.meta["ugsl_hands"]
    return HandExtractionService(
        DecodingService.from_registry(database(ctx), dataset, version, fps=fps, all_frames=all_frames), config,
    )


@hands_app.command("extract")
@report_errors
def extract(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    output: Annotated[Path | None, typer.Option(help="Write optional NPZ and JSON debug output here.")] = None,
):
    """Extract an approved sample and print its JSON report."""
    service = _registered(ctx, dataset, version)
    result = service.process(sample_id)
    if output is not None and result.arrays is not None:
        from landmarks.hand_debug import export_debug

        try:
            export_debug(result, output, check_permission=service.decoding.reader.check_permission)
        except FrameProcessingError as exc:
            service._record_failure(result.report, exc)
    finish_extraction(result)


@hands_app.command("visualize")
@report_errors
def visualize(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
    skeleton_only: Annotated[bool, typer.Option(help="Draw on black instead of the source image.")] = False,
):
    """Play an approved recording with its extracted landmarks. Q or Escape exits."""
    from landmarks.hand_visualizer import HandLandmarkVisualizer

    service = _registered(ctx, dataset, version)
    with HandLandmarkVisualizer(skeleton_only=skeleton_only) as preview:
        result = service.process(sample_id, on_frame=preview.show)
    finish_extraction(result, allow_early_exit=True)


@hands_app.command("inspect")
@report_errors
def inspect_sample(
    ctx: typer.Context, sample_id: str, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION],
):
    """Print metrics and diagnostics for an approved sample."""
    finish_extraction(_registered(ctx, dataset, version).process(sample_id))


@hands_app.command("inspect-file")
@report_errors
def inspect_direct_file(ctx: typer.Context, path: Path):
    """Inspect a diagnostic video without reading or updating registry data."""
    config, fps, all_frames = ctx.meta["ugsl_hands"]
    finish_extraction(HandExtractionService(None, config).inspect_file(path, fps=fps, all_frames=all_frames))


@hands_app.command("inspect-dataset")
@report_errors
def inspect_dataset(ctx: typer.Context, dataset: Annotated[str, DATASET], version: Annotated[str, VERSION]):
    """Print aggregate and per-sample results without exporting landmarks."""
    report = _registered(ctx, dataset, version).inspect_dataset()
    _json(report)
    if report["cancelled"]:
        raise typer.Exit(130)
    if report["failed"] or report["dataset_error"]:
        raise typer.Exit(1)


def mount(app: typer.Typer) -> None:
    app.add_typer(hands_app, name="hands")
