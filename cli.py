"""Capture and manage the UGSL training dataset."""

import signal
from pathlib import Path
from typing import Annotated

import typer

from config import DATASET_ROOT, REGISTRY_PATH, CaptureConfig, validate_identifier
from ingestion.capture import CameraCapture
from ingestion.exceptions import CaptureError
from registry import commands
from registry.database import RegistryError
from services.recording_service import capture_registered

app = typer.Typer(help="Capture and manage the UGSL training dataset.", no_args_is_help=True)
DEFAULTS = CaptureConfig()
commands.mount(app)


@app.callback()
def settings(
    ctx: typer.Context,
    db: Annotated[Path, typer.Option(envvar="UGSL_REGISTRY_DB", help="SQLite registry path.")] = REGISTRY_PATH,
    dataset_root: Annotated[Path, typer.Option(envvar="UGSL_DATASET_ROOT", help="Raw training dataset directory.")] = DATASET_ROOT,
):
    ctx.obj = commands.RegistryOptions(db, dataset_root.expanduser())


def _config(**settings) -> CaptureConfig:
    try:
        return CaptureConfig(**settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def preview(
    camera: Annotated[int, typer.Option(help="Camera device index.")] = DEFAULTS.camera_index,
    source: Annotated[str | None, typer.Option(help="Remote camera stream URL.")] = None,
) -> None:
    """Open a live preview. Press Q in the camera window to quit."""
    device = CameraCapture(_config(camera_index=camera, source=source))
    try:
        device.preview()
    except CaptureError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def capture(
    ctx: typer.Context,
    sign: Annotated[str, typer.Option(help="Registered sign ID or a standalone sign label.")],
    signer: Annotated[str | None, typer.Option(help="Signer label for standalone test captures.")] = None,
    session: Annotated[str | None, typer.Option(help="Recording session for registered captures.")] = None,
    source: Annotated[str | None, typer.Option(help="Remote camera stream URL.")] = None,
    camera: Annotated[int, typer.Option(help="Camera device index.")] = DEFAULTS.camera_index,
    width: Annotated[int, typer.Option(help="Requested width in pixels.")] = DEFAULTS.width,
    height: Annotated[int, typer.Option(help="Requested height in pixels.")] = DEFAULTS.height,
    fps: Annotated[float, typer.Option(help="Requested camera frame rate.")] = DEFAULTS.fps,
    duration: Annotated[float, typer.Option(help="Recording duration in seconds.")] = DEFAULTS.duration,
    countdown: Annotated[float, typer.Option(help="Countdown in seconds. Zero starts immediately.")] = DEFAULTS.countdown,
    repetitions: Annotated[int, typer.Option(help="Number of accepted samples to collect.")] = DEFAULTS.repetitions,
) -> None:
    """Record, replay and accept labelled video samples."""
    if (signer is None) == (session is None):
        raise typer.BadParameter("Provide --session for registered capture or --signer for standalone test capture.")
    if signer is not None:
        for value, label, option in ((sign, "Sign label", "--sign"), (signer, "Signer identifier", "--signer")):
            try:
                validate_identifier(value, label)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint=option) from exc
    config = _config(
        camera_index=camera, width=width, height=height, fps=fps,
        duration=duration, countdown=countdown, repetitions=repetitions, source=source,
    )
    root = ctx.obj.dataset_root
    try:
        device = CameraCapture(config)
        if session is not None:
            accepted = capture_registered(commands.database(ctx), device, root, sign_id=sign, session_id=session)
        else:
            standalone = root.resolve().parent / "standalone"
            if standalone.resolve().is_relative_to(root.resolve()):
                raise RegistryError("Standalone captures require a directory outside the raw training dataset.")
            accepted = device.capture(sign, signer, standalone)
    except (CaptureError, RegistryError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    label = "Accepted and registered" if session is not None else "Accepted standalone test"
    typer.echo(f"{label} {len(accepted)} of {repetitions} samples.")
    for sample in accepted:
        typer.echo(str(root / sample.file_path) if session is not None else sample.file_path)


def _terminate(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    app()
