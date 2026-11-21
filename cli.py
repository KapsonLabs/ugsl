"""Phase 1.1 camera capture commands."""

import signal
from typing import Annotated

import typer

from config import CaptureConfig, validate_identifier
from ingestion.capture import CameraCapture
from ingestion.exceptions import CaptureError

app = typer.Typer(help="Preview a camera and collect UGSL training samples.", no_args_is_help=True)
DEFAULTS = CaptureConfig()


def _config(**settings) -> CaptureConfig:
    try:
        return CaptureConfig(**settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def preview(
    camera: Annotated[int, typer.Option(help="Camera device index.")] = DEFAULTS.camera_index,
) -> None:
    """Open a live preview. Press Q in the camera window to quit."""
    device = CameraCapture(_config(camera_index=camera))
    try:
        device.preview()
    except CaptureError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def capture(
    sign: Annotated[str, typer.Option(help="Sign label or gloss.")],
    signer: Annotated[str, typer.Option(help="Identifier for the signer.")],
    camera: Annotated[int, typer.Option(help="Camera device index.")] = DEFAULTS.camera_index,
    width: Annotated[int, typer.Option(help="Requested width in pixels.")] = DEFAULTS.width,
    height: Annotated[int, typer.Option(help="Requested height in pixels.")] = DEFAULTS.height,
    fps: Annotated[float, typer.Option(help="Requested camera frame rate.")] = DEFAULTS.fps,
    duration: Annotated[float, typer.Option(help="Recording duration in seconds.")] = DEFAULTS.duration,
    countdown: Annotated[float, typer.Option(help="Countdown in seconds. Zero starts immediately.")] = DEFAULTS.countdown,
    repetitions: Annotated[int, typer.Option(help="Number of accepted samples to collect.")] = DEFAULTS.repetitions,
) -> None:
    """Record, replay and accept labelled video samples."""
    for value, label, option in ((sign, "Sign label", "--sign"), (signer, "Signer identifier", "--signer")):
        try:
            validate_identifier(value, label)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint=option) from exc
    config = _config(
        camera_index=camera, width=width, height=height, fps=fps,
        duration=duration, countdown=countdown, repetitions=repetitions,
    )
    try:
        accepted = CameraCapture(config).capture(sign, signer)
    except CaptureError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Accepted {len(accepted)} of {repetitions} samples.")
    for sample in accepted:
        typer.echo(sample.file_path)


def _terminate(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    app()
