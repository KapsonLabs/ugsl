"""Errors that can be reported directly to the capture operator."""


class CaptureError(Exception):
    """A camera, recording, or storage operation failed."""


class CameraError(CaptureError):
    """The camera could not deliver usable frames."""


class RecordingError(CaptureError):
    """A recording could not be encoded, validated, or saved."""
