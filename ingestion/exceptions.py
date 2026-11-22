"""Errors that can be reported directly to the capture operator."""


class CaptureError(Exception):
    """A camera, recording, or storage operation failed."""


class CameraError(CaptureError):
    """The camera could not deliver usable frames."""


class RecordingError(CaptureError):
    """A recording could not be encoded, validated, or saved."""


class MediaError(Exception):
    """A media failure with an operator-safe message and stable error code."""

    def __init__(self, code: str, message: str, *, operational: bool = False):
        super().__init__(message)
        self.code = code
        self.operational = operational
