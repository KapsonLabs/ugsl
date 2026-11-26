"""Failures reported by the decoding and sampling workflow."""

from registry.database import RegistryError


class FrameProcessingError(RegistryError):
    code = "PROCESSING_FAILED"


class SourceVideoError(FrameProcessingError):
    code = "INVALID_SOURCE_VIDEO"


class MissingVideoStream(FrameProcessingError):
    code = "MISSING_VIDEO_STREAM"


class UnsupportedCodec(FrameProcessingError):
    code = "UNSUPPORTED_CODEC"


class VideoDecodeError(FrameProcessingError):
    code = "DECODE_FAILED"


class UnexpectedEndOfStream(VideoDecodeError):
    code = "UNEXPECTED_END_OF_STREAM"


class MissingTimestamp(FrameProcessingError):
    code = "MISSING_TIMESTAMP"


class InvalidTimestamp(FrameProcessingError):
    code = "INVALID_TIMESTAMP"


class InvalidInterval(FrameProcessingError):
    code = "INVALID_INTERVAL"


class EmptyInterval(FrameProcessingError):
    code = "EMPTY_INTERVAL"


class InvalidSampleRate(FrameProcessingError):
    code = "INVALID_TARGET_FPS"


class UnsupportedImageFormat(FrameProcessingError):
    code = "UNSUPPORTED_IMAGE_FORMAT"


class SampleNotFound(FrameProcessingError):
    code = "SAMPLE_NOT_FOUND"


class DatasetAccessError(FrameProcessingError):
    code = "DATASET_ACCESS_DENIED"


class HandModelError(FrameProcessingError):
    code = "HAND_MODEL_ERROR"


class HandInferenceError(FrameProcessingError):
    code = "HAND_INFERENCE_FAILED"


class HandOutputError(FrameProcessingError):
    code = "HAND_OUTPUT_FAILED"
