"""Incremental PyAV decoding using frame presentation timestamps."""

import os
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_stream
from ingestion.metadata import local_video
from landmarks.exceptions import (
    EmptyInterval, InvalidInterval, InvalidTimestamp, MissingTimestamp, MissingVideoStream,
    SourceVideoError, UnexpectedEndOfStream, UnsupportedCodec, UnsupportedImageFormat, VideoDecodeError,
)
from landmarks.schemas import DecodedFrame, ProcessingSummary, TimestampIssue

VIDEO_FORMATS = "mov,avi,matroska,webm"
OPEN_OPTIONS = {"protocol_whitelist": "file", "format_whitelist": VIDEO_FORMATS}


def validate_interval(start_ms: int, end_ms: int | None, duration_ms: int | None = None) -> None:
    if (
        type(start_ms) is not int or start_ms < 0
        or (end_ms is not None and (type(end_ms) is not int or end_ms <= start_ms))
        or (duration_ms is not None and (start_ms >= duration_ms or (end_ms is not None and end_ms > duration_ms)))
    ):
        raise InvalidInterval("Use integer millisecond boundaries within the video with 0 ≤ start < end.")


class VideoDecoder:
    """Decode a local video. Exhaust or close the iterator to release the decoder."""

    def __init__(
        self, path: Path, sample_id: str = "", *, start_ms: int = 0, end_ms: int | None = None,
        expected_frames: int | None = None, expected_duration_ms: int | None = None,
        expected_sha256: str | None = None, expected_size: int | None = None,
        rotation_degrees: float | None = None, summary: ProcessingSummary | None = None,
    ):
        validate_interval(start_ms, end_ms, expected_duration_ms)
        self.path = Path(path)
        self.sample_id = sample_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.expected_frames = expected_frames
        self.expected_sha256 = expected_sha256
        self.expected_size = expected_size
        self.summary = summary if summary is not None else ProcessingSummary(sample_id)
        self.summary.source_video_path = str(self.path)
        self.summary.start_ms = start_ms
        self.summary.end_ms = end_ms
        self.summary.rotation_degrees = rotation_degrees
        self.duration_ms: Fraction | None = self._interval_duration()
        self._decoding = False
        self._initial_dimensions = (self.summary.width, self.summary.height)
        self._initial_source_fps = self.summary.source_fps
        self._rotation_degrees = rotation_degrees

    def _interval_duration(self) -> Fraction | None:
        return Fraction(self.end_ms - self.start_ms) if self.end_ms is not None else None

    def decode(self) -> Iterator[DecodedFrame]:
        if self._decoding:
            raise VideoDecodeError("Close the current iterator before decoding this source again.")
        self._decoding = True
        self.duration_ms = self._interval_duration()
        self.summary.source_frames_decoded = self.summary.decoded_frames = 0
        self.summary.source_origin_pts_ms = None
        self.summary.variable_timing = False
        self.summary.timestamp_irregularities.clear()
        self.summary.decoder_warnings.clear()
        self.summary.width, self.summary.height = self._initial_dimensions
        self.summary.source_fps = self._initial_source_fps
        self.summary.rotation_degrees = self._rotation_degrees
        try:
            yield from self._decode_file()
        finally:
            self._decoding = False

    def _decode_file(self) -> Iterator[DecodedFrame]:
        try:
            path = local_video(self.path)
        except MediaError as exc:
            raise SourceVideoError(str(exc)) from exc
        try:
            # A local file object keeps inputs on disk and avoids URL interpretation.
            with path.open("rb") as source:
                initial = os.fstat(source.fileno())
                if self.expected_sha256 is not None:
                    if initial.st_size != self.expected_size or sha256_stream(source) != self.expected_sha256:
                        raise SourceVideoError("The video checksum or size differs from the approved manifest.")
                    source.seek(0)
                with av.open(source, mode="r", options=OPEN_OPTIONS) as container:
                    stream = next((item for item in container.streams.video
                                   if not item.disposition & av.stream.Disposition.attached_pic), None)
                    if stream is None:
                        raise MissingVideoStream("The file has no decodable primary video stream.")
                    if stream.codec_context is None:
                        raise UnsupportedCodec("The primary video stream has no supported decoder.")
                    stream.codec_context.options = {**stream.codec_context.options, "err_detect": "explode"}
                    yield from self._frames(container, stream)
                final = os.fstat(source.fileno())
                if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
                    final.st_size, final.st_mtime_ns, final.st_ctime_ns
                ):
                    raise SourceVideoError("The source video changed during processing.")
        except (av.error.DecoderNotFoundError, av.error.NotImplementedError) as exc:
            raise UnsupportedCodec("PyAV could not decode the source video codec.") from exc
        except av.error.EOFError as exc:
            raise UnexpectedEndOfStream("Video decoding ended unexpectedly.") from exc
        except (av.FFmpegError, OSError) as exc:
            raise VideoDecodeError("PyAV could not open or decode the video. It may be corrupt or unsupported.") from exc

    def _frames(self, container, stream) -> Iterator[DecodedFrame]:
        summary = self.summary
        average_rate = stream.average_rate
        if average_rate and average_rate > 0:
            summary.source_fps = float(average_rate)
        declared_duration = (
            Fraction(stream.duration) * stream.time_base * 1000
            if stream.duration is not None and stream.duration > 0 and stream.time_base else None
        )
        nominal_period = Fraction(1000) / average_rate if average_rate and average_rate > 0 else None
        origin = previous = previous_delta = last_end = None
        for index, frame in enumerate(container.decode(stream)):
            summary.source_frames_decoded += 1
            if frame.is_corrupt:
                raise VideoDecodeError(f"The decoder marked source frame {index} as corrupt.")
            time_base = frame.time_base or stream.time_base
            if frame.pts is None or not time_base:
                summary.timestamp_irregularities.append(TimestampIssue("MISSING_PTS", index, None))
                raise MissingTimestamp(f"Source frame {index} has no presentation timestamp or time base.")
            if time_base < 0:
                raise InvalidTimestamp(f"Source frame {index} has a negative time base.")
            pts = Fraction(frame.pts) * time_base * 1000
            if origin is None:
                origin = pts
                summary.source_origin_pts_ms = origin // 1
                summary.width, summary.height = frame.width, frame.height
            rotation = getattr(frame, "rotation", 0)
            if rotation:
                summary.rotation_degrees = float(rotation)
            if previous is not None:
                delta = pts - previous
                if delta <= 0:
                    summary.timestamp_irregularities.append(TimestampIssue(
                        "DECREASING_PTS" if delta < 0 else "REPEATED_PTS", index, pts // 1, delta // 1,
                    ))
                    if delta < 0:
                        raise InvalidTimestamp(f"Presentation timestamp decreased at source frame {index}.")
                else:
                    tolerance = max(Fraction(1), 2 * time_base * 1000)
                    if previous_delta is not None and abs(delta - previous_delta) > tolerance:
                        summary.variable_timing = True
                    expected = nominal_period or previous_delta
                    if expected is not None and delta > 2 * expected + tolerance:
                        summary.timestamp_irregularities.append(TimestampIssue("SOURCE_GAP", index, pts // 1, delta // 1))
                    previous_delta = delta
            previous = pts
            offset = pts - origin
            if self.end_ms is not None and offset >= self.end_ms:
                break
            duration = frame.duration
            last_end = offset + duration * time_base * 1000 if duration and duration > 0 else None
            if offset < self.start_ms:
                continue
            summary.decoded_frames += 1
            try:
                image = frame.to_ndarray(format="rgb24")
            except (av.FFmpegError, ValueError, TypeError, NotImplementedError) as exc:
                raise UnsupportedImageFormat(f"Cannot convert source frame {index} to RGB24.") from exc
            if image.dtype != np.uint8 or image.shape != (frame.height, frame.width, 3):
                raise UnsupportedImageFormat(f"Source frame {index} did not decode as uint8 RGB.")
            yield DecodedFrame(
                self.sample_id, index, pts, offset - self.start_ms,
                image, frame.width, frame.height, summary.rotation_degrees,
            )
        else:
            expected = self.expected_frames if self.expected_frames is not None else (stream.frames or None)
            if expected is not None and summary.source_frames_decoded < expected:
                raise UnexpectedEndOfStream(
                    f"Decoded {summary.source_frames_decoded} source frames before EOF. Expected {expected}."
                )
            if self.expected_frames is not None and summary.source_frames_decoded != self.expected_frames:
                raise VideoDecodeError("The decoded frame count differs from the source manifest.")
            if last_end is not None and declared_duration is not None:
                # Large disagreements can reveal a truncated stream even when frame count is unknown.
                tolerance = max(Fraction(2), 2 * nominal_period if nominal_period else Fraction(100))
                if last_end + tolerance < declared_duration:
                    raise UnexpectedEndOfStream("The decoded stream ended before its declared video duration.")
            observed_end = last_end if last_end is not None else declared_duration
            if self.end_ms is not None:
                if observed_end is None:
                    raise UnexpectedEndOfStream("EOF arrived before the annotation end could be verified.")
                if observed_end + 1 < self.end_ms:
                    raise UnexpectedEndOfStream("The video ended before the annotated interval ended.")
                if last_end is None:
                    summary.decoder_warnings.append("Used the stream duration to check EOF because the last frame duration was absent.")
            elif observed_end is not None and observed_end > self.start_ms:
                self.duration_ms = observed_end - self.start_ms
            else:
                summary.decoder_warnings.append("The presentation duration is unknown. Effective FPS is unavailable.")

        if not summary.decoded_frames:
            raise EmptyInterval("The selected interval contains no decoded frames.")
