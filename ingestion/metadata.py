"""Local-only FFprobe access and media metadata parsing."""

import json
import math
import re
import stat
import subprocess
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from ingestion.exceptions import MediaError
from ingestion.schemas import VideoMetadata

INPUT_OPTIONS = ["-protocol_whitelist", "file", "-format_whitelist", "mov,avi,matroska,webm"]


def local_video(path: Path) -> Path:
    """Reject special files and URL inputs before invoking a media tool."""
    path = Path(path).expanduser()
    try:
        path = path.resolve(strict=True)
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise MediaError("INVALID_FILE", "Select a regular local video file.")
        with path.open("rb"):
            pass
        return path
    except (OSError, RuntimeError, ValueError) as exc:
        raise MediaError("UNREADABLE_FILE", "The local video file is missing or unreadable.") from exc


def media_command(args: list[str], timeout: float, error_code: str, error_message: str) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("MEDIA_TIMEOUT", f"{args[0]} exceeded its time limit.", operational=True) from exc
    except OSError as exc:
        raise MediaError("MEDIA_TOOL_UNAVAILABLE", f"Cannot run {args[0]}. Check that it is installed and on PATH.",
                         operational=True) from exc
    if result.returncode:
        if result.returncode < 0:
            raise MediaError("MEDIA_TOOL_INTERRUPTED", f"{args[0]} stopped unexpectedly.", operational=True)
        raise MediaError(error_code, error_message)
    return result.stdout


def _positive_number(value, convert=float):
    try:
        if isinstance(value, bool):
            return None
        result = convert(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _rate(value):
    return _positive_number(value, lambda item: float(Fraction(item)))


def timestamp_utc(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.utcoffset() is not None:
            return timestamp.astimezone(UTC).isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    raise MediaError("INVALID_TIMESTAMP", "Use an ISO 8601 recording timestamp with an explicit UTC offset.")


def _creation_time(*tags: dict) -> str | None:
    for values in tags:
        if value := values.get("creation_time"):
            try:
                return timestamp_utc(value)
            except MediaError:
                continue
    return None


def _extension(formats: str, tags: dict, path: Path) -> str:
    names = set(formats.split(","))
    if "avi" in names:
        return ".avi"
    if names & {"matroska", "webm"}:
        with path.open("rb") as file:
            header = file.read(4096)
        return ".webm" if re.search(rb"\x42\x82\x84webm", header) else ".mkv"
    if "mov" in names or "mp4" in names:
        brand = tags.get("major_brand", "").strip().lower()
        if brand.startswith(("3gp", "3g2")) or brand in {"m4a", "m4b", "mj2"}:
            raise MediaError("UNSUPPORTED_CONTAINER", "Use an MP4, MOV, AVI, MKV or WEBM video.")
        return ".mov" if brand == "qt" or not brand else ".mp4"
    raise MediaError("UNSUPPORTED_CONTAINER", "Use an MP4, MOV, AVI, MKV or WEBM video.")


class VideoMetadataExtractor:
    def __init__(self, timeout=20):
        self.timeout = timeout

    def _probe(self, path, options):
        output = media_command(
            ["ffprobe", "-v", "error", *INPUT_OPTIONS, *options, "-of", "json", "-i", str(path)],
            self.timeout, "INVALID_CONTAINER", "FFprobe could not read a supported video container.",
        )
        try:
            parsed = json.loads(output)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object.")
            return parsed
        except ValueError as exc:
            raise MediaError("INVALID_PROBE_OUTPUT", "FFprobe returned unreadable metadata.", operational=True) from exc

    def extract(self, path: Path, *, count_frames: bool = True) -> VideoMetadata:
        path = local_video(path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MediaError("UNREADABLE_FILE", "The local video file is missing or unreadable.") from exc
        if not size:
            raise MediaError("EMPTY_FILE", "The video file is empty.")
        video = self._probe(path, ["-select_streams", "V:0", "-show_streams", "-show_format"])
        streams = video.get("streams") or []
        if not streams:
            raise MediaError("NO_VIDEO_STREAM", "The file has no usable video stream.")
        stream = streams[0]
        container = video.get("format") or {}
        tags = container.get("tags") or {}
        stream_tags = stream.get("tags") or {}
        formats = container.get("format_name", "")
        try:
            extension = _extension(formats, tags, path)
        except OSError as exc:
            raise MediaError("MEDIA_IO", "Could not read the video container.", operational=True) from exc
        audio = self._probe(path, ["-select_streams", "a", "-show_entries", "stream=index"])
        frame_count = _positive_number(stream.get("nb_frames"), int)
        if frame_count is None and count_frames:
            counted = self._probe(path, ["-select_streams", "V:0", "-count_frames", "-show_entries", "stream=nb_read_frames"])
            decoded = counted.get("streams") or [{}]
            frame_count = _positive_number(decoded[0].get("nb_read_frames"), int)
        rotation = None
        for value in (*(item.get("rotation") for item in stream.get("side_data_list", [])), stream_tags.get("rotate")):
            try:
                candidate = float(value)
                if math.isfinite(candidate):
                    rotation = candidate
                    break
            except (TypeError, ValueError):
                continue
        width = _positive_number(stream.get("width"), int)
        height = _positive_number(stream.get("height"), int)
        orientation = None
        if width and height:
            display_width, display_height = (height, width) if rotation is not None and rotation % 180 == 90 else (width, height)
            orientation = "SQUARE" if display_width == display_height else ("LANDSCAPE" if display_width > display_height else "PORTRAIT")
        stream_duration = _positive_number(stream.get("duration"))
        return VideoMetadata(
            container_format=formats, extension=extension, video_codec=stream.get("codec_name"),
            pixel_format=stream.get("pix_fmt"), width=width, height=height, rotation=rotation,
            display_orientation=orientation, average_fps=_rate(stream.get("avg_frame_rate")),
            nominal_fps=_rate(stream.get("r_frame_rate")),
            frame_count=frame_count,
            duration_seconds=stream_duration or _positive_number(container.get("duration")),
            stream_duration_seconds=stream_duration,
            bitrate=_positive_number(stream.get("bit_rate"), int) or _positive_number(container.get("bit_rate"), int),
            file_size_bytes=size, has_audio=bool(audio.get("streams")),
            creation_time=_creation_time(stream_tags, tags),
        )
