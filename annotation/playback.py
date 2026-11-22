"""Source presentation timestamps and sequential OpenCV playback."""

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import cv2

from ingestion.metadata import INPUT_OPTIONS, local_video, media_command
from registry.database import RegistryError

ARROWS = {
    65361: "LEFT", 2424832: "LEFT", 63234: "LEFT", 2: "LEFT",
    65363: "RIGHT", 2555904: "RIGHT", 63235: "RIGHT", 3: "RIGHT",
}


def key_name(code: int) -> str:
    if code < 0:
        return ""
    return ARROWS.get(code, chr(code & 0xFF).lower())


def _number(value) -> Decimal | None:
    try:
        number = Decimal(str(value))
        if number.is_finite():
            return number
    except InvalidOperation:
        pass
    return None


@dataclass(frozen=True)
class Timeline:
    timestamps_ms: tuple[float, ...]
    duration_ms: int

    def seek(self, milliseconds: float) -> int:
        return min(bisect_left(self.timestamps_ms, milliseconds), len(self.timestamps_ms) - 1)

    def start_marker(self, frame_index: int) -> int:
        return min(self.duration_ms, round(self.timestamps_ms[frame_index]))

    def end_marker(self, frame_index: int) -> int:
        return self.duration_ms if frame_index == len(self.timestamps_ms) - 1 else self.start_marker(frame_index)


def probe_timeline(path: Path, *, timeout: float = 60) -> Timeline:
    path = local_video(path)
    output = media_command(
        ["ffprobe", "-v", "error", *INPUT_OPTIONS, "-select_streams", "V:0", "-show_frames",
         "-show_streams", "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time:stream=duration",
         "-of", "json", "-i", str(path)],
        timeout, "PLAYBACK_TIMELINE", "Could not read the source video's presentation timestamps.",
    )
    try:
        payload = json.loads(output)
        frames = payload["frames"]
        stamps = [_number(frame.get("best_effort_timestamp_time")) for frame in frames]
        if not stamps or any(stamp is None for stamp in stamps):
            raise ValueError
        if any(later < earlier for earlier, later in zip(stamps, stamps[1:])):
            raise ValueError
        offsets = tuple(float((stamp - stamps[0]) * 1000) for stamp in stamps)
        if not all(math.isfinite(offset) for offset in offsets):
            raise ValueError
        streams = payload.get("streams") or [{}]
        stream_duration = _number(streams[0].get("duration"))
        last_duration = _number(frames[-1].get("duration_time") or frames[-1].get("pkt_duration_time"))
        if stream_duration and stream_duration * 1000 > offsets[-1]:
            end = int(stream_duration * 1000)
        elif last_duration and last_duration > 0:
            end = int((stamps[-1] - stamps[0] + last_duration) * 1000)
        else:
            raise ValueError
        if end <= 0:
            raise ValueError
        return Timeline(offsets, end)
    except (InvalidOperation, TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
        raise RegistryError("The video has no reliable presentation timeline for annotation playback.") from exc


class FramePlayer:
    """Retain a single frame and restart decoding when stepping backwards."""

    def __init__(self, path: Path, timeline: Timeline):
        self.path = path
        self.timeline = timeline
        self.capture = None
        self.index = -1
        self.frame = None

    def __enter__(self):
        self._restart()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _restart(self) -> None:
        self.close()
        try:
            self.capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
            if self.capture.isOpened():
                self.index = -1
                self.frame = None
                return
        except cv2.error as exc:
            self.close()
            raise RegistryError("OpenCV could not initialize video playback.") from exc
        self.close()
        raise RegistryError("OpenCV could not open the video for playback.")

    def at(self, index: int):
        if not 0 <= index < len(self.timeline.timestamps_ms):
            raise RegistryError("The selected video frame is outside the playback timeline.")
        if index < self.index:
            self._restart()
        try:
            while self.index < index:
                ok, frame = self.capture.read()
                if not ok or frame is None or not frame.size:
                    raise RegistryError("OpenCV ended playback before the frame indicated by FFprobe.")
                self.index += 1
                self.frame = frame
        except cv2.error as exc:
            raise RegistryError("OpenCV could not decode a playback frame.") from exc
        return self.frame
