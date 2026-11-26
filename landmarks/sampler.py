"""Deterministic time-grid sampling with one pending frame."""

import math
from collections.abc import Iterable, Iterator
from fractions import Fraction

from landmarks.exceptions import EmptyInterval, InvalidSampleRate, InvalidTimestamp
from landmarks.schemas import DecodedFrame, SamplingGap


class FrameSampler:
    """Assign each frame to its nearest grid slot, with ties going to earlier slots and frames."""

    def __init__(self, fps: float = 30, *, all_frames: bool = False):
        try:
            invalid = type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
        except OverflowError:
            invalid = True
        if invalid:
            raise InvalidSampleRate("Target FPS must be a finite positive number.")
        self.fps = fps
        self.all_frames = all_frames
        self.period_ms = Fraction(1000) / Fraction(str(fps))
        self.gaps: list[SamplingGap] = []

    def sample(
        self, frames: Iterable[DecodedFrame], *, duration_ms: Fraction | None = None,
    ) -> Iterator[DecodedFrame]:
        self.gaps.clear()
        last_slot = math.ceil(duration_ms / self.period_ms) - 1 if duration_ms is not None else None
        slot = -1
        pending = None
        previous = None
        for frame in frames:
            time = frame.relative_timestamp_ms
            if time < 0 or (duration_ms is not None and time >= duration_ms):
                raise InvalidTimestamp("A source frame lies outside the sampling interval.")
            if previous is not None and (
                time < previous.relative_timestamp_ms or frame.timestamp_ms < previous.timestamp_ms
                or frame.frame_index <= previous.frame_index
            ):
                raise InvalidTimestamp("Sampling requires ordered source timestamps and distinct increasing frame indices.")
            previous = frame
            if self.all_frames:
                yield frame
                continue
            index = math.ceil(time / self.period_ms - Fraction(1, 2))
            if index != slot:
                if pending is not None:
                    yield pending
                self._gap(slot + 1, index)
                slot, pending = index, frame
            elif abs(time - slot * self.period_ms) < abs(pending.relative_timestamp_ms - slot * self.period_ms):
                pending = frame
        if previous is None:
            raise EmptyInterval("The selected interval contains no frames to sample.")
        if pending is not None:
            yield pending
            if last_slot is not None:
                self._gap(slot + 1, last_slot + 1)

    def _gap(self, start: int, stop: int) -> None:
        if stop > start:
            self.gaps.append(SamplingGap(
                start, stop - start, (start * self.period_ms) // 1, ((stop - 1) * self.period_ms) // 1,
            ))
