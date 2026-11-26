"""Exact time-grid decisions independent of image content or nominal FPS."""

from fractions import Fraction

import numpy as np
import pytest

from landmarks.decoder import VideoDecoder
from landmarks.exceptions import EmptyInterval, InvalidSampleRate, InvalidTimestamp
from landmarks.sampler import FrameSampler
from landmarks.schemas import DecodedFrame
from tests.test_video_decoder import clips


def make_frames(times):
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    return [DecodedFrame("test", index, Fraction(time) + 1000, Fraction(time), image, 1, 1)
            for index, time in enumerate(times)]


@pytest.mark.parametrize(("source_fps", "selected"), [(24, 24), (30, 30), (60, 30)])
def test_rate_conversion(clips, source_fps, selected):
    sampler = FrameSampler(30)
    output = list(sampler.sample(VideoDecoder(clips[source_fps]).decode(), duration_ms=Fraction(1000)))
    assert len(output) == selected
    assert len({frame.frame_index for frame in output}) == selected
    if source_fps == 60:
        assert [frame.frame_index for frame in output] == list(range(0, 60, 2))
    else:
        assert [frame.frame_index for frame in output] == list(range(source_fps))


def test_ties_gaps_and_determinism():
    frames = make_frames([10, 20, 80, 120, 420, 430])
    sampler = FrameSampler(10)
    results = [list(sampler.sample(frames, duration_ms=Fraction(500))) for _ in range(2)]
    assert [[frame.frame_index for frame in result] for result in results] == [[0, 2, 4], [0, 2, 4]]
    gap, = sampler.gaps
    assert (gap.first_grid_index, gap.missing_slots, gap.first_grid_timestamp_ms, gap.last_grid_timestamp_ms) == (2, 2, 200, 300)


def test_half_grid_tie_goes_to_earlier_slot_and_empty_slots_are_reported():
    sampler = FrameSampler(10)
    result = list(sampler.sample(make_frames([50, 250]), duration_ms=Fraction(400)))
    assert [frame.relative_timestamp_ms for frame in result] == [50, 250]
    assert [gap.first_grid_index for gap in sampler.gaps] == [1, 3]


def test_in_interval_tail_can_use_the_nearest_tick_at_the_end():
    sampler = FrameSampler(10)
    output = list(sampler.sample(make_frames([90, 195]), duration_ms=Fraction(200)))
    assert [frame.relative_timestamp_ms for frame in output] == [90, 195]
    assert sampler.gaps[0].first_grid_index == 0


def test_all_frames_bypasses_grid_and_preserves_source_timestamps():
    frames = make_frames([20, 45, 45, 150, 320])
    sampler = FrameSampler(10, all_frames=True)
    assert list(sampler.sample(frames, duration_ms=Fraction(400))) == frames
    assert not sampler.gaps


def test_sampling_streams_and_looks_ahead_one_slot():
    read = []
    frames = make_frames([0, 30, 110, 210])

    def source():
        for frame in frames:
            read.append(frame.frame_index)
            yield frame

    output = FrameSampler(10).sample(source(), duration_ms=Fraction(300))
    assert next(output) is frames[0] and read == [0, 1, 2]
    output.close()


@pytest.mark.parametrize("fps", [0, -1, False, float("inf"), float("nan"), "30"])
def test_invalid_fps(fps):
    with pytest.raises(InvalidSampleRate):
        FrameSampler(fps)


def test_empty_or_unordered_input():
    sampler = FrameSampler()
    with pytest.raises(EmptyInterval):
        list(sampler.sample([]))
    with pytest.raises(InvalidTimestamp):
        list(sampler.sample(make_frames([90, 80])))
