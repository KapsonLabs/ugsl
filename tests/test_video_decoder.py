"""Real synthetic timelines and fault injection for the PyAV decoder."""

from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from landmarks.decoder import VideoDecoder
from landmarks.exceptions import (
    EmptyInterval, InvalidInterval, InvalidTimestamp, MissingTimestamp, MissingVideoStream, SourceVideoError,
    UnexpectedEndOfStream, UnsupportedCodec, UnsupportedImageFormat, VideoDecodeError,
)


def write_video(path, pts, *, fps=30, rotation=0):
    base = Fraction(1, 60_000)
    pixels = np.zeros((72, 96, 3), dtype=np.uint8)
    pixels[..., 0] = 220
    pixels[:20, :30, 1] = 120
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 96, 72, "yuv420p"
        stream.codec_context.time_base = base
        if rotation:
            stream.set_display_rotation(rotation)
        for value in pts:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = value, base
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("timestamped-videos")
    files = {
        fps: write_video(root / f"{fps}fps.mp4", [index * (60_000 // fps) for index in range(fps)], fps=fps)
        for fps in (24, 30, 60)
    }
    files["vfr"] = write_video(root / "vfr.mp4", [60_000 + offset for offset in (0, 2400, 7200, 9600, 14400)], fps=25)
    files["rotated"] = write_video(root / "rotated.mp4", [0, 2000, 4000], rotation=90)
    return files


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_regular_presentation_timestamps_and_rgb(clips, fps):
    decoder = VideoDecoder(clips[fps], "clip")
    frames = list(decoder.decode())
    assert len(frames) == fps
    assert decoder.duration_ms == 1000
    assert [frame.timestamp_ms for frame in frames] == [Fraction(index * 1000, fps) for index in range(fps)]
    assert not decoder.summary.variable_timing
    first = frames[0]
    assert first.sample_id == "clip"
    assert first.image.dtype == np.uint8 and first.image.shape == (72, 96, 3)
    assert (first.width, first.height, first.frame_index) == (96, 72, 0)
    assert first.image[-1, -1, 0] > 200
    assert first.image[-1, -1, 2] < 10


def test_exact_half_open_bounds_and_source_indices(clips):
    decoder = VideoDecoder(clips[30], "clip", start_ms=100, end_ms=200)
    frames = list(decoder.decode())
    assert [frame.frame_index for frame in frames] == [3, 4, 5]
    assert [frame.timestamp_ms for frame in frames] == [100, Fraction(400, 3), Fraction(500, 3)]
    assert [frame.relative_timestamp_ms for frame in frames] == [0, Fraction(100, 3), Fraction(200, 3)]
    assert decoder.summary.decoded_frames == 3 and decoder.summary.source_frames_decoded == 7


def test_variable_rate_with_nonzero_source_origin(clips):
    decoder = VideoDecoder(clips["vfr"], start_ms=40, end_ms=160)
    frames = list(decoder.decode())
    assert [frame.timestamp_ms for frame in frames] == [1040, 1120]
    assert [frame.relative_timestamp_ms for frame in frames] == [0, 80]
    assert decoder.summary.source_origin_pts_ms == 1000
    assert decoder.summary.variable_timing


def test_rotation_is_metadata_and_pixels_keep_coded_resolution(clips):
    decoder = VideoDecoder(clips["rotated"])
    frame = list(decoder.decode())[0]
    assert frame.rotation_degrees == 90
    assert frame.image.shape == (72, 96, 3)
    assert frame.image[5, 5, 1] > frame.image[-1, -1, 1] + 70


@pytest.mark.parametrize("bounds", [(-1, 100), (100, 100), (100, 90), (True, 100), (0, 1001)])
def test_invalid_interval_bounds(clips, bounds):
    with pytest.raises(InvalidInterval):
        VideoDecoder(clips[30], start_ms=bounds[0], end_ms=bounds[1], expected_duration_ms=1000)


def test_empty_interval(clips):
    with pytest.raises(EmptyInterval):
        list(VideoDecoder(clips[30], start_ms=10, end_ms=20).decode())


def test_corrupt_or_missing_source(tmp_path):
    file = tmp_path / "bad.mp4"
    file.write_bytes(b"not a video container")
    with pytest.raises(VideoDecodeError):
        list(VideoDecoder(file).decode())
    with pytest.raises(SourceVideoError):
        list(VideoDecoder(tmp_path / "missing.mp4").decode())


def mock_frame(pts, *, pixels=None, corrupt=False):
    return SimpleNamespace(
        pts=pts, time_base=Fraction(1, 1000), width=4, height=3, duration=40, is_corrupt=corrupt, rotation=0,
        to_ndarray=lambda **_: np.zeros((3, 4, 3), dtype=np.uint8) if pixels is None else pixels,
    )


class MockContainer:
    def __init__(self, frames, *, count=0, cover_art=False):
        self.frames = frames
        stream = SimpleNamespace(
            duration=None, time_base=Fraction(1, 1000), average_rate=25, frames=count,
            disposition=av.stream.Disposition.attached_pic if cover_art else 0,
            codec_context=SimpleNamespace(options={}),
        )
        self.streams = SimpleNamespace(video=(stream,))
        self.closed = False
        self.source = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def decode(self, _):
        yield from self.frames


@pytest.mark.parametrize(("frames", "error", "issue"), [
    ([mock_frame(None)], MissingTimestamp, "MISSING_PTS"),
    ([mock_frame(40), mock_frame(30)], InvalidTimestamp, "DECREASING_PTS"),
    ([mock_frame(0, pixels=np.zeros((3, 4, 3), dtype=np.float32))], UnsupportedImageFormat, None),
    ([mock_frame(0, corrupt=True)], VideoDecodeError, None),
])
def test_invalid_decoded_frames_are_reported_and_resources_close(clips, monkeypatch, frames, error, issue):
    container = MockContainer(frames)

    def opener(source, **_):
        container.source = source
        return container

    monkeypatch.setattr(av, "open", opener)
    decoder = VideoDecoder(clips[30])
    with pytest.raises(error):
        list(decoder.decode())
    assert container.closed and container.source.closed
    if issue:
        assert decoder.summary.timestamp_irregularities[-1].kind == issue


def test_decoder_releases_resources_on_early_close(clips, monkeypatch):
    container = MockContainer([mock_frame(0), mock_frame(40)])
    monkeypatch.setattr(av, "open", lambda *_, **__: container)
    iterator = VideoDecoder(clips[30]).decode()
    assert next(iterator).timestamp_ms == 0 and not container.closed
    iterator.close()
    assert container.closed


def test_decoder_can_be_reused_after_closing_an_iterator(clips):
    decoder = VideoDecoder(clips[30], expected_frames=30)
    first = decoder.decode()
    assert next(first).frame_index == 0
    first.close()
    assert len(list(decoder.decode())) == decoder.summary.decoded_frames == 30


def test_no_primary_video_and_unexpected_eof(clips, monkeypatch):
    container = MockContainer([mock_frame(0)], cover_art=True)
    monkeypatch.setattr(av, "open", lambda *_, **__: container)
    with pytest.raises(MissingVideoStream):
        list(VideoDecoder(clips[30]).decode())
    container = MockContainer([mock_frame(0), mock_frame(40)], count=3)
    with pytest.raises(UnexpectedEndOfStream, match="Expected 3"):
        list(VideoDecoder(clips[30]).decode())


def test_unsupported_codec_is_a_distinct_error(clips, monkeypatch):
    def opener(*_, **__):
        raise av.error.DecoderNotFoundError(0, "missing")

    monkeypatch.setattr(av, "open", opener)
    with pytest.raises(UnsupportedCodec):
        list(VideoDecoder(clips[30]).decode())


def test_repeated_pts_and_large_source_gaps_are_reported(clips, monkeypatch):
    container = MockContainer([mock_frame(0), mock_frame(40), mock_frame(40), mock_frame(200)])
    monkeypatch.setattr(av, "open", lambda *_, **__: container)
    decoder = VideoDecoder(clips[30])
    assert [frame.timestamp_ms for frame in decoder.decode()] == [0, 40, 40, 200]
    assert [issue.kind for issue in decoder.summary.timestamp_irregularities] == ["REPEATED_PTS", "SOURCE_GAP"]
