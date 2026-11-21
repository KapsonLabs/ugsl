"""Headless tests for the public camera session and CLI."""

import errno
import json
from collections import defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import app
from config import CaptureConfig, validate_identifier
from ingestion import capture
from ingestion.capture import CameraCapture, State
from ingestion.exceptions import CameraError, RecordingError


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"camera_index": -1}, "Camera index"),
        ({"camera_index": 2**31}, "Camera index"),
        ({"width": 0}, "Width"),
        ({"height": 2.5}, "Height"),
        ({"fps": float("nan")}, "FPS"),
        ({"fps": 0}, "FPS"),
        ({"duration": float("inf")}, "Duration"),
        ({"duration": 0}, "Duration"),
        ({"countdown": -1}, "Countdown"),
        ({"repetitions": 0}, "Repetitions"),
        ({"codec": "mp4"}, "Codec"),
    ],
)
def test_configuration_validation(settings, message):
    with pytest.raises(ValueError, match=message):
        CaptureConfig(**settings)


@pytest.mark.parametrize("value", ["", "../WATER", "a/b", "a\\b", ".", "a b", "á", "x" * 65])
def test_invalid_path_components(value):
    with pytest.raises(ValueError, match="Sign label"):
        validate_identifier(value, "Sign label")


class FakeCamera:
    def __init__(self, rig):
        self.rig = rig
        self.settings = {}
        self.released = False

    def isOpened(self):
        return self.rig.camera_open

    def set(self, property_id, value):
        self.settings[property_id] = value
        return False  # The backend ignores requests and delivers its own mode.

    def read(self):
        if self.rig.state == State.RECORDING and self.rig.record_reads:
            if not self.rig.record_reads.popleft():
                return False, None
        return True, self.rig.original.copy()

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, rig, name, fps, resolution):
        self.rig = rig
        self.path = Path(name)
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.path.name != self.rig.fail_writer

    def write(self, frame):
        if self.rig.write_error:
            raise self.rig.write_error
        self.frames.append(frame.copy())

    def release(self):
        self.released = True
        if self.isOpened() and self.frames and not self.rig.empty_video:
            self.path.write_bytes(b"mock mp4")


class FakeDecoder:
    def __init__(self, rig, writer):
        self.writer = writer
        self.frames = iter(writer.frames[:-1] if rig.truncate_video else writer.frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, _):
        return self.writer.fps * 1.001  # Exercise reading back the actual encoded FPS.

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.released = True


class Rig:
    def __init__(self, monkeypatch):
        self.clock = 0.0
        self.state = State.READY
        self.states = []
        self.keys = defaultdict(deque)
        self.original = np.full((120, 160, 3), 42, np.uint8)
        self.record_reads = deque()
        self.camera_open = True
        self.camera = FakeCamera(self)
        self.writers = []
        self.decoders = []
        self.fail_writer = None
        self.write_error = None
        self.empty_video = False
        self.truncate_video = False
        self.windows_closed = []
        self.previews = []

        monkeypatch.setattr(capture.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(capture.cv2, "VideoCapture", self.open_capture)
        monkeypatch.setattr(capture.cv2, "VideoWriter", self.open_writer)
        monkeypatch.setattr(capture.cv2, "waitKey", self.wait_key)
        monkeypatch.setattr(capture.cv2, "namedWindow", lambda *_: None)
        monkeypatch.setattr(capture.cv2, "destroyWindow", self.windows_closed.append)
        monkeypatch.setattr(capture.cv2, "imshow", lambda _, frame: self.previews.append(frame))

        show = capture._Session._show

        def observe(session, frame, **kwargs):
            self.state = session.state
            if not self.states or self.states[-1] != self.state:
                self.states.append(self.state)
            show(session, frame, **kwargs)

        monkeypatch.setattr(capture._Session, "_show", observe)

    def open_capture(self, device):
        if isinstance(device, int):
            return self.camera
        decoder = FakeDecoder(self, self.writers[-1])
        self.decoders.append(decoder)
        return decoder

    def open_writer(self, name, _, fps, resolution):
        writer = FakeWriter(self, name, fps, resolution)
        self.writers.append(writer)
        return writer

    def wait_key(self, _):
        self.clock += 0.125
        assert self.clock < 30, "Test did not drive the session to completion."
        keys = self.keys[self.state]
        key = keys.popleft() if keys else None
        return -1 if key is None else ord(key)

    def assert_clean(self, root):
        assert self.camera.released
        assert self.windows_closed == [capture.WINDOW]
        assert all(writer.released for writer in self.writers)
        assert all(decoder.released for decoder in self.decoders)
        assert not list(root.parent.glob(".capture-*"))


@pytest.fixture
def rig(monkeypatch):
    return Rig(monkeypatch)


@pytest.fixture
def root(tmp_path):
    return tmp_path / "datasets" / "raw"


SETTINGS = CaptureConfig(duration=0.5, countdown=0.25)


def run(root, settings=SETTINGS):
    return CameraCapture(settings).capture("WATER", "SIGNER001", root)


def test_camera_preview_requests_defaults_and_draws_on_a_copy(rig):
    rig.keys[State.READY].extend([" ", "Q"])
    camera = CameraCapture()
    camera.preview()
    assert camera.resolution == (160, 120)
    assert rig.camera.settings == {
        capture.cv2.CAP_PROP_FRAME_WIDTH: 1280,
        capture.cv2.CAP_PROP_FRAME_HEIGHT: 720,
        capture.cv2.CAP_PROP_FPS: 30,
    }
    assert rig.previews and not np.array_equal(rig.previews[0], rig.original)
    assert not rig.writers
    assert rig.camera.released
    assert rig.windows_closed == [capture.WINDOW]


def test_review_replay_reject_accept_and_repetitions(rig, root):
    rig.keys[State.READY].extend(["a", " ", " ", " "])
    rig.keys[State.COUNTDOWN].append("r")
    rig.keys[State.RECORDING].append("p")
    rig.keys[State.REVIEW].extend([" ", "P", "R", "A", "A"])
    rig.record_reads.extend([True, True, True, True, True, False, True, True])

    samples = run(root, replace(SETTINGS, repetitions=2))

    assert rig.states == [
        State.READY, State.COUNTDOWN, State.RECORDING, State.REVIEW, State.PLAYBACK, State.REVIEW,
        State.READY, State.COUNTDOWN, State.RECORDING, State.REVIEW,
        State.READY, State.COUNTDOWN, State.RECORDING, State.REVIEW,
    ]
    assert len(rig.decoders) == 4  # Three validations and one replay.
    assert [sample.repetition for sample in samples] == [1, 2]
    assert len({sample.sample_id for sample in samples}) == 2
    for sample in samples:
        assert UUID(sample.sample_id).version == 4
        path = root / "WATER" / "SIGNER001" / f"{sample.sample_id}.mp4"
        assert Path(sample.file_path) == path
        assert path.stat().st_size == sample.file_size_bytes
        assert json.loads(path.with_suffix(".json").read_text()) == asdict(sample)
        assert datetime.fromisoformat(sample.recorded_at).utcoffset() == timedelta(0)
        assert (sample.width, sample.height) == (160, 120)
        assert sample.actual_duration_seconds == pytest.approx(0.5)
        assert sample.requested_duration_seconds == 0.5
        assert sample.requested_fps == 30
        assert sample.measured_fps == pytest.approx(sample.frame_count / sample.actual_duration_seconds)
        assert sample.encoded_fps == pytest.approx(sample.measured_fps * 1.001)
    assert samples[0].failed_frame_reads == 1
    assert samples[0].frame_count == 3
    assert len(list(root.rglob("*.mp4"))) == len(list(root.rglob("*.json"))) == 2
    assert all(np.array_equal(frame, rig.original) for writer in rig.writers for frame in writer.frames)
    rig.assert_clean(root)


@pytest.mark.parametrize("state", [State.COUNTDOWN, State.RECORDING, State.REVIEW])
def test_escape_discards_only_current_take(rig, root, state):
    rig.keys[State.READY].extend([" ", " "])
    rig.keys[state].append("\x1b")
    rig.keys[State.REVIEW].append("a")
    samples = run(root)
    assert len(samples) == samples[0].repetition == 1
    assert len(list(root.rglob("*.mp4"))) == 1
    rig.assert_clean(root)


def test_escape_from_playback_returns_to_review(rig, root):
    rig.keys[State.READY].append(" ")
    rig.keys[State.REVIEW].extend(["p", "a"])
    rig.keys[State.PLAYBACK].append("\x1b")
    assert len(run(root)) == 1
    assert rig.states[-3:] == [State.REVIEW, State.PLAYBACK, State.REVIEW]
    rig.assert_clean(root)


def test_quit_preserves_accepted_samples_and_discards_pending_video(rig, root):
    rig.keys[State.READY].extend([" ", " "])
    rig.keys[State.REVIEW].extend(["a", "p"])
    rig.keys[State.PLAYBACK].append("q")
    assert len(run(root, replace(SETTINGS, repetitions=3))) == 1
    assert len(list(root.rglob("*.mp4"))) == len(list(root.rglob("*.json"))) == 1
    rig.assert_clean(root)


def test_camera_unavailable_releases_device(rig):
    rig.camera_open = False
    with pytest.raises(CameraError, match="Cannot open camera"):
        CameraCapture().preview()
    assert rig.camera.released
    assert not rig.windows_closed


def test_camera_disconnection_discards_partial_take(rig, root):
    rig.keys[State.READY].append(" ")
    rig.record_reads.extend([True] + [False] * 8)
    with pytest.raises(CameraError, match="stopped delivering frames"):
        run(root, replace(SETTINGS, duration=2))
    assert not list(root.rglob("*.mp4"))
    rig.assert_clean(root)


@pytest.mark.parametrize(
    "failure, message",
    [
        ("probe.mp4", "initialize the MP4 writer"),
        ("take.mp4", "initialize the MP4 writer"),
        ("empty_video", "video is empty"),
        ("truncate_video", "video is incomplete"),
        ("odd_resolution", "requires even dimensions"),
        ("disk_full", "free space and permissions"),
    ],
)
def test_unusable_recordings_are_never_published(rig, root, failure, message):
    rig.keys[State.READY].append(" ")
    if failure.endswith(".mp4"):
        rig.fail_writer = failure
    elif failure == "odd_resolution":
        rig.original = rig.original[:, :-1].copy()
    elif failure == "disk_full":
        rig.write_error = OSError(errno.ENOSPC, "No space left on device")
    else:
        setattr(rig, failure, True)
    with pytest.raises(RecordingError, match=message):
        run(root)
    assert not list(root.rglob("*.mp4"))
    assert not list(root.rglob("*.json"))
    rig.assert_clean(root)


@pytest.mark.parametrize("failed_extension", [".json", ".mp4"])
def test_publish_failure_rolls_back_the_pair(rig, root, monkeypatch, failed_extension):
    rig.keys[State.READY].append(" ")
    rig.keys[State.REVIEW].append("a")
    link = capture.os.link

    def fail_publish(source, destination, *args, **kwargs):
        if Path(destination).suffix == failed_extension:
            raise PermissionError("permission denied")
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(capture.os, "link", fail_publish)
    with pytest.raises(RecordingError, match="Could not save"):
        run(root)
    assert not list(root.rglob("*.mp4"))
    assert not list(root.rglob("*.json"))
    rig.assert_clean(root)


def test_existing_uuid_is_never_overwritten(rig, root, monkeypatch):
    rig.keys[State.READY].append(" ")
    rig.keys[State.REVIEW].append("a")
    sample_id = UUID("8cbb4e58-b53c-4ce5-91ea-b83685a1507b")
    monkeypatch.setattr(capture, "uuid4", lambda: sample_id)
    video = root / "WATER" / "SIGNER001" / f"{sample_id}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"existing video")
    with pytest.raises(RecordingError, match="Could not save"):
        run(root)
    assert video.read_bytes() == b"existing video"
    assert not video.with_suffix(".json").exists()
    rig.assert_clean(root)


def test_interruption_releases_resources(rig, root):
    rig.keys[State.READY].append(" ")
    rig.write_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(root, replace(SETTINGS, countdown=0))
    assert not list(root.rglob("*.mp4"))
    rig.assert_clean(root)


@pytest.mark.parametrize(
    "args, message",
    [
        (["preview", "--camera", "-1"], "Camera index"),
        (["capture", "--sign", "../WATER", "--signer", "SIGNER001"], "Sign label"),
        (["capture", "--sign", "WATER", "--signer", "SIGNER001", "--fps", "nan"], "FPS"),
        (["capture", "--sign", "WATER", "--signer", "SIGNER001", "--repetitions", "0"], "Repetitions"),
    ],
)
def test_cli_rejects_invalid_arguments_before_opening_camera(rig, args, message):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert message in result.output
    assert not rig.camera.settings
