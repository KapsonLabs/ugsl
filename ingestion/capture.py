"""OpenCV camera access, interactive capture and review, and sample storage."""

import json
import math
import os
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from uuid import uuid4

import cv2

from config import DATASET_ROOT, CaptureConfig, validate_identifier
from ingestion.exceptions import CameraError, CaptureError, RecordingError
from ingestion.schemas import RecordingMetadata, RecordingState as State

WINDOW = "UGSL Camera Capture"
MAX_CONSECUTIVE_FAILURES = 5
TERMINAL_STATES = {State.COMPLETED, State.CANCELLED}


class CameraCapture:
    """Context-managed access to a configured camera and its delivered frame size."""

    def __init__(self, config: CaptureConfig | None = None):
        self.config = config if config is not None else CaptureConfig()
        self.resolution: tuple[int, int] | None = None
        self._camera = None

    def __enter__(self) -> "CameraCapture":
        try:
            self._camera = cv2.VideoCapture(self.config.camera_index)
            if not self._camera.isOpened():
                raise CameraError(f"Cannot open camera {self.config.camera_index}. Check the device and camera permissions.")
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            for _ in range(MAX_CONSECUTIVE_FAILURES):
                if self.read() is not None:
                    return self
            raise CameraError(f"Camera {self.config.camera_index} opened but delivered no frames.")
        except cv2.error as exc:
            self.close()
            raise CameraError(f"Could not initialize camera {self.config.camera_index}: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._camera is not None:
            camera, self._camera = self._camera, None
            camera.release()

    def read(self):
        """Return an untouched BGR frame, or None for a recoverable failed read."""
        try:
            if self._camera is None or not self._camera.isOpened():
                raise CameraError(f"Camera {self.config.camera_index} is closed or disconnected.")
            ok, frame = self._camera.read()
        except cv2.error as exc:
            raise CameraError(f"Camera {self.config.camera_index} could not read a frame: {exc}") from exc
        if not ok or frame is None or not frame.size:
            return None
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype.name != "uint8":
            raise CameraError("The camera must deliver 8-bit, three-channel BGR frames.")
        height, width = frame.shape[:2]
        self.resolution = (width, height)
        return frame

    def preview(self) -> None:
        """Run a camera-only preview until the operator quits."""
        self._run()

    def capture(
        self, sign: str, signer: str, dataset_root: Path = DATASET_ROOT
    ) -> list[RecordingMetadata]:
        """Return the samples accepted in this session, in repetition order."""
        validate_identifier(sign, "Sign label")
        validate_identifier(signer, "Signer identifier")
        return self._run(sign, signer, Path(dataset_root))

    def _run(self, sign=None, signer=None, root=None):
        try:
            with self:
                try:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                    session = _Session(self, sign, signer, root)
                    if self.resolution != (self.config.width, self.config.height):
                        width, height = self.resolution
                        session.notice = f"Camera supplied {width} x {height}"
                    return session.run()
                finally:
                    with suppress(cv2.error):
                        cv2.destroyWindow(WINDOW)
        except cv2.error as exc:
            raise CaptureError(f"OpenCV camera or display operation failed: {exc}") from exc
        except OSError as exc:
            raise RecordingError(f"Recording storage error. Check free space and permissions: {exc}") from exc


class _Session:
    def __init__(
        self, camera: CameraCapture, sign: str | None, signer: str | None, root: Path | None
    ):
        self.camera = camera
        self.config = camera.config
        self.sign = sign
        self.signer = signer
        self.root = root
        self.state = State.READY
        self.accepted: list[RecordingMetadata] = []
        self.pending: RecordingMetadata | None = None
        self.frame = None
        self.read_failures = 0
        self.countdown_end = 0.0
        self.notice = ""
        self.staging: Path | None = None

    def run(self):
        if self.root is None:
            return self._loop()
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".capture-", dir=self.root.resolve().parent) as directory:
            self.staging = Path(directory)
            return self._loop()

    def _loop(self):
        while self.state not in TERMINAL_STATES:
            if self.state == State.PLAYBACK:
                self._playback()
                continue
            if self.state == State.COUNTDOWN and time.monotonic() >= self.countdown_end:
                self._record()
                continue
            if self.state in {State.READY, State.COUNTDOWN}:
                self._read_live()
            self._show(self.frame)
            self._handle_key(self._key())
        return self.accepted

    def _read_live(self):
        frame = self.camera.read()
        if frame is None:
            self.read_failures += 1
            if self.read_failures >= MAX_CONSECUTIVE_FAILURES:
                raise CameraError(f"Camera {self.config.camera_index} stopped delivering frames. Check its connection.")
        else:
            self.read_failures = 0
            self.frame = frame
        return frame

    @staticmethod
    def _key(delay=1):
        return chr(cv2.waitKey(delay) & 0xFF).lower()

    def _handle_key(self, key):
        if key == "q":
            self.state = State.CANCELLED
        elif self.root is None:
            return
        elif self.state == State.READY and key == " " and self.frame is not None:
            self._start_countdown()
        elif self.state in {State.COUNTDOWN, State.RECORDING, State.REVIEW} and key == "\x1b":
            self._discard("Recording cancelled.")
        elif self.state == State.REVIEW:
            if key == "p":
                self.state = State.PLAYBACK
            elif key == "r":
                self._discard("Recording rejected.")
            elif key == "a":
                self._accept()
        elif self.state == State.PLAYBACK and key == "\x1b":
            self.state = State.REVIEW

    @property
    def _video(self):
        return self.staging / "take.mp4"

    def _start_countdown(self):
        width, height = self.camera.resolution
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise RecordingError(
                f"Camera supplied {width} x {height}. MP4 capture requires even dimensions "
                "to preserve the full image. Request a different camera mode."
            )
        # Test the codec, dimensions and output location before capturing a take.
        probe = self.staging / "probe.mp4"
        writer = self._open_writer(probe, self.config.fps, (width, height))
        writer.release()
        probe.unlink(missing_ok=True)
        self.notice = ""
        self.countdown_end = time.monotonic() + self.config.countdown
        self.state = State.COUNTDOWN

    def _open_writer(self, path, fps, resolution):
        writer = None
        opened = False
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, resolution)
            opened = writer.isOpened()
            if opened:
                return writer
        except cv2.error as exc:
            raise RecordingError(f"MP4 writer failed for {resolution} at {fps:.3f} FPS: {exc}") from exc
        finally:
            if writer is not None and not opened:
                writer.release()
        raise RecordingError(
            f"Cannot initialize the MP4 writer for {resolution} at {fps:.3f} FPS. "
            "Check codec support, available disk space and permissions."
        )

    def _record(self):
        resolution = self.camera.resolution
        # Reuse a single contiguous buffer when reading raw frames back for encoding.
        buffer = self.frame.copy(order="C")
        pixels = memoryview(buffer).cast("B")
        frame_count = failed_reads = 0
        with TemporaryFile(dir=self.staging) as raw:
            recorded_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            self.state = State.RECORDING
            start = time.monotonic()
            elapsed = 0.0
            while elapsed < self.config.duration:
                frame = self._read_live()
                if frame is None:
                    failed_reads += 1
                else:
                    if self.camera.resolution != resolution:
                        raise CameraError("Camera resolution changed during recording. Reopen the camera and retry.")
                    raw.write(frame.tobytes())
                    frame_count += 1
                self._show(self.frame, elapsed=time.monotonic() - start)
                self._handle_key(self._key())
                if self.state != State.RECORDING:
                    return
                elapsed = time.monotonic() - start

            if frame_count == 0 or elapsed <= 0:
                raise RecordingError("The camera delivered no usable frames for this recording.")
            measured_fps = frame_count / elapsed
            self._show(self.frame, elapsed=elapsed, controls="Preparing review...")
            self._handle_key(self._key())
            if self.state != State.RECORDING:
                return
            raw.seek(0)
            writer = self._open_writer(self._video, measured_fps, resolution)
            try:
                for _ in range(frame_count):
                    if raw.readinto(pixels) != len(pixels):
                        raise RecordingError("The temporary frame data is incomplete.")
                    writer.write(buffer)
            finally:
                writer.release()

        encoded_fps, size = self._validate_video(resolution, frame_count)
        sample_id = str(uuid4())
        path = self.root / self.sign / self.signer / f"{sample_id}.mp4"
        self.pending = RecordingMetadata(
            sample_id=sample_id,
            gloss=self.sign,
            signer_id=self.signer,
            repetition=len(self.accepted) + 1,
            camera_index=self.config.camera_index,
            width=resolution[0],
            height=resolution[1],
            requested_fps=self.config.fps,
            measured_fps=measured_fps,
            encoded_fps=encoded_fps,
            frame_count=frame_count,
            failed_frame_reads=failed_reads,
            requested_duration_seconds=self.config.duration,
            actual_duration_seconds=elapsed,
            file_size_bytes=size,
            file_path=str(path),
            recorded_at=recorded_at,
        )
        self.state = State.REVIEW

    def _validate_video(self, resolution, frame_count):
        if not self._video.is_file() or not (size := self._video.stat().st_size):
            raise RecordingError("The encoded video is empty. Check free disk space and codec support.")
        decoder = cv2.VideoCapture(str(self._video))
        try:
            fps = decoder.get(cv2.CAP_PROP_FPS)
            if not decoder.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise RecordingError("The encoded video cannot be opened at a valid playback rate.")
            for _ in range(frame_count):
                ok, frame = decoder.read()
                if not ok or frame is None or frame.shape[:2][::-1] != resolution:
                    raise RecordingError(
                        "The encoded video is incomplete, unreadable or has the wrong resolution. "
                        "Check free disk space and camera settings."
                    )
            if decoder.read()[0]:
                raise RecordingError("The encoded video contains more frames than the captured take.")
            return fps, size
        finally:
            decoder.release()

    def _playback(self):
        decoder = cv2.VideoCapture(str(self._video))
        try:
            if not decoder.isOpened():
                raise RecordingError("Could not reopen the temporary video for replay.")
            fps = self.pending.encoded_fps
            start = time.monotonic()
            for index in range(self.pending.frame_count):
                ok, frame = decoder.read()
                if not ok or frame is None:
                    raise RecordingError("Replay ended before all recorded frames could be read.")
                self._show(frame)
                deadline = start + (index + 1) / fps
                while True:
                    remaining = max(0.0, deadline - time.monotonic())
                    self._handle_key(self._key(max(1, math.ceil(min(0.02, remaining) * 1000))))
                    if self.state != State.PLAYBACK:
                        return
                    if time.monotonic() >= deadline:
                        break
            self.state = State.REVIEW
        finally:
            decoder.release()

    def _discard(self, notice):
        self.state = State.CANCELLED
        self.pending = None
        self._video.unlink(missing_ok=True)
        self.notice = notice
        self.state = State.READY

    def _accept(self):
        pending = self.pending
        video = Path(pending.file_path)
        sidecar = video.with_suffix(".json")
        staged_json = self.staging / "take.json"

        try:
            # Refuse symlinked gloss/signer directories so validated labels stay under the root.
            for directory in (video.parent.parent, video.parent):
                if directory.is_symlink():
                    raise RecordingError(f"Dataset directory cannot be a symlink: {directory}")
                directory.mkdir(exist_ok=True)

            with staged_json.open("x", encoding="utf-8") as file:
                json.dump(asdict(pending), file, indent=2, allow_nan=False)
                file.write("\n")

            # Publish metadata first so an interruption cannot expose a video without it.
            # Hard links fail if a destination already exists.
            os.link(staged_json, sidecar)
            os.link(self._video, video)
        except BaseException as exc:
            for source, destination in ((self._video, video), (staged_json, sidecar)):
                with suppress(FileNotFoundError):
                    if source.samefile(destination):
                        destination.unlink()
            if isinstance(exc, (OSError, ValueError)):
                raise RecordingError(f"Could not save the recording and metadata: {exc}") from exc
            raise

        self.accepted.append(pending)
        self.pending = None
        self._video.unlink()
        staged_json.unlink()
        self.notice = f"Sample {pending.repetition} saved."
        self.state = State.COMPLETED if len(self.accepted) == self.config.repetitions else State.READY

    def _show(self, frame, *, elapsed=0.0, controls=None):
        if frame is None:
            return
        preview = frame.copy()
        height, width = preview.shape[:2]
        if self.state in {State.READY, State.COUNTDOWN, State.RECORDING}:
            cv2.rectangle(preview, (width // 4, height // 8), (width * 3 // 4, height * 7 // 8), (128, 165, 150), 1)

        if self.root is None:
            lines = [f"Camera {self.config.camera_index}  |  {width} x {height}", "Q quit"]
        else:
            lines = [
                f"Sign: {self.sign}",
                f"Signer: {self.signer}",
                f"Sample: {len(self.accepted) + 1} of {self.config.repetitions}",
                f"Status: {self.state}",
            ]
            if self.state == State.COUNTDOWN:
                lines.append(f"Starts in {max(1, math.ceil(self.countdown_end - time.monotonic()))}")
            if self.state == State.RECORDING:
                percent = min(100, int(100 * (elapsed / self.config.duration)))
                lines.append(f"{percent}%  |  {elapsed:.1f} / {self.config.duration:g} s")
            if controls is None:
                controls = {
                    State.READY: "SPACE start  |  Q quit",
                    State.COUNTDOWN: "ESC cancel  |  Q quit",
                    State.RECORDING: "ESC cancel  |  Q quit",
                    State.REVIEW: "A accept  |  R reject  |  P replay  |  ESC cancel  |  Q quit",
                    State.PLAYBACK: "ESC return to review  |  Q quit",
                }[self.state]
            lines.append(controls)
        if self.notice:
            lines.append(self.notice)

        scale = max(0.25, min(0.7, width / 1000))
        spacing = max(13, round(33 * scale))
        for index, line in enumerate(lines, start=1):
            position = (8, spacing * index)
            color = (90, 100, 255) if line == "Status: RECORDING" else (245, 245, 245)
            cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, preview)
