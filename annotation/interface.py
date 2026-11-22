"""OpenCV annotation view and keyboard-driven review."""

import math
import time
from contextlib import suppress
from dataclasses import dataclass
from textwrap import wrap

import cv2
import typer

from annotation.models import AnnotationInput, Decision, REASONS
from annotation.playback import FramePlayer, Timeline, key_name, probe_timeline
from config import QualityConfig
from ingestion.exceptions import MediaError
from registry.database import RegistryError

WINDOW = "UGSL Annotation"
SPEEDS = (0.25, 0.5, 1.0, 2.0)


@dataclass
class PlaybackControls:
    timeline: Timeline
    index: int = 0
    speed_index: int = 2
    playing: bool = False
    stop_ms: int | None = None
    deadline: float = 0

    def _schedule(self, now: float) -> None:
        stamps = self.timeline.timestamps_ms
        after = stamps[self.index + 1] if self.index + 1 < len(stamps) else self.timeline.duration_ms
        self.deadline = now + max(1, after - stamps[self.index]) / (1000 * SPEEDS[self.speed_index])

    def pause(self) -> None:
        self.playing = False
        self.stop_ms = None

    def toggle(self, now: float) -> None:
        if self.playing:
            self.pause()
        else:
            if self.index == len(self.timeline.timestamps_ms) - 1:
                self.index = 0
            self.playing = True
            self._schedule(now)

    def navigate(self, key: str) -> None:
        self.pause()
        if key in {"LEFT", "RIGHT"}:
            delta = -1 if key == "LEFT" else 1
            self.index = max(0, min(len(self.timeline.timestamps_ms) - 1, self.index + delta))
        else:
            delta = -500 if key == "j" else 500
            self.index = self.timeline.seek(self.timeline.timestamps_ms[self.index] + delta)

    def replay(self, start: int, end: int, now: float) -> None:
        index = self.timeline.seek(start)
        if start >= end or self.timeline.timestamps_ms[index] >= end:
            raise RegistryError("The marked interval contains no displayed frames.")
        self.index = index
        self.stop_ms = end
        self.playing = True
        self._schedule(now)

    def change_speed(self, delta: int, now: float) -> None:
        self.speed_index = max(0, min(len(SPEEDS) - 1, self.speed_index + delta))
        if self.playing:
            self._schedule(now)

    def advance(self, now: float) -> None:
        if not self.playing or now < self.deadline:
            return
        after = self.index + 1
        stamps = self.timeline.timestamps_ms
        if after == len(stamps) or (self.stop_ms is not None and stamps[after] >= self.stop_ms):
            self.pause()
        else:
            self.index = after
            self._schedule(self.deadline)


class AnnotationWindow:
    def __init__(self, service, view, reviewer_code: str):
        self.service = service
        self.view = view
        self.reviewer_code = reviewer_code
        try:
            timeout = QualityConfig.from_environment().decode_timeout
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
        timeline = probe_timeline(view.path, timeout=timeout)
        timeline = Timeline(timeline.timestamps_ms, min(timeline.duration_ms, view.duration_ms))
        self.controls = PlaybackControls(timeline)
        self.start_ms, self.end_ms = self._saved_markers()
        self.notice = ""

    def _saved_markers(self):
        annotation = self.view.current
        return (annotation.start_ms, annotation.end_ms) if annotation else (None, None)

    @property
    def dirty(self) -> bool:
        return (self.start_ms, self.end_ms) != self._saved_markers()

    def draw(self, frame) -> None:
        preview = frame.copy()
        height, width = preview.shape[:2]
        scale = max(0.30, min(0.72, width / 1050))
        spacing = max(13, round(30 * scale))
        position = round(self.controls.timeline.timestamps_ms[self.controls.index])
        sign = self.view.sign
        annotated = self.view.annotated_sign
        lines = [
            f"Sample {self.view.sample.id}",
            f"Assigned {sign.gloss} ({sign.id})  |  Signer {self.view.signer_code}",
            f"Video {position} / {self.controls.timeline.duration_ms} ms  |  Frame {self.controls.index + 1} "
            f"of {len(self.controls.timeline.timestamps_ms)}  |  {SPEEDS[self.controls.speed_index]:g}x "
            f"{'playing' if self.controls.playing else 'paused'}",
            f"Start {self.start_ms if self.start_ms is not None else '-'} ms  |  "
            f"End {self.end_ms if self.end_ms is not None else '-'} ms"
            f"{'  |  Unsaved' if self.dirty else ''}",
            f"{self.view.sample.review_state}  |  {self.view.sample.technical_validation_status}"
            f"{'  |  Adjudication required' if self.view.needs_adjudication else ''}",
        ]
        if annotated:
            variant = f" [{annotated.variant_label}]" if annotated.variant_label else ""
            lines.append(f"Annotation v{self.view.current.version}  {annotated.gloss}{variant} ({annotated.id})")
        if self.notice:
            lines.append(self.notice)
        lines.extend(f"Quality: {warning}" for warning in self.view.warnings)
        lines.extend([
            "SPACE play  LEFT/RIGHT frame  J/L seek  +/- speed",
            "I/O mark  P replay  S save  A approve  R reject  U second review  N next  Q exit",
        ])
        # Draw onto the copied frame only.
        limit = max(20, int(width / (10.5 * scale)))
        y = spacing
        for line in lines:
            for part in wrap(line, width=limit):
                if y >= height:
                    break
                xy = (8, y)
                cv2.putText(preview, part, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 3, cv2.LINE_AA)
                cv2.putText(preview, part, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (242, 242, 242), 1, cv2.LINE_AA)
                y += spacing
        cv2.imshow(WINDOW, preview)

    def _refresh(self) -> None:
        self.view = self.service.view(self.view.sample.id, self.reviewer_code)
        self.start_ms, self.end_ms = self._saved_markers()

    def _mark(self, key: str) -> None:
        self.controls.pause()
        if key == "i":
            self.start_ms = self.controls.timeline.start_marker(self.controls.index)
            self.notice = f"Marked start at {self.start_ms} ms."
        else:
            self.end_ms = self.controls.timeline.end_marker(self.controls.index)
            self.notice = f"Marked end at {self.end_ms} ms."

    def _editing_bounds(self):
        if self.start_ms is None or self.end_ms is None:
            raise RegistryError("Mark both boundaries with I and O.")
        self.service.validator.interval(self.start_ms, self.end_ms, self.controls.timeline.duration_ms)
        return self.start_ms, self.end_ms

    def _prompt_ready(self, frame, text: str) -> None:
        self.controls.pause()
        self.notice = text
        self.draw(frame)
        cv2.waitKeyEx(1)
        typer.echo(text)

    def _save(self, frame) -> None:
        start_ms, end_ms = self._editing_bounds()
        active = self.view.current
        self._prompt_ready(frame, "Use the terminal to save the annotation.")
        default_sign = active.sign_id if active else self.view.sign.id
        try:
            sign = typer.prompt("Registered sign or variant ID", default=default_sign).strip().upper()
            if active:
                if not self.dirty and sign == active.sign_id:
                    self.notice = "The saved annotation is unchanged."
                    return
                reason = typer.prompt("Correction reason")
                self.service.correct(
                    self.view.sample.id, self.reviewer_code, reason=reason, sign_id=sign, start_ms=start_ms, end_ms=end_ms,
                    expected_annotation_id=active.id, expected_round=self.view.sample.review_round,
                )
            else:
                reason = typer.prompt("Reason for changing the assigned sign") if sign != default_sign else None
                self.service.create(
                    self.view.sample.id, self.reviewer_code, AnnotationInput(sign, start_ms, end_ms), reason=reason,
                )
        except typer.Abort:
            self.notice = "Save cancelled. Boundary changes are still unsaved."
            return
        self._refresh()
        self.notice = f"Saved annotation version {self.view.current.version}."

    def _review(self, key: str, frame) -> None:
        if self.dirty or self.view.current is None:
            raise RegistryError("Save the annotation before entering a review decision.")
        self.controls.pause()
        action = {"a": Decision.APPROVE, "r": Decision.REJECT, "u": Decision.REQUEST_SECOND_REVIEW}[key]
        reason, notes = None, None
        if key != "a":
            self._prompt_ready(frame, "Use the terminal to enter review details.")
            try:
                if key == "r":
                    typer.echo("Reasons: " + ", ".join(value.lower().replace("_", "-") for value in REASONS))
                    reason = typer.prompt("Rejection reason code")
                notes = typer.prompt("Notes", default="", show_default=False)
            except typer.Abort:
                self.notice = "Review decision cancelled."
                return
        decision = self.service.decide(
            self.view.sample.id, self.reviewer_code, action, reason_code=reason, notes=notes,
            expected_annotation_id=self.view.current.id, expected_round=self.view.sample.review_round,
        )
        self._refresh()
        self.notice = f"Decision recorded. State {decision.state}."

    def _leave(self, frame) -> bool:
        if not self.dirty:
            return True
        self._prompt_ready(frame, "There are unsaved annotation changes.")
        try:
            leaving = typer.confirm("Discard unsaved changes?", default=False)
        except typer.Abort:
            leaving = False
        if not leaving:
            self.notice = "Unsaved changes retained. Press S to save."
        return leaving

    def press(self, key: str, frame) -> str | None:
        now = time.monotonic()
        if key == " ":
            self.controls.toggle(now)
        elif key in {"LEFT", "RIGHT", "j", "l"}:
            self.controls.navigate(key)
        elif key in {"i", "o"}:
            self._mark(key)
        elif key == "p":
            start, end = self._editing_bounds()
            self.controls.replay(start, end, now)
        elif key in {"+", "=", "-"}:
            self.controls.change_speed(-1 if key == "-" else 1, now)
        elif key == "s":
            self._save(frame)
        elif key in {"a", "r", "u"}:
            self._review(key, frame)
        elif key in {"n", "q"} and self._leave(frame):
            return "next" if key == "n" else "quit"
        return None

    def run(self) -> str:
        try:
            with FramePlayer(self.view.path, self.controls.timeline) as player:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                while True:
                    frame = player.at(self.controls.index)
                    self.draw(frame)
                    remaining = (self.controls.deadline - time.monotonic()) * 1000
                    delay = max(1, min(40, math.ceil(remaining))) if self.controls.playing else 40
                    key = key_name(cv2.waitKeyEx(delay))
                    with suppress(cv2.error):
                        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                            key = "q"
                    try:
                        action = self.press(key, frame)
                        if action:
                            return action
                    except (RegistryError, MediaError) as exc:
                        self.controls.pause()
                        self.notice = str(exc)
                    self.controls.advance(time.monotonic())
        except (KeyboardInterrupt, typer.Abort):
            if self.dirty:
                typer.echo("Session interrupted. Unsaved boundary changes will be discarded. Saved annotations remain.", err=True)
            return "quit"
        except (RegistryError, MediaError, cv2.error):
            if self.dirty:
                typer.echo("Playback stopped. Unsaved boundary changes will be discarded. Saved annotations remain.", err=True)
            raise
        finally:
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW)


def open_video(service, sample_id: str, reviewer_code: str) -> str:
    try:
        return AnnotationWindow(service, service.view(sample_id, reviewer_code), reviewer_code).run()
    except cv2.error as exc:
        raise RegistryError("Could not run the OpenCV annotation window. Use a graphical desktop session.") from exc
