"""OpenCV playback on a copy of the source RGB frame or a black canvas."""

import os
import sys
from contextlib import suppress
from time import monotonic

import cv2
import numpy as np

from landmarks.exceptions import FrameProcessingError
from landmarks.schemas import HAND_SLOTS, DecodedFrame, HandFrame

WINDOW = "UGSL Hand Landmarks"
COLORS = ((255, 210, 55), (80, 225, 105))
UNASSIGNED = (100, 180, 255)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandLandmarkVisualizer:
    def __init__(self, *, skeleton_only: bool = False):
        self.skeleton_only = skeleton_only
        self._last_timestamp: int | None = None
        self._last_display: float | None = None

    @staticmethod
    def _caption(canvas, text, at, color, scale):
        cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def _draw_hand(self, canvas, landmarks, color, label):
        height, width = canvas.shape[:2]
        # Limit the drawing area only. Stored MediaPipe coordinates remain unchanged.
        points = [
            (round(float(x) * (width - 1)), round(float(y) * (height - 1)))
            for x, y in np.clip(landmarks[:, :2], 0, 1)
        ]
        for a, b in CONNECTIONS:
            cv2.line(canvas, points[a], points[b], color, 2, cv2.LINE_AA)
        for point in points[1:]:
            cv2.circle(canvas, point, 3, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, points[0], 5, color, 2, cv2.LINE_AA)
        x, y = points[0]
        self._caption(canvas, label, (max(0, min(x + 8, width - 130)), max(12, y - 8)), color, 0.45)

    def render(self, frame: DecodedFrame, landmarks: HandFrame) -> np.ndarray:
        canvas = np.zeros_like(frame.image) if self.skeleton_only else cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        for index, label in enumerate(HAND_SLOTS):
            if landmarks.presence_mask[index]:
                self._draw_hand(
                    canvas, landmarks.image_landmarks[index], COLORS[index],
                    f"{label} {landmarks.handedness_score[index]:.2f}",
                )
        for detection in landmarks.detections:
            if "unassigned_image_landmarks" in detection:
                predictions = detection["predictions"]
                first = predictions[0] if predictions else {}
                score = first.get("score")
                suffix = f" {first.get('label') or '?'} {score:.2f}" if score is not None else ""
                self._draw_hand(
                    canvas, np.asarray(detection["unassigned_image_landmarks"]), UNASSIGNED, f"Unassigned{suffix}",
                )
        status = " | ambiguous handedness" if landmarks.ambiguous_handedness else ""
        scale = max(0.3, min(0.65, canvas.shape[1] / 800))
        self._caption(
            canvas, f"Frame {frame.frame_index} | Source {landmarks.timestamp_ms} ms{status}",
            (6, max(12, round(32 * scale))), (255, 255, 255), scale,
        )
        return canvas

    def __enter__(self):
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise FrameProcessingError("OpenCV playback requires a graphical desktop session.")
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise FrameProcessingError("Could not open an OpenCV playback window.") from exc
        self._last_timestamp = self._last_display = None
        return self

    def __exit__(self, *_):
        with suppress(cv2.error):
            cv2.destroyWindow(WINDOW)

    def show(self, frame: DecodedFrame, landmarks: HandFrame) -> bool:
        """Play at source cadence when inference keeps up. Q or Escape ends playback."""
        try:
            canvas = self.render(frame, landmarks)
            now = monotonic()
            if self._last_timestamp is not None:
                deadline = self._last_display + (landmarks.timestamp_ms - self._last_timestamp) / 1000
                while now < deadline:
                    if not self._wait(min(50, max(1, round(1000 * (deadline - now))))):
                        return False
                    now = monotonic()
            cv2.imshow(WINDOW, canvas)
            self._last_timestamp, self._last_display = landmarks.timestamp_ms, monotonic()
            return self._wait(1)
        except cv2.error as exc:
            raise FrameProcessingError("Could not display the hand landmark playback.") from exc

    @staticmethod
    def _wait(delay_ms: int) -> bool:
        key = cv2.waitKeyEx(delay_ms)
        if key in (ord("q"), ord("Q"), 27):
            return False
        with suppress(cv2.error):
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                return False
        return True
