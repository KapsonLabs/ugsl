"""Draw anatomical pose landmarks on the decoded image or a black canvas."""

import cv2
import numpy as np

from landmarks.preview import LandmarkPlayback
from landmarks.schemas import POSE_GROUPS, DecodedFrame, PoseFrame, PoseLandmark

WINDOW = "UGSL Pose Landmarks"
LEFT_COLOR = (255, 210, 55)
RIGHT_COLOR = (80, 225, 105)
CENTER_COLOR = (235, 235, 235)
# MediaPipe face reference, arms/torso and leg/foot links.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def image_point(landmark, width: int, height: int) -> tuple[int, int] | None:
    x, y, z = (float(coordinate) for coordinate in landmark)
    if not np.isfinite((x, y, z)).all() or not (0 <= x < 1 and 0 <= y < 1):
        return None
    return min(width - 1, round(x * width)), min(height - 1, round(y * height))


def _side(index: int) -> str:
    name = PoseLandmark(index).name
    if "LEFT" in name:
        return "L"
    if "RIGHT" in name:
        return "R"
    return ""


def _color(side: str):
    return {"L": LEFT_COLOR, "R": RIGHT_COLOR}.get(side, CENTER_COLOR)


class PoseLandmarkVisualizer(LandmarkPlayback):
    window = WINDOW

    def __init__(
        self, *, model_variant: str = "full", full_body: bool = False, skeleton_only: bool = False,
        show_indices: bool = False, show_visibility: bool = False, visibility_threshold: float = 0.5,
        mirror_preview: bool = False,
    ):
        self.model_variant = model_variant
        self.indices = tuple(PoseLandmark) if full_body else POSE_GROUPS["upper_body"]
        self.skeleton_only = skeleton_only
        self.show_indices = show_indices
        self.show_visibility = show_visibility
        self.visibility_threshold = visibility_threshold
        self.mirror_preview = mirror_preview

    def render(self, frame: DecodedFrame, pose: PoseFrame) -> np.ndarray:
        canvas = np.zeros_like(frame.image) if self.skeleton_only else cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        height, width = canvas.shape[:2]
        if self.mirror_preview:
            canvas = cv2.flip(canvas, 1)
        points = {}
        if pose.pose_detected:
            for index in self.indices:
                score = float(pose.visibility[index])
                if not np.isfinite(score) or score < self.visibility_threshold:
                    continue
                point = image_point(pose.image_landmarks[index], width, height)
                if point is not None:
                    x, y = point
                    points[index] = (width - 1 - x if self.mirror_preview else x, y)
        for a, b in CONNECTIONS:
            if a in points and b in points:
                color = _color(_side(a)) if _side(a) == _side(b) else CENTER_COLOR
                cv2.line(canvas, points[a], points[b], color, 2, cv2.LINE_AA)
        for index, at in points.items():
            side = _side(index)
            color = _color(side)
            cv2.circle(canvas, at, 3, color, cv2.FILLED, cv2.LINE_AA)
            label = str(int(index)) if self.show_indices else ""
            if self.show_visibility:
                label = f"{label} {pose.visibility[index]:.2f}".strip()
            if label:
                self._caption(canvas, label, (min(at[0] + 5, width - 1), max(at[1] - 4, 9)), color, 0.35)

        status = "pose detected" if pose.pose_detected else "no valid pose"
        if pose.invalid:
            status += " (invalid result)"
        line = f"Frame {frame.frame_index} | Source {pose.timestamp_ms} ms | {status} | {self.model_variant}"
        scale = max(0.3, min(0.65, width / 880))
        self._caption(canvas, line, (6, max(12, round(32 * scale))), CENTER_COLOR, scale)
        self._caption(canvas, "L: anatomical left", (6, height - 7), LEFT_COLOR, scale)
        self._caption(canvas, "R: anatomical right", (max(6, width // 2), height - 7), RIGHT_COLOR, scale)
        return canvas
