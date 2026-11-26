"""Validate MediaPipe pose results while retaining temporal alignment."""

from math import isfinite

import numpy as np

from landmarks.schemas import POSE_LANDMARK_COUNT, PoseFrame


def _invalid(frame, code, **details):
    frame.invalid = True
    frame.warnings.append({"code": code, **details})


def _coordinates(points, frame: PoseFrame, space: str) -> np.ndarray | None:
    if len(points) != POSE_LANDMARK_COUNT:
        _invalid(
            frame, "UNEXPECTED_POSE_LANDMARK_COUNT",
            space=space, count=len(points), expected=POSE_LANDMARK_COUNT,
        )
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float32)
    except (AttributeError, OverflowError, TypeError, ValueError):
        _invalid(frame, "INVALID_POSE_COORDINATES", space=space)
        return None
    if coordinates.shape != (POSE_LANDMARK_COUNT, 3):
        _invalid(frame, "INVALID_POSE_COORDINATES", space=space)
        return None
    invalid = np.flatnonzero(~np.isfinite(coordinates).all(axis=1))
    if invalid.size:
        _invalid(frame, "INVALID_POSE_COORDINATES", space=space, indices=invalid.tolist())
        return None
    return coordinates


def _scores(points, name, target, frame):
    invalid = []
    for index, landmark in enumerate(points):
        score = getattr(landmark, name, None)
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError, OverflowError):
            invalid.append(index)
            continue
        if not isfinite(score) or not 0 <= score <= 1:
            invalid.append(index)
        else:
            target[index] = score
    if invalid:
        _invalid(frame, "INVALID_POSE_SCORE", score=name, indices=invalid)


def parse_pose_result(raw, index, source_timestamp, timestamp_ms, inference_timestamp_ms) -> PoseFrame:
    frame = PoseFrame(index, source_timestamp, timestamp_ms, inference_timestamp_ms)
    poses = getattr(raw, "pose_landmarks", None)
    worlds = getattr(raw, "pose_world_landmarks", None) or []
    if poses is None:
        _invalid(frame, "MISSING_POSE_RESULT_COMPONENT")
        return frame
    frame.raw_pose_count = len(poses)
    if len(poses) > 1 or len(worlds) > 1 or (worlds and not poses):
        _invalid(frame, "UNEXPECTED_POSE_COUNT", image=len(poses), world=len(worlds), maximum_poses=1)
    if len(poses) != 1:
        return frame

    points = poses[0] or []
    image = _coordinates(points, frame, "image")
    if image is None:
        return frame
    frame.image_landmarks = image
    frame.pose_detected = True
    _scores(points, "visibility", frame.visibility, frame)
    _scores(points, "presence", frame.presence_scores, frame)
    if not worlds or not worlds[0]:
        frame.warnings.append({"code": "POSE_WORLD_UNAVAILABLE"})
    elif len(worlds) == 1:
        world = _coordinates(worlds[0], frame, "world")
        if world is not None:
            frame.world_landmarks = world
            frame.world_available = True
    return frame
