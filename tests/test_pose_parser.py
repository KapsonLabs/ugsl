from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from landmarks.pose_parser import parse_pose_result
from landmarks.schemas import POSE_GROUPS, PoseArrays, PoseLandmark


def points(*, offset=0.0, visibility=0.9, presence=0.8, count=33):
    return [
        SimpleNamespace(x=offset + index / 32, y=0.25, z=-0.01, visibility=visibility, presence=presence)
        for index in range(count)
    ]


def raw(image=None, world=None):
    return SimpleNamespace(
        pose_landmarks=[] if image is None else [image],
        pose_world_landmarks=[] if world is None else [world],
    )


def parsed(result, index=0, time_ms=100, inference=None):
    source = Fraction(time_ms)
    timestamp = source // 1
    return parse_pose_result(result, index, source, timestamp, timestamp if inference is None else inference)


def test_landmark_order_and_coordinate_spaces_are_preserved():
    row = parsed(raw(points(offset=-0.25), points(offset=2)))
    assert list(map(int, PoseLandmark)) == list(range(33))
    assert POSE_GROUPS["upper_body"] == tuple(range(25))
    assert (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_FOOT_INDEX) == (15, 32)
    assert row.pose_detected and row.world_available
    assert row.image_landmarks.shape == row.world_landmarks.shape == (33, 3)
    assert row.image_landmarks[0, 0] == -0.25
    assert row.image_landmarks[15, 0] == pytest.approx(15 / 32 - 0.25)
    assert row.world_landmarks[32, 0] == 3
    assert row.visibility.dtype == row.presence_scores.dtype == np.float32
    assert row.visibility[32] == pytest.approx(0.9)
    assert row.presence_scores[32] == pytest.approx(0.8)


def test_no_pose_keeps_frame_and_arrays_masked():
    row = parsed(raw(), 8, Fraction(301, 3), 101)
    arrays = PoseArrays.from_frames([row])
    assert arrays.timestamps_ms.tolist() == [100]
    assert arrays.frame_indices.tolist() == [8]
    assert arrays.inference_timestamps_ms.tolist() == [101]
    assert arrays.timestamps_ms.dtype == arrays.frame_indices.dtype == np.int64
    assert arrays.pose_detected.dtype == arrays.world_available.dtype == np.bool_
    assert not arrays.pose_detected.any() and not arrays.world_available.any()
    for values in (arrays.image_landmarks, arrays.world_landmarks, arrays.visibility, arrays.presence_scores):
        assert np.isnan(values).all()
    assert not row.warnings and not row.invalid


def test_missing_scores_and_world_do_not_invalidate_image_geometry():
    image = points()
    del image[2].visibility
    image[3].presence = None
    row = parsed(raw(image))
    assert row.pose_detected and not row.world_available and not row.invalid
    assert np.isfinite(row.image_landmarks).all() and np.isnan(row.world_landmarks).all()
    assert np.isnan(row.visibility[2]) and np.isnan(row.presence_scores[3])
    assert row.warnings == [{"code": "POSE_WORLD_UNAVAILABLE"}]


@pytest.mark.parametrize(("image", "world", "pose_detected", "space"), [
    (32, 33, False, "image"), (33, 34, True, "world"),
])
def test_unexpected_count_preserves_frame_and_masks_affected_coordinates(image, world, pose_detected, space):
    row = parsed(raw(points(count=image), points(count=world)))
    assert row.invalid and row.pose_detected == pose_detected and not row.world_available
    assert row.warnings[0]["code"] == "UNEXPECTED_POSE_LANDMARK_COUNT"
    assert row.warnings[0]["space"] == space


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e80, None])
def test_bad_coordinates_never_contaminate_result(value):
    image = points()
    image[16].x = value
    row = parsed(raw(image, points()))
    assert row.invalid and not row.pose_detected and np.isnan(row.image_landmarks).all()
    assert row.warnings[0]["code"] == "INVALID_POSE_COORDINATES"
    row = parsed(raw(points(), image))
    assert row.invalid and row.pose_detected and not row.world_available


def test_bad_scores_are_masked_individually_and_flagged():
    image = points()
    image[0].visibility, image[6].presence, image[9].presence = -0.1, float("inf"), "bad"
    row = parsed(raw(image, points()))
    assert row.invalid and row.pose_detected and row.world_available
    assert np.isnan(row.visibility[0]) and np.isnan(row.presence_scores[[6, 9]]).all()
    assert row.warnings == [
        {"code": "INVALID_POSE_SCORE", "score": "visibility", "indices": [0]},
        {"code": "INVALID_POSE_SCORE", "score": "presence", "indices": [6, 9]},
    ]


def test_excess_poses_are_not_arbitrarily_selected():
    result = SimpleNamespace(pose_landmarks=[points(), points()], pose_world_landmarks=[])
    row = parsed(result)
    assert row.raw_pose_count == 2 and row.invalid and not row.pose_detected
    assert row.warnings[0]["code"] == "UNEXPECTED_POSE_COUNT"


def test_array_validation_rejects_misaligned_indices_or_masks():
    a = parsed(raw(), 2, 30)
    b = parsed(raw(), 2, 40)
    with pytest.raises(ValueError, match="strictly increasing"):
        PoseArrays.from_frames([a, b])
    a.world_available = True
    with pytest.raises(ValueError, match="valid image pose"):
        PoseArrays.from_frames([a])
