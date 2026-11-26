"""Canonical slots, raw predictions, and validation of MediaPipe output."""

from types import SimpleNamespace

import numpy as np
import pytest

from landmarks.hand_parser import parse_hand_result
from landmarks.schemas import HAND_SCHEMA_VERSION, HandArrays


def points(value=0.25, count=21):
    return [SimpleNamespace(x=value, y=0.75, z=-0.05) for _ in range(count)]


def prediction(label, score=0.9):
    return SimpleNamespace(category_name=label, score=score)


def raw(images=(), labels=(), worlds=()):
    return SimpleNamespace(hand_landmarks=images, hand_world_landmarks=worlds, handedness=labels)


def test_two_hands_use_anatomical_slots_instead_of_result_order():
    result = parse_hand_result(raw(
        [points(0.8), points(-0.02)], [[prediction("Right", 0.7)], [prediction("Left")]],
        [points(2.0), points(3.0)],
    ), 8, 266)
    assert result.presence_mask.tolist() == [True, True]
    assert result.handedness_score.tolist() == pytest.approx([0.9, 0.7])
    assert result.image_landmarks[:, 0, :].tolist() == [
        pytest.approx([-0.02, 0.75, -0.05]), pytest.approx([0.8, 0.75, -0.05]),
    ]
    assert result.world_landmarks[:, 0, 0].tolist() == [3.0, 2.0]
    assert result.detections[0]["predictions"] == [{"label": "Right", "score": 0.7}]


def test_empty_frame_and_world_absence_keep_fixed_arrays():
    first = parse_hand_result(raw([points()], [[prediction("Left")]]), 1, 50)
    empty = parse_hand_result(raw(), 2, 100)
    arrays = HandArrays.from_frames([first, empty])
    assert arrays.schema_version == HAND_SCHEMA_VERSION
    assert arrays.image_landmarks.shape == arrays.world_landmarks.shape == (2, 2, 21, 3)
    assert arrays.image_landmarks.dtype == arrays.world_landmarks.dtype == np.float32
    assert arrays.presence_mask.dtype == np.bool_
    assert arrays.timestamps_ms.dtype == np.int64
    assert arrays.timestamps_ms.tolist() == [50, 100]
    assert arrays.presence_mask.tolist() == [[True, False], [False, False]]
    assert np.isnan(arrays.image_landmarks[~arrays.presence_mask]).all()
    assert np.isnan(arrays.world_landmarks).all()
    assert np.isnan(arrays.handedness_score[~arrays.presence_mask]).all()


def test_mirrored_pixels_swap_assignment_only():
    result = parse_hand_result(raw([points(0.8)], [[prediction("Left")]]), 1, 0, mirrored_input=True)
    assert result.presence_mask.tolist() == [False, True]
    assert result.image_landmarks[1, 0, 0] == pytest.approx(0.8)
    assert result.detections == [{
        "index": 0, "assigned_slot": "RIGHT", "proposed_slot": "RIGHT",
        "predictions": [{"label": "Left", "score": 0.9}],
    }]


def test_duplicates_keep_both_diagnostics_without_guessing_identity():
    result = parse_hand_result(raw(
        [points(0.2), points(0.5)], [[prediction("Left", 0.95)], [prediction("Left", 0.7)]],
    ), 1, 0)
    assert result.ambiguous_handedness
    assert not result.presence_mask.any()
    assert len(result.detections) == 2
    for entry, score, x in zip(result.detections, (0.95, 0.7), (0.2, 0.5), strict=True):
        assert entry["predictions"] == [{"label": "Left", "score": score}]
        assert entry["assigned_slot"] is None
        assert entry["unassigned_image_landmarks"][0][0] == pytest.approx(x)
    assert {warning["code"] for warning in result.warnings} == {"DUPLICATE_HANDEDNESS"}


def test_equal_scores_and_unknown_labels_are_ambiguous():
    result = parse_hand_result(raw(
        [points(), points()], [[prediction("Left", 0.5), prediction("Right", 0.5)], [prediction("unknown")]],
    ), 2, 100)
    assert result.ambiguous_handedness and not result.presence_mask.any()
    assert len(result.detections[0]["predictions"]) == 2
    assert result.detections[1]["predictions"][0]["label"] == "unknown"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e99, None])
def test_invalid_coordinates_are_flagged_without_contaminating_arrays(value):
    result = parse_hand_result(raw(
        [points(value), points()], [[prediction("Left")], [prediction("Right")]], [points(), points(value)],
    ), 0, 0)
    assert result.invalid_landmark_values
    assert result.presence_mask.tolist() == [False, True]
    assert np.isnan(result.image_landmarks[0]).all()
    assert np.isnan(result.world_landmarks[1]).all()
    assert {warning["landmarks"] for warning in result.warnings} == {"image", "world"}


def test_unexpected_counts_are_observable():
    result = parse_hand_result(raw([points(count=20)], [[prediction("Left")]], [points(count=22)]), 0, 0)
    assert not result.presence_mask.any()
    assert [row["actual"] for row in result.warnings] == [20, 22]
    result = parse_hand_result(raw([points()] * 3, [[prediction("Left")]] * 3), 0, 0)
    assert len(result.detections) == 3
    assert result.ambiguous_handedness and not result.presence_mask.any()
    assert result.warnings[0]["code"] == "UNEXPECTED_DETECTION_COUNTS"


def test_time_alignment_and_shapes_are_checked():
    frame = parse_hand_result(raw(), 1, 30)
    with pytest.raises(ValueError, match="strictly increasing"):
        HandArrays.from_frames([frame, frame])
    with pytest.raises(ValueError, match="shape"):
        HandArrays(
            np.zeros((1, 1, 21, 3), dtype=np.float32), frame.world_landmarks[None],
            frame.presence_mask[None], frame.handedness_score[None], np.array([30], dtype=np.int64),
        )
