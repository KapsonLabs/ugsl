from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from landmarks.pose_visualizer import LEFT_COLOR, PoseLandmarkVisualizer, image_point
from tests.test_hand_extractor import frame
from tests.test_pose_parser import parsed, points, raw


@pytest.mark.parametrize(("xyz", "pixel"), [
    ((0.25, 0.5, -0.1), (60, 90)), ((0.999, 0.999, 0), (239, 179)),
    ((-0.01, 0.5, 0), None), ((1.2, 0.5, 0), None), ((0.5, float("nan"), 0), None),
])
def test_pixel_conversion(xyz, pixel):
    assert image_point(xyz, 240, 180) == pixel


def test_upper_full_and_visibility_filtered_connections(monkeypatch):
    circle = Mock()
    line = Mock()
    monkeypatch.setattr(cv2, "circle", circle)
    monkeypatch.setattr(cv2, "line", line)
    body = points()
    for index, point in enumerate(body):
        point.x = (index + 1) / 45
        point.y = 0.45
    body[13].visibility = 0.2
    body[17].x = -0.5
    pose = parsed(raw(body))
    visualizer = PoseLandmarkVisualizer(skeleton_only=True)
    visualizer.render(frame(), pose)
    assert circle.call_count == 23
    forbidden = {image_point(pose.image_landmarks[13], 240, 180)}
    assert all(not forbidden.intersection(call.args[1:3]) for call in line.call_args_list)
    assert all(call.args[1] != image_point(pose.image_landmarks[30], 240, 180) for call in circle.call_args_list)
    circle.reset_mock()
    PoseLandmarkVisualizer(full_body=True).render(frame(), pose)
    assert circle.call_count == 31
    circle.reset_mock()
    visualizer.render(frame(), parsed(raw()))
    circle.assert_not_called()


def test_mirrored_display_preserves_arrays_and_anatomical_color(monkeypatch):
    circle = Mock()
    monkeypatch.setattr(cv2, "circle", circle)
    image = np.full((180, 240, 3), [70, 20, 10], dtype=np.uint8)
    source = frame(pixels=image)
    landmarks = points(visibility=0.2)
    landmarks[15].visibility = 0.95
    pose = parsed(raw(landmarks))
    before = pose.image_landmarks.copy()
    overlay = PoseLandmarkVisualizer(mirror_preview=True, show_indices=True, show_visibility=True).render(source, pose)
    x, y = image_point(pose.image_landmarks[15], 240, 180)
    assert circle.call_args.args[1:4] == ((239 - x, y), 3, LEFT_COLOR)
    assert overlay[-1, -1].tolist() == [10, 20, 70]
    assert np.all(image == [70, 20, 10])
    np.testing.assert_array_equal(before, pose.image_landmarks)
