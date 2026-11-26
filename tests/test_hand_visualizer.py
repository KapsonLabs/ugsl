from unittest.mock import Mock

import cv2
import numpy as np

from landmarks.hand_parser import parse_hand_result
from landmarks.hand_visualizer import WINDOW, HandLandmarkVisualizer
from tests.test_hand_extractor import frame
from tests.test_hand_parser import points, prediction, raw


def test_render_rgb_overlay_or_black_background_without_mutating_source():
    rgb = np.broadcast_to(np.array([110, 60, 25], dtype=np.uint8), (180, 240, 3)).copy()
    decoded = frame(125, rgb, 3)
    result = parse_hand_result(raw([points(0.3), points(0.7)], [[prediction("Left")], [prediction("Right")]]), 3, 125)
    image = HandLandmarkVisualizer().render(decoded, result)
    black = HandLandmarkVisualizer(skeleton_only=True).render(decoded, result)
    assert image.shape == black.shape == rgb.shape
    assert image[-1, -1].tolist() == [25, 60, 110]
    assert black[-1, -1].tolist() == [0, 0, 0]
    assert black[134, 72:79].any() and black[134, 167:174].any()
    assert np.all(rgb == [110, 60, 25])


def test_playback_shows_frame_quits_and_closes_window(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":test")
    named = Mock()
    destroyed = Mock()
    displayed = Mock()
    monkeypatch.setattr(cv2, "namedWindow", named)
    monkeypatch.setattr(cv2, "destroyWindow", destroyed)
    monkeypatch.setattr(cv2, "imshow", displayed)
    monkeypatch.setattr(cv2, "waitKeyEx", Mock(return_value=ord("q")))
    with HandLandmarkVisualizer() as playback:
        assert not playback.show(frame(), parse_hand_result(raw(), 0, 0))
    named.assert_called_once_with(WINDOW, cv2.WINDOW_NORMAL)
    assert displayed.call_args.args[0] == WINDOW
    destroyed.assert_called_once_with(WINDOW)
