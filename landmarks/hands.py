"""Synchronous MediaPipe Tasks VIDEO hand inference on Phase 2.1 frames."""

from landmarks.exceptions import HandInferenceError, HandModelError
from landmarks.hand_parser import parse_hand_result
from landmarks.schemas import DecodedFrame, HandFrame
from landmarks.video_landmarker import VideoLandmarker


class HandExtractor(VideoLandmarker):
    """Own one hand tracking context, closed on completion or interruption."""

    kind = "hand"
    task = "HandLandmarker"
    model_members = {"hand_detector.tflite", "hand_landmarks_detector.tflite"}
    model_error = HandModelError
    inference_error = HandInferenceError

    def _task_options(self):
        return {
            "num_hands": 2,
            "min_hand_detection_confidence": self.config.min_detection_confidence,
            "min_hand_presence_confidence": self.config.min_presence_confidence,
            "min_tracking_confidence": self.config.min_tracking_confidence,
        }

    def extract(self, frame: DecodedFrame) -> HandFrame:
        raw, _, timestamp, _ = self._infer(frame)
        return parse_hand_result(raw, frame.frame_index, timestamp, mirrored_input=self.config.mirrored_input)
