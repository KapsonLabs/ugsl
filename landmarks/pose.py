"""MediaPipe Tasks VIDEO pose inference for one signer per frame."""

from landmarks.exceptions import InvalidTimestamp, PoseInferenceError, PoseModelError
from landmarks.pose_parser import parse_pose_result
from landmarks.schemas import DecodedFrame, PoseFrame
from landmarks.video_landmarker import VideoLandmarker


class PoseExtractor(VideoLandmarker):
    """Own one pose tracking context for the duration of a recording."""

    kind = "pose"
    task = "PoseLandmarker"
    model_members = {"pose_detector.tflite", "pose_landmarks_detector.tflite"}
    model_error = PoseModelError
    inference_error = PoseInferenceError
    adjust_inference_clock = True

    def _task_options(self):
        return {
            "num_poses": 1,
            "min_pose_detection_confidence": self.config.min_detection_confidence,
            "min_pose_presence_confidence": self.config.min_presence_confidence,
            "min_tracking_confidence": self.config.min_tracking_confidence,
            "output_segmentation_masks": False,
        }

    def extract(self, frame: DecodedFrame) -> PoseFrame:
        if type(frame.frame_index) is not int or not 0 <= frame.frame_index < 2**63:
            raise InvalidTimestamp("Pose extraction requires a nonnegative int64 source frame index.")
        raw, source, timestamp, inference = self._infer(frame)
        return parse_pose_result(raw, frame.frame_index, source, timestamp, inference)
