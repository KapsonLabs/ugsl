"""Opt-in development exports of approved hand landmarks."""

from pathlib import Path

from landmarks.debug_export import write_landmark_debug
from landmarks.exceptions import HandOutputError
from landmarks.schemas import HandExtractionResult


def export_debug(result: HandExtractionResult, root: Path, *, check_permission) -> Path:
    return write_landmark_debug(
        result, root, check_permission=check_permission,
        names=("hand_landmarks.npz", "extraction_report.json"),
        array_names=("image_landmarks", "world_landmarks", "presence_mask", "handedness_score", "timestamps_ms"),
        output_error=HandOutputError,
    )
