"""Opt-in development exports of approved pose landmarks."""

from dataclasses import fields
from pathlib import Path

from landmarks.debug_export import write_landmark_debug
from landmarks.exceptions import PoseOutputError
from landmarks.schemas import PoseArrays, PoseExtractionResult


def export_debug(result: PoseExtractionResult, root: Path, *, check_permission) -> Path:
    return write_landmark_debug(
        result, root, check_permission=check_permission,
        names=("pose_landmarks.npz", "pose_extraction_report.json"),
        array_names=tuple(field.name for field in fields(PoseArrays) if field.name != "schema_version"),
        output_error=PoseOutputError,
    )
