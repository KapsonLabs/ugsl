"""Per-landmark coverage and temporal diagnostics for pose extraction."""

from itertools import pairwise

import numpy as np

from landmarks.schemas import POSE_GROUPS, POSE_LANDMARK_COUNT, PoseFrame

COUNT_FIELDS = (
    "total_processed_frames", "frames_with_detected_pose", "frames_without_detected_pose",
    "invalid_extraction_results", "frames_with_available_world_landmarks", "frames_with_raw_pose",
    "missing_pose_interval_count", "timestamp_discontinuities", "sudden_displacement_warnings",
)


def _performance(count, processing_seconds, initialization_seconds):
    return {
        "processing_duration_seconds": processing_seconds,
        "model_initialization_seconds": initialization_seconds,
        "average_processing_time_per_frame_ms": 1000 * processing_seconds / count if count else None,
        "effective_throughput_fps": count / processing_seconds if processing_seconds > 0 else None,
    }


def _averages(scores):
    seen = np.isfinite(scores).sum(axis=0)
    totals = np.nansum(scores, axis=0, dtype=np.float64)
    return [float(total / count) if count else None for total, count in zip(totals, seen, strict=True)]


def _intervals(frames):
    intervals = []
    start = None
    for i, frame in enumerate(frames):
        if not frame.pose_detected and start is None:
            start = i
        if start is not None and (frame.pose_detected or i == len(frames) - 1):
            end = i if frame.pose_detected else i + 1
            intervals.append({
                "start_frame_index": frames[start].frame_index,
                "last_missing_frame_index": frames[end - 1].frame_index,
                "frame_count": end - start,
                "start_ms": frames[start].timestamp_ms,
                "end_ms": frame.timestamp_ms,
                "duration_ms": float(frame.source_timestamp - frames[start].source_timestamp),
            })
            start = None
    return intervals


class PoseQualityMetrics:
    """Measure observable model output. The metrics are diagnostic."""

    def __init__(self, config):
        self.config = config

    def evaluate(
        self, frames: list[PoseFrame], processing_seconds: float, initialization_seconds: float,
        *, nominal_fps: float | None = None,
    ) -> tuple[dict, list[dict]]:
        count = len(frames)
        pose_count = sum(frame.pose_detected for frame in frames)
        if frames:
            visibility = np.stack([frame.visibility for frame in frames])
            presence = np.stack([frame.presence_scores for frame in frames])
        else:
            visibility = presence = np.empty((0, POSE_LANDMARK_COUNT), dtype=np.float32)
        visible = visibility >= self.config.visibility_threshold
        present = presence >= self.config.presence_threshold

        time_gap_ms = max(self.config.timestamp_gap_ms, 2000 / nominal_fps) if nominal_fps else self.config.timestamp_gap_ms
        warnings = []
        for i, (previous, current) in enumerate(pairwise(frames), start=1):
            gap = float(current.source_timestamp - previous.source_timestamp)
            if gap > time_gap_ms:
                warnings.append({
                    "code": "POSE_TIMESTAMP_GAP", "previous_frame_index": previous.frame_index,
                    "frame_index": current.frame_index, "gap_ms": gap, "threshold_ms": time_gap_ms,
                })
                continue
            if previous.pose_detected and current.pose_detected:
                upper = list(POSE_GROUPS["upper_body"])
                reliable = visible[i - 1, upper] & visible[i, upper] & present[i - 1, upper] & present[i, upper]
                positions = current.image_landmarks[upper, :2].astype(np.float64)
                displacement = np.linalg.norm(positions - previous.image_landmarks[upper, :2], axis=1)
                flagged = np.flatnonzero(reliable & (displacement > self.config.displacement_threshold))
                if flagged.size:
                    warnings.append({
                        "code": "SUDDEN_POSE_DISPLACEMENT", "frame_index": current.frame_index,
                        "indices": [upper[index] for index in flagged],
                        "max_displacement": float(displacement[flagged].max()),
                    })

        intervals = _intervals(frames)
        return {
            "total_processed_frames": count,
            "frames_with_detected_pose": pose_count,
            "frames_without_detected_pose": count - pose_count,
            "frames_with_raw_pose": sum(frame.raw_pose_count > 0 for frame in frames),
            "pose_detection_coverage": pose_count / count if count else None,
            "invalid_extraction_results": sum(frame.invalid for frame in frames),
            "frames_with_available_world_landmarks": sum(frame.world_available for frame in frames),
            "average_visibility_per_landmark": _averages(visibility),
            "average_presence_per_landmark": _averages(presence),
            "visibility_coverage_per_landmark": visible.mean(axis=0).tolist() if count else [None] * POSE_LANDMARK_COUNT,
            "presence_coverage_per_landmark": present.mean(axis=0).tolist() if count else [None] * POSE_LANDMARK_COUNT,
            "visibility_coverage_by_group": {
                group: float(visible[:, indices].mean()) if count else None for group, indices in POSE_GROUPS.items()
            },
            "longest_consecutive_frames_without_pose": max((row["frame_count"] for row in intervals), default=0),
            "missing_pose_interval_count": len(intervals),
            "missing_pose_duration_ms": sum(row["duration_ms"] for row in intervals),
            "missing_pose_intervals": intervals,
            "timestamp_discontinuities": sum(row["code"] == "POSE_TIMESTAMP_GAP" for row in warnings),
            "sudden_displacement_warnings": sum(row["code"] == "SUDDEN_POSE_DISPLACEMENT" for row in warnings),
            **_performance(count, processing_seconds, initialization_seconds),
        }, warnings


def aggregate_pose_metrics(rows: list[dict]) -> dict:
    counts = {field: sum(row[field] for row in rows) for field in COUNT_FIELDS}
    count = counts["total_processed_frames"]
    return {
        **counts,
        "pose_detection_coverage": counts["frames_with_detected_pose"] / count if count else None,
        "missing_pose_duration_ms": sum(row["missing_pose_duration_ms"] for row in rows),
        "longest_consecutive_frames_without_pose": max(
            (row["longest_consecutive_frames_without_pose"] for row in rows), default=0,
        ),
        "visibility_coverage_by_group": {
            group: sum(
                (row["visibility_coverage_by_group"][group] or 0) * row["total_processed_frames"] for row in rows
            ) / count if count else None
            for group in POSE_GROUPS
        },
        **_performance(
            count, sum(row["processing_duration_seconds"] for row in rows),
            sum(row["model_initialization_seconds"] for row in rows),
        ),
    }
