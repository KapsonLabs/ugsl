"""Summarize detection coverage, diagnostics, and runtime independently of sign labels."""

import numpy as np

from landmarks.schemas import HandFrame

COUNT_FIELDS = (
    "total_processed_frames", "frames_with_both_hands", "frames_with_only_left_hand",
    "frames_with_only_right_hand", "frames_with_no_hands", "frames_with_no_raw_detections",
    "ambiguous_handedness_assignments", "frames_with_invalid_landmark_values", "handedness_scores_count",
)


def _rates(counts: dict, seconds: float, score_total: float) -> dict:
    total = counts["total_processed_frames"]
    left = counts["frames_with_both_hands"] + counts["frames_with_only_left_hand"]
    right = counts["frames_with_both_hands"] + counts["frames_with_only_right_hand"]
    return {
        **counts,
        "overall_frame_coverage": (total - counts["frames_with_no_hands"]) / total if total else None,
        "left_hand_coverage": left / total if total else None,
        "right_hand_coverage": right / total if total else None,
        "average_handedness_score": score_total / counts["handedness_scores_count"]
        if counts["handedness_scores_count"] else None,
        "processing_duration_seconds": seconds,
        "average_processing_time_per_frame_ms": seconds * 1000 / total if total else None,
        "effective_throughput_fps": total / seconds if seconds > 0 else None,
    }


def hand_metrics(frames: list[HandFrame], seconds: float) -> dict:
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts["total_processed_frames"] = len(frames)
    score_total = 0.0
    for frame in frames:
        left, right = frame.presence_mask
        if left and right:
            counts["frames_with_both_hands"] += 1
        elif left:
            counts["frames_with_only_left_hand"] += 1
        elif right:
            counts["frames_with_only_right_hand"] += 1
        else:
            counts["frames_with_no_hands"] += 1
        counts["frames_with_no_raw_detections"] += not frame.raw_image_detections
        counts["ambiguous_handedness_assignments"] += frame.ambiguous_handedness
        counts["frames_with_invalid_landmark_values"] += frame.invalid_landmark_values
        counts["handedness_scores_count"] += int(frame.presence_mask.sum())
        score_total += float(np.nansum(frame.handedness_score, dtype=np.float64))
    return _rates(counts, seconds, score_total)


def aggregate_hand_metrics(metrics: list[dict]) -> dict:
    counts = {field: sum(row[field] for row in metrics) for field in COUNT_FIELDS}
    seconds = sum(row["processing_duration_seconds"] for row in metrics)
    total = sum((row["average_handedness_score"] or 0) * row["handedness_scores_count"] for row in metrics)
    return _rates(counts, seconds, total)


def no_hand_warnings(frames: list[HandFrame], minimum_ms: int, *, completed: bool) -> list[dict]:
    warnings = []
    if completed and frames and not any(frame.raw_image_detections for frame in frames):
        warnings.append({"code": "NO_HANDS_DETECTED"})
    first = last = None
    for frame in [*frames, None]:
        if frame is not None and not frame.raw_image_detections:
            first = first if first is not None else frame
            last = frame
            continue
        if first is not None:
            duration = last.timestamp_ms - first.timestamp_ms
            if duration >= minimum_ms:
                warnings.append({
                    "code": "LONG_NO_HAND_SEQUENCE", "first_frame_index": first.frame_index,
                    "last_frame_index": last.frame_index, "start_ms": first.timestamp_ms,
                    "end_ms": last.timestamp_ms, "duration_ms": duration,
                })
        first = last = None
    return warnings
