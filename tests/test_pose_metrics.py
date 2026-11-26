import pytest

from config import PoseExtractionConfig
from landmarks.pose_metrics import PoseQualityMetrics, aggregate_pose_metrics
from tests.test_pose_parser import parsed, points, raw


def test_coverage_scores_and_performance_use_aligned_frames():
    a = points()
    a[0].visibility = 0.3
    a[0].presence = 0.2
    a[25].visibility = None
    b = points()
    b[0].visibility = 0.7
    b[0].presence = None
    frames = [
        parsed(raw(a, points()), 0, 0), parsed(raw(), 1, 100),
        parsed(raw(), 2, 200), parsed(raw(b), 3, 400), parsed(raw(), 5, 500),
    ]
    metrics, warnings = PoseQualityMetrics(PoseExtractionConfig()).evaluate(frames, 2, 0.25)
    assert not warnings
    assert metrics["pose_detection_coverage"] == 2 / 5
    assert metrics["frames_with_available_world_landmarks"] == 1
    assert metrics["average_visibility_per_landmark"][0] == pytest.approx(0.5)
    assert metrics["average_presence_per_landmark"][0] == pytest.approx(0.2)
    assert metrics["visibility_coverage_per_landmark"][0] == 1 / 5
    assert metrics["presence_coverage_per_landmark"][0] == 0
    assert metrics["visibility_coverage_by_group"]["upper_body"] == 49 / 125
    assert metrics["longest_consecutive_frames_without_pose"] == 2
    assert metrics["missing_pose_interval_count"] == 2 and metrics["missing_pose_duration_ms"] == 300
    assert metrics["missing_pose_intervals"][1]["duration_ms"] == 0
    assert metrics["model_initialization_seconds"] == 0.25
    assert metrics["average_processing_time_per_frame_ms"] == 400
    assert metrics["effective_throughput_fps"] == 2.5
    combined = aggregate_pose_metrics([metrics, metrics])
    assert combined["total_processed_frames"] == 10
    assert combined["pose_detection_coverage"] == 2 / 5


def test_displacement_score_thresholds_and_timestamp_discontinuities():
    latest = points(offset=0.35)
    latest[0].presence = 0.3
    frames = [
        parsed(raw(points()), 0, 0),
        parsed(raw(latest), 1, 100),
        parsed(raw(points()), 2, 500),
    ]
    metrics, warnings = PoseQualityMetrics(PoseExtractionConfig()).evaluate(frames, 1, 0.1)
    assert metrics["sudden_displacement_warnings"] == metrics["timestamp_discontinuities"] == 1
    assert warnings[0]["code"] == "SUDDEN_POSE_DISPLACEMENT" and 0 not in warnings[0]["indices"]
    assert warnings[0]["max_displacement"] == pytest.approx(0.35)
    assert warnings[1]["code"] == "POSE_TIMESTAMP_GAP" and warnings[1]["gap_ms"] == 400
    slow_metrics, _ = PoseQualityMetrics(PoseExtractionConfig()).evaluate(frames, 1, 0.1, nominal_fps=4)
    assert slow_metrics["timestamp_discontinuities"] == 0


def test_empty_results_have_no_defined_coverage():
    metrics, warnings = PoseQualityMetrics(PoseExtractionConfig()).evaluate([], 0, 0)
    assert metrics["pose_detection_coverage"] is None and metrics["effective_throughput_fps"] is None
    assert metrics["average_visibility_per_landmark"] == [None] * 33 and not warnings
