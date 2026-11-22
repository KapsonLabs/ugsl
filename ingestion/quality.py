"""Technical thresholds and bounded visual heuristics for isolated signs."""

from dataclasses import asdict, dataclass

import cv2

from config import QualityConfig
from ingestion.schemas import ValidationCheck, VideoMetadata


def technical_checks(video: VideoMetadata, config: QualityConfig) -> list[ValidationCheck]:
    long_edge, short_edge = sorted((video.width or 0, video.height or 0), reverse=True)
    required_long, required_short = sorted((config.min_width, config.min_height), reverse=True)
    checks = [ValidationCheck(
        "minimum_resolution", "PASS" if long_edge >= required_long and short_edge >= required_short else "FAIL",
        f"{video.width}x{video.height}" if video.width and video.height else None,
    )]
    for name, actual, minimum, maximum in (
        ("frame_rate", video.average_fps, config.min_fps, None),
        ("frame_count", video.frame_count, config.min_frame_count, None),
        ("duration", video.duration_seconds, config.min_duration_seconds, config.max_duration_seconds),
        ("file_size_bytes", video.file_size_bytes, 1, config.max_file_size_bytes),
    ):
        passed = actual is not None and actual >= minimum and (maximum is None or actual <= maximum)
        checks.append(ValidationCheck(name, "PASS" if passed else "FAIL", actual, minimum, maximum))
    return checks


@dataclass(frozen=True)
class VisualMetrics:
    sampled_frames: int
    mean_brightness: float
    brightness_variability: float
    underexposed_percent: float
    overexposed_percent: float
    mean_sharpness: float
    low_sharpness_percent: float
    mean_motion: float | None
    peak_motion: float | None


class VisualQualityAnalyzer:
    def __init__(self, max_samples=8):
        self.max_samples = max_samples

    def analyze(self, path, video: VideoMetadata) -> tuple[dict, list[ValidationCheck], list[str]]:
        from statistics import fmean, pstdev

        brightness, under, over, sharpness, motion = [], [], [], [], []
        capture = None
        previous = None
        samples = min(self.max_samples, video.frame_count or self.max_samples)
        try:
            capture = cv2.VideoCapture(str(path))
            if capture.isOpened():
                for index in range(samples):
                    timestamp = video.duration_seconds * (index + 0.5) / samples
                    if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                        continue
                    ok, frame = capture.read()
                    if not ok or frame is None or not frame.size:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness.append(float(gray.mean()))
                    under.append(float((gray < 25).mean()) * 100)
                    over.append(float((gray > 230).mean()) * 100)
                    sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                    if previous is not None and previous.shape == gray.shape:
                        motion.append(float(cv2.absdiff(previous, gray).mean()))
                    previous = gray
        except cv2.error:
            pass
        finally:
            if capture is not None:
                capture.release()

        checks, warnings = [], []
        if len(brightness) < samples:
            warnings.append("OpenCV could not sample all requested visual timestamps.")
            checks.append(ValidationCheck("visual_sampling", "WARNING", len(brightness)))
        if not brightness:
            return {}, checks, warnings

        metrics = VisualMetrics(
            len(brightness), fmean(brightness), pstdev(brightness), fmean(under), fmean(over),
            fmean(sharpness), sum(value < 50 for value in sharpness) * 100 / len(sharpness),
            fmean(motion) if motion else None, max(motion) if motion else None,
        )
        for name, actual, flagged, message in (
            ("visual_underexposure", metrics.underexposed_percent, metrics.underexposed_percent > 35,
             "A large portion of sampled pixels is underexposed."),
            ("visual_overexposure", metrics.overexposed_percent, metrics.overexposed_percent > 35,
             "A large portion of sampled pixels is overexposed."),
            ("visual_sharpness", metrics.low_sharpness_percent, metrics.low_sharpness_percent >= 50,
             "Several sampled frames have low sharpness."),
            ("visual_motion", metrics.mean_motion, metrics.mean_motion is not None and metrics.mean_motion < 2,
             "Sampled motion is low. This may be appropriate for a static sign."),
        ):
            checks.append(ValidationCheck(name, "WARNING" if flagged else "PASS", actual))
            if flagged:
                warnings.append(message)
        return asdict(metrics), checks, warnings
