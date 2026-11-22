"""Full-stream integrity checking and validation reports."""

import re
from dataclasses import asdict, replace
from pathlib import Path

from config import QualityConfig
from ingestion.exceptions import MediaError
from ingestion.metadata import INPUT_OPTIONS, VideoMetadataExtractor, local_video, media_command
from ingestion.quality import VisualQualityAnalyzer, technical_checks
from ingestion.schemas import ValidationCheck, ValidationReport, ValidationStatus, VideoMetadata


def failed_report(error: MediaError, check: str) -> ValidationReport:
    return ValidationReport(
        ValidationStatus.ERROR if error.operational else ValidationStatus.FAILED,
        [ValidationCheck(check, "ERROR" if error.operational else "FAIL")], [str(error)], [], {},
    )


class VideoIntegrityValidator:
    def __init__(self, config: QualityConfig):
        self.config = config

    def check_file(self, path: Path) -> Path:
        path = local_video(path)
        size = path.stat().st_size
        if not size:
            raise MediaError("EMPTY_FILE", "The video file is empty.")
        if size > self.config.max_file_size_bytes:
            raise MediaError("FILE_TOO_LARGE", "The video exceeds the configured file size limit.")
        return path

    def decode(self, path: Path, video: VideoMetadata) -> VideoMetadata:
        output = media_command(
            ["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-nostdin",
             *INPUT_OPTIONS, "-i", str(path), "-map", "0:V:0", "-an", "-sn", "-dn",
             "-fps_mode", "passthrough", "-f", "null", "-", "-progress", "pipe:1", "-nostats"],
            self.config.decode_timeout, "DECODE_FAILED", "FFmpeg could not decode the complete video stream.",
        )
        frames = re.findall(r"^frame=\s*(\d+)$", output, re.MULTILINE)
        if not frames or int(frames[-1]) == 0:
            raise MediaError("NO_USABLE_FRAMES", "The video has no decodable frames.")
        count = int(frames[-1])
        if video.frame_count is not None and count != video.frame_count:
            raise MediaError("FRAME_COUNT_MISMATCH", "Decoded frame count differs from container metadata.")
        times = re.findall(r"^out_time_us=(\d+)$", output, re.MULTILINE)
        decoded_duration = int(times[-1]) / 1_000_000 if times and int(times[-1]) else None
        if decoded_duration and video.stream_duration_seconds:
            tolerance = max(0.15, 2 / video.average_fps if video.average_fps else 0.25)
            if decoded_duration + tolerance < video.stream_duration_seconds:
                raise MediaError("TRUNCATED_VIDEO", "Decoded video ends before the declared stream duration.")
        return replace(video, frame_count=count,
                       duration_seconds=video.stream_duration_seconds or decoded_duration or video.duration_seconds)


class VideoValidator:
    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self.metadata = VideoMetadataExtractor(self.config.probe_timeout)
        self.integrity = VideoIntegrityValidator(self.config)
        self.visual = VisualQualityAnalyzer(self.config.visual_samples)

    def validate(self, path: Path) -> tuple[VideoMetadata | None, ValidationReport]:
        try:
            file = self.integrity.check_file(path)
            video = self.metadata.extract(file, count_frames=False)
            video = self.integrity.decode(file, video)
        except MediaError as exc:
            return None, failed_report(exc, "video_integrity")
        except OSError:
            error = MediaError("MEDIA_IO", "Could not read the video during validation.", operational=True)
            return None, failed_report(error, "video_integrity")
        checks = [ValidationCheck("video_integrity", "PASS"), *technical_checks(video, self.config)]
        errors = [
            f"{check.name.replace('_', ' ').capitalize()} "
            f"{'is missing or unknown' if check.actual is None else 'is outside the isolated-sign limits'}."
            for check in checks if check.status == "FAIL"
        ]
        visual, visual_checks, warnings = {}, [], []
        if not errors:
            visual, visual_checks, warnings = self.visual.analyze(file, video)
        status = ValidationStatus.FAILED if errors else (
            ValidationStatus.PASSED_WITH_WARNINGS if warnings else ValidationStatus.PASSED
        )
        report = ValidationReport(status, checks + visual_checks, errors, warnings, {"video": asdict(video), "visual": visual})
        return video, report
