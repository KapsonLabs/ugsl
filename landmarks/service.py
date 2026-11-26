"""Run decoding and sampling on approved exports or on a diagnostic input."""

import json
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import av

from dataset_export.reader import DatasetReader
from landmarks.decoder import VideoDecoder
from landmarks.exceptions import (
    DatasetAccessError, EmptyInterval, FrameProcessingError, HandInferenceError, PoseInferenceError,
    SampleNotFound, SourceVideoError,
)
from landmarks.hand_metrics import aggregate_hand_metrics, hand_metrics, no_hand_warnings
from landmarks.hands import HandExtractor
from landmarks.pose import PoseExtractor
from landmarks.pose_metrics import PoseQualityMetrics, aggregate_pose_metrics
from landmarks.sampler import FrameSampler
from landmarks.schemas import (
    HAND_SCHEMA_VERSION, HAND_SLOTS, POSE_SCHEMA_VERSION, BatchSummary, DecodedFrame, HandArrays,
    HandExtractionResult, PoseArrays, PoseExtractionResult, ProcessingSummary,
)
from registry import datasets
from registry.database import Database, RegistryError


class ProcessingRun:
    """Single-pass frames and a summary. Use as a context manager for early exit."""

    def __init__(self, summary: ProcessingSummary, frames: Iterator[DecodedFrame]):
        self.summary = summary
        self.frames = frames

    def __iter__(self):
        return self.frames

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.frames.close()
        if self.summary.status in {"PENDING", "PROCESSING"}:
            self.summary.status = "CANCELLED"


class DecodingService:
    def __init__(self, reader: DatasetReader, *, fps: float = 30, all_frames: bool = False):
        self.reader = reader
        FrameSampler(fps, all_frames=all_frames)
        self.fps = fps
        self.all_frames = all_frames

    @classmethod
    def from_registry(
        cls, db: Database, name: str, version: str, *, storage_root: Path | None = None,
        fps: float = 30, all_frames: bool = False,
    ):
        try:
            entry = datasets.show(Database(db.path, read_only=True), name, version)
            root = storage_root or json.loads(entry.configuration_json)["storage_root"]
            reader = DatasetReader(entry.output_path, storage_root=root, registry_path=db.path)
            if reader.metadata["dataset_id"] != entry.id:
                raise RegistryError("The export folder does not match the requested dataset version.")
        except (RegistryError, OSError, ValueError, KeyError) as exc:
            raise DatasetAccessError(str(exc)) from exc
        return cls(reader, fps=fps, all_frames=all_frames)

    def process(self, sample_id: str) -> ProcessingRun:
        requested = sample_id.strip().lower()
        try:
            with closing(self.reader.iter_records()) as records:
                for record in records:
                    if record.sample_id == requested:
                        return self._run(record)
        except (RegistryError, OSError) as exc:
            raise DatasetAccessError(str(exc)) from exc
        raise SampleNotFound(f"Sample {sample_id} is absent from this exported dataset.")

    def _run(self, record) -> ProcessingRun:
        summary = ProcessingSummary(
            record.sample_id, dataset_name=self.reader.metadata["name"], dataset_version=self.reader.metadata["version"],
            source_video_path=str(record.video_path),
            dataset_id=self.reader.metadata["dataset_id"], annotation_id=record.annotation.id,
            start_ms=record.annotation.start_ms, end_ms=record.annotation.end_ms,
            width=record.width, height=record.height, source_fps=record.average_fps, rotation_degrees=record.rotation,
        )

        def check_permission():
            try:
                self.reader.check_permission()
            except (RegistryError, OSError) as exc:
                raise DatasetAccessError(str(exc)) from exc

        def source_decoder():
            check_permission()
            try:
                sample = self.reader.resolve_sample(record)
                path = Path(sample.video_path)
                summary.source_video_path = str(path)
            except (RegistryError, OSError) as exc:
                raise SourceVideoError(str(exc)) from exc
            check_permission()
            return VideoDecoder(
                path, sample.sample_id, start_ms=sample.annotation.start_ms, end_ms=sample.annotation.end_ms,
                expected_frames=sample.frame_count, expected_duration_ms=sample.duration_ms,
                expected_sha256=sample.sha256, expected_size=sample.file_size_bytes,
                rotation_degrees=sample.rotation, summary=summary,
            )

        return _processing_run(summary, source_decoder, self.fps, self.all_frames, check_permission)

    def iter_runs(self) -> Iterator[ProcessingRun]:
        with closing(self.reader.iter_records()) as records:
            for record in records:
                yield self._run(record)

    def validate_dataset(self) -> BatchSummary:
        batch = BatchSummary(
            self.reader.metadata["name"], self.reader.metadata["version"], self.reader.metadata["total_samples"],
        )
        try:
            for record in self.reader.iter_records():
                with self._run(record) as run:
                    try:
                        for _ in run:
                            pass
                    except DatasetAccessError as exc:
                        batch.samples.append(run.summary)
                        batch.dataset_error = str(exc)
                        break
                    except FrameProcessingError:
                        pass
                batch.samples.append(run.summary)
        except (RegistryError, OSError) as exc:
            batch.dataset_error = str(exc)
        return batch


def inspect_file(path: Path, *, fps: float = 30, all_frames: bool = True) -> ProcessingRun:
    """A standalone diagnostic. It never constructs a dataset processing service."""
    path = Path(path)
    summary = ProcessingSummary(path.stem, source_video_path=str(path))
    return _processing_run(summary, lambda: VideoDecoder(path, summary.sample_id, summary=summary), fps, all_frames)


def _processing_run(summary, decoder_factory, fps, all_frames, check_permission=None) -> ProcessingRun:
    sampler = FrameSampler(fps, all_frames=all_frames)
    summary.target_fps = None if all_frames else fps
    summary.decoder_configuration = {
        "decoder": "PyAV", "pyav_version": av.__version__, "pixel_format": "rgb24", "apply_rotation": False,
        "all_frames": all_frames, "sampling_policy": "all_source_frames" if all_frames else "nearest_grid_slot",
        "timestamp_rounding_in_report": "floor_milliseconds",
    }

    def frames():
        summary.status = "PROCESSING"
        try:
            decoder = decoder_factory()
            with closing(decoder.decode()) as decoded, closing(sampler.sample(decoded, duration_ms=decoder.duration_ms)) as sampled:
                for frame in sampled:
                    if check_permission is not None:
                        check_permission()
                    if summary.first_selected_timestamp_ms is None:
                        summary.first_selected_timestamp_ms = frame.timestamp_ms // 1
                    summary.last_selected_timestamp_ms = frame.timestamp_ms // 1
                    summary.selected_frames += 1
                    yield frame
            if check_permission is not None:
                check_permission()
            if decoder.duration_ms:
                summary.effective_output_fps = round(float(summary.selected_frames * 1000 / decoder.duration_ms), 6)
            summary.status = "PROCESSED"
        except FrameProcessingError as exc:
            summary.status, summary.error_code, summary.error = "FAILED", exc.code, str(exc)
            raise
        finally:
            summary.skipped_frames = summary.decoded_frames - summary.selected_frames
            summary.sampling_gaps = list(sampler.gaps)
            if summary.status == "PROCESSING":
                summary.status = "CANCELLED"

    return ProcessingRun(summary, frames())


class LandmarkExtractionService:
    """Share the single-video and approved batch lifecycle across landmark tasks."""

    def __init__(self, decoding: DecodingService | None, config, *, extractor_factory=None):
        self.decoding = decoding
        self.config = config
        self.extractor_factory = extractor_factory or self.extractor_type

    def process(self, sample_id: str, *, on_frame=None):
        if self.decoding is None:
            raise DatasetAccessError("An approved dataset is required to process a sample ID.")
        try:
            run = self.decoding.process(sample_id)
        except FrameProcessingError as exc:
            report = self._report(sample_id, diagnostic=False)
            self._record_failure(report, exc)
            return self.result_type(report)
        return self._extract(run, diagnostic=False, on_frame=on_frame)

    def inspect_file(self, path: Path, *, fps: float = 30, all_frames: bool = False):
        return self._extract(inspect_file(path, fps=fps, all_frames=all_frames), diagnostic=True)

    def inspect_dataset(self) -> dict:
        if self.decoding is None:
            raise DatasetAccessError("An approved dataset is required to process a collection.")
        metadata = self.decoding.reader.metadata
        samples = []
        dataset_error = None
        cancelled = False
        try:
            with closing(self.decoding.iter_runs()) as runs:
                for run in runs:
                    result = self._extract(run, diagnostic=False)
                    samples.append(result.report)
                    if result.report["status"] == "CANCELLED":
                        cancelled = True
                        break
                    if result.report["error_code"] == DatasetAccessError.code:
                        dataset_error = result.report["error"]
                        break
        except (RegistryError, OSError) as exc:
            dataset_error = str(exc)
        except KeyboardInterrupt:
            cancelled = True
        if dataset_error:
            for sample in samples:
                sample["frames"] = []
        complete = [sample for sample in samples if sample["status"] == "PROCESSED"]
        return {
            "schema_version": self.schema_version,
            "dataset": metadata["name"], "version": metadata["version"],
            "total_samples": metadata["total_samples"], "processed": len(complete),
            "failed": sum(sample["status"] == "FAILED" for sample in samples),
            "cancelled": cancelled,
            "dataset_error": dataset_error,
            "metrics": self.aggregate_metrics([sample["metrics"] for sample in complete]),
            "samples": samples,
        }

    def _report(self, sample_id: str, *, diagnostic: bool) -> dict:
        report = {
            "sample_id": sample_id,
            "source": "diagnostic_file" if diagnostic else "approved_dataset",
            "schema_version": self.schema_version,
            "model_sha256": None,
            "mediapipe_version": None,
            "configuration": {
                **asdict(self.config), "model_path": str(self.config.model_path),
                "running_mode": "VIDEO", "timestamp_rounding": "floor_milliseconds", **self.task_configuration,
            },
            "status": "PENDING", "error_code": None, "error": None,
            "warnings": [], "frames": [], "metrics": self.aggregate_metrics([]),
        }
        self._enrich_report(report)
        return report

    def _enrich_report(self, report):
        pass

    @staticmethod
    def _record_failure(report: dict, exc: FrameProcessingError) -> None:
        report.update(status="FAILED", error_code=exc.code, error=str(exc))
        report["warnings"].append({"code": exc.code, "message": str(exc)})
        if isinstance(exc, DatasetAccessError):
            report["frames"] = []

    def _extract(self, run: ProcessingRun, *, diagnostic: bool, on_frame=None):
        report = self._report(run.summary.sample_id, diagnostic=diagnostic)
        started = perf_counter()
        display_seconds = 0.0
        extractor = self.extractor_factory(self.config)
        arrays = None
        frames = []
        try:
            with run, extractor:
                for decoded in run:
                    frame = extractor.extract(decoded)
                    frames.append(frame)
                    if on_frame is not None:
                        display_started = perf_counter()
                        try:
                            if not on_frame(decoded, frame):
                                break
                        finally:
                            display_seconds += perf_counter() - display_started
                else:
                    if not frames:
                        raise EmptyInterval(f"No sampled frames were available for {self.kind} extraction.")
                    try:
                        arrays = self.array_type.from_frames(frames)
                    except ValueError as exc:
                        raise self.inference_error(str(exc)) from exc
                    report["status"] = "PROCESSED"
                if arrays is None:
                    report["status"] = "CANCELLED"
        except FrameProcessingError as exc:
            arrays = None
            self._record_failure(report, exc)
        except KeyboardInterrupt:
            arrays = None
            report.update(status="CANCELLED", error_code="INTERRUPTED", error=f"{self.kind.capitalize()} extraction was interrupted.")
            report["warnings"].append({"code": "INTERRUPTED"})
        finally:
            report["model_sha256"] = extractor.model_sha256
            report["mediapipe_version"] = extractor.mediapipe_version
            report["decoding"] = asdict(run.summary)
            for issue in run.summary.timestamp_irregularities:
                report["warnings"].append({"code": "DECODER_TIMESTAMP_IRREGULARITY", **asdict(issue)})
            seconds = max(0.0, perf_counter() - started - display_seconds)
            metrics, warnings = self._analyze(frames, seconds, extractor, run.summary, report["status"])
            report["metrics"] = metrics
            report["warnings"].extend(warnings)
            if report["error_code"] != DatasetAccessError.code:
                report["frames"] = [self._frame_report(frame) for frame in frames]
        return self.result_type(report, arrays)


class HandExtractionService(LandmarkExtractionService):
    kind = "hand"
    extractor_type = HandExtractor
    array_type = HandArrays
    result_type = HandExtractionResult
    inference_error = HandInferenceError
    schema_version = HAND_SCHEMA_VERSION
    task_configuration = {"num_hands": 2}
    aggregate_metrics = staticmethod(aggregate_hand_metrics)

    def _enrich_report(self, report):
        report["slots"] = list(HAND_SLOTS)

    def _analyze(self, frames, seconds, extractor, summary, status):
        return hand_metrics(frames, seconds), no_hand_warnings(
            frames, self.config.no_hand_warning_ms, completed=status == "PROCESSED",
        )

    @staticmethod
    def _frame_report(frame):
        return {
            "frame_index": frame.frame_index, "timestamp_ms": frame.timestamp_ms,
            "raw_image_detections": frame.raw_image_detections,
            "ambiguous_handedness": frame.ambiguous_handedness,
            "invalid_landmark_values": frame.invalid_landmark_values,
            "detections": frame.detections, "warnings": frame.warnings,
        }


class PoseExtractionService(LandmarkExtractionService):
    kind = "pose"
    extractor_type = PoseExtractor
    array_type = PoseArrays
    result_type = PoseExtractionResult
    inference_error = PoseInferenceError
    schema_version = POSE_SCHEMA_VERSION
    task_configuration = {
        "num_poses": 1, "output_segmentation_masks": False,
        "inference_clock": "source_ms_with_explicit_rebase_or_collision_adjustment",
    }
    aggregate_metrics = staticmethod(aggregate_pose_metrics)

    def __init__(self, decoding, config, *, extractor_factory=None, multiple_participant_samples=()):
        super().__init__(decoding, config, extractor_factory=extractor_factory)
        self.multiple_participant_samples = {sample.strip().lower() for sample in multiple_participant_samples}

    def _enrich_report(self, report):
        report["model_variant"] = self.config.model_variant
        report["annotation_id"] = None
        report["dataset"] = None
        if report["source"] == "approved_dataset":
            metadata = self.decoding.reader.metadata
            report["dataset"] = {
                "id": metadata["dataset_id"], "name": metadata["name"], "version": metadata["version"],
            }
        if report["sample_id"].strip().lower() in self.multiple_participant_samples:
            report["warnings"].append({"code": "POSSIBLE_MULTIPLE_PARTICIPANTS", "source": "human_review"})

    def _extract(self, run, **kwargs):
        result = super()._extract(run, **kwargs)
        result.report["annotation_id"] = run.summary.annotation_id
        return result

    def _analyze(self, frames, seconds, extractor, summary, status):
        rates = [rate for rate in (summary.source_fps, summary.target_fps) if rate]
        metrics, warnings = PoseQualityMetrics(self.config).evaluate(
            frames, seconds, extractor.initialization_seconds, nominal_fps=min(rates) if rates else None,
        )
        metrics["timestamp_discontinuities"] += len(summary.timestamp_irregularities)
        if status == "PROCESSED" and not metrics["frames_with_raw_pose"]:
            warnings.append({"code": "NO_POSES_DETECTED"})
        return metrics, warnings

    @staticmethod
    def _frame_report(frame):
        warnings = list(frame.warnings)
        if frame.inference_timestamp_ms != frame.timestamp_ms:
            warnings.append({"code": "POSE_INFERENCE_TIMESTAMP_ADJUSTED"})
        return {
            "frame_index": frame.frame_index, "timestamp_ms": frame.timestamp_ms,
            "source_pts_ms": {
                "numerator": frame.source_timestamp.numerator, "denominator": frame.source_timestamp.denominator,
            },
            "inference_timestamp_ms": frame.inference_timestamp_ms,
            "pose_detected": frame.pose_detected, "world_available": frame.world_available,
            "raw_pose_count": frame.raw_pose_count, "invalid": frame.invalid, "warnings": warnings,
        }
