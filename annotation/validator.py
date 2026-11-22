"""Validate media eligibility and reviewer-supplied annotation fields."""

import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from annotation.models import AnnotationInput, TECHNICAL_STATUSES
from annotation.playback import probe_timeline
from config import QualityConfig, ReviewConfig
from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_file
from ingestion.metadata import local_video
from ingestion.storage import safe_path
from ingestion.validator import VideoValidator
from registry import reviewers, sessions, signers, vocabulary
from registry.database import RegistryError
from registry.repository import optional_text


class AnnotationValidator:
    def __init__(self, root: Path, config: ReviewConfig):
        self.root = root.resolve()
        self.config = config

    def media(self, connection, sample, *, verify=False):
        if sample.sample_type != "ISOLATED":
            raise RegistryError("This phase supports only isolated sign samples.")
        if sample.status == "EXCLUDED":
            raise RegistryError("This sample is excluded and cannot be annotated or reviewed.")
        if sample.technical_validation_status not in TECHNICAL_STATUSES:
            raise RegistryError("A sample needs PASSED or PASSED_WITH_WARNINGS technical validation.")
        session = sessions.find(connection, sample.session_id)
        signer = signers.find_by_id(connection, session.signer_id)
        if signer.consent_status != "GRANTED":
            raise RegistryError("This recording lacks current granted signer consent.")
        signers.require_recording_consent(connection, signer, sample.recorded_at)
        try:
            path = local_video(safe_path(self.root, sample.file_path))
            if path.stat().st_size != sample.file_size_bytes:
                raise RegistryError("The video size differs from the registry. Revalidate the recording.")
            if verify and sample.sha256 is not None and sha256_file(path) != sample.sha256:
                raise RegistryError("The video checksum differs from the registry. Revalidate the recording.")
            if verify and sample.sha256 is None:
                validator = VideoValidator()
                metadata = validator.metadata.extract(path, count_frames=False)
                validator.integrity.decode(path, metadata)
        except MediaError as exc:
            if exc.operational:
                raise RegistryError(str(exc)) from exc
            raise RegistryError("The recorded video is missing, unsafe, unreadable, or corrupt.") from exc
        except (OSError, RuntimeError) as exc:
            raise RegistryError("The recorded video is missing, unsafe, unreadable, or corrupt.") from exc
        return path, signer

    def duration_ms(self, sample, path: Path) -> int:
        if sample.duration_seconds is not None and math.isfinite(sample.duration_seconds) and sample.duration_seconds > 0:
            return int(Decimal(str(sample.duration_seconds)) * 1000)
        # The Phase 1.1 capture-loop duration is not the playback duration.
        return probe_timeline(path, timeout=QualityConfig().decode_timeout).duration_ms

    def interval(self, start_ms: int, end_ms: int, duration_ms: int) -> None:
        if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms < end_ms <= duration_ms:
            raise RegistryError(f"Annotation requires integer timestamps with 0 ≤ start < end ≤ {duration_ms} ms.")

    def annotation(self, connection, inputs: AnnotationInput, duration_ms: int) -> AnnotationInput:
        sign = vocabulary.find(connection, inputs.sign_id)
        if not sign.is_active:
            raise RegistryError("The selected sign or variant is inactive.")
        self.interval(inputs.start_ms, inputs.end_ms, duration_ms)
        confidence = inputs.confidence
        if confidence is not None and (
            type(confidence) not in (float, int) or not math.isfinite(confidence)
            or not self.config.confidence_min <= confidence <= self.config.confidence_max
        ):
            raise RegistryError(
                f"Reviewer confidence must be between {self.config.confidence_min:g} and {self.config.confidence_max:g}."
            )
        choices = {
            "hand_usage": ("ONE_HANDED", "TWO_HANDED", "UNKNOWN"),
            "dominant_hand": ("LEFT", "RIGHT", "BOTH", "UNKNOWN"),
            "movement_type": ("STATIC", "DYNAMIC", "UNKNOWN"),
            "facial_expression": ("PRESENT", "ABSENT", "UNKNOWN"),
            "completeness": ("COMPLETE", "INCOMPLETE", "UNKNOWN"),
        }
        labels = {name: reviewers.workflow_choice(getattr(inputs, name).strip(), name.replace("_", " ").capitalize(), allowed)
                  for name, allowed in choices.items()}
        return replace(inputs, sign_id=sign.id, notes=optional_text(inputs.notes), **labels)
