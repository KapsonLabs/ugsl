"""Opt-in development exports for approved samples."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import validate_identifier
from landmarks.exceptions import DatasetAccessError, HandOutputError
from landmarks.schemas import HandExtractionResult
from registry.database import RegistryError


def export_debug(result: HandExtractionResult, root: Path, *, check_permission) -> Path:
    arrays, report = result.arrays, result.report
    if arrays is None or report["source"] != "approved_dataset":
        raise HandOutputError("Debug exports require a completed extraction of an approved sample.")
    sample_id = report["sample_id"]

    def require_permission():
        try:
            check_permission()
        except (RegistryError, OSError) as exc:
            raise DatasetAccessError(str(exc)) from exc

    require_permission()
    try:
        validate_identifier(sample_id, "Sample ID")
        root = Path(root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / sample_id
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"{destination} already exists")
        with TemporaryDirectory(prefix=".hand-debug-", dir=root) as staging:
            pending = Path(staging) / sample_id
            pending.mkdir()
            np.savez_compressed(
                pending / "hand_landmarks.npz",
                image_landmarks=arrays.image_landmarks,
                world_landmarks=arrays.world_landmarks,
                presence_mask=arrays.presence_mask,
                handedness_score=arrays.handedness_score,
                timestamps_ms=arrays.timestamps_ms,
                schema_version=np.array(arrays.schema_version),
            )
            (pending / "extraction_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
            )
            require_permission()
            pending.rename(destination)
    except (OSError, ValueError) as exc:
        raise HandOutputError(f"Could not write debug outputs for {sample_id}: {exc}") from exc
    return destination
