"""Permission-checked, non-overwriting landmark debug output."""

import json
from contextlib import suppress
from os import link
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import validate_identifier
from landmarks.exceptions import DatasetAccessError
from registry.database import RegistryError


def write_landmark_debug(
    result, root: Path, *, check_permission, names: tuple[str, str], array_names, output_error,
) -> Path:
    arrays, report = result.arrays, result.report
    if arrays is None or report["source"] != "approved_dataset" or report["status"] != "PROCESSED":
        raise output_error("Debug exports require a completed extraction of an approved sample.")
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
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise FileExistsError(f"{destination} already exists and is not a regular sample directory")
        with TemporaryDirectory(prefix=".landmark-debug-", dir=root) as staging:
            pending = Path(staging)
            np.savez_compressed(
                pending / names[0], **{name: getattr(arrays, name) for name in array_names},
                schema_version=np.array(arrays.schema_version),
            )
            (pending / names[1]).write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
            )
            require_permission()
            destination.mkdir(exist_ok=True)
            if destination.is_symlink():
                raise FileExistsError(f"{destination} already exists and is a symbolic link")
            written = []
            try:
                # Hard links publish complete files without replacing another task's artifacts.
                for name in names:
                    target = destination / name
                    link(pending / name, target)
                    written.append(target)
            finally:
                if len(written) != len(names):
                    for target in written:
                        target.unlink()
                    with suppress(OSError):
                        destination.rmdir()
    except FileExistsError as exc:
        raise output_error(f"Debug output for {sample_id} already exists: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise output_error(f"Could not write debug outputs for {sample_id}: {exc}") from exc
    return destination
