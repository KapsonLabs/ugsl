"""Safe relative paths, build locks and byte-preserving export storage."""

import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from ingestion.hashing import sha256_file as digest_file
from registry.database import RegistryError

BUILD_MARKER = ".building"
REVOCATION_FILE = "revocation.json"


class ExportStorageUnavailable(RegistryError):
    """A build directory or lock is temporarily unavailable."""


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or any(ord(char) < 32 or char in "\\:" for char in value):
        raise RegistryError("Invalid relative dataset path.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or value != path.as_posix():
        raise RegistryError("Invalid relative dataset path.")
    return path


def safe_join(root: Path, relative: str) -> Path:
    path = root.joinpath(*relative_path(relative).parts)
    try:
        if path.resolve() != path:
            raise RegistryError("Dataset paths cannot use symlinks or escape their root.")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RegistryError("Cannot resolve the dataset path safely.") from exc
    return path


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RegistryError(f"Missing or unreadable dataset file: {path.name}")
    return digest_file(path)


def copy_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, destination.open("xb") as copied:
        shutil.copyfileobj(original, copied, length=1024 * 1024)


def check_space(root: Path, video_bytes: int) -> None:
    if shutil.disk_usage(root).free < video_bytes:
        raise RegistryError("Insufficient free space for the portable video copies.")


@contextmanager
def export_lock(root: Path):
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / ".ugsl-export.lock", os.O_NOFOLLOW | os.O_CREAT | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise ExportStorageUnavailable("Cannot access or lock the export directory.") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExportStorageUnavailable("Another export or reconciliation is running in this output directory.") from exc
        except OSError as exc:
            raise ExportStorageUnavailable("Cannot lock the export directory.") from exc
        yield
    finally:
        os.close(fd)


def publish(stage: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RegistryError("The export destination already exists.")
    stage.rename(destination)
