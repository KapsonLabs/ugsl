"""Staging journals and exclusive publication for media files."""

import errno
import fcntl
import json
import os
import shutil
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from ingestion.exceptions import MediaError
from ingestion.hashing import sha256_file

SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def canonical_uuid(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Expected a canonical UUID.")
    return value


def safe_path(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MediaError("UNSAFE_PATH", "The dataset path is unsafe.")
    path = root / relative
    if path.resolve() != path:
        raise MediaError("UNSAFE_PATH", "Symlinks are unsupported inside dataset storage.")
    return path


def _json(path: Path, payload: dict) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".pending-", dir=path.parent) as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def _link(source: Path, target: Path) -> None:
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError:
        if target.is_symlink() or not source.samefile(target):
            raise MediaError("DESTINATION_EXISTS", "A different file already exists at the destination.", operational=True)


class Stage:
    def __init__(self, directory: Path):
        self.directory = directory
        self.video = directory / "video"
        self.request = directory / "request.json"
        self.sidecar = directory / "sidecar.json"
        self.checksum = directory / "checksum.json"

    @contextmanager
    def lock(self, *, create=False):
        path = self.directory / ".lock"
        flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY)
        fd = os.open(path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def prepare(self, source: Path, request: dict, *, camera: bool, max_bytes: int) -> None:
        if camera:
            try:
                os.link(source, self.video, follow_symlinks=False)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
        if not self.video.exists():
            source_stat = source.stat()
            with source.open("rb") as original, self.video.open("xb") as staged:
                size = 0
                for chunk in iter(lambda: original.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > max_bytes:
                        raise MediaError("FILE_TOO_LARGE", "The video exceeds the configured file size limit.")
                    staged.write(chunk)
                staged.flush()
                os.fsync(staged.fileno())
            os.utime(self.video, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        _json(self.request, request)

    def read_request(self) -> dict:
        if self.request.is_symlink() or self.video.is_symlink() or not self.video.is_file():
            raise MediaError("UNSAFE_STAGE", "The staging journal or video is missing or unsafe.")
        return json.loads(self.request.read_text(encoding="utf-8"))

    def verify_checksum(self, digest: str, *, dry_run: bool) -> None:
        if self.checksum.is_symlink():
            raise MediaError("UNSAFE_STAGE", "The staging checksum is unsafe.", operational=True)
        if self.checksum.is_file():
            if json.loads(self.checksum.read_text(encoding="utf-8")).get("sha256") != digest:
                raise MediaError("CHECKSUM_CHANGED", "Staged bytes no longer match the ingestion checksum.", operational=True)
        elif not dry_run:
            _json(self.checksum, {"sha256": digest})

    def remove(self) -> None:
        shutil.rmtree(self.directory)


class DatasetStorage:
    def __init__(self, raw: Path):
        self.raw = Path(raw).resolve()
        self.staging = safe_path(self.raw.parent, "staging")
        self.quarantine = safe_path(self.raw.parent, "quarantine")
        if self.raw in {self.staging, self.quarantine}:
            raise MediaError("UNSAFE_PATH", "Raw, staging and quarantine must be distinct directories.")

    @contextmanager
    def stage(self, attempt_id):
        path = safe_path(self.staging, canonical_uuid(attempt_id))
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        stage = Stage(path)
        with stage.lock(create=True):
            yield stage

    def publish(self, stage: Stage, relative: Path, sidecar: dict, checksum: str) -> Path:
        video = safe_path(self.raw, relative)
        metadata = safe_path(self.raw, relative.with_suffix(".json"))
        if sha256_file(stage.video) != checksum:
            raise MediaError("CHECKSUM_CHANGED", "The staged video changed during ingestion.", operational=True)
        if stage.sidecar.is_symlink():
            raise MediaError("UNSAFE_STAGE", "The staging sidecar is unsafe.", operational=True)
        if stage.sidecar.exists():
            previous = json.loads(stage.sidecar.read_text(encoding="utf-8"))
            if (previous.get("sha256"), previous.get("file_path")) != (checksum, relative.as_posix()):
                raise MediaError("JOURNAL_MISMATCH", "Staged publication metadata does not match the video.", operational=True)
        else:
            _json(stage.sidecar, sidecar)
        video.parent.mkdir(parents=True, exist_ok=True)
        _link(stage.sidecar, metadata)
        _link(stage.video, video)
        return video

    def remove_owned_links(self, stage: Stage) -> None:
        if not stage.sidecar.is_file() or stage.sidecar.is_symlink():
            return
        info = json.loads(stage.sidecar.read_text(encoding="utf-8"))
        video = safe_path(self.raw, info["file_path"])
        for source, target in ((stage.video, video), (stage.sidecar, video.with_suffix(".json"))):
            with suppress(FileNotFoundError):
                if not target.is_symlink() and source.samefile(target):
                    target.unlink()

    def retain(self, stage: Stage, attempt_id: str, report: dict, retention_days: int) -> None:
        target = safe_path(self.quarantine, canonical_uuid(attempt_id))
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        _link(stage.video, target / "video")
        report = {**report, "expires_at": (datetime.now(UTC) + timedelta(days=retention_days)).isoformat()}
        marker = safe_path(target, "report.json")
        if not marker.exists():
            _json(marker, report)
