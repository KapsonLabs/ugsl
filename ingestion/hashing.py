"""Content hashing without reading the entire video into memory."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream) -> str:
    return hashlib.file_digest(stream, "sha256").hexdigest()
