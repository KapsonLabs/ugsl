"""Directory and CSV batch inputs for the shared ingestion service."""

import csv
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ingestion.exceptions import MediaError
from ingestion.pipeline import IngestionRequest, IngestionService
from ingestion.schemas import IngestionOutcome as Outcome
from ingestion.schemas import IngestionResult
from ingestion.storage import SUFFIXES


@dataclass(frozen=True)
class ImportEntry:
    file_path: Path
    sign_id: str
    session_id: str
    recorded_at: str | None = None
    notes: str | None = None
    source_reference: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    discovered: int
    duration_seconds: float
    results: list[IngestionResult]

    def summary(self) -> dict:
        counts = Counter(result.outcome for result in self.results)
        return {
            "batch_id": self.batch_id,
            "discovered": self.discovered,
            "processed": len(self.results),
            "ingested": counts[Outcome.INGESTED] + counts[Outcome.INGESTED_WITH_WARNINGS],
            "ingested_with_warnings": counts[Outcome.INGESTED_WITH_WARNINGS],
            "duplicates": counts[Outcome.DUPLICATE],
            "metadata_conflicts": counts[Outcome.CONFLICT],
            "rejected": counts[Outcome.REJECTED],
            "operational_failures": counts[Outcome.ERROR],
            "duration_seconds": self.duration_seconds,
            "storage_bytes": sum(result.storage_bytes for result in self.results if not result.dry_run),
            "projected_storage_bytes": sum(result.storage_bytes for result in self.results if result.dry_run),
        }


def media_files(directory: Path, *, recursive=True):
    """Discover regular, recognized files without traversing symlink directories."""
    for current, directories, filenames in os.walk(directory, followlinks=False):
        directories[:] = sorted(
            item for item in directories if recursive and not (Path(current) / item).is_symlink()
        )
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.suffix.lower() in SUFFIXES and not path.is_symlink() and path.is_file():
                yield path


def directory_entries(directory: Path, sign_id: str, session_id: str, *, recursive=False) -> list[ImportEntry]:
    if directory.is_symlink() or not directory.is_dir():
        raise MediaError("INVALID_DIRECTORY", "Select a local directory that is not a symlink.")
    return [ImportEntry(path, sign_id, session_id) for path in media_files(directory, recursive=recursive)]


def manifest_entries(manifest: Path) -> list[ImportEntry]:
    entries = []
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            required = {"file_path", "sign_id", "session_id"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise MediaError("INVALID_MANIFEST", "CSV requires unique headers including file_path, sign_id and session_id.")
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error:
                    entries.append(ImportEntry(
                        manifest, "", "", error=f"Manifest near line {reader.line_num} has invalid CSV quoting.",
                    ))
                    continue
                source, sign, session = ((row.get(field) or "").strip() for field in ("file_path", "sign_id", "session_id"))
                error = None
                if not source or not sign or not session or None in row:
                    error = f"Manifest line {reader.line_num} has missing required values or extra columns."
                entries.append(ImportEntry(
                    manifest.resolve().parent / source, sign, session, (row.get("recorded_at") or "").strip() or None,
                    (row.get("notes") or "").strip() or None, (row.get("source_reference") or "").strip() or None, error,
                ))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MediaError("INVALID_MANIFEST", "Could not read the CSV manifest. Check its encoding and CSV quoting.") from exc
    return entries


def import_batch(service: IngestionService, entries: list[ImportEntry], *, dry_run=False, fail_fast=False,
                 quarantine=False) -> BatchReport:
    started = time.monotonic()
    batch_id = str(uuid4())
    results = []
    simulated = {}
    for entry in entries:
        request = IngestionRequest(
            entry.sign_id, entry.session_id, source_reference=entry.source_reference or entry.file_path.name,
            recorded_at=entry.recorded_at, notes=entry.notes, batch_id=batch_id, quarantine=quarantine,
        )
        result = service.ingest(entry.file_path, request, dry_run=dry_run, input_error=entry.error,
                                simulated=simulated if dry_run else None)
        results.append(result)
        if fail_fast and result.outcome in {Outcome.CONFLICT, Outcome.REJECTED, Outcome.ERROR}:
            break
    return BatchReport(batch_id, len(entries), time.monotonic() - started, results)
