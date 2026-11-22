"""Export snapshots, integrity, lifecycle and recovery with synthetic clips."""

import csv
import json
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from annotation.models import AnnotationInput, Decision
from cli import app
from dataset_export import comparison, manifest, repository
from dataset_export.integrity import DatasetIntegrityVerifier
from dataset_export.models import ExportConfig
from dataset_export.reader import DatasetReader
from dataset_export.service import DatasetExportService
from dataset_export.storage import sha256_file
from dataset_export.versioning import next_version, version_key
from registry import datasets, samples, signers, vocabulary
from registry.database import Database, DatabaseError, RegistryError
from tests.test_annotation import annotate, setup
from tests.test_ingestion import media


@pytest.fixture
def system(setup):
    workflow, add, sign, variant, signer, session = setup
    service = DatasetExportService(workflow.db, workflow.validator.root)

    def approved(filename="landscape.mp4", *, chosen_sign=sign, **traits):
        sid = add(filename, sign_id=chosen_sign.id)
        workflow.create(sid, "AUTHOR", replace(
            AnnotationInput(chosen_sign.id, 100, 900),
            **traits,
        ))
        workflow.decide(sid, "AUTHOR", Decision.APPROVE)
        return sid

    return service, approved


def target(result):
    return Path(result.output_path)


def version(service, result):
    return datasets.show(service.db, result.name, result.version)


def verify(service, result, **kwargs):
    return DatasetIntegrityVerifier().verify(
        target(result), storage_root=service.raw, registry_path=service.db.path, **kwargs,
    )


def test_export_schema_migration_preserves_existing_review_metadata(system, setup, tmp_path):
    service, approved = system
    workflow, *_ = setup
    sid = approved()
    # Construct a v3 fixture from the existing DB after removing the Phase 1.5 objects.
    old = tmp_path / "old.db"
    shutil.copyfile(service.db.path, old)
    with sqlite3.connect(old) as db:
        triggers = db.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'dataset_%'").fetchall()
        for (name,) in triggers:
            db.execute(f"DROP TRIGGER {name}")
        for name in ("dataset_events", "dataset_members", "dataset_versions"):
            db.execute(f"DROP TABLE {name}")
        db.execute("PRAGMA user_version = 3")
    migrated = Database(old)
    assert samples.show(migrated, sid).current_annotation_id == workflow.history(sid)[0].id
    with migrated.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT count(*) FROM dataset_versions").fetchone()[0] == 0


def test_reference_version_membership_manifest_and_csv(system, setup):
    service, approved = system
    _, _, sign, _, signer, _ = setup
    sid = approved(hand_usage="two-handed", completeness="complete")
    result = service.export(ExportConfig("ugsl-basic", "1.0.0", csv=True))
    assert result.status == "ACTIVE" and result.total_samples == 1
    row = version(service, result)
    assert row.schema_version == "1.0" and row.finalized_at and row.total_signers == 1
    with service.db.connect() as connection:
        member, = repository.members(connection, row.id)
    assert (member.sample_id, member.sign_id, member.signer_id) == (sid, sign.id, signer.id)
    assert member.manifest_entry_checksum == manifest.sha256_text((target(result) / "manifest.jsonl").read_text().strip())
    assert [event["status"] for event in datasets.history(service.db, row.id)] == ["BUILDING", "ACTIVE"]
    folder = target(result)
    assert not (folder / "videos").exists() and not (folder / ".building").exists()
    data = json.loads((folder / "dataset.json").read_text())
    assert data["manifest_checksum"] == sha256_file(folder / "manifest.jsonl")
    assert data["total_samples"] == data["total_signs"] == data["total_signers"] == 1
    assert str(service.raw) not in (folder / "dataset.json").read_text()
    sample, = DatasetReader(folder, storage_root=service.raw, registry_path=service.db.path)
    assert sample.sample_id == sid and sample.annotation.id == member.annotation_id
    assert sample.annotation.hand_usage == "TWO_HANDED" and sample.annotation.start_ms == 100
    assert sample.video_path.is_relative_to(service.raw)
    with (folder / "manifest.csv").open() as file:
        flat, = csv.DictReader(file)
    assert flat["annotation_id"] == member.annotation_id and flat["sample_id"] == sid
    assert verify(service, result).integrity_ok
    with pytest.raises(RegistryError, match="already registered"):
        service.export(ExportConfig("UGSL-BASIC", "1.0.0"))


def test_semantic_versioning_and_deterministic_export(system):
    service, approved = system
    approved()
    first = service.export(ExportConfig("ugsl-example"))
    second = service.export(ExportConfig("ugsl-example"))
    third = service.export(ExportConfig("ugsl-example", bump="minor"))
    fourth = service.export(ExportConfig("ugsl-example", bump="major"))
    assert [result.version for result in (first, second, third, fourth)] == ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
    assert len({result.dataset_id for result in (first, second, third, fourth)}) == 4
    assert len({result.content_fingerprint for result in (first, second, third, fourth)}) == 1
    assert (target(first) / "manifest.jsonl").read_bytes() == (target(fourth) / "manifest.jsonl").read_bytes()
    assert (target(first) / "vocabulary.json").read_bytes() == (target(fourth) / "vocabulary.json").read_bytes()
    assert next_version(["1.0.0-rc.2", "1.0.0-rc.10", "1.0.0+local.1"], "patch") == "1.0.1"
    assert version_key("1.0.0-rc.2") < version_key("1.0.0-rc.10") < version_key("1.0.0")
    with pytest.raises(RegistryError, match="semantic version"):
        service.export(ExportConfig("invalid", "01.2.3"))


def test_dry_run_and_coverage_report_leave_storage_untouched(system, setup):
    service, approved = system
    _, _, sign, variant, _, _ = setup
    approved()
    db_bytes = service.db.path.read_bytes()
    output = service.raw.parent / "exports"
    request = ExportConfig("coverage", "1.0.0", sign_ids=(sign.id, variant.id), min_signers_per_sign=2, dry_run=True)
    result = service.export(request)
    assert result.status == "BLOCKED"
    assert {row["sign_id"]: row["unique_signers"] for row in result.threshold_failures} == {sign.id: 1, variant.id: 0}
    allowed = service.export(replace(request, allow_incomplete_coverage=True))
    assert allowed.status == "DRY_RUN" and allowed.total_samples == 1
    assert len(allowed.threshold_failures) == 2 and allowed.manifest_checksum
    assert service.db.path.read_bytes() == db_bytes and not output.exists()
    final = service.export(replace(request, dry_run=False, allow_incomplete_coverage=True))
    assert final.status == "ACTIVE" and final.content_fingerprint == allowed.content_fingerprint
    stats = comparison.load_statistics(service.db, version(service, final))
    assert len(stats["coverage"]["threshold_failures"]) == 2


def test_filters_warnings_and_deterministic_cap(system, setup):
    service, approved = system
    _, _, sign, variant, signer, session = setup
    vocabulary.update(service.db, sign.id, category="Kitchen")
    first = approved()
    second = approved("audio.mp4")
    warned = approved("dark.mp4")
    corresponding = approved("portrait.mov", chosen_sign=variant)
    preview = service.export(ExportConfig(
        "filtered", categories=("KITCHEN",), signer_ids=(signer.code,), session_ids=(session.id,),
        max_samples_per_sign=1, include_warnings=False,
    ))
    assert preview.total_samples == 1
    picked, = DatasetReader(target(preview), storage_root=service.raw)
    assert picked.sample_id == min(first, second)
    reasons = {row["sample_id"]: row["reason"] for row in preview.excluded}
    assert "cap" in reasons[max(first, second)]
    assert "warnings" in reasons[warned]
    assert "verified signs" in reasons[corresponding]


def test_exports_enforce_approval_consent_and_checksums(system, setup):
    service, approved = system
    workflow, add, _, _, signer, _ = setup
    accepted = approved()
    pending = add("audio.mp4")
    annotated = add("portrait.mov")
    annotate(setup, annotated)
    preview = service.export(ExportConfig("eligible", dry_run=True))
    assert preview.total_samples == 1
    excluded = {row["sample_id"]: row["reason"] for row in preview.excluded}
    assert "linguistic approval" in excluded[pending] and "linguistic approval" in excluded[annotated]
    with service.db.connect(write=True) as connection:
        connection.execute("UPDATE video_samples SET sha256 = NULL WHERE id = ?", (accepted,))
    missing = service.export(ExportConfig("eligible", dry_run=True))
    assert "registered SHA-256" in next(row["reason"] for row in missing.excluded if row["sample_id"] == accepted)
    signers.consent(workflow.db, signer.code, "WITHDRAWN")
    withdrawn = service.export(ExportConfig("eligible", dry_run=True))
    assert withdrawn.status == "BLOCKED" and withdrawn.total_samples == 0


def test_portable_round_trip_preserves_bytes_extensions_and_works_offline(system, setup, tmp_path, monkeypatch):
    service, approved = system
    _, _, sign, variant, _, _ = setup
    first = approved()
    second = approved("portrait.mov", chosen_sign=variant)
    result = service.export(ExportConfig("portable", "1.0.0", mode="PORTABLE"))
    source_dir = target(result)
    moved = tmp_path / "moved-dataset"
    shutil.copytree(source_dir, moved)
    monkeypatch.delenv("UGSL_REGISTRY_DB", raising=False)
    report = DatasetIntegrityVerifier().verify(moved)
    assert report.integrity_ok and report.permission.status == "UNKNOWN" and report.permission.permitted is None
    reader = DatasetReader(moved)
    assert reader.schema_version == "1.0"
    assert [sample.sample_id for sample in reader.iter_samples(sign.id)] == [first]
    by_id = {row.sample_id: row for row in reader}
    assert by_id[second].video_path.suffix == ".mov"
    for sid in [first, second]:
        source = service.raw / samples.show(service.db, sid).file_path
        original_export = source_dir / by_id[sid].video_path.relative_to(moved)
        assert source.stat().st_ino != original_export.stat().st_ino
        assert sha256_file(source) == sha256_file(by_id[sid].video_path) == by_id[sid].sha256
    assert [entry["class_index"] for entry in reader.vocabulary["classes"]] == [0, 1]
    assert "checksums.sha256  " not in (moved / "checksums.sha256").read_text()
    process = subprocess.run(
        [sys.executable, "-c", "from dataset_export.reader import DatasetReader; import sys; "
                              "assert 'cv2' not in sys.modules"], capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr


def test_reference_resolution_supports_a_configured_parent(system):
    service, approved = system
    sid = approved()
    root = service.raw.parent
    result = service.export(ExportConfig("parent-root", storage_root=root))
    with pytest.raises(RegistryError, match="require a configured storage root"):
        DatasetReader(target(result))
    record, = DatasetReader(target(result), storage_root=root)
    assert record.sample_id == sid and record.video_path == service.raw / samples.show(service.db, sid).file_path
    assert not DatasetIntegrityVerifier().verify(target(result), storage_root=service.raw).integrity_ok
    assert DatasetIntegrityVerifier().verify(target(result), storage_root=root).integrity_ok


def test_verify_detects_tampering_missing_files_and_membership_mismatch(system):
    service, approved = system
    approved()
    portable = service.export(ExportConfig("integrity", mode="portable"))
    root = target(portable)
    copied, = list((root / "videos").rglob("*.mp4"))
    original = copied.read_bytes()
    copied.write_bytes(b"x" * len(original))
    invalid = verify(service, portable)
    assert not invalid.integrity_ok and invalid.permission.permitted
    assert any("Checksum mismatch" in err for err in invalid.errors)
    copied.write_bytes(original)
    with service.db.connect(write=True) as connection:
        connection.execute("UPDATE dataset_members SET manifest_entry_checksum = ? WHERE dataset_version_id = ?",
                           ("b" * 64, portable.dataset_id))
    assert any("registry membership" in err for err in verify(service, portable).errors)
    copied.unlink()
    assert not verify(service, portable).integrity_ok


def test_reference_verifies_source_bytes_and_portable_stays_offline_readable(system):
    service, approved = system
    sid = approved()
    reference = service.export(ExportConfig("refs", mode="REFERENCE"))
    portable = service.export(ExportConfig("copies", mode="PORTABLE"))
    path = service.raw / samples.show(service.db, sid).file_path
    path.write_bytes(b"x" * path.stat().st_size)
    report = verify(service, reference)
    assert not report.integrity_ok and any("Source checksum mismatch" in error for error in report.errors)
    assert verify(service, portable).integrity_ok


def test_version_comparison_and_annotation_invalidation(system, setup):
    service, approved = system
    workflow, _, sign, variant, _, _ = setup
    first = approved()
    old_ann = workflow.history(first)[0]
    before = service.export(ExportConfig("compare", "1.0.0"))
    before_manifest = (target(before) / "manifest.jsonl").read_bytes()
    corrected = workflow.correct(first, "REVIEW1", reason="Verify variant", sign_id=variant.id)
    assert version(service, before).status == "REVOKED"
    affected, = datasets.affected(service.db, annotation_id=old_ann.id)
    assert affected.id == before.dataset_id
    workflow.decide(first, "REVIEW1", Decision.APPROVE)
    added = approved("portrait.mov", chosen_sign=variant)
    after = service.export(ExportConfig("compare", "1.1.0"))
    changes = comparison.compare(service.db, version(service, before), version(service, after))
    assert changes["added_samples"] == [added] and not changes["removed_samples"]
    assert changes["changed_annotations"][0]["to_annotation"] == corrected.id
    assert changes["added_signs"] == [variant.id] and changes["removed_signs"] == [sign.id]
    assert changes["duration_change_ms"] > 0 and changes["storage_change_bytes"] > 0
    assert not changes["equivalent_content"] and changes["class_mapping_changes"]
    assert (target(before) / "manifest.jsonl").read_bytes() == before_manifest
    stats = comparison.load_statistics(service.db, version(service, after))
    assert stats["annotations"]["corrected_classifications"] == 1
    assert stats["annotations"]["distinct_version_numbers"] == 2


def test_lifecycle_revoke_and_withdrawal_impact(system, setup):
    service, approved = system
    _, _, _, _, signer, _ = setup
    sid = approved()
    one = service.export(ExportConfig("life", "1.0.0", mode="portable"))
    two = service.export(ExportConfig("life", "1.1.0"))
    superseded = datasets.transition(service.db, one.name, one.version, "SUPERSEDED")
    assert superseded.status == "SUPERSEDED" and verify(service, one).permission.permitted
    reader = DatasetReader(target(one), registry_path=service.db.path)
    assert next(iter(reader)).sample_id == sid
    with pytest.raises(RegistryError, match="requires a reason"):
        datasets.transition(service.db, one.name, one.version, "REVOKED", "")
    signers.consent(service.db, signer.code, "WITHDRAWN")
    affected = datasets.affected(service.db, signer=signer.code, sample_id=sid)
    assert [entry.status for entry in affected] == ["REVOKED", "REVOKED"]
    assert all(entry.revocation_reason == "Signer consent is no longer granted" for entry in affected)
    result = verify(service, one)
    assert result.integrity_ok and result.permission.status == "REVOKED" and not result.permission.permitted
    with pytest.raises(RegistryError, match="blocked"):
        list(reader)
    dry = service.reconcile(dry_run=True)
    assert dry and not (target(one) / "revocation.json").exists()
    service.reconcile()
    assert (target(two) / "revocation.json").is_file()
    with pytest.raises(RegistryError, match="REVOKED"):
        DatasetReader(target(one))
    assert [event["status"] for event in datasets.history(service.db, one.dataset_id)] == [
        "BUILDING", "ACTIVE", "SUPERSEDED", "REVOKED",
    ]


def test_invalidation_during_build_blocks_publication(system, setup, monkeypatch):
    service, approved = system
    _, _, _, _, signer, _ = setup
    approved()
    original = service._verify_build

    def revoke_after_validation(*args):
        original(*args)
        signers.consent(service.db, signer.code, "WITHDRAWN")

    monkeypatch.setattr(service, "_verify_build", revoke_after_validation)
    with pytest.raises(RegistryError, match="no longer in BUILDING"):
        service.export(ExportConfig("consent-race", mode="PORTABLE"))
    row, = datasets.list_versions(service.db)
    assert row.status == "REVOKED" and not Path(row.output_path).exists()


def test_reconciliation_recovers_rename_then_database_failure(system, monkeypatch):
    service, approved = system
    approved()
    original = repository.activate

    def fail_after_rename(*_):
        raise DatabaseError("simulated database transition failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(repository, "activate", fail_after_rename)
        with pytest.raises(RegistryError, match="reconcile"):
            service.export(ExportConfig("recover", mode="PORTABLE"))
    row, = datasets.list_versions(service.db)
    root = Path(row.output_path)
    assert row.status == "BUILDING" and (root / ".building").is_file()
    with pytest.raises(RegistryError, match="BUILDING"):
        DatasetReader(root)
    before = service.db.path.read_bytes()
    assert service.reconcile(dry_run=True)[0]["action"] == "Would finalize"
    assert service.db.path.read_bytes() == before and (root / ".building").exists()
    assert repository.activate is original
    assert service.reconcile()[0]["action"] == "Finalized"
    assert datasets.show(service.db, row.name, row.version).status == "ACTIVE"
    assert not (root / ".building").exists() and list(DatasetReader(root))
    assert not service.reconcile()


def test_reconciliation_flags_incomplete_staging(system, monkeypatch):
    service, approved = system
    approved()

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(service, "_write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        service.export(ExportConfig("interrupted"))
    row, = datasets.list_versions(service.db)
    assert row.status == "BUILDING"
    assert service.reconcile(dry_run=True)[0]["error"]
    assert datasets.show(service.db, row.name, row.version).status == "BUILDING"
    assert service.reconcile()[0]["error"]
    assert datasets.show(service.db, row.name, row.version).status == "FAILED"


def test_path_safety_rejects_traversal_symlinks_and_existing_destinations(system):
    service, approved = system
    approved()
    for name in ("../elsewhere", "path/name", "/tmp"):
        with pytest.raises(RegistryError, match="Dataset name"):
            service.export(ExportConfig(name))
    with pytest.raises(RegistryError, match="outside raw"):
        service.export(ExportConfig("unsafe", output_directory=service.raw))
    parent = service.raw.parent / "exports"
    parent.mkdir()
    destination = parent / "collision-v1.0.0"
    destination.symlink_to(service.raw, target_is_directory=True)
    with pytest.raises(RegistryError, match="symlinks"):
        service.export(ExportConfig("collision"))
    assert destination.is_symlink()
    result = service.export(ExportConfig("safe", mode="PORTABLE"))
    root = target(result)
    checksums = root / "checksums.sha256"
    with checksums.open("a") as stream:
        stream.write("0" * 64 + "  ../outside\n")
    assert not verify(service, result).integrity_ok


def test_cli_export_verify_compare_and_revoke(system, setup):
    service, approved = system
    _, _, _, _, signer, _ = setup
    approved()
    runner = CliRunner()
    flags = ["--db", str(service.db.path), "--dataset-root", str(service.raw)]

    def invoke(*args, code=0):
        result = runner.invoke(app, flags + ["dataset", *map(str, args)])
        assert result.exit_code == code, (result.output, result.exception)
        return result.output

    preview = json.loads(invoke("export", "--name", "cli-data", "--dry-run", "--json"))
    assert preview["status"] == "DRY_RUN"
    first = json.loads(invoke("export", "--name", "cli-data", "--mode", "portable", "--json"))
    assert first["status"] == "ACTIVE"
    assert "cli-data" in invoke("list")
    assert "Lifecycle history" in invoke("show", "cli-data", "--version", "1.0.0")
    assert "Content integrity passed" in invoke("verify", "--path", first["output_path"])
    assert "total_samples" in invoke("stats", "cli-data", "--version", "1.0.0", "--json")
    invoke("export", "--name", "cli-data", "--bump", "minor")
    verify_diff = json.loads(invoke("compare", "cli-data", "--from-version", "1.0.0", "--to-version", "1.1.0", "--json"))
    assert verify_diff["equivalent_content"]
    assert "cli-data" in invoke("affected", "--signer", signer.code)
    assert "REVOKED" in invoke("revoke", "cli-data", "--version", "1.0.0", "--reason", "Consent review")
    output = invoke("verify", "cli-data", "--version", "1.0.0", code=1)
    assert "Content integrity passed" in output and "Use blocked" in output


def test_portable_capacity_failure_does_not_publish(system, monkeypatch):
    service, approved = system
    approved()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RegistryError, match="Insufficient free space"):
        service.export(ExportConfig("space", mode="portable"))
    record, = datasets.list_versions(service.db)
    assert record.status == "FAILED" and not Path(record.output_path).exists()


def test_skipping_candidate_preflight_does_not_bypass_source_integrity(system):
    service, approved = system
    sid = approved()
    path = service.raw / samples.show(service.db, sid).file_path
    path.write_bytes(b"z" * path.stat().st_size)
    with pytest.raises(RegistryError, match="Source checksum mismatch"):
        service.export(ExportConfig("source-changed", verify_all_source_checksums=False))
    record, = datasets.list_versions(service.db)
    assert record.status == "FAILED" and not Path(record.output_path).exists()


def test_cli_withdraw_reports_affected_and_writes_local_notice(system, setup, monkeypatch):
    service, approved = system
    _, _, _, _, signer, _ = setup
    approved()
    result = service.export(ExportConfig("local-notice", mode="portable"))
    outcome = CliRunner().invoke(app, [
        "--db", str(service.db.path), "--dataset-root", str(service.raw),
        "signer", "withdraw", signer.code,
    ])
    assert outcome.exit_code == 0 and "Revoked dataset versions" in outcome.output
    assert (target(result) / "revocation.json").is_file()
    monkeypatch.delenv("UGSL_REGISTRY_DB", raising=False)
    with pytest.raises(RegistryError, match="REVOKED"):
        DatasetReader(target(result))
