"""Linguistic review workflows with temporary registries and synthetic video."""

import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from annotation import commands as annotation_commands
from annotation import interface
from annotation import service as workflow
from annotation.eligibility import TrainingEligibilityService
from annotation.interface import AnnotationWindow, PlaybackControls
from annotation.models import AnnotationInput, Decision, ReviewState
from annotation.playback import FramePlayer, Timeline, key_name, probe_timeline
from annotation.review_queue import ReviewQueue
from annotation.service import AnnotationService
from cli import app
from config import ReviewConfig
from ingestion.pipeline import IngestionService
from ingestion.schemas import IngestionOutcome
from registry import database, reviewers, samples, sessions, signers, statistics, vocabulary
from registry.database import Database, DatabaseError, RegistryError
from tests.test_ingestion import ffmpeg, media


@pytest.fixture
def setup(tmp_path, media):
    db = Database(tmp_path / "review.db")
    root = tmp_path / "dataset" / "raw"
    sign = vocabulary.add(db, gloss="WATER", target_samples=4)
    variant = vocabulary.add(db, gloss="WATER", variant_label="CENTRAL")
    signer = signers.add(db, code="SIGNER001")
    signers.consent(db, signer.code, "GRANTED")
    session = sessions.create(db, signer_code=signer.code)
    for code in ("AUTHOR", "REVIEW1", "REVIEW2", "REVIEW3"):
        reviewers.add(db, code=code, proficiency="FLUENT", role="REVIEWER")
    reviewers.add(db, code="ADJ", role="ADJUDICATOR", proficiency="QUALIFIED_INTERPRETER")
    svc = AnnotationService(db, root)

    def add(filename="landscape.mp4", sign_id=sign.id, session_id=session.id):
        outcome = IngestionService(db, root).import_file(media / filename, sign_id=sign_id, session_id=session_id)
        assert outcome.outcome in {IngestionOutcome.INGESTED, IngestionOutcome.INGESTED_WITH_WARNINGS}
        return outcome.sample_id

    return svc, add, sign, variant, signer, session


def annotate(setup, sample_id, reviewer="AUTHOR", **changes):
    svc, _, sign, _, _, _ = setup
    inputs = replace(AnnotationInput(sign.id, 100, 900), **changes)
    return svc.create(sample_id, reviewer, inputs)


def test_migration_preserves_prior_samples_without_verifying_them(tmp_path, monkeypatch):
    db = Database(tmp_path / "registry.db")
    original = tmp_path / "migrations"
    original.mkdir()
    for script in sorted(database.MIGRATIONS.glob("*.sql"))[:2]:
        shutil.copyfile(script, original / script.name)
    sid = str(uuid4())
    old_approved = str(uuid4())
    with monkeypatch.context() as patcher:
        patcher.setattr(database, "MIGRATIONS", original)
        sign = vocabulary.add(db, gloss="HELLO")
        signer = signers.add(db, code="S1")
        signers.consent(db, signer.code, "GRANTED")
        session = sessions.create(db, signer_code=signer.code)
        with db.connect(write=True) as connection:
            for sample_id, status in ((sid, "CAPTURED"), (old_approved, "APPROVED")):
                connection.execute(
                    """INSERT INTO video_samples (id, sign_id, session_id, file_path, metadata_path, width, height,
                       frame_count, status, file_size_bytes, registered_at, technical_validation_status)
                       VALUES (?, ?, ?, ?, ?, 640, 480, 30, ?, 100, '2026-11-22T09:00:00Z', 'PASSED')""",
                    (sample_id, sign.id, session.id, sample_id + ".mp4", sample_id + ".json", status),
                )
    for _ in range(2):
        with db.connect() as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        sample = samples.show(db, sid)
        assert (sample.sign_id, sample.status, sample.review_state) == (sign.id, "CAPTURED", ReviewState.PENDING)
        assert sample.current_annotation_id is None and sample.review_round == 1
    assert samples.show(db, old_approved).status == "APPROVED"
    assert not TrainingEligibilityService(db, tmp_path).check(old_approved).eligible


def test_reviewer_registration_normalization_and_local_qualification(setup):
    svc, add, _, _, _, _ = setup
    reviewer = reviewers.add(
        svc.db, code="  field-01 ", proficiency="qualified-interpreter", role="annotator",
        display_name="Local participant", qualification_notes="Credential checked offline",
    )
    assert reviewer.code == "FIELD-01" and reviewer.proficiency == "QUALIFIED_INTERPRETER"
    assert reviewers.show(svc.db, "field-01") == reviewer
    with pytest.raises(RegistryError, match="already registered"):
        reviewers.add(svc.db, code="FIELD-01", proficiency="NATIVE", role="REVIEWER")
    sample = add()
    annotate(setup, sample, reviewer="FIELD-01")
    reviewers.add(svc.db, code="TRAINEE", proficiency="OTHER", role="REVIEWER")
    for code in ("FIELD-01", "TRAINEE"):
        with pytest.raises(RegistryError, match="declared linguistic proficiency"):
            svc.decide(sample, code, Decision.APPROVE)
    reviewers.deactivate(svc.db, "REVIEW1")
    with pytest.raises(RegistryError, match="inactive"):
        svc.decide(sample, "REVIEW1", Decision.APPROVE)
    with pytest.raises(RegistryError, match="Unknown reviewer"):
        svc.decide(sample, "unknown01", Decision.APPROVE)
    assert len(reviewers.list_reviewers(svc.db)) == 7


@pytest.mark.parametrize("fields, message", [
    ({"start_ms": -1}, "integer timestamps"),
    ({"start_ms": 0.1}, "integer timestamps"),
    ({"start_ms": 900}, "integer timestamps"),
    ({"end_ms": 1251}, "integer timestamps"),
    ({"confidence": float("nan")}, "Reviewer confidence"),
    ({"confidence": 1.1}, "Reviewer confidence"),
    ({"sign_id": "UGSL9999"}, "Unknown sign"),
    ({"hand_usage": "invented"}, "Hand usage"),
])
def test_annotation_field_and_boundary_validation(setup, fields, message):
    svc, add, _, _, _, _ = setup
    sample_id = add()
    with pytest.raises(RegistryError, match=message):
        annotate(setup, sample_id, **fields)
    assert not svc.history(sample_id)
    assert samples.show(svc.db, sample_id).review_state == ReviewState.PENDING


def test_initial_and_corrected_variants_preserve_assigned_label_and_bytes(setup):
    svc, add, sign, variant, _, _ = setup
    sample_id = add()
    before = samples.show(svc.db, sample_id)
    path = svc.validator.root / before.file_path
    contents = path.read_bytes()
    variant_request = svc.request_variant(sample_id, "AUTHOR", notes="Movement differs in the central variant.")
    assert variant_request.annotation_id is None and len(vocabulary.list_signs(svc.db)) == 2
    with pytest.raises(RegistryError, match="Changing the assigned sign requires"):
        annotate(setup, sample_id, sign_id=variant.id)
    v1 = annotate(setup, sample_id, end_ms=1250, hand_usage="two-handed", movement_type="dynamic", confidence=0.8)
    assert (v1.version, v1.hand_usage, v1.movement_type, v1.end_ms) == (1, "TWO_HANDED", "DYNAMIC", 1250)
    svc.decide(sample_id, "AUTHOR", Decision.APPROVE, expected_annotation_id=v1.id)
    eligibility = TrainingEligibilityService(svc.db, svc.validator.root)
    assert eligibility.check(sample_id).eligible
    v2 = svc.correct(sample_id, "REVIEW1", sign_id=variant.id, reason="Regional variant verified", start_ms=200,
                     expected_annotation_id=v1.id, expected_round=1)
    assert (v2.version, v2.supersedes_annotation_id, v2.correction_reason, v2.confidence) == (
        2, v1.id, "Regional variant verified", 0.8,
    )
    assert svc.history(sample_id) == [v1, v2]
    assert not eligibility.check(sample_id).eligible
    mid = samples.show(svc.db, sample_id)
    assert mid.status == "CAPTURED" and mid.current_annotation_id == v2.id and mid.sign_id == sign.id
    with pytest.raises(RegistryError, match="outdated"):
        svc.decide(sample_id, "AUTHOR", Decision.APPROVE, expected_annotation_id=v1.id)
    svc.decide(sample_id, "AUTHOR", Decision.APPROVE, expected_annotation_id=v2.id)
    assert eligibility.check(sample_id).verified_sign_id == variant.id
    after = samples.show(svc.db, sample_id)
    assert (after.sign_id, after.file_path, path.read_bytes()) == (before.sign_id, before.file_path, contents)
    with svc.db.connect(write=True) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE annotations SET start_ms = 42 WHERE id = ?", (v1.id,))


def test_single_review_states_duplicate_submission_and_reopening(setup):
    svc, add, _, _, _, _ = setup
    sample_id = add()
    annotate(setup, sample_id)
    svc.start_review(sample_id, "AUTHOR")
    assert samples.show(svc.db, sample_id).review_state == ReviewState.UNDER_REVIEW
    approval = svc.decide(sample_id, "AUTHOR", Decision.APPROVE)
    assert approval.state == ReviewState.APPROVED and approval.decision.is_final
    with pytest.raises(RegistryError, match="already submitted"):
        svc.decide(sample_id, "AUTHOR", Decision.APPROVE)
    event = svc.reopen(sample_id, "REVIEW1", reason="Inspect the end boundary")
    assert event.review_round == 1
    mid = samples.show(svc.db, sample_id)
    assert (mid.review_state, mid.status, mid.review_round) == (ReviewState.ANNOTATED, "CAPTURED", 2)
    assert not TrainingEligibilityService(svc.db, svc.validator.root).check(sample_id).eligible
    with pytest.raises(RegistryError, match="outdated"):
        svc.decide(sample_id, "REVIEW1", Decision.APPROVE, expected_round=1)
    svc.decide(sample_id, "AUTHOR", Decision.APPROVE, expected_round=2)
    votes, events = svc.review_history(sample_id)
    assert [vote.review_round for vote in votes] == [1, 2]
    assert [event.event for event in events] == ["REVIEW_STARTED", "REOPENED"]


def test_rejection_and_correction_require_standardized_reasons(setup):
    svc, add, _, _, _, _ = setup
    sample = add()
    annotate(setup, sample)
    with pytest.raises(RegistryError, match="requires a reason"):
        svc.decide(sample, "REVIEW1", Decision.REQUEST_CORRECTION)
    with pytest.raises(RegistryError, match="Reason code"):
        svc.decide(sample, "REVIEW1", Decision.REJECT, reason_code="free text")
    decision = svc.decide(sample, "REVIEW1", Decision.REQUEST_CORRECTION, reason_code="invalid-temporal-boundaries")
    assert decision.state == ReviewState.CORRECTION_REQUIRED
    with pytest.raises(RegistryError, match="unavailable"):
        svc.decide(sample, "REVIEW2", Decision.APPROVE)
    svc.correct(sample, "AUTHOR", reason="Corrected timing", start_ms=220)
    outcome = svc.decide(sample, "REVIEW1", "reject", reason_code="incomplete-sign", notes="Ends early")
    assert outcome.state == ReviewState.REJECTED and outcome.decision.reason_code == "INCOMPLETE_SIGN"
    assert not TrainingEligibilityService(svc.db, svc.validator.root).check(sample).eligible


def test_dual_policy_requires_independent_qualification(setup):
    svc, add, _, _, _, _ = setup
    sample_id = add()
    annotate(setup, sample_id)
    dual = AnnotationService(svc.db, svc.validator.root, ReviewConfig(policy="dual"))
    with pytest.raises(RegistryError, match="other than the original annotator"):
        dual.decide(sample_id, "AUTHOR", Decision.APPROVE)
    result = dual.decide(sample_id, "REVIEW1", Decision.APPROVE)
    assert result.decision.review_policy == "DUAL" and result.state == ReviewState.APPROVED
    assert TrainingEligibilityService(svc.db, svc.validator.root, dual.config).check(sample_id).eligible
    reviewers.deactivate(svc.db, "REVIEW1")
    assert TrainingEligibilityService(svc.db, svc.validator.root, dual.config).check(sample_id).eligible


def test_single_approval_does_not_satisfy_a_later_dual_policy(setup):
    svc, add, _, _, _, _ = setup
    sample = add()
    annotate(setup, sample)
    svc.decide(sample, "AUTHOR", Decision.APPROVE)
    eligibility = TrainingEligibilityService(svc.db, svc.validator.root, ReviewConfig("DUAL")).check(sample)
    assert not eligibility.eligible and "independent approval" in eligibility.reason


def test_explicit_second_review_requires_another_reviewer(setup):
    svc, add, _, _, _, _ = setup
    sample = add()
    annotate(setup, sample)
    result = svc.decide(sample, "AUTHOR", Decision.REQUEST_SECOND_REVIEW, notes="Variant uncertain")
    assert result.state == ReviewState.SECOND_REVIEW_REQUIRED
    with pytest.raises(RegistryError, match="already submitted"):
        svc.decide(sample, "AUTHOR", Decision.APPROVE)
    result = svc.decide(sample, "REVIEW1", Decision.APPROVE)
    assert result.state == ReviewState.APPROVED


def test_conflicting_decisions_require_and_preserve_adjudication(setup):
    svc, add, _, _, _, _ = setup
    sample = add()
    annotation = annotate(setup, sample)
    dual = AnnotationService(svc.db, svc.validator.root, ReviewConfig("DUAL"))
    verdict = dual.decide(sample, "REVIEW1", Decision.REJECT, reason_code="incorrect-label")
    assert verdict.state == ReviewState.SECOND_REVIEW_REQUIRED and not verdict.decision.is_final
    verdict = dual.decide(sample, "REVIEW2", Decision.APPROVE)
    assert verdict.needs_adjudication and not verdict.decision.is_final
    queue, = ReviewQueue(svc.db, svc.validator.root).list(state="second-review-required")
    assert queue.needs_adjudication
    with pytest.raises(RegistryError, match="independent adjudicator"):
        dual.decide(sample, "REVIEW3", Decision.APPROVE)
    with pytest.raises(RegistryError, match="independent adjudicator"):
        dual.correct(sample, "AUTHOR", reason="Do not bypass review", start_ms=120)
    verdict = dual.decide(sample, "ADJ", Decision.APPROVE, expected_annotation_id=annotation.id, notes="Variant confirmed")
    assert verdict.state == ReviewState.APPROVED and verdict.decision.is_adjudication
    votes, _ = dual.review_history(sample)
    assert [vote.decision for vote in votes] == ["REJECT", "APPROVE", "APPROVE"]
    assert [vote.is_final for vote in votes] == [False, False, True]
    assert TrainingEligibilityService(svc.db, svc.validator.root, dual.config).check(sample).eligible
    with svc.db.connect(write=True) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE review_decisions SET notes = 'Rewrite' WHERE id = ?", (votes[0].id,))


@pytest.mark.parametrize("technical", ["PENDING", "FAILED", "ERROR"])
def test_technical_failures_block_annotations_and_approvals(setup, technical):
    svc, add, _, _, _, _ = setup
    sample = add()
    annotation = annotate(setup, sample)
    with svc.db.connect(write=True) as connection:
        connection.execute("UPDATE video_samples SET technical_validation_status = ? WHERE id = ?", (technical, sample))
    with pytest.raises(RegistryError, match="technical validation"):
        svc.decide(sample, "REVIEW1", Decision.APPROVE)
    with pytest.raises(RegistryError, match="technical validation"):
        svc.correct(sample, "REVIEW1", reason="Correction", start_ms=120)
    assert svc.history(sample) == [annotation]
    assert not ReviewQueue(svc.db, svc.validator.root).list()


def test_consent_and_video_access_are_rechecked_for_eligibility(setup):
    svc, add, _, _, signer, _ = setup
    sid = add()
    annotate(setup, sid)
    eligibility = TrainingEligibilityService(svc.db, svc.validator.root)
    sample = samples.show(svc.db, sid)
    path = svc.validator.root / sample.file_path
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    with pytest.raises(RegistryError, match="checksum"):
        svc.decide(sid, "AUTHOR", Decision.APPROVE)
    path.write_bytes(original)
    svc.decide(sid, "AUTHOR", Decision.APPROVE)
    assert eligibility.check(sid).eligible
    renamed = path.with_suffix(".holding")
    path.rename(renamed)
    assert not eligibility.check(sid).eligible
    assert not ReviewQueue(svc.db, svc.validator.root).list(state="approved")
    renamed.rename(path)
    signers.consent(svc.db, signer.code, "WITHDRAWN")
    assert not eligibility.check(sid).eligible
    assert samples.show(svc.db, sid).status == "EXCLUDED"
    with pytest.raises(RegistryError, match="excluded"):
        svc.view(sid, "AUTHOR")


def test_queue_filters_orders_and_prioritizes(setup):
    svc, add, sign, variant, signer, session = setup
    pending = add()
    second = add("dark.mp4")
    correction = add("portrait.mov", sign_id=variant.id)
    annotate(setup, second)
    svc.decide(second, "AUTHOR", Decision.REQUEST_SECOND_REVIEW)
    svc.create(correction, "AUTHOR", AnnotationInput(variant.id, 100, 900))
    svc.decide(correction, "REVIEW1", Decision.REQUEST_CORRECTION, reason_code="incorrect-label")
    with svc.db.connect(write=True) as connection:
        for sid, date in ((pending, "2026-01-01"), (second, "2026-01-02"), (correction, "2026-01-03")):
            connection.execute("UPDATE video_samples SET registered_at = ? WHERE id = ?", (date + "T10:00:00Z", sid))
    queue = ReviewQueue(svc.db, svc.validator.root)
    assert [item.sample_id for item in queue.list()] == [pending, second, correction]
    assert [item.sample_id for item in queue.list(order="newest")] == [correction, second, pending]
    assert [item.sample_id for item in queue.list(prioritize=True)] == [second, correction, pending]
    matched = queue.list(
        sign_id=sign.id, signer=signer.id, session_id=session.id, reviewer="AUTHOR", source="import",
        since="2026-01-02", until="2026-01-02", technical_status="passed-with-warnings",
    )
    assert [item.sample_id for item in matched] == [second]
    assert [item.sample_id for item in queue.list(state="pending", signer=signer.code)] == [pending]
    assert not queue.list(source="REMOTE")


def test_statistics_distinguish_verified_labels_rejections_and_agreement(setup):
    svc, add, sign, variant, signer, _ = setup
    first = add()
    annotate(setup, first)
    svc.correct(first, "AUTHOR", reason="Correct variant", sign_id=variant.id)
    svc.decide(first, "AUTHOR", Decision.APPROVE)
    second = add("dark.mp4")
    annotate(setup, second)
    dual = AnnotationService(svc.db, svc.validator.root, ReviewConfig("DUAL"))
    dual.decide(second, "REVIEW1", Decision.REJECT, reason_code="hands-not-visible")
    dual.decide(second, "REVIEW2", Decision.REJECT, reason_code="hands-not-visible")
    add("portrait.mov")
    stats = statistics.annotation_statistics(svc.db, svc.validator.root)
    assert stats.counts["awaiting_annotation"] == 1 and stats.counts["annotated_samples"] == 2
    assert stats.counts["approved_samples"] == stats.counts["rejected_samples"] == stats.counts["training_eligible_samples"] == 1
    assert stats.counts["label_corrections"] == 1
    assert stats.counts["review_completion_percent"] == 100 and stats.counts["approval_rate_percent"] == 50
    assert stats.rejection_reasons == [{"reason": "HANDS_NOT_VISIBLE", "samples": 1}]
    assert stats.agreement == {"compared_rounds": 1, "agreed_rounds": 1, "agreement_percent": 100.0}
    approved = {row["sign_id"]: row for row in stats.approved_by_sign}
    assert approved[sign.id]["approved_by_original_label"] == 1 and approved[sign.id]["verified_approved"] == 0
    assert approved[variant.id]["training_eligible"] == 1 and approved[variant.id]["verified_approved"] == 1
    assert stats.approved_by_signer == [{"signer": signer.code, "approved_samples": 1, "training_eligible": 1}]


def test_state_change_failure_rolls_back_annotation_and_review(setup, monkeypatch):
    svc, add, _, _, _, _ = setup
    sid = add()

    def fail(*args, **kwargs):
        raise DatabaseError("Test transition failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(workflow, "_update_state", fail)
        with pytest.raises(DatabaseError):
            annotate(setup, sid)
    assert not svc.history(sid)
    annotate(setup, sid)
    with monkeypatch.context() as patcher:
        patcher.setattr(workflow, "_update_state", fail)
        with pytest.raises(DatabaseError):
            svc.decide(sid, "AUTHOR", Decision.APPROVE)
    assert svc.review_history(sid) == ([], [])
    assert samples.show(svc.db, sid).review_state == ReviewState.ANNOTATED


def test_vfr_timeline_and_frame_stepping_use_presentation_timestamps(tmp_path, media):
    path = tmp_path / "vfr.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-vf", "setpts=(N+floor(N/2))/(25*TB)",
        "-frames:v", "12", "-fps_mode", "vfr", "-c:v", "mpeg4", str(path),
    )
    timeline = probe_timeline(path)
    differences = {round(end - start) for start, end in zip(timeline.timestamps_ms, timeline.timestamps_ms[1:])}
    assert len(timeline.timestamps_ms) == 12 and len(differences) > 1
    assert timeline.start_marker(2) != round(2 * 1000 / 25)
    assert timeline.end_marker(11) == timeline.duration_ms
    with FramePlayer(path, timeline) as player:
        first = player.at(0).copy()
        assert player.at(11).shape == first.shape
        assert np.array_equal(player.at(0), first)
    controls = PlaybackControls(timeline)
    controls.toggle(10)
    controls.advance(10.041)
    assert controls.index == 1
    controls.navigate("RIGHT")
    assert controls.index == 2 and not controls.playing
    controls.navigate("LEFT")
    assert controls.index == 1
    start, end = timeline.start_marker(1), timeline.start_marker(3)
    controls.replay(start, end, 20)
    controls.change_speed(1, 20)
    controls.advance(30)
    controls.advance(30)
    assert controls.index == 2 and not controls.playing
    assert key_name(65361) == "LEFT" and key_name(63235) == "RIGHT" and key_name(ord("Q")) == "q"


def test_opencv_window_controls_save_approve_and_protect_drafts(setup, monkeypatch):
    svc, add, _, _, _, _ = setup
    sid = add()
    original = np.full((120, 160, 3), 85, dtype=np.uint8)
    captures, previews = [], []

    class Capture:
        def __init__(self, *_):
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            return True, original

        def release(self):
            self.released = True

    monkeypatch.setattr(interface, "probe_timeline", lambda *args, **kwargs: Timeline((0, 500, 900), 1250))
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "imshow", lambda _, frame: previews.append(frame))
    keys = iter([ord("i"), 65363, ord("o"), ord("s"), -1, ord("a"), ord("n")])
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _: next(keys))
    monkeypatch.setattr(interface.typer, "prompt", lambda _, default=None: default)
    assert interface.open_video(svc, sid, "AUTHOR") == "next"
    saved, = svc.history(sid)
    assert (saved.start_ms, saved.end_ms) == (0, 500)
    assert samples.show(svc.db, sid).status == "APPROVED"
    assert captures[0].released and previews and np.all(original == 85)
    assert not np.array_equal(previews[0], original)

    window = AnnotationWindow(svc, svc.view(sid, "AUTHOR"), "AUTHOR")
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _: -1)
    window.controls.index = 2
    window.press("o", original)
    with pytest.raises(RegistryError, match="Save the annotation"):
        window.press("a", original)
    monkeypatch.setattr(interface.typer, "confirm", lambda *_args, **_kwargs: False)
    assert window.press("q", original) is None and window.dirty
    monkeypatch.setattr(interface.typer, "confirm", lambda *_args, **_kwargs: True)
    assert window.press("q", original) == "quit"
    assert svc.history(sid) == [saved]


def test_cli_registration_annotation_review_and_stats(setup, monkeypatch):
    svc, add, sign, variant, _, _ = setup
    sid = add()
    cli = CliRunner()
    base = ["--db", str(svc.db.path), "--dataset-root", str(svc.validator.root)]
    monkeypatch.delenv("UGSL_REVIEW_POLICY", raising=False)

    def invoke(*args):
        result = cli.invoke(app, base + list(args))
        assert result.exit_code == 0, (result.output, result.exception)
        return result.output

    assert "OP1" in invoke("reviewer", "add", "--code", "OP1", "--proficiency", "fluent", "--role", "reviewer")
    assert sid in invoke("annotation", "queue", "--sign", sign.id, "--status", "pending")
    assert "Version: 1" in invoke("annotation", "create", sid, "--reviewer", "OP1", "--sign", sign.id,
                                  "--start-ms", "100", "--end-ms", "900", "--hand-usage", "two-handed")
    assert "APPROVED" in invoke("review", "approve", sid, "--reviewer", "OP1")
    assert "Eligible: True" in invoke("sample", "eligibility", sid)
    assert "ANNOTATED" in invoke("review", "reopen", sid, "--reviewer", "OP1", "--reason", "Revisit classification")
    assert "Version: 2" in invoke("annotation", "correct", sid, "--reviewer", "OP1", "--sign", variant.id,
                                  "--reason", "Verified variant")
    assert "Version: 1" in invoke("annotation", "history", sid)
    assert "REOPENED" in invoke("review", "history", sid)
    assert "training eligible samples" in invoke("stats", "--annotations")
    assert "Agreement between independent reviewers" in invoke("stats", "--reviews")
    assert "Approved coverage by signer" in invoke("stats", "--approved-coverage")


def test_annotation_open_can_continue_to_next_sample(setup, monkeypatch):
    svc, add, _, _, _, _ = setup
    first = add()
    second = add("portrait.mov")
    opened = []

    def open_video(_, sample_id, reviewer):
        assert reviewer == "AUTHOR"
        opened.append(sample_id)
        return "next" if sample_id == first else "quit"

    monkeypatch.setattr(annotation_commands, "open_video", open_video)
    result = CliRunner().invoke(app, [
        "--db", str(svc.db.path), "--dataset-root", str(svc.validator.root),
        "annotation", "open", first, "--reviewer", "AUTHOR",
    ])
    assert result.exit_code == 0 and opened == [first, second]


def test_interrupted_window_keeps_saved_version_and_warns_about_draft(setup, monkeypatch, capsys):
    svc, add, _, _, _, _ = setup
    sample_id = add()
    saved = annotate(setup, sample_id)
    window = AnnotationWindow(svc, svc.view(sample_id, "AUTHOR"), "AUTHOR")
    window.start_ms = 50
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: None)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cv2, "waitKeyEx", interrupt)
    assert window.run() == "quit"
    assert "Unsaved boundary changes will be discarded" in capsys.readouterr().err
    assert svc.history(sample_id) == [saved]
