"""Typer entry points for annotation and reviewer accountability."""

from dataclasses import asdict
from typing import Annotated

import typer

from annotation.eligibility import TrainingEligibilityService
from annotation.interface import open_video
from annotation.models import AnnotationInput, Decision
from annotation.review_queue import ReviewQueue
from annotation.service import AnnotationService
from config import ReviewConfig
from ingestion.exceptions import MediaError
from registry import reviewers
from registry.commands import _details, _report_revoked_datasets, _table, database, report_errors, sample_app
from registry.database import RegistryError

reviewer_app = typer.Typer(help="Manage local linguistic reviewers.", no_args_is_help=True)
annotation_app = typer.Typer(help="Annotate isolated signs and inspect annotation history.", no_args_is_help=True)
review_app = typer.Typer(help="Verify and review saved annotations.", no_args_is_help=True)
REVIEWER = typer.Option(help="Active reviewer code.")
ANNOTATION_ID = typer.Option(help="Expected current annotation UUID.")
ROUND = typer.Option("--round", help="Expected current review round.")


def configured_review() -> ReviewConfig:
    try:
        return ReviewConfig.from_environment()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _service(ctx):
    return AnnotationService(database(ctx), ctx.obj.dataset_root, configured_review())


@reviewer_app.command("add")
@report_errors
def reviewer_add(
    ctx: typer.Context, code: Annotated[str, typer.Option()], proficiency: Annotated[str, typer.Option()],
    role: Annotated[str, typer.Option()], display_name: Annotated[str | None, typer.Option()] = None,
    qualification_notes: Annotated[str | None, typer.Option()] = None,
):
    _details(reviewers.add(database(ctx), code=code, proficiency=proficiency, role=role,
                           display_name=display_name, qualification_notes=qualification_notes))


@reviewer_app.command("list")
@report_errors
def reviewer_list(ctx: typer.Context):
    _table([asdict(item) for item in reviewers.list_reviewers(database(ctx))],
           ("code", "display_name", "proficiency", "role", "is_active"))


@reviewer_app.command("show")
@report_errors
def reviewer_show(ctx: typer.Context, code: str):
    _details(reviewers.show(database(ctx), code))


@reviewer_app.command("deactivate")
@report_errors
def reviewer_deactivate(ctx: typer.Context, code: str):
    _details(reviewers.deactivate(database(ctx), code))


@annotation_app.command("queue")
@report_errors
def annotation_queue(
    ctx: typer.Context, sign: Annotated[str | None, typer.Option()] = None,
    signer: Annotated[str | None, typer.Option(help="Signer code or UUID.")] = None,
    session: Annotated[str | None, typer.Option()] = None,
    technical_status: Annotated[str | None, typer.Option("--technical-status", "--validation-status")] = None,
    status: Annotated[str | None, typer.Option(help="Review workflow state.")] = None,
    reviewer: Annotated[str | None, typer.Option(help="Current annotation author or review participant.")] = None,
    source: Annotated[str | None, typer.Option(help="CAMERA, IMPORT, LOCAL or REMOTE.")] = None,
    since: Annotated[str | None, typer.Option(help="Earliest UTC registration date (YYYY-MM-DD).")] = None,
    until: Annotated[str | None, typer.Option(help="Latest UTC registration date (YYYY-MM-DD).")] = None,
    order: Annotated[str, typer.Option(help="oldest, newest or sign.")] = "oldest",
    priority: Annotated[bool, typer.Option(help="Move second review and corrections to the front.")] = False,
):
    rows = ReviewQueue(database(ctx), ctx.obj.dataset_root).list(
        sign_id=sign, signer=signer, session_id=session, technical_status=technical_status, state=status,
        reviewer=reviewer, source=source, since=since, until=until, order=order, prioritize=priority,
    )
    _table([asdict(row) for row in rows], ("sample_id", "sign_id", "gloss", "signer_code", "source", "review_state",
                                         "technical_validation_status", "reviewer_code", "needs_adjudication"))


@annotation_app.command("open")
@report_errors
def annotation_open(ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER]):
    service = _service(ctx)
    if open_video(service, sample_id, reviewer) == "next":
        queue = ReviewQueue(service.db, ctx.obj.dataset_root).list()
        _open_queue(service, reviewer, (row for row in queue if row.sample_id != sample_id.strip().lower()))


def _open_queue(service, reviewer, queue):
    for row in queue:
        try:
            if open_video(service, row.sample_id, reviewer) == "quit":
                return
        except (RegistryError, MediaError) as exc:
            typer.echo(f"{row.sample_id}: {exc}", err=True)
    typer.echo("End of annotation queue.")


@annotation_app.command("run")
@report_errors
def annotation_run(
    ctx: typer.Context, reviewer: Annotated[str, REVIEWER],
    sign: Annotated[str | None, typer.Option()] = None, status: Annotated[str | None, typer.Option()] = None,
    order: Annotated[str, typer.Option()] = "oldest",
    priority: Annotated[bool, typer.Option()] = False,
):
    service = _service(ctx)
    with service.db.connect() as connection:
        reviewers.require_active(connection, reviewer)
    queue = ReviewQueue(service.db, ctx.obj.dataset_root).list(sign_id=sign, state=status, order=order, prioritize=priority)
    if not queue:
        typer.echo("No eligible samples in the queue.")
        return
    _open_queue(service, reviewer, queue)


@annotation_app.command("create")
@report_errors
def annotation_create(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER], sign: Annotated[str, typer.Option()],
    start_ms: Annotated[int, typer.Option()], end_ms: Annotated[int, typer.Option()],
    hand_usage: Annotated[str, typer.Option()] = "unknown",
    dominant_hand: Annotated[str, typer.Option()] = "unknown",
    movement_type: Annotated[str, typer.Option()] = "unknown",
    facial_expression: Annotated[str, typer.Option()] = "unknown",
    completeness: Annotated[str, typer.Option()] = "unknown",
    confidence: Annotated[float | None, typer.Option()] = None, notes: Annotated[str | None, typer.Option()] = None,
    reason: Annotated[str | None, typer.Option(help="Required if changing the originally assigned sign.")] = None,
):
    inputs = AnnotationInput(sign, start_ms, end_ms, hand_usage, dominant_hand, movement_type,
                             facial_expression, completeness, confidence, notes)
    _details(_service(ctx).create(sample_id, reviewer, inputs, reason=reason))


@annotation_app.command("correct")
@report_errors
def annotation_correct(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER], reason: Annotated[str, typer.Option()],
    sign: Annotated[str | None, typer.Option()] = None,
    start_ms: Annotated[int | None, typer.Option()] = None, end_ms: Annotated[int | None, typer.Option()] = None,
    hand_usage: Annotated[str | None, typer.Option()] = None, dominant_hand: Annotated[str | None, typer.Option()] = None,
    movement_type: Annotated[str | None, typer.Option()] = None,
    facial_expression: Annotated[str | None, typer.Option()] = None, completeness: Annotated[str | None, typer.Option()] = None,
    confidence: Annotated[float | None, typer.Option()] = None,
    clear_confidence: Annotated[bool, typer.Option(help="Remove the confidence score from the new version.")] = False,
    notes: Annotated[str | None, typer.Option()] = None,
    annotation_id: Annotated[str | None, ANNOTATION_ID] = None, review_round: Annotated[int | None, ROUND] = None,
):
    changes = {
        name: value for name, value in dict(
            sign_id=sign, start_ms=start_ms, end_ms=end_ms, hand_usage=hand_usage, dominant_hand=dominant_hand,
            movement_type=movement_type, facial_expression=facial_expression, completeness=completeness,
            confidence=confidence, notes=notes,
        ).items() if value is not None
    }
    if clear_confidence:
        if confidence is not None:
            raise typer.BadParameter("Choose either --confidence or --clear-confidence.")
        changes["confidence"] = None
    _details(_service(ctx).correct(sample_id, reviewer, reason=reason, expected_annotation_id=annotation_id,
                                   expected_round=review_round, **changes))
    _report_revoked_datasets(database(ctx), sample_id=sample_id.strip().lower())


@annotation_app.command("request-variant")
@report_errors
def annotation_variant(ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER],
                       notes: Annotated[str, typer.Option()]):
    _details(_service(ctx).request_variant(sample_id, reviewer, notes=notes))


@annotation_app.command("history")
@report_errors
def annotation_history(ctx: typer.Context, sample_id: str):
    history = _service(ctx).history(sample_id)
    if not history:
        typer.echo("No saved annotations.")
    for annotation in history:
        _details(annotation)
        typer.echo()


@review_app.command("start")
@report_errors
def review_start(ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER]):
    _service(ctx).start_review(sample_id, reviewer)
    typer.echo("Review state UNDER_REVIEW")


def _decision(ctx, sample_id, reviewer, action, reason, notes, annotation_id, review_round):
    result = _service(ctx).decide(sample_id, reviewer, action, reason_code=reason, notes=notes,
                                  expected_annotation_id=annotation_id, expected_round=review_round)
    _details(result.decision)
    typer.echo(f"Review state {result.state}")
    if result.needs_adjudication:
        typer.echo("Conflicting decisions require an independent adjudicator.")


@review_app.command("approve")
@report_errors
def review_approve(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER],
    notes: Annotated[str | None, typer.Option()] = None, annotation_id: Annotated[str | None, ANNOTATION_ID] = None,
    review_round: Annotated[int | None, ROUND] = None,
):
    _decision(ctx, sample_id, reviewer, Decision.APPROVE, None, notes, annotation_id, review_round)


@review_app.command("reject")
@report_errors
def review_reject(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER], reason: Annotated[str, typer.Option()],
    notes: Annotated[str | None, typer.Option()] = None, annotation_id: Annotated[str | None, ANNOTATION_ID] = None,
    review_round: Annotated[int | None, ROUND] = None,
):
    _decision(ctx, sample_id, reviewer, Decision.REJECT, reason, notes, annotation_id, review_round)


@review_app.command("request-correction")
@report_errors
def review_correction(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER], reason: Annotated[str, typer.Option()],
    notes: Annotated[str | None, typer.Option()] = None, annotation_id: Annotated[str | None, ANNOTATION_ID] = None,
    review_round: Annotated[int | None, ROUND] = None,
):
    _decision(ctx, sample_id, reviewer, Decision.REQUEST_CORRECTION, reason, notes, annotation_id, review_round)


@review_app.command("request-second")
@report_errors
def review_second(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER],
    notes: Annotated[str | None, typer.Option()] = None, annotation_id: Annotated[str | None, ANNOTATION_ID] = None,
    review_round: Annotated[int | None, ROUND] = None,
):
    _decision(ctx, sample_id, reviewer, Decision.REQUEST_SECOND_REVIEW, None, notes, annotation_id, review_round)


@review_app.command("reopen")
@report_errors
def review_reopen(
    ctx: typer.Context, sample_id: str, reviewer: Annotated[str, REVIEWER], reason: Annotated[str, typer.Option()],
    annotation_id: Annotated[str | None, ANNOTATION_ID] = None, review_round: Annotated[int | None, ROUND] = None,
):
    _details(_service(ctx).reopen(sample_id, reviewer, reason=reason, expected_annotation_id=annotation_id,
                                  expected_round=review_round))
    typer.echo("Review state ANNOTATED")
    _report_revoked_datasets(database(ctx), sample_id=sample_id.strip().lower())


@review_app.command("history")
@report_errors
def review_history(ctx: typer.Context, sample_id: str):
    decisions, events = _service(ctx).review_history(sample_id)
    typer.echo("Review decisions")
    _table([asdict(decision) for decision in decisions],
           ("annotation_id", "review_round", "reviewer_id", "review_policy", "decision", "reason_code", "notes",
            "is_final", "is_adjudication", "reviewed_at"))
    typer.echo("\nReview events")
    _table([asdict(event) for event in events],
           ("annotation_id", "review_round", "reviewer_id", "event", "notes", "created_at"))


@sample_app.command("eligibility")
@report_errors
def sample_eligibility(ctx: typer.Context, sample_id: str):
    service = TrainingEligibilityService(database(ctx), ctx.obj.dataset_root, configured_review())
    _details(service.check(sample_id))


def mount(app: typer.Typer) -> None:
    app.add_typer(reviewer_app, name="reviewer")
    app.add_typer(annotation_app, name="annotation")
    app.add_typer(review_app, name="review")
