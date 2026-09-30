"""
Settles training jobs whose worker never reported back.

A job normally finishes with its worker's final webhook. That update never arrives if
Modal kills the container at its timeout, the container crashes, or the request is lost,
and the job would then stay RUNNING for good: its credits reserved, the user never charged,
and Modal's bill unpaid. The reconciler asks Modal about each unfinished job every few
minutes and settles the ones whose call has ended.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text

from db import SessionLocal
from models.database import JobStatus, TrainingJob
from services.custom_models import remove_replaced_model_files
from services.job_orchestrator import _fail_job_submission, finish_job
from services.modal_worker import TRAINING_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_SECONDS = 600
STARTUP_DELAY_SECONDS = 60

# A QUEUED job with no Modal call this long after it was created was never spawned: its
# submission died before recording one
UNSPAWNED_AFTER = timedelta(minutes=15)

TIMEOUT_MINUTES = TRAINING_TIMEOUT_SECONDS / 60

ACTIVE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


@dataclass
class CallOutcome:
    """How a finished training call ended."""
    status: str  # "completed" or "failed"
    error_message: Optional[str] = None
    # Container time Modal billed, when the call reported it (or it hit the timeout)
    actual_minutes: Optional[float] = None
    # True when Modal no longer has the call's record, so its duration is unknown
    expired: bool = False


async def _call_outcome(modal_call_id: str) -> Optional[CallOutcome]:
    """
    How a training call ended, or None while it's still queued or running.

    Raises if Modal can't be asked (the job is then left for the next round): a failure to
    reach Modal must never be mistaken for the job failing.
    """
    import modal
    from modal.call_graph import InputStatus

    call = modal.FunctionCall.from_id(modal_call_id)
    inputs = [i for i in await call.get_call_graph.aio() if i.function_call_id == modal_call_id]
    if not inputs:
        return CallOutcome("failed", "Lost track of the training job", expired=True)

    status = inputs[0].status
    if status == InputStatus.PENDING:
        return None
    if status == InputStatus.TIMEOUT:
        return CallOutcome(
            "failed",
            f"Training exceeded the {TIMEOUT_MINUTES:.0f}-minute time limit",
            actual_minutes=TIMEOUT_MINUTES,
        )
    if status != InputStatus.SUCCESS:
        return CallOutcome("failed", "Training stopped unexpectedly")

    # The worker returns how the job went ({"status", "actual_minutes", "error_message"})
    try:
        result = await call.get.aio(timeout=0)
    except modal.exception.OutputExpiredError:
        return CallOutcome("failed", "Lost track of the training job", expired=True)

    if result.get("status") == "completed":
        return CallOutcome("completed", result.get("error_message"), result.get("actual_minutes"))
    return CallOutcome(
        "failed",
        result.get("error_message") or "Training stopped",
        result.get("actual_minutes"),
    )


def _billable_minutes(job: TrainingJob, outcome: CallOutcome) -> float:
    """
    Minutes to charge a job for.

    What the worker reported, when it did. Otherwise, the time since the job was spawned,
    which is at least what Modal billed (checks run every few minutes, so it's close), up
    to the timeout. If Modal no longer has the call's record, the backend was down for
    days and that time says nothing about the job, so the job's estimate is charged instead.
    """
    if outcome.actual_minutes is not None:
        return outcome.actual_minutes
    if outcome.expired:
        return min(job.estimated_minutes, TIMEOUT_MINUTES)
    started_at = job.started_at or job.created_at
    return min((datetime.utcnow() - started_at).total_seconds() / 60, TIMEOUT_MINUTES)


async def _settle(job_id: str, outcome: CallOutcome) -> None:
    """Finish a job whose call has ended, unless its final update got there first."""
    db = SessionLocal()
    try:
        db.execute(text("SET LOCAL lock_timeout = '5s'"))
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).with_for_update().first()
        if job is None or job.status not in ACTIVE_STATUSES:
            db.rollback()
            return

        user_id, model_name = job.user_id, job.model_name
        if outcome.error_message:
            job.error_message = outcome.error_message
        replaced_models = await finish_job(
            db, job, outcome.status, _billable_minutes(job, outcome), outcome.error_message
        )
        db.commit()
    finally:
        db.close()

    logger.warning("Reconciled training job %s: %s (%s)", job_id, outcome.status, outcome.error_message)
    await remove_replaced_model_files(user_id, replaced_models, model_name)


def _fail_unspawned(job_id: str) -> None:
    """Fail a job whose submission never spawned it, refunding its reservation in full."""
    db = SessionLocal()
    try:
        _fail_job_submission(db, job_id, "Job submission didn't complete")
    finally:
        db.close()


async def reconcile_training_jobs() -> None:
    """Settle every unfinished training job whose Modal call has ended."""
    db = SessionLocal()
    try:
        jobs = [
            (job.job_id, job.status, job.modal_call_id, job.created_at)
            for job in db.query(TrainingJob).filter(TrainingJob.status.in_(ACTIVE_STATUSES))
        ]
    finally:
        db.close()

    for job_id, status, modal_call_id, created_at in jobs:
        try:
            if modal_call_id is None:
                if status == JobStatus.QUEUED and datetime.utcnow() - created_at > UNSPAWNED_AFTER:
                    logger.warning("Failing training job %s: it was never spawned", job_id)
                    _fail_unspawned(job_id)
                continue

            outcome = await _call_outcome(modal_call_id)
            if outcome is not None:
                await _settle(job_id, outcome)
        except Exception:
            logger.exception("Failed to reconcile training job %s", job_id)


async def run_training_job_reconciler() -> None:
    """Reconcile training jobs shortly after startup, then periodically, until cancelled."""
    await asyncio.sleep(STARTUP_DELAY_SECONDS)
    while True:
        try:
            await reconcile_training_jobs()
        except Exception:
            logger.exception("Training job reconciliation failed")
        await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)
