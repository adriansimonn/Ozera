"""
Circuit Tracing jobs (an attribution graph of one prompt): submitting one to the circuits
worker, and settling it once its call ends.

A job is charged its estimate when submitted. When its call ends, it's marked completed (its
graph is then on the SAE volume, and its summary on its row), or failed and refunded all but the
GPU time it used (Modal bills that either way). The job's endpoint settles it when it's polled,
and the reconciler settles any nobody polls.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from core.open_source import get_gpu_tier
from db import SessionLocal
from models.database import CircuitJob, TransactionType
from services.credit_service import calculate_circuit_cost, charge_flat, credit_refund
from services.modal_circuits import CIRCUIT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
# A queued job with no Modal call this long after it was created was never spawned
UNSPAWNED_AFTER = timedelta(minutes=15)
# Characters of a prompt a transaction's description shows
DESCRIPTION_CHARS = 60


def new_job_id() -> str:
    return f"circuit_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _brief(prompt: str) -> str:
    text = " ".join(prompt.split())
    return text if len(text) <= DESCRIPTION_CHARS else text[: DESCRIPTION_CHARS - 1] + "…"


def create_job(
    db: Session,
    user_id: int,
    model_id: str,
    transcoders: str,
    prompt: str,
    target: Optional[str],
    cost: float,
    estimated_seconds: float,
) -> CircuitJob:
    """
    Record a job and charge for it, in one transaction.

    Raises:
        InsufficientBalanceError: the balance doesn't cover it (nothing is recorded)
    """
    job = CircuitJob(
        job_id=new_job_id(),
        user_id=user_id,
        model_id=model_id,
        transcoders=transcoders,
        prompt=prompt,
        target=target or None,
        status="queued",
        cost_usd=cost,
        estimated_seconds=estimated_seconds,
        created_at=datetime.utcnow(),
    )
    db.add(job)
    charge_flat(db, user_id, cost, TransactionType.CIRCUIT_CHARGE, f"Circuit graph ({model_id}): {_brief(prompt)}")
    db.refresh(job)
    return job


def _worker(model_id: str):
    from services.modal_circuits import get_circuit_worker

    return get_circuit_worker(get_gpu_tier(model_id))


async def spawn_job(db: Session, job: CircuitJob) -> None:
    """
    Start a recorded job on the circuits worker. If that fails, the job fails with a full
    refund and the error is raised.
    """
    try:
        call = await _worker(job.model_id)().trace.spawn.aio(job.job_id, job.model_id, job.transcoders, job.prompt, job.target)
    except Exception as e:
        finish(db, job.job_id, Outcome("failed", "The job couldn't be started", seconds=0.0))
        raise RuntimeError(f"Couldn't start the circuit job: {e}")
    db.query(CircuitJob).filter(CircuitJob.job_id == job.job_id, CircuitJob.status == "queued").update(
        {CircuitJob.modal_call_id: call.object_id, CircuitJob.status: "running", CircuitJob.started_at: datetime.utcnow()},
        synchronize_session=False,
    )
    db.commit()


@dataclass
class Outcome:
    """How a job's call ended."""

    status: str  # "completed" or "failed"
    error: Optional[str] = None
    # Seconds the job kept its worker busy, when known (the job reports them; a timeout runs the whole limit)
    seconds: Optional[float] = None
    # A completed job's graph in brief (core.circuits.graph.summary)
    summary: Optional[dict] = None


async def call_outcome(modal_call_id: str) -> Optional[Outcome]:
    """
    How a job's call ended, or None while it's still queued or running.

    Raises if Modal can't be asked (the job is then left for later): a failure to reach Modal
    must never be mistaken for the job failing.
    """
    import modal
    from modal.call_graph import InputStatus

    call = modal.FunctionCall.from_id(modal_call_id)
    inputs = [i for i in await call.get_call_graph.aio() if i.function_call_id == modal_call_id]
    if not inputs:
        return Outcome("failed", "Lost track of the job")
    status = inputs[0].status
    if status == InputStatus.PENDING:
        return None
    if status == InputStatus.TIMEOUT:
        return Outcome("failed", "The job ran out of time", seconds=float(CIRCUIT_TIMEOUT_SECONDS))
    if status != InputStatus.SUCCESS:
        return Outcome("failed", "The job stopped unexpectedly")
    try:
        result = await call.get.aio(timeout=0)
    except modal.exception.OutputExpiredError:
        return Outcome("failed", "Lost track of the job")
    return Outcome(result.get("status", "failed"), result.get("error"), result.get("seconds"), result.get("summary"))


def finish(db: Session, job_id: str, outcome: Outcome) -> bool:
    """
    Settle a job whose call ended, once: completed, or failed with a refund of all but the GPU
    time it ran (from its report, else the time since it started). Returns whether this call settled it.
    """
    job = db.query(CircuitJob).filter(CircuitJob.job_id == job_id).with_for_update().populate_existing().first()
    if job is None or job.status not in ACTIVE:
        db.rollback()
        return False
    job.finished_at = datetime.utcnow()
    if outcome.status == "completed" and outcome.summary is not None:
        job.status = "completed"
        job.summary = outcome.summary
        db.commit()
        return True

    seconds = outcome.seconds
    if seconds is None:
        started = job.started_at or job.created_at
        seconds = min((job.finished_at - started).total_seconds(), CIRCUIT_TIMEOUT_SECONDS)
    used = calculate_circuit_cost(job.model_id, seconds) if seconds > 0 else 0.0
    # To the ledger's four decimals before it's compared with the smallest refund made (a cent)
    refund = round(max(0.0, float(job.cost_usd) - used), 4)
    job.status = "failed"
    job.error_message = (outcome.error or "The job failed")[:1000]
    refunded = credit_refund(
        db, job.user_id, refund, TransactionType.CIRCUIT_REFUND,
        f"Circuit graph refund ({job.model_id}): {job.error_message[:200]}",
    )
    # What was given back, which is nothing when the unused part is under a cent
    job.refunded_usd = refund if refunded is not None else 0
    db.commit()
    return True


async def refresh(db: Session, job: CircuitJob) -> CircuitJob:
    """Settle a job if its call has ended (asking Modal); returns it as it now is."""
    if job.status not in ACTIVE:
        return job
    try:
        if job.modal_call_id is None:
            if datetime.utcnow() - job.created_at > UNSPAWNED_AFTER:
                finish(db, job.job_id, Outcome("failed", "The job was never started", seconds=0.0))
        else:
            outcome = await call_outcome(job.modal_call_id)
            if outcome is not None:
                finish(db, job.job_id, outcome)
    except Exception:
        logger.exception("Failed to check circuit job %s", job.job_id)
        db.rollback()
    db.refresh(job)
    return job


async def progress(job_id: str) -> Optional[dict]:
    """A running job's progress, {"stage", "fraction"}, as its worker last reported it."""
    try:
        import modal

        from services.modal_circuits import PROGRESS_DICT_NAME

        return await modal.Dict.from_name(PROGRESS_DICT_NAME).get.aio(job_id)
    except Exception:
        return None


async def reconcile_circuit_jobs() -> None:
    """Settle every unfinished circuit job whose call has ended (for jobs nobody polls)."""
    db = SessionLocal()
    try:
        jobs = db.query(CircuitJob).filter(CircuitJob.status.in_(ACTIVE)).all()
        for job in jobs:
            before = job.status
            job = await refresh(db, job)
            if job.status != before:
                logger.warning("Reconciled circuit job %s: %s (%s)", job.job_id, job.status, job.error_message)
    finally:
        db.close()
