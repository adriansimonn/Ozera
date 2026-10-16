"""
Atlas jobs (the Geometry Lab's feature statistics over a corpus): submitting one to Modal, and
settling it once its call ends.

A job is charged its estimate when submitted. When its call ends, it's marked completed, or
failed and refunded all but the GPU time it used (Modal bills that either way). The status
endpoint settles a job when it's polled, and the reconciler settles any nobody polls.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from db import SessionLocal
from models.database import AtlasJob, TransactionType
from services import atlas_store
from services.credit_service import calculate_atlas_cost, charge_flat, credit_refund
from services.modal_feature_stats import ATLAS_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
# A queued job with no Modal call this long after it was created was never spawned
UNSPAWNED_AFTER = timedelta(minutes=15)


def new_job_id() -> str:
    return f"atlas_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def active_job(db: Session, atlas_key: str) -> Optional[AtlasJob]:
    """The job computing an atlas now, if any (anyone's)."""
    return (
        db.query(AtlasJob)
        .filter(AtlasJob.atlas_key == atlas_key, AtlasJob.status.in_(ACTIVE))
        .order_by(AtlasJob.created_at.desc())
        .first()
    )


def create_job(
    db: Session,
    user_id: int,
    model_id: str,
    sae: dict,
    sae_name: str,
    atlas_key: str,
    corpus: str,
    tokens: int,
    cost: float,
    estimated_seconds: float,
) -> AtlasJob:
    """
    Record a job and charge for it, in one transaction.

    Raises:
        InsufficientBalanceError: the balance doesn't cover it (nothing is recorded)
    """
    job = AtlasJob(
        job_id=new_job_id(),
        user_id=user_id,
        model_id=model_id,
        sae=sae,
        sae_name=sae_name,
        atlas_key=atlas_key,
        corpus=corpus,
        tokens=tokens,
        status="queued",
        cost_usd=cost,
        estimated_seconds=estimated_seconds,
        created_at=datetime.utcnow(),
    )
    db.add(job)
    charge_flat(
        db, user_id, cost, TransactionType.GEOMETRY_CHARGE,
        f"Geometry atlas ({model_id}, {sae_name}): {tokens:,} {corpus} tokens",
    )
    db.refresh(job)
    return job


async def spawn_job(db: Session, job: AtlasJob, gpu_tier: str, layer: int, seed: int) -> None:
    """
    Start a recorded job on Modal. If that fails, the job fails with a full refund and the
    error is raised.
    """
    from services.modal_feature_stats import get_feature_stats_function

    try:
        call = await get_feature_stats_function(gpu_tier).spawn.aio(
            job.job_id, job.model_id, job.sae, job.sae_name, layer, job.corpus, job.tokens, seed,
        )
    except Exception as e:
        finish(db, job.job_id, Outcome("failed", "The job couldn't be started", seconds=0.0))
        raise RuntimeError(f"Couldn't start the atlas job: {e}")
    db.query(AtlasJob).filter(AtlasJob.job_id == job.job_id, AtlasJob.status == "queued").update(
        {AtlasJob.modal_call_id: call.object_id, AtlasJob.status: "running", AtlasJob.started_at: datetime.utcnow()},
        synchronize_session=False,
    )
    db.commit()


@dataclass
class Outcome:
    """How a job's call ended."""

    status: str  # "completed" or "failed"
    error: Optional[str] = None
    # Seconds the job ran, when known (the job reports them; a timeout runs the whole limit)
    seconds: Optional[float] = None


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
        return Outcome("failed", "The job ran out of time", seconds=float(ATLAS_TIMEOUT_SECONDS))
    if status != InputStatus.SUCCESS:
        return Outcome("failed", "The job stopped unexpectedly")
    try:
        result = await call.get.aio(timeout=0)
    except modal.exception.OutputExpiredError:
        return Outcome("failed", "Lost track of the job")
    return Outcome(result.get("status", "failed"), result.get("error"), result.get("seconds"))


def finish(db: Session, job_id: str, outcome: Outcome) -> bool:
    """
    Settle a job whose call ended, once: completed, or failed with a refund of all but the GPU
    time it ran (from its report, else the time since it started). Returns whether this call settled it.
    """
    job = db.query(AtlasJob).filter(AtlasJob.job_id == job_id).with_for_update().first()
    if job is None or job.status not in ACTIVE:
        db.rollback()
        return False
    job.finished_at = datetime.utcnow()
    if outcome.status == "completed":
        job.status = "completed"
        db.commit()
        _forget(job)
        return True

    seconds = outcome.seconds
    if seconds is None:
        started = job.started_at or job.created_at
        seconds = min((job.finished_at - started).total_seconds(), ATLAS_TIMEOUT_SECONDS)
    used = calculate_atlas_cost(job.model_id, seconds) if seconds > 0 else 0.0
    refund = max(0.0, float(job.cost_usd) - used)
    job.status = "failed"
    job.error_message = (outcome.error or "The job failed")[:1000]
    job.refunded_usd = round(refund, 4)
    credit_refund(
        db, job.user_id, refund, TransactionType.GEOMETRY_REFUND,
        f"Geometry atlas refund ({job.model_id}, {job.sae_name}): {job.error_message[:200]}",
    )
    db.commit()
    return True


def _forget(job: AtlasJob) -> None:
    from core.geometry.atlas import atlas_directory
    from core.probes.sae import sae_directory

    try:
        atlas_store.forget(atlas_directory(sae_directory(job.sae), job.model_id, job.corpus))
    except Exception:
        pass


async def refresh(db: Session, job: AtlasJob) -> AtlasJob:
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
        logger.exception("Failed to check atlas job %s", job.job_id)
        db.rollback()
    db.refresh(job)
    return job


async def progress(job_id: str) -> Optional[dict]:
    """A running job's progress, {"stage", "fraction"}, as its container last reported it."""
    try:
        import modal

        from services.modal_feature_stats import PROGRESS_DICT_NAME

        return await modal.Dict.from_name(PROGRESS_DICT_NAME).get.aio(job_id)
    except Exception:
        return None


async def reconcile_atlas_jobs() -> None:
    """Settle every unfinished atlas job whose call has ended (for jobs nobody polls)."""
    db = SessionLocal()
    try:
        jobs = db.query(AtlasJob).filter(AtlasJob.status.in_(ACTIVE)).all()
        for job in jobs:
            before = job.status
            job = await refresh(db, job)
            if job.status != before:
                logger.warning("Reconciled atlas job %s: %s (%s)", job.job_id, job.status, job.error_message)
    finally:
        db.close()
