"""
Webhook endpoints for receiving updates from Modal training jobs.
"""
import hmac
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models.database import TrainingJob, JobStatus
from services.job_orchestrator import settle_completed_job, handle_failed_job

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Get webhook secret from environment
MODAL_WEBHOOK_SECRET = os.environ.get("MODAL_WEBHOOK_SECRET", "")


class TrainingProgressUpdate(BaseModel):
    """Progress update from Modal training job."""
    status: str
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_ppl: Optional[float] = None
    val_ppl: Optional[float] = None
    elapsed_seconds: Optional[int] = None
    actual_minutes: Optional[float] = None
    error_message: Optional[str] = None


def verify_modal_secret(x_modal_secret: Optional[str] = Header(None)) -> bool:
    """
    Verify the Modal webhook secret.

    Args:
        x_modal_secret: Secret from X-Modal-Secret header

    Returns:
        True if valid

    Raises:
        HTTPException if invalid
    """
    if not MODAL_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Webhook secret not configured"
        )

    if not x_modal_secret or not hmac.compare_digest(x_modal_secret, MODAL_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    return True


@router.post("/training/{job_id}/progress")
async def receive_training_progress(
    job_id: str,
    update: TrainingProgressUpdate,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_modal_secret),
):
    """
    Receive progress updates from Modal training jobs.

    This endpoint is called by the Modal training function to report:
    - Epoch progress
    - Loss metrics
    - Job completion/failure

    Security: Protected by X-Modal-Secret header validation.
    """
    # Get the job from database, locked so a concurrent cancel or submission can't change its
    # status between this read and the commit below (their updates wait, then see this one).
    # Nothing before the commit may suspend (settle_completed_job and handle_failed_job
    # never do), or other requests would block on the lock with the event loop.
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).with_for_update().first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # A finished job (completed, failed, or cancelled) is already settled; ignore late
    # updates so it can't be charged twice
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        db.rollback()
        return {"status": "ignored", "job_id": job_id}

    # Update progress fields
    job.current_epoch = update.current_epoch
    job.total_epochs = update.total_epochs

    if update.train_loss is not None:
        job.train_loss = update.train_loss
    if update.val_loss is not None:
        job.val_loss = update.val_loss
    if update.train_ppl is not None:
        job.train_ppl = update.train_ppl
    if update.val_ppl is not None:
        job.val_ppl = update.val_ppl
    if update.error_message:
        job.error_message = update.error_message

    # Handle status changes
    if update.status == "completed":
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        # Settle the job (charge actual cost, refund difference)
        if update.actual_minutes is not None:
            await settle_completed_job(db, job_id, update.actual_minutes)
        else:
            # Calculate from elapsed_seconds if actual_minutes not provided
            actual_minutes = (update.elapsed_seconds or 0) / 60
            await settle_completed_job(db, job_id, actual_minutes)

    elif update.status == "failed":
        job.status = JobStatus.FAILED

        # Handle failed job (partial charge, refund rest)
        actual_minutes = update.actual_minutes or (update.elapsed_seconds or 0) / 60
        await handle_failed_job(
            db, job_id, actual_minutes, update.error_message or "Unknown error"
        )

    elif update.status == "running":
        # Update status if not already running (e.g., from queued)
        if job.status != JobStatus.RUNNING:
            job.status = JobStatus.RUNNING
            if not job.started_at:
                job.started_at = datetime.utcnow()

    db.commit()

    return {"status": "ok", "job_id": job_id}


@router.get("/training/{job_id}/status")
async def get_job_status_internal(
    job_id: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_modal_secret),
):
    """
    Get job status (for Modal function to check if job was cancelled).

    Returns:
        Job status and whether it should continue running
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        return {"status": "not_found", "should_continue": False}

    # Job should continue if status is RUNNING or QUEUED
    should_continue = job.status in [JobStatus.RUNNING, JobStatus.QUEUED]

    return {
        "status": job.status.value,
        "should_continue": should_continue,
    }
