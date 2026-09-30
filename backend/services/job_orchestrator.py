"""
Job orchestration layer for Modal cloud training.

Handles job submission to Modal, credit reservation, and job lifecycle management.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from models.database import TrainingJob, JobStatus, Dataset
from services.credit_service import (
    GPU_PRICING,
    reserve_credits,
    charge_credits,
    refund_credits,
)
from services.modal_volumes import (
    datasets_volume,
    check_dataset_exists,
    check_generic_dataset_exists,
    is_generic_dataset,
    parse_dataset_id,
)

# GPU throughput for cost estimation (tokens/second)
# Note: These are conservative estimates based on real-world training runs.
# Actual throughput varies with batch size, sequence length, and other factors.
GPU_THROUGHPUT = {
    "nano": {"t4": 25000, "a10g": 45000, "a100": 80000},
    "mini": {"t4": 5000, "a10g": 12000, "a100": 25000},
}

# Default GPU rates (from credit_service.py)
GPU_RATES = {gpu: info["rate_per_hour"] for gpu, info in GPU_PRICING.items()}

# Container time Modal bills that the job's reported minutes don't include: startup before
# the training function runs. (Workers report time through saving the model, and training
# containers shut down 2s after finishing.)
TRAINING_STARTUP_MINUTES = 1.0


def training_charge(gpu_type: str, minutes: float) -> float:
    """
    Charge for a training job that ran for `minutes`, including container startup.

    Unknown GPU types are charged at the highest rate so a job is never billed below cost.
    """
    gpu_rate = GPU_RATES.get(gpu_type, max(GPU_RATES.values()))
    return ((minutes + TRAINING_STARTUP_MINUTES) / 60) * gpu_rate

# Credit reservation buffer (20%)
RESERVATION_BUFFER = 1.2


def estimate_training_cost(
    num_tokens: int,
    model_config: str,
    epochs: int,
    gpu_type: str,
) -> Tuple[float, float]:
    """
    Estimate training time and cost.

    Args:
        num_tokens: Number of tokens in the dataset
        model_config: Model configuration ('nano' or 'mini')
        epochs: Number of training epochs
        gpu_type: GPU type ('t4', 'a10g', 'a100')

    Returns:
        Tuple of (estimated_minutes, estimated_cost_usd)
    """
    # Get throughput for this model/GPU combination
    throughput = GPU_THROUGHPUT.get(model_config, {}).get(gpu_type, 40000)

    # Calculate total tokens to process
    total_tokens = num_tokens * epochs

    # Estimate time
    estimated_seconds = total_tokens / throughput
    estimated_minutes = estimated_seconds / 60

    # Estimate cost
    estimated_cost = training_charge(gpu_type, estimated_minutes)

    return round(estimated_minutes, 1), round(estimated_cost, 2)


def generate_job_id() -> str:
    """Generate a unique job ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"job_{timestamp}_{unique_id}"


async def submit_training_job(
    db: Session,
    user_id: int,
    dataset_id: str,
    dataset_name: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    gpu_type: str,
    num_tokens: int,
) -> Tuple[Optional[TrainingJob], Optional[str]]:
    """
    Submit a training job to Modal.

    Args:
        db: Database session
        user_id: User ID
        dataset_id: Dataset ID
        dataset_name: Human-readable dataset name
        model_config: Model configuration ('nano' or 'mini')
        model_name: Name for the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seq_len: Sequence length
        gpu_type: GPU type ('t4', 'a10g', 'a100')
        num_tokens: Number of tokens in dataset

    Returns:
        Tuple of (TrainingJob, error_message). If error, TrainingJob is None.
    """
    # Estimate cost
    estimated_minutes, estimated_cost = estimate_training_cost(
        num_tokens, model_config, epochs, gpu_type
    )

    # Calculate reservation amount (with buffer), in whole cents so it releases exactly
    reservation_amount = round(estimated_cost * RESERVATION_BUFFER, 2)

    # Verify the dataset before reserving anything - generic datasets are already in the
    # volume, user datasets must have been uploaded
    is_generic, actual_dataset_id = parse_dataset_id(dataset_id)
    if is_generic:
        if not await check_generic_dataset_exists(actual_dataset_id):
            return None, f"Generic dataset not found: {actual_dataset_id}"
    elif not await check_dataset_exists(user_id, dataset_id):
        return None, f"Dataset not found in storage: {dataset_id}"

    job_id = generate_job_id()

    # Create the job as QUEUED: the worker stops if it doesn't see QUEUED or RUNNING, and it
    # can start before the RUNNING update below is committed
    job = TrainingJob(
        job_id=job_id,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        model_config=model_config,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seq_len=seq_len,
        gpu_type=gpu_type,
        status=JobStatus.QUEUED,
        estimated_cost_usd=estimated_cost,
        estimated_minutes=estimated_minutes,
        reserved_credits_usd=reservation_amount,
        total_epochs=epochs,
        created_at=datetime.utcnow(),
    )
    db.add(job)

    # Saves the job and its reservation in one transaction (or neither)
    try:
        reserved = reserve_credits(db, user_id, reservation_amount, job_id)
    except Exception as e:
        db.rollback()
        return None, f"Failed to submit job: {str(e)}"
    if not reserved:
        db.rollback()
        return None, f"Insufficient credits. Required: ${reservation_amount:.2f}"

    modal_call_id = None
    try:
        # Submit to Modal (asynchronously)
        modal_call_id = await _spawn_modal_job(
            job_id=job_id,
            user_id=user_id,
            dataset_id=dataset_id,
            model_config=model_config,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len=seq_len,
            dataset_name=dataset_name,
            gpu_type=gpu_type,
        )

        # Record the Modal call, and mark the job RUNNING only if it's still QUEUED: a cancel
        # during the spawn has already made it CANCELLED and released its reservation, and
        # overwriting that would let the job train unbilled
        job_rows = db.query(TrainingJob).filter(TrainingJob.job_id == job_id)
        job_rows.update({TrainingJob.modal_call_id: modal_call_id}, synchronize_session=False)
        started = job_rows.filter(TrainingJob.status == JobStatus.QUEUED).update(
            {TrainingJob.status: JobStatus.RUNNING, TrainingJob.started_at: datetime.utcnow()},
            synchronize_session=False,
        )
        db.commit()

    except Exception as e:
        db.rollback()
        if modal_call_id:
            # Spawned but not recorded as running: stop it rather than leave it running unbilled
            try:
                await _stop_modal_call(modal_call_id)
            except Exception:
                pass
        _fail_job_submission(db, job_id, f"Job submission failed: {str(e)}")
        return None, f"Failed to submit job: {str(e)}"

    if not started:
        # Cancelled during the spawn: stop the call too (if this fails, the worker still sees
        # the job is cancelled when it starts and stops itself)
        try:
            await _stop_modal_call(modal_call_id)
        except Exception as e:
            print(f"Failed to stop cancelled job {job_id} on Modal: {e}")

    db.refresh(job)
    return job, None


async def _stop_modal_call(modal_call_id: str) -> None:
    """Stop a spawned training call on Modal, terminating its container so it stops billing."""
    import modal
    await modal.FunctionCall.from_id(modal_call_id).cancel.aio(terminate_containers=True)


def _fail_job_submission(db: Session, job_id: str, error_message: str) -> None:
    """
    Mark a job whose submission failed as FAILED and release its reservation.

    Both are committed in one transaction, so the job can't be left QUEUED with its
    reservation already released (cancelling it would then release it again). A job
    cancelled in the meantime stays CANCELLED; its cancel already released the reservation.
    """
    failed = (
        db.query(TrainingJob)
        .filter(TrainingJob.job_id == job_id, TrainingJob.status == JobStatus.QUEUED)
        .update(
            {
                TrainingJob.status: JobStatus.FAILED,
                TrainingJob.error_message: error_message,
                TrainingJob.completed_at: datetime.utcnow(),
            },
            synchronize_session=False,
        )
    )
    if failed:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        refund_credits(db, job.user_id, 0, job.reserved_credits_usd, job_id, error_message[:500])
    db.commit()


async def _spawn_modal_job(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
    gpu_type: str,
) -> str:
    """
    Spawn a training job on Modal.

    Returns:
        Modal call ID
    """
    from services.modal_worker import get_training_function

    # Get the appropriate function for the GPU type
    training_fn = get_training_function(gpu_type)

    # Spawn the job (non-blocking)
    call = training_fn.spawn(
        job_id=job_id,
        user_id=user_id,
        dataset_id=dataset_id,
        model_config=model_config,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seq_len=seq_len,
        dataset_name=dataset_name,
    )

    return call.object_id


async def settle_completed_job(
    db: Session,
    job_id: str,
    actual_minutes: float,
) -> bool:
    """
    Settle a completed job - charge actual cost and refund difference.

    Args:
        db: Database session
        job_id: Job ID
        actual_minutes: Actual training duration in minutes

    Returns:
        True if successful
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        return False

    # Calculate actual cost
    actual_cost = training_charge(job.gpu_type, actual_minutes)

    # Update job with actual values
    job.actual_minutes = actual_minutes
    job.actual_cost_usd = round(actual_cost, 2)
    job.completed_at = datetime.utcnow()

    # Charge credits (this handles refund of difference)
    charge_credits(
        db=db,
        user_id=job.user_id,
        amount_usd=actual_cost,
        reserved_amount=job.reserved_credits_usd,
        job_id=job_id,
        description=f"Training job {job_id} on {job.gpu_type.upper()} - {actual_minutes:.1f} min"
    )

    db.commit()
    return True


async def handle_failed_job(
    db: Session,
    job_id: str,
    actual_minutes: float,
    error_message: str,
) -> bool:
    """
    Handle a failed job - charge for partial work and refund rest.

    Args:
        db: Database session
        job_id: Job ID
        actual_minutes: Actual training duration before failure
        error_message: Error message

    Returns:
        True if successful
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        return False

    # Charge for the container time Modal billed before the failure
    partial_cost = training_charge(job.gpu_type, actual_minutes)

    # Update job
    job.status = JobStatus.FAILED
    job.actual_minutes = actual_minutes
    job.actual_cost_usd = round(partial_cost, 2)
    job.error_message = error_message
    job.completed_at = datetime.utcnow()

    # Refund credits (minus partial charge)
    refund_credits(
        db=db,
        user_id=job.user_id,
        amount_usd=partial_cost,
        reserved_amount=job.reserved_credits_usd,
        job_id=job_id,
        description=f"Training job {job_id} failed: {error_message[:100]}"
    )

    db.commit()
    return True


async def cancel_job(
    db: Session,
    job_id: str,
    user_id: int,
) -> Tuple[bool, Optional[str]]:
    """
    Cancel a queued or running job.

    The job can change status while this runs: its submission marks it RUNNING (with a
    Modal call to stop) and its worker reports it finished. So the cancel only applies if
    the job still has the status it was seen with, and otherwise looks again. Overwriting a
    new status could leave a job running on Modal unbilled, or undo a finished job's
    settlement. Statuses only move forward, so this ends after a few tries at most.

    Args:
        db: Database session
        job_id: Job ID
        user_id: User ID (for authorization)

    Returns:
        Tuple of (success, error_message)
    """
    while True:
        job = db.query(TrainingJob).filter(
            TrainingJob.job_id == job_id,
            TrainingJob.user_id == user_id,
        ).first()

        if not job:
            return False, "Job not found"

        seen_status = job.status
        if seen_status not in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
            return False, f"Cannot cancel job with status: {seen_status.value}"

        # Stop the job on Modal; otherwise it keeps running (and billing) to completion. A
        # QUEUED job's call is stopped by its submission, which sees the cancel.
        if job.modal_call_id and seen_status == JobStatus.RUNNING:
            try:
                await _stop_modal_call(job.modal_call_id)
            except Exception:
                return False, "Failed to stop the job on Modal, please try again"

        # Calculate time so far, and the partial cost. Elapsed time runs from when the job
        # was spawned, so it already covers container startup.
        now = datetime.utcnow()
        elapsed = (now - job.started_at).total_seconds() / 60 if job.started_at else 0
        gpu_rate = GPU_RATES.get(job.gpu_type, max(GPU_RATES.values()))
        partial_cost = (elapsed / 60) * gpu_rate

        cancelled = (
            db.query(TrainingJob)
            .filter(TrainingJob.job_id == job_id, TrainingJob.status == seen_status)
            .update(
                {
                    TrainingJob.status: JobStatus.CANCELLED,
                    TrainingJob.actual_minutes: elapsed,
                    TrainingJob.actual_cost_usd: round(partial_cost, 2),
                    TrainingJob.completed_at: now,
                },
                synchronize_session=False,
            )
        )
        if not cancelled:
            # The status changed since it was read; look again
            db.rollback()
            continue

        # Refund credits (committed together with the cancel)
        refund_credits(
            db=db,
            user_id=user_id,
            amount_usd=partial_cost,
            reserved_amount=job.reserved_credits_usd,
            job_id=job_id,
            description=f"Training job {job_id} cancelled"
        )

        db.commit()
        return True, None


def get_user_jobs(
    db: Session,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[TrainingJob]:
    """
    Get all jobs for a user.

    Args:
        db: Database session
        user_id: User ID
        limit: Maximum number of jobs to return
        offset: Offset for pagination

    Returns:
        List of TrainingJob objects
    """
    return (
        db.query(TrainingJob)
        .filter(TrainingJob.user_id == user_id)
        .order_by(TrainingJob.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_job_by_id(db: Session, job_id: str) -> Optional[TrainingJob]:
    """Get a job by ID."""
    return db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
