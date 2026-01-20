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
    check_sufficient_balance,
)
from services.modal_volumes import (
    datasets_volume,
    upload_dataset_to_volume,
    check_dataset_exists,
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

# Credit reservation buffer (20%)
RESERVATION_BUFFER = 1.2

# Local datasets directory
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"


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
    gpu_rate = GPU_RATES.get(gpu_type, 0.80)

    # Calculate total tokens to process
    total_tokens = num_tokens * epochs

    # Estimate time
    estimated_seconds = total_tokens / throughput
    estimated_minutes = estimated_seconds / 60

    # Estimate cost
    estimated_hours = estimated_minutes / 60
    estimated_cost = estimated_hours * gpu_rate

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

    # Calculate reservation amount (with buffer)
    reservation_amount = estimated_cost * RESERVATION_BUFFER

    # Check if user has sufficient balance
    if not check_sufficient_balance(db, user_id, reservation_amount):
        return None, f"Insufficient credits. Required: ${reservation_amount:.2f}"

    # Generate job ID
    job_id = generate_job_id()

    # Reserve credits
    if not reserve_credits(db, user_id, reservation_amount, job_id):
        return None, "Failed to reserve credits"

    try:
        # Create job record in database
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
            status=JobStatus.PENDING,
            estimated_cost_usd=estimated_cost,
            estimated_minutes=estimated_minutes,
            reserved_credits_usd=reservation_amount,
            total_epochs=epochs,
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Upload dataset to Modal volume if not already there
        dataset_exists = await check_dataset_exists(user_id, dataset_id)
        if not dataset_exists:
            local_dataset_path = DATASETS_DIR / dataset_id / "raw.txt"
            if local_dataset_path.exists():
                await upload_dataset_to_volume(local_dataset_path, user_id, dataset_id)
            else:
                # Try user-scoped path
                user_dataset_path = DATASETS_DIR / str(user_id) / dataset_id / "raw.txt"
                if user_dataset_path.exists():
                    await upload_dataset_to_volume(user_dataset_path, user_id, dataset_id)

        # Update status to queued
        job.status = JobStatus.QUEUED
        db.commit()

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

        # Store Modal call ID
        job.modal_call_id = modal_call_id
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        return job, None

    except Exception as e:
        # Refund credits on failure
        refund_credits(db, user_id, 0, reservation_amount, job_id,
                       f"Job submission failed: {str(e)}")
        db.rollback()
        return None, f"Failed to submit job: {str(e)}"


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
    gpu_rate = GPU_RATES.get(job.gpu_type, 0.80)
    actual_hours = actual_minutes / 60
    actual_cost = actual_hours * gpu_rate

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

    # Calculate partial cost (only charge if meaningful work was done)
    gpu_rate = GPU_RATES.get(job.gpu_type, 0.80)
    actual_hours = actual_minutes / 60
    partial_cost = actual_hours * gpu_rate if actual_minutes > 0.5 else 0

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
    Cancel a running job.

    Args:
        db: Database session
        job_id: Job ID
        user_id: User ID (for authorization)

    Returns:
        Tuple of (success, error_message)
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.job_id == job_id,
        TrainingJob.user_id == user_id,
    ).first()

    if not job:
        return False, "Job not found"

    if job.status not in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
        return False, f"Cannot cancel job with status: {job.status.value}"

    # Try to cancel on Modal if running
    if job.modal_call_id and job.status == JobStatus.RUNNING:
        try:
            import modal
            # Cancel the Modal function call
            # Note: This may not immediately stop the job
            pass  # Modal doesn't have a direct cancel API for spawned calls
        except Exception:
            pass

    # Calculate time so far
    if job.started_at:
        elapsed = (datetime.utcnow() - job.started_at).total_seconds() / 60
    else:
        elapsed = 0

    # Update job status
    job.status = JobStatus.CANCELLED
    job.actual_minutes = elapsed
    job.completed_at = datetime.utcnow()

    # Calculate partial cost
    gpu_rate = GPU_RATES.get(job.gpu_type, 0.80)
    partial_cost = (elapsed / 60) * gpu_rate if elapsed > 0.5 else 0
    job.actual_cost_usd = round(partial_cost, 2)

    # Refund credits
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
