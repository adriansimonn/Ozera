"""
Training job API endpoints with Modal cloud GPU support.

Supports multiple concurrent jobs per user with credit-based billing.
"""
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db import get_db
from api.schemas.training import (
    JobStatus,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingProgress,
    TrainingJobListItem,
    TrainingJobListResponse,
    TrainingEstimateRequest,
    TrainingEstimateResponse,
    CustomModelInfo,
)
from api.datasets import load_dataset_metadata
from middleware.auth_middleware import get_current_user, get_optional_current_user
from models.database import User, TrainingJob as TrainingJobModel, JobStatus as DBJobStatus
from services.job_orchestrator import (
    estimate_training_cost,
    submit_training_job,
    cancel_job,
    get_user_jobs,
    get_job_by_id,
    GPU_THROUGHPUT,
)
from services.credit_service import GPU_PRICING, check_sufficient_balance


router = APIRouter(prefix="/training", tags=["training"])


@router.post("/estimate", response_model=TrainingEstimateResponse)
async def get_training_estimate(
    request: TrainingEstimateRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Get cost and time estimate for training.

    Returns estimated training time and cost based on dataset size,
    model configuration, and GPU type.
    """
    from services.modal_volumes import parse_dataset_id, get_generic_dataset_metadata

    # Load dataset metadata - handle both generic and user-uploaded datasets
    is_generic, actual_dataset_id = parse_dataset_id(request.dataset_id)

    if is_generic:
        metadata = get_generic_dataset_metadata(actual_dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Generic dataset not found: {actual_dataset_id}")
    else:
        metadata = load_dataset_metadata(request.dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    num_tokens = metadata["num_tokens"]

    # Get GPU type from request (default to a10g)
    gpu_type = request.gpu_type

    # Calculate estimate
    estimated_minutes, estimated_cost = estimate_training_cost(
        num_tokens=num_tokens,
        model_config=request.base_model,
        epochs=request.epochs,
        gpu_type=gpu_type,
    )

    total_tokens = num_tokens * request.epochs

    warning = None
    if estimated_minutes > 120:
        warning = "Large dataset may take over 2 hours"
    elif estimated_minutes > 60:
        warning = "Training may take over 1 hour"

    return TrainingEstimateResponse(
        estimated_minutes=estimated_minutes,
        estimated_cost_usd=estimated_cost,
        total_tokens=total_tokens,
        tokens_per_epoch=num_tokens,
        warning=warning,
    )


# Reserved model names that cannot be used for custom models
RESERVED_MODEL_NAMES = ["ozera-nano", "ozera-mini"]


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training_job(
    request: TrainingJobRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start a new training job on Modal cloud GPU.

    Requires authentication and sufficient credit balance.
    Credits are reserved upfront and charged based on actual usage.
    """
    from services.modal_volumes import parse_dataset_id, get_generic_dataset_metadata

    # Validate model name is not reserved
    if request.model_name.lower() in RESERVED_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Model name '{request.model_name}' is reserved for default Ozera models"
        )

    # Validate dataset exists - handle both generic and user-uploaded datasets
    is_generic, actual_dataset_id = parse_dataset_id(request.dataset_id)

    if is_generic:
        metadata = get_generic_dataset_metadata(actual_dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Generic dataset not found: {actual_dataset_id}")
    else:
        metadata = load_dataset_metadata(request.dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    # Get GPU type from request
    gpu_type = request.gpu_type
    if gpu_type not in GPU_PRICING:
        raise HTTPException(status_code=400, detail=f"Invalid GPU type: {gpu_type}")

    # Check if user already has a custom model (limit to 1)
    existing_models = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .all()
    )

    if existing_models:
        if not request.overwrite_existing:
            raise HTTPException(
                status_code=409,
                detail="You already have a custom model. Enable overwrite to replace it."
            )
        # Delete all existing models (enforcing 1 model limit)
        for model in existing_models:
            db.delete(model)
        db.commit()

    # Estimate cost
    estimated_minutes, estimated_cost = estimate_training_cost(
        num_tokens=metadata["num_tokens"],
        model_config=request.base_model,
        epochs=request.epochs,
        gpu_type=gpu_type,
    )

    # Check sufficient balance (with 20% buffer)
    required_balance = estimated_cost * 1.2
    if not check_sufficient_balance(db, current_user.id, required_balance):
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: ${required_balance:.2f}"
        )

    # Submit job to Modal
    job, error = await submit_training_job(
        db=db,
        user_id=current_user.id,
        dataset_id=request.dataset_id,
        dataset_name=metadata["name"],
        model_config=request.base_model,
        model_name=request.model_name,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        seq_len=request.seq_len,
        gpu_type=gpu_type,
        num_tokens=metadata["num_tokens"],
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    return TrainingJobResponse(
        job_id=job.job_id,
        status=JobStatus(job.status.value),
        model_name=job.model_name,
        dataset_id=job.dataset_id,
        config={
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "seq_len": job.seq_len,
            "model_config": job.model_config,
            "gpu_type": job.gpu_type,
        },
        created_at=job.created_at,
        estimated_minutes=job.estimated_minutes,
        estimated_cost_usd=job.estimated_cost_usd,
    )


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    List training jobs.

    Returns only the authenticated user's jobs, or empty list if not logged in.
    """
    if not current_user:
        return TrainingJobListResponse(jobs=[])

    jobs = get_user_jobs(db, current_user.id)

    job_list = [
        TrainingJobListItem(
            job_id=job.job_id,
            status=JobStatus(job.status.value),
            model_name=job.model_name,
            dataset_name=job.dataset_name or "",
            created_at=job.created_at,
            current_epoch=job.current_epoch or 0,
            total_epochs=job.total_epochs or job.epochs,
        )
        for job in jobs
    ]

    return TrainingJobListResponse(jobs=job_list)


@router.get("/jobs/{job_id}", response_model=TrainingProgress)
async def get_job_status(
    job_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Get the current status of a training job."""
    job = get_job_by_id(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check authorization (job owner or public view for demo)
    if current_user and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Calculate elapsed and remaining time
    elapsed_seconds = 0
    estimated_remaining = 0

    if job.started_at:
        elapsed_seconds = int((datetime.utcnow() - job.started_at).total_seconds())

    if job.status == DBJobStatus.RUNNING and job.estimated_minutes:
        estimated_total_seconds = job.estimated_minutes * 60
        estimated_remaining = max(0, int(estimated_total_seconds - elapsed_seconds))

    return TrainingProgress(
        job_id=job.job_id,
        status=JobStatus(job.status.value),
        current_epoch=job.current_epoch or 0,
        total_epochs=job.total_epochs or job.epochs,
        current_step=0,
        total_steps=0,
        train_loss=job.train_loss,
        val_loss=job.val_loss,
        train_ppl=job.train_ppl,
        val_ppl=job.val_ppl,
        elapsed_seconds=elapsed_seconds,
        estimated_remaining_seconds=estimated_remaining,
        last_update=None,
        error_message=job.error_message,
    )


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(
    job_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Stream training job progress via Server-Sent Events.

    Sends progress updates every 2 seconds until the job completes or fails.
    Progress is read from the database (updated by Modal webhooks).
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_stream():
        last_epoch = -1

        while True:
            # Refresh job from database
            db.expire_all()
            job = get_job_by_id(db, job_id)

            if not job:
                data = json.dumps({"type": "error", "message": "Job not found"})
                yield f"data: {data}\n\n"
                break

            status = job.status.value

            # Calculate elapsed time
            elapsed_seconds = 0
            if job.started_at:
                elapsed_seconds = int((datetime.utcnow() - job.started_at).total_seconds())

            estimated_remaining = 0
            if job.status == DBJobStatus.RUNNING and job.estimated_minutes:
                estimated_total = job.estimated_minutes * 60
                estimated_remaining = max(0, int(estimated_total - elapsed_seconds))

            # Send progress update
            data = json.dumps({
                "type": "progress",
                "job_id": job_id,
                "status": status,
                "current_epoch": job.current_epoch or 0,
                "total_epochs": job.total_epochs or job.epochs,
                "train_loss": job.train_loss,
                "val_loss": job.val_loss,
                "train_ppl": job.train_ppl,
                "val_ppl": job.val_ppl,
                "elapsed_seconds": elapsed_seconds,
                "estimated_remaining_seconds": estimated_remaining,
                "gpu_type": job.gpu_type,
                "estimated_cost_usd": job.estimated_cost_usd,
                "actual_cost_usd": job.actual_cost_usd,
            })
            yield f"data: {data}\n\n"

            # Check for terminal states
            if status == "completed":
                data = json.dumps({
                    "type": "completed",
                    "job_id": job_id,
                    "model_name": job.model_name,
                    "actual_cost_usd": job.actual_cost_usd,
                    "actual_minutes": job.actual_minutes,
                })
                yield f"data: {data}\n\n"
                break

            if status == "failed":
                data = json.dumps({
                    "type": "error",
                    "job_id": job_id,
                    "message": job.error_message or "Training failed",
                })
                yield f"data: {data}\n\n"
                break

            if status == "cancelled":
                data = json.dumps({
                    "type": "cancelled",
                    "job_id": job_id,
                    "actual_cost_usd": job.actual_cost_usd,
                })
                yield f"data: {data}\n\n"
                break

            # Wait before next update
            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancel a running training job."""
    success, error = await cancel_job(db, job_id, current_user.id)

    if not success:
        if error == "Job not found":
            raise HTTPException(status_code=404, detail=error)
        raise HTTPException(status_code=400, detail=error)

    return {"status": "cancelled", "job_id": job_id}


# GPU Pricing endpoint
@router.get("/gpu-pricing")
async def get_gpu_pricing():
    """Get GPU pricing information."""
    pricing = []
    for gpu_type, info in GPU_PRICING.items():
        # Get throughput for a reference model
        nano_throughput = GPU_THROUGHPUT.get("nano", {}).get(gpu_type, 40000)
        mini_throughput = GPU_THROUGHPUT.get("mini", {}).get(gpu_type, 15000)

        pricing.append({
            "gpu_type": gpu_type,
            "display_name": info["display_name"],
            "rate_per_hour": info["rate_per_hour"],
            "description": info["description"],
            "nano_tokens_per_sec": nano_throughput,
            "mini_tokens_per_sec": mini_throughput,
        })

    return {"pricing": pricing, "default_gpu": "a10g"}


# Custom model management

@router.get("/models/count")
async def get_custom_model_count(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Get the count of custom models for the current user."""
    if not current_user:
        return {"count": 0, "max_allowed": 1}

    count = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .count()
    )

    return {"count": count, "max_allowed": 1}


@router.get("/models", response_model=list[CustomModelInfo])
async def list_custom_models(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """List all custom trained models for the current user."""
    if not current_user:
        return []

    # Get completed jobs for this user
    jobs = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .order_by(TrainingJobModel.completed_at.desc())
        .all()
    )

    models = []
    for job in jobs:
        models.append(CustomModelInfo(
            model_id=job.model_name,
            name=job.model_name,
            base_config=job.model_config,
            dataset_id=job.dataset_id,
            dataset_name=job.dataset_name or "",
            trained_at=job.completed_at or job.created_at,
            val_loss=job.val_loss or 0.0,
            parameters=0,  # Would need to load from metadata
        ))

    return models


@router.delete("/models/{model_id}")
async def delete_custom_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a custom trained model."""
    # Find the job that created this model
    job = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.model_name == model_id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .first()
    )

    if not job:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Delete from Modal volume (would need to implement)
    # For now, just mark the job as deleted by updating status
    # In production, you'd also delete the model files from Modal volume

    # Delete the job record
    db.delete(job)
    db.commit()

    return {"status": "deleted", "model_id": model_id}


@router.get("/models/{model_id}/download")
async def download_custom_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download a trained model as a zip file.

    Returns the model weights and config as a downloadable zip archive.
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse
    from services.modal_volumes import models_volume

    # Find the job that created this model
    job = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.model_name == model_id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .first()
    )

    if not job:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    try:
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        remote_dir = f"/{current_user.id}/{model_id}"

        # Debug: list what's in the volume
        print(f"[Download] Looking for model files in: {remote_dir}")

        # First, list root to see volume structure
        try:
            print(f"[Download] Volume root contents:")
            for root_entry in models_volume.listdir("/"):
                print(f"[Download]   root: {root_entry.path}")
        except Exception as e:
            print(f"[Download] Error listing root: {e}")

        files_found = 0

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # List and download all model files
            for entry in models_volume.listdir(remote_dir):
                print(f"[Download] Found entry: {entry.path}, type: {getattr(entry, 'type', 'unknown')}")
                # Only download model.pt file
                if str(entry.path).endswith('.pt'):
                    file_name = Path(entry.path).name
                    # Read file content from volume
                    file_content = b""
                    for chunk in models_volume.read_file(entry.path):
                        file_content += chunk
                    zip_file.writestr(file_name, file_content)
                    files_found += 1
                    print(f"[Download] Added {file_name} ({len(file_content)} bytes)")

        print(f"[Download] Total files found: {files_found}")

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{model_id}.zip"'
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {str(e)}"
        )
