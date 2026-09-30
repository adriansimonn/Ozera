"""
Training job API endpoints with Modal cloud GPU support.

Supports multiple concurrent jobs per user with credit-based billing.
"""
import json
import re
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db import SessionLocal, get_db
from middleware.rate_limit import limiter
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
    UploadedModelInfo,
    ModelUploadResponse,
    CustomModelCount,
)
from core.model_names import is_valid_model_name, model_folder
from core.transformer.checkpoint import InvalidCheckpointError, read_safetensors_header
from core.open_source import OPEN_SOURCE_MODELS
from services.inference_router import BASE_MODELS
from services.modal_volumes import read_dataset_metadata_from_volume
from middleware.auth_middleware import get_current_user, get_optional_current_user
from models.database import User, TrainingJob as TrainingJobModel, JobStatus as DBJobStatus, UploadedModel
from services.job_orchestrator import (
    estimate_training_cost,
    submit_training_job,
    cancel_job,
    get_user_jobs,
    get_job_by_id,
    training_time_error,
    GPU_THROUGHPUT,
)
from services.credit_service import GPU_PRICING, check_sufficient_balance
from services.custom_models import delete_replaced_models, remove_replaced_model_files
from services.model_specs import uploaded_model_fields


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
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required for user datasets")
        metadata = read_dataset_metadata_from_volume(current_user.id, request.dataset_id)
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

    warning = training_time_error(estimated_minutes)
    if warning is None and estimated_minutes > 60:
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


def _validate_model_name(model_name: str) -> None:
    """
    Reject a custom model name that is malformed or reserved (400).

    Names become folder names on the models volume (see core.model_names). Requests name
    models by name, and the base and open-source models' IDs take precedence, so a custom
    model can't use one of them.
    """
    if not is_valid_model_name(model_name):
        raise HTTPException(
            status_code=400,
            detail="Model name must be 1-64 characters: letters, digits, '_' and '-'"
        )

    reserved = {name.lower() for name in [*RESERVED_MODEL_NAMES, *BASE_MODELS, *OPEN_SOURCE_MODELS]}
    if model_name.lower() in reserved:
        raise HTTPException(
            status_code=400,
            detail=f"Model name '{model_name}' is reserved for a built-in model"
        )


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

    _validate_model_name(request.model_name)

    # Validate dataset exists - handle both generic and user-uploaded datasets
    is_generic, actual_dataset_id = parse_dataset_id(request.dataset_id)

    if is_generic:
        metadata = get_generic_dataset_metadata(actual_dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Generic dataset not found: {actual_dataset_id}")
    else:
        metadata = read_dataset_metadata_from_volume(current_user.id, request.dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    # Get GPU type from request
    gpu_type = request.gpu_type
    if gpu_type not in GPU_PRICING:
        raise HTTPException(status_code=400, detail=f"Invalid GPU type: {gpu_type}")

    # Check if user already has a custom model (limit to 1 across trained + uploaded). The
    # existing models are replaced when the new job completes (see webhooks.py), so they're
    # kept if it fails or is cancelled.
    if get_total_custom_model_count(db, current_user.id) > 0 and not request.overwrite_existing:
        raise HTTPException(
            status_code=409,
            detail="You already have a custom model. Enable overwrite to replace it."
        )

    # Estimate cost
    estimated_minutes, estimated_cost = estimate_training_cost(
        num_tokens=metadata["num_tokens"],
        model_config=request.base_model,
        epochs=request.epochs,
        gpu_type=gpu_type,
    )

    # A job has to finish within the training container's timeout
    time_error = training_time_error(estimated_minutes)
    if time_error:
        raise HTTPException(status_code=400, detail=time_error)

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the current status of a training job. Requires authentication."""
    job = get_job_by_id(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check authorization — only the job owner can view
    if job.user_id != current_user.id:
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


def _read_job_progress(job_id: str) -> Optional[dict]:
    """
    Snapshot of a job's progress for the progress stream, read in a session of its own.

    Returns None if the job doesn't exist.
    """
    db = SessionLocal()
    try:
        job = get_job_by_id(db, job_id)
        if not job:
            return None

        elapsed_seconds = 0
        if job.started_at:
            elapsed_seconds = int((datetime.utcnow() - job.started_at).total_seconds())

        estimated_remaining = 0
        if job.status == DBJobStatus.RUNNING and job.estimated_minutes:
            estimated_remaining = max(0, int(job.estimated_minutes * 60 - elapsed_seconds))

        def usd(value) -> Optional[float]:
            # Numeric columns load as Decimal, which JSON can't encode
            return None if value is None else float(value)

        return {
            "status": job.status.value,
            "model_name": job.model_name,
            "current_epoch": job.current_epoch or 0,
            "total_epochs": job.total_epochs or job.epochs,
            "train_loss": job.train_loss,
            "val_loss": job.val_loss,
            "train_ppl": job.train_ppl,
            "val_ppl": job.val_ppl,
            "elapsed_seconds": elapsed_seconds,
            "estimated_remaining_seconds": estimated_remaining,
            "gpu_type": job.gpu_type,
            "estimated_cost_usd": usd(job.estimated_cost_usd),
            "actual_cost_usd": usd(job.actual_cost_usd),
            "actual_minutes": job.actual_minutes,
            "error_message": job.error_message,
        }
    finally:
        db.close()


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Stream training job progress via Server-Sent Events.

    Sends progress updates every 2 seconds until the job completes or fails.
    Progress is read from the database (updated by Modal webhooks).
    Requires authentication — only the job owner can stream progress.

    A stream can stay open for the whole training run, so it holds no database connection
    between polls: each poll reads the job in its own short session, off the event loop.
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check authorization — only the job owner can stream progress
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Return the request's connection to the pool now, rather than when the stream ends
    db.close()

    async def event_stream():
        while True:
            job = await run_in_threadpool(_read_job_progress, job_id)

            if job is None:
                data = json.dumps({"type": "error", "message": "Job not found"})
                yield f"data: {data}\n\n"
                break

            status = job["status"]

            # Send progress update
            data = json.dumps({
                "type": "progress",
                "job_id": job_id,
                "status": status,
                "current_epoch": job["current_epoch"],
                "total_epochs": job["total_epochs"],
                "train_loss": job["train_loss"],
                "val_loss": job["val_loss"],
                "train_ppl": job["train_ppl"],
                "val_ppl": job["val_ppl"],
                "elapsed_seconds": job["elapsed_seconds"],
                "estimated_remaining_seconds": job["estimated_remaining_seconds"],
                "gpu_type": job["gpu_type"],
                "estimated_cost_usd": job["estimated_cost_usd"],
                "actual_cost_usd": job["actual_cost_usd"],
            })
            yield f"data: {data}\n\n"

            # Check for terminal states
            if status == "completed":
                data = json.dumps({
                    "type": "completed",
                    "job_id": job_id,
                    "model_name": job["model_name"],
                    "actual_cost_usd": job["actual_cost_usd"],
                    "actual_minutes": job["actual_minutes"],
                })
                yield f"data: {data}\n\n"
                break

            if status == "failed":
                data = json.dumps({
                    "type": "error",
                    "job_id": job_id,
                    "message": job["error_message"] or "Training failed",
                })
                yield f"data: {data}\n\n"
                break

            if status == "cancelled":
                data = json.dumps({
                    "type": "cancelled",
                    "job_id": job_id,
                    "actual_cost_usd": job["actual_cost_usd"],
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

# Maximum file size for model upload: 500MB
MAX_MODEL_FILE_SIZE = 500 * 1024 * 1024


async def _delete_model_files(db: Session, user_id: int, model_name: str, deleting: list) -> None:
    """
    Delete a model's folder from the models volume, before deleting its records `deleting`.

    Raises a 500 if that fails, so the records are kept and the delete can be retried. The
    folder stays if another of the user's models with the same name (so the same folder)
    isn't being deleted.
    """
    from services.modal_volumes import delete_model_from_volume

    same_folder = [
        *db.query(TrainingJobModel).filter(
            TrainingJobModel.user_id == user_id,
            TrainingJobModel.model_name == model_name,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        ),
        *db.query(UploadedModel).filter(UploadedModel.user_id == user_id, UploadedModel.name == model_name),
    ]
    if any(model not in deleting for model in same_folder):
        return

    if not await delete_model_from_volume(user_id, model_name):
        raise HTTPException(status_code=500, detail="Failed to delete the model's files, please try again")


def get_total_custom_model_count(db: Session, user_id: int) -> int:
    """Get the total count of custom models (trained + uploaded) for a user."""
    trained_count = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == user_id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .count()
    )

    uploaded_count = (
        db.query(UploadedModel)
        .filter(UploadedModel.user_id == user_id)
        .count()
    )

    return trained_count + uploaded_count


@router.get("/models/count", response_model=CustomModelCount)
async def get_custom_model_count(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """Get the count of custom models (trained + uploaded) for the current user."""
    if not current_user:
        return CustomModelCount(trained_count=0, uploaded_count=0, total_count=0, max_allowed=1)

    trained_count = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .count()
    )

    uploaded_count = (
        db.query(UploadedModel)
        .filter(UploadedModel.user_id == current_user.id)
        .count()
    )

    return CustomModelCount(
        trained_count=trained_count,
        uploaded_count=uploaded_count,
        total_count=trained_count + uploaded_count,
        max_allowed=1,
    )


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
    """Delete a custom trained model: its weights on the models volume and its job record."""
    # Find the job(s) that created this model; completed jobs with the same name all stored
    # their model in the same folder
    jobs = (
        db.query(TrainingJobModel)
        .filter(
            TrainingJobModel.user_id == current_user.id,
            TrainingJobModel.model_name == model_id,
            TrainingJobModel.status == DBJobStatus.COMPLETED,
        )
        .all()
    )

    if not jobs:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Delete the weights first (keeping the records if that fails)
    await _delete_model_files(db, current_user.id, model_id, jobs)

    # Delete the job records
    for job in jobs:
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
        remote_dir = f"/{model_folder(current_user.id, job.model_name)}"

        files_found = 0

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # List and download all model files
            for entry in models_volume.listdir(remote_dir):
                # Download model.safetensors file (or .pt for backwards compatibility)
                if str(entry.path).endswith('.safetensors') or str(entry.path).endswith('.pt'):
                    file_name = Path(entry.path).name
                    # Read file content from volume
                    file_content = b""
                    for chunk in models_volume.read_file(entry.path):
                        file_content += chunk
                    zip_file.writestr(file_name, file_content)
                    files_found += 1

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id)}.zip"'
            }
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to download model"
        )


# Uploaded model management

@router.post("/models/upload", response_model=ModelUploadResponse)
@limiter.limit("3/minute")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
    overwrite_existing: bool = Form(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a .safetensors model file.

    The model will be available for text generation and visualization.
    Users are limited to 1 custom model total (trained or uploaded).

    - Maximum file size: 500MB
    - Accepted formats: .safetensors
    """
    from services.modal_volumes import upload_model_to_volume

    # Validate file extension
    if not file.filename or not file.filename.endswith(".safetensors"):
        raise HTTPException(
            status_code=400,
            detail="Only .safetensors files are accepted"
        )

    _validate_model_name(model_name)

    # Check if user already has a custom model (limit to 1). The existing models are only
    # replaced once the new file is validated and stored.
    if get_total_custom_model_count(db, current_user.id) > 0 and not overwrite_existing:
        raise HTTPException(
            status_code=409,
            detail="You already have a custom model. Enable overwrite to replace it."
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file size
    if file_size > MAX_MODEL_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_MODEL_FILE_SIZE // (1024*1024)}MB"
        )

    if file_size < 1024:  # Less than 1KB is suspiciously small
        raise HTTPException(
            status_code=400,
            detail="File too small. Model files should be at least 1KB"
        )

    # Read the model's specs from the header the same way the inference workers build the
    # model from it, so they're known without loading the model on a GPU
    try:
        metadata, shapes = read_safetensors_header(content)
        specs = uploaded_model_fields(metadata, shapes)
    except InvalidCheckpointError as e:
        raise HTTPException(status_code=400, detail=f"Invalid model file: {e}")

    # Save to temporary file and upload to Modal volume
    # Use model_name as model_id since that's the folder name on the volume
    model_id = model_name

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    try:
        success = await upload_model_to_volume(tmp_path, current_user.id, model_name)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload model to storage"
            )
    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

    # Replace the existing models, now that the new one is stored
    replaced_models = set()
    if overwrite_existing:
        replaced_models = delete_replaced_models(db, current_user.id)

    # Save to database
    uploaded_model = UploadedModel(
        model_id=model_id,
        user_id=current_user.id,
        name=model_name,
        file_size_bytes=file_size,
        **specs,
    )
    db.add(uploaded_model)
    db.commit()
    db.refresh(uploaded_model)

    await remove_replaced_model_files(current_user.id, replaced_models, model_name)

    return ModelUploadResponse(
        model_id=model_id,
        name=model_name,
        file_size_bytes=file_size,
        num_parameters=specs["num_parameters"],
        num_layers=specs["num_layers"],
        status="uploaded",
    )


@router.get("/models/uploaded", response_model=list[UploadedModelInfo])
async def list_uploaded_models(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """List all uploaded models for the current user."""
    if not current_user:
        return []

    models = (
        db.query(UploadedModel)
        .filter(UploadedModel.user_id == current_user.id)
        .order_by(UploadedModel.created_at.desc())
        .all()
    )

    return [
        UploadedModelInfo(
            model_id=model.model_id,
            name=model.name,
            file_size_bytes=model.file_size_bytes,
            num_parameters=model.num_parameters,
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            hidden_dim=model.hidden_dim,
            vocab_size=model.vocab_size,
            max_seq_len=model.max_seq_len,
            uploaded_at=model.created_at,
            model_type="uploaded",
        )
        for model in models
    ]


@router.delete("/models/uploaded/{model_id}")
async def delete_uploaded_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an uploaded model."""
    model = (
        db.query(UploadedModel)
        .filter(
            UploadedModel.user_id == current_user.id,
            UploadedModel.model_id == model_id,
        )
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail=f"Uploaded model not found: {model_id}")

    # Delete from Modal volume (first, keeping the record if that fails)
    await _delete_model_files(db, current_user.id, model.name, [model])

    # Delete from database
    db.delete(model)
    db.commit()

    return {"status": "deleted", "model_id": model_id}
