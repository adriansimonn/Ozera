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
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db import get_db
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
from services.modal_volumes import read_dataset_metadata_from_volume
from middleware.auth_middleware import get_current_user, get_optional_current_user
from models.database import User, TrainingJob as TrainingJobModel, JobStatus as DBJobStatus, UploadedModel
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
        metadata = read_dataset_metadata_from_volume(current_user.id, request.dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    # Get GPU type from request
    gpu_type = request.gpu_type
    if gpu_type not in GPU_PRICING:
        raise HTTPException(status_code=400, detail=f"Invalid GPU type: {gpu_type}")

    # Check if user already has a custom model (limit to 1 across trained + uploaded)
    from services.modal_volumes import delete_model_from_volume

    total_models = get_total_custom_model_count(db, current_user.id)

    if total_models > 0:
        if not request.overwrite_existing:
            raise HTTPException(
                status_code=409,
                detail="You already have a custom model. Enable overwrite to replace it."
            )
        # Delete all existing trained models
        existing_trained = (
            db.query(TrainingJobModel)
            .filter(
                TrainingJobModel.user_id == current_user.id,
                TrainingJobModel.status == DBJobStatus.COMPLETED,
            )
            .all()
        )
        for model in existing_trained:
            await delete_model_from_volume(current_user.id, model.model_name)
            db.delete(model)

        # Delete all existing uploaded models
        existing_uploaded = (
            db.query(UploadedModel)
            .filter(UploadedModel.user_id == current_user.id)
            .all()
        )
        for model in existing_uploaded:
            await delete_model_from_volume(current_user.id, model.name)
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
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check authorization — only the job owner can stream progress
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

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

# Maximum file size for model upload: 500MB
MAX_MODEL_FILE_SIZE = 500 * 1024 * 1024


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
    from services.modal_volumes import upload_model_to_volume, delete_model_from_volume

    # Validate file extension
    if not file.filename or not file.filename.endswith(".safetensors"):
        raise HTTPException(
            status_code=400,
            detail="Only .safetensors files are accepted"
        )

    # Validate model name
    if not model_name or len(model_name) < 1 or len(model_name) > 64:
        raise HTTPException(
            status_code=400,
            detail="Model name must be between 1 and 64 characters"
        )

    if model_name.lower() in RESERVED_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Model name '{model_name}' is reserved for default Ozera models"
        )

    # Check if user already has a custom model (limit to 1)
    total_models = get_total_custom_model_count(db, current_user.id)

    if total_models > 0:
        if not overwrite_existing:
            raise HTTPException(
                status_code=409,
                detail="You already have a custom model. Enable overwrite to replace it."
            )

        # Delete existing trained models
        existing_trained = (
            db.query(TrainingJobModel)
            .filter(
                TrainingJobModel.user_id == current_user.id,
                TrainingJobModel.status == DBJobStatus.COMPLETED,
            )
            .all()
        )
        for model in existing_trained:
            await delete_model_from_volume(current_user.id, model.model_name)
            db.delete(model)

        # Delete existing uploaded models
        existing_uploaded = (
            db.query(UploadedModel)
            .filter(UploadedModel.user_id == current_user.id)
            .all()
        )
        for model in existing_uploaded:
            await delete_model_from_volume(current_user.id, model.name)
            db.delete(model)

        db.commit()

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

    # Validate safetensors format (basic header check)
    # Safetensors files start with a little-endian uint64 header size
    if len(content) < 8:
        raise HTTPException(
            status_code=400,
            detail="Invalid safetensors file: too small"
        )

    # Parse header size (first 8 bytes as little-endian uint64)
    import struct
    header_size = struct.unpack("<Q", content[:8])[0]

    if header_size > len(content) - 8 or header_size > 100 * 1024 * 1024:  # Header shouldn't be > 100MB
        raise HTTPException(
            status_code=400,
            detail="Invalid safetensors file: invalid header size"
        )

    # Try to parse the header JSON to extract model info
    num_parameters = None
    num_layers = None
    num_heads = None
    hidden_dim = None
    vocab_size = None
    max_seq_len = None

    try:
        header_json = content[8:8 + header_size].decode("utf-8")
        header = json.loads(header_json)

        # Try to extract model metadata if present
        metadata = header.get("__metadata__", {})
        if metadata:
            # Common metadata fields in safetensors
            if "parameters" in metadata:
                num_parameters = int(metadata["parameters"])
            if "num_layers" in metadata:
                num_layers = int(metadata["num_layers"])
            if "num_heads" in metadata:
                num_heads = int(metadata["num_heads"])
            if "hidden_dim" in metadata:
                hidden_dim = int(metadata["hidden_dim"])
            if "vocab_size" in metadata:
                vocab_size = int(metadata["vocab_size"])
            if "max_seq_len" in metadata:
                max_seq_len = int(metadata["max_seq_len"])

        # Count parameters from tensor shapes if not in metadata
        if num_parameters is None:
            total_params = 0
            for key, tensor_info in header.items():
                if key != "__metadata__" and isinstance(tensor_info, dict):
                    shape = tensor_info.get("shape", [])
                    if shape:
                        param_count = 1
                        for dim in shape:
                            param_count *= dim
                        total_params += param_count
            if total_params > 0:
                num_parameters = total_params

    except Exception as e:
        print(f"Warning: Could not parse safetensors header metadata: {e}")
        # Continue anyway - we can still upload the file

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

    # Save to database
    uploaded_model = UploadedModel(
        model_id=model_id,
        user_id=current_user.id,
        name=model_name,
        file_size_bytes=file_size,
        num_parameters=num_parameters,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    db.add(uploaded_model)
    db.commit()
    db.refresh(uploaded_model)

    return ModelUploadResponse(
        model_id=model_id,
        name=model_name,
        file_size_bytes=file_size,
        num_parameters=num_parameters,
        num_layers=num_layers,
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
    from services.modal_volumes import delete_model_from_volume

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

    # Delete from Modal volume
    await delete_model_from_volume(current_user.id, model.name)

    # Delete from database
    db.delete(model)
    db.commit()

    return {"status": "deleted", "model_id": model_id}
