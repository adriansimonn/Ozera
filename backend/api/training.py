"""
Training job API endpoints.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
from services.job_manager import (
    get_job_manager,
    load_job_config,
    load_job_progress,
    list_all_jobs,
    get_job_dir,
)
from services.training_worker import run_training_job
from api.datasets import load_dataset_metadata


router = APIRouter(prefix="/training", tags=["training"])

# Cost estimation constants (configurable)
TOKENS_PER_SECOND = {
    "nano": 50000,  # ~50K tokens/sec on consumer GPU
    "mini": 15000,  # ~15K tokens/sec on consumer GPU
}
COST_PER_HOUR = 0.50  # USD


def estimate_training(
    num_tokens: int,
    model_config: str,
    epochs: int,
) -> dict:
    """Calculate training time and cost estimate."""
    tokens_per_second = TOKENS_PER_SECOND.get(model_config, 50000)
    total_tokens = num_tokens * epochs
    estimated_seconds = total_tokens / tokens_per_second
    estimated_minutes = estimated_seconds / 60
    estimated_cost = (estimated_seconds / 3600) * COST_PER_HOUR

    warning = None
    if estimated_minutes > 120:
        warning = "Large dataset may take over 2 hours"
    elif estimated_minutes > 60:
        warning = "Training may take over 1 hour"

    return {
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_cost_usd": round(estimated_cost, 2),
        "total_tokens": total_tokens,
        "tokens_per_epoch": num_tokens,
        "warning": warning,
    }


@router.post("/estimate", response_model=TrainingEstimateResponse)
async def get_training_estimate(request: TrainingEstimateRequest):
    """
    Get cost and time estimate for training.

    Returns estimated training time and cost based on dataset size and model configuration.
    """
    # Load dataset to get token count
    metadata = load_dataset_metadata(request.dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    num_tokens = metadata["num_tokens"]

    estimate = estimate_training(
        num_tokens=num_tokens,
        model_config=request.base_model,
        epochs=request.epochs,
    )

    return TrainingEstimateResponse(**estimate)


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training_job(request: TrainingJobRequest):
    """
    Start a new training job.

    Only one training job can run at a time. Returns error if a job is already running.
    """
    job_manager = get_job_manager()

    # Check if a job is already running
    running_job = job_manager.get_running_job_id()
    if running_job:
        raise HTTPException(
            status_code=409,
            detail=f"A training job is already running: {running_job}"
        )

    # Validate dataset exists
    metadata = load_dataset_metadata(request.dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    # Check if model name already exists
    custom_models_dir = Path(__file__).parent.parent / "models" / "custom"
    if (custom_models_dir / request.model_name).exists():
        raise HTTPException(
            status_code=409,
            detail=f"A model with name '{request.model_name}' already exists"
        )

    # Get estimate
    estimate = estimate_training(
        num_tokens=metadata["num_tokens"],
        model_config=request.base_model,
        epochs=request.epochs,
    )

    # Create job
    job_id = job_manager.create_job(
        dataset_id=request.dataset_id,
        dataset_name=metadata["name"],
        model_config=request.base_model,
        model_name=request.model_name,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        seq_len=request.seq_len,
        estimated_minutes=estimate["estimated_minutes"],
        estimated_cost_usd=estimate["estimated_cost_usd"],
    )

    # Start job
    started = job_manager.start_job(job_id, run_training_job)
    if not started:
        raise HTTPException(
            status_code=409,
            detail="Failed to start training job - another job may have started"
        )

    # Load config to return
    config = load_job_config(job_id)

    return TrainingJobResponse(
        job_id=job_id,
        status=JobStatus.RUNNING,
        model_name=request.model_name,
        dataset_id=request.dataset_id,
        config={
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "seq_len": request.seq_len,
            "model_config": request.base_model,
        },
        created_at=datetime.fromisoformat(config["created_at"]),
        estimated_minutes=estimate["estimated_minutes"],
        estimated_cost_usd=estimate["estimated_cost_usd"],
    )


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs():
    """List all training jobs."""
    jobs = list_all_jobs()
    return TrainingJobListResponse(
        jobs=[TrainingJobListItem(**j) for j in jobs]
    )


@router.get("/jobs/{job_id}", response_model=TrainingProgress)
async def get_job_status(job_id: str):
    """Get the current status of a training job."""
    job_manager = get_job_manager()
    progress = job_manager.get_job_status(job_id)

    if not progress:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return progress


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Stream training job progress via Server-Sent Events.

    Sends progress updates every 2 seconds until the job completes or fails.
    """
    progress = load_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_stream():
        last_epoch = -1

        while True:
            progress = load_job_progress(job_id)
            if not progress:
                data = json.dumps({"type": "error", "message": "Job not found"})
                yield f"data: {data}\n\n"
                break

            status = progress.get("status", JobStatus.PENDING.value)

            # Always send progress update
            data = json.dumps({
                "type": "progress",
                "job_id": job_id,
                "status": status,
                "current_epoch": progress.get("current_epoch", 0),
                "total_epochs": progress.get("total_epochs", 0),
                "train_loss": progress.get("train_loss"),
                "val_loss": progress.get("val_loss"),
                "train_ppl": progress.get("train_ppl"),
                "val_ppl": progress.get("val_ppl"),
                "elapsed_seconds": progress.get("elapsed_seconds", 0),
                "estimated_remaining_seconds": progress.get("estimated_remaining_seconds", 0),
            })
            yield f"data: {data}\n\n"

            # Check for terminal states
            if status == JobStatus.COMPLETED.value:
                # Get model name from config
                config = load_job_config(job_id)
                model_name = config.get("model_name", "") if config else ""
                data = json.dumps({
                    "type": "completed",
                    "job_id": job_id,
                    "model_name": model_name,
                })
                yield f"data: {data}\n\n"
                break

            if status == JobStatus.FAILED.value:
                data = json.dumps({
                    "type": "error",
                    "job_id": job_id,
                    "message": progress.get("error_message", "Training failed"),
                })
                yield f"data: {data}\n\n"
                break

            if status == JobStatus.CANCELLED.value:
                data = json.dumps({
                    "type": "cancelled",
                    "job_id": job_id,
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
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    job_manager = get_job_manager()

    progress = load_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    status = progress.get("status", JobStatus.PENDING.value)
    if status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {status}"
        )

    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")

    return {"status": "cancelled", "job_id": job_id}


# Custom model management

@router.get("/models", response_model=list[CustomModelInfo])
async def list_custom_models():
    """List all custom trained models."""
    custom_models_dir = Path(__file__).parent.parent / "models" / "custom"
    if not custom_models_dir.exists():
        return []

    models = []
    for model_dir in custom_models_dir.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    models.append(CustomModelInfo(**metadata))

    # Sort by trained_at descending
    models.sort(key=lambda x: x.trained_at, reverse=True)
    return models


@router.delete("/models/{model_id}")
async def delete_custom_model(model_id: str):
    """Delete a custom trained model."""
    import shutil

    custom_models_dir = Path(__file__).parent.parent / "models" / "custom"
    model_dir = custom_models_dir / model_id

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    shutil.rmtree(model_dir)

    return {"status": "deleted", "model_id": model_id}
