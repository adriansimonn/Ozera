"""
Pydantic models for dataset and training API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    QUEUED = "queued"  # Waiting in Modal queue
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Dataset schemas

class DatasetMetadata(BaseModel):
    """Dataset metadata."""
    dataset_id: str
    name: str
    size_bytes: int
    num_tokens: int
    created_at: datetime


class DatasetDetail(DatasetMetadata):
    """Dataset details including preview."""
    preview: str = Field(description="First 500 characters of the dataset")


class DatasetListResponse(BaseModel):
    """Response for listing datasets."""
    datasets: list[DatasetMetadata]


# Training job schemas

class TrainingJobRequest(BaseModel):
    """Request to start a training job."""
    dataset_id: str = Field(..., description="ID of the dataset to train on")
    base_model: str = Field(..., pattern="^(nano|mini)$", description="Model architecture")
    model_name: str = Field(..., min_length=1, max_length=64, description="Name for the trained model")
    epochs: int = Field(default=20, ge=5, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Batch size")
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2, description="Learning rate")
    seq_len: int = Field(default=256, ge=64, le=1024, description="Sequence length")
    gpu_type: str = Field(default="a10g", pattern="^(t4|a10g|a100)$", description="GPU type for training")
    overwrite_existing: bool = Field(default=False, description="If true, delete existing custom model before training")


class TrainingJobResponse(BaseModel):
    """Response after starting a training job."""
    job_id: str
    status: JobStatus
    model_name: str
    dataset_id: str
    config: dict
    created_at: datetime
    estimated_minutes: float
    estimated_cost_usd: float


class TrainingProgress(BaseModel):
    """Training job progress."""
    job_id: str
    status: JobStatus
    current_epoch: int
    total_epochs: int
    current_step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_ppl: Optional[float] = None
    val_ppl: Optional[float] = None
    elapsed_seconds: int = 0
    estimated_remaining_seconds: int = 0
    last_update: Optional[datetime] = None
    error_message: Optional[str] = None


class TrainingJobListItem(BaseModel):
    """Training job list item."""
    job_id: str
    status: JobStatus
    model_name: str
    dataset_name: str
    created_at: datetime
    current_epoch: int = 0
    total_epochs: int = 0


class TrainingJobListResponse(BaseModel):
    """Response for listing training jobs."""
    jobs: list[TrainingJobListItem]


# Cost estimation schemas

class TrainingEstimateRequest(BaseModel):
    """Request for training cost estimate."""
    dataset_id: str
    base_model: str = Field(..., pattern="^(nano|mini)$")
    epochs: int = Field(default=20, ge=5, le=100)
    batch_size: int = Field(default=32, ge=8, le=128)
    seq_len: int = Field(default=256, ge=64, le=1024)
    gpu_type: str = Field(default="a10g", pattern="^(t4|a10g|a100)$", description="GPU type for training")


class TrainingEstimateResponse(BaseModel):
    """Training cost estimate response."""
    estimated_minutes: float
    estimated_cost_usd: float
    total_tokens: int
    tokens_per_epoch: int
    warning: Optional[str] = None


# Custom model schemas

class CustomModelInfo(BaseModel):
    """Custom model information."""
    model_id: str
    name: str
    base_config: str
    dataset_id: str
    dataset_name: str
    trained_at: datetime
    val_loss: float
    parameters: int
