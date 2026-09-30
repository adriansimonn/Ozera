"""
Resolving a requested model name to a model the requesting user may run.

Base and open-source models are shared. A custom model (trained or uploaded) belongs to one
user and lives in that user's folder on the models volume, so two users can each have a
model with the same name. Requests name models by name only; resolve_model maps the name
to the caller's own model, and the inference workers load it from its owner's folder.
"""
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.open_source import OPEN_SOURCE_MODELS
from models.database import JobStatus, TrainingJob, UploadedModel
from services.inference_router import BASE_MODELS


@dataclass(frozen=True)
class ModelRef:
    """A model a request may run. Custom models carry their owner and version."""

    name: str
    owner_id: Optional[int] = None
    # Changes whenever the model's weights do (retrained or re-uploaded under the same name),
    # so workers never serve a cached copy of the old weights
    version: Optional[str] = None

    def worker_kwargs(self) -> dict:
        """Arguments telling an inference worker which custom model to load."""
        return {"owner_id": self.owner_id, "version": self.version}


def resolve_model(db: Session, user_id: int, model_name: str) -> ModelRef:
    """
    The model a user's request names: a shared model, or the user's own custom model.

    Raises:
        HTTPException: 404 if the name is neither a shared model nor one of the user's
            models (other users' models are treated as nonexistent)
    """
    if model_name in BASE_MODELS or model_name in OPEN_SOURCE_MODELS:
        return ModelRef(model_name)

    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.user_id == user_id,
            TrainingJob.model_name == model_name,
            TrainingJob.status == JobStatus.COMPLETED,
        )
        .order_by(TrainingJob.completed_at.desc())
        .first()
    )
    if job:
        return ModelRef(model_name, user_id, job.job_id)

    uploaded = (
        db.query(UploadedModel)
        .filter(UploadedModel.user_id == user_id, UploadedModel.name == model_name)
        .first()
    )
    if uploaded:
        return ModelRef(model_name, user_id, f"upload-{uploaded.created_at:%Y%m%d%H%M%S%f}")

    raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


def list_custom_model_names(db: Session, user_id: int) -> list[str]:
    """Names of the user's custom models (completed training jobs and uploads)."""
    trained = (
        db.query(TrainingJob.model_name)
        .filter(TrainingJob.user_id == user_id, TrainingJob.status == JobStatus.COMPLETED)
        .all()
    )
    uploaded = db.query(UploadedModel.name).filter(UploadedModel.user_id == user_id).all()
    return list(dict.fromkeys(name for (name,) in [*trained, *uploaded]))
