"""
Resolving a requested model name to a model the requesting user may run.

Base and open-source models are shared. A custom model (trained or uploaded) belongs to one
user and lives in that user's folder on the models volume, so two users can each have a
model with the same name. Requests name models by name only; resolve_model maps the name
to the caller's own model, and the inference workers load it from its owner's folder.

Users have at most one custom model, so a new one replaces the user's existing models.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.open_source import OPEN_SOURCE_MODELS
from models.database import JobStatus, TrainingJob, UploadedModel
from services.inference_router import BASE_MODELS
from services.modal_volumes import delete_model_from_volume


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


def delete_replaced_models(
    db: Session,
    user_id: int,
    existing_at: Optional[datetime] = None,
    new_job_id: Optional[str] = None,
) -> set[str]:
    """
    Delete the rows of a user's custom models that a new model replaces (users have at most
    one custom model). Nothing is committed.

    A new model only replaces the old ones once it exists (its upload is stored, or its
    training job completed), so a failed upload or training job leaves them in place.

    Args:
        existing_at: Only replace models that already existed then. A training job replaces
            the models the user had when submitting it, which they agreed to replace.
        new_job_id: The training job of the new model, which is kept

    Returns:
        Names of the replaced models. After committing, pass them to
        remove_replaced_model_files to delete their weights.
    """
    trained = db.query(TrainingJob).filter(
        TrainingJob.user_id == user_id,
        TrainingJob.status == JobStatus.COMPLETED,
    )
    uploaded = db.query(UploadedModel).filter(UploadedModel.user_id == user_id)
    if new_job_id is not None:
        trained = trained.filter(TrainingJob.job_id != new_job_id)
    if existing_at is not None:
        trained = trained.filter(
            or_(TrainingJob.completed_at.is_(None), TrainingJob.completed_at <= existing_at)
        )
        uploaded = uploaded.filter(UploadedModel.created_at <= existing_at)

    names = set()
    for job in trained.all():
        names.add(job.model_name)
        db.delete(job)
    for model in uploaded.all():
        names.add(model.name)
        db.delete(model)
    return names


async def remove_replaced_model_files(user_id: int, replaced_names: Iterable[str], new_name: str) -> None:
    """
    Delete replaced models' weights from the models volume, once their rows are deleted.

    A replaced model with the new model's name shared its folder, which now holds the new
    model's weights, so that folder stays.
    """
    for name in set(replaced_names) - {new_name}:
        if not await delete_model_from_volume(user_id, name):
            print(f"Failed to delete replaced model {name!r} of user {user_id} from the models volume")
