"""
Specs of the models a user can run, without loading them on a GPU.

Base and open-source models' specs are fixed. A trained model's come from its base config
and training settings, an uploaded model's from its file's header, read when it's uploaded
(see core.transformer.checkpoint), so they match what the inference workers build.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from core.open_source import OPEN_SOURCE_MODELS
from core.tokenizer import get_tokenizer
from core.transformer.checkpoint import (
    InvalidCheckpointError,
    config_from_checkpoint,
    count_parameters,
    read_safetensors_header,
)
from core.transformer.config import TransformerConfig, get_config
from models.database import JobStatus, TrainingJob, UploadedModel

logger = logging.getLogger(__name__)

# The base models' checkpoint configs (mirrored by STATIC_BASE_MODEL_INFO in the frontend)
BASE_MODEL_CONFIGS = {
    "nano": TransformerConfig(vocab_size=50257, d_model=192, num_layers=6, num_heads=6, d_ff=768, max_seq_len=512),
    "mini": TransformerConfig(vocab_size=50257, d_model=512, num_layers=8, num_heads=8, d_ff=2048, max_seq_len=512),
}


@dataclass(frozen=True)
class ModelSpecs:
    """
    A model's architecture. Fields are None only for an uploaded model whose file couldn't
    be read (its specs are filled in once it can).
    """

    name: str
    parameters: Optional[int]
    layers: Optional[int]
    heads: Optional[int]
    hidden_dim: Optional[int]
    vocab_size: Optional[int]
    # Context window (the most tokens an Ozera model attends over)
    max_seq_len: Optional[int]
    # Neurons per MLP layer (d_ff); unknown for uploaded models
    mlp_neurons: Optional[int] = None


def _specs_from_config(name: str, config: TransformerConfig, parameters: Optional[int] = None) -> ModelSpecs:
    return ModelSpecs(
        name=name,
        parameters=parameters if parameters is not None else config.count_parameters(),
        layers=config.num_layers,
        heads=config.num_heads,
        hidden_dim=config.d_model,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        mlp_neurons=config.d_ff,
    )


def trained_model_config(job: TrainingJob) -> TransformerConfig:
    """The config a completed training job's model was built with (see training_logic.py)."""
    base = get_config(job.model_config)
    return TransformerConfig(
        vocab_size=get_tokenizer().vocab_size,
        max_seq_len=job.seq_len,
        d_model=base.d_model,
        num_layers=base.num_layers,
        num_heads=base.num_heads,
        d_ff=base.d_ff,
        dropout_rate=base.dropout_rate,
    )


def uploaded_model_fields(metadata: dict, shapes: dict) -> dict:
    """
    UploadedModel columns for a safetensors file, from its header.

    Raises:
        InvalidCheckpointError: the file can't be built into an Ozera model
    """
    config = config_from_checkpoint(metadata, shapes)
    try:
        num_parameters = int(metadata["parameters"])
    except (KeyError, TypeError, ValueError):
        num_parameters = count_parameters(shapes)
    return {
        "num_parameters": num_parameters,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "hidden_dim": config.d_model,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
    }


async def _fill_uploaded_model_specs(db: Session, model: UploadedModel) -> None:
    """
    Fill in an uploaded model's missing specs from its file's header on the models volume.

    Models uploaded before the upload read their config this way can lack some. Failures
    are logged and leave the model as it was.
    """
    from services.modal_volumes import read_model_header_from_volume

    try:
        metadata, shapes = read_safetensors_header(await read_model_header_from_volume(model.user_id, model.name))
        fields = uploaded_model_fields(metadata, shapes)
    except (FileNotFoundError, InvalidCheckpointError) as e:
        logger.warning("Can't read the specs of uploaded model %r of user %s: %s", model.name, model.user_id, e)
        return
    except Exception:
        logger.exception("Failed to read uploaded model %r of user %s", model.name, model.user_id)
        return

    for column, value in fields.items():
        if getattr(model, column) is None:
            setattr(model, column, value)
    db.commit()


async def model_specs(db: Session, user_id: Optional[int], model_name: str) -> ModelSpecs:
    """
    Specs of a model the user can run: a shared model, or the user's own custom model.

    Resolves names like services.custom_models.resolve_model (a trained model before an
    uploaded one of the same name).

    Raises:
        FileNotFoundError: the name is neither a shared model nor one of the user's
    """
    if model_name in BASE_MODEL_CONFIGS:
        return _specs_from_config(model_name, BASE_MODEL_CONFIGS[model_name])

    if model_name in OPEN_SOURCE_MODELS:
        cfg = OPEN_SOURCE_MODELS[model_name]
        return ModelSpecs(
            name=model_name,
            parameters=cfg.parameters,
            layers=cfg.num_layers,
            heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            mlp_neurons=cfg.intermediate_dim,
        )

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
        return _specs_from_config(model_name, trained_model_config(job))

    uploaded = (
        db.query(UploadedModel)
        .filter(UploadedModel.user_id == user_id, UploadedModel.name == model_name)
        .first()
    )
    if uploaded is None:
        raise FileNotFoundError(f"Model '{model_name}' not found")

    columns = ("num_parameters", "num_layers", "num_heads", "hidden_dim", "vocab_size", "max_seq_len")
    if any(getattr(uploaded, column) is None for column in columns):
        await _fill_uploaded_model_specs(db, uploaded)

    return ModelSpecs(
        name=model_name,
        parameters=uploaded.num_parameters,
        layers=uploaded.num_layers,
        heads=uploaded.num_heads,
        hidden_dim=uploaded.hidden_dim,
        vocab_size=uploaded.vocab_size,
        max_seq_len=uploaded.max_seq_len,
    )
