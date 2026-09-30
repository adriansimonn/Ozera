"""
API endpoints for open-source model management and inference.

Provides endpoints for:
- Listing available open-source models
- Triggering model downloads to Modal cache
- Checking model cache status
- Managing cached models
"""

import logging

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional
import json

logger = logging.getLogger(__name__)

from core.open_source import OPEN_SOURCE_MODELS, ModelFamily
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import TransactionType, User
from db import get_db
from services.credit_service import (
    calculate_inference_cost,
    charge_flat,
    charge_inference,
    check_sufficient_balance,
    min_gpu_request_charge,
    InsufficientBalanceError,
)
from inference.activation_store import get_activation_store

router = APIRouter(prefix="/open-source", tags=["Open Source Models"])

activation_store = get_activation_store()


# Request/Response Models

class OpenSourceModelInfo(BaseModel):
    """Information about an open-source model."""
    id: str
    hf_id: str
    display_name: str
    family: str
    parameters: int
    layers: int
    heads: int
    kv_heads: int
    hidden_dim: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    gpu_tier: str


class ModelCacheStatus(BaseModel):
    """Cache status for a model."""
    status: str  # "ready", "not_cached", "incomplete", "error"
    hf_id: str
    path: Optional[str] = None
    has_model: Optional[bool] = None
    has_tokenizer: Optional[bool] = None
    total_size_mb: Optional[float] = None
    file_count: Optional[int] = None
    error: Optional[str] = None


class DownloadResponse(BaseModel):
    """Response from model download request."""
    status: str  # "downloading", "cached", "error"
    hf_id: str
    path: Optional[str] = None
    error: Optional[str] = None


class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Text prompt to start generation")
    model: str = Field(..., description="Open-source model ID (e.g., 'smollm-135m')")
    max_tokens: int = Field(default=200, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=40, ge=1, le=100, description="Top-k sampling")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")


# Helper Functions

def _get_gpu_tier_for_model(model_id: str) -> str:
    """Get the GPU tier required for a model."""
    if model_id not in OPEN_SOURCE_MODELS:
        raise ValueError(f"Unknown open-source model: {model_id}")
    return OPEN_SOURCE_MODELS[model_id].gpu_tier


def _get_inference_worker(model_id: str):
    """Get the appropriate Modal inference worker for a model."""
    from services.modal_inference import get_inference_worker
    return get_inference_worker(_get_gpu_tier_for_model(model_id))


def _get_download_functions():
    """Get Modal download functions."""
    import modal
    download_model = modal.Function.from_name("ozera-hf-download", "download_model")
    check_model_status = modal.Function.from_name("ozera-hf-download", "check_model_status")
    list_cached_models = modal.Function.from_name("ozera-hf-download", "list_cached_models")
    delete_cached_model = modal.Function.from_name("ozera-hf-download", "delete_cached_model")
    return download_model, check_model_status, list_cached_models, delete_cached_model


# Model Listing Endpoints

@router.get("/models", response_model=list[OpenSourceModelInfo])
async def list_models():
    """
    List all available open-source models.

    Returns metadata for all supported open-source models (SmolLM, Gemma, Qwen).
    """
    return [
        OpenSourceModelInfo(
            id=cfg.model_id,
            hf_id=cfg.hf_id,
            display_name=cfg.display_name,
            family=cfg.family.value,
            parameters=cfg.parameters,
            layers=cfg.num_layers,
            heads=cfg.num_heads,
            kv_heads=cfg.num_kv_heads,
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            gpu_tier=cfg.gpu_tier,
        )
        for cfg in OPEN_SOURCE_MODELS.values()
    ]


@router.get("/models/{model_id}", response_model=OpenSourceModelInfo)
async def get_model_info(model_id: str):
    """
    Get detailed info for a specific open-source model.

    Args:
        model_id: Internal model ID (e.g., "smollm-135m", "gemma-3-1b")
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_id}. Available: {list(OPEN_SOURCE_MODELS.keys())}"
        )

    cfg = OPEN_SOURCE_MODELS[model_id]
    return OpenSourceModelInfo(
        id=cfg.model_id,
        hf_id=cfg.hf_id,
        display_name=cfg.display_name,
        family=cfg.family.value,
        parameters=cfg.parameters,
        layers=cfg.num_layers,
        heads=cfg.num_heads,
        kv_heads=cfg.num_kv_heads,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        gpu_tier=cfg.gpu_tier,
    )


@router.get("/families")
async def list_families():
    """
    List model families with their models.

    Returns models grouped by family (smollm, gemma, qwen).
    """
    families = {}
    for model_id, cfg in OPEN_SOURCE_MODELS.items():
        family = cfg.family.value
        if family not in families:
            families[family] = {
                "name": family,
                "display_name": family.title(),
                "models": []
            }
        families[family]["models"].append({
            "id": model_id,
            "display_name": cfg.display_name,
            "parameters": cfg.parameters,
            "gpu_tier": cfg.gpu_tier,
        })

    return list(families.values())


# Cache Management Endpoints

@router.get("/cache", response_model=list[dict])
async def list_cached_models(current_user: User = Depends(get_current_user)):
    """
    List all models cached in the HuggingFace volume.

    Returns list of cached models with their sizes.
    Requires authentication.
    """
    try:
        _, _, list_cached_fn, _ = _get_download_functions()
        cached = await list_cached_fn.remote.aio()
        return cached
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Failed to list cached models")


@router.get("/cache/{model_id}/status", response_model=ModelCacheStatus)
async def get_cache_status(model_id: str, current_user: User = Depends(get_current_user)):
    """
    Check if a model is cached and ready for inference.
    Requires authentication.

    Args:
        model_id: Internal model ID (e.g., "smollm-135m")
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    cfg = OPEN_SOURCE_MODELS[model_id]

    try:
        _, check_status_fn, _, _ = _get_download_functions()
        status = await check_status_fn.remote.aio(cfg.hf_id)
        return ModelCacheStatus(**status)
    except Exception as e:
        return ModelCacheStatus(
            status="error",
            hf_id=cfg.hf_id,
            error="Failed to check cache status"
        )


@router.post("/cache/{model_id}/download", response_model=DownloadResponse)
async def trigger_download(model_id: str, force: bool = False, current_user: User = Depends(get_current_user)):
    """
    Trigger download of a model to the Modal HuggingFace cache.
    Requires authentication.

    Args:
        model_id: Internal model ID (e.g., "smollm-135m")
        force: Force re-download even if already cached
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    cfg = OPEN_SOURCE_MODELS[model_id]

    try:
        download_fn, _, _, _ = _get_download_functions()
        result = await download_fn.remote.aio(cfg.hf_id, force=force)
        return DownloadResponse(**result)
    except Exception as e:
        return DownloadResponse(
            status="error",
            hf_id=cfg.hf_id,
            error="Failed to download model"
        )


@router.delete("/cache/{model_id}")
async def delete_cached_model(model_id: str, current_user: User = Depends(get_current_user)):
    """
    Delete a cached model from the HuggingFace volume.
    Requires authentication.

    Args:
        model_id: Internal model ID (e.g., "smollm-135m")
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    cfg = OPEN_SOURCE_MODELS[model_id]

    try:
        _, _, _, delete_fn = _get_download_functions()
        result = await delete_fn.remote.aio(cfg.hf_id)

        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Model not cached: {model_id}")

        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Failed to delete model")


# Warmup Endpoint

@router.post("/models/{model_id}/warmup")
async def warmup_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Pre-load a model into GPU memory for faster inference.
    Requires authentication and charges user credits (a GPU request's minimum charge).

    This triggers the model to be loaded into the inference worker's cache.
    Subsequent generation requests will be faster.

    Args:
        model_id: Internal model ID (e.g., "smollm-135m")
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    cfg = OPEN_SOURCE_MODELS[model_id]
    cost = min_gpu_request_charge(model_id)
    if not check_sufficient_balance(db, current_user.id, cost):
        raise HTTPException(status_code=402, detail="Insufficient credits.")

    try:
        worker = _get_inference_worker(model_id)
        success = await worker().warmup.remote.aio(model_id)

        if success:
            charge_flat(db, current_user.id, cost, TransactionType.INFERENCE_CHARGE,
                        f"Model warmup ({model_id})")
            return {
                "status": "ready",
                "model": model_id,
                "display_name": cfg.display_name,
                "parameters": cfg.parameters,
                "gpu_tier": cfg.gpu_tier,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to warmup model. Ensure it's downloaded first."
            )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Warmup failed")


# Generation Endpoints

@router.post("/generate")
@limiter.limit("10/minute")
async def generate(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate text using an open-source model.

    Requires authentication and charges user credits.
    """
    if body.model not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")

    # Check credits (with model-size-based pricing)
    estimated_cost = calculate_inference_cost(
        prompt_tokens=len(body.prompt.split()) * 2,
        generated_tokens=body.max_tokens,
        model_id=body.model,
    )
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please add more credits to continue."
        )

    try:
        worker = _get_inference_worker(body.model)
        result = await worker().generate.remote.aio(
            model_id=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
        )

        # Charge user — fail if charge fails
        charge_inference(
            db=db,
            user_id=current_user.id,
            prompt_tokens=result.get('prompt_tokens', 0),
            generated_tokens=result.get('generated_tokens', 0),
            model_name=body.model,
        )
        result['charged'] = True

        return result

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Generation failed")


@router.post("/generate/stream")
@limiter.limit("10/minute")
async def generate_stream(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Stream text generation using an open-source model.

    Returns Server-Sent Events with generated tokens.
    Requires authentication and charges user credits.
    """
    if body.model not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")

    # Check credits (with model-size-based pricing)
    estimated_cost = calculate_inference_cost(
        prompt_tokens=len(body.prompt.split()) * 2,
        generated_tokens=body.max_tokens,
        model_id=body.model,
    )
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please add more credits to continue."
        )

    # Charge upfront based on estimated cost before streaming begins
    prompt_token_estimate = len(body.prompt.split()) * 2
    try:
        charge_inference(
            db=db,
            user_id=current_user.id,
            prompt_tokens=prompt_token_estimate,
            generated_tokens=body.max_tokens,
            model_name=body.model,
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except Exception:
        logger.exception("Failed to charge credits before streaming")
        raise HTTPException(status_code=500, detail="Failed to reserve credits for generation")

    async def event_stream():
        token_count = 0

        try:
            data = json.dumps({'type': 'start', 'prompt': body.prompt}, ensure_ascii=False)
            yield f"data: {data}\n\n".encode('utf-8')

            worker = _get_inference_worker(body.model)

            async for token in worker().generate_stream.remote_gen.aio(
                model_id=body.model,
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
            ):
                token_count += 1
                data = json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)
                yield f"data: {data}\n\n".encode('utf-8')

            done_payload = {
                'type': 'done',
                'token_count': token_count,
                'charged': True,
            }
            data = json.dumps(done_payload, ensure_ascii=False)
            yield f"data: {data}\n\n".encode('utf-8')

        except Exception as e:
            data = json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)
            yield f"data: {data}\n\n".encode('utf-8')

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )


@router.post("/generate/with-activations")
@limiter.limit("10/minute")
async def generate_with_activations(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate text and capture activations for interpretability visualization.

    Returns an activation_id that can be used to retrieve full activation data.
    Requires authentication and charges user credits.
    """
    if body.model not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}")

    # Check credits (with model-size-based pricing)
    estimated_cost = calculate_inference_cost(
        prompt_tokens=len(body.prompt.split()) * 2,
        generated_tokens=body.max_tokens,
        model_id=body.model,
    )
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please add more credits to continue."
        )

    try:
        worker = _get_inference_worker(body.model)
        result = await worker().generate_with_activations.remote.aio(
            model_id=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
        )

        # Store activations and return ID
        if 'activations' in result:
            activation_id = activation_store.store_activations(
                user_id=current_user.id,
                activations=result['activations'],
                tokens=result.get('tokens', []),
                prompt=result['prompt'],
                model_name=result.get('model', body.model),
                metadata={
                    'temperature': result.get('temperature'),
                    'top_k': result.get('top_k'),
                    'top_p': result.get('top_p'),
                    'prompt_tokens': result.get('prompt_tokens'),
                    'generated_tokens': result.get('generated_tokens'),
                    'total_tokens': result.get('total_tokens'),
                    'generated_text': result.get('text'),
                    'decoded_tokens': result.get('decoded_tokens', []),
                    'model_family': OPEN_SOURCE_MODELS[body.model].family.value,
                }
            )
            result['activation_id'] = activation_id
            del result['activations']  # Don't send large activations inline

        # Charge user — fail if charge fails
        charge_inference(
            db=db,
            user_id=current_user.id,
            prompt_tokens=result.get('prompt_tokens', 0),
            generated_tokens=result.get('generated_tokens', 0),
            model_name=body.model,
        )
        result['charged'] = True

        return result

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Generation failed")


class DecodeTokensRequest(BaseModel):
    """Request for decoding token IDs."""
    model: str = Field(..., description="Model ID to use for decoding")
    token_ids: list[int] = Field(..., description="Token IDs to decode")


@router.post("/decode-tokens")
async def decode_tokens(
    request: DecodeTokensRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Decode token IDs to their string representations using the model's tokenizer.

    This endpoint uses the correct tokenizer for the specified model, ensuring
    that token IDs are decoded accurately for visualization purposes. The tokenizer
    lives on the GPU worker, so this requires authentication and charges a GPU
    request's minimum.
    """
    try:
        if request.model not in OPEN_SOURCE_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        cost = min_gpu_request_charge(request.model)
        if not check_sufficient_balance(db, current_user.id, cost):
            raise HTTPException(status_code=402, detail="Insufficient credits.")

        worker = _get_inference_worker(request.model)
        decoded = await worker().decode_tokens.remote.aio(request.model, request.token_ids)

        charge_flat(db, current_user.id, cost, TransactionType.INFERENCE_CHARGE,
                    f"Token decoding ({request.model}): {len(request.token_ids)} tokens")

        return {"decoded_tokens": decoded}

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Token decoding failed")
