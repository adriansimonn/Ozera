"""
FastAPI application for Ozera inference API.

Supports both local and Modal cloud inference based on INFERENCE_MODE env var.
"""

# Load environment variables BEFORE any other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional
from slowapi.errors import RateLimitExceeded
import logging
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

from inference import ModelLoader
from inference.activation_store import get_activation_store
from services.inference_router import get_inference_router
from core.open_source import OPEN_SOURCE_MODELS
from services.credit_service import (
    calculate_inference_cost,
    charge_inference,
    check_sufficient_balance,
    get_credit_balance,
    InsufficientBalanceError,
)
from api.datasets import router as datasets_router
from api.training import router as training_router
from api.auth import router as auth_router
from api.credits import router as credits_router
from api.payments import router as payments_router
from api.webhooks import router as webhooks_router
from api.open_source import router as open_source_router
from api.patching import router as patching_router
from api.analysis import router as analysis_router
from api.export import router as export_router
from api.sae import router as sae_router
from api.settings import router as settings_router
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter, rate_limit_exceeded_handler
from models.database import User
from db import get_db, SessionLocal
from api.error_utils import safe_detail

_app_env = os.getenv("APP_ENV", "development")

# Validate critical environment variables in production
if _app_env == "production":
    _required_env_vars = ["MODAL_WEBHOOK_SECRET", "STRIPE_SECRET_KEY", "SUPABASE_URL"]
    _missing = [v for v in _required_env_vars if not os.getenv(v)]
    if _missing:
        raise RuntimeError(f"Missing required environment variables for production: {', '.join(_missing)}")

app = FastAPI(
    title="Ozera API",
    description="Text generation API for Ozera language models",
    version="1.0.0"
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# CORS middleware for frontend
_cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
] if _app_env == "development" else [
    "https://ozera.app",
    "https://www.ozera.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

# Global model loader - use absolute path relative to backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
model_loader = ModelLoader(models_dir=MODELS_DIR)
activation_store = get_activation_store()

# Inference router for local/Modal routing
inference_router = get_inference_router(models_dir=MODELS_DIR)

# Register routers for authentication, datasets, training, credits, payments, webhooks, and open-source models
app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(credits_router)
app.include_router(payments_router)
app.include_router(webhooks_router)
app.include_router(open_source_router)
app.include_router(patching_router)
app.include_router(analysis_router)
app.include_router(export_router)
app.include_router(sae_router)
app.include_router(settings_router)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to start generation")
    model: str = Field(default="nano", description="Model to use ('nano' or 'mini')")
    max_tokens: int = Field(default=200, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=40, ge=1, le=100, description="Top-k sampling")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    model: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]


class ModelInfo(BaseModel):
    name: str
    parameters: int
    layers: int
    heads: int
    hidden_dim: int
    vocab_size: int


class HealthResponse(BaseModel):
    status: str
    available_models: list[str]


@app.get("/", response_model=dict)
async def root():
    # Root endpoint
    return {
        "name": "Ozera API",
        "version": "1.0.0",
        "description": "Text generation API for Ozera language models"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    # Health check endpoint — verifies DB connectivity.
    from sqlalchemy import text
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception:
        logger.exception("Health check: database unreachable")
        raise HTTPException(status_code=503, detail="Database unreachable")

    return {
        "status": "healthy",
        "available_models": model_loader.list_available_models()
    }


@app.get("/inference-mode")
async def get_inference_mode():
    """
    Get current inference mode configuration.

    Returns:
        Current inference mode ('local' or 'modal')
    """
    return {
        "mode": inference_router.get_inference_mode(),
        "is_modal": inference_router.is_modal_mode(),
    }


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate text from a prompt.

    Routes to local or Modal inference based on INFERENCE_MODE env var.
    Requires authentication and charges user credits.

    Args:
        request: Generation request parameters
        current_user: Authenticated user
        db: Database session

    Returns:
        Generated text and metadata
    """
    # Check if user has sufficient balance
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
        result = await inference_router.generate(
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
            prompt_tokens=result['prompt_tokens'],
            generated_tokens=result['generated_tokens'],
            model_name=body.model,
        )

        return GenerateResponse(
            text=result['text'],
            prompt=result['prompt'],
            model=result.get('model', body.model),
            prompt_tokens=result['prompt_tokens'],
            generated_tokens=result['generated_tokens'],
            total_tokens=result['total_tokens'],
            temperature=result['temperature'],
            top_k=result['top_k'],
            top_p=result['top_p']
        )

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid request"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Resource not found"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models", response_model=list[str])
async def list_models():
    # List available models (including custom models from database)
    return model_loader.list_available_models(include_remote=True)


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Routes to local or Modal inference based on INFERENCE_MODE env var.

    Args:
        model_name: Name of the model ('nano', 'mini', or custom model name)

    Returns:
        Model information
    """
    try:
        info = await inference_router.get_model_info(model_name)

        return ModelInfo(
            name=info["name"],
            parameters=info["parameters"],
            layers=info["layers"],
            heads=info["heads"],
            hidden_dim=info["hidden_dim"],
            vocab_size=info["vocab_size"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid request"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Resource not found"))
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/models/{model_name}/prepare")
async def prepare_model(model_name: str):
    """
    Pre-load a model to ensure it's available for inference.

    For local mode: Downloads from Modal volume if needed and caches locally.
    For Modal mode: Warms up the Modal container with the model.

    This is useful to trigger model loading before the first generation request.

    Args:
        model_name: Name of the model

    Returns:
        Status and model info
    """
    try:
        # Warmup using the inference router
        success = await inference_router.warmup_model(model_name)

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to prepare model: {model_name}")

        # Get model info
        model_info = await inference_router.get_model_info(model_name)

        return {
            "status": "ready",
            "model": model_name,
            "parameters": model_info.get("parameters"),
            "layers": model_info.get("layers"),
            "inference_mode": inference_router.get_inference_mode(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid request"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Resource not found"))
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Failed to prepare model")


@app.post("/generate/stream")
@limiter.limit("10/minute")
async def generate_stream(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate text from a prompt with streaming (Server-Sent Events).

    Routes to local or Modal inference based on INFERENCE_MODE env var.
    Requires authentication and charges user credits.
    """
    # Check if user has sufficient balance (estimate based on max_tokens, with model-based pricing)
    estimated_cost = calculate_inference_cost(
        prompt_tokens=len(body.prompt.split()) * 2,  # Rough estimate
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

    try:
        async def event_stream():
            token_count = 0
            try:
                data = json.dumps({'type': 'start', 'prompt': body.prompt}, ensure_ascii=False)
                yield f"data: {data}\n\n".encode('utf-8')

                async for token in inference_router.generate_stream(
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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid request"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Resource not found"))
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/generate/with-activations")
@limiter.limit("10/minute")
async def generate_with_activations(
    request: Request,
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate text and capture activations for visualization.

    Note: For Modal mode, activations are returned inline. For local mode,
    an activation_id is returned for later retrieval.
    Requires authentication and charges user credits.
    """
    # Check if user has sufficient balance (estimate based on max_tokens, with model-based pricing)
    estimated_cost = calculate_inference_cost(
        prompt_tokens=len(body.prompt.split()) * 2,  # Rough estimate
        generated_tokens=body.max_tokens,
        model_id=body.model,
    )
    if not check_sufficient_balance(db, current_user.id, estimated_cost):
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please add more credits to continue."
        )

    try:
        result = await inference_router.generate_with_activations(
            model_id=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
        )

        # If Modal mode returned inline activations, store them locally
        if 'activations' in result and 'activation_id' not in result:
            activation_id = activation_store.store_activations(
                activations=result['activations'],
                tokens=result.get('tokens', []),
                prompt=result['prompt'],
                model_name=result.get('model', body.model),
                metadata={
                    'temperature': result['temperature'],
                    'top_k': result['top_k'],
                    'top_p': result['top_p'],
                    'prompt_tokens': result['prompt_tokens'],
                    'generated_tokens': result['generated_tokens'],
                    'total_tokens': result['total_tokens'],
                    'generated_text': result['text'],
                    'decoded_tokens': result.get('decoded_tokens', []),
                    'model_family': OPEN_SOURCE_MODELS[body.model].family.value if body.model in OPEN_SOURCE_MODELS else 'ozera',
                }
            )
            result['activation_id'] = activation_id
            del result['activations']  # Don't send large activations to frontend

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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid request"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=safe_detail(e, "Resource not found"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/activations/{activation_id}")
async def get_activations(
    activation_id: str,
    current_user: User = Depends(get_current_user),
):
    """Retrieve stored activations by ID. Requires authentication."""
    activations = activation_store.get_activations(activation_id)

    if activations is None:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return activations


@app.get("/activations/{activation_id}/summary")
async def get_activation_summary(
    activation_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get metadata summary for activations without full tensors. Requires authentication."""
    summary = activation_store.get_activation_summary(activation_id)

    if summary is None:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return summary


@app.post("/decode-tokens")
async def decode_tokens(request: dict):
    """
    Decode token IDs to their string representations.

    Args:
        request: Dictionary with 'token_ids' list and optional 'model' string

    Returns:
        List of decoded token strings
    """
    try:
        from core.tokenizer import get_tokenizer

        token_ids = request.get('token_ids', [])

        # Use shared tokenizer (all models use the same BPE tokenizer)
        tokenizer = get_tokenizer()

        # Decode each token ID individually
        decoded = [tokenizer.decode([tid]) for tid in token_ids]

        return {'decoded_tokens': decoded}

    except Exception:
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail="Error decoding tokens")


@app.get("/activations")
async def list_activations(
    current_user: User = Depends(get_current_user),
):
    """List all stored activations (summaries only). Requires authentication."""
    return activation_store.list_activations()


@app.get("/activations/{activation_id}/layer/{layer_idx}")
async def get_layer_activations(
    activation_id: str,
    layer_idx: int,
    current_user: User = Depends(get_current_user),
):
    """
    Get activations for a specific layer (lazy loading).

    This endpoint supports lazy loading of activation data by allowing
    the frontend to fetch individual layers on-demand instead of loading
    all layers at once.

    Args:
        activation_id: UUID of stored activations
        layer_idx: Index of the layer to retrieve (0-indexed)

    Returns:
        Layer activation data including attention weights, hidden states, etc.
    """
    result = activation_store.get_layer_activations(activation_id, layer_idx)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Layer {layer_idx} not found for activation {activation_id}"
        )

    return result


@app.get("/activations/{activation_id}/tensor/{tensor_name}")
async def get_tensor_activation(
    activation_id: str,
    tensor_name: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get a specific top-level tensor activation (lazy loading).

    This endpoint supports lazy loading of activation data by allowing
    the frontend to fetch specific tensors on-demand.

    Args:
        activation_id: UUID of stored activations
        tensor_name: Name of tensor to retrieve. Valid options:
            - token_embeddings
            - positional_embeddings
            - combined_embeddings
            - final_layer_norm
            - logits

    Returns:
        Tensor data with values, shape, and statistics
    """
    valid_tensors = ['token_embeddings', 'positional_embeddings', 'combined_embeddings', 'final_layer_norm', 'logits']
    if tensor_name not in valid_tensors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tensor name: {tensor_name}. Valid options: {valid_tensors}"
        )

    result = activation_store.get_tensor_activation(activation_id, tensor_name)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tensor {tensor_name} not found for activation {activation_id}"
        )

    return result


@app.delete("/activations/{activation_id}")
async def delete_activations(
    activation_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Delete stored activations.

    Args:
        activation_id: UUID of stored activations

    Returns:
        Success status
    """
    success = activation_store.delete_activations(activation_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return {"status": "deleted", "activation_id": activation_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
