"""
FastAPI application for Ozera inference API.

Supports both local and Modal cloud inference based on INFERENCE_MODE env var.
"""

# Load environment variables BEFORE any other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import ModelLoader
from inference.activation_store import get_activation_store
from services.inference_router import get_inference_router
from api.datasets import router as datasets_router
from api.training import router as training_router
from api.auth import router as auth_router
from api.credits import router as credits_router
from api.payments import router as payments_router
from api.webhooks import router as webhooks_router

app = FastAPI(
    title="Ozera API",
    description="Text generation API for Ozera language models",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader - use absolute path relative to backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
model_loader = ModelLoader(models_dir=MODELS_DIR)
activation_store = get_activation_store()

# Inference router for local/Modal routing
inference_router = get_inference_router(models_dir=MODELS_DIR)

# Register routers for authentication, datasets, training, credits, payments, and webhooks
app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(credits_router)
app.include_router(payments_router)
app.include_router(webhooks_router)


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
    # Health check endpoint.
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
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    Routes to local or Modal inference based on INFERENCE_MODE env var.

    Args:
        request: Generation request parameters

    Returns:
        Generated text and metadata
    """
    try:
        result = await inference_router.generate(
            model_id=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        return GenerateResponse(
            text=result['text'],
            prompt=result['prompt'],
            model=result.get('model', request.model),
            prompt_tokens=result['prompt_tokens'],
            generated_tokens=result['generated_tokens'],
            total_tokens=result['total_tokens'],
            temperature=result['temperature'],
            top_k=result['top_k'],
            top_p=result['top_p']
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare model: {str(e)}")


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text from a prompt with streaming (Server-Sent Events).

    Routes to local or Modal inference based on INFERENCE_MODE env var.

    Args:
        request: Generation request parameters

    Returns:
        Stream of generated tokens
    """
    try:
        # Create streaming generator function
        async def event_stream():
            try:
                # Send initial metadata
                data = json.dumps({'type': 'start', 'prompt': request.prompt}, ensure_ascii=False)
                yield f"data: {data}\n\n".encode('utf-8')

                # Stream tokens using the inference router
                async for token in inference_router.generate_stream(
                    model_id=request.model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                ):
                    data = json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)
                    yield f"data: {data}\n\n".encode('utf-8')

                # Send completion
                data = json.dumps({'type': 'done'}, ensure_ascii=False)
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
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate/with-activations")
async def generate_with_activations(request: GenerateRequest):
    """
    Generate text and capture activations for visualization.

    Note: For Modal mode, activations are returned inline. For local mode,
    an activation_id is returned for later retrieval.

    Args:
        request: Generation request parameters

    Returns:
        Generated text, metadata, and activation ID or inline activations
    """
    try:
        result = await inference_router.generate_with_activations(
            model_id=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        # If Modal mode returned inline activations, store them locally
        if 'activations' in result and 'activation_id' not in result:
            activation_id = activation_store.store_activations(
                activations=result['activations'],
                tokens=result.get('tokens', []),
                prompt=result['prompt'],
                model_name=result.get('model', request.model),
                metadata={
                    'temperature': result['temperature'],
                    'top_k': result['top_k'],
                    'top_p': result['top_p'],
                    'prompt_tokens': result['prompt_tokens'],
                    'generated_tokens': result['generated_tokens'],
                    'total_tokens': result['total_tokens'],
                    'generated_text': result['text'],
                    'decoded_tokens': result.get('decoded_tokens', []),
                }
            )
            result['activation_id'] = activation_id
            del result['activations']  # Don't send large activations to frontend

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/activations/{activation_id}")
async def get_activations(activation_id: str):
    """
    Retrieve stored activations by ID.

    Args:
        activation_id: UUID of stored activations

    Returns:
        Complete activation data
    """
    activations = activation_store.get_activations(activation_id)

    if activations is None:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return activations


@app.get("/activations/{activation_id}/summary")
async def get_activation_summary(activation_id: str):
    """
    Get metadata summary for activations without full tensors.

    Args:
        activation_id: UUID of stored activations

    Returns:
        Activation metadata summary
    """
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding tokens: {str(e)}")


@app.get("/activations")
async def list_activations():
    """
    List all stored activations (summaries only).

    Returns:
        List of activation summaries
    """
    return activation_store.list_activations()


@app.delete("/activations/{activation_id}")
async def delete_activations(activation_id: str):
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
