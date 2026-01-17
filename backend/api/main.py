"""
FastAPI application for Ozera inference API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import ModelLoader, TextGenerator
from inference.activation_store import get_activation_store

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

# Global model loader
model_loader = ModelLoader(models_dir="models")
generators = {}
activation_store = get_activation_store()


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


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    Args:
        request: Generation request parameters

    Returns:
        Generated text and metadata
    """
    try:
        # Load model if not already loaded
        if request.model not in generators:
            model, config = model_loader.load_model(request.model, device='cpu')
            device = str(next(model.parameters()).device)
            generators[request.model] = TextGenerator(model, device=device, model_name=request.model)

        generator = generators[request.model]

        # Generate text
        result = generator.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            return_metadata=True
        )

        return GenerateResponse(
            text=result['text'],
            prompt=result['prompt'],
            model=request.model,
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
    # List available models
    return model_loader.list_available_models()


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model ('nano' or 'mini')

    Returns:
        Model information
    """
    try:
        model, config = model_loader.load_model(model_name)

        return ModelInfo(
            name=model_name,
            parameters=config.count_parameters(),
            layers=config.num_layers,
            heads=config.num_heads,
            hidden_dim=config.d_model,
            vocab_size=config.vocab_size
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text from a prompt with streaming (Server-Sent Events).

    Args:
        request: Generation request parameters

    Returns:
        Stream of generated tokens
    """
    try:
        # Load model if not already loaded
        if request.model not in generators:
            model, _ = model_loader.load_model(request.model, device='cpu')
            device = str(next(model.parameters()).device)
            generators[request.model] = TextGenerator(model, device=device, model_name=request.model)

        generator = generators[request.model]

        # Create streaming generator function
        def event_stream():
            try:
                # Send initial metadata
                data = json.dumps({'type': 'start', 'prompt': request.prompt}, ensure_ascii=False)
                yield f"data: {data}\n\n".encode('utf-8')

                # Stream tokens
                for token in generator.generate_stream(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p
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

    Args:
        request: Generation request parameters

    Returns:
        Generated text, metadata, and activation ID
    """
    try:
        # Load model if not already loaded
        if request.model not in generators:
            model, _ = model_loader.load_model(request.model, device='cpu')
            device = str(next(model.parameters()).device)
            generators[request.model] = TextGenerator(model, device=device, model_name=request.model)

        generator = generators[request.model]

        # Generate with activation capture
        result = generator.generate_with_activations(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

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
        token_ids = request.get('token_ids', [])
        model_name = request.get('model', 'nano')

        # Load model to get tokenizer
        if model_name not in generators:
            model, _ = model_loader.load_model(model_name, device='cpu')
            device = str(next(model.parameters()).device)
            generators[model_name] = TextGenerator(model, device=device, model_name=model_name)

        generator = generators[model_name]

        # Decode each token ID individually
        decoded = [generator.tokenizer.decode([tid]) for tid in token_ids]

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
