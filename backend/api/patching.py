"""
API endpoints for activation patching experiments.

Provides endpoints for:
- Capturing source activations from a prompt
- Running patching experiments (baseline vs patched generation)
- Managing captured activation storage
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.open_source import OPEN_SOURCE_MODELS, OpenSourceModelLoader, get_loader_for_model
from core.patching import get_patching_engine, PatchConfig
from api.schemas.patching import (
    CaptureActivationsRequest,
    CaptureActivationsResponse,
    CapturedActivationSummary,
    CapturedActivationDetail,
    RunPatchingRequest,
    RunPatchingWithCapturedRequest,
    PatchingResult,
    PatchSpec,
    EffectSummary,
    ChangedToken,
)
from middleware.auth_middleware import get_optional_current_user
from models.database import User
from db import get_db
from services.credit_service import (
    estimate_patching_cost,
    charge_patching,
    check_sufficient_balance,
)

router = APIRouter(prefix="/patching", tags=["Activation Patching"])

# Cache for model loaders (avoid reloading models repeatedly)
_model_loaders: dict = {}

# Base Ozera models
BASE_MODELS = {"nano", "mini"}


def _get_model_type(model_id: str) -> str:
    """Determine if a model is Ozera or open-source."""
    if model_id in BASE_MODELS:
        return "ozera"
    if model_id in OPEN_SOURCE_MODELS:
        return "open_source"
    # Check for custom trained models (assume ozera-based)
    return "ozera"


def _get_open_source_loader(model_id: str) -> OpenSourceModelLoader:
    """Get or create an open-source model loader."""
    cache_key = f"os_{model_id}"
    if cache_key not in _model_loaders:
        if model_id not in OPEN_SOURCE_MODELS:
            raise ValueError(f"Unknown open-source model: {model_id}")
        loader = get_loader_for_model(model_id)
        # Load the model (models are loaded from Modal inference server)
        # For now, we'll initialize but the actual load happens on first use
        _model_loaders[cache_key] = loader
    return _model_loaders[cache_key]


def _get_inference_worker(model_id: str):
    """Get the appropriate Modal inference worker for a model."""
    import modal

    # Ozera base models use T4 GPU
    if model_id in BASE_MODELS:
        return modal.Cls.from_name("ozera-inference", "InferenceWorkerT4")

    # Open-source models use tier specified in config
    if model_id in OPEN_SOURCE_MODELS:
        config = OPEN_SOURCE_MODELS[model_id]
        gpu_tier = config.gpu_tier
        worker_class = "InferenceWorkerT4" if gpu_tier == "t4" else "InferenceWorkerA10G"
        return modal.Cls.from_name("ozera-inference", worker_class)

    raise ValueError(f"Unknown model: {model_id}")


def _get_ozera_generator(model_id: str):
    """Get or create an Ozera model text generator."""
    cache_key = f"ozera_{model_id}"
    if cache_key not in _model_loaders:
        from inference.model_loader import ModelLoader
        from inference.text_generator import TextGenerator

        # Get models directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(backend_dir, "models")

        loader = ModelLoader(models_dir=models_dir)
        model, config = loader.load_model(model_id)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _model_loaders[cache_key] = TextGenerator(
            model=model,
            device=device,
            model_name=model_id,
        )
    return _model_loaders[cache_key]


def _get_ozera_tokenizer():
    """Get the Ozera tokenizer."""
    from core.tokenizer import get_tokenizer
    return get_tokenizer()


def _patch_spec_to_config(spec: PatchSpec) -> PatchConfig:
    """Convert API PatchSpec to engine PatchConfig."""
    return PatchConfig(
        layer=spec.layer,
        patch_type=spec.patch_type,
        positions=spec.positions,
        heads=spec.heads,
        neurons=spec.neurons,
        blend_factor=spec.blend_factor,
        intervention_type=spec.intervention_type,
    )


# Activation capture endpoints

@router.post("/capture", response_model=CaptureActivationsResponse)
async def capture_activations(request: CaptureActivationsRequest):
    """
    Capture activations from a source prompt.

    This captures all layer activations from a forward pass through the model,
    which can later be used for patching experiments.

    Args:
        request: Contains prompt and model ID

    Returns:
        Activation ID and metadata for the captured activations
    """
    import torch
    from core.patching import CapturedActivations

    model_type = _get_model_type(request.model)
    engine = get_patching_engine()

    try:
        # Open-source models and Ozera base models run on Modal GPU
        if model_type == "open_source" or request.model in BASE_MODELS:
            worker = _get_inference_worker(request.model)

            # Capture activations on Modal
            result = worker().capture_activations.remote(
                model_id=request.model,
                prompt=request.prompt,
            )

            # Reconstruct activations locally for caching
            reconstructed_activations = {}
            for key, value in result['activations'].items():
                if isinstance(value, list):
                    reconstructed_activations[key] = torch.tensor(value)
                else:
                    reconstructed_activations[key] = value

            # Create and cache CapturedActivations object locally
            captured = CapturedActivations(
                id=result['id'],
                prompt=result['prompt'],
                tokens=result['tokens'],
                decoded_tokens=result['decoded_tokens'],
                activations=reconstructed_activations,
                model_type=result['model_type'],
                model_id=result['model_id'],
                num_layers=result['num_layers'],
            )
            engine._captured_activations[captured.id] = captured

            return CaptureActivationsResponse(
                activation_id=captured.id,
                prompt=captured.prompt,
                model_type=captured.model_type,
                model_id=captured.model_id,
                num_tokens=len(captured.tokens),
                num_layers=captured.num_layers,
                decoded_tokens=captured.decoded_tokens,
            )
        else:
            # Custom trained Ozera models run locally
            generator = _get_ozera_generator(request.model)
            tokenizer = _get_ozera_tokenizer()
            captured = engine.capture_source_activations(
                prompt=request.prompt,
                model_loader=generator,
                model_type="ozera",
                model_id=request.model,
                tokenizer=tokenizer,
            )

            return CaptureActivationsResponse(
                activation_id=captured.id,
                prompt=captured.prompt,
                model_type=captured.model_type,
                model_id=captured.model_id,
                num_tokens=len(captured.tokens),
                num_layers=captured.num_layers,
                decoded_tokens=captured.decoded_tokens,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture activations: {str(e)}")


@router.get("/activations", response_model=list[CapturedActivationSummary])
async def list_captured_activations():
    """
    List all captured activations.

    Returns summaries of all captured activation sets currently stored in memory.
    """
    engine = get_patching_engine()
    activations = engine.list_captured_activations()

    return [
        CapturedActivationSummary(
            id=act['id'],
            prompt=act['prompt'],
            model_type=act['model_type'],
            model_id=act['model_id'],
            num_tokens=act['num_tokens'],
            num_layers=act['num_layers'],
        )
        for act in activations
    ]


@router.get("/activations/{activation_id}", response_model=CapturedActivationDetail)
async def get_captured_activation(activation_id: str):
    """
    Get details for a specific captured activation set.

    Returns full metadata including token information.
    """
    engine = get_patching_engine()
    captured = engine.get_captured_activations(activation_id)

    if captured is None:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return CapturedActivationDetail(
        id=captured.id,
        prompt=captured.prompt,
        model_type=captured.model_type,
        model_id=captured.model_id,
        num_tokens=len(captured.tokens),
        num_layers=captured.num_layers,
        tokens=captured.tokens,
        decoded_tokens=captured.decoded_tokens,
        available_keys=list(captured.activations.keys()),
    )


@router.delete("/activations/{activation_id}")
async def delete_captured_activation(activation_id: str):
    """
    Delete captured activations.

    Frees memory by removing stored activations.
    """
    engine = get_patching_engine()
    success = engine.delete_captured_activations(activation_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return {"status": "deleted", "activation_id": activation_id}


@router.delete("/activations")
async def clear_all_activations():
    """
    Clear all captured activations.

    Frees all memory used by stored activations.
    """
    engine = get_patching_engine()
    engine.clear_all_activations()
    return {"status": "cleared"}


# Patching experiment endpoints

@router.post("/run", response_model=PatchingResult)
async def run_patching_experiment(
    request: RunPatchingRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Run a patching experiment.

    This performs the following steps:
    1. Captures activations from the source prompt (if needed for patching)
    2. Generates baseline output from the target prompt (no patches)
    3. Generates patched output from target prompt with interventions applied
    4. Compares baseline and patched outputs

    For ablation experiments (zero_ablate, mean_ablate, noise_ablate), source_prompt is not required.
    For standard patching, source_prompt is required.

    Charges user credits if authenticated.

    Args:
        request: Contains source prompt (optional for ablation), target prompt, model, and patch specifications

    Returns:
        Baseline and patched outputs with comparison metrics
    """
    model_type = _get_model_type(request.model)

    # Check if any patches require source activations (patch intervention type)
    requires_source = any(p.intervention_type == 'patch' for p in request.patches)

    if requires_source and not request.source_prompt:
        raise HTTPException(
            status_code=400,
            detail="source_prompt is required when using 'patch' intervention type. Use ablation types (zero_ablate, mean_ablate, noise_ablate) or provide a source prompt."
        )

    # Check credits before running experiment
    if current_user:
        source_prompt_length = len(request.source_prompt) if request.source_prompt else 0
        estimated_cost = estimate_patching_cost(
            source_prompt_length=source_prompt_length,
            target_prompt_length=len(request.target_prompt),
            max_tokens=request.max_tokens,
            model_id=request.model,
            num_patches=len(request.patches),
        )
        if not check_sufficient_balance(db, current_user.id, estimated_cost):
            raise HTTPException(
                status_code=402,
                detail="INSUFFICIENT_CREDITS"
            )

    try:
        # Open-source models and Ozera base models run on Modal GPU workers
        if model_type == "open_source" or request.model in BASE_MODELS:
            worker = _get_inference_worker(request.model)

            # Convert patches to dict format for Modal
            patches_dicts = [
                {
                    "layer": p.layer,
                    "patch_type": p.patch_type,
                    "positions": p.positions,
                    "heads": p.heads,
                    "neurons": p.neurons,
                    "blend_factor": p.blend_factor,
                    "intervention_type": p.intervention_type,
                }
                for p in request.patches
            ]

            # Run patching on Modal
            result = worker().run_patching_experiment.remote(
                model_id=request.model,
                source_prompt=request.source_prompt,  # Can be None for ablation
                target_prompt=request.target_prompt,
                patches=patches_dicts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # Charge user after successful experiment
            if current_user:
                try:
                    # Calculate actual token counts from result
                    source_tokens = len(request.source_prompt.split()) if request.source_prompt else 0
                    target_tokens = len(result.get('baseline_tokens', []))
                    generated_tokens = len(result.get('patched_tokens', [])) - target_tokens

                    charge_patching(
                        db=db,
                        user_id=current_user.id,
                        source_tokens=source_tokens,
                        target_tokens=target_tokens,
                        generated_tokens=max(generated_tokens, request.max_tokens),
                        model_name=request.model,
                        num_patches=len(request.patches),
                    )
                except Exception as charge_error:
                    print(f"Warning: Failed to charge user {current_user.id}: {charge_error}")

            # Convert Modal result to response
            return _convert_modal_result_to_response(result, request.patches)

        else:
            # Custom trained Ozera models run locally (fallback)
            engine = get_patching_engine()
            generator = _get_ozera_generator(request.model)
            tokenizer = _get_ozera_tokenizer()

            # Only capture source activations if we have a source prompt
            captured = None
            if request.source_prompt:
                captured = engine.capture_source_activations(
                    prompt=request.source_prompt,
                    model_loader=generator,
                    model_type="ozera",
                    model_id=request.model,
                    tokenizer=tokenizer,
                )

            patches = [_patch_spec_to_config(p) for p in request.patches]

            result = engine.run_patched_generation(
                target_prompt=request.target_prompt,
                source_activation_id=captured.id if captured else None,
                patches=patches,
                model_loader=generator,
                model_type="ozera",
                tokenizer=tokenizer,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # Charge user after successful experiment
            if current_user:
                try:
                    source_tokens = len(captured.tokens) if captured else 0
                    charge_patching(
                        db=db,
                        user_id=current_user.id,
                        source_tokens=source_tokens,
                        target_tokens=len(result.baseline_tokens),
                        generated_tokens=len(result.patched_tokens) - len(result.baseline_tokens),
                        model_name=request.model,
                        num_patches=len(request.patches),
                    )
                except Exception as charge_error:
                    print(f"Warning: Failed to charge user {current_user.id}: {charge_error}")

            # Convert result to response
            return _convert_result_to_response(result, request.patches)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patching experiment failed: {str(e)}")


@router.post("/run-with-captured", response_model=PatchingResult)
async def run_patching_with_captured(
    request: RunPatchingWithCapturedRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Run a patching experiment using pre-captured activations.

    Use this when you've already captured source activations and want to run
    multiple experiments with different target prompts or patch configurations.

    Charges user credits if authenticated.

    Args:
        request: Contains activation ID, target prompt, model, and patch specifications

    Returns:
        Baseline and patched outputs with comparison metrics
    """
    import torch

    engine = get_patching_engine()

    # Verify activations exist
    captured = engine.get_captured_activations(request.source_activation_id)
    if captured is None:
        raise HTTPException(
            status_code=404,
            detail=f"Source activations not found: {request.source_activation_id}"
        )

    # Verify model matches
    if captured.model_id != request.model:
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch: activations are from '{captured.model_id}', but request specifies '{request.model}'"
        )

    model_type = captured.model_type

    # Check credits before running experiment
    if current_user:
        estimated_cost = estimate_patching_cost(
            source_prompt_length=len(captured.prompt),
            target_prompt_length=len(request.target_prompt),
            max_tokens=request.max_tokens,
            model_id=request.model,
            num_patches=len(request.patches),
        )
        if not check_sufficient_balance(db, current_user.id, estimated_cost):
            raise HTTPException(
                status_code=402,
                detail="INSUFFICIENT_CREDITS"
            )

    try:
        # Open-source models and Ozera base models run on Modal GPU
        if model_type == "open_source" or request.model in BASE_MODELS:
            worker = _get_inference_worker(request.model)

            # Serialize activations for transfer to Modal
            serialized_activations = {
                'id': captured.id,
                'prompt': captured.prompt,
                'tokens': captured.tokens,
                'decoded_tokens': captured.decoded_tokens,
                'model_type': captured.model_type,
                'model_id': captured.model_id,
                'num_layers': captured.num_layers,
                'activations': {
                    key: tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor
                    for key, tensor in captured.activations.items()
                },
            }

            # Convert patches to dict format for Modal
            patches_dicts = [
                {
                    "layer": p.layer,
                    "patch_type": p.patch_type,
                    "positions": p.positions,
                    "heads": p.heads,
                    "neurons": p.neurons,
                    "blend_factor": p.blend_factor,
                    "intervention_type": p.intervention_type,
                }
                for p in request.patches
            ]

            # Run patching on Modal with pre-captured activations
            result = worker().run_patching_with_activations.remote(
                model_id=request.model,
                target_prompt=request.target_prompt,
                patches=patches_dicts,
                source_activations=serialized_activations,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # Charge user after successful experiment
            if current_user:
                try:
                    charge_patching(
                        db=db,
                        user_id=current_user.id,
                        source_tokens=len(captured.tokens),
                        target_tokens=len(result.get('baseline_tokens', [])),
                        generated_tokens=len(result.get('patched_tokens', [])) - len(result.get('baseline_tokens', [])),
                        model_name=request.model,
                        num_patches=len(request.patches),
                    )
                except Exception as charge_error:
                    print(f"Warning: Failed to charge user {current_user.id}: {charge_error}")

            return _convert_modal_result_to_response(result, request.patches)

        else:
            # Custom trained Ozera models run locally
            patches = [_patch_spec_to_config(p) for p in request.patches]
            generator = _get_ozera_generator(request.model)
            tokenizer = _get_ozera_tokenizer()
            result = engine.run_patched_generation(
                target_prompt=request.target_prompt,
                source_activation_id=request.source_activation_id,
                patches=patches,
                model_loader=generator,
                model_type="ozera",
                tokenizer=tokenizer,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # Charge user after successful experiment
            if current_user:
                try:
                    charge_patching(
                        db=db,
                        user_id=current_user.id,
                        source_tokens=len(captured.tokens),
                        target_tokens=len(result.baseline_tokens),
                        generated_tokens=len(result.patched_tokens) - len(result.baseline_tokens),
                        model_name=request.model,
                        num_patches=len(request.patches),
                    )
                except Exception as charge_error:
                    print(f"Warning: Failed to charge user {current_user.id}: {charge_error}")

            return _convert_result_to_response(result, request.patches)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patching experiment failed: {str(e)}")


def _convert_result_to_response(result, patches: list[PatchSpec]) -> PatchingResult:
    """Convert engine PatchingResult to API response."""
    # Convert changed tokens
    changed_tokens = [
        ChangedToken(
            position=ct['position'],
            baseline_token=ct['baseline_token'],
            patched_token=ct['patched_token'],
        )
        for ct in result.effect_summary.get('changed_tokens', [])
    ]

    effect_summary = EffectSummary(
        first_divergence_position=result.effect_summary.get('first_divergence_position'),
        token_changes=result.effect_summary.get('token_changes', 0),
        changed_tokens=changed_tokens,
        baseline_length=result.effect_summary.get('baseline_length', len(result.baseline_tokens)),
        patched_length=result.effect_summary.get('patched_length', len(result.patched_tokens)),
    )

    return PatchingResult(
        baseline_output=result.baseline_output,
        patched_output=result.patched_output,
        baseline_tokens=result.baseline_tokens,
        patched_tokens=result.patched_tokens,
        baseline_decoded=result.baseline_decoded,
        patched_decoded=result.patched_decoded,
        source_activation_id=result.source_activation_id,
        patches_applied=patches,
        effect_summary=effect_summary,
    )


def _convert_modal_result_to_response(result: dict, patches: list[PatchSpec]) -> PatchingResult:
    """Convert Modal worker result dict to API response."""
    # Convert changed tokens
    effect_summary_data = result.get('effect_summary', {})
    changed_tokens = [
        ChangedToken(
            position=ct['position'],
            baseline_token=ct['baseline_token'],
            patched_token=ct['patched_token'],
        )
        for ct in effect_summary_data.get('changed_tokens', [])
    ]

    effect_summary = EffectSummary(
        first_divergence_position=effect_summary_data.get('first_divergence_position'),
        token_changes=effect_summary_data.get('token_changes', 0),
        changed_tokens=changed_tokens,
        baseline_length=effect_summary_data.get('baseline_length', len(result.get('baseline_tokens', []))),
        patched_length=effect_summary_data.get('patched_length', len(result.get('patched_tokens', []))),
    )

    return PatchingResult(
        baseline_output=result['baseline_output'],
        patched_output=result['patched_output'],
        baseline_tokens=result['baseline_tokens'],
        patched_tokens=result['patched_tokens'],
        baseline_decoded=result['baseline_decoded'],
        patched_decoded=result['patched_decoded'],
        source_activation_id=result['source_activation_id'],
        patches_applied=patches,
        effect_summary=effect_summary,
    )


# Model info endpoint for patching

@router.get("/models")
async def list_available_models():
    """
    List models available for patching experiments.

    Returns both Ozera base models and open-source models that support
    activation patching.
    """
    models = []

    # Add Ozera base models
    for model_id in BASE_MODELS:
        models.append({
            "model_id": model_id,
            "model_type": "ozera",
            "display_name": f"Ozera {model_id.title()}",
            "num_layers": 6,  # Default for Ozera models
            "num_heads": 8,   # Default for Ozera models
        })

    # Add open-source models
    for model_id, config in OPEN_SOURCE_MODELS.items():
        models.append({
            "model_id": model_id,
            "model_type": "open_source",
            "display_name": config.display_name,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
        })

    return models


# Base model configurations (avoid loading model just for config)
BASE_MODEL_CONFIGS = {
    "nano": {"num_layers": 6, "num_heads": 6},
    "mini": {"num_layers": 8, "num_heads": 8},
}


@router.get("/models/{model_id}/layers")
async def get_model_layers(model_id: str):
    """
    Get layer information for a model.

    Returns the number of layers and available patch points for each layer.
    """
    model_type = _get_model_type(model_id)

    if model_type == "open_source":
        if model_id not in OPEN_SOURCE_MODELS:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        config = OPEN_SOURCE_MODELS[model_id]
        num_layers = config.num_layers
        num_heads = config.num_heads
    elif model_id in BASE_MODEL_CONFIGS:
        # Use hardcoded config for base models (avoid loading model on CPU)
        config = BASE_MODEL_CONFIGS[model_id]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
    else:
        # Custom trained Ozera models - need to load to get config
        try:
            from inference.model_loader import ModelLoader
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(backend_dir, "models")
            loader = ModelLoader(models_dir=models_dir)
            _, model_config = loader.load_model(model_id)
            num_layers = model_config.num_layers
            num_heads = model_config.num_heads
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Available patch types at each layer
    patch_types = ['attention', 'mlp', 'residual', 'attn_output', 'ff_output', 'post_attn', 'post_ff']

    layers = []
    for i in range(num_layers):
        layers.append({
            "index": i,
            "patch_types": patch_types,
            "num_heads": num_heads,
        })

    return {
        "model_id": model_id,
        "model_type": model_type,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "layers": layers,
        "patch_types": patch_types,
    }
