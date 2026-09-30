"""
API endpoints for activation patching experiments.

Provides endpoints for:
- Capturing source activations from a prompt
- Running patching experiments (baseline vs patched generation)
- Managing captured activation storage
"""

from fastapi import APIRouter, HTTPException, Depends
from api.error_utils import safe_detail
from sqlalchemy.orm import Session
from typing import Optional

from core.activation_limits import ActivationLimitError
from core.open_source import OPEN_SOURCE_MODELS, OpenSourceModelLoader, get_gpu_tier, get_loader_for_model
from core.patching import get_patching_engine, PatchConfig
from core.patching.engine import CaptureTooLargeError, PatchError, patch_activation_key, validate_patches
from core.tensor_codec import encode_tensor, is_tensor_entry, to_float32
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
from middleware.auth_middleware import get_optional_current_user, get_current_user
from services.credit_service import InsufficientBalanceError
from services.custom_models import list_custom_model_names, resolve_model
from services.model_specs import BASE_MODEL_CONFIGS, model_specs
from models.database import User, TransactionType
from db import get_db
from services.credit_service import (
    estimate_patching_cost,
    charge_flat,
    charge_patching,
    check_sufficient_balance,
    min_gpu_request_charge,
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
    from services.modal_inference import get_inference_worker

    # Open-source models use the tier specified in their config; base and custom
    # Ozera models use L4
    return get_inference_worker(get_gpu_tier(model_id))


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


async def _check_patches(
    db: Session,
    user_id: int,
    model_id: str,
    patches: list[PatchSpec],
    source_tokens: Optional[int] = None,
) -> None:
    """
    Reject (400) patches the model can't apply, before starting a GPU for them.

    The worker checks again with everything it knows (e.g. the prompts' token counts).
    """
    specs = await model_specs(db, user_id, model_id)
    if specs.layers is None or specs.heads is None:
        return
    try:
        validate_patches(
            [_patch_spec_to_config(p) for p in patches],
            num_layers=specs.layers,
            num_heads=specs.heads,
            mlp_neurons=specs.mlp_neurons,
            source_tokens=source_tokens,
        )
    except PatchError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Activation capture endpoints

@router.post("/capture", response_model=CaptureActivationsResponse)
async def capture_activations(
    request: CaptureActivationsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Capture activations from a source prompt.

    This captures all layer activations from a forward pass through the model,
    which can later be used for patching experiments. Charged as a GPU request
    (the model's per-request minimum).

    Args:
        request: Contains prompt and model ID

    Returns:
        Activation ID and metadata for the captured activations
    """
    import torch
    from core.patching import CapturedActivations

    model = resolve_model(db, current_user.id, request.model)
    engine = get_patching_engine()

    cost = min_gpu_request_charge(request.model)
    if not check_sufficient_balance(db, current_user.id, cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    try:
        # All models run on Modal GPU (base, open-source, and custom)
        worker = _get_inference_worker(request.model)

        # Capture activations on Modal
        result = await worker().capture_activations.remote.aio(
            model_id=request.model,
            prompt=request.prompt,
            **model.worker_kwargs(),
        )

        # Reconstruct activations locally for caching (without logits, which nothing uses
        # and which workers from before they were dropped still send)
        reconstructed_activations = {}
        for key, value in result['activations'].items():
            if key == 'logits':
                continue
            if is_tensor_entry(value):
                reconstructed_activations[key] = torch.tensor(to_float32(value))
            elif isinstance(value, list):
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
            user_id=current_user.id,
        )
        engine.add_captured_activations(captured)

        charge_flat(db, current_user.id, cost, TransactionType.PATCHING_CHARGE,
                    f"Activation capture ({request.model}): {len(captured.tokens)} tokens")

        return CaptureActivationsResponse(
            activation_id=captured.id,
            prompt=captured.prompt,
            model_type=captured.model_type,
            model_id=captured.model_id,
            num_tokens=len(captured.tokens),
            num_layers=captured.num_layers,
            decoded_tokens=captured.decoded_tokens,
        )

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    except ActivationLimitError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CaptureTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to capture activations")


@router.get("/activations", response_model=list[CapturedActivationSummary])
async def list_captured_activations(
    current_user: User = Depends(get_current_user),
):
    """
    List the user's captured activations.

    Returns summaries of the user's captured activation sets currently stored in memory.
    """
    engine = get_patching_engine()
    activations = engine.list_captured_activations(current_user.id)

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
async def get_captured_activation(
    activation_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get details for a specific captured activation set.

    Returns full metadata including token information.
    """
    engine = get_patching_engine()
    captured = engine.get_captured_activations(activation_id, current_user.id)

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
async def delete_captured_activation(
    activation_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Delete captured activations.

    Frees memory by removing stored activations.
    """
    engine = get_patching_engine()
    success = engine.delete_captured_activations(activation_id, current_user.id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Activations not found: {activation_id}")

    return {"status": "deleted", "activation_id": activation_id}


@router.delete("/activations")
async def clear_all_activations(
    current_user: User = Depends(get_current_user),
):
    """
    Clear the user's captured activations.

    Frees the memory used by the user's stored activations.
    """
    engine = get_patching_engine()
    engine.clear_all_activations(current_user.id)
    return {"status": "cleared"}


# Patching experiment endpoints

@router.post("/run", response_model=PatchingResult)
async def run_patching_experiment(
    request: RunPatchingRequest,
    current_user: User = Depends(get_current_user),
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
    model = resolve_model(db, current_user.id, request.model)

    # Check if any patches require source activations (patch intervention type)
    requires_source = any(p.intervention_type == 'patch' for p in request.patches)

    if requires_source and not request.source_prompt:
        raise HTTPException(
            status_code=400,
            detail="source_prompt is required when using 'patch' intervention type. Use ablation types (zero_ablate, mean_ablate, noise_ablate) or provide a source prompt."
        )

    await _check_patches(db, current_user.id, request.model, request.patches)

    # Check credits before running experiment
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
        # All models run on Modal GPU workers (base, open-source, and custom)
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
        result = await worker().run_patching_experiment.remote.aio(
            model_id=request.model,
            source_prompt=request.source_prompt,  # Can be None for ablation
            target_prompt=request.target_prompt,
            patches=patches_dicts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **model.worker_kwargs(),
        )

        # Charge user after successful experiment — fail if charge fails
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

        # Convert Modal result to response
        return _convert_modal_result_to_response(result, request.patches)

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except (ActivationLimitError, PatchError) as e:
        # The source prompt is too long to capture on this model, or a patch can't be applied
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid patching request"))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Patching experiment failed")


@router.post("/run-with-captured", response_model=PatchingResult)
async def run_patching_with_captured(
    request: RunPatchingWithCapturedRequest,
    current_user: User = Depends(get_current_user),
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
    captured = engine.get_captured_activations(request.source_activation_id, current_user.id)
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

    model = resolve_model(db, current_user.id, request.model)
    await _check_patches(db, current_user.id, request.model, request.patches, len(captured.tokens))

    # Check credits before running experiment
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
        # All models run on Modal GPU (base, open-source, and custom)
        worker = _get_inference_worker(request.model)

        # Serialize activations for transfer to Modal: only those the patches replace (a
        # capture holds every layer's), as compact raw bytes
        needed_keys = {
            patch_activation_key(p.patch_type, p.layer)
            for p in request.patches
            if p.intervention_type == 'patch'
        }
        serialized_activations = {
            'id': captured.id,
            'prompt': captured.prompt,
            'tokens': captured.tokens,
            'decoded_tokens': captured.decoded_tokens,
            'model_type': captured.model_type,
            'model_id': captured.model_id,
            'num_layers': captured.num_layers,
            'activations': {
                key: encode_tensor(tensor) if isinstance(tensor, torch.Tensor) else tensor
                for key, tensor in captured.activations.items()
                if key in needed_keys
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
        result = await worker().run_patching_with_activations.remote.aio(
            model_id=request.model,
            target_prompt=request.target_prompt,
            patches=patches_dicts,
            source_activations=serialized_activations,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **model.worker_kwargs(),
        )

        # Charge user after successful experiment — fail if charge fails
        charge_patching(
            db=db,
            user_id=current_user.id,
            source_tokens=len(captured.tokens),
            target_tokens=len(result.get('baseline_tokens', [])),
            generated_tokens=len(result.get('patched_tokens', [])) - len(result.get('baseline_tokens', [])),
            model_name=request.model,
            num_patches=len(request.patches),
        )

        return _convert_modal_result_to_response(result, request.patches)

    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="Insufficient credits.")
    except PatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=safe_detail(e, "Invalid patching request"))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Patching experiment failed")


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
async def list_available_models(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    List models available for patching experiments.

    Returns Ozera base models, open-source models, and user's custom models
    (trained and uploaded) that support activation patching.
    """
    def entry(specs, model_type: str, display_name: str) -> dict:
        return {
            "model_id": specs.name,
            "model_type": model_type,
            "display_name": display_name,
            "num_layers": specs.layers,
            "num_heads": specs.heads,
        }

    user_id = current_user.id if current_user else None
    models = [
        entry(await model_specs(db, user_id, model_id), "ozera", f"Ozera {model_id.title()}")
        for model_id in BASE_MODEL_CONFIGS
    ]
    models += [
        entry(await model_specs(db, user_id, model_id), "open_source", config.display_name)
        for model_id, config in OPEN_SOURCE_MODELS.items()
    ]

    # Add user's custom models (trained and uploaded) if authenticated
    if current_user:
        for name in list_custom_model_names(db, current_user.id):
            models.append(entry(await model_specs(db, current_user.id, name), "custom", name))

    return models


@router.get("/models/{model_id}/layers")
async def get_model_layers(
    model_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Get layer information for a model.

    Returns the number of layers and heads, the neurons per MLP layer (null if unknown),
    and the patch points of each layer. Every patch type applies to every model; heads can be
    chosen for attention patches, neurons for MLP patches.
    """
    if current_user is None and model_id not in BASE_MODEL_CONFIGS and model_id not in OPEN_SOURCE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    try:
        specs = await model_specs(db, current_user.id if current_user else None, model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if specs.layers is None or specs.heads is None:
        raise HTTPException(status_code=503, detail="Couldn't read the model's file, please try again")

    # Available patch types at each layer
    patch_types = ['attention', 'mlp', 'residual', 'attn_output', 'ff_output', 'post_attn', 'post_ff']

    layers = []
    for i in range(specs.layers):
        layers.append({
            "index": i,
            "patch_types": patch_types,
            "num_heads": specs.heads,
        })

    return {
        "model_id": model_id,
        "model_type": _get_model_type(model_id),
        "num_layers": specs.layers,
        "num_heads": specs.heads,
        "mlp_neurons": specs.mlp_neurons,
        "layers": layers,
        "patch_types": patch_types,
    }
