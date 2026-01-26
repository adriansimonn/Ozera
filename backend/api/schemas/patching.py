"""
Pydantic models for activation patching API endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# Patch specification schemas

class PatchSpec(BaseModel):
    """Specification for a single activation patch."""
    layer: int = Field(..., ge=0, description="Layer index to patch")
    patch_type: Literal['attention', 'mlp', 'residual', 'attn_output', 'ff_output', 'post_attn', 'post_ff'] = Field(
        ..., description="Type of activation to patch"
    )
    positions: Optional[list[int]] = Field(
        default=None, description="Token positions to patch (None = all)"
    )
    heads: Optional[list[int]] = Field(
        default=None, description="Attention heads to patch (None = all, for attention types)"
    )
    neurons: Optional[list[int]] = Field(
        default=None, description="MLP neurons to patch (None = all, for MLP types)"
    )
    blend_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Blend factor (0 = original, 1 = full replacement)"
    )
    intervention_type: Literal['patch', 'zero_ablate', 'mean_ablate', 'noise_ablate'] = Field(
        default='patch',
        description="Type of intervention: 'patch' (replace with source activations), 'zero_ablate' (zero out), 'mean_ablate' (replace with mean), 'noise_ablate' (replace with noise)"
    )


# Request schemas

class CaptureActivationsRequest(BaseModel):
    """Request to capture source activations."""
    prompt: str = Field(..., min_length=1, description="Source prompt to capture activations from")
    model: str = Field(..., description="Model ID (e.g., 'nano', 'mini', 'smollm-135m')")


class RunPatchingRequest(BaseModel):
    """Request to run a patching experiment."""
    source_prompt: Optional[str] = Field(
        default=None,
        description="Source prompt (activations to copy from). Required for 'patch' intervention type, optional for ablation types."
    )
    target_prompt: str = Field(..., min_length=1, description="Target prompt (to run generation on)")
    model: str = Field(..., description="Model ID")
    patches: list[PatchSpec] = Field(..., min_length=1, description="List of patches to apply")
    max_tokens: int = Field(default=50, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0 = deterministic)")


class RunPatchingWithCapturedRequest(BaseModel):
    """Request to run patching with pre-captured activations."""
    source_activation_id: str = Field(..., description="ID of pre-captured source activations")
    target_prompt: str = Field(..., min_length=1, description="Target prompt to run generation on")
    model: str = Field(..., description="Model ID (must match source activations)")
    patches: list[PatchSpec] = Field(..., min_length=1, description="List of patches to apply")
    max_tokens: int = Field(default=50, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")


# Response schemas

class CapturedActivationSummary(BaseModel):
    """Summary of captured activations."""
    id: str
    prompt: str
    model_type: Literal['ozera', 'open_source']
    model_id: str
    num_tokens: int
    num_layers: int


class CapturedActivationDetail(CapturedActivationSummary):
    """Detailed captured activation info including tokens."""
    tokens: list[int]
    decoded_tokens: list[str]
    available_keys: list[str] = Field(description="Available activation keys")


class ChangedToken(BaseModel):
    """Information about a changed token."""
    position: int
    baseline_token: str
    patched_token: str


class EffectSummary(BaseModel):
    """Summary of patching effects."""
    first_divergence_position: Optional[int] = Field(
        description="Position where baseline and patched outputs first differ"
    )
    token_changes: int = Field(description="Total number of differing tokens")
    changed_tokens: list[ChangedToken] = Field(description="Details of first 10 changed tokens")
    baseline_length: int
    patched_length: int


class PatchingResult(BaseModel):
    """Result of a patching experiment."""
    baseline_output: str = Field(description="Generated text without patches")
    patched_output: str = Field(description="Generated text with patches applied")
    baseline_tokens: list[int]
    patched_tokens: list[int]
    baseline_decoded: list[str]
    patched_decoded: list[str]
    source_activation_id: str
    patches_applied: list[PatchSpec]
    effect_summary: EffectSummary


class CaptureActivationsResponse(BaseModel):
    """Response from capturing activations."""
    activation_id: str
    prompt: str
    model_type: Literal['ozera', 'open_source']
    model_id: str
    num_tokens: int
    num_layers: int
    decoded_tokens: list[str]
