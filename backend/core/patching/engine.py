"""
Core patching engine for activation patching experiments.

Orchestrates capturing source activations, applying patches during generation,
and comparing baseline vs patched outputs.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Any
import os
import torch
import torch.nn.functional as F
import uuid

from .hooks import (
    PositionOffset,
    create_replacement_hook,
    create_attention_patch_hook,
    create_mlp_patch_hook,
    create_residual_patch_hook,
    create_zero_ablation_hook,
)


# Captured activation (per layer) that each patch type replaces
PATCH_TYPE_ACTIVATION_KEYS = {
    'attention': 'attn_output',
    'attn_output': 'attn_output',
    'mlp': 'ff_output',
    'ff_output': 'ff_output',
    'residual': 'post_ff',
    'post_attn': 'post_attn',
    'post_ff': 'post_ff',
}


def patch_activation_key(patch_type: str, layer: int) -> str:
    """Key of the captured activation a patch replaces, e.g. 'layer_2_attn_output'."""
    return f"layer_{layer}_{PATCH_TYPE_ACTIVATION_KEYS.get(patch_type, patch_type)}"


# Interventions that only change the positions they're applied to. The rest (mean and
# noise ablation) are computed from the whole sequence in each forward pass.
POSITION_LOCAL_INTERVENTIONS = ('patch', 'zero_ablate')

# Bounds on the captures kept with add_captured_activations (the backend's store for the
# patching and analysis pages; Modal workers only hold a capture during its request).
# Captures are float32: Qwen3-4B takes ~2 MB per token plus its attention weights.
CAPTURE_CACHE_MAX_BYTES = int(os.getenv("PATCHING_CAPTURE_CACHE_MB", "1024")) * 1024 * 1024
CAPTURE_MAX_BYTES = CAPTURE_CACHE_MAX_BYTES // 2
CAPTURES_PER_USER = 20


class CaptureTooLargeError(ValueError):
    """A capture is too large to keep (see CAPTURE_MAX_BYTES)."""


def _capture_bytes(captured: 'CapturedActivations') -> int:
    return sum(
        t.numel() * t.element_size() for t in captured.activations.values() if isinstance(t, torch.Tensor)
    )


@dataclass
class PatchConfig:
    """
    Configuration for a single activation patch.

    Specifies which layer, what type of activation, which positions/heads
    to patch, and how to blend the patched values.
    """
    layer: int
    patch_type: Literal['attention', 'mlp', 'residual', 'attn_output', 'ff_output', 'post_attn', 'post_ff']
    positions: Optional[list[int]] = None  # None = all positions
    heads: Optional[list[int]] = None  # None = all heads (for attention)
    neurons: Optional[list[int]] = None  # None = all neurons (for MLP)
    blend_factor: float = 1.0  # 1.0 = full replacement
    intervention_type: Literal['patch', 'zero_ablate', 'mean_ablate', 'noise_ablate'] = 'patch'

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'layer': self.layer,
            'patch_type': self.patch_type,
            'positions': self.positions,
            'heads': self.heads,
            'neurons': self.neurons,
            'blend_factor': self.blend_factor,
            'intervention_type': self.intervention_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PatchConfig':
        """Create from dictionary."""
        return cls(
            layer=data['layer'],
            patch_type=data['patch_type'],
            positions=data.get('positions'),
            heads=data.get('heads'),
            neurons=data.get('neurons'),
            blend_factor=data.get('blend_factor', 1.0),
            intervention_type=data.get('intervention_type', 'patch'),
        )


@dataclass
class CapturedActivations:
    """
    Container for captured activations from a forward pass.
    """
    id: str
    prompt: str
    tokens: list[int]
    decoded_tokens: list[str]
    activations: dict[str, torch.Tensor]
    model_type: Literal['ozera', 'open_source']
    model_id: str
    num_layers: int
    user_id: Optional[int] = None  # Owner, on the backend (the Modal worker doesn't track users)

    def get_layer_activation(self, layer: int, key: str) -> Optional[torch.Tensor]:
        """Get a specific activation from a layer."""
        full_key = f"layer_{layer}_{key}"
        return self.activations.get(full_key)

    def get_activation(self, key: str) -> Optional[torch.Tensor]:
        """Get a top-level activation (embeddings, logits, etc)."""
        return self.activations.get(key)


@dataclass
class PatchingResult:
    """
    Result of a patching experiment.
    """
    baseline_output: str
    patched_output: str
    baseline_tokens: list[int]
    patched_tokens: list[int]
    baseline_decoded: list[str]
    patched_decoded: list[str]
    source_activation_id: Optional[str]  # None for ablation-only experiments
    patches_applied: list[PatchConfig]
    effect_summary: dict = field(default_factory=dict)


class PatchingEngine:
    """
    Core engine for activation patching experiments.

    Supports both Ozera custom models and open-source HuggingFace models.
    Captures activations from a source prompt, then applies patches during
    generation on a target prompt.
    """

    def __init__(self):
        """Initialize the patching engine."""
        self._captured_activations: dict[str, CapturedActivations] = {}
        self._intervention_hooks: list = []

    def capture_source_activations(
        self,
        prompt: str,
        model_loader: Any,
        model_type: Literal['ozera', 'open_source'],
        model_id: str,
        tokenizer: Any = None,
    ) -> CapturedActivations:
        """
        Capture activations from a source prompt.

        Captures one forward pass over the prompt, so positions match the prompt's tokens
        for both model types. Logits aren't kept: nothing patches or analyzes them.

        Args:
            prompt: Source prompt to capture activations from.
            model_loader: Model loader (OpenSourceModelLoader for HF, TextGenerator for Ozera).
            model_type: 'ozera' or 'open_source'.
            model_id: Model identifier string.
            tokenizer: Tokenizer (required for Ozera models).

        Returns:
            CapturedActivations object with source activations.
        """
        activation_id = str(uuid.uuid4())

        if model_type == 'open_source':
            result = model_loader.capture_prompt_activations(prompt)
            activations = result['activations']
            tokens = result['tokens']
            decoded_tokens = result['decoded_tokens']
            num_layers = model_loader.config.num_layers

        else:  # Ozera model
            # Use TextGenerator's activation capture
            if tokenizer is None:
                raise ValueError("Tokenizer required for Ozera models")

            # Encode prompt
            prompt_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)

            # Move input_ids to the model's device
            if hasattr(model_loader, 'device'):
                input_ids = input_ids.to(model_loader.device)
            elif hasattr(model_loader, 'model') and hasattr(model_loader.model, 'token_embedding'):
                input_ids = input_ids.to(next(model_loader.model.parameters()).device)
            else:
                # model_loader is the model itself (e.g., raw TransformerLM)
                input_ids = input_ids.to(next(model_loader.parameters()).device)

            # Get the model
            if hasattr(model_loader, 'model'):
                model = model_loader.model
            else:
                model = model_loader

            # Forward pass with activation capture
            with torch.no_grad():
                _, _, raw_activations = model.forward(
                    input_ids,
                    return_attention=True,
                    capture_activations=True,
                )

            # Convert to flat dict format
            activations = self._flatten_ozera_activations(raw_activations)
            tokens = input_ids[0].cpu().tolist()
            decoded_tokens = [tokenizer.decode([t]) for t in tokens]
            num_layers = model.config.num_layers

        captured = CapturedActivations(
            id=activation_id,
            prompt=prompt,
            tokens=tokens,
            decoded_tokens=decoded_tokens,
            activations=activations,
            model_type=model_type,
            model_id=model_id,
            num_layers=num_layers,
        )

        self._captured_activations[activation_id] = captured
        return captured

    def _flatten_ozera_activations(self, raw_activations: dict) -> dict[str, torch.Tensor]:
        """Flatten Ozera model activations to a flat dict format."""
        activations = {}

        # Top-level activations
        for key in ['token_embeddings', 'positional_embeddings', 'combined_embeddings', 'final_layer_norm']:
            if key in raw_activations and raw_activations[key] is not None:
                activations[key] = raw_activations[key].clone()

        # Layer activations
        if 'layers' in raw_activations:
            for layer_idx, layer_data in enumerate(raw_activations['layers']):
                if layer_data is None:
                    continue
                for key, tensor in layer_data.items():
                    if tensor is not None:
                        activations[f"layer_{layer_idx}_{key}"] = tensor.clone()

        return activations

    def run_patched_generation(
        self,
        target_prompt: str,
        source_activation_id: Optional[str],
        patches: list[PatchConfig],
        model_loader: Any,
        model_type: Literal['ozera', 'open_source'],
        tokenizer: Any = None,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> PatchingResult:
        """
        Run generation with patches applied.

        First generates baseline output without patches, then generates
        patched output with interventions applied.

        Args:
            target_prompt: Prompt to run generation on.
            source_activation_id: ID of captured source activations. Can be None for ablation-only experiments.
            patches: List of patch configurations to apply.
            model_loader: Model loader or generator.
            model_type: 'ozera' or 'open_source'.
            tokenizer: Tokenizer (required for Ozera).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            PatchingResult with baseline and patched outputs.
        """
        # Check if any patches require source activations (patch intervention type)
        requires_source = any(
            (patch.intervention_type if hasattr(patch, 'intervention_type') else 'patch') == 'patch'
            for patch in patches
        )

        source = None
        if source_activation_id:
            source = self._captured_activations.get(source_activation_id)
            if source is None and requires_source:
                raise ValueError(f"Source activations not found: {source_activation_id}")
        elif requires_source:
            raise ValueError("Source activations required for patching interventions. Use ablation types (zero_ablate, mean_ablate, noise_ablate) or provide source activations.")

        # Generate baseline (no patches)
        if model_type == 'open_source':
            baseline_result = model_loader.generate(
                prompt=target_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                # Same setting as the patched run, so the two stay comparable
                use_cache=self._open_source_use_cache(patches),
            )
            baseline_output = baseline_result['text']
            baseline_tokens = baseline_result['tokens']
            baseline_decoded = [model_loader.tokenizer.decode([t]) for t in baseline_tokens]
        else:
            # Ozera model - handle both TextGenerator wrapper and raw TransformerLM
            if tokenizer is None:
                raise ValueError("Tokenizer required for Ozera models")

            # Check if this is a TextGenerator wrapper (has .model attribute) or raw TransformerLM
            if hasattr(model_loader, 'model'):
                # TextGenerator wrapper
                model = model_loader.model
                device = model_loader.device if hasattr(model_loader, 'device') else 'cpu'
            else:
                # Raw TransformerLM model
                model = model_loader
                device = next(model.parameters()).device

            # Tokenize and generate baseline
            prompt_ids = tokenizer.encode(target_prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

            with torch.no_grad():
                gen_ids, _ = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            baseline_tokens = gen_ids[0].cpu().tolist()
            baseline_output = tokenizer.decode(baseline_tokens)
            baseline_decoded = [tokenizer.decode([t]) for t in baseline_tokens]

        # Generate with patches
        patched_output, patched_tokens, patched_decoded = self._generate_with_patches(
            target_prompt=target_prompt,
            source=source,
            patches=patches,
            model_loader=model_loader,
            model_type=model_type,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Compute effect summary
        effect_summary = self._compute_effect_summary(
            baseline_tokens=baseline_tokens,
            patched_tokens=patched_tokens,
            baseline_decoded=baseline_decoded,
            patched_decoded=patched_decoded,
        )

        return PatchingResult(
            baseline_output=baseline_output,
            patched_output=patched_output,
            baseline_tokens=baseline_tokens,
            patched_tokens=patched_tokens,
            baseline_decoded=baseline_decoded,
            patched_decoded=patched_decoded,
            source_activation_id=source_activation_id,
            patches_applied=patches,
            effect_summary=effect_summary,
        )

    def _generate_with_patches(
        self,
        target_prompt: str,
        source: Optional[CapturedActivations],
        patches: list[PatchConfig],
        model_loader: Any,
        model_type: Literal['ozera', 'open_source'],
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, list[int], list[str]]:
        """Generate with patches applied via hooks."""

        if model_type == 'open_source':
            return self._generate_with_patches_open_source(
                target_prompt, source, patches, model_loader,
                max_new_tokens, temperature,
            )
        else:
            return self._generate_with_patches_ozera(
                target_prompt, source, patches, model_loader,
                tokenizer, max_new_tokens, temperature,
            )

    def _generate_with_patches_open_source(
        self,
        target_prompt: str,
        source: Optional[CapturedActivations],
        patches: list[PatchConfig],
        model_loader: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, list[int], list[str]]:
        """
        Apply patches to open-source model generation.

        Patches address absolute token positions, as they do for Ozera models (which
        recompute the whole sequence each step). With the KV cache, each forward pass after
        the prompt computes only the newest token, so the hooks track where each pass starts
        in the sequence. Mean and noise ablation depend on the whole sequence at every step,
        so experiments using them generate without the cache (see _open_source_use_cache).
        """
        offset = PositionOffset()
        registered_hooks = [offset.attach(self._get_open_source_decoder(model_loader))]

        try:
            # Register intervention hooks
            for patch in patches:
                hook_handle = self._register_open_source_patch_hook(
                    model_loader, source, patch, offset
                )
                if hook_handle is not None:
                    registered_hooks.append(hook_handle)

            # Generate with patches active
            result = model_loader.generate(
                prompt=target_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_cache=self._open_source_use_cache(patches),
            )

            output = result['text']
            tokens = result['tokens']
            decoded = [model_loader.tokenizer.decode([t]) for t in tokens]

            return output, tokens, decoded

        finally:
            # Clean up hooks
            for hook in registered_hooks:
                hook.remove()

    def _register_open_source_patch_hook(
        self,
        model_loader: Any,
        source: Optional[CapturedActivations],
        patch: PatchConfig,
        offset: PositionOffset,
    ):
        """Register a patch hook on an open-source model."""
        activation_key = PATCH_TYPE_ACTIVATION_KEYS.get(patch.patch_type, patch.patch_type)

        # Get the module to hook
        module = self._get_open_source_module(model_loader, patch.layer, activation_key)
        if module is None:
            return None

        # Get intervention type
        intervention_type = patch.intervention_type if hasattr(patch, 'intervention_type') else 'patch'

        # Handle ablation types (don't need source activations)
        if intervention_type in ['zero_ablate', 'mean_ablate', 'noise_ablate']:
            hook_fn = create_zero_ablation_hook(
                positions=patch.positions,
                blend_factor=patch.blend_factor,
                ablation_type=intervention_type,
                offset=offset,
            )
            return module.register_forward_hook(hook_fn)

        # Standard patching requires source activations
        if source is None:
            return None

        full_key = f"layer_{patch.layer}_{activation_key}"
        source_activation = source.activations.get(full_key)
        if source_activation is None:
            return None

        # Move the source to the model's device once, rather than in every forward pass
        source_activation = source_activation.to(next(model_loader.model.parameters()).device)

        # Create appropriate hook
        if patch.patch_type in ['attention', 'attn_output']:
            hook_fn = create_attention_patch_hook(
                source_attention=source_activation,
                heads=patch.heads,
                positions=patch.positions,
                blend_factor=patch.blend_factor,
                offset=offset,
            )
        elif patch.patch_type in ['mlp', 'ff_output']:
            hook_fn = create_mlp_patch_hook(
                source_mlp_output=source_activation,
                positions=patch.positions,
                neurons=patch.neurons,
                blend_factor=patch.blend_factor,
                offset=offset,
            )
        else:
            hook_fn = create_residual_patch_hook(
                source_residual=source_activation,
                positions=patch.positions,
                blend_factor=patch.blend_factor,
                offset=offset,
            )

        return module.register_forward_hook(hook_fn)

    @staticmethod
    def _open_source_use_cache(patches: list[PatchConfig]) -> bool:
        """
        Whether open-source generation can use the KV cache with these patches.

        Patches and zero ablations only change their own positions, so applying them to
        each token once, as the cached pass computes it, gives what recomputing and
        patching the whole sequence every step gives. Mean and noise ablation are
        computed from the whole sequence in each pass, so they need every pass to
        recompute it, like Ozera models do.
        """
        return all(patch.intervention_type in POSITION_LOCAL_INTERVENTIONS for patch in patches)

    def _get_open_source_decoder(self, model_loader: Any):
        """The decoder stack of an open-source model (whose passes PositionOffset tracks)."""
        model = model_loader.model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer
        raise ValueError(f"Unsupported model structure for patching: {type(model).__name__}")

    def _get_open_source_module(self, model_loader: Any, layer: int, activation_key: str):
        """Get the module to hook for an open-source model."""
        model = model_loader.model

        # Try common HuggingFace model structures
        # SmolLM/Llama-style
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if layer >= len(layers):
                return None
            layer_module = layers[layer]

            if activation_key == 'attn_output':
                return layer_module.self_attn if hasattr(layer_module, 'self_attn') else None
            elif activation_key == 'ff_output':
                return layer_module.mlp if hasattr(layer_module, 'mlp') else None
            elif activation_key == 'post_ff':
                return layer_module
            elif activation_key == 'post_attn':
                # This is after attention but before MLP
                return layer_module.post_attention_layernorm if hasattr(layer_module, 'post_attention_layernorm') else None

        # GPT-2 style
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            if layer >= len(layers):
                return None
            layer_module = layers[layer]

            if activation_key == 'attn_output':
                return layer_module.attn if hasattr(layer_module, 'attn') else None
            elif activation_key == 'ff_output':
                return layer_module.mlp if hasattr(layer_module, 'mlp') else None
            elif activation_key in ['post_ff', 'post_attn']:
                return layer_module

        return None

    def _generate_with_patches_ozera(
        self,
        target_prompt: str,
        source: Optional[CapturedActivations],
        patches: list[PatchConfig],
        model_loader: Any,
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, list[int], list[str]]:
        """Apply patches to Ozera model generation."""
        # Get the model
        if hasattr(model_loader, 'model'):
            model = model_loader.model
            device = model_loader.device if hasattr(model_loader, 'device') else 'cpu'
        else:
            model = model_loader
            device = next(model.parameters()).device

        # Encode prompt
        prompt_ids = tokenizer.encode(target_prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        model.eval()

        # Set up patches in model's expected format
        patch_dict = {}
        for patch in patches:
            full_key = patch_activation_key(patch.patch_type, patch.layer)

            # For ablation types, we don't need source activations
            intervention_type = patch.intervention_type if hasattr(patch, 'intervention_type') else 'patch'

            if intervention_type == 'patch':
                # Standard patching requires source activations
                if source is None:
                    raise ValueError(
                        f"Source activations required for patch intervention at layer {patch.layer}. "
                        "Use ablation types (zero_ablate, mean_ablate, noise_ablate) or provide source activations."
                    )
                source_activation = source.activations.get(full_key)
                if source_activation is None:
                    available_keys = [k for k in source.activations.keys() if f"layer_{patch.layer}" in k]
                    raise ValueError(
                        f"Cannot find activation for patch type '{patch.patch_type}' at layer {patch.layer}. "
                        f"Tried key '{full_key}'. Available keys for layer {patch.layer}: {available_keys}"
                    )

                # Check for NaN/inf in source activation
                if torch.isnan(source_activation).any() or torch.isinf(source_activation).any():
                    raise ValueError(
                        f"Source activation for '{full_key}' contains NaN/inf values. "
                        "The model may have produced unstable activations during capture."
                    )

                patch_dict[full_key] = {
                    'source': source_activation.to(device),
                    'positions': patch.positions,
                    'blend_factor': patch.blend_factor,
                    'intervention_type': intervention_type,
                }
            else:
                # Ablation types don't need source activations
                patch_dict[full_key] = {
                    'positions': patch.positions,
                    'blend_factor': patch.blend_factor,
                    'intervention_type': intervention_type,
                }

        # Use the model's built-in generate_with_patches method if available
        if hasattr(model, 'generate_with_patches'):
            with torch.no_grad():
                generated_ids, _ = model.generate_with_patches(
                    input_ids=input_ids,
                    patches=patch_dict,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            tokens = generated_ids[0].cpu().tolist()
            output = tokenizer.decode(tokens)
            decoded = [tokenizer.decode([t]) for t in tokens]
            return output, tokens, decoded

        # Fallback to manual implementation if model doesn't have the method
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if needed
                idx_cond = input_ids if input_ids.size(1) <= model.config.max_seq_len else input_ids[:, -model.config.max_seq_len:]

                # Forward pass with patching using model's method
                if hasattr(model, 'forward_with_patches'):
                    logits, _, _ = model.forward_with_patches(
                        idx_cond, patch_dict, return_attention=False, capture_activations=False
                    )
                else:
                    logits, _, _ = self._forward_with_patches_ozera(
                        model, idx_cond, patch_dict
                    )

                # Get logits for last position
                logits = logits[:, -1, :]

                # Check for NaN/inf in logits (can happen with incompatible patches)
                has_nan = torch.isnan(logits).any()
                all_inf = torch.isinf(logits).all()

                if has_nan or all_inf:
                    # Try to recover: if any valid values, use those; otherwise sample random
                    valid_mask = ~(torch.isnan(logits) | torch.isinf(logits))
                    if valid_mask.any():
                        # Replace invalid values with very negative number
                        logits = torch.where(valid_mask, logits, torch.tensor(-1e10, device=logits.device))
                    else:
                        # Fall back to random sampling if completely invalid
                        next_token = torch.randint(0, tokenizer.vocab_size, (logits.shape[0], 1), device=logits.device)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        continue

                # Clamp logits for numerical stability
                logits = torch.clamp(logits, min=-100, max=100)

                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)

                    # Final safety check for multinomial
                    if torch.isnan(probs).any() or (probs <= 0).all():
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        # Ensure probabilities are valid (positive and sum to 1)
                        probs = torch.clamp(probs, min=1e-10)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)

                # Clamp and append
                next_token = torch.clamp(next_token, 0, tokenizer.vocab_size - 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode
        tokens = input_ids[0].cpu().tolist()
        output = tokenizer.decode(tokens)
        decoded = [tokenizer.decode([t]) for t in tokens]

        return output, tokens, decoded

    def _forward_with_patches_ozera(
        self,
        model: Any,
        input_ids: torch.Tensor,
        patch_dict: dict,
    ) -> tuple:
        """
        Forward pass with patches applied to Ozera model.

        Patches are applied by modifying activations during the forward pass.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        import math
        token_emb = model.token_embedding(input_ids)
        token_emb = token_emb * math.sqrt(model.config.d_model)

        # Positional embeddings
        if model.config.learned_pos_emb:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = model.pos_embedding(positions)
        else:
            pos_emb = model.pos_embedding[:seq_len, :].unsqueeze(0)

        x = token_emb + pos_emb
        x = model.emb_dropout(x)

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        # Process each block
        for layer_idx, block in enumerate(model.blocks):
            # Pre-norm attention
            attn_input = block.ln1(x)

            # Check for attn_input patch
            patch_key = f"layer_{layer_idx}_attn_input"
            if patch_key in patch_dict:
                attn_input = self._apply_patch(attn_input, patch_dict[patch_key])

            attn_output, attn_weights = block.attention(attn_input, causal_mask, return_attention=False)

            # Check for attn_output patch
            patch_key = f"layer_{layer_idx}_attn_output"
            if patch_key in patch_dict:
                attn_output = self._apply_patch(attn_output, patch_dict[patch_key])

            x = x + attn_output

            # Check for post_attn patch
            patch_key = f"layer_{layer_idx}_post_attn"
            if patch_key in patch_dict:
                x = self._apply_patch(x, patch_dict[patch_key])

            # Pre-norm feed-forward
            ff_input = block.ln2(x)

            # Check for ff_input patch
            patch_key = f"layer_{layer_idx}_ff_input"
            if patch_key in patch_dict:
                ff_input = self._apply_patch(ff_input, patch_dict[patch_key])

            ff_output = block.feed_forward(ff_input)

            # Check for ff_output patch
            patch_key = f"layer_{layer_idx}_ff_output"
            if patch_key in patch_dict:
                ff_output = self._apply_patch(ff_output, patch_dict[patch_key])

            x = x + ff_output

            # Check for post_ff patch
            patch_key = f"layer_{layer_idx}_post_ff"
            if patch_key in patch_dict:
                x = self._apply_patch(x, patch_dict[patch_key])

        # Final layer norm
        x = model.ln_f(x)

        # Project to vocabulary
        logits = model.lm_head(x)

        return logits, None, None

    def _apply_patch(self, tensor: torch.Tensor, patch_info: dict) -> torch.Tensor:
        """Apply a patch to a tensor.

        Supports multiple intervention types:
        - 'patch': Replace with source activations (standard activation patching)
        - 'zero_ablate': Zero out activations
        - 'mean_ablate': Replace with mean activation (computed from current tensor)
        - 'noise_ablate': Replace with Gaussian noise matching activation statistics
        """
        intervention_type = patch_info.get('intervention_type', 'patch')
        positions = patch_info.get('positions')
        blend_factor = patch_info.get('blend_factor', 1.0)

        result = tensor.clone()

        if intervention_type == 'zero_ablate':
            # Zero ablation: set activations to zero
            if positions is None:
                if blend_factor == 1.0:
                    result.zero_()
                else:
                    result = (1 - blend_factor) * tensor
            else:
                for pos in positions:
                    if pos < tensor.shape[1]:
                        if blend_factor == 1.0:
                            result[:, pos] = 0.0
                        else:
                            result[:, pos] = (1 - blend_factor) * tensor[:, pos]
            return result

        elif intervention_type == 'mean_ablate':
            # Mean ablation: replace with mean activation
            # Compute mean across sequence dimension for each batch
            mean_activation = tensor.mean(dim=1, keepdim=True)  # [batch, 1, d_model]

            if positions is None:
                if blend_factor == 1.0:
                    result = mean_activation.expand_as(tensor)
                else:
                    result = (1 - blend_factor) * tensor + blend_factor * mean_activation.expand_as(tensor)
            else:
                for pos in positions:
                    if pos < tensor.shape[1]:
                        if blend_factor == 1.0:
                            result[:, pos] = mean_activation.squeeze(1)
                        else:
                            result[:, pos] = (1 - blend_factor) * tensor[:, pos] + blend_factor * mean_activation.squeeze(1)
            return result

        elif intervention_type == 'noise_ablate':
            # Noise ablation: replace with Gaussian noise matching activation statistics
            mean = tensor.mean()
            std = tensor.std()
            noise = torch.randn_like(tensor) * std + mean

            if positions is None:
                if blend_factor == 1.0:
                    result = noise
                else:
                    result = (1 - blend_factor) * tensor + blend_factor * noise
            else:
                for pos in positions:
                    if pos < tensor.shape[1]:
                        if blend_factor == 1.0:
                            result[:, pos] = noise[:, pos]
                        else:
                            result[:, pos] = (1 - blend_factor) * tensor[:, pos] + blend_factor * noise[:, pos]
            return result

        else:
            # Standard patching: replace with source activations
            source = patch_info.get('source')
            if source is None:
                # No source provided, return original tensor unchanged
                return tensor

            # Ensure source is on the same device
            source = source.to(tensor.device)

            # Ensure source matches batch size
            if source.shape[0] != tensor.shape[0]:
                source = source.expand(tensor.shape[0], -1, -1)

            # Check for NaN/inf in source (shouldn't happen, but be safe)
            if torch.isnan(source).any() or torch.isinf(source).any():
                # Replace invalid values with corresponding tensor values
                valid_mask = ~(torch.isnan(source) | torch.isinf(source))
                source = torch.where(valid_mask, source, tensor[:, :source.shape[1], :] if tensor.shape[1] >= source.shape[1] else torch.zeros_like(source))

            if positions is None:
                # Patch all positions
                min_seq = min(tensor.shape[1], source.shape[1])
                if blend_factor == 1.0:
                    result[:, :min_seq] = source[:, :min_seq]
                else:
                    result[:, :min_seq] = (
                        (1 - blend_factor) * tensor[:, :min_seq] +
                        blend_factor * source[:, :min_seq]
                    )
            else:
                for pos in positions:
                    if pos < tensor.shape[1] and pos < source.shape[1]:
                        if blend_factor == 1.0:
                            result[:, pos] = source[:, pos]
                        else:
                            result[:, pos] = (
                                (1 - blend_factor) * tensor[:, pos] +
                                blend_factor * source[:, pos]
                            )

            return result

    def _compute_effect_summary(
        self,
        baseline_tokens: list[int],
        patched_tokens: list[int],
        baseline_decoded: list[str],
        patched_decoded: list[str],
    ) -> dict:
        """Compute summary of patching effects."""
        # Find first divergence
        first_divergence = None
        for i in range(min(len(baseline_tokens), len(patched_tokens))):
            if baseline_tokens[i] != patched_tokens[i]:
                first_divergence = i
                break

        # Count token changes
        min_len = min(len(baseline_tokens), len(patched_tokens))
        token_changes = sum(
            1 for i in range(min_len)
            if baseline_tokens[i] != patched_tokens[i]
        )
        token_changes += abs(len(baseline_tokens) - len(patched_tokens))

        # Find changed token pairs
        changed_tokens = []
        for i in range(min_len):
            if baseline_tokens[i] != patched_tokens[i]:
                changed_tokens.append({
                    'position': i,
                    'baseline_token': baseline_decoded[i] if i < len(baseline_decoded) else '?',
                    'patched_token': patched_decoded[i] if i < len(patched_decoded) else '?',
                })

        return {
            'first_divergence_position': first_divergence,
            'token_changes': token_changes,
            'changed_tokens': changed_tokens[:10],  # Limit to first 10
            'baseline_length': len(baseline_tokens),
            'patched_length': len(patched_tokens),
        }

    # The store is shared by every user of the process. The backend passes user_id to every
    # method below, which then only sees that user's activations; other users' IDs behave as
    # if they don't exist.

    def add_captured_activations(self, captured: CapturedActivations) -> None:
        """
        Keep captured activations under their ID, within the store's bounds.

        Makes room by evicting the least recently used captures: the user's own beyond
        CAPTURES_PER_USER, then anyone's until the store fits CAPTURE_CACHE_MAX_BYTES.

        Raises:
            CaptureTooLargeError: the capture alone is over CAPTURE_MAX_BYTES (nothing is stored)
        """
        # Nothing patches or analyzes logits ([seq, vocab], 262k wide for Gemma)
        captured.activations.pop('logits', None)

        size = _capture_bytes(captured)
        if size > CAPTURE_MAX_BYTES:
            raise CaptureTooLargeError(
                f"These activations are too large to keep ({size / 1024**2:.0f} MB, limit "
                f"{CAPTURE_MAX_BYTES / 1024**2:.0f} MB). Use a shorter prompt or a smaller model."
            )

        # Dicts keep insertion order, and reads move entries to the end: oldest first
        users_captures = [k for k, v in self._captured_activations.items() if v.user_id == captured.user_id]
        for activation_id in users_captures[:max(0, len(users_captures) - CAPTURES_PER_USER + 1)]:
            del self._captured_activations[activation_id]

        total = sum(_capture_bytes(c) for c in self._captured_activations.values())
        while self._captured_activations and total + size > CAPTURE_CACHE_MAX_BYTES:
            oldest_id = next(iter(self._captured_activations))
            total -= _capture_bytes(self._captured_activations.pop(oldest_id))

        self._captured_activations[captured.id] = captured

    def get_captured_activations(
        self, activation_id: str, user_id: Optional[int] = None
    ) -> Optional[CapturedActivations]:
        """Retrieve captured activations by ID (marking them recently used)."""
        captured = self._captured_activations.get(activation_id)
        if captured is None or (user_id is not None and captured.user_id != user_id):
            return None
        self._captured_activations[activation_id] = self._captured_activations.pop(activation_id)
        return captured

    def list_captured_activations(self, user_id: Optional[int] = None) -> list[dict]:
        """List captured activation summaries."""
        return [
            {
                'id': act.id,
                'prompt': act.prompt,
                'model_type': act.model_type,
                'model_id': act.model_id,
                'num_tokens': len(act.tokens),
                'num_layers': act.num_layers,
            }
            for act in self._captured_activations.values()
            if user_id is None or act.user_id == user_id
        ]

    def delete_captured_activations(self, activation_id: str, user_id: Optional[int] = None) -> bool:
        """Delete captured activations."""
        if self.get_captured_activations(activation_id, user_id) is None:
            return False
        del self._captured_activations[activation_id]
        return True

    def clear_all_activations(self, user_id: Optional[int] = None):
        """Clear captured activations (all of them, or only the user's)."""
        if user_id is None:
            self._captured_activations.clear()
            return
        for activation_id in [k for k, v in self._captured_activations.items() if v.user_id == user_id]:
            del self._captured_activations[activation_id]


# Global patching engine instance
_global_engine = None


def get_patching_engine() -> PatchingEngine:
    """Get global patching engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = PatchingEngine()
    return _global_engine
