"""
Core patching engine for activation patching experiments.

Orchestrates capturing source activations, applying patches during generation,
and comparing baseline vs patched outputs.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Any
import os
import torch
import uuid

from core.activation_limits import ActivationLimitError
from .hooks import (
    LayerInput,
    PositionOffset,
    create_attention_patch_hook,
    create_mlp_patch_hook,
    create_residual_patch_hook,
    create_unit_intervention_hook,
    create_zero_ablation_hook,
    residual_stream_hook,
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


# Patch types whose patches can pick attention heads, and MLP neurons
HEAD_PATCH_TYPES = ('attention', 'attn_output')
NEURON_PATCH_TYPES = ('mlp', 'ff_output')


class PatchError(ValueError):
    """A patch can't be applied as requested. The message says why, for the user."""


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


def unit_projection_key(patch: PatchConfig) -> Optional[str]:
    """
    Key of the projection input a patch of chosen heads or neurons changes, e.g.
    'layer_2_attn_heads' (the attention output projection's input) or 'layer_2_mlp_neurons'
    (the MLP down projection's input). None for a patch of a whole activation.
    """
    if patch.heads is not None:
        return f"layer_{patch.layer}_attn_heads"
    if patch.neurons is not None:
        return f"layer_{patch.layer}_mlp_neurons"
    return None


def _index_problem(indices: list[int], count: Optional[int], unit: str) -> Optional[str]:
    if not indices:
        return f"no {unit}s chosen (leave them empty for all {unit}s)"
    bad = [i for i in indices if i < 0 or (count is not None and i >= count)]
    if bad:
        if count is None:
            return f"{unit}s can't be negative"
        return f"the model has {count} {unit}s per layer (0-{count - 1}), so {bad} don't exist"
    return None


def _patch_problem(
    patch: PatchConfig,
    num_layers: int,
    num_heads: int,
    mlp_neurons: Optional[int],
    source_tokens: Optional[int],
    max_tokens: Optional[int],
) -> Optional[str]:
    if not 0 <= patch.layer < num_layers:
        return f"the model has layers 0-{num_layers - 1}"

    if patch.heads is not None and patch.neurons is not None:
        return "choose heads or neurons, not both"
    if patch.heads is not None:
        if patch.patch_type not in HEAD_PATCH_TYPES:
            return "heads can only be chosen for attention patches"
        problem = _index_problem(patch.heads, num_heads, "head")
        if problem:
            return problem
    if patch.neurons is not None:
        if patch.patch_type not in NEURON_PATCH_TYPES:
            return "neurons can only be chosen for MLP patches"
        problem = _index_problem(patch.neurons, mlp_neurons, "neuron")
        if problem:
            return problem

    if patch.positions is not None:
        if not patch.positions:
            return "no positions chosen (leave them empty for all positions)"
        if min(patch.positions) < 0:
            return "positions can't be negative"
        last = max(patch.positions)
        if patch.intervention_type == 'patch' and source_tokens is not None and last >= source_tokens:
            return f"the source prompt only has positions 0-{source_tokens - 1}"
        if max_tokens is not None and last >= max_tokens:
            return f"the sequence only reaches position {max_tokens - 1} (prompt plus max tokens)"
    return None


def validate_patches(
    patches: list[PatchConfig],
    num_layers: int,
    num_heads: int,
    mlp_neurons: Optional[int] = None,
    source_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """
    Check that every patch can be applied as requested, rather than silently skipping it.

    Args:
        patches: The patches
        num_layers: The model's layers
        num_heads: The model's attention heads per layer
        mlp_neurons: The model's neurons per MLP layer, if known
        source_tokens: Tokens in the source prompt, if known (a patch only covers those)
        max_tokens: The most tokens the patched sequence can have, if known

    Raises:
        PatchError: for the first patch that can't be applied
    """
    for number, patch in enumerate(patches, start=1):
        problem = _patch_problem(patch, num_layers, num_heads, mlp_neurons, source_tokens, max_tokens)
        if problem:
            raise PatchError(f"Patch {number} (layer {patch.layer}, {patch.patch_type}): {problem}")


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

            # Positions past the context window have no position embedding
            if input_ids.shape[1] > model.config.max_seq_len:
                raise ActivationLimitError(
                    f"Capturing activations from {model_id} is limited to its context window of "
                    f"{model.config.max_seq_len} tokens; this prompt has {input_ids.shape[1]}."
                )

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

        Raises:
            PatchError: a patch can't be applied as requested (checked before generating)
        """
        # Check if any patches require source activations (patch intervention type)
        requires_source = any(patch.intervention_type == 'patch' for patch in patches)

        source = None
        if source_activation_id:
            source = self._captured_activations.get(source_activation_id)
            if source is None and requires_source:
                raise ValueError(f"Source activations not found: {source_activation_id}")
        elif requires_source:
            raise ValueError("Source activations required for patching interventions. Use ablation types (zero_ablate, mean_ablate, noise_ablate) or provide source activations.")

        # Check every patch can be applied, before generating anything
        if model_type == 'open_source':
            target_tokens = model_loader.encode_prompt(target_prompt).input_ids.shape[1]
            config = model_loader.config
            num_layers, num_heads, mlp_neurons = config.num_layers, config.num_heads, config.intermediate_dim
        else:
            if tokenizer is None:
                raise ValueError("Tokenizer required for Ozera models")
            config = self._ozera_model(model_loader).config
            target_tokens = len(tokenizer.encode(target_prompt))
            num_layers, num_heads, mlp_neurons = config.num_layers, config.num_heads, config.d_ff
            # Past the context window, generation only attends over the last max_seq_len
            # tokens, where the patched positions would no longer be the ones asked for
            if target_tokens + max_new_tokens > config.max_seq_len:
                raise PatchError(
                    f"Patching is limited to the model's context window of {config.max_seq_len} tokens "
                    f"(prompt plus generated). This prompt has {target_tokens}, so generate at most "
                    f"{max(config.max_seq_len - target_tokens, 0)} tokens."
                )
        validate_patches(
            patches,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_neurons=mlp_neurons,
            source_tokens=len(source.tokens) if source is not None else None,
            max_tokens=target_tokens + max_new_tokens,
        )
        unit_sources = self._unit_projection_sources(patches, source, model_loader, model_type)

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
            model = self._ozera_model(model_loader)

            # Tokenize and generate baseline
            prompt_ids = tokenizer.encode(target_prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(next(model.parameters()).device)

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
            unit_sources=unit_sources,
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
        unit_sources: dict[str, torch.Tensor],
    ) -> tuple[str, list[int], list[str]]:
        """Generate with patches applied via hooks."""

        if model_type == 'open_source':
            return self._generate_with_patches_open_source(
                target_prompt, source, patches, model_loader,
                max_new_tokens, temperature, unit_sources,
            )
        else:
            return self._generate_with_patches_ozera(
                target_prompt, source, patches, model_loader,
                tokenizer, max_new_tokens, temperature, unit_sources,
            )

    @staticmethod
    def _ozera_model(model_loader: Any):
        """The TransformerLM behind an Ozera model loader (a TextGenerator, or the model itself)."""
        return model_loader.model if hasattr(model_loader, 'model') else model_loader

    def _unit_projection(self, model_loader: Any, model_type: str, layer: int, kind: str) -> torch.nn.Module:
        """
        The projection whose input a head or neuron patch changes (see unit_projection_key):
        the attention output projection ('attn_heads') or the MLP down projection ('mlp_neurons').
        """
        if model_type == 'open_source':
            layer_module = self._open_source_layer(model_loader, layer)
            return layer_module.self_attn.o_proj if kind == 'attn_heads' else layer_module.mlp.down_proj
        block = self._ozera_model(model_loader).blocks[layer]
        return block.attention.Wo if kind == 'attn_heads' else block.feed_forward.fc2

    def _unit_projection_sources(
        self,
        patches: list[PatchConfig],
        source: Optional[CapturedActivations],
        model_loader: Any,
        model_type: str,
    ) -> dict[str, torch.Tensor]:
        """
        The source prompt's inputs to the projections that head and neuron patches replace.

        Captures don't keep these (the MLP neurons alone are several times the size of the
        rest of a layer's activations), so they're recomputed with one forward pass over the
        source's tokens.
        """
        keys = {
            unit_projection_key(patch)
            for patch in patches
            if patch.intervention_type == 'patch' and unit_projection_key(patch) is not None
        }
        if not keys:
            return {}

        inputs: dict[str, torch.Tensor] = {}

        def recorder(key):
            def hook(module, args):
                inputs[key] = args[0].detach()
            return hook

        handles = []
        try:
            for key in keys:
                _, layer, kind = key.split('_', 2)
                module = self._unit_projection(model_loader, model_type, int(layer), kind)
                handles.append(module.register_forward_pre_hook(recorder(key)))

            with torch.no_grad():
                if model_type == 'open_source':
                    device = next(model_loader.model.parameters()).device
                    input_ids = torch.tensor([source.tokens], dtype=torch.long, device=device)
                    model_loader.model(input_ids, use_cache=False, logits_to_keep=1)
                else:
                    model = self._ozera_model(model_loader)
                    device = next(model.parameters()).device
                    model.forward(torch.tensor([source.tokens], dtype=torch.long, device=device))
        finally:
            for handle in handles:
                handle.remove()
        return inputs

    def _generate_with_patches_open_source(
        self,
        target_prompt: str,
        source: Optional[CapturedActivations],
        patches: list[PatchConfig],
        model_loader: Any,
        max_new_tokens: int,
        temperature: float,
        unit_sources: dict[str, torch.Tensor],
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
                registered_hooks.extend(
                    self._register_open_source_patch_hooks(model_loader, source, patch, offset, unit_sources)
                )

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

    def _register_open_source_patch_hooks(
        self,
        model_loader: Any,
        source: Optional[CapturedActivations],
        patch: PatchConfig,
        offset: PositionOffset,
        unit_sources: dict[str, torch.Tensor],
    ) -> list:
        """
        Register the hooks that apply a patch to an open-source model.

        Returns:
            The hook handles

        Raises:
            PatchError: the patch can't be applied
        """
        intervention_type = patch.intervention_type
        device = next(model_loader.model.parameters()).device

        # Chosen heads or neurons: intervene on their slice of a projection's input
        unit_key = unit_projection_key(patch)
        if unit_key is not None:
            kind = unit_key.split('_', 2)[2]
            projection = self._unit_projection(model_loader, 'open_source', patch.layer, kind)
            hook_fn = create_unit_intervention_hook(
                units=patch.heads if patch.heads is not None else patch.neurons,
                unit_size=projection.in_features // model_loader.config.num_heads if kind == 'attn_heads' else 1,
                intervention_type=intervention_type,
                source_input=unit_sources.get(unit_key),
                positions=patch.positions,
                blend_factor=patch.blend_factor,
                offset=offset,
            )
            return [projection.register_forward_pre_hook(hook_fn)]

        activation_key = PATCH_TYPE_ACTIVATION_KEYS[patch.patch_type]

        if intervention_type in ['zero_ablate', 'mean_ablate', 'noise_ablate']:
            hook_fn = create_zero_ablation_hook(
                positions=patch.positions,
                blend_factor=patch.blend_factor,
                ablation_type=intervention_type,
                offset=offset,
            )
        else:
            full_key = f"layer_{patch.layer}_{activation_key}"
            source_activation = source.activations.get(full_key) if source is not None else None
            if source_activation is None:
                raise PatchError(
                    f"The source activations have no '{full_key}'. Capture the source prompt again."
                )
            # Move the source to the model's device once, rather than in every forward pass
            source_activation = source_activation.to(device)

            if activation_key == 'attn_output':
                hook_fn = create_attention_patch_hook(
                    source_attention=source_activation,
                    positions=patch.positions,
                    blend_factor=patch.blend_factor,
                    offset=offset,
                )
            elif activation_key == 'ff_output':
                hook_fn = create_mlp_patch_hook(
                    source_mlp_output=source_activation,
                    positions=patch.positions,
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

        layer_module = self._open_source_layer(model_loader, patch.layer)
        if activation_key == 'post_attn':
            # No module outputs the residual stream after attention: the layer adds its
            # attention branch's output to its input, so the patch changes that branch's output.
            # The branch ends at self_attn, or (Gemma, which normalizes the attention output
            # and has a separate pre-FFN norm) at post_attention_layernorm.
            layer_input = LayerInput()
            branch_end = (
                layer_module.post_attention_layernorm
                if hasattr(layer_module, 'pre_feedforward_layernorm')
                else layer_module.self_attn
            )
            return [
                layer_input.attach(layer_module),
                branch_end.register_forward_hook(residual_stream_hook(hook_fn, layer_input)),
            ]

        module = {
            'attn_output': layer_module.self_attn,
            'ff_output': layer_module.mlp,
            'post_ff': layer_module,
        }[activation_key]
        return [module.register_forward_hook(hook_fn)]

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

    def _open_source_layer(self, model_loader: Any, layer: int) -> torch.nn.Module:
        """A decoder layer of an open-source model (Llama, Qwen and Gemma layouts)."""
        model = model_loader.model
        if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
            raise PatchError(f"Patching isn't supported for {type(model).__name__} models")
        return model.model.layers[layer]

    def _generate_with_patches_ozera(
        self,
        target_prompt: str,
        source: Optional[CapturedActivations],
        patches: list[PatchConfig],
        model_loader: Any,
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
        unit_sources: dict[str, torch.Tensor],
    ) -> tuple[str, list[int], list[str]]:
        """
        Apply patches to Ozera model generation.

        Patches of whole activations go to the model's generate_with_patches (a list per
        activation, applied in order). Patches of chosen heads or neurons are hooks on the
        attention output and MLP down projections. The model recomputes the whole sequence
        every step, so positions are absolute.
        """
        model = self._ozera_model(model_loader)
        device = next(model.parameters()).device

        # Encode prompt
        prompt_ids = tokenizer.encode(target_prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        model.eval()

        # Set up patches in model's expected format
        patch_dict: dict[str, list[dict]] = {}
        unit_hooks = []
        for patch in patches:
            intervention_type = patch.intervention_type

            unit_key = unit_projection_key(patch)
            if unit_key is not None:
                kind = unit_key.split('_', 2)[2]
                hook_fn = create_unit_intervention_hook(
                    units=patch.heads if patch.heads is not None else patch.neurons,
                    unit_size=model.config.d_k if kind == 'attn_heads' else 1,
                    intervention_type=intervention_type,
                    source_input=unit_sources.get(unit_key),
                    positions=patch.positions,
                    blend_factor=patch.blend_factor,
                )
                unit_hooks.append((self._unit_projection(model, 'ozera', patch.layer, kind), hook_fn))
                continue

            full_key = patch_activation_key(patch.patch_type, patch.layer)
            patch_info = {
                'positions': patch.positions,
                'blend_factor': patch.blend_factor,
                'intervention_type': intervention_type,
            }

            if intervention_type == 'patch':
                # Standard patching requires source activations
                source_activation = source.activations.get(full_key) if source is not None else None
                if source_activation is None:
                    raise PatchError(
                        f"The source activations have no '{full_key}'. Capture the source prompt again."
                    )

                # Check for NaN/inf in source activation
                if torch.isnan(source_activation).any() or torch.isinf(source_activation).any():
                    raise PatchError(
                        f"Source activation for '{full_key}' contains NaN/inf values. "
                        "The model may have produced unstable activations during capture."
                    )

                patch_info['source'] = source_activation.to(device)

            patch_dict.setdefault(full_key, []).append(patch_info)

        handles = [projection.register_forward_pre_hook(hook_fn) for projection, hook_fn in unit_hooks]
        try:
            with torch.no_grad():
                generated_ids, _ = model.generate_with_patches(
                    input_ids=input_ids,
                    patches=patch_dict,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
        finally:
            for handle in handles:
                handle.remove()

        tokens = generated_ids[0].cpu().tolist()
        output = tokenizer.decode(tokens)
        decoded = [tokenizer.decode([t]) for t in tokens]
        return output, tokens, decoded

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
