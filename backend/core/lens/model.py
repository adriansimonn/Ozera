"""
A model and one forward pass through it, as the logit lens and logit attribution read them.

LensModel is what both need from a model: how it turns the residual stream into logits (its
final norm and unembedding), and for each layer, how the attention output splits into heads
and whether the layer normalizes a branch before adding it to the stream (Gemma does).

Capture is one forward pass over a sequence: the residual stream entering the first layer,
what each layer's attention and MLP wrote, each layer's output, and the model's logits.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from .norms import FrozenNorm, LensError, frozen_norm


@dataclass
class LayerParts:
    """
    One decoder layer, for attribution.

    Attributes:
        out_proj: [d, heads * head_dim] the attention output projection, whose input holds the
            heads' outputs side by side
        out_proj_bias: [d] its bias (part of the attention output that no head wrote), or None
        attn_norm, mlp_norm: The norm the layer applies to a branch's output before adding it
            to the residual stream, or None when the output is added as it is
    """

    out_proj: torch.Tensor
    out_proj_bias: Optional[torch.Tensor] = None
    attn_norm: Optional[FrozenNorm] = None
    mlp_norm: Optional[FrozenNorm] = None


class LensModel:
    """A model's final norm, unembedding and layers (see the module docstring)."""

    def __init__(
        self,
        final_norm: torch.nn.Module,
        lm_head: torch.nn.Module,
        layers: Sequence[LayerParts],
        num_heads: int,
        token_text: Callable[[int], str],
        softcap: Optional[float] = None,
    ):
        """
        Args:
            final_norm: The norm before the unembedding
            lm_head: The unembedding (a linear layer)
            layers: Every decoder layer, in order
            num_heads: Attention heads per layer
            token_text: Decodes one token ID
            softcap: The model's final logit soft cap (logits = tanh(logits / cap) * cap), if any
        """
        self._final_norm = final_norm
        self._lm_head = lm_head
        self.final_norm = frozen_norm(final_norm)
        self.layers = list(layers)
        self.num_heads = num_heads
        self.token_text = token_text
        self.softcap = softcap

    @property
    def unembedding(self) -> torch.Tensor:
        """[vocab, d]"""
        return self._lm_head.weight

    @property
    def unembedding_bias(self) -> Optional[torch.Tensor]:
        return getattr(self._lm_head, "bias", None)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits [n, vocab] the model's own final norm and unembedding give for hidden states [n, d]."""
        logits = self._lm_head(self._final_norm(hidden))
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    @classmethod
    def for_ozera(cls, model, tokenizer) -> "LensModel":
        """An Ozera model (core.transformer.model_torch.TransformerLM)."""
        return cls(
            final_norm=model.ln_f,
            lm_head=model.lm_head,
            layers=[
                LayerParts(out_proj=block.attention.Wo.weight, out_proj_bias=block.attention.Wo.bias)
                for block in model.blocks
            ],
            num_heads=model.config.num_heads,
            token_text=lambda token_id: tokenizer.decode([token_id]),
        )

    @classmethod
    def for_open_source(cls, loader) -> "LensModel":
        """An open-source model, through its loader (core.open_source)."""
        model = loader.model
        layers = []
        for layer in model.model.layers:
            # Gemma normalizes each branch's output before the residual connection; Llama and
            # Qwen layers have no such norms (their post_attention_layernorm is the pre-MLP norm)
            sandwich = hasattr(layer, "pre_feedforward_layernorm")
            layers.append(LayerParts(
                out_proj=layer.self_attn.o_proj.weight,
                out_proj_bias=layer.self_attn.o_proj.bias,
                attn_norm=frozen_norm(layer.post_attention_layernorm) if sandwich else None,
                mlp_norm=frozen_norm(layer.post_feedforward_layernorm) if sandwich else None,
            ))
        return cls(
            final_norm=model.model.norm,
            lm_head=model.lm_head,
            layers=layers,
            num_heads=loader.config.num_heads,
            token_text=lambda token_id: loader.tokenizer.decode([token_id]),
            softcap=getattr(model.config, "final_logit_softcapping", None),
        )


@dataclass
class Capture:
    """
    One forward pass over a sequence. Tensors are on the model's device, in its dtype.

    Attributes:
        token_ids: The sequence
        embeddings: [seq, d] the residual stream entering the first layer
        attn_out, mlp_out: Per layer, [seq, d] what its attention and MLP output (before any
            norm the layer applies to the branch)
        resid_post: Per layer, [seq, d] the residual stream after it
        head_outputs: Per layer, [seq, heads * head_dim] the attention output projection's input
        logits: [seq, vocab] the model's logits
    """

    token_ids: list[int]
    embeddings: torch.Tensor
    attn_out: list[torch.Tensor]
    mlp_out: list[torch.Tensor]
    resid_post: list[torch.Tensor]
    head_outputs: list[torch.Tensor]
    logits: torch.Tensor

    @classmethod
    def from_ozera(cls, model, activations: dict, logits: torch.Tensor, token_ids: Sequence[int]) -> "Capture":
        """
        From TransformerLM.forward(capture_activations=True)'s activations.

        The forward pass doesn't keep the heads' outputs before the output projection; they're
        recomputed from each layer's attention input and weights, which it does keep. (No
        hooks: other requests may be running the same model.)
        """
        layers = activations["layers"]
        head_outputs = []
        with torch.no_grad():
            for block, layer in zip(model.blocks, layers):
                attention = block.attention
                attn_input, weights = layer["attn_input"], layer["attn_weights"]
                batch, seq, width = attn_input.shape
                values = attention.Wv(attn_input).view(batch, seq, attention.num_heads, attention.d_k).transpose(1, 2)
                heads = torch.matmul(weights, values).transpose(1, 2).contiguous().view(batch, seq, width)
                head_outputs.append(heads[0])
        return cls(
            token_ids=list(token_ids),
            embeddings=activations["combined_embeddings"][0],
            attn_out=[layer["attn_output"][0] for layer in layers],
            mlp_out=[layer["ff_output"][0] for layer in layers],
            resid_post=[layer["post_ff"][0] for layer in layers],
            head_outputs=head_outputs,
            logits=logits[0] if logits.dim() == 3 else logits,
        )

    @classmethod
    def from_open_source(cls, activations: dict, num_layers: int, token_ids: Sequence[int]) -> "Capture":
        """From an open-source loader's capture pass run with readout_inputs (see _capture_forward)."""
        missing = [
            key for key in (
                ["resid_pre", "logits"]
                + [f"layer_{i}_{name}" for i in range(num_layers) for name in ("attn_output", "ff_output", "post_ff", "attn_heads")]
            ) if key not in activations
        ]
        if missing:
            raise LensError(f"The capture has no {missing[0]}")
        logits = activations["logits"]
        return cls(
            token_ids=list(token_ids),
            embeddings=activations["resid_pre"][0],
            attn_out=[activations[f"layer_{i}_attn_output"][0] for i in range(num_layers)],
            mlp_out=[activations[f"layer_{i}_ff_output"][0] for i in range(num_layers)],
            resid_post=[activations[f"layer_{i}_post_ff"][0] for i in range(num_layers)],
            head_outputs=[activations[f"layer_{i}_attn_heads"][0] for i in range(num_layers)],
            logits=logits[0] if logits.dim() == 3 else logits,
        )
