"""
Direct logit attribution: how much each part of the model wrote toward a token's logit.

The residual stream a position ends with is a sum: the embeddings, plus every layer's attention
output and MLP output (and an attention output is itself a sum over heads). The logit is read
from that sum through the final norm and the token's unembedding. With the final norm's scale
frozen at the value it took (see norms.py), that reading is linear, so each part has a share
of the logit and the shares add up to it, less what the norm's and unembedding's biases add.

This is the direct effect only: what a part wrote that the unembedding reads. What it did by
changing later layers' inputs is in those layers' shares.
"""

import torch

from .model import Capture, LensModel
from .norms import LensError


def attribute(model: LensModel, capture: Capture, target_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Each part's share of the target token's logit at every position.

    Args:
        target_ids: [seq] the token whose logit is attributed at each position

    Returns:
        CPU float32 tensors: "embed" [seq]; "attn" and "mlp" [seq, layers]; "heads"
        [seq, layers, heads]; "attn_bias" [seq, layers] (the attention output projection's
        bias, the part of "attn" no head wrote); "bias" [seq] (the final norm's and
        unembedding's biases); "logit" [seq] the model's own logit, which embed + attn + mlp
        + bias add up to

    Raises:
        LensError: the model caps its logits, so they aren't a linear readout
    """
    if model.softcap:
        raise LensError("Logit attribution doesn't support models that cap their logits")

    target_ids = target_ids.to(capture.logits.device)
    seq_len = len(capture.token_ids)
    final = capture.resid_post[-1]

    with torch.no_grad():
        unembed = model.unembedding[target_ids]  # [seq, d]
        # The direction in the residual stream each position's logit is read along
        directions = model.final_norm.pull_back(unembed, final)
        bias = model.final_norm.bias_share(unembed)
        if model.unembedding_bias is not None:
            bias = bias + model.unembedding_bias[target_ids].float()

        def share(part: torch.Tensor, along: torch.Tensor) -> torch.Tensor:
            return (part.float() * along).sum(dim=-1)

        attn, mlp, heads, attn_bias = [], [], [], []
        for index, layer in enumerate(model.layers):
            attn_out, mlp_out = capture.attn_out[index], capture.mlp_out[index]

            # A layer that normalizes a branch before adding it: frozen the same way
            along = directions if layer.attn_norm is None else layer.attn_norm.pull_back(directions, attn_out)
            total = share(attn_out, along)
            if layer.attn_norm is not None:
                total = total + layer.attn_norm.bias_share(directions)
            attn.append(total)

            # Heads: the output projection's input holds them side by side
            through_projection = along @ layer.out_proj.float()  # [seq, heads * head_dim]
            per_head = capture.head_outputs[index].float() * through_projection
            heads.append(per_head.view(seq_len, model.num_heads, -1).sum(dim=-1))
            attn_bias.append(
                along @ layer.out_proj_bias.float() if layer.out_proj_bias is not None
                else torch.zeros(seq_len, device=along.device)
            )

            along = directions if layer.mlp_norm is None else layer.mlp_norm.pull_back(directions, mlp_out)
            total = share(mlp_out, along)
            if layer.mlp_norm is not None:
                total = total + layer.mlp_norm.bias_share(directions)
            mlp.append(total)

        out = {
            "embed": share(capture.embeddings, directions),
            "attn": torch.stack(attn, dim=1),
            "mlp": torch.stack(mlp, dim=1),
            "heads": torch.stack(heads, dim=1),
            "attn_bias": torch.stack(attn_bias, dim=1),
            "bias": bias,
            "logit": capture.logits.gather(1, target_ids[:, None])[:, 0],
        }
    return {key: value.float().cpu() for key, value in out.items()}
