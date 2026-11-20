"""
Patching one model's activations into the other: exact activation patching
(core.attribution.patching) across two models of the same shape, on one prompt.

The donor's activation at one layer and position replaces the recipient's there, the
recipient runs on, and its next-token prediction at the prompt's last position is compared
with the donor's: KL(donor ‖ patched recipient). Unpatched, that KL is what separates the two
models on this prompt; a patch that brings it down carries part of what the donor does
differently. A layer is also patched at every position at once, which runs the donor up to
that layer and the recipient from there on.

What's patched is the residual stream entering a layer (layer 0: the embeddings), or a
layer's attention or MLP output: the sites of core.attribution.taps.
"""

from typing import Sequence

import torch

from core.attribution.metric import last_logits
from core.attribution.taps import Taps
from core.patching.hooks import create_masked_replacement_hook
from core.probes.runner import ProbeTarget

from .budget import PATCH_KINDS, TOKENS_PER_BATCH, DiffInputError

# Next-token predictions listed for each model
TOP_PREDICTIONS = 5


def _predictions(target: ProbeTarget, log_probs: torch.Tensor) -> list[dict]:
    probs, ids = log_probs.exp().topk(min(TOP_PREDICTIONS, log_probs.shape[-1]))
    return [{"token": target.token_text(int(i)), "prob": round(float(p), 4)} for p, i in zip(probs, ids)]


def run_patch(
    donor: ProbeTarget,
    recipient: ProbeTarget,
    ids: Sequence[int],
    kind: str = "resid",
    tokens_per_batch: int = TOKENS_PER_BATCH,
) -> dict:
    """
    Patch a donor model's activations into a recipient model, one layer and position at a
    time, on one prompt. Caller holds both targets' locks.

    Args:
        ids: The prompt's token IDs, as both models read them
        kind: "resid", "attn" or "mlp"

    Returns:
        "kind", "num_layers", "tokens" (the prompt's), "kl_base" (KL(donor ‖ recipient) at the
        last position, nats), "grid": [layers][tokens] the KL with that one activation
        patched, "all_positions": [layers] with a layer patched at every position,
        "predictions": {"donor", "recipient": [{"token", "prob"}]}, "token_passes"

    Raises:
        DiffInputError: an unknown kind
    """
    if kind not in PATCH_KINDS:
        raise DiffInputError(f"What's patched is one of: {', '.join(PATCH_KINDS)}")
    num_layers = len(donor.layers)
    device = donor.device
    tokens = len(ids)
    final = ("resid", num_layers)
    sites = [(kind, layer) for layer in range(num_layers)]
    prompt = torch.tensor([list(ids)], device=device)
    mask = torch.ones_like(prompt)
    lengths = torch.tensor([tokens], device=device)

    with Taps(donor, read=sites + [final]) as taps, torch.no_grad():
        taps.run(lambda: donor.forward(prompt, mask))
        sources = {site: taps.acts[site].detach() for site in sites}
        log_donor = torch.log_softmax(last_logits(donor, taps.acts[final], lengths), dim=-1)[0]

    def kl(logits: torch.Tensor) -> torch.Tensor:
        """KL(donor ‖ each row's prediction) for rows of next-token logits."""
        return (log_donor.exp() * (log_donor - torch.log_softmax(logits, dim=-1))).sum(dim=-1).clamp_min(0)

    # A row per patch: the prompt with one position patched, and a last row with all of them
    rows = tokens + 1
    per_batch = max(1, tokens_per_batch // tokens)
    grid = torch.zeros(num_layers, tokens)
    all_positions = torch.zeros(num_layers)
    with Taps(recipient, read=[final], replace_at=sites) as taps, torch.no_grad():
        taps.run(lambda: recipient.forward(prompt, mask))
        log_recipient = torch.log_softmax(last_logits(recipient, taps.acts[final], lengths), dim=-1)[0]
        kl_base = float(kl(log_recipient[None])[0])

        for layer, site in enumerate(sites):
            values = []
            for start in range(0, rows, per_batch):
                chunk = range(start, min(start + per_batch, rows))
                patched = torch.zeros(len(chunk), tokens, 1, dtype=torch.bool, device=device)
                for row, index in enumerate(chunk):
                    patched[row, slice(None) if index == tokens else index] = True
                taps.replace = {site: create_masked_replacement_hook(sources[site].expand(len(chunk), -1, -1), patched)}
                taps.run(lambda: recipient.forward(prompt.expand(len(chunk), -1), mask.expand(len(chunk), -1)))
                values.append(kl(last_logits(recipient, taps.acts[final], lengths.expand(len(chunk)))).cpu())
            taps.replace = {}
            values = torch.cat(values)
            grid[layer] = values[:tokens]
            all_positions[layer] = values[tokens]

    return {
        "kind": kind,
        "num_layers": num_layers,
        "tokens": [donor.token_text(i) for i in ids],
        "kl_base": round(kl_base, 6),
        "grid": [[round(v, 6) for v in row] for row in grid.tolist()],
        "all_positions": [round(v, 6) for v in all_positions.tolist()],
        "predictions": {"donor": _predictions(donor, log_donor), "recipient": _predictions(recipient, log_recipient)},
        "token_passes": 2 * tokens + num_layers * rows * tokens,
    }
