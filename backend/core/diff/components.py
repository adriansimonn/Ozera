"""
The directions two models differ along at one residual stream position, and whether the base
model uses them.

For every compared token, d = (the other model's activation) − (the base model's). The
components are the principal directions of d on one half of the texts, without centering:
each one's share is its share of the mean squared difference, so a constant offset between
the models shows up as a component of its own (usually the first) instead of being
subtracted away. Each component points the way the other model differs on average. Its
tokens are the ones whose difference lies furthest along it, at either end.

Tokens with massive activations at the position (see budget.MASSIVE_NORM) are left out of
the decomposition and get no edit.

The test, on the other half of the texts: the base model is run with each token's own
difference along a component added at that position, a' = a + (d·v)v, and its next-token
predictions are compared with the other model's. A component the models' behaviour runs
through closes part of the KL between them. Any edit of that size moves predictions, so each
component is compared with random directions carrying the same per-token amounts,
a' = a + (d·v)u. Two references bound the result: the base model unchanged, and with the
whole difference added at every position (a' = b, the tokenizer's prefix included), which
leaves only what the later layers do differently.
"""

from typing import Optional, Sequence

import torch

from core.probes.capture import ResidualReader
from core.probes.runner import ProbeTarget, forward_batches

from .budget import (
    COMPONENTS,
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    CONTROL_DIRECTIONS,
    EXTREME_TOKENS,
    KL_CHUNK,
    TESTED_COMPONENTS,
    TOKENS_PER_BATCH,
    DiffInputError,
)
from .edit import StreamEdit
from .profile import batch_token_index, encoded, ordinary_tokens, span_mask, token_offsets
from .texts import DiffText

# Below this KL between the two models (nats per token) there is nothing to close
MIN_KL = 1e-6


def difference_components(differences: torch.Tensor, count: int = COMPONENTS) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    The principal directions of differences [n, d], uncentered.

    Returns:
        (components [k, d] unit rows, largest first, each pointing the way the mean difference
        does; each one's share [k] of the mean squared difference; the mean squared difference)
    """
    d = differences.float()
    moment = d.T @ d / d.shape[0]
    total = float(moment.trace())
    if total <= 0:
        raise DiffInputError("The two models' activations are identical at this layer: there is no difference to decompose")
    values, vectors = torch.linalg.eigh(moment.double())
    k = min(count, vectors.shape[1])
    components = vectors[:, -k:].flip(dims=[1]).T.float()
    shares = (values[-k:].flip(dims=[0]).clamp_min(0) / values.clamp_min(0).sum()).float()
    signs = torch.where((d @ components.T).mean(dim=0) < 0, -1.0, 1.0)
    return components * signs[:, None], shares, total


def control_directions(component: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """Random unit directions [count, d] orthogonal to a unit component."""
    random = torch.randn(count, component.shape[0], generator=generator).to(component.device)
    random = random - (random @ component)[:, None] * component
    return random / random.norm(dim=1, keepdim=True)


def _contexts(texts: Sequence[DiffText], target: ProbeTarget, offsets: Sequence[int], values: torch.Tensor, order: Sequence[int]) -> list[dict]:
    """The tokens at some global indices, each with the tokens around it and its value."""
    out = []
    for index in order:
        text = max(i for i, offset in enumerate(offsets) if offset <= index)
        position = texts[text].start + index - offsets[text]
        ids = texts[text].ids
        first = max(texts[text].lead, position - CONTEXT_BEFORE)
        out.append({
            "text": texts[text].source,
            "position": position,
            "value": round(float(values[index]), 4),
            "before": [target.token_text(i) for i in ids[first:position]],
            "token": target.token_text(ids[position]),
            "after": [target.token_text(i) for i in ids[position + 1:position + 1 + CONTEXT_AFTER]],
        })
    return out


@torch.no_grad()
def run_directions(
    base: ProbeTarget,
    other: ProbeTarget,
    texts: Sequence[DiffText],
    position: int,
    seed: int = 0,
    components: int = COMPONENTS,
    tested: int = TESTED_COMPONENTS,
    controls: int = CONTROL_DIRECTIONS,
    tokens_per_batch: int = TOKENS_PER_BATCH,
) -> dict:
    """
    Decompose the difference between two models at a residual stream position, and test its
    largest components. Caller holds both targets' locks.

    Args:
        position: Residual stream position (0: the embeddings, i: layer i-1's output)
        seed: Seed of the control directions

    Returns:
        "layer" (the position), "num_layers", "hidden_dim"; "fit_texts", "test_texts",
        "fit_tokens", "test_tokens", "total_tokens" (tokens one model read), "massive_tokens"
        (left out for their massive activations);
        "relative_difference" (root mean squared ‖d‖ over root mean squared ‖a‖, on the fitted
        half), "mean_share" (the share of the mean squared difference that is the same for
        every token);
        "components": [{"index" (from 1), "share", "mean_cosine" (its cosine with the mean
        difference), "high", "low": tokens at its ends (see _contexts)}];
        "test": {"kl_base" (KL(other ‖ base) per token on the held-out half), "kl_full" (with
        the whole difference added), "agreement_base", "agreement_full" (shares of tokens
        whose top prediction is the other model's), "components": [{"index", "kl", "closed"
        (1 − kl / kl_base), "agreement", "controls": [{"kl", "closed"}]}], "joint": {"count",
        "kl", "closed", "agreement"} for the tested components together}; "closed" is None
        when the models already agree

    Raises:
        DiffInputError: the position doesn't exist, or the models are identical there
    """
    num_layers = len(base.layers)
    if not 0 <= position <= num_layers:
        raise DiffInputError(f"The models have residual stream positions 0-{num_layers}")
    device = base.device
    offsets = token_offsets(texts)
    count = sum(text.compared for text in texts)
    fit = [i for i in range(len(texts)) if i % 2 == 0]
    test = [i for i in range(len(texts)) if i % 2 == 1]

    # Every token's difference at the position, and the base model's squared norm there
    differences = torch.zeros(count, base.hidden_dim, device=device)
    square_norm = torch.zeros(count, device=device)
    ordinary = torch.zeros(count, dtype=torch.bool, device=device)
    for batch in forward_batches(base, encoded(texts, range(len(texts))), tokens_per_batch):
        rows = batch.idx.tolist()
        in_span = span_mask(batch)
        index = batch_token_index(texts, offsets, rows, device)
        read: dict[str, torch.Tensor] = {}
        for name, target in (("base", base), ("other", other)):
            with ResidualReader(target.layers, [position], lambda _, hidden, name=name: read.update({name: hidden[in_span].float()})) as reader:
                reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
        differences[index] = read["other"] - read["base"]
        square_norm[index] = read["base"].pow(2).sum(dim=1)
        ordinary[index] = ordinary_tokens(read["base"], read["other"])
    # A massive token has no part in the components, and gets no edit
    differences[~ordinary] = 0

    in_fit = torch.zeros(count, dtype=torch.bool, device=device)
    for i in fit:
        in_fit[offsets[i]:offsets[i] + texts[i].compared] = True
    fit_tokens = int(in_fit.sum())
    in_fit &= ordinary
    if int(in_fit.sum()) < 2:
        raise DiffInputError("Too few tokens to decompose the difference; use more texts")
    vectors, shares, mean_square = difference_components(differences[in_fit], components)
    mean = differences[in_fit].mean(dim=0)
    mean_norm = float(mean.norm())
    projections = differences @ vectors.T  # [tokens, k]
    listed = ordinary.nonzero()[:, 0]

    described = []
    for k in range(vectors.shape[0]):
        ordered = listed[torch.argsort(projections[listed, k], descending=True)]
        described.append({
            "index": k + 1,
            "share": round(float(shares[k]), 5),
            "mean_cosine": round(float(vectors[k] @ mean) / mean_norm, 4) if mean_norm > 0 else None,
            "high": _contexts(texts, base, offsets, projections[:, k], ordered[:EXTREME_TOKENS].tolist()),
            "low": _contexts(texts, base, offsets, projections[:, k], ordered.flip(dims=[0])[:EXTREME_TOKENS].tolist()),
        })

    # The test: what the base model predicts on the held-out half with each edit
    tested = min(tested, vectors.shape[0])
    generator = torch.Generator().manual_seed(seed)
    randoms = [control_directions(vectors[k], controls, generator) for k in range(tested)]
    conditions: list[tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]] = [("base", None, None), ("full", None, None)]
    for k in range(tested):
        conditions.append((f"component_{k}", vectors[k:k + 1], vectors[k:k + 1]))
        for j in range(controls):
            conditions.append((f"control_{k}_{j}", vectors[k:k + 1], randoms[k][j:j + 1]))
    conditions.append(("joint", vectors[:tested], vectors[:tested]))

    kl = {name: 0.0 for name, _, _ in conditions}
    agree = {name: 0 for name, _, _ in conditions}
    compared = 0
    final = num_layers
    with StreamEdit(base, position) as edit:
        for batch in forward_batches(base, encoded(texts, test), tokens_per_batch):
            rows = [test[i] for i in batch.idx.tolist()]
            in_span = span_mask(batch)
            index = batch_token_index(texts, offsets, rows, device)
            batch_differences = differences[index]
            read: dict[str, torch.Tensor] = {}

            def keep_other(at: int, hidden: torch.Tensor) -> None:
                if at == position:
                    read["other_at"] = hidden.float()
                if at == final:
                    read["other"] = hidden[in_span]

            def keep_base(at: int, hidden: torch.Tensor) -> None:
                # The unchanged pass's stream at the position (later passes read it edited)
                if at == position and edit.delta is None:
                    read["base_at"] = hidden.float()
                if at == final:
                    read["base"] = hidden[in_span]

            with ResidualReader(other.layers, [position, final], keep_other) as reader:
                reader.run(lambda: other.forward(batch.input_ids, batch.attention_mask))
            finals = {}
            with ResidualReader(base.layers, [position, final], keep_base) as reader:
                for name, reads, writes in conditions:
                    if name == "base":
                        edit.delta = None
                    elif name == "full":
                        # Every position, the tokenizer's prefix too: the other model's stream itself
                        edit.delta = read["other_at"] - read["base_at"]
                    else:
                        delta = torch.zeros(*in_span.shape, base.hidden_dim, device=device)
                        delta[in_span] = (batch_differences @ reads.T) @ writes
                        edit.delta = delta
                    reader.run(lambda: base.forward(batch.input_ids, batch.attention_mask))
                    finals[name] = read.pop("base")
                edit.delta = None
            for start in range(0, len(index), KL_CHUNK):
                chunk = slice(start, start + KL_CHUNK)
                log_other = torch.log_softmax(other.logits(read["other"][chunk]).float(), dim=-1)
                top_other = log_other.argmax(dim=-1)
                for name, _, _ in conditions:
                    log_base = torch.log_softmax(base.logits(finals[name][chunk]).float(), dim=-1)
                    kl[name] += float((log_other.exp() * (log_other - log_base)).sum())
                    agree[name] += int((log_base.argmax(dim=-1) == top_other).sum())
            compared += len(index)

    def mean_kl(name: str) -> float:
        return max(kl[name] / max(compared, 1), 0.0)

    kl_base = mean_kl("base")

    def closed(name: str) -> Optional[float]:
        return round(1 - mean_kl(name) / kl_base, 4) if kl_base > MIN_KL else None

    def result(name: str) -> dict:
        return {"kl": round(mean_kl(name), 6), "closed": closed(name), "agreement": round(agree[name] / max(compared, 1), 4)}

    return {
        "layer": position,
        "num_layers": num_layers,
        "hidden_dim": base.hidden_dim,
        "fit_texts": len(fit),
        "test_texts": len(test),
        "fit_tokens": fit_tokens,
        "test_tokens": compared,
        "total_tokens": sum(len(text.ids) for text in texts),
        "massive_tokens": int((~ordinary).sum()),
        "relative_difference": round((mean_square / max(float(square_norm[in_fit].mean()), 1e-30)) ** 0.5, 5),
        "mean_share": round(mean_norm ** 2 / mean_square, 5),
        "components": described,
        "test": {
            "kl_base": round(kl_base, 6),
            "kl_full": round(mean_kl("full"), 6),
            "agreement_base": round(agree["base"] / max(compared, 1), 4),
            "agreement_full": round(agree["full"] / max(compared, 1), 4),
            "full_closed": closed("full"),
            "components": [
                {"index": k + 1, **result(f"component_{k}"),
                 "controls": [{"kl": round(mean_kl(f"control_{k}_{j}"), 6), "closed": closed(f"control_{k}_{j}")} for j in range(controls)]}
                for k in range(tested)
            ],
            "joint": {"count": tested, **result("joint")},
        },
    }
