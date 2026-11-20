"""
Where two models part: their residual streams compared at every position on the same tokens,
and their next-token predictions compared at every token.

One pass of each model over the texts. Per residual stream position (0: the embeddings, i:
layer i-1's output), over every compared token: the relative distance between the two
models' activations, their mean cosine, linear CKA and Procrustes R² (see similarity.py).
Per token: KL(other ‖ base) between the two models' next-token distributions, and each
model's top prediction.

Tokens with massive activations at a position (see budget.MASSIVE_NORM) are left out of that
position's statistics, and counted.

The texts are split in two halves (every other text). Each measure is also read on each half
alone, so a layer profile can be told from an accident of the sample: where the halves
disagree, the profile says more about the texts than about the models. (Linear CKA and
Procrustes have halves only on models narrow enough to keep their second moments twice: see
budget.similarity_halves.)
"""

from typing import Optional, Sequence

import torch

from core.probes.capture import ResidualReader
from core.probes.runner import EncodedText, ProbeTarget, forward_batches

from .budget import KL_CHUNK, MASSIVE_DEVIATIONS, MASSIVE_NORM, MIN_NORM_TOKENS, TOKENS_PER_BATCH, similarity_halves
from .similarity import StreamSimilarity
from .texts import DiffText


def span_mask(batch) -> torch.Tensor:
    """[batch, seq] bool: each row's compared positions."""
    seq = torch.arange(batch.input_ids.shape[1], device=batch.input_ids.device)
    return (seq[None, :] >= batch.starts[:, None]) & (seq[None, :] < batch.ends[:, None])


def encoded(texts: Sequence[DiffText], members: Sequence[int]) -> list[EncodedText]:
    return [EncodedText(texts[i].ids, texts[i].start, len(texts[i].ids)) for i in members]


def token_offsets(texts: Sequence[DiffText]) -> list[int]:
    """Where each text's compared tokens start among all the compared tokens, in order."""
    offsets, total = [], 0
    for text in texts:
        offsets.append(total)
        total += text.compared
    return offsets


def batch_token_index(texts: Sequence[DiffText], offsets: Sequence[int], rows: Sequence[int], device) -> torch.Tensor:
    """The global indices of a batch's compared tokens, in the order hidden[span_mask] gives them."""
    return torch.cat([offsets[i] + torch.arange(texts[i].compared, device=device) for i in rows])


def ordinary_tokens(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    [m] bool: the tokens whose activations [m, d] aren't massive in either model (see
    budget.MASSIVE_NORM), among a batch's tokens at one position.
    """
    keep = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
    if a.shape[0] < MIN_NORM_TOKENS:
        return keep
    for activations in (a, b):
        norm = activations.float().norm(dim=1)
        median = norm.median()
        spread = 1.4826 * (norm - median).abs().median()
        keep &= ~((norm > MASSIVE_NORM * median) & (norm > median + MASSIVE_DEVIATIONS * spread))
    return keep


def prediction_shift(reference: ProbeTarget, model: ProbeTarget, reference_final: torch.Tensor, final: torch.Tensor) -> dict:
    """
    How far a model's next-token predictions are from a reference model's, token by token,
    from the last layer's outputs [m, d] of each at the same tokens.

    Returns:
        "kl": [m] KL(reference ‖ model) in nats; "agree": [m] bool, whether their top
        predictions are the same token; "reference_top", "model_top": [m] token IDs;
        "reference_prob", "model_prob": [m] each model's probability of its own top token
    """
    out = {key: [] for key in ("kl", "agree", "reference_top", "model_top", "reference_prob", "model_prob")}
    for i in range(0, final.shape[0], KL_CHUNK):
        log_reference = torch.log_softmax(reference.logits(reference_final[i:i + KL_CHUNK]).float(), dim=-1)
        log_model = torch.log_softmax(model.logits(final[i:i + KL_CHUNK]).float(), dim=-1)
        out["kl"].append((log_reference.exp() * (log_reference - log_model)).sum(dim=-1).clamp_min(0))
        reference_prob, reference_top = log_reference.max(dim=-1)
        model_prob, model_top = log_model.max(dim=-1)
        out["agree"].append(reference_top == model_top)
        out["reference_top"].append(reference_top)
        out["model_top"].append(model_top)
        out["reference_prob"].append(reference_prob.exp())
        out["model_prob"].append(model_prob.exp())
    return {key: torch.cat(values) for key, values in out.items()}


def _rounded(values: torch.Tensor, digits: int = 4) -> list[float]:
    return [round(v, digits) for v in values.tolist()]


@torch.no_grad()
def run_layers(base: ProbeTarget, other: ProbeTarget, texts: Sequence[DiffText], tokens_per_batch: int = TOKENS_PER_BATCH) -> dict:
    """
    Compare two models of the same shape on a set of texts. Caller holds both targets' locks.

    Returns:
        "num_layers", "hidden_dim", "num_texts", "num_tokens" (compared tokens), "total_tokens"
        (tokens one model read), "half_tokens": [2];
        "distance", "cosine", "cka", "cka_chance", "procrustes": [positions] over all tokens;
        "halves": {"distance", "cosine": [2][positions], "cka", "procrustes": the same, or
        None on models too wide for them};
        "kl" (mean KL(other ‖ base) per token, nats), "half_kl": [2], "top1_agreement";
        "texts": per text {"source", "half", "lead" (the text's tokens before the compared
        ones: its first, on a model without a BOS), "tokens" (its compared tokens), "kl",
        "base_top", "other_top", "base_prob", "other_prob": one per compared token};
        "token_distance": [positions, num_tokens] float32 (CPU), each token's relative distance
        at each position, tokens in the texts' order (−1 where a token was left out for its
        massive activations); "massive": [positions] how many were
    """
    num_layers = len(base.layers)
    positions = num_layers + 1
    device = base.device
    count = sum(text.compared for text in texts)
    offsets = token_offsets(texts)
    half_of = [i % 2 for i in range(len(texts))]

    stats = StreamSimilarity(positions, base.hidden_dim, similarity_halves(num_layers, base.hidden_dim), device)
    token_distance = torch.zeros(positions, count, device=device)
    massive = [0] * positions
    token = {
        "kl": torch.zeros(count, device=device),
        "agree": torch.zeros(count, dtype=torch.bool, device=device),
        "reference_top": torch.zeros(count, dtype=torch.long, device=device),
        "model_top": torch.zeros(count, dtype=torch.long, device=device),
        "reference_prob": torch.zeros(count, device=device),
        "model_prob": torch.zeros(count, device=device),
    }

    for half in (0, 1):
        members = [i for i in range(len(texts)) if half_of[i] == half]
        for batch in forward_batches(base, encoded(texts, members), tokens_per_batch):
            rows = [members[i] for i in batch.idx.tolist()]
            in_span = span_mask(batch)
            index = batch_token_index(texts, offsets, rows, device)
            base_hidden: list[Optional[torch.Tensor]] = [None] * positions

            def keep(position: int, hidden: torch.Tensor) -> None:
                base_hidden[position] = hidden[in_span]

            with ResidualReader(base.layers, None, keep) as reader:
                reader.run(lambda: base.forward(batch.input_ids, batch.attention_mask))
            if any(hidden is None for hidden in base_hidden):
                raise RuntimeError("The base model's pass didn't reach every residual stream position")

            def compare(position: int, hidden: torch.Tensor) -> None:
                a, b = base_hidden[position], hidden[in_span]
                ordinary = ordinary_tokens(a, b)
                distances = torch.full((a.shape[0],), -1.0, device=device)
                distances[ordinary] = stats.add(half, position, a[ordinary], b[ordinary])
                token_distance[position, index] = distances
                massive[position] += int((~ordinary).sum())
                if position == positions - 1:
                    for key, values in prediction_shift(other, base, b, a).items():
                        token[key][index] = values
                base_hidden[position] = None

            with ResidualReader(other.layers, None, compare) as reader:
                reader.run(lambda: other.forward(batch.input_ids, batch.attention_mask))
            if any(hidden is not None for hidden in base_hidden):
                raise RuntimeError("The other model's pass didn't reach every residual stream position")

    for position in range(positions):
        scale = stats.scale(position)
        read = token_distance[position] >= 0
        token_distance[position, read] = token_distance[position, read] / scale if scale > 0 else 0.0

    whole = stats.alignment()
    half_alignment = [stats.alignment(half) for half in (0, 1)] if stats.matrix_halves else None
    half_tokens = [sum(text.compared for i, text in enumerate(texts) if half_of[i] == half) for half in (0, 1)]
    half_kl = []
    for half in (0, 1):
        mask = torch.zeros(count, dtype=torch.bool, device=device)
        for i, text in enumerate(texts):
            if half_of[i] == half:
                mask[offsets[i]:offsets[i] + text.compared] = True
        half_kl.append(float(token["kl"][mask].mean()) if half_tokens[half] else None)

    kl = token["kl"].cpu()
    base_top, other_top = token["model_top"].cpu().tolist(), token["reference_top"].cpu().tolist()
    base_prob, other_prob = _rounded(token["model_prob"].cpu()), _rounded(token["reference_prob"].cpu())
    kl_values = _rounded(kl)
    names: dict[int, str] = {}

    def name(token_id: int) -> str:
        if token_id not in names:
            names[token_id] = base.token_text(token_id)
        return names[token_id]

    out_texts = []
    for i, text in enumerate(texts):
        span = slice(offsets[i], offsets[i] + text.compared)
        out_texts.append({
            "source": text.source,
            "half": half_of[i],
            "lead": [name(token_id) for token_id in text.ids[text.lead:text.start]],
            "tokens": [name(token_id) for token_id in text.ids[text.start:]],
            "kl": kl_values[span],
            "base_top": [name(token_id) for token_id in base_top[span]],
            "other_top": [name(token_id) for token_id in other_top[span]],
            "base_prob": base_prob[span],
            "other_prob": other_prob[span],
        })

    return {
        "num_layers": num_layers,
        "hidden_dim": base.hidden_dim,
        "num_texts": len(texts),
        "num_tokens": count,
        "total_tokens": sum(len(text.ids) for text in texts),
        "half_tokens": half_tokens,
        "distance": stats.distance((0, 1)),
        "cosine": stats.mean_cosine((0, 1)),
        "cka": whole["cka"],
        "cka_chance": whole["cka_chance"],
        "procrustes": whole["procrustes"],
        "halves": {
            "distance": [stats.distance((half,)) for half in (0, 1)],
            "cosine": [stats.mean_cosine((half,)) for half in (0, 1)],
            "cka": [h["cka"] for h in half_alignment] if half_alignment else None,
            "procrustes": [h["procrustes"] for h in half_alignment] if half_alignment else None,
        },
        "kl": float(kl.mean()),
        "half_kl": half_kl,
        "top1_agreement": float(token["agree"].float().mean()),
        "texts": out_texts,
        "token_distance": token_distance.cpu(),
        "massive": massive,
    }
