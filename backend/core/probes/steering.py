"""
Steering with a probe's direction: generations with α × the direction added to the residual
stream where the probe reads, side by side with an unsteered baseline and, optionally, a
generation with the direction ablated everywhere.

Every generation is then read by the unsteered model: the probe's score of the generated
text (does the output now read as the concept?) and the text's perplexity (has steering
broken the model's fluency?). Neither is ground truth: the probe is the thing being tested,
and an unusual but fluent continuation is also surprising to the clean model. Read them as a
dose-response curve next to the texts themselves.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from core.patching.engine import PatchingEngine

from .budget import ProbeInputError
from .directions import ablation_patches, steering_patch
from .runner import ProbeTarget, encode_texts, score_texts


@dataclass
class ScoringProbe:
    """A probe, as it reads texts."""

    layer: int  # Decoder layer whose output it reads
    weights: Sequence[float]
    bias: float
    pooling: str
    chat_template: bool = False
    read_span: str = "text"


def _finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _divergence(baseline: list[int], ids: list[int]) -> tuple[Optional[int], int]:
    """(first position where a continuation differs from the baseline's, or None; tokens that differ)"""
    overlap = min(len(baseline), len(ids))
    changed = [i for i in range(overlap) if baseline[i] != ids[i]]
    first = changed[0] if changed else (overlap if len(baseline) != len(ids) else None)
    return first, len(changed) + abs(len(baseline) - len(ids))


def _prompt_tokens(target: ProbeTarget, prompt: str) -> int:
    if target.model_type == "open_source":
        return int(target.model_loader.encode_prompt(prompt).input_ids.shape[1])
    return len(target.engine_tokenizer.encode(prompt))


def _read_back(target: ProbeTarget, ids: list[int], prompt_tokens: int, probe: ScoringProbe) -> dict:
    """What the unsteered model makes of a generation: its text, perplexity, and probe score."""
    continuation = ids[prompt_tokens:]
    text = target.decode(continuation)
    result = {
        "text": text,
        "generated_tokens": len(continuation),
        "perplexity": None,
        "probe_score": None,
        "scored": None,
    }
    if continuation:
        logprobs = target.token_logprobs(ids, prompt_tokens)
        result["perplexity"] = _finite(math.exp(-sum(logprobs) / len(logprobs)))
    if text.strip():
        encoded = encode_texts(target, [text.strip()], probe.chat_template, probe.read_span)
        scored = score_texts(target, encoded, probe.layer, probe.weights, probe.bias, probe.pooling)[0]
        if all(math.isfinite(s) for s in scored["scores"]) and math.isfinite(scored["score"]):
            result["probe_score"] = scored["score"]
            result["scored"] = {"text": text.strip(), **scored}
    return result


def run_steering(
    target: ProbeTarget,
    prompt: str,
    probe: ScoringProbe,
    vector: Sequence[float],
    alphas: Sequence[float],
    ablate: bool = False,
    generated_only: bool = False,
    max_tokens: int = 40,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    Generate from a prompt with the probe's direction added at its layer, at each α.

    Caller holds target.lock().

    Args:
        probe: The probe the direction comes from (also scores every generation)
        vector: The steering vector for α = 1, [hidden_dim]
        alphas: Multiples of the vector to steer with (0 is the baseline, always generated)
        ablate: Also generate with the vector's direction projected out of the residual
            stream everywhere
        generated_only: Steer only the generated tokens' positions, not the prompt's
        temperature: 0 for greedy decoding; otherwise every generation samples from the same seed

    Returns:
        {"prompt_tokens", "generations": [{"kind": "baseline" | "steer" | "ablate", "alpha",
         "text", "generated_tokens", "perplexity", "probe_score", "scored" (the probe's
         per-token scores of the text, as for the token heatmap), "first_divergence",
         "tokens_changed"}]}

    Raises:
        ProbeInputError, PatchError: the request doesn't fit the model
    """
    if not 0 <= probe.layer < len(target.layers):
        raise ProbeInputError(f"Layer {probe.layer} is out of range for this model ({len(target.layers)} layers)")
    v = torch.as_tensor(vector, dtype=torch.float32, device=target.device)
    if v.numel() != target.hidden_dim:
        raise ProbeInputError(f"The direction has {v.numel()} values; this model's activations have {target.hidden_dim}")

    prompt_tokens = _prompt_tokens(target, prompt)
    positions = list(range(prompt_tokens, prompt_tokens + max_tokens)) if generated_only else None

    conditions: list[tuple[str, Optional[float], list]] = [("baseline", 0.0, [])]
    for alpha in alphas:
        if alpha != 0:
            conditions.append(("steer", float(alpha), [steering_patch(probe.layer, alpha * v, positions)]))
    if ablate:
        conditions.append(("ablate", None, ablation_patches(len(target.layers), v / v.norm())))

    engine = PatchingEngine()
    generations = []
    baseline: list[int] = []
    for kind, alpha, patches in conditions:
        if temperature > 0:
            torch.manual_seed(seed)
        _, ids, _ = engine.run_generation(
            prompt, patches, target.model_loader, target.model_type, target.engine_tokenizer,
            max_new_tokens=max_tokens, temperature=temperature,
        )
        continuation = ids[prompt_tokens:]
        if kind == "baseline":
            baseline = continuation
        first, changed = _divergence(baseline, continuation)
        generations.append({
            "kind": kind,
            "alpha": alpha,
            **_read_back(target, ids, prompt_tokens, probe),
            "first_divergence": first,
            "tokens_changed": changed,
        })

    return {"prompt_tokens": prompt_tokens, "generations": generations}
