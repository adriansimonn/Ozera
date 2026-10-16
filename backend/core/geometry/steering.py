"""
Steering along a concept's shape: does the model use it?

The manifold is the class means at one residual stream position, in their top m principal
components, joined in value order by a spline (closed for cyclic concepts). An eval prompt that
ends where the model would say a value ("Today is {x}. Tomorrow is", filled with a source
value) is edited at its read position, at that residual stream position, at steps + 1 points
of a walk from the source value to a target value, two ways:

- manifold: the activation's coordinates in the subspace are set to the spline's point and the
  rest of it is left alone, h' = h + B(c* − Bᵀ(h − μ));
- linear: t × (mean[target] − mean[source]) is added, the straight line between the two class
  means (difference-in-means steering).

Both end at the target's class mean (within the subspace) and differ on the way, where the
straight line cuts through the inside of a curve. The readout is the next-token distribution
at the prompt's last position (forward passes only, no generation):

- the probability of each value's first token as the prompt's continuation (with its leading
  space). Values whose first tokens collide are dropped; tokens every value starts with (a lone
  space, say) are appended to the prompt first;
- the total probability on those tokens, which drops where an edit leaves the concept;
- the Hellinger distance to the nearest unsteered value's distribution (the prompt filled with
  each value, unedited): how far an edit takes the model from anything it does by itself.

A read position spanning several tokens (the mean over a value's tokens) is edited by shifting
each of them by the same vector, so their mean lands where the edit puts it.
"""

import threading
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch

from core.probes.capture import ResidualReader
from core.probes.pooling import pool_hidden
from core.probes.runner import EncodedText, ProbeTarget, forward_batches

from .budget import MAX_STEER_DIMS, MAX_STEER_STEPS, MIN_STEER_STEPS, GeometryInputError, steer_passes
from .concepts import READS, fill, validate_eval_template
from .encoding import ExampleTokens, encode_concept, encode_example, read_span

# Centripetal Catmull-Rom (α = 0.5): interpolates the class means without the overshoot and
# small loops a uniform cubic spline makes between unevenly spaced points
_CENTRIPETAL = 0.5
# Points per segment of the curve drawn for the views
CURVE_POINTS_PER_SEGMENT = 8
# Most likely tokens reported per pass
TOP_TOKENS = 3


# The manifold

def spline_points(points: torch.Tensor, closed: bool, s: torch.Tensor) -> torch.Tensor:
    """
    Points on a centripetal Catmull-Rom spline through `points` in order, at value indices s
    (s = i is points[i]; between them, the curve). Open curves extend past their ends by
    reflecting the neighbouring point; closed ones wrap round (s is taken modulo k).

    Args:
        points: [k, m] (k >= 3)
        s: [S] value indices

    Returns:
        [S, m] float32
    """
    k = points.shape[0]
    P = points.double()
    s = s.double().to(P.device)
    if closed:
        E = torch.cat([P[-1:], P, P[:2]])
        s = torch.remainder(s, k)
        segment = s.floor().long().clamp(0, k - 1)
    else:
        E = torch.cat([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
        s = s.clamp(0, k - 1)
        segment = s.floor().long().clamp(0, k - 2)
    u = (s - segment)[:, None]
    P0, P1, P2, P3 = (E[segment + j] for j in range(4))

    def gap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (b - a).norm(dim=1, keepdim=True).pow(_CENTRIPETAL).clamp_min(1e-12)

    # Barry–Goldman's pyramid, knots t0 = 0 < t1 < t2 < t3 spaced by √distance
    t1 = gap(P0, P1)
    t2 = t1 + gap(P1, P2)
    t3 = t2 + gap(P2, P3)
    t = t1 + u * (t2 - t1)
    A1 = ((t1 - t) * P0 + t * P1) / t1
    A2 = ((t2 - t) * P1 + (t - t1) * P2) / (t2 - t1)
    A3 = ((t3 - t) * P2 + (t - t2) * P3) / (t3 - t2)
    B1 = ((t2 - t) * A1 + t * A2) / t2
    B2 = ((t3 - t) * A2 + (t - t1) * A3) / (t3 - t1)
    return (((t2 - t) * B1 + (t - t1) * B2) / (t2 - t1)).float()


def walk(k: int, source: int, target: int, closed: bool, steps: int) -> torch.Tensor:
    """
    The value index at each of steps + 1 evenly spaced points from source to target: the
    shorter way round a closed curve (halfway ties go forward), kept in [0, k). [steps + 1] float64
    """
    delta = float(target - source)
    if closed:
        delta = float((target - source) % k)
        if delta > k / 2:
            delta -= k
    s = source + torch.linspace(0, 1, steps + 1, dtype=torch.float64) * delta
    return torch.remainder(s, k) if closed else s


# The intervention

@dataclass
class SpanEdits:
    """One batch's edits (see SubspaceSetter), all on the target's device."""

    starts: torch.Tensor  # [batch] each row's span [start, end)
    ends: torch.Tensor
    coords: torch.Tensor  # [batch, m] the coordinates to give each span's mean
    set_coords: torch.Tensor  # [batch] bool: rows whose coordinates are set (the others' coords are ignored)
    shift: torch.Tensor  # [batch, d] then added to every row


class SubspaceSetter:
    """
    Edits a probe target's residual stream at one position (0: the first layer's input, i:
    layer i-1's output) in forward passes over whole sequences (no KV cache).

    Each row's span of tokens is shifted by one vector: the one that gives the span's mean the
    row's coordinates in a subspace, h' = h + B(c − Bᵀ(h̄ − μ)), plus the row's added shift.
    Set `edits` before each pass, or None to leave passes unchanged; after a pass, `before`
    holds each row's span mean before the edit and `delta` the vector it was shifted by.

    Same pattern as core.probes.directions.DirectionAblation: enter it before a ResidualReader
    on the same target, so the reader sees the edited stream, and the hooks only act in the
    thread that entered.
    """

    def __init__(self, target: ProbeTarget, position: int, basis: torch.Tensor, center: torch.Tensor):
        """
        Args:
            position: Residual stream position (0..len(target.layers))
            basis: [d, m] orthonormal columns B
            center: [d] the subspace's origin μ
        """
        if not 0 <= position <= len(target.layers):
            raise GeometryInputError(f"Position {position} is out of range for this model ({len(target.layers)} layers)")
        self.target = target
        self.position = position
        self.basis = basis.float()
        self.center = center.float()
        self.edits: Optional[SpanEdits] = None
        self.before: Optional[torch.Tensor] = None
        self.delta: Optional[torch.Tensor] = None
        self._handles: list = []
        self._thread: Optional[int] = None

    def _edit(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        edits = self.edits
        if edits is None or threading.get_ident() != self._thread:
            return None
        h = hidden.float()
        seq = torch.arange(h.shape[1], device=h.device)
        in_span = (seq[None, :] >= edits.starts[:, None]) & (seq[None, :] < edits.ends[:, None])  # [batch, seq]
        mean = (h * in_span[..., None]).sum(dim=1) / in_span.sum(dim=1, keepdim=True)
        current = (mean - self.center) @ self.basis
        setting = (edits.coords - current) @ self.basis.T
        delta = torch.where(edits.set_coords[:, None], setting, torch.zeros_like(setting)) + edits.shift
        self.before, self.delta = mean, delta
        # Tokens outside the spans keep their exact values
        return torch.where(in_span[..., None], (h + delta[:, None, :]).to(hidden.dtype), hidden)

    def __enter__(self) -> "SubspaceSetter":
        self._thread = threading.get_ident()
        layers = self.target.layers

        if self.position == 0:
            def on_input(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                edited = self._edit(hidden)
                if edited is None:
                    return None
                if args:
                    return (edited,) + tuple(args[1:]), kwargs
                return args, {**kwargs, "hidden_states": edited}

            self._handles.append(layers[0].register_forward_pre_hook(on_input, with_kwargs=True))
        else:
            def on_output(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                edited = self._edit(tensor)
                if edited is None:
                    return None
                return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited

            self._handles.append(layers[self.position - 1].register_forward_hook(on_output))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# The readout

def _continuations(
    target: ProbeTarget,
    eval_template: str,
    source_value: str,
    values: list[str],
    chat_template: bool,
    prompt_ids: list[int],
) -> list[Optional[list[int]]]:
    """
    Each value's tokens as the eval prompt's continuation (with a leading space; right after
    the generation prompt in a chat template), or None where the prompt's own tokens don't
    stay a prefix.
    """
    text = fill(eval_template, source_value)
    if target.model_type == "ozera":
        encode = lambda value: target.tokenizer.encode(f"{text} {value}")  # noqa: E731
    elif chat_template:
        rendered = target.tokenizer.apply_chat_template(
            target.loader.chat_messages(text),
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        encode = lambda value: target.tokenizer(rendered + value, add_special_tokens=False)["input_ids"]  # noqa: E731
    else:
        encode = lambda value: target.tokenizer(f"{text} {value}")["input_ids"]  # noqa: E731

    n = len(prompt_ids)
    continuations = []
    for value in values:
        ids = list(encode(value))
        continuations.append(ids[n:] if ids[:n] == list(prompt_ids) and len(ids) > n else None)
    return continuations


def readout_tokens(continuations: list[Optional[list[int]]]) -> tuple[list[int], list[Optional[int]], list[int]]:
    """
    Where each value is read in the next-token distribution.

    Returns:
        (the tokens every value's continuation starts with, appended to the prompt before
         reading; each value's first token after them, or None; the values that can be read:
         those whose first token no other value shares)
    """
    valid = [c for c in continuations if c]
    prefix: list[int] = []
    if valid:
        shortest = min(len(c) for c in valid)
        while len(prefix) < shortest - 1 and all(c[len(prefix)] == valid[0][len(prefix)] for c in valid):
            prefix.append(valid[0][len(prefix)])
    first = [c[len(prefix)] if c else None for c in continuations]
    counts = Counter(token for token in first if token is not None)
    readable = [v for v, token in enumerate(first) if token is not None and counts[token] == 1]
    return prefix, first, readable


# A run

@dataclass
class SteeringPlan:
    """A steering run, tokenized and checked (see plan_steering)."""

    concept: dict
    read: str
    position: int
    dims: int
    source: int
    target: int
    steps: int
    eval_template: str
    examples: list[EncodedText]  # The concept's examples, read at the read position
    prompts: list[ExampleTokens]  # The eval prompt filled with each value
    prefix: list[int]
    first_tokens: list[Optional[int]]
    readable: list[int]
    concept_tokens: int
    eval_tokens: int
    eval_passes: int


def plan_steering(
    target: ProbeTarget,
    concept: dict,
    read: str,
    chat_template: bool,
    position: int,
    dims: int,
    source: int,
    target_value: int,
    steps: int,
    eval_template: str,
) -> SteeringPlan:
    """
    Check a steering run and tokenize everything it reads. Caller holds target.lock() if the
    tokenizer is shared.

    Args:
        concept: A validated concept (cyclic or ordinal: the spline follows the values' order)
        read: Where examples are read and the eval prompt is edited ("value_last", ...)
        position: Residual stream position the class means are read and the edits made at
        dims: Subspace size m (at most k - 1)
        source, target_value: Value indices the walk starts and ends at
        steps: The walk's steps (steps + 1 points, both ends included)
        eval_template: A template ending where the model would say a value

    Raises:
        GeometryInputError: the run can't be done as asked, with what to change
    """
    k = len(concept["values"])
    if concept["structure"] == "categorical":
        raise GeometryInputError("Steering follows the values' order; categorical concepts have none")
    if read not in READS:
        raise GeometryInputError(f"Unknown read position: {read}")
    if not 0 <= position <= len(target.layers):
        raise GeometryInputError(f"Position {position} is out of range for this model ({len(target.layers)} layers)")
    if not 1 <= dims <= min(k - 1, target.hidden_dim, MAX_STEER_DIMS):
        raise GeometryInputError(f"The subspace needs 1 to {min(k - 1, target.hidden_dim, MAX_STEER_DIMS)} dimensions")
    if not (0 <= source < k and 0 <= target_value < k):
        raise GeometryInputError(f"Choose values 1 to {k}")
    if source == target_value:
        raise GeometryInputError("Choose a target value different from the source")
    if not MIN_STEER_STEPS <= steps <= MAX_STEER_STEPS:
        raise GeometryInputError(f"Use {MIN_STEER_STEPS} to {MAX_STEER_STEPS} steps")
    eval_template = validate_eval_template(eval_template)

    examples, _ = encode_concept(target, concept["values"], concept["templates"], read, chat_template)
    prompts = [encode_example(target, eval_template, value, chat_template) for value in concept["values"]]
    continuations = _continuations(
        target, eval_template, concept["values"][source], concept["values"], chat_template, prompts[source].ids,
    )
    prefix, first_tokens, readable = readout_tokens(continuations)
    if len(readable) < 2:
        values = concept["values"]
        shared = Counter(token for token in first_tokens if token is not None).most_common(1)
        example = ""
        if shared and shared[0][1] > 1:
            a, b = [values[v] for v, token in enumerate(first_tokens) if token == shared[0][0]][:2]
            example = f' ("{a}" and "{b}" both start with "{target.token_text(shared[0][0])}")'
        raise GeometryInputError(
            f"Fewer than two values can be told apart by their first token after the eval prompt on this model{example}"
        )
    longest = max(len(p.ids) for p in prompts) + len(prefix)
    if longest > target.max_tokens:
        raise GeometryInputError(f"The eval prompt is longer than {target.max_tokens} tokens")

    passes = steer_passes(k, steps)
    eval_tokens = sum(len(p.ids) + len(prefix) for p in prompts) + (passes - k) * (len(prompts[source].ids) + len(prefix))
    return SteeringPlan(
        concept=concept,
        read=read,
        position=position,
        dims=dims,
        source=source,
        target=target_value,
        steps=steps,
        eval_template=eval_template,
        examples=examples,
        prompts=prompts,
        prefix=prefix,
        first_tokens=first_tokens,
        readable=readable,
        concept_tokens=sum(len(e.ids) for e in examples),
        eval_tokens=eval_tokens,
        eval_passes=passes,
    )


def class_means_at(target: ProbeTarget, examples: list[EncodedText], k: int, position: int) -> torch.Tensor:
    """
    Each value's mean activation at one position over a concept's examples (example i has
    value i % k), pooled over each example's read span. Caller holds target.lock(). [k, d] float32
    """
    d = target.hidden_dim
    sums = torch.zeros(k, d, dtype=torch.float32, device=target.device)
    counts = torch.zeros(k, dtype=torch.float32, device=target.device)
    current: dict = {}

    def on_hidden(_: int, hidden: torch.Tensor) -> None:
        batch = current["batch"]
        values = batch.idx % k
        sums.index_add_(0, values, pool_hidden(hidden, batch.starts, batch.ends, ("mean",))["mean"])
        counts.index_add_(0, values, torch.ones_like(values, dtype=torch.float32))

    with ResidualReader(target.layers, [position], on_hidden) as reader:
        for batch in forward_batches(target, examples):
            current["batch"] = batch
            reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
    return sums / counts.clamp_min(1)[:, None]


def _top(target: ProbeTarget, probs: torch.Tensor, ids: torch.Tensor) -> list[list[dict]]:
    return [
        [{"token": target.token_text(int(i)), "prob": float(p)} for p, i in zip(row_p, row_i)]
        for row_p, row_i in zip(probs.tolist(), ids.tolist())
    ]


def run_steering(target: ProbeTarget, plan: SteeringPlan) -> dict:
    """
    Walk the eval prompt from the source value to the target along the manifold and along the
    straight line, reading the next-token distribution at every step. Caller holds
    target.lock().

    Returns:
        {
          "prompt": the eval prompt with the source value, "edited_tokens": the tokens edited,
          "prefix": tokens appended before reading, "readable": the values read (indices),
          "tokens": each readable value's token,
          "path": [S] value index at each step (S = steps + 1),
          "explained": share of the class means' variance in the subspace,
          "eval_decoding": share of values whose eval prompt's activation is nearest its own
              class mean in the subspace,
          "reference": the prompt with each value, unedited: {"probs": [k][R] on the readable
              values' tokens, "mass": [k], "top": [k][3] most likely tokens,
              "nearest_other": [k] Hellinger distance to the nearest other value's distribution},
          "methods": {"manifold" | "linear": {"probs": [S][R], "mass": [S], "hellinger": [S]
              to the nearest unsteered value's distribution, "nearest": [S] that value,
              "to_target": [S] Hellinger distance to the target's, "top": [S][3],
              "edit_size": [S] ‖edit‖ / ‖activation‖}},
          "plot": on the class means' top two components, scaled to their RMS radius:
              {"means": [k, 2], "eval": [k, 2] the eval prompt's activations, "curve": [C, 2]
               the spline, "manifold", "linear": [S, 2] the edited activations},
        }
    """
    concept = plan.concept
    k = len(concept["values"])
    m = plan.dims
    closed = concept["structure"] == "cyclic"
    device = target.device

    means = class_means_at(target, plan.examples, k, plan.position)
    center = means.mean(dim=0)
    _, S, Vh = torch.linalg.svd(means - center, full_matrices=False)
    r = min(k - 1, target.hidden_dim)
    V = Vh[:r].T  # [d, r]
    lam = S[:r].double() ** 2
    if float(lam.sum()) <= 0:
        raise GeometryInputError("The values' class means coincide at this layer, so there's no shape to steer along")
    B = V[:, :m]
    coords = (means - center) @ B  # [k, m]

    path = walk(k, plan.source, plan.target, closed, plan.steps)
    steps = path.shape[0]
    t = torch.linspace(0, 1, steps, device=device)
    on_curve = spline_points(coords, closed, path).to(device)
    gap = means[plan.target] - means[plan.source]

    # Rows: each value's prompt unedited, then the source's prompt along the curve, then along the line
    rows = [EncodedText(p.ids + plan.prefix, *read_span(p, plan.read)) for p in plan.prompts]
    source_prompt = plan.prompts[plan.source]
    rows += [EncodedText(source_prompt.ids + plan.prefix, *read_span(source_prompt, plan.read))] * (2 * steps)
    n = len(rows)
    manifold = slice(k, k + steps)
    linear = slice(k + steps, n)
    set_coords = torch.zeros(n, dtype=torch.bool, device=device)
    set_coords[manifold] = True
    targets = torch.zeros(n, m, device=device)
    targets[manifold] = on_curve
    shift_scale = torch.zeros(n, device=device)
    shift_scale[linear] = t
    readout_at = torch.tensor([len(row.ids) - 1 for row in rows], device=device)
    value_ids = torch.tensor([plan.first_tokens[v] for v in plan.readable], device=device)

    value_probs = torch.empty(n, len(plan.readable), device=device)
    top_probs = torch.empty(n, TOP_TOKENS, device=device)
    top_ids = torch.empty(n, TOP_TOKENS, dtype=torch.long, device=device)
    before = torch.empty(n, target.hidden_dim, device=device)
    delta = torch.empty(n, target.hidden_dim, device=device)
    roots: Optional[torch.Tensor] = None  # [n, vocab] √p, for Hellinger distances
    current: dict = {}

    def on_final(_: int, hidden: torch.Tensor) -> None:
        current["final"] = hidden

    with SubspaceSetter(target, plan.position, B, center) as setter, \
            ResidualReader(target.layers, [len(target.layers)], on_final) as reader:
        for batch in forward_batches(target, rows):
            idx = batch.idx
            setter.edits = SpanEdits(
                batch.starts, batch.ends, targets[idx], set_coords[idx], shift_scale[idx, None] * gap[None, :],
            )
            reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
            final = current.pop("final")
            with torch.no_grad():
                last = final[torch.arange(len(idx), device=device), readout_at[idx]]
                probs = torch.softmax(target.logits(last).float(), dim=-1)
            if roots is None:
                roots = torch.empty(n, probs.shape[1], device=device)
            roots[idx] = probs.sqrt()
            value_probs[idx] = probs[:, value_ids]
            top = probs.topk(TOP_TOKENS, dim=-1)
            top_probs[idx], top_ids[idx] = top.values, top.indices
            before[idx], delta[idx] = setter.before, setter.delta
        setter.edits = None

    # Hellinger distance between every pass's distribution and each value's unedited one
    hellinger = (1 - roots @ roots[:k].T).clamp(0, 1).sqrt()  # [n, k]
    del roots
    others = hellinger[:k].clone()
    others.fill_diagonal_(float("inf"))
    nearest_distance, nearest = hellinger.min(dim=1)
    mass = value_probs.sum(dim=1)
    edit_size = delta.norm(dim=1) / before.norm(dim=1).clamp_min(1e-12)

    # Does the subspace read the eval prompt's own value?
    eval_coords = (before[:k] - center) @ B
    decoded = torch.cdist(eval_coords[None], coords[None])[0].argmin(dim=1)
    eval_decoding = float((decoded == torch.arange(k, device=device)).float().mean())

    scale = float((lam.sum() / k).sqrt())
    V2 = V[:, :2]

    def plane(x: torch.Tensor) -> list:
        return ((x - center) @ V2 / scale).tolist()

    dense = torch.linspace(0, k if closed else k - 1, (k if closed else k - 1) * CURVE_POINTS_PER_SEGMENT + 1)
    if m >= 2:
        curve = (spline_points(coords, closed, dense)[:, :2] / scale).tolist()
    else:
        curve = spline_points((means - center) @ V2 / scale, closed, dense).tolist()

    def method(rows: slice) -> dict:
        return {
            "probs": value_probs[rows].tolist(),
            "mass": mass[rows].tolist(),
            "hellinger": nearest_distance[rows].tolist(),
            "nearest": nearest[rows].tolist(),
            "to_target": hellinger[rows, plan.target].tolist(),
            "top": _top(target, top_probs[rows], top_ids[rows]),
            "edit_size": edit_size[rows].tolist(),
        }

    start, end = read_span(source_prompt, plan.read)
    return {
        "prompt": fill(plan.eval_template, concept["values"][plan.source]),
        "edited_tokens": [target.token_text(i) for i in source_prompt.ids[start:end]],
        "prefix": [target.token_text(i) for i in plan.prefix],
        "readable": plan.readable,
        "tokens": [target.token_text(plan.first_tokens[v]) for v in plan.readable],
        "path": path.tolist(),
        "explained": float(lam[:m].sum() / lam.sum()),
        "eval_decoding": eval_decoding,
        "reference": {
            "probs": value_probs[:k].tolist(),
            "mass": mass[:k].tolist(),
            "top": _top(target, top_probs[:k], top_ids[:k]),
            "nearest_other": others.min(dim=1).values.tolist(),
        },
        "methods": {"manifold": method(manifold), "linear": method(linear)},
        "plot": {
            "means": plane(means),
            "eval": plane(before[:k]),
            "curve": curve,
            "manifold": plane(before[manifold] + delta[manifold]),
            "linear": plane(before[linear] + delta[linear]),
        },
    }
