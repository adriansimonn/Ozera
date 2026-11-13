"""
One feature through autointerp.

The explainer reads the feature's explain set (shuffled) and writes an explanation. The scorer
rates it on the held-out score set:

- detection: the activating contexts and the negatives, unmarked and shuffled,
  DETECTION_PER_PROMPT per prompt: is the feature active anywhere in each?
- fuzzing: the activating contexts, half marked at their true firing tokens and half at as many
  random read tokens, FUZZING_PER_PROMPT per prompt: are the marks where it's active?
- the token baseline: the feature's top 3 firing tokens as an explanation ("the tokens 'X', 'Y',
  'Z'"), scored by detection on exactly the same prompts.

That's 1 explain request and 13 scoring requests for a full set. Prompts are built from a seed,
so a run is reproducible and a rerun with another seed reshuffles them.

A refused explanation marks the feature refused; a refused scoring prompt drops its examples,
and the feature is refused if detection or fuzzing can't be scored without them. Run alone
(run_feature), any other failed request fails the run (the caller decides what to keep); in a
batch job (core.autointerp.jobs), a failed scoring request drops its examples like a refusal,
so an explanation already paid for is kept, unscored if it has to be.

Testing an explanation (AUTOINTERP_PLAN.md, Phase 3):

- A hypothesis is a user's own explanation, scored by the same prompts on the same held-out set
  (run_hypothesis): its scores compare with the autointerp explanation's and the token baseline's.
- A generation test asks whether an explanation is specific enough to write text the feature
  fires on, not just to recognize it: Claude writes texts the explanation says it fires on and
  near-misses it says it doesn't (generation_prompt), the model and SAE read them
  (core.geometry.atlas.read_feature_on_texts, on a GPU), and generation_result counts the
  texts that reach a share of the feature's maximum over the corpus.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import scoring
from .budget import (
    DETECTION_PER_PROMPT,
    FUZZING_PER_PROMPT,
    GENERATION_FIRING,
    GENERATION_NEAR_MISSES,
    GENERATION_TEXT_CHARS,
    GENERATION_THRESHOLD,
    MAX_EXPLANATION_CHARS,
    MAX_EXPLANATION_TOKENS,
    MODELS,
)
from .client import ClaudeClient, Reply, Usage, request_params
from .examples import Context, FeatureSets, activation_scale, escape, highlighted, plain
from .prompts import (
    ANSWERS_SCHEMA,
    DETECTION_SYSTEM,
    EXPLAIN_SYSTEM,
    EXPLANATION_SCHEMA,
    FUZZING_SYSTEM,
    GENERATE_SYSTEM,
    GENERATION_SCHEMA,
    explain_message,
    generation_message,
    scoring_message,
    token_description,
)

TOKEN_BASELINE_TOKENS = 3


@dataclass
class Item:
    """One example in a scoring prompt."""

    group: str  # "activating", "random" or "token"
    index: int  # its index in that group of the feature's sets
    truth: bool  # detection: the feature is active in it; fuzzing: the marks are correct
    decile: Optional[int]
    marks: Optional[list[int]] = None  # fuzzing: the positions marked


@dataclass
class Prompt:
    kind: str  # "explain", "detection", "fuzzing" or "token"
    params: dict
    items: list[Item] = field(default_factory=list)


def _rng(seed: int, feature: int, kind: str) -> random.Random:
    return random.Random(f"{seed}:{feature}:{kind}")


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def explain_example(ctx: Context) -> str:
    marks = [level > 0 for level in ctx.levels]
    acts = ", ".join(
        f"({json.dumps(escape(token), ensure_ascii=False)}, {activation_scale(level)})"
        for token, level in zip(ctx.tokens, ctx.levels) if level > 0
    )
    return f"{highlighted(ctx.tokens, marks)}\nActivations: {acts}"


def explain_prompt(sets: FeatureSets, explainer: str, seed: int = 0) -> Prompt:
    contexts = list(sets.explain)
    _rng(seed, sets.feature, "explain").shuffle(contexts)
    user = explain_message([explain_example(ctx) for ctx in contexts])
    return Prompt("explain", request_params(explainer, EXPLAIN_SYSTEM, user, EXPLANATION_SCHEMA, MODELS[explainer].explain_max_tokens))


def token_baseline(sets: FeatureSets) -> Optional[str]:
    """The token baseline's explanation, from the feature's top firing tokens (None if it has none)."""
    tokens: list[str] = []
    for text, _ in sets.top_tokens:
        shown = escape(text.strip())
        if shown and shown not in tokens:
            tokens.append(shown)
        if len(tokens) == TOKEN_BASELINE_TOKENS:
            break
    return token_description(tokens) if tokens else None


def _context(sets: FeatureSets, item: Item) -> Context:
    return getattr(sets, item.group)[item.index]


def detection_items(sets: FeatureSets, seed: int = 0) -> list[Item]:
    items = (
        [Item("activating", i, True, c.decile) for i, c in enumerate(sets.activating)]
        + [Item("random", i, False, None) for i in range(len(sets.random))]
        + [Item("token", i, False, None) for i in range(len(sets.token))]
    )
    _rng(seed, sets.feature, "detection").shuffle(items)
    return items


def fuzzing_items(sets: FeatureSets, seed: int = 0) -> list[Item]:
    """The activating contexts, half marked at their firing tokens, half at as many random read tokens."""
    rng = _rng(seed, sets.feature, "fuzzing")
    order = list(range(len(sets.activating)))
    rng.shuffle(order)
    correct = set(order[:(len(order) + 1) // 2])
    items = []
    for i, ctx in enumerate(sets.activating):
        firing = [j for j, level in enumerate(ctx.levels) if level > 0]
        others = [j for j, (level, read) in enumerate(zip(ctx.levels, ctx.read)) if level == 0 and read]
        if i in correct or not others:
            items.append(Item("activating", i, True, ctx.decile, firing))
        else:
            items.append(Item("activating", i, False, ctx.decile, sorted(rng.sample(others, min(len(firing), len(others))))))
    rng.shuffle(items)
    return items


def _scoring_prompt(kind: str, description: str, items: list[Item], sets: FeatureSets, scorer: str) -> Prompt:
    if kind == "fuzzing":
        examples = []
        for item in items:
            ctx = _context(sets, item)
            marked = set(item.marks or [])
            examples.append(highlighted(ctx.tokens, [j in marked for j in range(len(ctx.tokens))]))
        system = FUZZING_SYSTEM
    else:
        examples = [plain(_context(sets, item).tokens) for item in items]
        system = DETECTION_SYSTEM
    user = scoring_message(description, examples)
    return Prompt(kind, request_params(scorer, system, user, ANSWERS_SCHEMA, MODELS[scorer].score_max_tokens), items)


def scoring_prompts(sets: FeatureSets, explanation: str, scorer: str, seed: int = 0, baseline: bool = True) -> list[Prompt]:
    """The scoring requests for an explanation (and its token baseline's, on the same detection prompts)."""
    description = escape(explanation)
    detection = _chunks(detection_items(sets, seed), DETECTION_PER_PROMPT)
    prompts = [_scoring_prompt("detection", description, chunk, sets, scorer) for chunk in detection]
    prompts += [_scoring_prompt("fuzzing", description, chunk, sets, scorer) for chunk in _chunks(fuzzing_items(sets, seed), FUZZING_PER_PROMPT)]
    token = token_baseline(sets) if baseline else None
    if token is not None:
        prompts += [_scoring_prompt("token", token, chunk, sets, scorer) for chunk in detection]
    return prompts


def request_bounds(prompts: Sequence[Prompt], counts: Sequence[int]) -> list[tuple[str, int, int]]:
    """
    (model, input tokens at most, output tokens at most) of each request, from the exact counts
    of prompts built with an empty explanation (detection and fuzzing prompts get room for any
    explanation; the explain and token prompts are counted as they are).
    """
    out = []
    for prompt, count in zip(prompts, counts):
        extra = MAX_EXPLANATION_TOKENS if prompt.kind in ("detection", "fuzzing") else 0
        out.append((prompt.params["model"], count + extra, prompt.params["max_tokens"]))
    return out


def planned_prompts(sets: FeatureSets, explainer: str, scorer: str, seed: int = 0) -> list[Prompt]:
    """Every request a run may send, scoring ones with an empty explanation (for request_bounds)."""
    prompts = [explain_prompt(sets, explainer, seed)]
    if sets.scorable:
        prompts += scoring_prompts(sets, "", scorer, seed)
    return prompts


# Replies

def parse_explanation(reply: Reply) -> Optional[str]:
    """The explanation, whitespace collapsed and capped, with the prompts' lookalikes for < and > read back."""
    data = reply.json()
    text = data.get("explanation") if data else None
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).replace("‹", "<").replace("›", ">").strip()
    return text[:MAX_EXPLANATION_CHARS] or None


def parse_answers(reply: Reply, n: int) -> tuple[list[Optional[bool]], Optional[str]]:
    """The reply's n booleans (None where missing), and what was wrong with it, if anything."""
    data = reply.json()
    values = data.get("answers") if data else None
    if not isinstance(values, list):
        if reply.refused:
            return [None] * n, "refused"
        return [None] * n, reply.error or "no answers"
    answers = [v if isinstance(v, bool) else None for v in values[:n]] + [None] * max(0, n - len(values))
    return answers, None if len(values) == n else f"{len(values)} answers for {n} examples"


@dataclass
class FeatureResult:
    feature: int
    explainer: str
    scorer: str
    status: str = "failed"  # scored, unscored, refused or failed
    explanation: Optional[str] = None
    note: Optional[str] = None
    detection: Optional[float] = None
    detection_hard: Optional[float] = None
    fuzzing: Optional[float] = None
    score: Optional[float] = None
    token_detection: Optional[float] = None
    decile_recall: Optional[list] = None
    details: dict = field(default_factory=dict)
    verdicts: Optional[dict] = None
    usage: dict[str, Usage] = field(default_factory=dict)
    # Requests that may have been billed with no usage reported (charged their upper bound), by
    # position in planned_prompts' order: 0 the explain request, then the scoring requests
    unreported: list[int] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, replies: Sequence[Reply], start: int) -> None:
        for j, reply in enumerate(replies):
            self.usage[reply.model] = self.usage.get(reply.model, Usage()) + reply.usage
            if reply.maybe_billed:
                self.unreported.append(start + j)

    @property
    def total_usage(self) -> dict[str, dict]:
        return {model: usage.to_dict() for model, usage in self.usage.items()}


def score_replies(result: FeatureResult, prompts: Sequence[Prompt], replies: Sequence[Reply]) -> None:
    """Fill in a result's scores and verdicts from its scoring prompts' replies."""
    verdicts: dict[str, list] = {"detection": [], "fuzzing": [], "token": []}
    deciles: list[Optional[int]] = []
    refusals = 0
    for prompt, reply in zip(prompts, replies):
        answers, problem = parse_answers(reply, len(prompt.items))
        if problem == "refused":
            refusals += 1
        elif problem:
            result.problems.append(f"{prompt.kind}: {problem}")
        for item, verdict in zip(prompt.items, answers):
            if prompt.kind == "fuzzing":
                verdicts["fuzzing"].append([item.index, item.truth, verdict, item.marks])
            else:
                verdicts[prompt.kind].append([item.group, item.index, item.truth, verdict])
                if prompt.kind == "detection":
                    deciles.append(item.decile)

    def columns(kind: str, groups: Optional[tuple[str, ...]] = None) -> tuple[list[bool], list[Optional[bool]]]:
        rows = [v for v in verdicts[kind] if groups is None or v[0] in groups]
        return [r[2] for r in rows], [r[3] for r in rows]

    truth, verdict = columns("detection")
    result.detection = scoring.balanced_accuracy(truth, verdict)
    result.details["detection_ci"] = scoring.interval(truth, verdict)
    result.detection_hard = scoring.balanced_accuracy(*columns("detection", ("activating", "token")))
    result.decile_recall = scoring.decile_recall(deciles, truth, verdict)

    fuzz_truth = [r[1] for r in verdicts["fuzzing"]]
    fuzz_verdict = [r[2] for r in verdicts["fuzzing"]]
    result.fuzzing = scoring.balanced_accuracy(fuzz_truth, fuzz_verdict)
    result.details["fuzzing_ci"] = scoring.interval(fuzz_truth, fuzz_verdict)
    result.score = scoring.mean(result.detection, result.fuzzing)
    result.token_detection = scoring.balanced_accuracy(*columns("token")) if verdicts["token"] else None

    result.details["refused_prompts"] = refusals
    result.details["examples"] = {kind: len(rows) for kind, rows in verdicts.items()}
    result.verdicts = verdicts


def apply_explanation(result: FeatureResult, sets: FeatureSets, reply: Reply) -> bool:
    """Record the explainer's reply in a result; returns whether the explanation is to be scored."""
    result.add([reply], 0)
    if reply.refused:
        result.status, result.note = "refused", "The explainer declined these contexts"
        return False
    result.explanation = parse_explanation(reply)
    if result.explanation is None:
        result.problems.append(f"explain: {reply.error or 'no explanation'}")
        return False
    if not sets.scorable:
        result.status, result.note = "unscored", sets.note
        return False
    return True


def apply_scores(result: FeatureResult, prompts: Sequence[Prompt], replies: Sequence[Reply], start: int = 1) -> None:
    """
    Score a result's explanation from its scoring prompts' replies (failed ones drop their
    examples). start: the first scoring request's position among the run's requests (1 after an
    explain request, 0 for a hypothesis).
    """
    result.add(replies, start)
    failed = [r for r in replies if r.error]
    if failed:
        result.problems.append(f"{len(failed)} scoring requests failed ({failed[0].error})")
    score_replies(result, prompts, replies)
    if result.detection is not None and result.fuzzing is not None:
        result.status = "scored"
    elif result.details.get("refused_prompts"):
        result.status, result.note = "refused", "The scorer declined these contexts"
    else:
        result.status, result.note = "unscored", "Scoring failed"


async def run_feature(
    client: ClaudeClient,
    sets: FeatureSets,
    explainer: str,
    scorer: str,
    seed: int = 0,
    deadline: Optional[float] = None,
) -> FeatureResult:
    """Explain a feature, then score the explanation (when its sets allow) before the deadline."""
    result = FeatureResult(sets.feature, explainer, scorer)
    [reply] = await client.send_all([explain_prompt(sets, explainer, seed).params], deadline)
    if not apply_explanation(result, sets, reply):
        return result

    prompts = scoring_prompts(sets, result.explanation, scorer, seed)
    replies = await client.send_all([p.params for p in prompts], deadline)
    failed = [r for r in replies if r.error]
    if failed:
        result.add(replies, 1)
        result.problems.append(f"{len(failed)} scoring requests failed ({failed[0].error})")
        return result
    apply_scores(result, prompts, replies)
    return result


# Hypotheses

def clean_text(text: str, limit: int = MAX_EXPLANATION_CHARS) -> str:
    """A user's explanation as it's scored and stored: whitespace collapsed, capped like an explanation."""
    return " ".join(text.split())[:limit].strip()


def hypothesis_prompts(sets: FeatureSets, text: str, scorer: str, seed: int = 0, baseline: bool = False) -> list[Prompt]:
    """
    The scoring requests for a user's explanation: the same detection and fuzzing prompts an
    autointerp explanation gets with this seed (and the token baseline's, if asked).
    """
    return scoring_prompts(sets, text, scorer, seed, baseline)


def exact_bounds(prompts: Sequence[Prompt], counts: Sequence[int]) -> list[tuple[str, int, int]]:
    """(model, input tokens, output tokens at most) of requests whose text is final (counted as they'll be sent)."""
    return [(prompt.params["model"], count, prompt.params["max_tokens"]) for prompt, count in zip(prompts, counts)]


async def run_hypothesis(
    client: ClaudeClient,
    sets: FeatureSets,
    text: str,
    scorer: str,
    seed: int = 0,
    deadline: Optional[float] = None,
    baseline: bool = False,
) -> FeatureResult:
    """
    Score a user's explanation of a feature on its held-out set. The result's status is scored,
    refused, unscored (the scorer's answers couldn't be read) or failed (a request failed; like
    run_feature, the caller decides what to keep).
    """
    result = FeatureResult(sets.feature, "", scorer, explanation=text)
    prompts = hypothesis_prompts(sets, text, scorer, seed, baseline)
    replies = await client.send_all([p.params for p in prompts], deadline)
    failed = [r for r in replies if r.error]
    if failed:
        result.add(replies, 0)
        result.problems.append(f"{len(failed)} scoring requests failed ({failed[0].error})")
        return result
    apply_scores(result, prompts, replies, start=0)
    return result


# Generation tests

def generation_prompt(explanation: str, generator: str) -> Prompt:
    """The request writing a generation test's texts for an explanation."""
    user = generation_message(escape(explanation))
    return Prompt("generate", request_params(generator, GENERATE_SYSTEM, user, GENERATION_SCHEMA, MODELS[generator].generate_max_tokens))


def _texts(values, limit: int) -> list[str]:
    """Up to limit distinct texts from a reply's list: whitespace collapsed, the prompts' lookalikes read back, cut to size."""
    out: list[str] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str):
            continue
        text = " ".join(value.split()).replace("‹", "<").replace("›", ">")[:GENERATION_TEXT_CHARS].strip()
        if text and text not in out:
            out.append(text)
        if len(out) == limit:
            break
    return out


def parse_generation(reply: Reply) -> Optional[tuple[list[str], list[str]]]:
    """(texts the explanation says the feature fires on, near-misses), or None if the reply has neither kind."""
    data = reply.json()
    if not data:
        return None
    firing = _texts(data.get("firing"), GENERATION_FIRING)
    near = [t for t in _texts(data.get("near_misses"), GENERATION_NEAR_MISSES + len(firing)) if t not in firing][:GENERATION_NEAR_MISSES]
    if not firing or not near:
        return None
    return firing, near


def generation_result(firing: Sequence[float], near_misses: Sequence[float], maximum: float, threshold: float = GENERATION_THRESHOLD) -> dict:
    """
    A generation test's result, from the feature's strongest activation on each text (peaks)
    and its maximum over the corpus: the share of the firing texts that reach threshold × maximum
    on some token, and the share of the near-misses that stay below it.
    """
    cut = threshold * maximum
    hits = sum(1 for peak in firing if peak >= cut)
    quiet = sum(1 for peak in near_misses if peak < cut)
    return {
        "threshold": threshold,
        "maximum": maximum,
        "firing": len(firing),
        "firing_hits": hits,
        "firing_rate": hits / len(firing) if firing else None,
        "near_misses": len(near_misses),
        "near_misses_quiet": quiet,
        "near_miss_rate": quiet / len(near_misses) if near_misses else None,
    }
