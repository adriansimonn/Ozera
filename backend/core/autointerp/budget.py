"""
Autointerp's models, limits and token estimates.

Claude explains and scores features (see core.autointerp.pipeline). Each model is called at the
lowest thinking setting it allows, since chain of thought didn't help explainers (Paulo et al.,
ICML '25): none on Haiku 4.5, between_tools on Sonnet 5.5, effort low on Opus 5.5 (whose
thinking can't be turned off). Prices are in services/credit_service.py (CLAUDE_PRICES).
"""

from dataclasses import dataclass
from typing import Optional


class AutointerpInputError(ValueError):
    """A request autointerp can't serve (shown to the user)."""


@dataclass(frozen=True)
class ClaudeModel:
    id: str
    name: str
    # Sent as the request's `thinking` (None leaves it out)
    thinking: Optional[dict]
    # output_config.effort (None leaves it out; Haiku 4.5 has no effort setting)
    effort: Optional[str]
    # Output tokens a request may produce, thinking included: the explanation's JSON, or a
    # scoring prompt's booleans
    explain_max_tokens: int
    score_max_tokens: int
    # Thinking tokens a request is estimated to produce on top of its answer
    explain_thinking_tokens: int = 0
    score_thinking_tokens: int = 0
    # Output tokens of a generation test's texts, and of a community's name (thinking included)
    generate_max_tokens: int = 2000
    name_max_tokens: int = 100
    # Input tokens per Haiku 4.5 token on the same text (the token constants below are in Haiku
    # tokens). Measured 1.35 for Sonnet 5.5 (calibration, 2026-11-12); assumed the same for
    # Opus 5.5, which shares its tokenizer generation
    input_token_factor: float = 1.0


HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"

MODELS = {
    HAIKU: ClaudeModel(HAIKU, "Claude Haiku 4.5", None, None, 200, 256),
    SONNET: ClaudeModel(SONNET, "Claude Sonnet 5.5", {"type": "between_tools"}, None, 200, 256, input_token_factor=1.35),
    OPUS: ClaudeModel(OPUS, "Claude Opus 5.5", None, "low", 4096, 4096, 300, 250, generate_max_tokens=8192, name_max_tokens=4096, input_token_factor=1.35),
}

# Defaults (decided 2026-10-16): Sonnet 5.5 explains, Haiku 4.5 scores
DEFAULT_EXPLAINER = SONNET
DEFAULT_SCORER = HAIKU
# Models a single feature may be explained and scored with (synchronously, at full price);
# MIN_AUTOINTERP_CHARGE covers the priciest pair of them. Jobs may use any of MODELS.
SINGLE_FEATURE_MODELS = (HAIKU, SONNET)
JOB_MODELS = tuple(MODELS)


def model(model_id: str) -> ClaudeModel:
    if model_id not in MODELS:
        raise AutointerpInputError(f"Unknown model: {model_id}")
    return MODELS[model_id]


# Tokens per feature, measured by calibration (2026-11-12: Gemma 3 270M, Gemma Scope 2 layer 9
# 16k, 264 features and controls; input in Haiku 4.5 tokens, see input_token_factor). Sonnet 5.5
# explained (3,638 input, 49 output) and Haiku 4.5 scored (14,049 input, 546 output).
# Scoring covers detection (5 prompts), fuzzing (3) and the token baseline's detection (5).
EXPLAIN_INPUT_TOKENS = 2_700
EXPLAIN_OUTPUT_TOKENS = 50
SCORE_INPUT_TOKENS = 14_050
SCORE_OUTPUT_TOKENS = 550
SCORE_PROMPTS = 13
# Of those, the prompts that carry the explanation (detection and fuzzing)
EXPLAINED_PROMPTS = 8


def estimated_tokens(explainer: str, scorer: str, scored: bool = True) -> dict[str, dict[str, int]]:
    """{model: {"input", "output"}} one feature is estimated to use."""
    out: dict[str, dict[str, int]] = {}

    def add(model_id: str, inp: int, outp: int) -> None:
        entry = out.setdefault(model_id, {"input": 0, "output": 0})
        entry["input"] += inp
        entry["output"] += outp

    add(explainer, round(EXPLAIN_INPUT_TOKENS * MODELS[explainer].input_token_factor), EXPLAIN_OUTPUT_TOKENS + MODELS[explainer].explain_thinking_tokens)
    if scored:
        add(scorer, round(SCORE_INPUT_TOKENS * MODELS[scorer].input_token_factor), SCORE_OUTPUT_TOKENS + SCORE_PROMPTS * MODELS[scorer].score_thinking_tokens)
    return out


def estimated_score_tokens(scorer: str, baseline: bool = True) -> dict[str, int]:
    """{"input", "output"} one scoring pass is estimated to use: detection and fuzzing, and the token baseline's detection if asked."""
    prompts = SCORE_PROMPTS if baseline else EXPLAINED_PROMPTS
    return {
        "input": round(SCORE_INPUT_TOKENS * MODELS[scorer].input_token_factor * prompts / SCORE_PROMPTS),
        "output": round(SCORE_OUTPUT_TOKENS * prompts / SCORE_PROMPTS) + prompts * MODELS[scorer].score_thinking_tokens,
    }


def estimated_bound_tokens(explainer: str, scorer: str, scored: bool = True) -> dict[str, dict[str, int]]:
    """
    {model: {"input", "output"}}: an estimate of the upper bound a job holds per feature (the
    estimated input, room for any explanation in detection and fuzzing prompts, and every
    request's max_tokens of output). The hold itself comes from counting the job's prompts.
    """
    out = {model_id: {"input": t["input"], "output": 0} for model_id, t in estimated_tokens(explainer, scorer, scored).items()}
    out[explainer]["output"] += MODELS[explainer].explain_max_tokens
    if scored:
        out[scorer]["input"] += EXPLAINED_PROMPTS * MAX_EXPLANATION_TOKENS
        out[scorer]["output"] += SCORE_PROMPTS * MODELS[scorer].score_max_tokens
    return out


# Eligibility. An SAE is refused when its atlas L0 is over this share of its width (built-in
# nano/mini SAEs measure 72–96%, Gemma Scope about 0.1–1%); a retrained one qualifies with no
# code change. Features are explained but left unscored when fewer activating contexts than
# MIN_SCORE_SEQUENCES are held out for scoring, or fewer non-activating windows than
# MIN_RANDOM_NEGATIVES can be found.
DENSE_L0_FRACTION = 0.05
MIN_SCORE_SEQUENCES = 20
MIN_RANDOM_NEGATIVES = 15

# An explanation whose detection doesn't beat the token baseline's by this much is tagged a
# token feature: calibration's rerun put the 95% spread of (detection − token detection) at
# ±0.088 (2026-11-12; 115 of 200 features tagged, against 94 at the provisional 0.05)
TOKEN_MARGIN = 0.088
# How much an explanation's score moves when the feature is explained and scored again (95%,
# calibration's rerun, 2026-11-12). Shown as the score's ±: the per-score intervals (mean 0.064)
# count only the scorer's sampling, not a new explanation
SCORE_PM = 0.076

# Explanations: words the explainer is asked for, and characters kept. Labels are meant to stay
# within 15 words; Sonnet 5.5 overshoots a stated limit by a few (asked for 15, the pilot's ran 20)
EXPLANATION_WORDS = 12
MAX_EXPLANATION_CHARS = 200
# Upper bound on the tokens of any explanation (at least one byte per token, 4 bytes a character)
MAX_EXPLANATION_TOKENS = 4 * MAX_EXPLANATION_CHARS

# Contexts per scoring prompt
DETECTION_PER_PROMPT = 20
FUZZING_PER_PROMPT = 17

# One feature, explained and scored synchronously: requests in flight at once, the SDK's
# per-request timeout and retries, and the deadline for the whole run (counting tokens,
# explaining, scoring). The frontend's explain timeout (AUTOINTERP_EXPLAIN_TIMEOUT_MS in
# frontend/src/api/client.ts) is derived from EXPLAIN_DEADLINE_SECONDS + CLAIM_GRACE_SECONDS.
PARALLEL_REQUESTS = 16
REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 2
EXPLAIN_DEADLINE_SECONDS = 240
# A run claimed this long ago that hasn't finished is taken to have died
CLAIM_GRACE_SECONDS = 30

# Features a label lookup may ask for at once
MAX_LABEL_FEATURES = 5000

# Whole-SAE jobs (core.autointerp.jobs), through the Message Batches API at half price. A job
# explains at most MAX_JOB_FEATURES features (plus the random-direction controls), and scores
# all of them or a sample of SCORE_SAMPLE (scoring is about 85% of the tokens). Every scoring
# request sits in the batch queue at once, which Anthropic limits per rate-limit tier (100k
# requests on tier 1, 500k on tier 4): MAX_JOB_FEATURES × 13 must fit the workspace's tier.
MAX_JOB_FEATURES = 16_384
DEFAULT_JOB_TOP = 1_000
SCORE_SAMPLE = 200
# Features whose explanations a job's results file holds per part (the backend imports a part at once)
RESULTS_PART = 1_000
# Upper bounds without a request per prompt: prompts' texts are counted together, as blocks of one
# message of at most COUNT_GROUP_BYTES (under the 200k context of every model, even at a token
# per byte), and each prompt is allowed COUNT_MARGIN_TOKENS for tokens merged across the joins
COUNT_GROUP_BYTES = 150_000
COUNT_MARGIN_TOKENS = 16
# The backend asks Anthropic how a job's batches are doing at most this often
BATCH_CHECK_SECONDS = 30

# Testing explanations (AUTOINTERP_PLAN.md, Phase 3)

# A hypothesis is a user's own explanation of a feature, scored by the same scorer on the same
# held-out set as autointerp's; it's kept as short as an explanation
MAX_HYPOTHESIS_CHARS = MAX_EXPLANATION_CHARS
# A user's hypotheses, a feature's explanations and its generation tests listed at most
HYPOTHESES_SHOWN = 20
VERSIONS_SHOWN = 20
TESTS_SHOWN = 20

# Generation test (SAEExplainer's generative accuracy, for one explanation): Claude writes
# GENERATION_FIRING texts the explanation says the feature fires on and GENERATION_NEAR_MISSES
# near-misses it says it doesn't; the model and SAE read them in one GPU call. A text counts as
# firing when the feature reaches GENERATION_THRESHOLD of its maximum over the corpus on some
# token of it. Texts are cut to GENERATION_TEXT_CHARS characters, and read up to
# GENERATION_TEXT_TOKENS tokens each (an atlas's sequence length).
GENERATION_FIRING = 10
GENERATION_NEAR_MISSES = 5
GENERATION_THRESHOLD = 0.5
GENERATION_TEXT_CHARS = 400
GENERATION_TEXT_TOKENS = 256
# Tokens the request writing the texts is estimated to use (replaced by measured means, like
# the constants above), and the longest it may take
GENERATE_INPUT_TOKENS = 450
GENERATE_OUTPUT_TOKENS = 900
GENERATION_DEADLINE_SECONDS = 120

# Atlas community names: one request per community, from its features' explanations (the
# COMMUNITY_MEMBERS most active explained ones), the tokens it fires on and COMMUNITY_CONTEXTS of
# its contexts. A community is named once COMMUNITY_MIN_SHARE of its features (and at least
# COMMUNITY_MIN_EXPLAINED) have explanations: a name from a few of them would stand for the rest.
COMMUNITY_MEMBERS = 24
COMMUNITY_CONTEXTS = 12
COMMUNITY_TOP_WORDS = 8
COMMUNITY_MIN_SHARE = 0.5
COMMUNITY_MIN_EXPLAINED = 2
COMMUNITY_NAME_WORDS = 6
MAX_COMMUNITY_NAME_CHARS = 80
# Tokens a naming request is estimated to use
NAME_INPUT_TOKENS = 1_300
NAME_OUTPUT_TOKENS = 20


def estimated_generation_tokens(generator: str) -> dict[str, int]:
    """{"input", "output"} the request writing a generation test's texts is estimated to use."""
    return {"input": GENERATE_INPUT_TOKENS, "output": GENERATE_OUTPUT_TOKENS + MODELS[generator].explain_thinking_tokens}


def estimated_naming_tokens(namer: str, bound: bool = False) -> dict[str, int]:
    """{"input", "output"} naming one community is estimated to use (bound: its max_tokens of output)."""
    output = MODELS[namer].name_max_tokens if bound else NAME_OUTPUT_TOKENS + MODELS[namer].explain_thinking_tokens
    return {"input": NAME_INPUT_TOKENS, "output": output}
