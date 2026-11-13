"""
Autointerp on the backend: one feature explained and scored synchronously (core.autointerp on
Claude, about 14 requests in parallel), and the explanations Postgres keeps for everyone with
the SAE.

Example sets come from v2 atlases on the SAE volume (read through services.atlas_store's
cache). Explaining a feature is charged its usage, at least MIN_AUTOINTERP_CHARGE: the upper
bound (exact input counted by the API, plus max_tokens of output) is charged when it starts and
whatever it didn't use is refunded when it ends, so a run is never charged less than it cost.
Two requests for the same feature at once run it once: the first claims it (a "running" row,
unique per feature, models and versions), the second waits for its result.

Testing explanations (AUTOINTERP_PLAN.md, Phase 3), charged the same way (a hold, settled on
usage): a user's hypothesis scored on a feature's held-out set (score_hypothesis, private), and
a generation test of an explanation or a hypothesis (generation_test: Claude writes texts, a GPU
call reads them through the model and SAE). Also here: a feature's explanations over time
(versions; the newest is the one shown) and an atlas's community names.
"""

import asyncio
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.autointerp import examples as ai
from core.autointerp import scoring
from core.autointerp.budget import (
    CLAIM_GRACE_SECONDS,
    EXPLAIN_DEADLINE_SECONDS,
    GENERATION_DEADLINE_SECONDS,
    GENERATION_THRESHOLD,
    HYPOTHESES_SHOWN,
    MODELS,
    TESTS_SHOWN,
    VERSIONS_SHOWN,
    AutointerpInputError,
)
from core.autointerp.client import ClaudeClient
from core.autointerp.pipeline import (
    FeatureResult,
    clean_text,
    exact_bounds,
    generation_prompt,
    generation_result,
    hypothesis_prompts,
    parse_generation,
    planned_prompts,
    request_bounds,
    run_feature,
    run_hypothesis,
)
from core.autointerp.prompts import COMMUNITY_PROMPT_VERSION, GENERATION_PROMPT_VERSION, PROMPT_VERSION
from core.geometry.atlas import CATALOG_COLUMNS
from models.database import CommunityName, FeatureExplanation, FeatureHypothesis, GenerationTest, TransactionType
from services import atlas_store
from services.credit_service import (
    MIN_AUTOINTERP_CHARGE,
    MIN_HYPOTHESIS_CHARGE,
    InsufficientBalanceError,
    autointerp_bound,
    calculate_autointerp_cost,
    charge_flat,
    credit_refund,
)

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = "openwebtext"
FINAL = ("scored", "unscored", "refused")
# A feature's percentile among the random-direction controls needs at least this many scored
MIN_CONTROLS = 8
# Seed of the prompts' order and fuzzing marks for explanations served to users
SEED = 0


class AutointerpUnavailable(Exception):
    """No Anthropic API key is configured."""


class AutointerpFailed(Exception):
    """A run failed; charged is what it was charged for the tokens it used."""

    def __init__(self, message: str, charged: float):
        super().__init__(message)
        self.charged = charged


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


_client: Optional[ClaudeClient] = None


def claude_client() -> ClaudeClient:
    global _client
    if not available():
        raise AutointerpUnavailable()
    if _client is None:
        _client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


# Atlases

@dataclass
class Atlas:
    """A stored atlas, as autointerp reads it."""

    directory: str  # also its key
    sae_key: str
    model_id: str
    corpus: str
    meta: dict

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def build(self) -> dict:
        return {key: self.meta.get(key) for key in ("created_at", "tokens", "seed", "version")}


def state(meta: dict) -> tuple[str, Optional[str]]:
    """Whether an atlas's features can be explained: ("ready" | "recompute" | "dense", why not)."""
    if int(meta.get("version", 1)) < 2:
        return "recompute", "This atlas predates autointerp; recompute it to explain features"
    if not meta.get("autointerp"):
        l0, width = meta.get("l0"), meta.get("width")
        detail = f" (L0 {l0:.0f} of {width} features)" if l0 is not None and width else ""
        return "dense", f"This SAE is too dense to explain{detail}"
    return "ready", None


_files: "OrderedDict[tuple[str, str], ai.AtlasFiles]" = OrderedDict()
FILES_CACHED = 4


async def atlas_files(atlas: Atlas) -> ai.AtlasFiles:
    """An atlas's corpus sample (parsed once per build, a few at a time)."""
    key = (atlas.directory, str(atlas.meta.get("created_at")))
    if key in _files:
        _files.move_to_end(key)
        return _files[key]
    base = f"{atlas.directory}/{ai.DIRECTORY}"
    tokens, read, vocabulary = await asyncio.gather(
        atlas_store.read_bytes(f"{base}/tokens.bin"),
        atlas_store.read_bytes(f"{base}/read.bin"),
        atlas_store.read_bytes(f"{base}/vocabulary.json"),
    )
    files = ai.load_files(tokens, read, vocabulary, int(atlas.meta["seq_len"]), int(atlas.meta["skip"]))
    _files[key] = files
    while len(_files) > FILES_CACHED:
        _files.popitem(last=False)
    return files


async def feature_sets(atlas: Atlas, feature: int) -> Optional[ai.FeatureSets]:
    """A feature's example sets (None if it was never active over the corpus)."""
    try:
        data = await atlas_store.read_bytes(ai.shard_path(atlas.directory, feature))
    except FileNotFoundError:
        return None
    shard = ai.Shard.from_bytes(data)
    if feature not in shard:
        return None
    return ai.feature_sets(await atlas_files(atlas), shard, feature)


# Stored explanations

def _final(db: Session, atlas_key: str, feature: int):
    return db.query(FeatureExplanation).filter(
        FeatureExplanation.atlas_key == atlas_key,
        FeatureExplanation.feature == feature,
        FeatureExplanation.status.in_(FINAL),
    )


def stored(db: Session, atlas_key: str, feature: int, explainer: str, scorer: str) -> Optional[FeatureExplanation]:
    """The newest finished explanation of a feature with these models and the current versions."""
    return (
        _final(db, atlas_key, feature)
        .filter(
            FeatureExplanation.explainer_model == explainer,
            FeatureExplanation.scorer_model == scorer,
            FeatureExplanation.prompt_version == PROMPT_VERSION,
            FeatureExplanation.examples_version == ai.EXAMPLES_VERSION,
        )
        .order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc())
        .first()
    )


def newest(db: Session, atlas_key: str, feature: int) -> Optional[FeatureExplanation]:
    """The explanation a feature shows: its newest, whatever the models (never the best-scoring one)."""
    return _final(db, atlas_key, feature).order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc()).first()


def control_scores(db: Session, atlas_key: str, explainer: Optional[str] = None, scorer: Optional[str] = None) -> list[float]:
    """
    Scores of the atlas's random-direction controls (each one's newest): those explained and
    scored with the given models when at least MIN_CONTROLS are, else all of them.
    """
    rows = (
        db.query(FeatureExplanation.feature, FeatureExplanation.score, FeatureExplanation.explainer_model, FeatureExplanation.scorer_model)
        .filter(
            FeatureExplanation.atlas_key == atlas_key,
            FeatureExplanation.is_control.is_(True),
            FeatureExplanation.status == "scored",
        )
        .order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc())
        .all()
    )

    def newest(matching) -> list[float]:
        seen: dict[int, float] = {}
        for r in matching:
            if r.score is not None:
                seen.setdefault(r.feature, r.score)
        return list(seen.values())

    if explainer is not None and scorer is not None:
        same = newest(r for r in rows if r.explainer_model == explainer and r.scorer_model == scorer)
        if len(same) >= MIN_CONTROLS:
            return same
    return newest(rows)


def control_percentile(
    db: Session, atlas_key: str, score: Optional[float], explainer: Optional[str] = None, scorer: Optional[str] = None,
) -> tuple[Optional[float], int]:
    """A score's percentile among the atlas's scored controls (None with fewer than MIN_CONTROLS), and how many there are."""
    controls = control_scores(db, atlas_key, explainer, scorer)
    if len(controls) < MIN_CONTROLS:
        return None, len(controls)
    return scoring.percentile(score, controls), len(controls)


def explained_count(db: Session, atlas_key: str) -> int:
    return (
        db.query(func.count(func.distinct(FeatureExplanation.feature)))
        .filter(FeatureExplanation.atlas_key == atlas_key, FeatureExplanation.is_control.is_(False), FeatureExplanation.status.in_(FINAL))
        .scalar()
    ) or 0


def label_atlas(db: Session, sae_key: str, model_id: Optional[str]) -> Optional[tuple[str, str, str]]:
    """
    The atlas whose explanations label an SAE's features (on a model, if given): its
    OpenWebText atlas's when it has explanations, else the one explained most recently.
    Returns (atlas key, model, corpus).
    """
    query = db.query(
        FeatureExplanation.atlas_key, FeatureExplanation.model_id, FeatureExplanation.corpus,
        func.max(FeatureExplanation.created_at).label("latest"),
    ).filter(
        FeatureExplanation.sae_key == sae_key,
        FeatureExplanation.is_control.is_(False),
        FeatureExplanation.status.in_(FINAL),
    )
    if model_id is not None:
        query = query.filter(FeatureExplanation.model_id == model_id)
    atlases = query.group_by(FeatureExplanation.atlas_key, FeatureExplanation.model_id, FeatureExplanation.corpus).all()
    if not atlases:
        return None
    atlases.sort(key=lambda a: (a.corpus == DEFAULT_CORPUS, a.latest), reverse=True)
    return atlases[0].atlas_key, atlases[0].model_id, atlases[0].corpus


def labels(db: Session, atlas_key: str, features: list[int]) -> list:
    """Each feature's newest explanation on an atlas (the columns labels show, not the verdicts)."""
    rows = (
        db.query(
            FeatureExplanation.feature, FeatureExplanation.explanation, FeatureExplanation.status,
            FeatureExplanation.score, FeatureExplanation.detection, FeatureExplanation.fuzzing,
            FeatureExplanation.token_detection, FeatureExplanation.explainer_model, FeatureExplanation.scorer_model,
        )
        .filter(
            FeatureExplanation.atlas_key == atlas_key,
            FeatureExplanation.feature.in_(features),
            FeatureExplanation.is_control.is_(False),
            FeatureExplanation.status.in_(("scored", "unscored")),
            FeatureExplanation.explanation.isnot(None),
        )
        .order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc())
        .all()
    )
    seen: dict[int, object] = {}
    for row in rows:
        seen.setdefault(row.feature, row)
    return list(seen.values())


def newest_ids(db: Session, atlas_key: str, controls: bool = False, models: Optional[tuple[str, str]] = None):
    """
    A subquery of the IDs of each feature's newest explanation on an atlas (the one it shows), of
    its features or its controls, among those with the given (explainer, scorer) if any.
    """
    query = db.query(
        FeatureExplanation.id,
        func.row_number().over(
            partition_by=FeatureExplanation.feature,
            order_by=(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc()),
        ).label("rank"),
    ).filter(
        FeatureExplanation.atlas_key == atlas_key,
        FeatureExplanation.is_control.is_(controls),
        FeatureExplanation.status.in_(FINAL),
    )
    if models is not None:
        query = query.filter(FeatureExplanation.explainer_model == models[0], FeatureExplanation.scorer_model == models[1])
    ranked = query.subquery()
    return db.query(ranked.c.id).filter(ranked.c.rank == 1)


def search_filter(db: Session, text: str):
    """Explanations matching a search: Postgres full-text search (its GIN index), or every word as a substring elsewhere."""
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        document = func.to_tsvector("english", func.coalesce(FeatureExplanation.explanation, ""))
        return document.op("@@")(func.plainto_tsquery("english", text))
    from sqlalchemy import and_

    return and_(*(FeatureExplanation.explanation.ilike(f"%{word}%") for word in text.split()))


def context_json(ctx: ai.Context, marks: Optional[list[int]] = None) -> dict:
    """A context as the Feature view shows it (marks: fuzzing's highlighted positions, if any)."""
    return {
        "tokens": ctx.tokens,
        "levels": ctx.levels,
        "read": ctx.read,
        "centre": ctx.centre,
        "act": round(ctx.act, 4),
        "decile": ctx.decile,
        "marks": marks,
    }


def _refund(db: Session, user_id: int, hold: float, charged: float, description: str) -> float:
    """
    Give back what a run didn't use of its hold (not committed). Returns what the run cost the
    user in the end: the charge, or the whole hold when the difference is under a cent (the
    ledger doesn't record refunds that small).
    """
    refunded = credit_refund(db, user_id, hold - charged, TransactionType.AUTOINTERP_REFUND, description)
    return charged if refunded is not None else hold


# Claiming a feature, so concurrent requests run it once

def _stale_before() -> datetime:
    return datetime.utcnow() - timedelta(seconds=EXPLAIN_DEADLINE_SECONDS + CLAIM_GRACE_SECONDS)


def _running(db: Session, atlas_key: str, feature: int, explainer: str, scorer: str) -> Optional[FeatureExplanation]:
    return db.query(FeatureExplanation).filter(
        FeatureExplanation.atlas_key == atlas_key,
        FeatureExplanation.feature == feature,
        FeatureExplanation.explainer_model == explainer,
        FeatureExplanation.scorer_model == scorer,
        FeatureExplanation.prompt_version == PROMPT_VERSION,
        FeatureExplanation.examples_version == ai.EXAMPLES_VERSION,
        FeatureExplanation.status == "running",
    ).first()


def claim(db: Session, atlas: Atlas, feature: int, explainer: str, scorer: str, user_id: int) -> Optional[FeatureExplanation]:
    """Claim a feature's run (removing a claim whose run died); None if another request holds it."""
    for _ in range(2):
        row = FeatureExplanation(
            atlas_key=atlas.directory, sae_key=atlas.sae_key, model_id=atlas.model_id, corpus=atlas.corpus,
            atlas=atlas.build, feature=feature, is_control=False, explainer_model=explainer, scorer_model=scorer,
            prompt_version=PROMPT_VERSION, examples_version=ai.EXAMPLES_VERSION, status="running", user_id=user_id,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        try:
            db.commit()
            return row
        except IntegrityError:
            db.rollback()
        other = _running(db, atlas.directory, feature, explainer, scorer)
        if other is None or other.created_at >= _stale_before():
            if other is not None:
                return None
            continue
        logger.warning("Removing a dead autointerp claim on %s feature %d", atlas.directory, feature)
        db.delete(other)
        db.commit()
    return None


async def wait_for(db: Session, atlas_key: str, feature: int, explainer: str, scorer: str) -> Optional[FeatureExplanation]:
    """The result of another request's run of a feature, once it ends (None if it failed or never ends)."""
    until = time.monotonic() + EXPLAIN_DEADLINE_SECONDS + CLAIM_GRACE_SECONDS
    while time.monotonic() < until:
        await asyncio.sleep(1.0)
        db.expire_all()
        done = stored(db, atlas_key, feature, explainer, scorer)
        if done is not None:
            return done
        if _running(db, atlas_key, feature, explainer, scorer) is None:
            db.commit()
            return None
        db.commit()
    return None


def _release(db: Session, row: FeatureExplanation) -> None:
    try:
        db.rollback()
        db.query(FeatureExplanation).filter(FeatureExplanation.id == row.id, FeatureExplanation.status == "running").delete()
        db.commit()
    except Exception:
        logger.exception("Couldn't release the autointerp claim %s", row.id)
        db.rollback()


def _save(db: Session, row: FeatureExplanation, result: FeatureResult, cost: float) -> None:
    row.status = result.status
    row.explanation = result.explanation
    row.note = (result.note or None) and result.note[:500]
    row.detection = result.detection
    row.detection_hard = result.detection_hard
    row.fuzzing = result.fuzzing
    row.score = result.score
    row.token_detection = result.token_detection
    row.decile_recall = result.decile_recall
    row.details = {**result.details, "problems": result.problems[:20]}
    row.verdicts = result.verdicts
    row.usage = result.total_usage
    row.cost_usd = round(cost, 4)


async def explain(
    db: Session,
    user_id: int,
    atlas: Atlas,
    feature: int,
    explainer: str,
    scorer: str,
    description: str,
) -> tuple[FeatureExplanation, float, bool]:
    """
    A feature's explanation with these models: the stored one, or a new run (charged).

    Returns (the row, what this request was charged, whether it was already stored).

    Raises:
        AutointerpInputError: the feature can't be explained (shown to the user)
        AutointerpUnavailable: no API key
        InsufficientBalanceError: the balance doesn't cover the run's upper bound
        AutointerpFailed: requests failed (charged for the tokens they used)
    """
    done = stored(db, atlas.directory, feature, explainer, scorer)
    if done is not None:
        return done, 0.0, True
    client = claude_client()
    sets = await feature_sets(atlas, feature)
    if sets is None:
        raise AutointerpInputError(f"Feature {feature} was never active over the corpus")

    row = claim(db, atlas, feature, explainer, scorer, user_id)
    if row is None:
        done = await wait_for(db, atlas.directory, feature, explainer, scorer)
        if done is not None:
            return done, 0.0, True
        raise AutointerpFailed("This feature is being explained by another request; try again shortly", 0.0)
    # Another request may have finished it between the check above and the claim
    done = stored(db, atlas.directory, feature, explainer, scorer)
    if done is not None:
        _release(db, row)
        return done, 0.0, True

    deadline = time.monotonic() + EXPLAIN_DEADLINE_SECONDS
    hold = 0.0
    try:
        planned = planned_prompts(sets, explainer, scorer, SEED)
        counts = await asyncio.wait_for(client.count_all([p.params for p in planned]), timeout=max(1.0, deadline - time.monotonic()))
        bounds = request_bounds(planned, counts)
        hold = max(autointerp_bound(bounds), MIN_AUTOINTERP_CHARGE)
        charge_flat(db, user_id, hold, TransactionType.AUTOINTERP_CHARGE, f"Autointerp: {description}")
    except InsufficientBalanceError:
        _release(db, row)
        raise
    except Exception:
        logger.exception("Couldn't price autointerp of %s feature %d", atlas.directory, feature)
        _release(db, row)
        raise AutointerpFailed("Couldn't reach Claude; you weren't charged", 0.0)

    try:
        result = await run_feature(client, sets, explainer, scorer, SEED, deadline)
    except BaseException:
        # A bug or a cancellation mid-run: what it used is unknown, so the hold stands
        logger.exception("Autointerp of %s feature %d stopped", atlas.directory, feature)
        _release(db, row)
        raise

    used = calculate_autointerp_cost(result.usage) + autointerp_bound([bounds[i] for i in result.unreported])
    charged = min(hold, used if result.status == "failed" else max(used, MIN_AUTOINTERP_CHARGE))
    if result.status == "failed":
        db.rollback()
        db.query(FeatureExplanation).filter(FeatureExplanation.id == row.id, FeatureExplanation.status == "running").delete()
        charged = _refund(db, user_id, hold, charged, f"Autointerp refund (failed): {description}")
        db.commit()
        logger.warning("Autointerp of %s feature %d failed: %s", atlas.directory, feature, "; ".join(result.problems))
        raise AutointerpFailed(f"Claude's requests failed; you were charged ${charged:.3f} for the tokens they used", charged)

    charged = _refund(db, user_id, hold, charged, f"Autointerp refund (unused hold): {description}")
    _save(db, row, result, charged)
    db.commit()
    db.refresh(row)
    return row, charged, False


def model_name(model_id: str) -> str:
    return MODELS[model_id].name if model_id in MODELS else model_id


# A feature's explanations over time. Re-explaining a feature (with other models, or after a
# prompt version changes) keeps the earlier ones; the newest is the one shown, never the
# best-scoring one: choosing by score on the held-out set the score is reported on would game it.

def versions(db: Session, atlas_key: str, feature: int) -> list[FeatureExplanation]:
    """Every finished explanation of a feature on an atlas, newest first (the first is the one shown)."""
    return (
        _final(db, atlas_key, feature)
        .order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc())
        .limit(VERSIONS_SHOWN)
        .all()
    )


def explanation_by_id(db: Session, atlas_key: str, feature: int, explanation_id: int) -> Optional[FeatureExplanation]:
    return _final(db, atlas_key, feature).filter(FeatureExplanation.id == explanation_id).first()


def _same_build(row, atlas: Atlas) -> bool:
    """Whether a row was made from the atlas as it's stored now (its example sets and communities)."""
    return (row.atlas or {}).get("created_at") == atlas.meta.get("created_at")


# Hypotheses: a user's own explanations, scored like autointerp's (private)

def hypotheses(db: Session, user_id: int, atlas_key: str, feature: int) -> list[FeatureHypothesis]:
    """A user's hypotheses about a feature on an atlas, newest first."""
    return (
        db.query(FeatureHypothesis)
        .filter(FeatureHypothesis.user_id == user_id, FeatureHypothesis.atlas_key == atlas_key, FeatureHypothesis.feature == feature)
        .order_by(FeatureHypothesis.created_at.desc(), FeatureHypothesis.id.desc())
        .limit(HYPOTHESES_SHOWN)
        .all()
    )


def hypothesis_by_id(db: Session, user_id: int, hypothesis_id: int) -> Optional[FeatureHypothesis]:
    return db.query(FeatureHypothesis).filter(FeatureHypothesis.id == hypothesis_id, FeatureHypothesis.user_id == user_id).first()


def delete_hypothesis(db: Session, row: FeatureHypothesis) -> None:
    """Delete a hypothesis and its generation tests."""
    db.query(GenerationTest).filter(GenerationTest.hypothesis_id == row.id).delete(synchronize_session=False)
    db.delete(row)
    db.commit()


def token_baseline(db: Session, user_id: int, atlas: Atlas, feature: int, scorer: str) -> Optional[float]:
    """
    The token baseline's detection on a feature's held-out set with a scorer, if it's already
    measured on this atlas build with the current versions: by an explanation's scoring, or by
    one of the user's hypotheses. (It's the same requests whatever explanation it ran next to.)
    """
    explained = (
        db.query(FeatureExplanation)
        .filter(
            FeatureExplanation.atlas_key == atlas.directory,
            FeatureExplanation.feature == feature,
            FeatureExplanation.scorer_model == scorer,
            FeatureExplanation.prompt_version == PROMPT_VERSION,
            FeatureExplanation.examples_version == ai.EXAMPLES_VERSION,
            FeatureExplanation.status == "scored",
            FeatureExplanation.token_detection.isnot(None),
        )
        .order_by(FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc())
        .all()
    )
    own = (
        db.query(FeatureHypothesis)
        .filter(
            FeatureHypothesis.user_id == user_id,
            FeatureHypothesis.atlas_key == atlas.directory,
            FeatureHypothesis.feature == feature,
            FeatureHypothesis.scorer_model == scorer,
            FeatureHypothesis.prompt_version == PROMPT_VERSION,
            FeatureHypothesis.examples_version == ai.EXAMPLES_VERSION,
            FeatureHypothesis.token_detection.isnot(None),
        )
        .order_by(FeatureHypothesis.created_at.desc(), FeatureHypothesis.id.desc())
        .all()
    )
    return next((row.token_detection for row in [*explained, *own] if _same_build(row, atlas)), None)


async def score_hypothesis(
    db: Session,
    user_id: int,
    atlas: Atlas,
    feature: int,
    text: str,
    scorer: str,
    description: str,
) -> tuple[FeatureHypothesis, float, bool]:
    """
    Score a user's explanation of a feature with the scorer, on the feature's held-out set and
    with the prompts an autointerp explanation gets (so the scores compare), and store it for
    the user. The token baseline's detection is measured too unless it already is.

    Charged its usage, at least MIN_HYPOTHESIS_CHARGE; the upper bound (every request's exact
    input and max_tokens of output) is held while it runs.

    Returns (the row, what this request was charged, whether it was already stored).

    Raises:
        AutointerpInputError: the text is empty, or the feature can't be scored (shown to the user)
        AutointerpUnavailable: no API key
        InsufficientBalanceError: the balance doesn't cover the upper bound
        AutointerpFailed: requests failed (charged for the tokens they used)
    """
    text = clean_text(text)
    if not text:
        raise AutointerpInputError("Write an explanation to score")
    same = (
        db.query(FeatureHypothesis)
        .filter(
            FeatureHypothesis.user_id == user_id,
            FeatureHypothesis.atlas_key == atlas.directory,
            FeatureHypothesis.feature == feature,
            FeatureHypothesis.text == text,
            FeatureHypothesis.scorer_model == scorer,
            FeatureHypothesis.prompt_version == PROMPT_VERSION,
            FeatureHypothesis.examples_version == ai.EXAMPLES_VERSION,
        )
        .order_by(FeatureHypothesis.created_at.desc(), FeatureHypothesis.id.desc())
        .all()
    )
    done = next((row for row in same if _same_build(row, atlas)), None)
    if done is not None:
        return done, 0.0, True

    client = claude_client()
    sets = await feature_sets(atlas, feature)
    if sets is None:
        raise AutointerpInputError(f"Feature {feature} was never active over the corpus")
    if not sets.scorable:
        raise AutointerpInputError(f"This feature can't be scored: {(sets.note or 'no held-out set').lower()}")
    known_baseline = token_baseline(db, user_id, atlas, feature, scorer)
    baseline = known_baseline is None

    deadline = time.monotonic() + EXPLAIN_DEADLINE_SECONDS
    try:
        prompts = hypothesis_prompts(sets, text, scorer, SEED, baseline)
        counts = await asyncio.wait_for(client.count_all([p.params for p in prompts]), timeout=max(1.0, deadline - time.monotonic()))
        bounds = exact_bounds(prompts, counts)
        hold = max(autointerp_bound(bounds), MIN_HYPOTHESIS_CHARGE)
        charge_flat(db, user_id, hold, TransactionType.AUTOINTERP_CHARGE, f"Autointerp hypothesis: {description}")
    except InsufficientBalanceError:
        raise
    except Exception:
        logger.exception("Couldn't price a hypothesis on %s feature %d", atlas.directory, feature)
        raise AutointerpFailed("Couldn't reach Claude; you weren't charged", 0.0)

    try:
        result = await run_hypothesis(client, sets, text, scorer, SEED, deadline, baseline)
    except BaseException:
        # A bug or a cancellation mid-run: what it used is unknown, so the hold stands
        logger.exception("Scoring a hypothesis on %s feature %d stopped", atlas.directory, feature)
        raise

    used = calculate_autointerp_cost(result.usage) + autointerp_bound([bounds[i] for i in result.unreported])
    charged = min(hold, used if result.status == "failed" else max(used, MIN_HYPOTHESIS_CHARGE))
    if result.status == "failed":
        charged = _refund(db, user_id, hold, charged, f"Autointerp refund (failed): hypothesis, {description}")
        db.commit()
        logger.warning("Scoring a hypothesis on %s feature %d failed: %s", atlas.directory, feature, "; ".join(result.problems))
        raise AutointerpFailed(f"Claude's requests failed; you were charged ${charged:.3f} for the tokens they used", charged)

    charged = _refund(db, user_id, hold, charged, f"Autointerp refund (unused hold): hypothesis, {description}")
    row = FeatureHypothesis(
        user_id=user_id, atlas_key=atlas.directory, sae_key=atlas.sae_key, model_id=atlas.model_id, corpus=atlas.corpus,
        atlas=atlas.build, feature=feature, text=text, scorer_model=scorer, prompt_version=PROMPT_VERSION,
        examples_version=ai.EXAMPLES_VERSION, status=result.status, note=(result.note or None) and result.note[:500],
        detection=result.detection, detection_hard=result.detection_hard, fuzzing=result.fuzzing, score=result.score,
        token_detection=result.token_detection if baseline else known_baseline, decile_recall=result.decile_recall,
        details={**result.details, "problems": result.problems[:20]}, verdicts=result.verdicts, usage=result.total_usage,
        cost_usd=round(charged, 4), created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, charged, False


# Generation tests

async def feature_maximum(atlas: Atlas, feature: int) -> Optional[float]:
    """A feature's maximum activation over the atlas's corpus (None if it was never active)."""
    try:
        raw = await atlas_store.read_bytes(f"{atlas.directory}/catalog.bin")
    except FileNotFoundError:
        return None
    catalog = np.frombuffer(raw, dtype="<f4").reshape(-1, CATALOG_COLUMNS)
    rows = catalog[catalog[:, 0] == feature]
    return float(rows[0, 3]) if rows.shape[0] and rows[0, 3] > 0 else None


def generation_tests(db: Session, user_id: int, atlas_key: str, feature: int) -> list[GenerationTest]:
    """A feature's generation tests a user can see, newest first: those of its explanations (anyone's), and of the user's own hypotheses."""
    return (
        db.query(GenerationTest)
        .filter(
            GenerationTest.atlas_key == atlas_key,
            GenerationTest.feature == feature,
            (GenerationTest.explanation_id.isnot(None)) | (GenerationTest.user_id == user_id),
        )
        .order_by(GenerationTest.created_at.desc(), GenerationTest.id.desc())
        .limit(TESTS_SHOWN)
        .all()
    )


def _stored_test(db: Session, atlas: Atlas, source: Union[FeatureExplanation, FeatureHypothesis], generator: str) -> Optional[GenerationTest]:
    column = GenerationTest.explanation_id if isinstance(source, FeatureExplanation) else GenerationTest.hypothesis_id
    rows = (
        db.query(GenerationTest)
        .filter(column == source.id, GenerationTest.generator_model == generator, GenerationTest.prompt_version == GENERATION_PROMPT_VERSION)
        .order_by(GenerationTest.created_at.desc(), GenerationTest.id.desc())
        .all()
    )
    return next((row for row in rows if _same_build(row, atlas)), None)


def _generation_texts(kinds: list[str], read: list[dict], peaks: list[float], maximum: float, threshold: float) -> list[dict]:
    """Each text as it's stored: its tokens, the feature's level on each (percent of its corpus maximum), and its peak."""
    out = []
    for kind, entry, peak in zip(kinds, read, peaks):
        out.append({
            "kind": kind,
            "tokens": entry["tokens"],
            "levels": [max(1, round(100 * act / maximum)) if act > 0 else 0 for act in entry["acts"]],
            "read": entry["read"],
            "peak": round(peak, 4),
            "fired": peak >= threshold * maximum,
        })
    return out


async def generation_test(
    db: Session,
    user_id: int,
    atlas: Atlas,
    feature: int,
    source: Union[FeatureExplanation, FeatureHypothesis],
    generator: str,
    read: Callable[[list[str]], Awaitable[dict]],
    read_bound: float,
    read_cost: Callable[[int, int, int], float],
    description: str,
) -> tuple[GenerationTest, float, bool]:
    """
    Test whether an explanation is specific enough to write text its feature fires on: Claude
    writes texts it says the feature fires on and near-misses it says it doesn't, the model and
    SAE read them in one GPU call, and the result is the share of the firing texts that reach
    GENERATION_THRESHOLD of the feature's maximum over the corpus, and the share of the
    near-misses that stay below it. Stored: shared for an autointerp explanation, private for a
    user's hypothesis.

    Args:
        source: The explanation or hypothesis tested
        read: Reads texts through the model and SAE on a GPU ({"texts": [{"tokens", "acts",
            "read"}], "total_tokens", "width"})
        read_bound: The most the GPU call is charged
        read_cost: The GPU call's charge, from (texts, tokens read, SAE width)

    Charged Claude's usage and the GPU call; their upper bound is held while it runs. A test
    whose texts couldn't be written or read is charged the tokens Claude used.

    Returns (the row, what this request was charged, whether it was already stored).

    Raises:
        AutointerpInputError, AutointerpUnavailable, InsufficientBalanceError, AutointerpFailed
        (as explain does)
    """
    text = source.explanation if isinstance(source, FeatureExplanation) else source.text
    if not text:
        raise AutointerpInputError("There's no explanation to test")
    done = _stored_test(db, atlas, source, generator)
    if done is not None:
        return done, 0.0, True
    client = claude_client()
    maximum = await feature_maximum(atlas, feature)
    if maximum is None:
        raise AutointerpInputError(f"Feature {feature} was never active over the corpus")

    deadline = time.monotonic() + GENERATION_DEADLINE_SECONDS
    prompt = generation_prompt(text, generator)
    try:
        counts = await asyncio.wait_for(client.count_all([prompt.params]), timeout=max(1.0, deadline - time.monotonic()))
        bounds = exact_bounds([prompt], counts)
        hold = autointerp_bound(bounds) + read_bound
        charge_flat(db, user_id, hold, TransactionType.AUTOINTERP_CHARGE, f"Autointerp generation test: {description}")
    except InsufficientBalanceError:
        raise
    except Exception:
        logger.exception("Couldn't price a generation test on %s feature %d", atlas.directory, feature)
        raise AutointerpFailed("Couldn't reach Claude; you weren't charged", 0.0)

    def settle(charged: float, why: str) -> float:
        return _refund(db, user_id, hold, min(hold, charged), f"Autointerp refund ({why}): generation test, {description}")

    try:
        [reply] = await client.send_all([prompt.params], deadline)
    except BaseException:
        # A bug or a cancellation mid-run: what it used is unknown, so the hold stands
        logger.exception("A generation test on %s feature %d stopped", atlas.directory, feature)
        raise
    claude_used = calculate_autointerp_cost({reply.model: reply.usage}) + (autointerp_bound(bounds) if reply.maybe_billed else 0.0)
    texts = None if reply.error else parse_generation(reply)
    if texts is None:
        charged = settle(claude_used, "failed")
        db.commit()
        logger.warning("A generation test on %s feature %d got no texts: %s", atlas.directory, feature, reply.error or reply.stop_reason)
        problem = "Claude declined to write texts for this explanation" if reply.refused else "Claude didn't write the texts"
        raise AutointerpFailed(f"{problem}; you were charged ${charged:.3f} for the tokens it used", charged)

    firing, near_misses = texts
    kinds = ["firing"] * len(firing) + ["near_miss"] * len(near_misses)
    try:
        result = await read(firing + near_misses)
        if len(result["texts"]) != len(kinds):
            raise RuntimeError(f"Read {len(result['texts'])} of {len(kinds)} texts")
    except Exception as e:
        # A GPU call that fails isn't charged, as elsewhere; the tokens Claude wrote are
        charged = settle(claude_used, "failed")
        db.commit()
        logger.exception("A generation test's texts couldn't be read on %s feature %d", atlas.directory, feature)
        shown = str(e) if e.__class__.__name__ in ("ProbeInputError", "GeometryInputError") else "The texts couldn't be read through the model"
        raise AutointerpFailed(f"{shown}; you were charged ${charged:.3f} for the tokens Claude used", charged)

    # Each text's strongest activation; the rates and each text's "fired" come from the same comparison
    peaks = [max(entry["acts"], default=0.0) for entry in result["texts"]]
    scores = generation_result(
        [p for kind, p in zip(kinds, peaks) if kind == "firing"], [p for kind, p in zip(kinds, peaks) if kind == "near_miss"],
        maximum, GENERATION_THRESHOLD,
    )
    stored = _generation_texts(kinds, result["texts"], peaks, maximum, GENERATION_THRESHOLD)
    charged = settle(claude_used + read_cost(len(kinds), int(result["total_tokens"]), int(result["width"])), "unused hold")
    row = GenerationTest(
        user_id=user_id, atlas_key=atlas.directory, sae_key=atlas.sae_key, model_id=atlas.model_id, corpus=atlas.corpus,
        atlas=atlas.build, feature=feature,
        explanation_id=source.id if isinstance(source, FeatureExplanation) else None,
        hypothesis_id=source.id if isinstance(source, FeatureHypothesis) else None,
        text=text, generator_model=generator, prompt_version=GENERATION_PROMPT_VERSION, threshold=scores["threshold"],
        maximum=maximum, firing_rate=scores["firing_rate"], near_miss_rate=scores["near_miss_rate"], texts=stored,
        usage={reply.model: reply.usage.to_dict()}, cost_usd=round(charged, 4), created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, charged, False


# Atlas communities and their names

_communities: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()


async def atlas_communities(atlas: Atlas) -> list[dict]:
    """An atlas's communities, as its summary lists them (parsed once per build, a few at a time)."""
    key = (atlas.directory, str(atlas.meta.get("created_at")))
    if key in _communities:
        _communities.move_to_end(key)
        return _communities[key]
    try:
        communities = (await atlas_store.read_json(f"{atlas.directory}/summary.json")).get("communities", [])
    except FileNotFoundError:
        communities = []
    _communities[key] = communities
    while len(_communities) > FILES_CACHED:
        _communities.popitem(last=False)
    return communities


def _explained(db: Session, atlas_key: str, *columns):
    return db.query(*columns).filter(
        FeatureExplanation.atlas_key == atlas_key,
        FeatureExplanation.is_control.is_(False),
        FeatureExplanation.status.in_(("scored", "unscored")),
        FeatureExplanation.explanation.isnot(None),
    )


def explained_features(db: Session, atlas_key: str) -> set[int]:
    """Features of an atlas with an explanation."""
    return {feature for (feature,) in _explained(db, atlas_key, FeatureExplanation.feature).distinct()}


def explanation_texts(db: Session, atlas_key: str) -> dict[int, str]:
    """Each explained feature's explanation on an atlas (its newest)."""
    rows = _explained(db, atlas_key, FeatureExplanation.feature, FeatureExplanation.explanation).order_by(
        FeatureExplanation.created_at.desc(), FeatureExplanation.id.desc(),
    )
    out: dict[int, str] = {}
    for feature, explanation in rows:
        out.setdefault(feature, explanation)
    return out


def community_names(db: Session, atlas: Atlas) -> dict[int, CommunityName]:
    """Each named community's newest name, for the atlas as it's stored now (names of an earlier build's communities aren't its)."""
    rows = (
        db.query(CommunityName)
        .filter(CommunityName.atlas_key == atlas.directory, CommunityName.atlas_created_at == str(atlas.meta.get("created_at")))
        .order_by(CommunityName.created_at.desc(), CommunityName.id.desc())
        .all()
    )
    out: dict[int, CommunityName] = {}
    for row in rows:
        out.setdefault(row.community, row)
    return out


def named_from(db: Session, atlas: Atlas, namer: str) -> dict[int, int]:
    """
    {community: explained features it was named from} for the names a model wrote for this atlas
    build with the current naming prompt: a naming job skips these until more of a community's
    features are explained.
    """
    rows = db.query(CommunityName.community, func.max(CommunityName.explained)).filter(
        CommunityName.atlas_key == atlas.directory,
        CommunityName.atlas_created_at == str(atlas.meta.get("created_at")),
        CommunityName.namer_model == namer,
        CommunityName.prompt_version == COMMUNITY_PROMPT_VERSION,
    ).group_by(CommunityName.community)
    return {community: explained for community, explained in rows}
