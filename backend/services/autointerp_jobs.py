"""
Autointerp jobs on the backend: choosing a job's features, starting each of its stages on Modal
(core.autointerp.jobs: prepare → submit → collect explain → collect score), waiting on its
batches in between, importing its results into Postgres, and charging for it.

Charging: a job never costs more than it was charged, and is never charged less than it cost.
When submitted it holds what its first stage can cost (its CPU timeout). Once its requests are
counted, it holds its upper bound: every request's exact input (scoring requests with room for
any explanation) and max_tokens of output at batch prices, and every remaining stage running to
its timeout, all with markup. When it ends it's charged the usage Anthropic reported and the CPU
seconds its stages ran, and refunded the rest. A failed job keeps only what it used: the usage
of the batches it read, the bound of any it sent but couldn't read (they're canceled), and its
stages' CPU time.

A job can instead name the atlas's communities from the explanations already stored (selection
kind "communities"): its naming requests ride in the explain batches, and its names are imported
with its results.

The status endpoints advance a job when it's polled, and the reconciler advances any nobody polls.
Each step runs once however many polls race it: a stage is claimed before it's started, and a
stage's end is recorded under the job's row lock (read afresh: populate_existing), committed
with no await in between (sessions are synchronous on the event loop).
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from core.autointerp import examples as ai
from core.autointerp.budget import BATCH_CHECK_SECONDS, MAX_JOB_FEATURES, SCORE_SAMPLE, AutointerpInputError
from core.autointerp.client import Usage
from core.autointerp.communities import NamePlan, plan_names
from core.autointerp.jobs import job_directory, stratified
from core.autointerp.prompts import COMMUNITY_PROMPT_VERSION, PROMPT_VERSION
from core.geometry.atlas import CATALOG_COLUMNS
from core.probes.sae import sae_directory
from db import SessionLocal
from models.database import AutointerpJob, CommunityName, FeatureExplanation, TransactionType
from services import atlas_store, autointerp
from services.credit_service import (
    AUTOINTERP_STAGES,
    InsufficientBalanceError,
    autointerp_cpu_cost,
    autointerp_stage_bound,
    autointerp_tokens_cost,
    calculate_autointerp_cost,
    charge_flat,
    charge_uncommitted,
    credit_refund,
)
from services.modal_feature_stats import AUTOINTERP_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
# A job's stages, and the Modal function each step runs
STAGES = ("preparing", "explaining", "scoring", "importing")
PREPARE, SUBMIT, COLLECT_EXPLAIN, COLLECT_SCORE = "prepare", "submit", "collect explain", "collect score"
FUNCTIONS = {PREPARE: "autointerp_prepare", SUBMIT: "autointerp_submit", COLLECT_EXPLAIN: "autointerp_collect", COLLECT_SCORE: "autointerp_collect"}
# A stage being started that has no Modal call this long after is started again
SPAWN_STALE = timedelta(minutes=15)
SPAWN_ATTEMPTS = 3
SPAWNING = "spawning"
# Seconds a poll spends importing results before answering (the reconciler finishes the rest)
IMPORT_SECONDS = 5.0
SELECTION_SEED = 0


def new_job_id() -> str:
    return f"autointerp_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def active_job(db: Session, atlas_key: str) -> Optional[AutointerpJob]:
    """The job explaining an atlas's features now, if any (anyone's)."""
    return (
        db.query(AutointerpJob)
        .filter(AutointerpJob.atlas_key == atlas_key, AutointerpJob.status.in_(ACTIVE))
        .order_by(AutointerpJob.created_at.desc())
        .first()
    )


# Choosing a job's features

async def live_features(atlas: autointerp.Atlas) -> tuple[np.ndarray, np.ndarray]:
    """(IDs, tokens active on) of the atlas's live features, most active first."""
    raw = await atlas_store.read_bytes(f"{atlas.directory}/catalog.bin")
    catalog = np.frombuffer(raw, dtype="<f4").reshape(-1, CATALOG_COLUMNS)
    order = np.argsort(-catalog[:, 1], kind="stable")
    return catalog[order, 0].astype(np.int64), catalog[order, 1].astype(np.int64)


def explained_with(db: Session, atlas_key: str, explainer: str, scorer: str, scoring: bool) -> set[int]:
    """
    Features (and controls) already explained on an atlas with these models and versions, which a
    job skips. A job that scores doesn't skip those explained by a job that left them out of its
    scoring sample.
    """
    rows = (
        db.query(FeatureExplanation.feature, FeatureExplanation.status, FeatureExplanation.details)
        .filter(
            FeatureExplanation.atlas_key == atlas_key,
            FeatureExplanation.explainer_model == explainer,
            FeatureExplanation.scorer_model == scorer,
            FeatureExplanation.prompt_version == PROMPT_VERSION,
            FeatureExplanation.examples_version == ai.EXAMPLES_VERSION,
            FeatureExplanation.status.in_(autointerp.FINAL),
        )
        .all()
    )
    return {r.feature for r in rows if not (scoring and r.status == "unscored" and (r.details or {}).get("sampled_out"))}


def choose(
    ids: np.ndarray, counts: np.ndarray, width: int, selection: dict, done: set[int],
) -> tuple[list[int], list[int], int]:
    """
    A selection's features (live ones, less those done), those of them to score, and how many
    were skipped as done. selection: {"kind": "top" (n), "range" (start, end, inclusive),
    "list" (features) or "all", "scoring": "all" or "sample"}.
    """
    kind = selection["kind"]
    if kind == "top":
        picked = ids[: int(selection["n"])]
    elif kind == "range":
        picked = ids[(ids >= int(selection["start"])) & (ids <= int(selection["end"]))]
    elif kind == "list":
        wanted = set(int(f) for f in selection["features"])
        if any(f < 0 or f >= width for f in wanted):
            raise AutointerpInputError(f"This SAE has {width} features")
        picked = ids[np.isin(ids, list(wanted))]
    else:
        picked = ids
    count = dict(zip(ids.tolist(), counts.tolist()))
    features = [int(f) for f in picked if int(f) not in done]
    skipped = len(picked) - len(features)
    if len(features) > MAX_JOB_FEATURES:
        raise AutointerpInputError(f"{len(features):,} features to explain; a job explains at most {MAX_JOB_FEATURES:,}")
    if selection.get("scoring", "all") == "all" or len(features) <= SCORE_SAMPLE:
        scored = sorted(features)
    else:
        scored = stratified([(f, count[f]) for f in features], SCORE_SAMPLE, SELECTION_SEED)
    return sorted(features), scored, skipped


def controls_to_run(width: int, done: set[int]) -> list[int]:
    """The atlas's random-direction controls not yet explained with these models (dead ones are dropped later)."""
    return [c for c in range(width, width + ai.CONTROLS) if c not in done]


async def naming_plan(db: Session, atlas: autointerp.Atlas, namer: str) -> tuple[list[NamePlan], dict, int]:
    """
    What a job naming an atlas's communities with a model would do: the communities it would
    name (those with enough explained features that the model hasn't named from as many), the
    "naming" part of its spec (the explanations of those communities' features, and what's
    named already), and how many nameable communities it skips as named.
    """
    communities = await autointerp.atlas_communities(atlas)
    explanations = autointerp.explanation_texts(db, atlas.directory)
    done = autointerp.named_from(db, atlas, namer)
    plans = plan_names(communities, explanations, done)
    planned = {plan.community for plan in plans}
    shown = {f: explanations[f] for c in communities if c["id"] in planned for f in c["features"] if f in explanations}
    skipped = len(plan_names(communities, explanations)) - len(plans)
    return plans, {"explanations": shown, "done": done}, skipped


# Creating and starting a job

def create_job(
    db: Session,
    user_id: int,
    atlas: autointerp.Atlas,
    sae: dict,
    sae_name: str,
    explainer: str,
    scorer: str,
    selection: dict,
    features: int,
    scored: int,
    controls: int,
    estimate: float,
) -> AutointerpJob:
    """
    Record a job and hold what its first stage can cost, in one transaction.

    Raises:
        InsufficientBalanceError: the balance doesn't cover it (nothing is recorded)
    """
    hold = autointerp_stage_bound(1)
    job = AutointerpJob(
        job_id=new_job_id(),
        user_id=user_id,
        model_id=atlas.model_id,
        sae=sae,
        sae_name=sae_name,
        atlas_key=atlas.directory,
        corpus=atlas.corpus,
        atlas=atlas.build,
        explainer_model=explainer,
        scorer_model=scorer,
        selection=selection,
        features=features,
        scored=scored,
        controls=controls,
        status="queued",
        stage="preparing",
        cost_usd=round(hold, 4),
        estimated_usd=round(estimate, 4),
        refunded_usd=0,
        cpu_seconds=0.0,
        usage={},
        batch_ids={},
        details={"stages": 0},
        created_at=datetime.utcnow(),
    )
    db.add(job)
    charge_flat(db, user_id, hold, TransactionType.AUTOINTERP_CHARGE, f"Autointerp job: {_describe(job)} (hold while counting)")
    db.refresh(job)
    return job


def _describe(job: AutointerpJob) -> str:
    if (job.selection or {}).get("kind") == "communities":
        return f"{job.sae_name} ({job.model_id}, {job.corpus}), community names"
    return f"{job.sae_name} ({job.model_id}, {job.corpus}), {job.features:,} features"


def _job_dir(job: AutointerpJob) -> str:
    return job_directory(sae_directory(job.sae), job.job_id)


async def start(db: Session, job: AutointerpJob, spec: dict) -> None:
    """Start a recorded job's first stage. If it can't be started, the job fails (refunded in full) and the error is raised."""
    if not await _spawn(db, job, PREPARE, spec):
        job = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).first()
        _fail(db, job, "The job couldn't be started")
        raise RuntimeError("Couldn't start the autointerp job")


async def _spawn(db: Session, job: AutointerpJob, step: str, *args) -> bool:
    """
    Start a step's Modal call, once: the job is claimed for it first (a poll and the reconciler
    can't both start it). Returns whether it started; one that didn't is tried again on a later poll.
    """
    claimed = (
        db.query(AutointerpJob)
        .filter(AutointerpJob.job_id == job.job_id, AutointerpJob.modal_call_id.is_(None), AutointerpJob.status.in_(ACTIVE))
        .update({AutointerpJob.modal_call_id: f"{SPAWNING}:{datetime.utcnow().isoformat()}"}, synchronize_session=False)
    )
    db.commit()
    if not claimed:
        return False
    from services.modal_feature_stats import get_function

    job_dir = _job_dir(job)
    args = {
        PREPARE: (job.job_id, job_dir, job.atlas_key, *args),
        SUBMIT: (job.job_id, job_dir),
        COLLECT_EXPLAIN: (job.job_id, job_dir, "explain"),
        COLLECT_SCORE: (job.job_id, job_dir, "score"),
    }[step]
    try:
        call = await get_function(FUNCTIONS[step]).spawn.aio(*args)
    except Exception:
        logger.exception("Couldn't start %s of autointerp job %s", step, job.job_id)
        db.rollback()
        fresh = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).with_for_update().populate_existing().first()
        fresh.modal_call_id = None
        details = dict(fresh.details or {})
        details["spawn_failures"] = details.get("spawn_failures", 0) + 1
        fresh.details = details
        db.commit()
        if details["spawn_failures"] >= SPAWN_ATTEMPTS and step != PREPARE:
            _fail(db, fresh, "The job couldn't be continued")
        return False
    fresh = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).with_for_update().populate_existing().first()
    details = dict(fresh.details or {})
    details.update(call=step, spawned_at=datetime.utcnow().isoformat(), spawn_failures=0)
    fresh.details = details
    fresh.modal_call_id = call.object_id
    if fresh.status == "queued":
        fresh.status, fresh.started_at = "running", datetime.utcnow()
    db.commit()
    return True


# How a stage's call ended

async def call_outcome(modal_call_id: str) -> Optional[dict]:
    """
    How a stage's call ended ({"status", "error", "seconds", "report"}; seconds None when
    unknown), or None while it's queued or running. Raises if Modal can't be asked: that must
    never be mistaken for the stage failing.
    """
    import modal
    from modal.call_graph import InputStatus

    call = modal.FunctionCall.from_id(modal_call_id)
    inputs = [i for i in await call.get_call_graph.aio() if i.function_call_id == modal_call_id]
    if not inputs:
        return {"status": "failed", "error": "Lost track of the job", "seconds": None, "report": None}
    status = inputs[0].status
    if status == InputStatus.PENDING:
        return None
    if status == InputStatus.TIMEOUT:
        return {"status": "failed", "error": "The job ran out of time", "seconds": float(AUTOINTERP_TIMEOUT_SECONDS), "report": None}
    if status != InputStatus.SUCCESS:
        return {"status": "failed", "error": "The job stopped unexpectedly", "seconds": None, "report": None}
    try:
        return await call.get.aio(timeout=0)
    except modal.exception.OutputExpiredError:
        return {"status": "failed", "error": "Lost track of the job", "seconds": None, "report": None}


def _seconds_since(iso: Optional[str], default: datetime) -> float:
    started = datetime.fromisoformat(iso) if iso else default
    return min((datetime.utcnow() - started).total_seconds(), float(AUTOINTERP_TIMEOUT_SECONDS))


def _finish_step(db: Session, job_id: str, call_id: str, outcome: dict) -> Optional[str]:
    """
    Record how a step's call ended, once (the job's row is locked, and the call must still be
    the job's). Returns the next step to start, if any.
    """
    job = db.query(AutointerpJob).filter(AutointerpJob.job_id == job_id).with_for_update().populate_existing().first()
    if job is None or job.status not in ACTIVE or job.modal_call_id != call_id:
        db.rollback()
        return None
    details = dict(job.details or {})
    step = details.get("call")
    seconds = outcome.get("seconds")
    if seconds is None:
        seconds = _seconds_since(details.get("spawned_at"), job.created_at)
    job.cpu_seconds = (job.cpu_seconds or 0.0) + float(seconds)
    details["stages"] = details.get("stages", 0) + 1
    details["call"] = None
    job.modal_call_id = None
    job.details = details
    if outcome.get("status") != "completed":
        _fail(db, job, outcome.get("error") or "The job failed", commit=False)
        db.commit()
        return None

    report = outcome["report"] or {}
    if step == PREPARE:
        return _prepared(db, job, report)
    usage = dict(job.usage or {})
    batches = dict(job.batch_ids or {})
    if step == SUBMIT:
        batches["explain"] = report["batches"]
        job.batch_ids = batches
        details["batches"] = None
    elif step == COLLECT_EXPLAIN:
        usage["explain"] = report["usage"]
        job.usage = usage
        details["explained"] = {key: report[key] for key in ("explained", "refused", "failed", "scoring")}
        if report.get("names"):
            details["names"] = {**report["names"], "imported": False}
        if report.get("batches"):
            batches["score"] = report["batches"]
            job.batch_ids = batches
            job.stage = "scoring"
            details["batches"] = None
        else:
            job.stage = "importing"
            details["results"] = {**report["results"], "imported": 0}
    elif step == COLLECT_SCORE:
        usage["score"] = report["usage"]
        job.usage = usage
        job.stage = "importing"
        details["results"] = {**report["results"], "imported": 0}
    job.details = dict(details)
    db.commit()
    return None


def _prepared(db: Session, job: AutointerpJob, report: dict) -> Optional[str]:
    """
    The job's requests are counted: hold its upper bound (tokens at batch prices, the stage that
    ran, and the rest running to their timeouts), and send the explain requests. A balance that
    doesn't cover it fails the job.
    """
    bound = report["bound"]
    tokens = autointerp_tokens_cost(bound["explain"]) + autointerp_tokens_cost(bound["score"])
    cpu = autointerp_cpu_cost(job.cpu_seconds, 1) + autointerp_stage_bound(AUTOINTERP_STAGES - 1)
    bound_usd = round(tokens + cpu, 4)
    details = dict(job.details)
    details["prepared"] = report
    job.details = details
    job.features, job.controls, job.scored = report["features"], report["controls"], report["scored_features"]
    extra = bound_usd - float(job.cost_usd)
    if extra > 0:
        try:
            charge_uncommitted(db, job.user_id, extra, TransactionType.AUTOINTERP_CHARGE, f"Autointerp job: {_describe(job)} (upper bound)")
        except InsufficientBalanceError:
            _fail(db, job, f"Holding the job's upper bound needs ${bound_usd:.2f}; add credits and submit it again", commit=False)
            db.commit()
            return None
        job.cost_usd = bound_usd
    job.bound_usd = bound_usd
    job.stage = "explaining"
    db.commit()
    return SUBMIT


# Settling

def _used(job: AutointerpJob) -> float:
    """What a job has cost so far: the usage of the batches it read (the bound of any it sent but couldn't read), and its stages' CPU time."""
    usage: dict[str, Usage] = {}
    for stage_usage in (job.usage or {}).values():
        for model_id, u in stage_usage.items():
            usage[model_id] = usage.get(model_id, Usage()) + Usage.from_dict(u)
    cost = calculate_autointerp_cost(usage, batch=True)
    bound = ((job.details or {}).get("prepared") or {}).get("bound", {})
    for stage in ("explain", "score"):
        if (job.batch_ids or {}).get(stage) and stage not in (job.usage or {}):
            cost += autointerp_tokens_cost(bound.get(stage, {}))
    return cost + autointerp_cpu_cost(job.cpu_seconds or 0.0, (job.details or {}).get("stages", 0))


def _settle(db: Session, job: AutointerpJob, description: str) -> None:
    """Charge a job what it used (never more than it holds) and refund the rest. Doesn't commit."""
    held = float(job.cost_usd) - float(job.refunded_usd or 0)
    used = min(_used(job), held)
    refund = held - used
    job.settled_usd = round(used, 4)
    job.refunded_usd = round(float(job.refunded_usd or 0) + refund, 4)
    job.finished_at = datetime.utcnow()
    credit_refund(db, job.user_id, refund, TransactionType.AUTOINTERP_REFUND, description)


def _fail(db: Session, job: AutointerpJob, error: str, commit: bool = True) -> None:
    """Fail a job: it keeps what it used (see _used), the rest is refunded; batches still running are canceled."""
    job.status = "failed"
    job.error_message = error[:1000]
    job.modal_call_id = None
    _settle(db, job, f"Autointerp refund ({job.sae_name}): {job.error_message[:200]}")
    if commit:
        db.commit()
    if any((job.batch_ids or {}).get(stage) for stage in ("explain", "score")):
        _cancel_later(dict(job.batch_ids))


_background: set = set()


def _cancel_later(batch_ids: dict) -> None:
    async def cancel() -> None:
        try:
            client = autointerp.claude_client()
            for ids in batch_ids.values():
                for batch_id in ids or []:
                    try:
                        await client.cancel_batch(batch_id)
                    except Exception:
                        pass
        except Exception:
            logger.exception("Couldn't cancel a failed autointerp job's batches")

    try:
        task = asyncio.get_running_loop().create_task(cancel())
    except RuntimeError:
        return
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _volume_batches(job: AutointerpJob) -> dict:
    """The batches a job's stages recorded on the volume (a stage that died may have sent some it never reported)."""
    try:
        data = json.loads(await atlas_store.read_uncached(f"{_job_dir(job)}/batches.json"))
    except (FileNotFoundError, ValueError):
        return {}
    return {stage: [entry[0] for _, entry in sorted(parts.items(), key=lambda p: int(p[0]))] for stage, parts in data.items()}


# Waiting on batches

async def _check_batches(db: Session, job: AutointerpJob, stage: str) -> Optional[str]:
    """Ask Anthropic how a stage's batches are doing (at most every BATCH_CHECK_SECONDS); returns the collect step once they've all ended."""
    details = dict(job.details or {})
    checked = (details.get("batches") or {}).get("checked_at")
    if checked and time.time() - checked < BATCH_CHECK_SECONDS:
        return None
    client = autointerp.claude_client()
    statuses = [await client.batch_status(batch_id) for batch_id in job.batch_ids[stage]]
    counts: dict[str, int] = {}
    for s in statuses:
        for key, n in s["counts"].items():
            counts[key] = counts.get(key, 0) + n
    ended = all(s["status"] == "ended" for s in statuses)
    db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).update(
        {AutointerpJob.details: {**details, "batches": {"stage": stage, "counts": counts, "ended": ended, "checked_at": time.time()}}},
        synchronize_session=False,
    )
    db.commit()
    return (COLLECT_EXPLAIN if stage == "explain" else COLLECT_SCORE) if ended else None


# Importing results

def _row(job: AutointerpJob, line: dict) -> FeatureExplanation:
    usage = {model_id: Usage.from_dict(u) for model_id, u in (line["usage"] or {}).items()}
    return FeatureExplanation(
        atlas_key=job.atlas_key,
        sae_key=sae_directory(job.sae),
        model_id=job.model_id,
        corpus=job.corpus,
        atlas=job.atlas,
        feature=int(line["feature"]),
        is_control=bool(line["is_control"]),
        explanation=line["explanation"],
        explainer_model=job.explainer_model,
        scorer_model=job.scorer_model,
        prompt_version=PROMPT_VERSION,
        examples_version=ai.EXAMPLES_VERSION,
        status=line["status"],
        note=(line["note"] or None) and line["note"][:500],
        detection=line["detection"],
        detection_hard=line["detection_hard"],
        fuzzing=line["fuzzing"],
        score=line["score"],
        token_detection=line["token_detection"],
        decile_recall=line["decile_recall"],
        details=line["details"],
        verdicts=line["verdicts"],
        usage={model_id: u.to_dict() for model_id, u in usage.items()},
        cost_usd=round(calculate_autointerp_cost(usage, batch=True), 4),
        job_id=job.job_id,
        user_id=job.user_id,
        created_at=datetime.utcnow(),
    )


def _name_row(job: AutointerpJob, line: dict) -> CommunityName:
    usage = {model_id: Usage.from_dict(u) for model_id, u in (line.get("usage") or {}).items()}
    return CommunityName(
        atlas_key=job.atlas_key,
        sae_key=sae_directory(job.sae),
        model_id=job.model_id,
        corpus=job.corpus,
        atlas_created_at=str((job.atlas or {}).get("created_at")),
        community=int(line["community"]),
        name=line["name"],
        namer_model=job.explainer_model,
        prompt_version=COMMUNITY_PROMPT_VERSION,
        explained=int(line["explained"]),
        size=int(line["size"]),
        usage={model_id: u.to_dict() for model_id, u in usage.items()},
        cost_usd=round(calculate_autointerp_cost(usage, batch=True), 4),
        job_id=job.job_id,
        user_id=job.user_id,
        created_at=datetime.utcnow(),
    )


async def _import_names(db: Session, job: AutointerpJob) -> None:
    """Store a job's community names, once (with the flag that they're stored, in one transaction)."""
    names = (job.details or {}).get("names")
    if job.status not in ACTIVE or not names or names.get("imported"):
        return
    data = await atlas_store.read_uncached(f"{_job_dir(job)}/names.json")
    locked = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).with_for_update().populate_existing().first()
    details = dict(locked.details or {})
    if locked.status in ACTIVE and details.get("names") and not details["names"].get("imported"):
        db.add_all([_name_row(locked, line) for line in json.loads(data) if line["status"] == "named"])
        details["names"] = {**details["names"], "imported": True}
        locked.details = details
        db.commit()
    else:
        db.rollback()
    db.refresh(job)


async def _import(db: Session, job: AutointerpJob, budget_seconds: Optional[float]) -> None:
    """
    Store a job's results, a part at a time (each with its count in one transaction, so a part is
    stored once), within budget_seconds (None: all), and its community names; then settle the job.
    """
    until = None if budget_seconds is None else time.monotonic() + budget_seconds
    while until is None or time.monotonic() < until:
        if job.status not in ACTIVE:
            return
        results = (job.details or {}).get("results") or {}
        part = results.get("imported", 0)
        if part >= results.get("parts", 0):
            break
        data = await atlas_store.read_uncached(f"{_job_dir(job)}/results/{part}.jsonl")
        locked = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).with_for_update().populate_existing().first()
        details = dict(locked.details or {})
        if locked.status not in ACTIVE or details.get("results", {}).get("imported") != part:
            db.rollback()
            db.refresh(job)
            continue
        lines = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        db.add_all([_row(locked, line) for line in lines if line["status"] in autointerp.FINAL])
        details["results"] = {**details["results"], "imported": part + 1}
        locked.details = details
        db.commit()
        db.refresh(job)

    await _import_names(db, job)
    job = db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).with_for_update().populate_existing().first()
    results = (job.details or {}).get("results") or {}
    names = (job.details or {}).get("names")
    if job.status in ACTIVE and results.get("imported", 0) >= results.get("parts", 0) and (not names or names.get("imported")):
        job.status, job.stage = "completed", None
        _settle(db, job, f"Autointerp refund (unused hold): {_describe(job)}")
        db.commit()
        logger.info("Autointerp job %s completed: %s", job.job_id, results.get("statuses"))
    else:
        db.rollback()


# Advancing a job

async def refresh(db: Session, job: AutointerpJob, import_seconds: Optional[float] = IMPORT_SECONDS) -> AutointerpJob:
    """Advance a job as far as it can go now (asking Modal and Anthropic); returns it as it now is."""
    for _ in range(6):
        if job.status not in ACTIVE:
            break
        before = (job.status, job.stage, job.modal_call_id, json.dumps(job.details, sort_keys=True, default=str))
        try:
            await _step(db, job, import_seconds)
        except Exception:
            logger.exception("Failed to advance autointerp job %s", job.job_id)
            db.rollback()
            break
        db.refresh(job)
        if (job.status, job.stage, job.modal_call_id, json.dumps(job.details, sort_keys=True, default=str)) == before:
            break
    db.refresh(job)
    return job


async def _step(db: Session, job: AutointerpJob, import_seconds: Optional[float]) -> None:
    details = job.details or {}
    call_id = job.modal_call_id
    if call_id is not None and call_id.startswith(SPAWNING):
        if datetime.utcnow() - datetime.fromisoformat(call_id.split(":", 1)[1]) > SPAWN_STALE:
            db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id, AutointerpJob.modal_call_id == call_id).update(
                {AutointerpJob.modal_call_id: None}, synchronize_session=False,
            )
            db.commit()
            if job.status == "queued":
                _fail(db, db.query(AutointerpJob).filter(AutointerpJob.job_id == job.job_id).first(), "The job was never started")
        return
    if call_id is not None:
        outcome = await call_outcome(call_id)
        if outcome is None:
            return
        if outcome.get("status") != "completed":
            # Batches a dying stage sent but never reported are canceled and counted at their bound
            recorded = await _volume_batches(job)
            if recorded:
                job.batch_ids = {**(job.batch_ids or {}), **recorded}
                db.commit()
        following = _finish_step(db, job.job_id, call_id, outcome)
        if following is not None:
            db.refresh(job)
            await _spawn(db, job, following)
        return
    if job.status == "queued":
        if datetime.utcnow() - job.created_at > SPAWN_STALE:
            _fail(db, job, "The job was never started")
        return
    if job.stage == "explaining" and details.get("call") is None:
        if not (job.batch_ids or {}).get("explain"):
            await _spawn(db, job, SUBMIT)
        else:
            following = await _check_batches(db, job, "explain")
            if following:
                await _spawn(db, job, following)
    elif job.stage == "scoring":
        following = await _check_batches(db, job, "score")
        if following:
            await _spawn(db, job, following)
    elif job.stage == "importing":
        await _import(db, job, import_seconds)


def progress(job: AutointerpJob, modal_progress: Optional[dict]) -> Optional[dict]:
    """A running job's progress, {"stage", "fraction"}: its stage running on Modal, its batches, or its import."""
    if job.status not in ACTIVE:
        return None
    details = job.details or {}
    if job.modal_call_id is not None:
        if modal_progress:
            return modal_progress
        labels = {PREPARE: "Counting tokens", SUBMIT: "Sending explain requests", COLLECT_EXPLAIN: "Reading explanations", COLLECT_SCORE: "Reading scores"}
        return {"stage": labels.get(details.get("call"), "Starting"), "fraction": 0.0}
    batches = details.get("batches") or {}
    if job.stage in ("explaining", "scoring") and batches.get("counts"):
        counts = batches["counts"]
        total = sum(counts.values())
        fraction = (total - counts.get("processing", 0)) / total if total else 0.0
        return {"stage": "Explaining" if job.stage == "explaining" else "Scoring", "fraction": round(fraction, 3)}
    if job.stage == "importing":
        results = details.get("results") or {}
        return {"stage": "Storing results", "fraction": results.get("imported", 0) / max(results.get("parts", 1), 1)}
    return {"stage": job.stage.capitalize() if job.stage else "Starting", "fraction": 0.0}


async def modal_progress(job: AutointerpJob) -> Optional[dict]:
    """The progress the stage running on Modal last reported (each stage reports under "<job ID>:<stage>")."""
    step = (job.details or {}).get("call")
    if step is None:
        return None
    try:
        import modal

        from services.modal_feature_stats import PROGRESS_DICT_NAME

        return await modal.Dict.from_name(PROGRESS_DICT_NAME).get.aio(f"{job.job_id}:{step}")
    except Exception:
        return None


async def reconcile_autointerp_jobs() -> None:
    """Advance every unfinished autointerp job (for jobs nobody polls), importing results in full."""
    db = SessionLocal()
    try:
        for job in db.query(AutointerpJob).filter(AutointerpJob.status.in_(ACTIVE)).all():
            before = (job.status, job.stage)
            job = await refresh(db, job, import_seconds=None)
            if job.status != before[0] or job.stage != before[1]:
                logger.warning("Reconciled autointerp job %s: %s/%s (%s)", job.job_id, job.status, job.stage, job.error_message)
    finally:
        db.close()
