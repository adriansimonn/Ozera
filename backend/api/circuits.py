"""
API endpoints for Circuit Tracing: the models that have transcoders, estimating and starting a
graph job, the user's graphs, and interventions on a graph's feature nodes.

A graph is built by a job on the circuits worker (services/modal_circuits.py), charged its
estimate when submitted, and stored on the SAE volume; reading it back is free. An intervention
is one charged call to the same worker. Everything else is answered without a GPU.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from api.schemas.circuits import (
    CircuitJobInfo,
    CircuitJobList,
    CircuitModelInfo,
    InterveneEstimate,
    InterveneRequest,
    InterveneResponse,
    TraceEstimate,
    TraceRequest,
)
from core.circuits import graph as graphs
from core.circuits.budget import (
    MAX_JOB_SECONDS,
    CircuitInputError,
    check_prompt,
    estimate_intervene_seconds,
    estimate_prompt_tokens,
    estimate_trace_seconds,
)
from core.circuits.registry import TRANSCODERS, TranscoderSet, transcoder_set
from core.circuits.store import graph_path, is_job_id
from core.open_source import OPEN_SOURCE_MODELS, get_gpu_tier
from db import get_db
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import CircuitJob, TransactionType, User
from services import atlas_store, circuit_jobs
from services.credit_service import InsufficientBalanceError, calculate_circuit_cost, charge_flat, check_sufficient_balance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuits", tags=["Circuit Tracing"])

# Graphs a user may have building at once, and how many the list of graphs shows
MAX_ACTIVE_JOBS = 3
LISTED_JOBS = 30


def _get_circuit_worker(model_id: str):
    from services.modal_circuits import get_circuit_worker

    return get_circuit_worker(get_gpu_tier(model_id))


# Models

def _set_info(transcoders: TranscoderSet) -> dict:
    return {
        "id": transcoders.id, "name": transcoders.name, "reference": transcoders.reference, "width": transcoders.width,
        "l0": transcoders.l0, "affine": transcoders.affine, "source": transcoders.source,
    }


@router.get("/models", response_model=list[CircuitModelInfo])
async def list_models():
    """The models that can be traced (those with public transcoders), each with its transcoder sets."""
    models = []
    for model_id, sets in TRANSCODERS.items():
        config = OPEN_SOURCE_MODELS[model_id]
        models.append({
            "model_id": model_id,
            "display_name": config.display_name,
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "parameters": config.parameters,
            "is_instruct": config.is_instruct,
            "gpu_tier": config.gpu_tier,
            "transcoders": [_set_info(s) for s in sets],
        })
    return models


def _transcoders(model_id: str, set_id: Optional[str]) -> TranscoderSet:
    """A request's transcoder set (400/404 if the model can't be traced or has no such set)."""
    if model_id not in TRANSCODERS:
        raise HTTPException(status_code=400, detail=f"{model_id} has no transcoders to trace it through")
    transcoders = transcoder_set(model_id, set_id)
    if transcoders is None:
        raise HTTPException(status_code=404, detail=f"{model_id} has no transcoder set {set_id}")
    return transcoders


# Graph jobs

def _trace_estimate(body: TraceRequest) -> tuple[TranscoderSet, int, float, float]:
    """(the transcoder set, estimated prompt tokens, estimated seconds, cost) of a graph job."""
    transcoders = _transcoders(body.model, body.transcoders)
    try:
        check_prompt(body.prompt)
    except CircuitInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    tokens = estimate_prompt_tokens(body.prompt, OPEN_SOURCE_MODELS[body.model].is_instruct)
    seconds = estimate_trace_seconds(body.model, transcoders, tokens)
    return transcoders, tokens, seconds, calculate_circuit_cost(body.model, seconds)


@router.post("/estimate", response_model=TraceEstimate)
async def estimate_trace(body: TraceRequest, current_user: User = Depends(get_current_user)):
    """Estimate a graph's cost before starting it."""
    return await estimate_trace_for(current_user, body)


async def estimate_trace_for(current_user: User, body: TraceRequest) -> dict:
    """POST /circuits/estimate as the user (also the assistant's quotes). Raises HTTPException as the route does."""
    _, tokens, seconds, cost = _trace_estimate(body)
    return {
        "estimated_cost": cost,
        "estimated_seconds": seconds,
        "estimated_tokens": tokens,
        "max_seconds": MAX_JOB_SECONDS,
        "within_limit": seconds <= MAX_JOB_SECONDS,
    }


async def _job_info(job: CircuitJob) -> dict:
    transcoders = transcoder_set(job.model_id, job.transcoders)
    return {
        "job_id": job.job_id,
        "model": job.model_id,
        "transcoders": job.transcoders,
        "transcoders_name": transcoders.name if transcoders else job.transcoders,
        "prompt": job.prompt,
        "target": job.target,
        "status": job.status,
        "progress": await circuit_jobs.progress(job.job_id) if job.status == "running" else None,
        "cost": float(job.cost_usd),
        "refunded": float(job.refunded_usd) if job.refunded_usd is not None else None,
        "error": job.error_message,
        "estimated_seconds": job.estimated_seconds,
        "summary": job.summary,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@router.post("/jobs", response_model=CircuitJobInfo)
@limiter.limit("10/minute")
async def submit_job(
    request: Request,
    body: TraceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trace a prompt on the circuits worker, in the background: the attribution graph of one
    target token at the prompt's last position (the next word given, or the model's top
    prediction), through the model's transcoders. Token nodes, the feature nodes active on the
    prompt, an error node per layer and position, and the target's logit node, with the direct
    effect of each on each; pruned to the nodes and edges that carry the target's attribution.

    Charged its estimate when submitted; a job that fails is refunded all but the GPU time it
    used. Poll GET /circuits/jobs/{id}, then read GET /circuits/jobs/{id}/graph.
    """
    return await submit_job_for(db, current_user, body)


async def submit_job_for(db: Session, current_user: User, body: TraceRequest) -> dict:
    """POST /circuits/jobs as the user (also the assistant's trace_circuit). Raises HTTPException as the route does."""
    transcoders, _, seconds, cost = _trace_estimate(body)
    if seconds > MAX_JOB_SECONDS:
        raise HTTPException(status_code=400, detail="This prompt is too long to trace")
    active = db.query(CircuitJob).filter(CircuitJob.user_id == current_user.id, CircuitJob.status.in_(circuit_jobs.ACTIVE)).all()
    active = [job for job in [await circuit_jobs.refresh(db, job) for job in active] if job.status in circuit_jobs.ACTIVE]
    if len(active) >= MAX_ACTIVE_JOBS:
        raise HTTPException(status_code=409, detail=f"{MAX_ACTIVE_JOBS} of your graphs are still being built; wait for one to finish")

    target = body.target if body.target else None
    try:
        job = circuit_jobs.create_job(db, current_user.id, body.model, transcoders.id, body.prompt, target, cost, seconds)
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")
    try:
        await circuit_jobs.spawn_job(db, job)
    except RuntimeError:
        logger.exception("Circuit job %s couldn't be started", job.job_id)
        raise HTTPException(status_code=502, detail="The job couldn't be started; you weren't charged")
    db.refresh(job)
    return await _job_info(job)


@router.get("/jobs", response_model=CircuitJobList)
@limiter.limit("60/minute")
async def list_jobs(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The user's graphs, newest first (jobs whose Modal call has ended are settled here)."""
    return await list_jobs_for(db, current_user)


async def list_jobs_for(db: Session, current_user: User, limit: int = LISTED_JOBS) -> dict:
    """GET /circuits/jobs as the user (also the assistant's reads)."""
    jobs = (
        db.query(CircuitJob)
        .filter(CircuitJob.user_id == current_user.id)
        .order_by(CircuitJob.created_at.desc())
        .limit(limit)
        .all()
    )
    jobs = [await circuit_jobs.refresh(db, job) for job in jobs]
    return {"jobs": [await _job_info(job) for job in jobs]}


def _own_job(db: Session, user: User, job_id: str) -> CircuitJob:
    """A job of the user's (404 for anyone else's, as for one that doesn't exist)."""
    job = db.query(CircuitJob).filter(CircuitJob.job_id == job_id, CircuitJob.user_id == user.id).first() if is_job_id(job_id) else None
    if job is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    return job


@router.get("/jobs/{job_id}", response_model=CircuitJobInfo)
@limiter.limit("120/minute")
async def get_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """A graph job's status and progress (settled here once its Modal call has ended)."""
    return await job_for(db, current_user, job_id)


async def job_for(db: Session, current_user: User, job_id: str) -> dict:
    """GET /circuits/jobs/{id} as the user."""
    job = await circuit_jobs.refresh(db, _own_job(db, current_user, job_id))
    return await _job_info(job)


async def _graph_bytes(db: Session, user: User, job_id: str) -> bytes:
    job = await circuit_jobs.refresh(db, _own_job(db, user, job_id))
    if job.status != "completed":
        raise HTTPException(status_code=409 if job.status in circuit_jobs.ACTIVE else 404, detail="This graph isn't built" if job.status == "failed" else "This graph is still being built")
    try:
        return await atlas_store.read_bytes(graph_path(job.job_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="This graph's file is missing")


async def graph_for(db: Session, current_user: User, job_id: str) -> dict:
    """A completed job's graph (core.circuits.trace.trace's result) as the user."""
    return json.loads(await _graph_bytes(db, current_user, job_id))


@router.get("/jobs/{job_id}/graph")
@limiter.limit("120/minute")
async def get_graph(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    A completed job's graph: the prompt's tokens, the target and the model's reading of it,
    the scores (how much of the target's attribution flows through features rather than error
    nodes), and the pruned nodes and edges (see core.circuits.graph).
    """
    return Response(content=await _graph_bytes(db, current_user, job_id), media_type="application/json")


# Interventions

def _intervention(graph: dict, body: InterveneRequest) -> tuple[TranscoderSet, list[tuple[int, int, int]], float, float]:
    """(the graph's transcoder set, the chosen nodes as (layer, position, feature), estimated seconds, cost)."""
    model_id = graph["model"]
    transcoders = _transcoders(model_id, graph["transcoders"]["id"])
    in_graph = {node["id"] for node in graph["nodes"] if node["kind"] == graphs.FEATURE}
    nodes = []
    for node_id in body.nodes:
        parsed = graphs.parse_feature_id(node_id)
        if parsed is None or node_id not in in_graph:
            raise HTTPException(status_code=400, detail=f"{node_id} isn't a feature node of this graph")
        nodes.append(parsed)
    if len(set(nodes)) != len(nodes):
        raise HTTPException(status_code=400, detail="A node is listed twice")
    if body.scale == 1:
        raise HTTPException(status_code=400, detail="A factor of 1 changes nothing; 0 ablates the nodes")
    seconds = estimate_intervene_seconds(model_id, transcoders, body.control_sets)
    return transcoders, nodes, seconds, calculate_circuit_cost(model_id, seconds)


@router.post("/intervene/estimate", response_model=InterveneEstimate)
async def estimate_intervention(
    body: InterveneRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Estimate an intervention's cost before running it."""
    return await estimate_intervention_for(db, current_user, body)


async def estimate_intervention_for(db: Session, current_user: User, body: InterveneRequest) -> dict:
    """POST /circuits/intervene/estimate as the user (also the assistant's quotes)."""
    _, _, seconds, cost = _intervention(await graph_for(db, current_user, body.job_id), body)
    return {"estimated_cost": cost, "estimated_seconds": seconds}


@router.post("/intervene", response_model=InterveneResponse)
@limiter.limit("10/minute")
async def intervene(
    request: Request,
    body: InterveneRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Test a graph, in one GPU call: its chosen feature nodes' activations are multiplied by a
    factor in the real model (0 ablates them; nothing is frozen), and the target token is read
    again. So are control sets of as many other feature nodes active on the prompt, scaled the
    same way: nodes matched in activation anywhere in the prompt, and nodes at the chosen
    nodes' own layers and positions.

    Charged a request to the circuits worker (by the model's size and the number of control sets).
    """
    return await intervene_for(db, current_user, body)


async def intervene_for(db: Session, current_user: User, body: InterveneRequest) -> dict:
    """POST /circuits/intervene as the user (also the assistant's intervene_circuit). Raises HTTPException as the route does."""
    graph = await graph_for(db, current_user, body.job_id)
    transcoders, nodes, _, cost = _intervention(graph, body)
    if not check_sufficient_balance(db, current_user.id, cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    model_id = graph["model"]
    try:
        result = await _get_circuit_worker(model_id)().intervene.remote.aio(
            model_id=model_id,
            set_id=transcoders.id,
            token_ids=graph["token_ids"],
            target_id=graph["target"]["token_id"],
            nodes=[list(node) for node in nodes],
            scale=body.scale,
            control_sets=body.control_sets,
            seed=body.seed,
        )
    except Exception:
        logger.exception("Intervention on %s failed", body.job_id)
        raise HTTPException(status_code=500, detail="The intervention failed")
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    try:
        charge_flat(
            db, current_user.id, cost, TransactionType.CIRCUIT_CHARGE,
            f"Circuit intervention ({model_id}): {len(nodes)} nodes × {body.scale:g}",
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")

    return {
        "job_id": body.job_id,
        "model": model_id,
        "target": {key: graph["target"][key] for key in ("token", "token_id")},
        "scale": body.scale,
        "base": result["base"],
        "chosen": result["chosen"],
        "controls": result["controls"],
        "predictions": result["predictions"],
        "nodes": [
            {"id": graphs.node_id(graphs.FEATURE, n["layer"], n["position"], n["feature"]), **n}
            for n in result["nodes"]
        ],
        "seed": body.seed,
        "seconds": result["seconds"],
        "cost": cost,
    }
