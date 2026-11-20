"""
Circuit Tracing's backend: graphs as Ozera stores them (influence, scores, packaging), the
choice of control nodes for interventions, the registry and budget, and the API's jobs and
interventions with their billing against a stand-in for the circuits worker.

What needs circuit-tracer and a GPU (tracing itself, interventions on a real model) is measured
on Modal: `modal run backend/services/modal_circuits.py::calibrate`.

Run from backend/: python -m pytest tests/test_circuits.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import graph as graphs  # noqa: E402
from core.circuits.budget import (  # noqa: E402
    CONTROL_KINDS,
    MAX_INTERVENE_NODES,
    MAX_PROMPT_CHARS,
    CircuitInputError,
    batch_size,
    check_prompt,
    estimate_intervene_seconds,
    estimate_trace_seconds,
    traced_features,
)
from core.circuits.intervene import check_request, choose_controls  # noqa: E402
from core.circuits.registry import TRANSCODERS, transcoder_set, transcoder_sets  # noqa: E402
from core.circuits.store import graph_path, is_job_id  # noqa: E402


# A small graph: 2 layers, 3 positions, 4 feature nodes, one target.
# Nodes: features 0-3, errors 4-9 (layer × position), tokens 10-12, logit 13.
FEATURES = torch.tensor([[0, 1, 7], [0, 2, 3], [1, 2, 5], [1, 2, 9]])
ACTIVATIONS = torch.tensor([2.0, 1.0, 4.0, 0.5])


def small_graph() -> tuple[torch.Tensor, graphs.Layout, torch.Tensor]:
    layout = graphs.Layout(n_features=4, n_layers=2, n_positions=3, n_logits=1)
    a = torch.zeros(layout.size, layout.size)
    token = lambda p: layout.token_start + p  # noqa: E731
    error = lambda layer, p: layout.error_start + layer * layout.n_positions + p  # noqa: E731
    logit = layout.logit_start
    # Layer-0 features read tokens
    a[0, token(1)] = 2.0
    a[1, token(2)] = 1.0
    a[1, token(1)] = -1.0
    # Layer-1 features read layer-0 features, tokens and a layer-0 error
    a[2, 0] = 3.0
    a[2, token(2)] = 1.0
    a[3, 1] = 0.5
    a[3, error(0, 2)] = 0.5
    # The logit reads features, an error and a token
    a[logit, 2] = 6.0
    a[logit, 3] = -1.0
    a[logit, 0] = 1.0
    a[logit, error(1, 2)] = 1.5
    a[logit, token(2)] = 0.5
    weights = torch.zeros(layout.size)
    weights[logit] = 1.0
    return a, layout, weights


def test_layout_and_ids():
    _, layout, _ = small_graph()
    assert [layout.kind(i) for i in (0, 3, 4, 9, 10, 12, 13)] == ["feature", "feature", "error", "error", "token", "token", "logit"]
    assert layout.size == 14
    assert graphs.node_id("feature", 3, 5, 120) == "f:3:5:120"
    assert graphs.node_id("error", 3, 5) == "e:3:5" and graphs.node_id("token", position=2) == "t:2"
    assert graphs.parse_feature_id("f:3:5:120") == (3, 5, 120)
    for bad in ("e:3:5", "f:3:5", "f:a:1:2", "f:-1:2:3", "", "f:1:2:3:4"):
        assert graphs.parse_feature_id(bad) is None


def test_influence_sums_paths():
    a, layout, weights = small_graph()
    shares = graphs.normalized(a)
    influence = graphs.node_influence(shares, weights)
    # The same as the series' closed form: w (I - S)^-1 - w
    closed = weights @ torch.linalg.inv(torch.eye(layout.size) - shares) - weights
    assert torch.allclose(influence, closed, atol=1e-6)
    # Every path starts at a token or an error node, so their influences add up to 1
    sources = influence[layout.error_start:layout.logit_start].sum()
    assert sources == pytest.approx(1.0, abs=1e-6)
    # The logit's own inputs: 6, -1, 1, 1.5, 0.5 of 10 in absolute value
    assert influence[3] == pytest.approx(0.1, abs=1e-6)
    # Feature 0: its direct edge (0.1) plus all of feature 2's inputs' share through it (0.6 * 3/4)
    assert influence[0] == pytest.approx(0.1 + 0.6 * 0.75, abs=1e-6)


def test_support_keeps_signs():
    a, layout, weights = small_graph()
    influence = graphs.node_influence(graphs.normalized(a), weights)
    support = graphs.node_support(a, weights)
    # Never larger than the influence, and equal to it along all-positive paths
    assert (support.abs() <= influence + 1e-6).all()
    assert support[2] == pytest.approx(0.6, abs=1e-6) and support[0] == pytest.approx(float(influence[0]), abs=1e-6)
    # Feature 3 holds the target down (its edge to the logit is -1 of 10)
    assert support[3] == pytest.approx(-0.1, abs=1e-6)
    # Token 1 raises it through feature 0 and lowers it through features 1 and 3 (a negative
    # edge into a feature that holds the target down raises it): the paths partly cancel
    token = layout.token_start + 1
    through_0 = float(influence[0])  # token 1 is feature 0's only input
    through_1 = -0.5 * (-0.1 * 0.5)  # -1 of 2 into feature 1, then 0.5 of 1 into feature 3, then -1 of 10
    assert support[token] == pytest.approx(through_0 + through_1, abs=1e-6)
    assert float(influence[token]) == pytest.approx(through_0 + abs(through_1), abs=1e-6)


def test_influence_refuses_cycles():
    shares = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    with pytest.raises(RuntimeError):
        graphs.node_influence(shares, torch.tensor([0.0, 0.0, 1.0]), max_steps=50)


def test_scores():
    a, layout, weights = small_graph()
    s = graphs.scores(a, layout, weights)
    influence = graphs.node_influence(graphs.normalized(a), weights)
    tokens = float(influence[layout.token_start:layout.logit_start].sum())
    assert s["replacement"] == pytest.approx(tokens, abs=1e-6)
    assert 0 < s["replacement"] < 1 and 0 < s["completeness"] < 1
    # The target's direct inputs: features 8 of 10, errors 1.5, tokens 0.5
    assert s["target_inputs"]["absolute"] == pytest.approx({"feature": 0.8, "error": 0.15, "token": 0.05})
    assert s["target_inputs"]["sum"] == pytest.approx(8.0)
    assert s["target_inputs"]["signed"]["feature"] == pytest.approx(6.0 / 8.0)

    # Without error nodes' edges everything flows from tokens
    clean = a.clone()
    clean[:, layout.error_start:layout.token_start] = 0
    perfect = graphs.scores(clean, layout, weights)
    assert perfect["replacement"] == pytest.approx(1.0) and perfect["completeness"] == pytest.approx(1.0)


def package(max_nodes=100, max_edges=1000, drop=()):
    a, layout, weights = small_graph()
    node_mask = torch.ones(layout.size, dtype=torch.bool)
    for index in drop:
        node_mask[index] = False
    edge_mask = a != 0
    return graphs.package(a, layout, weights, FEATURES, ACTIVATIONS, node_mask, edge_mask, max_nodes, max_edges), a, layout


def test_package_keeps_what_pruning_kept():
    packed, a, layout = package(drop=(1, 5))
    nodes = packed["nodes"]
    ids = [n.get("id") for n in nodes]
    assert "f:0:2:3" not in ids and "e:0:1" not in ids
    assert {n["kind"] for n in nodes} == {"feature", "error", "token", "logit"}
    feature = next(n for n in nodes if n.get("id") == "f:1:2:5")
    assert feature["layer"] == 1 and feature["position"] == 2 and feature["feature"] == 5 and feature["activation"] == 4.0
    assert feature["influence"] == pytest.approx(0.6, abs=1e-6) and feature["support"] == pytest.approx(0.6, abs=1e-6)
    assert next(n for n in nodes if n.get("id") == "f:1:2:9")["support"] == pytest.approx(-0.1, abs=1e-6)

    edges = packed["edges"]
    count = len(edges["source"])
    assert count == len(edges["target"]) == len(edges["weight"]) == len(edges["share"])
    assert all(0 <= i < len(nodes) for i in edges["source"] + edges["target"])
    # No edge touches a dropped node, and each edge carries its direct effect
    by_pair = {(nodes[s].get("id"), nodes[t].get("id", "logit")): w for s, t, w in zip(edges["source"], edges["target"], edges["weight"])}
    assert by_pair[("f:1:2:5", "logit")] == 6.0 and by_pair[("f:0:1:7", "f:1:2:5")] == 3.0
    assert not any("f:0:2:3" in pair for pair in by_pair)
    # Strongest share first: the logit's largest input
    assert edges["share"] == sorted(edges["share"], reverse=True)
    assert (nodes[edges["source"][0]]["id"], edges["share"][0]) == ("f:1:2:5", pytest.approx(0.6, abs=1e-6))
    assert packed["pruned"] == {"feature_nodes": 3, "error_nodes": 5, "edges": int((a != 0).sum()), "kept_edges": count}


def test_package_caps_nodes_and_edges():
    packed, _, layout = package(max_nodes=2)
    inner = [n for n in packed["nodes"] if n["kind"] in ("feature", "error")]
    # The two most influential of the features and errors; tokens and the logit always stay
    assert sorted(n["id"] for n in inner) == ["f:0:1:7", "f:1:2:5"]
    assert sum(n["kind"] == "token" for n in packed["nodes"]) == layout.n_positions
    assert sum(n["kind"] == "logit" for n in packed["nodes"]) == 1
    capped, _, _ = package(max_edges=3)
    assert len(capped["edges"]["source"]) == 3 and capped["pruned"]["kept_edges"] == 3


def test_summary():
    packed, a, layout = package()
    _, _, weights = small_graph()
    graph = {
        **packed,
        "target": {"token": " Paris", "token_id": 5, "prob": 0.4, "logit": 8.0, "rank": 0},
        "tokens": ["<bos>", "The", " capital"],
        "scores": graphs.scores(a, layout, weights),
        "counts": {"active_features": 9, "traced_features": 4},
        "seconds": 1.5,
    }
    brief = graphs.summary(graph)
    assert brief["target"]["token"] == " Paris" and brief["tokens"] == 3
    assert brief["feature_nodes"] == 4 and brief["error_nodes"] == 6 and brief["edges"] == len(packed["edges"]["source"])
    assert brief["error_inputs"] == pytest.approx(0.15)
    assert brief["active_features"] == 9 and brief["traced_features"] == 4 and brief["seconds"] == 1.5


# Controls

def active_features(seed: int = 0, layers: int = 4, positions: int = 5, per_site: int = 8):
    generator = torch.Generator().manual_seed(seed)
    rows = [(layer, position, feature) for layer in range(layers) for position in range(1, positions) for feature in range(per_site)]
    values = torch.rand(len(rows), generator=generator) * 20 + 0.1
    return torch.tensor(rows), values


def test_controls_match_and_avoid_the_chosen():
    active, values = active_features()
    chosen = [3, 40, 41, 100]
    controls = choose_controls(active, values, chosen, 5, torch.Generator().manual_seed(1))
    assert set(controls) == set(CONTROL_KINDS)
    for kind, sets in controls.items():
        assert len(sets) == 5
        for picked in sets:
            assert len(picked) == len(chosen) and len(set(picked)) == len(picked)
            assert not set(picked) & set(chosen)
    # Site controls sit where the chosen nodes do
    for picked in controls["site"]:
        assert [tuple(active[i, :2].tolist()) for i in picked] == [tuple(active[i, :2].tolist()) for i in chosen]
    # Activation controls are near the chosen nodes' activations: nearer than a random node on average
    gaps = torch.tensor([abs(float(values[c].log() - values[i].log())) for picked in controls["activation"] for c, i in zip(picked, chosen)])
    assert gaps.mean() < 0.2
    # Sets differ from one another, and the same seed gives the same sets
    assert len({tuple(p) for p in controls["activation"]}) > 1
    assert controls == choose_controls(active, values, chosen, 5, torch.Generator().manual_seed(1))


def test_controls_run_out_gracefully():
    # One site with two features: a chosen node has one site match, and a second chosen node none
    active = torch.tensor([[0, 1, 4], [0, 1, 9]])
    values = torch.tensor([1.0, 2.0])
    controls = choose_controls(active, values, [0], 2, torch.Generator().manual_seed(0))
    assert controls["site"] == [[1], [1]]
    assert choose_controls(active, values, [0, 1], 1, torch.Generator().manual_seed(0)) == {"activation": [[]], "site": [[]]}


def test_intervention_requests_are_checked():
    check_request([(0, 1, 2)], 0.0, 5)
    check_request([(0, 1, 2), (0, 1, 3)], 2.0, 0)
    for nodes, scale, sets in (
        ([], 0.0, 5),
        ([(0, 1, 2), (0, 1, 2)], 0.0, 5),
        ([(0, 1, i) for i in range(MAX_INTERVENE_NODES + 1)], 0.0, 5),
        ([(0, 1, 2)], 1.0, 5),
        ([(0, 1, 2)], 100.0, 5),
        ([(0, 1, 2)], 0.0, 50),
    ):
        with pytest.raises(CircuitInputError):
            check_request(nodes, scale, sets)


# Registry, budget and store

def test_registry():
    from core.open_source.registry import OPEN_SOURCE_MODELS

    assert TRANSCODERS, "some model can be traced"
    for model_id, sets in TRANSCODERS.items():
        assert model_id in OPEN_SOURCE_MODELS
        assert sets and len({s.id for s in sets}) == len(sets)
        assert transcoder_set(model_id, None) is sets[0]
        for s in sets:
            assert transcoder_set(model_id, s.id) is s
            assert s.reference.startswith(s.repo) and "/" not in s.directory
            # An instruct model reads its own checkpoint's transcoders
            assert s.repo.endswith("-it") == OPEN_SOURCE_MODELS[model_id].is_instruct
    assert transcoder_sets("nano") == () and transcoder_set("nano", None) is None
    assert transcoder_set("gemma-3-270m", "nope") is None


def test_budget():
    check_prompt("The capital of France is")
    for bad in ("", "   ", "x" * (MAX_PROMPT_CHARS + 1)):
        with pytest.raises(CircuitInputError):
            check_prompt(bad)
    small = transcoder_set("gemma-3-270m", None)
    large = transcoder_set("gemma-3-1b", None)
    assert estimate_trace_seconds("gemma-3-270m", small, 10) < estimate_trace_seconds("gemma-3-270m", small, 60)
    assert estimate_trace_seconds("gemma-3-270m", small, 10) < estimate_trace_seconds("gemma-3-1b", large, 10)
    # An instruct model is priced as its base model
    assert estimate_trace_seconds("gemma-3-1b-it", transcoder_set("gemma-3-1b-it", None), 10) == estimate_trace_seconds("gemma-3-1b", large, 10)
    assert estimate_intervene_seconds("gemma-3-270m", small, 0) < estimate_intervene_seconds("gemma-3-270m", small, 5)
    # A denser transcoder set has more active features to read
    assert estimate_trace_seconds("gemma-3-270m", small, 30) < estimate_trace_seconds("gemma-3-270m", transcoder_set("gemma-3-270m", "gemma-scope-2-16k-big"), 30)
    # Long prompts take fewer source nodes per backward pass, and prompts dense with active features trace fewer of them
    assert batch_size("gemma-3-270m", 8) == 256 and batch_size("gemma-3-270m", 64) == 64 and batch_size("gemma-3-1b-it", 64) == 24
    assert batch_size("gemma-3-1b", 1000) == 16
    assert traced_features(3000, 200) == 3000 and traced_features(30_000, 1000) == 7500
    assert traced_features(250_000, 2000) == 1_000_000_000 // 252_000


def test_store_paths():
    assert is_job_id("circuit_20261119120000_0a1b2c3d")
    assert graph_path("circuit_20261119120000_0a1b2c3d") == "/saes/circuits/graphs/circuit_20261119120000_0a1b2c3d/graph.json"
    for bad in ("", "atlas_20261119120000_0a1b2c3d", "circuit_../x", "circuit_20261119120000_0a1b2c3d/.."):
        assert not is_job_id(bad)
        with pytest.raises(ValueError):
            graph_path(bad)


# The API, against a stand-in for the circuits worker

import asyncio  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from fastapi import HTTPException  # noqa: E402


def stored_graph(job_id: str, model_id: str, set_id: str, prompt: str, target) -> dict:
    """A graph as the worker stores one, made from the small graph above."""
    from core.circuits.registry import transcoder_set as find_set

    packed, a, layout = package()
    _, _, weights = small_graph()
    transcoders = find_set(model_id, set_id)
    for node in packed["nodes"]:
        if node["kind"] == "feature":
            node.update(promotes=[" Paris"], suppresses=[" London"], fires=[0.0, 1.0, 2.0])
        elif node["kind"] == "token":
            node["token"] = ["<bos>", "The", " capital"][node["position"]]
        elif node["kind"] == "logit":
            node.update(id="l:5", token=target or " Paris", token_id=5, prob=0.4, position=2)
    return {
        "version": graphs.GRAPH_VERSION, "job_id": job_id, "model": model_id,
        "transcoders": {"id": transcoders.id, "name": transcoders.name, "reference": transcoders.reference, "width": transcoders.width},
        "prompt": prompt, "chat_template": False, "tokens": ["<bos>", "The", " capital"], "token_ids": [2, 10, 11], "num_layers": 2,
        "target": {"token": target or " Paris", "token_id": 5, "logit": 8.0, "prob": 0.4, "rank": 0}, "target_given": bool(target),
        "predictions": [{"token": " Paris", "token_id": 5, "prob": 0.4}],
        "scores": graphs.scores(a, layout, weights),
        "counts": {"active_features": 9, "traced_features": 4, **packed["pruned"]},
        "settings": {}, "nodes": packed["nodes"], "edges": packed["edges"], "seconds": 1.5, "gpu_peak_gib": 3.5,
    }


class FakeWorker:
    """The circuits worker's calls as the backend makes them: jobs end when `finish` says so."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: dict[str, dict] = {}
        self.interventions: list[dict] = []
        self.fail_spawn = False
        self.intervene_error = None
        # End every job as soon as it's started
        self.auto_finish = False

    def __call__(self):
        async def spawn(job_id, model_id, set_id, prompt, target):
            if self.fail_spawn:
                raise RuntimeError("no capacity")
            call_id = f"fc-{len(self.calls)}"
            self.calls[call_id] = {"job_id": job_id, "args": (model_id, set_id, prompt, target), "outcome": None}
            if self.auto_finish:
                self.finish(call_id)
            return SimpleNamespace(object_id=call_id)

        async def intervene(**kwargs):
            self.interventions.append(kwargs)
            if self.intervene_error:
                return {"error": self.intervene_error, "seconds": 0.5}
            reading = lambda logit, **extra: {"logit": logit, "prob": 0.4, "rank": 0, **extra}  # noqa: E731
            controls = [reading(7.5, change=-0.5, nodes=len(kwargs["nodes"]), activation=2.0) for _ in range(kwargs["control_sets"])]
            return {
                "base": reading(8.0),
                "chosen": reading(3.0, change=-5.0, nodes=len(kwargs["nodes"]), activation=3.0),
                "controls": {"activation": controls, "site": controls},
                "predictions": {"base": [{"token": " Paris", "token_id": 5, "prob": 0.4}], "chosen": [{"token": " London", "token_id": 6, "prob": 0.3}]},
                "nodes": [{"layer": n[0], "position": n[1], "feature": n[2], "activation": 3.0} for n in kwargs["nodes"]],
                "scale": kwargs["scale"], "seconds": 2.0,
            }

        return SimpleNamespace(trace=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)), intervene=SimpleNamespace(remote=SimpleNamespace(aio=intervene)))

    def finish(self, call_id: str, error=None, seconds: float = 20.0):
        """End a job's call: its graph is written (unless it failed) and its outcome reported."""
        from services.circuit_jobs import Outcome

        call = self.calls[call_id]
        if error:
            call["outcome"] = Outcome("failed", error, seconds)
            return
        graph = stored_graph(call["job_id"], *call["args"])
        self.files[graph_path(call["job_id"])] = json.dumps(graph).encode()
        call["outcome"] = Outcome("completed", None, seconds, graphs.summary(graph))


@pytest.fixture()
def api(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.circuits
    from models.database import Base, CreditBalance, User
    from services import assistant_runs, atlas_store, circuit_jobs, credit_service

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(assistant_runs, "SessionLocal", Session)
    # SQLite has no row locks (Postgres's FOR UPDATE and lock_timeout)
    monkeypatch.setattr(credit_service, "get_credit_balance_for_update",
                        lambda db, user_id: db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first())
    worker = FakeWorker()
    monkeypatch.setattr(api.circuits, "_get_circuit_worker", lambda model_id: worker)
    monkeypatch.setattr(circuit_jobs, "_worker", lambda model_id: worker)

    async def call_outcome(call_id):
        return worker.calls[call_id]["outcome"]

    async def progress(job_id):
        return {"stage": "Tracing", "fraction": 0.5}

    async def read_bytes(path):
        if path not in worker.files:
            raise FileNotFoundError(path)
        return worker.files[path]

    monkeypatch.setattr(circuit_jobs, "call_outcome", call_outcome)
    monkeypatch.setattr(circuit_jobs, "progress", progress)
    monkeypatch.setattr(atlas_store, "read_bytes", read_bytes)
    db = Session()
    users = [User(email="circuits@example.com", is_active=True), User(email="other@example.com", is_active=True)]
    db.add_all(users)
    db.commit()
    for user in users:
        db.add(CreditBalance(user_id=user.id, balance_usd=1, reserved_usd=0))
    db.commit()
    yield db, users[0], users[1], worker
    db.close()


def _transactions(db, user) -> list:
    from models.database import Transaction

    return db.query(Transaction).filter(Transaction.user_id == user.id).order_by(Transaction.id).all()


def _balance(db, user) -> float:
    from models.database import CreditBalance

    db.expire_all()
    return float(db.query(CreditBalance).filter(CreditBalance.user_id == user.id).first().balance_usd)


def test_models_list_those_with_transcoders():
    from api.circuits import list_models
    from api.schemas.circuits import CircuitModelInfo

    models = [CircuitModelInfo.model_validate(m) for m in asyncio.run(list_models())]
    assert {m.model_id for m in models} == set(TRANSCODERS)
    gemma = next(m for m in models if m.model_id == "gemma-3-270m")
    assert gemma.num_layers == 18 and gemma.transcoders[0].width == 16384 and not gemma.is_instruct


def test_a_graph_job_is_charged_built_and_read(api):
    from api.circuits import estimate_trace_for, graph_for, job_for, list_jobs_for, submit_job_for
    from api.schemas.circuits import CircuitJobInfo, CircuitJobList, TraceEstimate, TraceRequest
    from models.database import TransactionType
    from services.credit_service import calculate_circuit_cost

    db, user, other, worker = api
    body = TraceRequest(model="gemma-3-270m", prompt="The capital of France is")
    estimate = TraceEstimate.model_validate(asyncio.run(estimate_trace_for(user, body)))
    assert estimate.within_limit and estimate.estimated_cost == calculate_circuit_cost("gemma-3-270m", estimate.estimated_seconds)
    # Never below what a lone request costs Ozera: the cold start, the run, and the warm window
    from services.credit_service import CIRCUIT_COLD_START_SECONDS, CIRCUIT_SCALEDOWN_SECONDS, PRICE_MARKUP, container_cost_per_second
    from services.modal_circuits import CIRCUIT_CPU, CIRCUIT_MEMORY_MB
    lifetime = CIRCUIT_COLD_START_SECONDS + estimate.estimated_seconds + CIRCUIT_SCALEDOWN_SECONDS["l4"]
    assert estimate.estimated_cost >= lifetime * container_cost_per_second("l4", CIRCUIT_CPU, CIRCUIT_MEMORY_MB / 1024) * PRICE_MARKUP

    job = CircuitJobInfo.model_validate(asyncio.run(submit_job_for(db, user, body)))
    assert job.status == "running" and job.cost == estimate.estimated_cost and job.target is None and job.summary is None
    assert job.transcoders == "gemma-scope-2-16k-small" and job.transcoders_name.startswith("Gemma Scope 2")
    assert worker.calls["fc-0"]["args"] == ("gemma-3-270m", "gemma-scope-2-16k-small", "The capital of France is", None)
    charges = _transactions(db, user)
    assert [c.transaction_type for c in charges] == [TransactionType.CIRCUIT_CHARGE] and float(-charges[0].amount_usd) == job.cost
    assert _balance(db, user) == pytest.approx(1 - job.cost)

    # While it runs: progress, and no graph yet
    polled = CircuitJobInfo.model_validate(asyncio.run(job_for(db, user, job.job_id)))
    assert polled.status == "running" and polled.progress.stage == "Tracing"
    with pytest.raises(HTTPException) as e:
        asyncio.run(graph_for(db, user, job.job_id))
    assert e.value.status_code == 409

    worker.finish("fc-0")
    done = CircuitJobInfo.model_validate(asyncio.run(job_for(db, user, job.job_id)))
    assert done.status == "completed" and done.progress is None and done.finished_at is not None
    assert done.summary["target"]["token"] == " Paris" and done.summary["feature_nodes"] == 4 and 0 < done.summary["replacement"] < 1
    graph = asyncio.run(graph_for(db, user, job.job_id))
    assert graph["job_id"] == job.job_id and graph["prompt"] == body.prompt and len(graph["nodes"]) == 14
    # Settled once: no refund, nothing more charged
    assert len(_transactions(db, user)) == 1 and _balance(db, user) == pytest.approx(1 - job.cost)

    listed = CircuitJobList.model_validate(asyncio.run(list_jobs_for(db, user)))
    assert [j.job_id for j in listed.jobs] == [job.job_id]
    # A graph is its user's alone
    assert asyncio.run(list_jobs_for(db, other))["jobs"] == []
    for read in (job_for, graph_for):
        with pytest.raises(HTTPException) as e:
            asyncio.run(read(db, other, job.job_id))
        assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        asyncio.run(job_for(db, user, "circuit_../../etc"))
    assert e.value.status_code == 404


def test_a_failed_job_is_refunded_all_but_its_gpu_time(api):
    from api.circuits import job_for, submit_job_for
    from api.schemas.circuits import TraceRequest
    from models.database import TransactionType
    from services.credit_service import calculate_circuit_cost

    db, user, _, worker = api
    body = TraceRequest(model="gemma-3-1b", prompt="The capital of France is", target=" Lyon")
    job = asyncio.run(submit_job_for(db, user, body))
    assert job["target"] == " Lyon" and worker.calls["fc-0"]["args"][3] == " Lyon"
    worker.finish("fc-0", error="The prompt is 120 tokens on this model; graphs are of at most 64", seconds=4.0)
    failed = asyncio.run(job_for(db, user, job["job_id"]))
    used = calculate_circuit_cost("gemma-3-1b", 4.0)
    assert failed["status"] == "failed" and "120 tokens" in failed["error"]
    assert 0 < used < job["cost"] and failed["refunded"] == pytest.approx(job["cost"] - used)
    types = [c.transaction_type for c in _transactions(db, user)]
    assert types == [TransactionType.CIRCUIT_CHARGE, TransactionType.CIRCUIT_REFUND]
    assert _balance(db, user) == pytest.approx(1 - used)
    # Polling again settles nothing more
    asyncio.run(job_for(db, user, job["job_id"]))
    assert len(_transactions(db, user)) == 2

    # A job that can't be started is refunded in full
    worker.fail_spawn = True
    before = _balance(db, user)
    with pytest.raises(HTTPException) as e:
        asyncio.run(submit_job_for(db, user, body))
    assert e.value.status_code == 502 and _balance(db, user) == pytest.approx(before)


def test_jobs_are_checked_before_a_gpu_starts(api):
    from api.circuits import MAX_ACTIVE_JOBS, submit_job_for
    from api.schemas.circuits import TraceRequest
    from models.database import CreditBalance

    db, user, other, worker = api
    for body, status in (
        (TraceRequest(model="nano", prompt="Hello"), 400),
        (TraceRequest(model="smollm-135m", prompt="Hello"), 400),
        (TraceRequest(model="gemma-3-270m", transcoders="nope", prompt="Hello"), 404),
        (TraceRequest(model="gemma-3-270m", prompt="   "), 400),
    ):
        with pytest.raises(HTTPException) as e:
            asyncio.run(submit_job_for(db, user, body))
        assert e.value.status_code == status, body
    assert not worker.calls and not _transactions(db, user)

    # A balance that doesn't cover the estimate: nothing is recorded or started
    db.query(CreditBalance).filter(CreditBalance.user_id == other.id).update({"balance_usd": 0.05})
    db.commit()
    with pytest.raises(HTTPException) as e:
        asyncio.run(submit_job_for(db, other, TraceRequest(model="gemma-3-270m", prompt="Hello there")))
    assert e.value.status_code == 402 and not worker.calls

    # Only a few graphs at once
    body = TraceRequest(model="gemma-3-270m", prompt="Hello there")
    for _ in range(MAX_ACTIVE_JOBS):
        asyncio.run(submit_job_for(db, user, body))
    with pytest.raises(HTTPException) as e:
        asyncio.run(submit_job_for(db, user, body))
    assert e.value.status_code == 409 and len(worker.calls) == MAX_ACTIVE_JOBS


def _built(db, user, worker, model: str = "gemma-3-270m") -> str:
    from api.circuits import job_for, submit_job_for
    from api.schemas.circuits import TraceRequest

    job = asyncio.run(submit_job_for(db, user, TraceRequest(model=model, prompt="The capital of France is")))
    worker.finish(f"fc-{len(worker.calls) - 1}")
    asyncio.run(job_for(db, user, job["job_id"]))
    return job["job_id"]


def test_an_intervention_runs_on_the_graphs_nodes_and_is_charged(api):
    from api.circuits import estimate_intervention_for, intervene_for
    from api.schemas.circuits import InterveneRequest, InterveneResponse
    from models.database import TransactionType

    db, user, other, worker = api
    job_id = _built(db, user, worker)
    body = InterveneRequest(job_id=job_id, nodes=["f:1:2:5", "f:0:1:7"], scale=0.0)
    estimate = asyncio.run(estimate_intervention_for(db, user, body))
    result = InterveneResponse.model_validate(asyncio.run(intervene_for(db, user, body)))
    call = worker.interventions[0]
    assert call["model_id"] == "gemma-3-270m" and call["set_id"] == "gemma-scope-2-16k-small"
    assert call["token_ids"] == [2, 10, 11] and call["target_id"] == 5 and call["nodes"] == [[1, 2, 5], [0, 1, 7]] and call["scale"] == 0.0
    assert result.chosen.change == -5.0 and [len(sets) for sets in result.controls.values()] == [5, 5]
    assert [n.id for n in result.nodes] == body.nodes and result.target["token"] == " Paris"
    assert result.cost == estimate["estimated_cost"] > 0
    charges = _transactions(db, user)
    assert charges[-1].transaction_type == TransactionType.CIRCUIT_CHARGE and float(-charges[-1].amount_usd) == result.cost

    # Refused before the worker is called: nodes that aren't the graph's features, a factor of 1, someone else's graph
    charged = len(charges)
    for bad in (
        InterveneRequest(job_id=job_id, nodes=["f:1:2:999"]),
        InterveneRequest(job_id=job_id, nodes=["e:0:2"]),
        InterveneRequest(job_id=job_id, nodes=["f:1:2:5", "f:1:2:5"]),
        InterveneRequest(job_id=job_id, nodes=["f:1:2:5"], scale=1.0),
    ):
        with pytest.raises(HTTPException) as e:
            asyncio.run(intervene_for(db, user, bad))
        assert e.value.status_code == 400, bad
    with pytest.raises(HTTPException) as e:
        asyncio.run(intervene_for(db, other, body))
    assert e.value.status_code == 404
    assert len(worker.interventions) == 1 and len(_transactions(db, user)) == charged

    # What the worker refuses isn't charged
    worker.intervene_error = "Feature 5 of layer 1 isn't active at position 2 of this prompt"
    with pytest.raises(HTTPException) as e:
        asyncio.run(intervene_for(db, user, body))
    assert e.value.status_code == 400 and "isn't active" in e.value.detail and len(_transactions(db, user)) == charged


def test_prices_cover_modals_bill():
    """A run's price is never below Modal's cost for the container's whole lifetime, plus markup."""
    from services.credit_service import (
        CIRCUIT_COLD_START_SECONDS, CIRCUIT_SCALEDOWN_SECONDS, PRICE_MARKUP, calculate_circuit_cost, container_cost_per_second,
    )
    from services.modal_circuits import CIRCUIT_CPU, CIRCUIT_MEMORY_MB

    rate = container_cost_per_second("l4", CIRCUIT_CPU, CIRCUIT_MEMORY_MB / 1024)
    for model_id in TRANSCODERS:
        for seconds in (0.0, 10.0, 120.0):
            lifetime = CIRCUIT_COLD_START_SECONDS + seconds + CIRCUIT_SCALEDOWN_SECONDS["l4"]
            assert calculate_circuit_cost(model_id, seconds) >= lifetime * rate * PRICE_MARKUP
    with pytest.raises(ValueError):
        calculate_circuit_cost("qwen3-4b", 10.0)


# The assistant

def _quote_and_run(name: str, user_id: int, args: dict):
    from core.assistant.loop import RunQuote
    from services import assistant_runs
    from services.assistant_tools import ToolContext

    ctx = ToolContext(user_id)
    quoted = asyncio.run(assistant_runs.quote(name, ctx, args))
    assert isinstance(quoted, RunQuote), getattr(quoted, "content", quoted)
    outcome = asyncio.run(assistant_runs.run(name, ctx, args))
    return quoted, outcome


def _read(name: str, user_id: int, args: dict):
    from services import assistant_tools

    return asyncio.run(assistant_tools.execute(name, assistant_tools.ToolContext(user_id), args))


def test_the_assistant_traces_reads_and_intervenes(api, monkeypatch):
    from services import assistant_runs, assistant_tools

    db, user, other, worker = api
    monkeypatch.setattr(assistant_tools, "SessionLocal", assistant_runs.SessionLocal)
    monkeypatch.setattr(assistant_runs, "CIRCUIT_POLL_SECONDS", 0)
    worker.auto_finish = True

    args = {"model": "gemma-3-270m", "transcoders": "", "prompt": "The capital of France is", "target": ""}
    quoted, traced = _quote_and_run("trace_circuit", user.id, args)
    assert not traced.error, traced.content
    details = dict(quoted.details)
    assert details["Target"] == "the model's top prediction" and details["Transcoders"].startswith("Gemma Scope 2") and quoted.estimate > 0
    graph = traced.content
    assert graph["status"] == "completed" and graph["graph_id"].startswith("circuit_") and graph["target"]["token"] == " Paris"
    assert 0 < graph["replacement_score"] < 1 and graph["error_share"] == pytest.approx(1 - graph["replacement_score"], abs=2e-3)
    top = graph["top_features"]
    assert [f["node"] for f in top][:2] == ["f:1:2:5", "f:0:1:7"] and top[0]["promotes"] == [" Paris"]
    assert top[0]["inputs"][0] == {"from": "f:0:1:7", "weight": 3.0} and top[1]["inputs"][0]["from"] == "token 'The'"
    assert traced.charged == pytest.approx(quoted.estimate) and traced.result["page"] == "circuits"
    assert traced.result["data"]["job"]["job_id"] == graph["graph_id"] and traced.result["data"]["job"]["status"] == "completed"
    assert len(json.dumps(graph)) < 6000, "the summary fits a tool result"

    # Reading it again is free; another user can't
    read = _read("get_circuit", user.id, {"graph_id": graph["graph_id"]})
    assert not read.error and read.content["top_features"] == top
    listed = _read("get_circuit", user.id, {"graph_id": ""})
    assert [g["graph_id"] for g in listed.content["graphs"]] == [graph["graph_id"]] and listed.content["graphs"][0]["target"] == " Paris"
    assert _read("get_circuit", other.id, {"graph_id": graph["graph_id"]}).error
    jobs = _read("list_jobs", user.id, {})
    assert jobs.content["circuits"][0]["graph_id"] == graph["graph_id"]
    assert "transcoders" in next(m for m in _read("list_models", user.id, {}).content["models"] if m["id"] == "gemma-3-270m")
    assert "transcoders" not in next(m for m in _read("list_models", user.id, {}).content["models"] if m["id"] == "nano")

    quoted, tested = _quote_and_run("intervene_circuit", user.id, {"graph_id": graph["graph_id"], "nodes": [], "top": 2, "scale": 0})
    assert not tested.error, tested.content
    assert dict(quoted.details)["Change"] == "ablated (× 0)" and dict(quoted.details)["Nodes"].startswith("2: f:1:2:5, f:0:1:7")
    out = tested.content
    assert out["nodes"] == ["f:1:2:5", "f:0:1:7"] and out["target_logit"] == {"before": 8.0, "after": 3.0, "change": -5.0}
    assert out["control_changes"] == {"activation": [-0.5] * 5, "site": [-0.5] * 5} and out["larger_than_every_control"] == {"activation": True, "site": True}
    assert out["top_prediction_after"] == " London" and tested.charged == pytest.approx(quoted.estimate)
    assert tested.result["page"] == "circuits" and tested.result["data"]["response"]["job_id"] == graph["graph_id"]
    assert worker.interventions[-1]["nodes"] == [[1, 2, 5], [0, 1, 7]]

    # Bad input is refused before anything is priced as runnable
    from core.assistant.loop import ToolOutcome
    from services.assistant_tools import ToolContext
    for name, bad in (
        ("trace_circuit", {**args, "model": "nano"}),
        ("trace_circuit", {**args, "prompt": ""}),
        ("intervene_circuit", {"graph_id": graph["graph_id"], "nodes": ["f:1:2:5"], "top": 3, "scale": 0}),
        ("intervene_circuit", {"graph_id": graph["graph_id"], "nodes": [], "top": 0, "scale": 0}),
        ("intervene_circuit", {"graph_id": graph["graph_id"], "nodes": ["f:9:9:9"], "top": 0, "scale": 0}),
        ("intervene_circuit", {"graph_id": graph["graph_id"], "nodes": [], "top": 2, "scale": 1}),
        ("intervene_circuit", {"graph_id": "circuit_20000101000000_00000000", "nodes": [], "top": 2, "scale": 0}),
    ):
        refused = asyncio.run(assistant_runs.quote(name, ToolContext(user.id), bad))
        assert isinstance(refused, ToolOutcome) and refused.error, (name, bad)


def test_a_trace_that_fails_tells_the_assistant(api, monkeypatch):
    from services import assistant_runs

    db, user, _, worker = api
    monkeypatch.setattr(assistant_runs, "CIRCUIT_POLL_SECONDS", 0)
    original = worker.finish
    worker.auto_finish = True
    worker.finish = lambda call_id: original(call_id, error="The prompt is 120 tokens on this model; graphs are of at most 64", seconds=3.0)
    quoted, traced = _quote_and_run("trace_circuit", user.id, {"model": "gemma-3-270m", "transcoders": "", "prompt": "Hello there", "target": ""})
    assert traced.error and traced.content["status"] == "failed" and "120 tokens" in traced.content["error"]
    # What it was charged is the GPU time it used (a cold start, three seconds and the warm
    # window): the estimate less the refund, which is nothing when that comes to under a cent
    from services.credit_service import calculate_circuit_cost
    unused = round(quoted.estimate - calculate_circuit_cost("gemma-3-270m", 3.0), 4)
    refund = unused if unused >= 0.01 else 0
    assert traced.content["refunded_usd"] == pytest.approx(refund) and traced.charged == pytest.approx(quoted.estimate - refund)
    assert 0 < traced.charged <= quoted.estimate


def test_the_graph_is_drawn_as_a_figure():
    from api.export import DataFigureRequest, export_data_figure

    packed, _, layout = package()
    nodes = [
        {"kind": n["kind"], "layer": n.get("layer"), "position": n.get("position", layout.n_positions - 1), "influence": n["influence"],
         "support": n["support"], "pinned": n.get("id") == "f:1:2:5"}
        for n in packed["nodes"]
    ]
    edges = [list(edge) for edge in zip(packed["edges"]["source"], packed["edges"]["target"], packed["edges"]["weight"], packed["edges"]["share"])]
    graph = {"tokens": ["<bos>", "The", " capital"], "num_layers": 2, "target": " Paris", "nodes": nodes, "edges": edges}
    user = SimpleNamespace(id=1)
    response = asyncio.run(export_data_figure(DataFigureRequest(kind="graph", graph=graph, title="The capital → Paris"), user))
    assert response.media_type == "image/png" and response.body[:4] == b"\x89PNG" and len(response.body) > 5000
    assert b"<svg" in asyncio.run(export_data_figure(DataFigureRequest(kind="graph", graph=graph, config={"format": "svg"}), user)).body
    for bad in (
        DataFigureRequest(kind="graph"),
        DataFigureRequest(kind="graph", graph={**graph, "edges": [[0, 99, 1.0, 0.5]]}),
        DataFigureRequest(kind="graph", graph={**graph, "nodes": [{"kind": "feature", "layer": 5, "position": 0}]}),
        DataFigureRequest(kind="graph", graph={**graph, "nodes": [{"kind": "token", "position": 7}]}),
    ):
        with pytest.raises(HTTPException):
            asyncio.run(export_data_figure(bad, user))
