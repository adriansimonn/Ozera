"""
The assistant's lab tools: sweeps and geometry runs kept for a while (services/lab_runs.py), a
sweep's probe saved as the Probe Lab's Save button saves it, two kept runs compared, the user's
own rows and concepts, steering a kept run's concept, and estimate pricing a run from its own
input. Run from backend/: python -m pytest tests/test_assistant_lab_tools.py
"""

import asyncio
import base64
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.assistant.loop import RunQuote, ToolOutcome  # noqa: E402
from core.assistant.pages import check_labels, check_rows  # noqa: E402
from services import lab_runs  # noqa: E402
from services.assistant_tools import ToolContext  # noqa: E402

USER = SimpleNamespace(id=7)


@pytest.fixture(autouse=True)
def fresh_runs():
    lab_runs.clear()
    yield
    lab_runs.clear()


def _wire(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array, dtype=np.float32)
    return {"encoding": "base64_float32", "values": base64.b64encode(array.tobytes()).decode(), "shape": list(array.shape)}


# Kept runs

def test_runs_are_kept_per_user_and_kind(monkeypatch):
    first = lab_runs.keep(1, "sweep", {"n": 1}, test_fraction=0.3)
    assert first.startswith("sweep-")
    assert lab_runs.get(1, "sweep", first) == ({"n": 1}, {"test_fraction": 0.3})
    assert lab_runs.get(2, "sweep", first) is None, "another user's run"
    assert lab_runs.get(1, "geometry", first) is None, "another kind"
    ids = [lab_runs.keep(1, "sweep", {"n": i}) for i in range(lab_runs.KEEP_PER_USER)]
    assert lab_runs.get(1, "sweep", first) is None, "the oldest goes past the per-user limit"
    assert all(lab_runs.get(1, "sweep", i) for i in ids)
    # A run is forgotten after its time
    import time

    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + lab_runs.TTL_SECONDS + 1)
    assert lab_runs.get(1, "sweep", ids[-1]) is None


# Rows and labels

def test_rows_are_checked():
    rows = [{"text": f"t{i}", "label": i % 2} for i in range(20)]
    checked, problem = check_rows(rows)
    assert problem is None and len(checked) == 20
    assert check_rows(rows[:19])[1].startswith("Each label needs at least 10")
    assert check_rows([*rows, {"text": "x", "label": 2}])[1] == "Row 21's label is 0 or 1"
    assert check_rows([*rows, {"text": " ", "label": 0}])[1] == "Row 21 needs a text"
    assert check_rows("not json")[1]
    assert check_labels("false, true") == (["false", "true"], None)
    assert check_labels(["a"])[1]


def test_dataset_specs():
    from services.assistant_runs import _dataset_size, _dataset_spec

    rows = [{"text": f"text {i}", "label": i % 2} for i in range(20)]
    spec = _dataset_spec({"dataset": "", "rows": rows, "label_names": ["no", "yes"], "dataset_name": " My  set "})
    assert spec.builtin_id is None and len(spec.rows) == 20 and spec.label_names == ["no", "yes"] and spec.name == "My set"
    assert _dataset_size(spec) == (20, sum(len(r["text"]) for r in rows), "My set")
    with pytest.raises(HTTPException) as e:
        _dataset_spec({"dataset": "truth", "rows": rows})
    assert "not both" in e.value.detail
    with pytest.raises(HTTPException):
        _dataset_spec({"dataset": "", "rows": []})
    assert _dataset_spec({"dataset": "", "rows": []}, required=False) is None


# Saving a sweep's probe

def _sweep(positions: int = 3, d: int = 4) -> tuple[dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    logreg, means = rng.normal(size=(positions, d)), rng.normal(size=(positions, d))
    metrics = {k: [0.5 + 0.1 * p for p in range(positions)] for k in ("test_auroc", "control_test_auroc", "score_std", "score_mean")}

    def method(weights):
        return {"weights": _wire(weights), "biases": [0.1 * p for p in range(positions)], "metrics": metrics}

    run = {
        "model": "nano", "num_layers": positions - 1, "chat_template": False, "read_span": "text", "seed": 3,
        "dataset": {"name": "Truth", "builtin_id": "truth", "label_names": ["false", "true"], "n_train": 8, "n_test": 2,
                    "n_ood": 0, "ood_name": None},
        "poolings": {"mean": {"methods": {"logreg": {**method(logreg), "l2": [1.0] * positions}, "diff_means": method(means)},
                              "act_norms": [2.0] * positions}},
    }
    return run, logreg, means


def test_a_sweeps_probe_is_saved_as_the_page_saves_it():
    from services.assistant_runs import _save_request

    run, logreg, means = _sweep()
    run_id = lab_runs.keep(USER.id, "sweep", run, test_fraction=0.3)
    body = _save_request(USER, {"run_id": run_id, "layer": 2, "pooling": "mean", "method": "logreg", "name": ""})
    assert body.layer == 1, "the decoder layer is the sweep position minus 1"
    assert np.allclose(body.weights, logreg[2], atol=1e-6)
    expected_gap = float(logreg[2] @ means[2] / np.linalg.norm(logreg[2]))
    assert body.normalization["class_gap"] == pytest.approx(expected_gap, rel=1e-5)
    assert body.name == "Truth · L1 mean LR"
    assert body.metrics["test_auroc"] == pytest.approx(0.7) and body.metrics["embedding_test_auroc"] == pytest.approx(0.5)
    assert body.dataset["seed"] == 3 and body.dataset["test_fraction"] == 0.3
    with pytest.raises(HTTPException) as e:
        _save_request(USER, {"run_id": run_id, "layer": 0, "pooling": "mean", "method": "logreg", "name": ""})
    assert e.value.status_code == 400, "the embeddings can't be saved"
    with pytest.raises(HTTPException) as e:
        _save_request(SimpleNamespace(id=8), {"run_id": run_id, "layer": 1, "pooling": "mean", "method": "logreg", "name": ""})
    assert e.value.status_code == 404, "another user's sweep"


# Geometry: comparing and steering kept runs

def _geometry_run(values: list[str], structure: str = "cyclic", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "model": "nano", "num_layers": 2, "read": "value_last", "chat_template": False, "seed": 0,
        "concept": {"builtin_id": None, "name": "Seasons", "structure": structure, "values": values,
                    "templates": ["It is {x}.", "We met in {x}.", "Back in {x}.", "During {x} it rained."],
                    "eval_templates": ["After {x} comes"], "positions": None, "groups": None, "helix_periods": []},
        "means": _wire(rng.normal(size=(3, len(values), 4))),
    }


VALUES = ["spring", "summer", "autumn", "winter", "monsoon"]


def test_two_kept_runs_are_compared():
    from services.assistant_tools import compare_geometry

    a = lab_runs.keep(USER.id, "geometry", _geometry_run(VALUES))
    ctx = ToolContext(USER.id)
    out = asyncio.run(compare_geometry(None, USER, ctx, {"run_id": a, "layer": 2, "other_run_id": a, "dims": 0}))
    assert not out.error
    assert out.content["best_aligned"]["layer"] == 2 and out.content["best_aligned"]["r2"] == pytest.approx(1.0, abs=1e-3)
    assert len(out.content["by_layer"]) == 3
    assert out.result["page"] == "geometry" and out.result["data"]["other_run_id"] == a
    other = lab_runs.keep(USER.id, "geometry", _geometry_run(["a", "b", "c", "d", "e"], seed=1))
    assert asyncio.run(compare_geometry(None, USER, ctx, {"run_id": a, "layer": 1, "other_run_id": other, "dims": 0})).error
    missing = asyncio.run(compare_geometry(None, USER, ctx, {"run_id": "geometry-0", "layer": 1, "other_run_id": a, "dims": 0}))
    assert missing.error and "No kept run" in missing.content["error"]


def test_a_kept_runs_concept_is_steered():
    from services.assistant_runs import _geometry_steer_request

    run_id = lab_runs.keep(USER.id, "geometry", _geometry_run(VALUES))
    args = {"run_id": run_id, "layer": 1, "dims": 3, "source": "Summer", "target": "winter", "steps": 8, "eval_template": ""}
    _, _, body = _geometry_steer_request(USER, args)
    assert (body.source, body.target, body.position, body.steps) == (1, 3, 1, 8)
    assert body.eval_template == "After {x} comes", "the concept's first eval prompt by default"
    assert body.concept.custom is not None and body.concept.custom.values == VALUES
    with pytest.raises(HTTPException):
        _geometry_steer_request(USER, {**args, "source": "never"})
    categorical = lab_runs.keep(USER.id, "geometry", _geometry_run(VALUES, "categorical"))
    with pytest.raises(HTTPException) as e:
        _geometry_steer_request(USER, {**args, "run_id": categorical})
    assert "categorical" in e.value.detail


def test_custom_concepts_are_checked_before_pricing():
    from api.geometry import _concept
    from services.assistant_runs import _concept_spec

    custom = {"name": "Seasons", "structure": "cyclic", "values": VALUES, "groups": [],
              "templates": ["It is {x}.", "We met in {x}.", "Back in {x}.", "During {x} it rained."], "eval_templates": []}
    spec = _concept_spec({"concept": "", "custom": custom})
    assert spec.custom.values == VALUES and spec.custom.groups is None
    concept, builtin = _concept(spec)
    assert builtin is None and concept["values"] == VALUES
    with pytest.raises(HTTPException):
        _concept(_concept_spec({"concept": "", "custom": {**custom, "values": VALUES[:4]}}))
    with pytest.raises(HTTPException):
        _concept_spec({"concept": "", "custom": {**custom, "groups": ["a"]}})
    with pytest.raises(HTTPException):
        _concept_spec({"concept": "days", "custom": custom})


def test_atlas_corpora_match():
    from core.assistant.tools import ATLAS_CORPORA
    from core.geometry.atlas import CORPORA

    assert set(ATLAS_CORPORA) == set(CORPORA)


# Estimates: a run's own quote

def test_estimate_prices_a_run_from_its_input(monkeypatch):
    import json

    from services import assistant_runs
    from services.assistant_tools import estimate

    seen = {}

    async def fake_quote(name, ctx, args):
        seen.update(name=name, args=args)
        return RunQuote("Concept geometry", [["Model", "Ozera Nano"]], 0.0123, hold=0.05, note="n")

    monkeypatch.setattr(assistant_runs, "quote", fake_quote)
    ctx = ToolContext(USER.id)
    good = {"model": "nano", "concept": "", "custom": {"name": "S", "structure": "cyclic", "values": VALUES, "groups": [],
                                                      "templates": ["{x}"] * 4, "eval_templates": []},
            "read": "value_last", "chat_template": False, "seed": 0}
    out = asyncio.run(estimate(None, USER, ctx, {"action": "run_geometry", "input": json.dumps(good)}))
    assert not out.error and seen["name"] == "run_geometry" and seen["args"] == good
    assert out.content["estimate_usd"] == 0.0123 and out.content["hold_usd"] == 0.05 and out.content["details"] == {"Model": "Ozera Nano"}
    assert asyncio.run(estimate(None, USER, ctx, {"action": "run_geometry", "input": "{"})).error
    missing = asyncio.run(estimate(None, USER, ctx, {"action": "run_geometry", "input": json.dumps({"model": "nano"})}))
    assert missing.error and missing.content["error"].startswith("input is missing")
    assert asyncio.run(estimate(None, USER, ctx, {"action": "list_models", "input": "{}"})).error

    async def failing_quote(name, ctx, args):
        return ToolOutcome({"error": "Unknown built-in concept"}, error=True)

    monkeypatch.setattr(assistant_runs, "quote", failing_quote)
    failed = asyncio.run(estimate(None, USER, ctx, {"action": "run_geometry", "input": json.dumps(good)}))
    assert failed.error and failed.chip == "Estimate · run geometry"


# The Probe Lab's runs end to end: quoted, run (on a fake GPU worker) and charged, on a database

class _Remote:
    def __init__(self, fn):
        self.aio = fn


class _Method:
    def __init__(self, fn):
        self.remote = _Remote(fn)


class _FakeWorker:
    """The inference worker's probe methods, returning results of the shapes the real ones do."""

    def __init__(self):
        self.score_probe = _Method(self._score)
        self.steer_probe = _Method(self._steer)
        self.ablate_probe = _Method(self._ablate)
        self.generalize_probes = _Method(self._generalize)

    async def _score(self, texts, **kwargs):
        return [{"tokens": ["a", " b", " c"], "scores": [0.2, -1.5, 0.9], "span": [0, 3], "score": 0.4} for _ in texts]

    async def _steer(self, alphas, ablate, **kwargs):
        def gen(kind, alpha):
            return {"kind": kind, "alpha": alpha, "text": f"{kind} {alpha}", "generated_tokens": 5, "perplexity": 12.0,
                    "probe_score": alpha or 0.0, "scored": None, "first_divergence": None if kind == "baseline" else 1,
                    "tokens_changed": 0 if kind == "baseline" else 3}
        generations = [gen("baseline", 0.0)] + [gen("steer", a) for a in alphas] + ([gen("ablate", None)] if ablate else [])
        return {"prompt_tokens": 3, "generations": generations}

    async def _ablate(self, texts, **kwargs):
        curve = [0.5, 0.7, 0.9, 0.8, 0.8, 0.7, 0.7]
        return {"num_layers": 6, "hidden_dim": 192, "total_tokens": 10 * len(texts), "compared_tokens": 8 * len(texts),
                "curves": {k: curve for k in ("clean", "probe_frozen", "probe_retrained", "control_frozen", "control_retrained")},
                "behaviour": {"probe": {"kl": 0.2, "top1_agreement": 0.8, "energy_fraction": 0.05},
                              "control": {"kl": 0.01, "top1_agreement": 0.99, "energy_fraction": 0.05, "kl_range": [0.0, 0.02]}}}

    async def _generalize(self, datasets, transfer_model_id, **kwargs):
        n = len(datasets) * (2 if transfer_model_id else 1)
        matrix = [[[0.5 + 0.05 * p if i == j else 0.55 for p in range(7)] for j in range(n)] for i in range(n)]
        return {"num_layers": 6, "hidden_dim": 192, "auroc": {"logreg": matrix, "diff_means": matrix},
                "acc": {"logreg": matrix, "diff_means": matrix}, "total_tokens": 1000}


@pytest.fixture()
def lab_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.probes
    from models.database import Base, CreditBalance, User
    from services import assistant_runs, credit_service

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(assistant_runs, "SessionLocal", Session)
    # SQLite has no row locks (Postgres's FOR UPDATE and lock_timeout)
    monkeypatch.setattr(credit_service, "get_credit_balance_for_update",
                        lambda db, user_id: db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first())
    monkeypatch.setattr(api.probes, "_get_inference_worker", lambda model_id: _FakeWorker)
    db = Session()
    user = User(email="labs@example.com", is_active=True)
    db.add(user)
    db.commit()
    db.add(CreditBalance(user_id=user.id, balance_usd=10, reserved_usd=0))
    db.commit()
    yield user.id
    db.close()


def _quote_and_run(name: str, user_id: int, args: dict) -> tuple[RunQuote, ToolOutcome]:
    from services import assistant_runs

    ctx = ToolContext(user_id)
    quoted = asyncio.run(assistant_runs.quote(name, ctx, args))
    assert isinstance(quoted, RunQuote), getattr(quoted, "content", quoted)
    outcome = asyncio.run(assistant_runs.run(name, ctx, args))
    assert not outcome.error, outcome.content
    return quoted, outcome


def test_probe_runs_end_to_end(lab_db):
    user_id = lab_db
    run, _, _ = _sweep(positions=7, d=192)
    run["model"], run["num_layers"] = "nano", 6
    # The class means apart along the probe's direction (a positive class gap, so it can steer)
    methods = run["poolings"]["mean"]["methods"]
    methods["diff_means"]["weights"] = methods["logreg"]["weights"]
    run_id = lab_runs.keep(user_id, "sweep", run, test_fraction=0.2)

    quoted, saved = _quote_and_run("save_probe", user_id, {"run_id": run_id, "layer": 3, "pooling": "mean", "method": "logreg", "name": "Mine"})
    assert quoted.estimate == 0.0 and saved.charged == 0.0
    probe_id = saved.content["probe_id"]
    assert saved.content["layer"] == 2 and saved.result["data"]["probe"]["id"] == probe_id

    _, scored = _quote_and_run("score_probe", user_id, {"probe_id": probe_id, "texts": ["It is true.", "It is false."]})
    assert scored.charged > 0 and scored.content["texts"][0]["reads_as"] == "true"
    assert scored.content["texts"][0]["strongest_tokens"][0]["token"] == " b"
    assert scored.result["data"]["labels"] == ["false", "true"]

    args = {"probe_id": probe_id, "prompt": "The sky is", "alphas": [-2, 0, 2], "ablate": True, "generated_only": False,
            "max_tokens": 10, "temperature": 0}
    quoted, steered = _quote_and_run("steer_probe", user_id, args)
    assert "+2" in dict(quoted.details)["Strengths"] and steered.charged > 0
    assert [g["kind"] for g in steered.content["generations"]] == ["steer", "baseline", "steer", "ablate"]

    _, ablated = _quote_and_run("ablate_probe", user_id, {"probe_id": probe_id, "dataset": "", "rows": [], "label_names": [],
                                                          "dataset_name": "", "test_fraction": 0, "seed": -1})
    from core.probes.datasets import builtin_datasets

    # The probe's own built-in dataset
    assert ablated.content["dataset"] == builtin_datasets()["truth"]["name"] and ablated.content["behaviour"]["probe"]["kl"] == 0.2
    assert [row["position"] for row in ablated.content["test_auroc"]] == [2, 3], "the probe's position and the best clean one"

    datasets = [{"dataset": d, "part": "main", "rows": [], "label_names": [], "name": ""} for d in ("truth", "negation")]
    _, matrix = _quote_and_run("generalize_probes", user_id, {"model": "nano", "datasets": datasets, "transfer": False,
                                                              "pooling": "mean", "chat_template": False, "test_fraction": 0.3, "seed": 0})
    logreg = matrix.content["by_method"]["logreg"]
    assert logreg["position"] == 6 and logreg["auroc_train_by_test"][0] == [0.8, 0.55]


class _FakeGeometryWorker:
    """The inference worker's geometry steering and SAE methods, with results of the real ones' shapes."""

    def __init__(self):
        self.steer_manifold = _Method(self._steer)
        self.sae_geometry = _Method(self._sae)

    async def _steer(self, concept, steps, source, target_value, **kwargs):
        k, n = len(concept["values"]), steps + 1
        path = [source + (target_value - source) * i / steps for i in range(n)]
        nearest = [round(p) for p in path]

        def method():
            return {"probs": [[0.1] * k] * n, "mass": [0.5] * n, "hellinger": [0.1] * n, "nearest": nearest,
                    "to_target": [0.9 - 0.8 * i / steps for i in range(n)], "top": [[{"token": "x", "prob": 0.3}]] * n,
                    "edit_size": [0.1 * i for i in range(n)]}

        return {
            "num_layers": 6, "prompt": "After summer comes", "edited_tokens": ["After"], "prefix": [], "readable": list(range(k)),
            "tokens": concept["values"], "path": path, "explained": 0.8, "eval_decoding": 0.6,
            "reference": {"probs": [[0.2] * k] * k, "mass": [0.6] * k, "top": [[{"token": "x", "prob": 0.3}]] * k, "nearest_other": [0.4] * k},
            "methods": {"manifold": method(), "linear": method()},
            "plot": {"means": [[0.0, 0.0]] * k, "eval": [[0.0, 0.0]] * k, "curve": [[0.0, 0.0]] * 10, "manifold": [[0.0, 0.0]] * n, "linear": [[0.0, 0.0]] * n},
            "total_tokens": 200, "concept_tokens": 150, "eval_tokens": 50,
        }

    async def _sae(self, concept, **kwargs):
        k = len(concept["values"])
        fit = {"dims": 3, "variance": 0.9, "ceiling": 0.95, "order": [4, 2, 9], "r2": [0.5, 0.8, 0.92], "cos": [0.6, 0.5, 0.4],
               "random": {"mean": [0.1, 0.2, 0.3], "max": [0.2, 0.3, 0.4]}, "cosine_order": [4, 9, 2], "cosine_r2": [0.5, 0.7, 0.9],
               "n90": 3, "spans": 3, "compact": 4, "fields": {"width": 0.3, "coverage": 1.0, "contiguous": 0.8, "coactive": 1.5},
               "verdict": "capture"}
        return {
            "num_layers": 6, "hidden_dim": 192, "sae": {"tokens": 40, "width": 1536, "l0": 30.0, "fvu": 0.2},
            "active_features": 300, "varying_features": 120,
            "features": {"ids": [4, 2, 9], "tuning": _wire(np.zeros((3, k))), "active": _wire(np.zeros((3, k))),
                         "top_example": [0, 1, 2], "top_activation": [1.0, 0.5, 0.2]},
            "fits": [fit, {**fit, "dims": None}], "total_tokens": 200,
        }


def test_geometry_runs_end_to_end(lab_db, monkeypatch):
    import api.geometry

    monkeypatch.setattr(api.geometry, "_get_inference_worker", lambda model_id: _FakeGeometryWorker)
    user_id = lab_db
    run_id = lab_runs.keep(user_id, "geometry", {**_geometry_run(VALUES), "num_layers": 6})

    args = {"run_id": run_id, "layer": 3, "dims": 3, "source": "summer", "target": "winter", "steps": 4, "eval_template": ""}
    quoted, steered = _quote_and_run("steer_geometry", user_id, args)
    assert dict(quoted.details)["Walk"] == "summer → winter, 4 steps" and steered.charged > 0
    assert steered.content["manifold"]["nearest_value_by_step"] == ["summer", "autumn", "autumn", "autumn", "winter"]
    assert steered.content["manifold"]["reaches_target_at_step"] == 4
    assert steered.result["data"]["run_id"] == run_id

    quoted, read = _quote_and_run("geometry_sae", user_id, {"run_id": run_id, "sae": "ozera:nano:2"})
    assert read.content["fits"][0]["verdict"] == "capture" and read.content["fits"][1]["dims"] == "all"
    assert read.content["density"] == pytest.approx(30 / 1536, abs=1e-4)
    assert read.content["fits"][0]["top_features"][0]["link"].startswith("ozera:feature?sae=ozera:nano:2&n=4")


def test_attention_runs_capture_or_reuse(lab_db, monkeypatch):
    """compare_attention captures the side given a prompt and reuses the side given a capture; mine_patterns likewise."""
    import api.analysis
    import api.patching
    import core.patching

    captures = {"cap-1": SimpleNamespace(id="cap-1", model_id="nano", prompt="The cat sat", tokens=[1, 2, 3], num_layers=6, activations={})}
    engine = SimpleNamespace(get_captured_activations=lambda activation_id, user_id: captures.get(activation_id))
    monkeypatch.setattr(core.patching, "get_patching_engine", lambda: engine)
    made = []

    async def capture(request, current_user, db):
        made.append(request.prompt)
        return SimpleNamespace(activation_id=f"new-{len(made)}")

    def compare(request, current_user, db):
        return api.analysis.CompareAttentionResponse(
            prompt1="a", prompt2="b", activation_id1=request.activation_id_1, activation_id2=request.activation_id_2,
            layer_similarities=[{"layer": 0, "mean_similarity": 0.9, "min_similarity": 0.5, "max_similarity": 1.0}],
            head_differences=[{"layer": 0, "head": 1, "similarity": 0.4, "difference": 0.6}], common_patterns=["x"], divergent_patterns=[],
        )

    def mine(request, current_user, db):
        return api.analysis.MinePatternResponse(
            model_id="nano", activation_ids=[request.activation_id], head_importance=[], cross_layer_patterns=[],
            classification_summary={"induction": 2, "mixed": 0}, top_heads=[[1, 2, 0.8]], circuit_candidates=[],
        )

    monkeypatch.setattr(api.patching, "capture_activations", capture)
    monkeypatch.setattr(api.analysis, "compare_attention", compare)
    monkeypatch.setattr(api.analysis, "mine_patterns", mine)
    user_id = lab_db

    args = {"model": "nano", "prompt_1": "", "activation_id_1": "cap-1", "prompt_2": "The dog ran", "activation_id_2": ""}
    quoted, compared = _quote_and_run("compare_attention", user_id, args)
    assert "captures" in dict(quoted.details)["Charges"], "one side is captured, and charged for"
    assert made == ["The dog ran"] and compared.content["activation_ids"] == ["cap-1", "new-1"]
    assert compared.result["data"]["activation_ids"] == ["cap-1", "new-1"]

    quoted, mined = _quote_and_run("mine_patterns", user_id, {"model": "", "prompt": "", "activation_id": "cap-1"})
    assert "Charges" not in dict(quoted.details), "a reused capture isn't captured again"
    assert mined.content["counts"] == {"induction": 2} and mined.content["top_heads"][0] == {"layer": 1, "head": 2, "importance": 0.8}
