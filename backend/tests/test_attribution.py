"""
Attribution (core.attribution, api/attribution.py; INSTRUMENTS_PLAN.md's Phase 2): prompt pairs
and their checks, attribution patching against finite differences, exact patching and
ablation against the patching engine's own hooks, the API's routes over a worker run on the
CPU, and the assistant's three tools. Run from backend/: python -m pytest tests/test_attribution.py

The open-weight architectures (Llama, Qwen3, Gemma 3) are tested on small random models when
transformers is installed (it's only in the Modal images).
"""

import asyncio
import dataclasses
import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import attribution  # noqa: E402
from core.attribution import budget  # noqa: E402
from core.attribution.ablation import random_sets  # noqa: E402
from core.attribution.metric import spearman  # noqa: E402
from core.attribution.pairs import OzeraVocabulary, classes, pairs_key, prepare  # noqa: E402
from core.attribution.tasks import INDUCTION_LENGTH, TASKS, build_pairs, normalize_rows, parse_upload, task_summary  # noqa: E402
from core.patching.hooks import create_unit_intervention_hook  # noqa: E402
from core.probes.datasets import DatasetParseError  # noqa: E402
from core.probes.runner import OzeraTarget  # noqa: E402
from core.transformer.config import TransformerConfig  # noqa: E402
from core.transformer.model_torch import TransformerLM  # noqa: E402


class _WordTokenizer:
    """Words (with their leading space) as tokens, numbered as they're first seen."""

    vocab_size = 400
    eos_token_id = 399

    def __init__(self):
        self.ids, self.words = {}, []

    def encode(self, text):
        out = []
        for piece in re.findall(r" ?[^ ]+", text):
            if piece not in self.ids:
                self.ids[piece] = len(self.words)
                self.words.append(piece)
            out.append(self.ids[piece])
        return out

    def decode(self, ids):
        return "".join(self.words[i] if i < len(self.words) else f" w{i}" for i in ids)


def _ozera_target() -> OzeraTarget:
    """A small Ozera model whose norms and biases aren't at their initial values."""
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=400, d_model=48, num_layers=3, num_heads=4, d_ff=96, max_seq_len=64, initializer_range=0.3,
    )
    model = TransformerLM(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0, 0.3)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(0, 0.2)
    return OzeraTarget(model, config, _WordTokenizer())


def _capital_rows(target, count: int = 6, answers: bool = True) -> list[dict]:
    """Pairs of one template that differ at one position (3), oriented so the model's clean metric is the higher."""
    countries = ["France", "Spain", "Italy", "Japan", "Peru", "Cuba", "Kenya", "Egypt"]
    rows = []
    for i in range(count):
        a, b = countries[i], countries[(i + 3) % len(countries)]
        row = {"clean": f"The capital of {a} is", "corrupted": f"The capital of {b} is", "correct": None, "incorrect": None, "group": None}
        if answers:
            row.update(correct=f" city{i}", incorrect=f" town{i}")
        rows.append(row)
    if answers:
        # A random model has no opinion: point each pair the way it happens to lean
        run = attribution.sweep(target, "l4", "tiny", {"rows": rows})
        for row, pair in zip(rows, run["pairs"]):
            if pair["clean_value"] < pair["corrupted_value"]:
                row["correct"], row["incorrect"] = row["incorrect"], row["correct"]
    return rows


@pytest.fixture(scope="module")
def ozera():
    target = _ozera_target()
    rows = _capital_rows(target)
    return target, {"rows": rows}, attribution.sweep(target, "l4", "tiny", {"rows": rows})


# Tasks and rows

def test_built_in_tasks_build_the_same_pairs_every_time():
    for task_id, task in TASKS.items():
        summary = task_summary(task)
        if task["kind"] == "tokens":
            assert summary["examples"] == []
            with pytest.raises(budget.AttributionInputError):
                build_pairs(task_id, 5, 0)
            continue
        pairs = build_pairs(task_id, 40, 0)
        assert len(pairs) == 40 and pairs == build_pairs(task_id, 40, 0)
        assert pairs[:10] == build_pairs(task_id, 10, 0), "fewer pairs are the first of more"
        assert pairs != build_pairs(task_id, 40, 1)
        assert len({(p["clean"], p["corrupted"]) for p in pairs}) == 40
        for pair in pairs:
            assert pair["clean"] != pair["corrupted"] and pair["correct"] != pair["incorrect"]
            assert pair["correct"].startswith(" ") and not pair["clean"].endswith(" ")
            assert len(pair["clean"].split()) == len(pair["corrupted"].split()), "a pair's prompts have as many words"
        assert len(summary["examples"]) == 3
    with pytest.raises(budget.AttributionInputError):
        build_pairs("ioi", budget.MAX_PAIRS + 1, 0)


def test_rows_are_normalized():
    rows = normalize_rows([{"clean": "A b ", "corrupted": "A c", "correct": "x", "incorrect": " y", "group": 3}])
    assert rows == [{"clean": "A b", "corrupted": "A c", "correct": " x", "incorrect": " y", "group": "3"}]
    assert normalize_rows([{"clean": "a", "corrupted": "b"}])[0]["correct"] is None
    assert normalize_rows([{"clean": "a", "corrupted": "b", "correct": "'s", "incorrect": "."}])[0]["correct"] == "'s", "only words get a space"
    for bad in (
        [], [{"clean": "a", "corrupted": "a"}], [{"clean": "a", "corrupted": ""}],
        [{"clean": "a", "corrupted": "b", "correct": "x"}], [{"clean": "a", "corrupted": "b", "correct": "x", "incorrect": "x"}],
        [{"clean": "a" * (budget.MAX_PROMPT_CHARS + 1), "corrupted": "b"}],
    ):
        with pytest.raises(budget.AttributionInputError):
            normalize_rows(bad)


def test_uploads_are_parsed():
    csv = "clean,corrupted,correct,incorrect\nThe cat,The dog,meows,barks\n,skipped,,\nA b,A c,x,y\n"
    parsed = parse_upload("pairs.csv", csv.encode())
    assert len(parsed["rows"]) == 2 and parsed["has_answers"] and parsed["rows"][0]["correct"] == " meows"
    assert any("Skipped 1" in w for w in parsed["warnings"]) and any("leading space" in w for w in parsed["warnings"])
    jsonl = '{"clean": "a b", "corrupted": "a c"}\n{"clean": "d e", "corrupted": "d f", "correct": " x", "incorrect": " y"}\n'
    parsed = parse_upload("pairs.jsonl", jsonl.encode())
    assert not parsed["has_answers"] and any("KL" in w for w in parsed["warnings"])
    with pytest.raises(DatasetParseError):
        parse_upload("pairs.csv", b"text,label\na,1\n")
    with pytest.raises(DatasetParseError):
        parse_upload("pairs.csv", b"clean,corrupted\nsame,same\n")


def test_pairs_are_checked_and_grouped():
    target = _ozera_target()
    rows = normalize_rows([
        {"clean": "The cat sat", "corrupted": "The dog sat", "correct": "down", "incorrect": "up", "group": "a"},
        {"clean": "The cat sat", "corrupted": "The big dog sat", "correct": "down", "incorrect": "up"},
        {"clean": "The cow sat", "corrupted": "The dog sat", "correct": "down", "incorrect": "down now"},
        {"clean": "The pig sat", "corrupted": "The hen sat", "correct": "down", "incorrect": "up", "group": "a"},
        {"clean": "A pig sat here", "corrupted": "A hen sat here", "group": "a"},
    ])
    pairs, dropped = prepare(target, {"rows": rows})
    assert [p.source for p in pairs] == [0, 3, 4]
    assert [d["index"] for d in dropped] == [1, 2]
    assert "3 and 4 tokens" in dropped[0]["reason"] and "same token" in dropped[1]["reason"]
    assert [p.cls for p in pairs] == [0, 0, 1], "a class is one group and one length"
    groups = classes(target, pairs)
    assert groups[0]["pairs"] == [0, 1] and groups[0]["labels"] == ["The", None, " sat"]
    assert groups[1]["labels"] == ["A", " pig", " sat", " here"]
    assert pairs[2].correct_id is None and pairs[0].correct_id != pairs[0].incorrect_id
    assert pairs_key(pairs) == pairs_key(prepare(target, {"rows": rows})[0]) != pairs_key(pairs[:2])

    with pytest.raises(budget.AttributionInputError) as e:
        prepare(target, {"rows": rows[1:2]})
    assert "None of the pairs can be used" in str(e.value)


def test_induction_pairs_repeat_random_tokens():
    target = _ozera_target()
    for word in ("alpha", "beta", "gamma", "delta"):
        target.tokenizer.encode(f" {word}")
    pairs, dropped = prepare(target, {"induction": {"n": 4, "seed": 1}})
    k = INDUCTION_LENGTH
    assert len(pairs) == 4 and not dropped and {p.cls for p in pairs} == {0}
    for pair in pairs:
        first, again = pair.clean_ids[:k], pair.clean_ids[k:]
        assert again == first[:-1] and pair.correct_id == first[-1]
        assert pair.corrupted_ids[k:] == again and not set(pair.corrupted_ids[:k]) & set(first)
        assert pair.incorrect_id == pair.corrupted_ids[k - 1] and len(set(first)) == k
    assert [p.clean_ids for p in pairs] == [p.clean_ids for p in prepare(target, {"induction": {"n": 4, "seed": 1}})[0]]
    # The same pairs from the tokenizer alone, as the backend checks an Ozera model's
    vocabulary = OzeraVocabulary(target.tokenizer, 64, target.model.config.vocab_size)
    assert [p.clean_ids for p in prepare(vocabulary, {"induction": {"n": 4, "seed": 1}})[0]] == [p.clean_ids for p in pairs]


# The sweep

def _logit_diff(target, ids, pair, hooks=()):
    """The model's own logit difference at the last position, with forward pre-hooks [(module, hook)] in place."""
    handles = [module.register_forward_pre_hook(hook) for module, hook in hooks]
    try:
        with torch.no_grad():
            logits, _, _ = target.model(torch.tensor([ids]))
    finally:
        for handle in handles:
            handle.remove()
    return float(logits[0, -1, pair.correct_id] - logits[0, -1, pair.incorrect_id])


def _projection_input(target, ids, layer):
    """What a layer's attention output projection is given on a prompt: [1, seq, heads * head_dim]."""
    seen = {}
    module = target.head_projection(layer)
    handle = module.register_forward_pre_hook(lambda m, args: seen.update(x=args[0].detach()))
    with torch.no_grad():
        target.model(torch.tensor([ids]))
    handle.remove()
    return seen["x"]


def test_sweep_reports_the_models_own_metric(ozera):
    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    assert run["metric"] == "logit_diff" and run["normalized"] and run["num_pairs"] == len(pairs)
    for pair, result in zip(pairs, run["pairs"]):
        assert result["clean_value"] == pytest.approx(_logit_diff(target, pair.clean_ids, pair), abs=1e-4)
        assert result["corrupted_value"] == pytest.approx(_logit_diff(target, pair.corrupted_ids, pair), abs=1e-4)
    assert run["gap"] == pytest.approx(run["clean_mean"] - run["corrupted_mean"], abs=1e-5) and run["solved"] == 1.0
    layers, heads = run["num_layers"], run["num_heads"]
    assert run["heads"].shape == (layers, heads) and run["heads_pairs"].shape == (len(pairs), layers, heads)
    assert run["layers"].shape == (2, layers) and run["dla"].shape == (layers, heads)
    assert [p.shape for p in run["position_pairs"]] == [(3, layers, pair.tokens) for pair in pairs]
    assert torch.allclose(run["position_classes"][0], torch.stack(run["position_pairs"]).mean(dim=0))
    assert run["attention"]["pairs"] == list(range(len(pairs)))
    pattern = run["attention"]["patterns"][0]
    assert pattern.shape == (layers, heads, pairs[0].tokens, pairs[0].tokens) and pattern.dtype == torch.uint8
    assert bool((pattern.float().sum(dim=-1) / 255 - 1).abs().max() < 0.03), "each token's attention sums to 1"
    assert bool((pattern.triu(1) == 0).all()), "attention is causal"
    assert not any(p.requires_grad for p in target.model.parameters()), "the model is frozen for the gradients"


def test_estimates_are_the_metrics_first_order_change(ozera):
    """An estimate is the metric's slope along the line from the corrupted activation to the clean one."""
    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    pair, epsilon = pairs[0], 1e-3
    base = _logit_diff(target, pair.corrupted_ids, pair)
    head_dim = target.model.config.d_k
    for layer in range(run["num_layers"]):
        clean = _projection_input(target, pair.clean_ids, layer)
        for head in range(run["num_heads"]):
            hook = create_unit_intervention_hook([head], head_dim, "patch", source_input=clean, blend_factor=epsilon)
            moved = _logit_diff(target, pair.corrupted_ids, pair, [(target.head_projection(layer), hook)])
            slope = (moved - base) / epsilon
            estimate = float(run["heads_pairs"][0, layer, head]) * run["gap"]
            assert estimate == pytest.approx(slope, rel=0.05, abs=5e-3), (layer, head)


def test_heads_add_up_to_their_layers_attention(ozera):
    _, _, run = ozera
    assert torch.allclose(run["heads"].sum(dim=-1), run["layers"][0], atol=1e-4)
    # And each position's attention and MLP effects to their layer's
    totals = torch.stack([p[1:].sum(dim=-1) for p in run["position_pairs"]]).mean(dim=0)
    assert torch.allclose(totals, run["layers"], atol=1e-5)


def test_direct_effects_are_what_heads_write_to_the_logit_difference(ozera):
    """Heads, the projections' biases, MLPs and embeddings write the whole logit difference: check the heads' part against core.lens."""
    from core.lens import Capture, LensModel, attribute

    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    lens = LensModel.for_ozera(target.model, target.tokenizer)
    expected = torch.zeros_like(run["dla"])
    for pair in pairs:
        ids = torch.tensor([pair.clean_ids])
        with torch.no_grad():
            logits, _, activations = target.model.forward(ids, return_attention=True, capture_activations=True)
        capture = Capture.from_ozera(target.model, activations, logits, pair.clean_ids)
        correct = attribute(lens, capture, torch.full((pair.tokens,), pair.correct_id))["heads"][-1]
        incorrect = attribute(lens, capture, torch.full((pair.tokens,), pair.incorrect_id))["heads"][-1]
        expected += (correct - incorrect) / len(pairs)
    assert torch.allclose(run["dla"], expected, atol=1e-4)


# Exact patching

def test_patching_the_changed_token_restores_the_clean_run(ozera):
    target, spec, run = ozera
    cells = [{"kind": "resid", "layer": 0, "position": 3}, {"kind": "resid", "layer": 0, "position": 2}]
    result = attribution.verify(target, "l4", "tiny", spec, cells, expected_key=run["pairs_key"])
    changed, same = result["cells"]
    assert changed["effect"] == pytest.approx(1.0, abs=1e-4) and changed["n"] == run["num_pairs"] and changed["cls"] == 0
    assert same["effect"] == pytest.approx(0.0, abs=1e-5), "a token both prompts share changes nothing"
    assert result["gap"] == pytest.approx(run["gap"], abs=1e-5)
    # On one pair, in that pair's own share of the gap
    one = attribution.verify(target, "l4", "tiny", spec, [{"kind": "resid", "layer": 0, "position": 3, "pair": 1}])["cells"][0]
    own = (run["pairs"][1]["clean_value"] - run["pairs"][1]["corrupted_value"]) / run["gap"]
    assert one["effect"] == pytest.approx(own, abs=1e-4) and one["n"] == 1 and one["se"] == 0


def test_exact_patching_matches_the_patching_engines_hooks(ozera):
    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    layer, head, head_dim = 1, 2, target.model.config.d_k
    cells = [{"kind": "head", "layer": layer, "head": head}, {"kind": "head", "layer": layer, "head": head, "position": 3},
             {"kind": "attn", "layer": layer}, {"kind": "mlp", "layer": 0, "position": 4}]
    result = attribution.verify(target, "l4", "tiny", spec, cells)

    def expected(positions):
        effects = []
        for pair in pairs:
            clean = _projection_input(target, pair.clean_ids, layer)
            hook = create_unit_intervention_hook([head], head_dim, "patch", source_input=clean, positions=positions)
            patched = _logit_diff(target, pair.corrupted_ids, pair, [(target.head_projection(layer), hook)])
            effects.append(patched - _logit_diff(target, pair.corrupted_ids, pair))
        return sum(effects) / len(effects) / run["gap"]

    assert result["cells"][0]["effect"] == pytest.approx(expected(None), abs=1e-4)
    assert result["cells"][1]["effect"] == pytest.approx(expected([3]), abs=1e-4)
    # Every head of a layer patched at once is its attention output patched (the projection is linear)
    every = [{"kind": "head", "layer": layer, "head": h} for h in range(run["num_heads"])]
    assert abs(result["cells"][2]["effect"]) > 1e-3
    assert all(c["n"] == len(pairs) for c in result["cells"]) and result["cells"][0]["se"] > 0
    assert len(attribution.verify(target, "l4", "tiny", spec, every)["cells"]) == run["num_heads"]


def test_cells_are_checked(ozera):
    target, spec, run = ozera
    for bad in (
        [], [{"kind": "resid", "layer": 0}], [{"kind": "head", "layer": 0}], [{"kind": "head", "layer": 9, "head": 0}],
        [{"kind": "mlp", "layer": 0, "head": 1}], [{"kind": "attn", "layer": 0, "position": 99}],
        [{"kind": "attn", "layer": 0, "pair": 99}], [{"kind": "neuron", "layer": 0}],
    ):
        with pytest.raises(budget.AttributionInputError):
            attribution.verify(target, "l4", "tiny", spec, bad)
    with pytest.raises(budget.AttributionInputError) as e:
        attribution.verify(target, "l4", "tiny", spec, [{"kind": "attn", "layer": 0}], expected_key="another")
    assert "run the sweep again" in str(e.value)
    # Pairs whose tokens don't line up: a position needs a pair or a class
    mixed = {"rows": spec["rows"] + normalize_rows([{"clean": "A b", "corrupted": "A c", "correct": "x", "incorrect": "y"}])}
    with pytest.raises(budget.AttributionInputError):
        attribution.verify(target, "l4", "tiny", mixed, [{"kind": "attn", "layer": 0, "position": 1}])
    assert attribution.verify(target, "l4", "tiny", mixed, [{"kind": "attn", "layer": 0, "position": 1, "cls": 1}])["cells"][0]["n"] == 1


# Ablation

def test_ablation_matches_the_patching_engines_hooks(ozera):
    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    layer, head, head_dim = 2, 1, target.model.config.d_k
    result = attribution.ablate(target, "l4", "tiny", spec, [{"kind": "head", "layer": layer, "head": head}], ["zero", "resample", "mean"])
    assert result["mean_over"] == "position" and list(result["methods"]) == ["zero", "mean", "resample"]

    def expected(method):
        values = []
        for pair in pairs:
            if method == "zero":
                hook = create_unit_intervention_hook([head], head_dim, "zero_ablate")
            else:
                hook = create_unit_intervention_hook([head], head_dim, "patch", source_input=_projection_input(target, pair.corrupted_ids, layer))
            ablated = _logit_diff(target, pair.clean_ids, pair, [(target.head_projection(layer), hook)])
            values.append(ablated - _logit_diff(target, pair.corrupted_ids, pair))
        return sum(values) / len(values) / run["gap"]

    assert result["methods"]["zero"]["value"] == pytest.approx(expected("zero"), abs=1e-4)
    assert result["methods"]["resample"]["value"] == pytest.approx(expected("resample"), abs=1e-4)
    assert len(result["methods"]["zero"]["random"]) == budget.RANDOM_SETS == len(result["random_sets"])
    # Nothing ablated is the clean run: every component resampled from itself would be 1
    assert result["clean_mean"] == pytest.approx(run["clean_mean"], abs=1e-5)


def test_mean_ablation_averages_over_the_pair_set(ozera):
    target, spec, run = ozera
    pairs, _ = prepare(target, spec)
    layer = 1
    result = attribution.ablate(target, "l4", "tiny", spec, [{"kind": "attn", "layer": layer}], ["mean"])
    # The layer's attention output replaced by its mean over every prompt, position by position
    outputs = []
    for pair in pairs:
        for ids in (pair.clean_ids, pair.corrupted_ids):
            with torch.no_grad():
                _, _, activations = target.model.forward(torch.tensor([ids]), capture_activations=True)
            outputs.append(activations["layers"][layer]["attn_output"][0])
    mean = torch.stack(outputs).mean(dim=0)
    values = []
    for pair in pairs:
        handle = target.attention_branch(layer).register_forward_hook(lambda m, a, out: (mean[None], out[1]))
        try:
            ablated = _logit_diff(target, pair.clean_ids, pair)
        finally:
            handle.remove()
        values.append(ablated - _logit_diff(target, pair.corrupted_ids, pair))
    assert result["methods"]["mean"]["value"] == pytest.approx(sum(values) / len(values) / run["gap"], abs=1e-4)


def test_random_sets_match_the_chosen_components():
    chosen = [{"kind": "head", "layer": 0, "head": 1}, {"kind": "head", "layer": 2, "head": 0}, {"kind": "mlp", "layer": 1, "head": None}]
    sets = random_sets(chosen, 3, 4, seed=0)
    assert len(sets) == budget.RANDOM_SETS and sets == random_sets(chosen, 3, 4, seed=0) != random_sets(chosen, 3, 4, seed=1)
    for components in sets:
        assert sorted(c["kind"] for c in components) == ["head", "head", "mlp"]
        assert not [c for c in components if c in chosen]
        assert len({(c["kind"], c["layer"], c["head"]) for c in components}) == 3
    # Not enough other MLP layers to draw from: no random sets
    assert random_sets([{"kind": "mlp", "layer": layer, "head": None} for layer in range(2)], 3, 4, seed=0) == []


def test_components_are_checked(ozera):
    target, spec, _ = ozera
    for bad in ([], [{"kind": "head", "layer": 0}], [{"kind": "resid", "layer": 0}], [{"kind": "mlp", "layer": 5}]):
        with pytest.raises(budget.AttributionInputError):
            attribution.ablate(target, "l4", "tiny", spec, bad, ["zero"])
    with pytest.raises(budget.AttributionInputError):
        attribution.ablate(target, "l4", "tiny", spec, [{"kind": "mlp", "layer": 0}], [])


# The KL metric

def test_pairs_without_answers_are_measured_by_kl():
    target = _ozera_target()
    spec = {"rows": _capital_rows(target, answers=False)}
    run = attribution.sweep(target, "l4", "tiny", spec)
    assert run["metric"] == "kl" and run["dla"] is None
    assert run["clean_mean"] == pytest.approx(0.0, abs=1e-6) and run["corrupted_mean"] < 0 and run["gap"] > 0
    result = attribution.verify(target, "l4", "tiny", spec, [{"kind": "resid", "layer": 0, "position": 3}])
    assert result["cells"][0]["effect"] == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(budget.AttributionInputError):
        attribution.sweep(target, "l4", "tiny", spec, metric="logit_diff")
    assert attribution.sweep(target, "l4", "tiny", {"rows": _capital_rows(target, 3)}, metric="kl")["metric"] == "kl"


# Budget and statistics

def test_budget():
    assert budget.grad_tokens_per_batch(6, 192, 768, 4) == budget.TOKENS_PER_BATCH
    large = budget.grad_tokens_per_batch(36, 2560, 9728, 2)
    assert budget.MIN_GRAD_TOKENS <= large < 512
    sweep = budget.estimate_sweep_seconds("l4", 135_000_000, 40, 600)
    assert sweep < budget.estimate_sweep_seconds("l4", 135_000_000, 40, 6000) < budget.estimate_sweep_seconds("l4", 1_700_000_000, 40, 6000)
    assert budget.estimate_verify_seconds("l4", 135_000_000, 40, 600, 0) < budget.estimate_verify_seconds("l4", 135_000_000, 40, 600, 60_000)
    assert budget.estimate_ablate_seconds("l4", 135_000_000, 40, 600, ("zero",)) < budget.estimate_ablate_seconds("l4", 135_000_000, 40, 600, ("zero", "mean", "resample"))
    budget.check_run_size("l4", "nano", 60)
    with pytest.raises(budget.AttributionInputError):
        budget.check_run_size("l4", "qwen3-1.7b", 121)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 2, 2, 5]) == pytest.approx(0.8333, abs=1e-3)
    assert spearman([1, 1, 1], [1, 2, 3]) is None and spearman([1, 2], [1, 2]) is None


# Open-weight architectures

class _CharTokenizer:
    """A BOS, then a token per character."""

    pad_token_id = 1
    eos_token_id = 1
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return 120

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        return {"input_ids": [2] + [3 + (ord(c) % 90) for c in text], "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<bos>" if i == 2 else chr(i - 3 + 90 if i - 3 < 32 else i - 3) for i in ids)


def _open_source_target(family: str, dtype=torch.float32):
    transformers = pytest.importorskip("transformers")
    from core.open_source.gemma import GemmaLoader
    from core.open_source.qwen import QwenLoader
    from core.open_source.smollm import SmolLMLoader
    from core.probes.runner import OpenSourceTarget

    shape = dict(vocab_size=120, hidden_size=48, intermediate_size=96, num_hidden_layers=3, max_position_embeddings=128)
    if family == "llama":
        config = transformers.LlamaConfig(num_attention_heads=6, num_key_value_heads=2, **shape)
        model, loader = transformers.LlamaForCausalLM(config), SmolLMLoader("smollm-135m", device="cpu")
    elif family == "qwen":
        config = transformers.Qwen3Config(num_attention_heads=4, num_key_value_heads=2, head_dim=16, **shape)
        model, loader = transformers.Qwen3ForCausalLM(config), QwenLoader("qwen3-0.6b", device="cpu")
    else:
        config = transformers.Gemma3TextConfig(num_attention_heads=4, num_key_value_heads=1, head_dim=16, sliding_window=64, **shape)
        model, loader = transformers.Gemma3ForCausalLM(config), GemmaLoader("gemma-3-270m", device="cpu")

    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() > 1:
                parameter.mul_(4.0)
            else:
                parameter.add_(torch.randn_like(parameter) * 0.3)
    model.set_attn_implementation("eager")
    loader.model, loader.tokenizer = model.eval().to(dtype), _CharTokenizer()
    loader.config = dataclasses.replace(
        loader.config, num_layers=config.num_hidden_layers, num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads, hidden_dim=config.hidden_size, vocab_size=config.vocab_size,
        intermediate_dim=config.intermediate_size, is_instruct=False,
    )
    return OpenSourceTarget(loader)


def _letter_rows(target) -> dict:
    """Pairs that differ in one character (position 3, after the BOS), pointed the way the model leans."""
    rows = [
        {"clean": f"ab{a}de", "corrupted": f"ab{b}de", "correct": "x", "incorrect": "y", "group": None}
        for a, b in ("cq", "fr", "gs", "ht", "ju", "kv")
    ]
    run = attribution.sweep(target, "l4", "tiny", {"rows": rows})
    for row, pair in zip(rows, run["pairs"]):
        if pair["clean_value"] < pair["corrupted_value"]:
            row["correct"], row["incorrect"] = row["incorrect"], row["correct"]
    return {"rows": rows}


def _slope(target, pair, site, head, epsilon: float = 1e-3) -> float:
    """
    The logit difference's slope as a site of the corrupted run moves toward the clean run's
    (one head's part of it, or all of it), by finite differences.
    """
    from core.attribution.taps import Taps
    from core.patching.hooks import create_masked_replacement_hook

    final = ("resid", len(target.layers))
    mask = torch.ones(1, pair.tokens, dtype=torch.long)

    def read(taps, ids):
        taps.run(lambda: target.forward(torch.tensor([ids]), mask))
        logits = target.logits(taps.acts[final][:, -1]).double()
        return taps.acts[site], float(logits[0, pair.correct_id] - logits[0, pair.incorrect_id])

    with Taps(target, read=[site, final], replace_at=[site]) as taps, torch.no_grad():
        clean, _ = read(taps, pair.clean_ids)
        corrupted, base = read(taps, pair.corrupted_ids)
        dims = torch.ones(clean.shape[-1], dtype=torch.bool)
        if head is not None:
            width = clean.shape[-1] // target.num_heads
            dims = torch.zeros_like(dims)
            dims[head * width:(head + 1) * width] = True
        taps.replace = {site: create_masked_replacement_hook(corrupted + epsilon * (clean - corrupted), dims[None, None])}
        _, moved = read(taps, pair.corrupted_ids)
    return (moved - base) / epsilon


@pytest.mark.parametrize("family", ["llama", "qwen", "gemma"])
def test_open_weight_architectures(family):
    target = _open_source_target(family)
    spec = _letter_rows(target)
    # (transformers leaves hooks of its own on the model the first time attention weights are asked for)
    hooks = lambda: sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in target.loader.model.modules())  # noqa: E731
    before = hooks()
    run = attribution.sweep(target, "l4", "tiny", spec)
    layers, heads = run["num_layers"], run["num_heads"]
    assert run["normalized"] and run["heads"].shape == (layers, heads) and bool(torch.isfinite(run["heads"]).all())
    assert run["pairs"][0]["clean_tokens"][0] == "<bos>" and run["position_pairs"][0].shape == (3, layers, 6)
    assert run["attention"] is not None and run["attention"]["patterns"][0].shape == (layers, heads, 6, 6)
    if family != "gemma":
        # Gemma normalizes attention's output before adding it, so its heads don't add up exactly
        assert torch.allclose(run["heads"].sum(dim=-1), run["layers"][0], atol=1e-3)
        assert run["dla"] is not None

    # Estimates are the metric's slope from the corrupted activation toward the clean one, for
    # heads (through Gemma's norm of attention's output too) and for the stream, attention and MLP
    pairs, _ = prepare(target, spec)
    for layer in range(layers):
        for head in range(heads):
            estimate = float(run["heads_pairs"][0, layer, head]) * run["gap"]
            assert estimate == pytest.approx(_slope(target, pairs[0], ("heads", layer), head), rel=0.05, abs=0.02), (layer, head)
        for k, kind in enumerate(("resid", "attn", "mlp")):
            estimate = float(run["position_pairs"][0][k, layer].sum()) * run["gap"]
            assert estimate == pytest.approx(_slope(target, pairs[0], (kind, layer), None), rel=0.05, abs=0.02), (kind, layer)

    cells = [{"kind": "resid", "layer": 0, "position": 3}, {"kind": "resid", "layer": 0, "position": 1}]
    cells += [{"kind": "head", "layer": layer, "head": head} for layer in range(layers) for head in range(heads)]
    result = attribution.verify(target, "l4", "tiny", spec, cells, expected_key=run["pairs_key"])
    assert result["cells"][0]["effect"] == pytest.approx(1.0, abs=1e-3), "the changed token carries the whole difference"
    assert result["cells"][1]["effect"] == pytest.approx(0.0, abs=1e-4)
    assert len(result["cells"]) == 2 + layers * heads and all(c["n"] == len(pairs) for c in result["cells"])

    components = [{"kind": "head", "layer": 1, "head": 0}, {"kind": "mlp", "layer": 0}]
    ablated = attribution.ablate(target, "l4", "tiny", spec, components, ["zero", "mean", "resample"])
    assert all(len(m["random"]) == budget.RANDOM_SETS for m in ablated["methods"].values())
    # Resampling from the corrupted prompt can't take the metric outside what the two runs span by much
    assert -0.5 < ablated["methods"]["resample"]["value"] < 1.5
    assert not target.loader._hooks and hooks() == before, "no hooks left behind"


def test_half_precision_gradients():
    """
    The workers load open-weight models in bfloat16. The sweep's backward pass runs in it, and
    exact patching still restores the clean run. (How well bfloat16 estimates track exact
    effects is a property of a real model: it's measured there, not on random weights.)
    """
    target = _open_source_target("llama", torch.bfloat16)
    spec = _letter_rows(target)
    run = attribution.sweep(target, "l4", "tiny", spec)
    assert bool(torch.isfinite(run["heads"]).all()) and float(run["heads"].abs().max()) > 0
    assert all(bool(torch.isfinite(p).all()) for p in run["position_pairs"])
    result = attribution.verify(target, "l4", "tiny", spec, [{"kind": "resid", "layer": 0, "position": 3}])
    assert result["cells"][0]["effect"] == pytest.approx(1.0, abs=1e-3)


# The API over a worker on the CPU, and the assistant's tools

def _local_worker():
    """The inference worker's own methods, run on the CPU over a random nano (no Modal, no GPU)."""
    import services.modal_inference as mi
    from services.model_specs import BASE_MODEL_CONFIGS

    worker = object.__new__(mi._InferenceWorker)
    mi._InferenceWorker.__dict__["setup"]._get_raw_f()(worker)
    torch.manual_seed(0)
    config = BASE_MODEL_CONFIGS["nano"]
    worker._models[("nano", None, None)] = (TransformerLM(config).eval(), config)

    class Local:
        def __getattr__(self, name):
            raw = mi._InferenceWorker.__dict__[name]._get_raw_f()

            async def aio(**kwargs):
                return await asyncio.to_thread(raw, worker, **kwargs)

            return SimpleNamespace(remote=SimpleNamespace(aio=aio))

    return Local


@pytest.fixture()
def api_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.attribution
    from models.database import Base, CreditBalance, User
    from services import assistant_runs, credit_service, lab_runs

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(assistant_runs, "SessionLocal", Session)
    # SQLite has no row locks (Postgres's FOR UPDATE and lock_timeout)
    monkeypatch.setattr(credit_service, "get_credit_balance_for_update",
                        lambda db, user_id: db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first())
    worker = _local_worker()
    monkeypatch.setattr(api.attribution, "_get_inference_worker", lambda model_id: worker)
    lab_runs.clear()
    db = Session()
    user = User(email="attribution@example.com", is_active=True)
    db.add(user)
    db.commit()
    db.add(CreditBalance(user_id=user.id, balance_usd=10, reserved_usd=0))
    db.commit()
    yield db, user
    db.close()
    lab_runs.clear()


def _charges(db, user) -> list:
    from models.database import Transaction

    return db.query(Transaction).filter(Transaction.user_id == user.id).order_by(Transaction.id).all()


def test_api_runs_a_sweep_then_verifies_and_ablates(api_db):
    from api.attribution import ablate_for, estimate_for, run_sweep_for, verify_for
    from api.schemas.attribution import (
        AblateRequest, AblateResponse, Cell, Component, EstimateRequest, PairSpec, SweepRequest, SweepResponse, VerifyRequest, VerifyResponse,
    )
    from core.tensor_codec import to_float32
    from models.database import TransactionType
    from services import lab_runs
    from services.credit_service import min_gpu_request_charge

    db, user = api_db
    pairs = PairSpec(task="ioi", n=8, seed=0)
    estimate = asyncio.run(estimate_for(db, user, EstimateRequest(model="nano", pairs=pairs)))
    assert estimate["checked"] and estimate["within_limit"] and estimate["estimated_cost"] >= min_gpu_request_charge("nano")
    assert estimate["num_pairs"] + len(estimate["dropped"]) == 8

    run = SweepResponse.model_validate(asyncio.run(run_sweep_for(db, user, SweepRequest(model="nano", pairs=pairs)))).model_dump(mode="json")
    assert run["num_pairs"] == estimate["num_pairs"] and run["total_tokens"] == estimate["estimated_tokens"]
    assert run["num_layers"] == 6 and run["num_heads"] == 6 and run["task_name"] == "Indirect object"
    assert to_float32(run["heads"]).shape == (6, 6) and run["heads"]["encoding"] == "base64_float32"
    assert len(run["position_pairs"]) == run["num_pairs"] and len(run["attention"]["patterns"]) == len(run["attention"]["pairs"])
    kept, extra = lab_runs.get(user.id, "attribution", run["run_id"])
    assert kept["attention"] is None and extra["pairs"]["task"] == "ioi", "the kept run leaves the attention patterns out"

    heads = to_float32(run["heads"])
    cells = [Cell(kind="head", layer=layer, head=head, estimate=float(heads[layer, head])) for layer in range(6) for head in range(6)]
    verified = VerifyResponse.model_validate(asyncio.run(verify_for(db, user, VerifyRequest(
        model="nano", pairs=pairs, cells=cells, pairs_key=run["pairs_key"], metric=run["metric"],
    )))).model_dump()
    assert len(verified["cells"]) == 36 and verified["cells"][0]["estimate"] == pytest.approx(float(heads[0, 0]))
    assert verified["rank_correlation"] is not None and verified["gap"] == pytest.approx(run["gap"], abs=1e-4)

    ablated = AblateResponse.model_validate(asyncio.run(ablate_for(db, user, AblateRequest(
        model="nano", pairs=pairs, components=[Component(kind="head", layer=2, head=3), Component(kind="mlp", layer=4)],
        methods=["zero", "resample"], pairs_key=run["pairs_key"],
    )))).model_dump()
    assert set(ablated["methods"]) == {"zero", "resample"} and ablated["mean_over"] is None and len(ablated["random_sets"]) == 5

    charges = _charges(db, user)
    assert [c.transaction_type for c in charges] == [TransactionType.PATCHING_CHARGE] * 3
    assert all(float(-c.amount_usd) >= min_gpu_request_charge("nano") for c in charges)
    assert float(-charges[0].amount_usd) == pytest.approx(run["cost"])

    # Other pairs than the sweep's are refused, and nothing is charged for them
    with pytest.raises(HTTPException) as e:
        asyncio.run(verify_for(db, user, VerifyRequest(model="nano", pairs=PairSpec(task="ioi", n=8, seed=1), cells=cells[:2], pairs_key=run["pairs_key"])))
    assert e.value.status_code == 400 and "run the sweep again" in e.value.detail and len(_charges(db, user)) == 3


def test_api_checks_before_a_gpu_starts(api_db, monkeypatch):
    import api.attribution
    from api.attribution import run_sweep_for, verify_for
    from api.schemas.attribution import Cell, PairRow, PairSpec, SweepRequest, VerifyRequest
    from models.database import CreditBalance

    db, user = api_db

    def no_worker(model_id):
        raise AssertionError("a GPU was started")

    monkeypatch.setattr(api.attribution, "_get_inference_worker", no_worker)
    cases = [
        (PairSpec(task="nope"), 404),
        (PairSpec(task="ioi", rows=[PairRow(clean="a", corrupted="b")]), 400),
        (PairSpec(), 400),
        # An Ozera model's pairs are tokenized on the backend: these two can't be used
        (PairSpec(rows=[PairRow(clean="The cat sat", corrupted="The big dog sat", correct="x", incorrect="y")]), 400),
        (PairSpec(rows=[PairRow(clean="The cat", corrupted="The dog", correct="down", incorrect="down now")]), 400),
    ]
    for pairs, status in cases:
        with pytest.raises(HTTPException) as e:
            asyncio.run(run_sweep_for(db, user, SweepRequest(model="nano", pairs=pairs)))
        assert e.value.status_code == status, (pairs, e.value.detail)
    with pytest.raises(HTTPException) as e:
        asyncio.run(verify_for(db, user, VerifyRequest(model="nano", pairs=PairSpec(task="ioi", n=4), cells=[Cell(kind="head", layer=6, head=0)])))
    assert e.value.status_code == 400 and "layers 0-5" in e.value.detail

    db.query(CreditBalance).filter(CreditBalance.user_id == user.id).update({"balance_usd": 0})
    db.commit()
    with pytest.raises(HTTPException) as e:
        asyncio.run(run_sweep_for(db, user, SweepRequest(model="nano", pairs=PairSpec(task="ioi", n=4))))
    assert e.value.status_code == 402


def test_the_induction_task_and_the_kl_metric_through_the_api(api_db):
    from api.attribution import run_sweep_for
    from api.schemas.attribution import PairRow, PairSpec, SweepRequest

    db, user = api_db
    run = asyncio.run(run_sweep_for(db, user, SweepRequest(model="nano", pairs=PairSpec(task="induction", n=5, seed=2))))
    assert run["num_pairs"] == 5 and run["total_tokens"] == 5 * (2 * INDUCTION_LENGTH - 1) and len(run["classes"]) == 1
    assert run["pairs"][0]["clean_tokens"][:INDUCTION_LENGTH - 1] == run["pairs"][0]["clean_tokens"][INDUCTION_LENGTH:]
    rows = [PairRow(clean="The capital of France is", corrupted="The capital of Spain is"),
            PairRow(clean="The capital of Italy is", corrupted="The capital of Japan is")]
    run = asyncio.run(run_sweep_for(db, user, SweepRequest(model="nano", pairs=PairSpec(rows=rows, name="Mine"))))
    assert run["metric"] == "kl" and run["task"] is None and run["task_name"] == "Mine" and run["dla"] is None


def _quote_and_run(name: str, user_id: int, args: dict):
    from core.assistant.loop import RunQuote
    from services import assistant_runs
    from services.assistant_tools import ToolContext

    ctx = ToolContext(user_id)
    quoted = asyncio.run(assistant_runs.quote(name, ctx, args))
    assert isinstance(quoted, RunQuote), getattr(quoted, "content", quoted)
    outcome = asyncio.run(assistant_runs.run(name, ctx, args))
    assert not outcome.error, outcome.content
    return quoted, outcome


def test_the_assistant_sweeps_verifies_and_ablates(api_db):
    import json

    from services import assistant_runs
    from services.assistant_tools import ToolContext

    db, user = api_db
    args = {"model": "nano", "task": "agreement", "pairs": 6, "rows": [], "seed": 0, "metric": "auto"}
    quoted, swept = _quote_and_run("run_attribution", user.id, args)
    assert dict(quoted.details)["Pairs"].startswith("Subject-verb agreement (") and quoted.estimate > 0
    summary = swept.content
    assert summary["run_id"].startswith("attribution-") and len(summary["top_heads"]) == 10 and "direct_logits" in summary["top_heads"][0]
    assert summary["top_positions"]["cells"] and swept.charged == pytest.approx(swept.result["data"]["run"]["cost"])
    assert swept.result["page"] == "attribution" and swept.result["data"]["pairs"]["task"] == "agreement"
    assert len(json.dumps(summary)) < 6000, "the summary fits a tool result"

    quoted, verified = _quote_and_run("verify_attribution", user.id, {"run_id": summary["run_id"], "cells": "top", "count": 8})
    assert dict(quoted.details)["Cells"] == "the sweep's top 8" and verified.content["cells_patched"] == 8
    assert verified.content["rank_correlation"] is not None and verified.result["data"]["what"] == "top"
    assert len(verified.result["data"]["response"]["cells"]) == 8

    components = [{"kind": "head", "layer": 1, "head": 2}, {"kind": "attn", "layer": 3, "head": -1}]
    quoted, ablated = _quote_and_run("ablate_components", user.id, {"run_id": summary["run_id"], "components": components, "methods": []})
    assert dict(quoted.details)["Components"] == "L1H2, L3 attn"
    assert set(ablated.content["methods"]) == {"zero", "mean", "resample"} and ablated.content["methods"]["zero"]["random_sets"]["min"] is not None

    # The user's own pairs, and what can't run
    rows = [{"clean": "The cat sat", "corrupted": "The dog sat", "correct": "down", "incorrect": "up"},
            {"clean": "A cat ran", "corrupted": "A dog ran", "correct": "", "incorrect": ""}]
    _, own = _quote_and_run("run_attribution", user.id, {"model": "nano", "task": "", "pairs": 0, "rows": rows, "seed": 0, "metric": "auto"})
    assert own.content["metric"] == "kl" and own.content["pairs"] == 2
    ctx = ToolContext(user.id)
    for name, bad in (
        ("run_attribution", {**args, "rows": rows}),
        ("run_attribution", {**args, "task": "nope"}),
        ("verify_attribution", {"run_id": "attribution-none", "cells": "top", "count": 5}),
        ("ablate_components", {"run_id": summary["run_id"], "components": [{"kind": "head", "layer": 9, "head": 0}], "methods": []}),
    ):
        outcome = asyncio.run(assistant_runs.quote(name, ctx, bad))
        assert getattr(outcome, "error", False), (name, bad)


def test_figures_are_drawn_from_the_pages_data():
    from api.export import DataFigureRequest, export_data_figure

    user = SimpleNamespace(id=1)
    heatmap = DataFigureRequest(kind="heatmap", values=[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]], row_labels=["0", "1"], column_labels=["The", " cat", " sat"])
    response = asyncio.run(export_data_figure(heatmap, user))
    assert response.media_type == "image/png" and response.body[:4] == b"\x89PNG"
    scatter = DataFigureRequest(kind="scatter", x=[0.1, 0.2, -0.3], y=[0.12, 0.1, -0.2], errors=[0.01, 0.02, 0.03], note="Spearman 0.50",
                                config={"format": "svg"})
    assert b"<svg" in asyncio.run(export_data_figure(scatter, user)).body
    bars = DataFigureRequest(kind="bars", bars=[{"label": "Chosen", "value": 0.2, "error": 0.05}, {"label": "Random", "value": 0.9, "marks": [0.8, 1.0]}],
                             baseline=1.0, baseline_label="Clean", config={"format": "pdf"})
    assert asyncio.run(export_data_figure(bars, user)).body[:4] == b"%PDF"
    for bad in (DataFigureRequest(kind="heatmap", values=[[1.0, 2.0], [1.0]]), DataFigureRequest(kind="scatter", x=[1.0], y=[]),
                DataFigureRequest(kind="bars")):
        with pytest.raises(HTTPException):
            asyncio.run(export_data_figure(bad, user))
