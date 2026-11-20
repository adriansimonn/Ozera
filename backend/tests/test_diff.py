"""
Model Diff (core.diff, api/diff.py; INSTRUMENTS_PLAN.md's Phase 4): the text samples and how
two models read them, the layer statistics against the Geometry Lab's own functions, a planted
difference found by the layer profile, by the directions and by patching, the API's routes
over a worker run on the CPU, and the assistant's three tools. Run from backend/:
python -m pytest tests/test_diff.py

The open-weight architectures (Llama, Qwen3, Gemma 3) are tested on small random models when
transformers is installed (it's only in the Modal images).
"""

import asyncio
import copy
import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import diff  # noqa: E402
from core.diff import budget  # noqa: E402
from core.diff.components import control_directions, difference_components  # noqa: E402
from core.diff.similarity import StreamSimilarity  # noqa: E402
from core.diff.texts import builtin_samples, build_texts, encode_texts, normalize_texts, sample_summary  # noqa: E402
from core.geometry.subspace import linear_cka, procrustes_r2  # noqa: E402
from core.probes.runner import OzeraTarget  # noqa: E402
from core.transformer.config import TransformerConfig  # noqa: E402
from core.transformer.model_torch import TransformerLM  # noqa: E402
from tests.test_attribution import _WordTokenizer  # noqa: E402

NAMES = ("base", "other")
# The layer whose MLP is changed in the planted pairs, and the residual stream position after it
CHANGED = 1
AFTER = CHANGED + 1
# How far the changed layer's output is moved, along a unit direction
PLANT = 12.0


def _model(seed: int = 0) -> tuple[TransformerLM, TransformerConfig]:
    torch.manual_seed(seed)
    config = TransformerConfig(
        vocab_size=400, d_model=48, num_layers=4, num_heads=4, d_ff=96, max_seq_len=64, initializer_range=0.3,
    )
    model = TransformerLM(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0, 0.3)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(0, 0.2)
    return model, config


def _texts(count: int = 12) -> list[str]:
    words = "the cat sat on a mat while rain fell over hills and rivers ran down to sea birds".split()
    generator = torch.Generator().manual_seed(1)
    return [" ".join(words[int(i)] for i in torch.randint(len(words), (6 + n % 5,), generator=generator)) for n in range(count)]


@pytest.fixture(scope="module")
def planted():
    """
    A base model, and a copy whose layer-1 MLP writes one more fixed vector (its output bias
    is moved along a direction): the two differ by that vector from the layer on.
    """
    model, config = _model()
    changed = copy.deepcopy(model)
    direction = torch.nn.functional.normalize(torch.randn(config.d_model, generator=torch.Generator().manual_seed(5)), dim=0)
    with torch.no_grad():
        changed.blocks[CHANGED].feed_forward.fc2.bias += PLANT * direction
    tokenizer = _WordTokenizer()
    return OzeraTarget(model, config, tokenizer), OzeraTarget(changed, config, tokenizer), direction


# Texts

def test_samples_build_the_same_texts_every_time():
    samples = builtin_samples()
    assert list(samples) == ["prose", "instructions"]
    for sample_id, sample in samples.items():
        assert len(sample["texts"]) >= 48 and len(set(sample["texts"])) == len(sample["texts"])
        assert all(0 < len(text) <= budget.MAX_TEXT_CHARS and text == text.strip() for text in sample["texts"])
        texts = build_texts(sample_id, 32, 0)
        assert len(texts) == 32 and texts == build_texts(sample_id, 32, 0) and texts[:10] == build_texts(sample_id, 10, 0)
        assert texts != build_texts(sample_id, 32, 1)
        summary = sample_summary(sample)
        assert summary["num_texts"] == len(sample["texts"]) and len(summary["examples"]) == 3
        with pytest.raises(budget.DiffInputError):
            build_texts(sample_id, len(sample["texts"]) + 1, 0)
    with pytest.raises(budget.DiffInputError):
        build_texts("nope", 8, 0)


def test_texts_are_normalized():
    assert normalize_texts(["  a b ", "c", "", "a b", "d", "e"]) == ["a b", "c", "d", "e"]
    for bad in (["a", "b", "c"], ["x" * (budget.MAX_TEXT_CHARS + 1), "a", "b", "c", "d"], [f"t{i}" for i in range(budget.MAX_TEXTS + 1)]):
        with pytest.raises(budget.DiffInputError):
            normalize_texts(bad)


def test_both_models_read_the_same_tokens(planted):
    base, other, _ = planted
    texts = _texts(6) + ["word " * 100]
    encoded, dropped, cut = encode_texts(base, other, texts, False)
    assert [t.source for t in encoded] == list(range(7)) and not dropped and cut == 1
    assert len(encoded[-1].ids) == 64, "a long text is cut to what both models can read"
    assert all(t.lead == 0 and t.start == 1 and t.compared == len(t.ids) - 1 for t in encoded), "a text's first token isn't compared"
    _, dropped, _ = encode_texts(base, other, texts[:5] + ["word"], False)
    assert dropped == [{"index": 5, "reason": "it has no tokens after its first"}]
    with pytest.raises(budget.DiffInputError):
        encode_texts(base, other, texts[:2], False)
    with pytest.raises(budget.DiffInputError) as e:
        encode_texts(base, other, texts, True)
    assert "chat template" in str(e.value)

    # A model with a smaller vocabulary can't read every token; one with another shape can't be compared
    small_config = dataclasses.replace(base.model.config, vocab_size=5)
    small = OzeraTarget(TransformerLM(small_config).eval(), small_config, base.tokenizer)
    with pytest.raises(budget.DiffInputError) as e:
        encode_texts(base, small, texts, False)
    assert "doesn't know" in str(e.value)
    narrow_config = dataclasses.replace(base.model.config, d_model=32, d_ff=64)
    narrow = OzeraTarget(TransformerLM(narrow_config).eval(), narrow_config, base.tokenizer)
    with pytest.raises(budget.DiffInputError) as e:
        diff.layers(base, narrow, "l4", NAMES, texts)
    assert "shape" in str(e.value)


# Layers

def test_streaming_similarity_matches_the_geometry_labs():
    """Linear CKA and Procrustes from sums over batches are the Geometry Lab's, from the activations themselves."""
    generator = torch.Generator().manual_seed(0)
    n, d = 300, 24
    # A mean far larger than the spread, as residual streams have
    a = torch.randn(n, d, generator=generator) * torch.linspace(0.2, 3, d) + 200 * torch.randn(d, generator=generator)
    rotation = torch.linalg.qr(torch.randn(d, d, generator=generator)).Q
    b = 0.7 * (a - a.mean(dim=0)) @ rotation + 0.4 * torch.randn(n, d, generator=generator) - 50
    half = torch.arange(n) % 2

    for matrix_halves in (True, False):
        stats = StreamSimilarity(2, d, matrix_halves, torch.device("cpu"))
        distances = torch.zeros(n)
        for start in range(0, n, 64):
            for h in (0, 1):
                rows = (torch.arange(n) >= start) & (torch.arange(n) < start + 64) & (half == h)
                stats.add(h, 0, a[rows], a[rows])
                distances[rows] = stats.add(h, 1, a[rows], b[rows])
        whole = stats.alignment()
        assert whole["cka"][0] == pytest.approx(1.0, abs=1e-4) and whole["procrustes"][0] == pytest.approx(1.0, abs=1e-4)
        assert whole["cka"][1] == pytest.approx(float(linear_cka(a, b)), abs=2e-4)
        assert whole["procrustes"][1] == pytest.approx(float(procrustes_r2(a, b)), abs=2e-4)
        assert 0 < whole["cka_chance"][1] < whole["cka"][1]
        assert torch.allclose(distances, (a - b).norm(dim=1), rtol=1e-4)
        if matrix_halves:
            for h in (0, 1):
                part = stats.alignment(h)
                assert part["cka"][1] == pytest.approx(float(linear_cka(a[half == h], b[half == h])), abs=2e-4)
                assert part["procrustes"][1] == pytest.approx(float(procrustes_r2(a[half == h], b[half == h])), abs=2e-4)
        else:
            with pytest.raises(ValueError):
                stats.alignment(0)

        # Distance: the root mean squared difference over the spread of both models' activations
        spread = ((a - a.mean(dim=0)).pow(2).sum() + (b - b.mean(dim=0)).pow(2).sum()) / 2
        assert stats.distance((0, 1))[0] == 0.0
        assert stats.distance((0, 1))[1] == pytest.approx(float(((a - b).pow(2).sum() / spread).sqrt()), rel=1e-4)
        assert stats.scale(1) == pytest.approx(float((spread / n).sqrt()), rel=1e-4)
        assert stats.mean_cosine((0, 1))[1] == pytest.approx(float(torch.nn.functional.cosine_similarity(a, b).mean()), abs=1e-5)

    # Tokens paired at random score about the chance level
    shuffled = StreamSimilarity(1, d, False, torch.device("cpu"))
    shuffled.add(0, 0, a, b[torch.randperm(n, generator=generator)])
    result = shuffled.alignment()
    assert result["cka"][0] < 3 * result["cka_chance"][0] < 0.5


def test_identical_models_dont_differ(planted):
    base, _, _ = planted
    run = diff.layers(base, base, "l4", NAMES, _texts())
    positions = run["num_layers"] + 1
    assert run["num_texts"] == 12 and run["num_tokens"] == run["total_tokens"] - 12 == sum(len(t["tokens"]) for t in run["texts"])
    assert run["massive"] == [0] * positions
    assert run["distance"] == [0.0] * positions and run["kl"] == pytest.approx(0.0, abs=1e-6) and run["top1_agreement"] == 1.0
    for key in ("cosine", "cka", "procrustes"):
        assert run[key] == pytest.approx([1.0] * positions, abs=1e-4), key
    assert float(run["token_distance"].abs().max()) == 0.0


def test_the_layer_profile_shows_where_a_model_was_changed(planted):
    base, other, _ = planted
    run = diff.layers(base, other, "l4", NAMES, _texts())
    positions = run["num_layers"] + 1
    # Nothing differs before the changed layer writes
    assert run["distance"][:AFTER] == [0.0] * AFTER and all(v > 0.05 for v in run["distance"][AFTER:])
    assert run["cka"][:AFTER] == pytest.approx([1.0] * AFTER, abs=1e-4)
    assert run["kl"] > 0 and 0 <= run["top1_agreement"] <= 1
    assert run["token_distance"].shape == (positions, run["num_tokens"])
    # The root mean square of the tokens' distances is the profile
    read = run["token_distance"][AFTER]
    assert int((read < 0).sum()) == run["massive"][AFTER]
    assert float(read[read >= 0].pow(2).mean().sqrt()) == pytest.approx(run["distance"][AFTER], rel=1e-4)

    halves = run["halves"]
    assert run["half_tokens"][0] + run["half_tokens"][1] == run["num_tokens"]
    assert [t["half"] for t in run["texts"]] == [i % 2 for i in range(12)]
    for key in ("distance", "cosine", "cka", "procrustes"):
        assert len(halves[key]) == 2 and len(halves[key][0]) == positions, key
    assert halves["distance"][0][AFTER] != halves["distance"][1][AFTER], "the halves are read separately"
    weights = torch.tensor(run["half_tokens"], dtype=torch.float64)
    assert float((torch.tensor(run["half_kl"], dtype=torch.float64) * weights).sum() / weights.sum()) == pytest.approx(run["kl"], rel=1e-4)

    text = run["texts"][3]
    assert len(text["kl"]) == len(text["tokens"]) == len(text["base_top"]) == len(text["other_prob"])
    assert all(v >= 0 for v in text["kl"]) and all(0 < p <= 1 for p in text["base_prob"])

    # The next-token KL is the models' own
    ids = torch.tensor([base.tokenizer.encode(_texts()[3])])
    with torch.no_grad():
        log_base = torch.log_softmax(base.model(ids)[0][0], dim=-1)
        log_other = torch.log_softmax(other.model(ids)[0][0], dim=-1)
    expected = (log_other.exp() * (log_other - log_base)).sum(dim=-1)[1:]
    assert text["kl"] == pytest.approx(expected.tolist(), abs=2e-4)
    assert text["other_top"] == [base.tokenizer.decode([i]) for i in log_other.argmax(dim=-1).tolist()[1:]]
    assert text["tokens"] == [base.tokenizer.decode([i]) for i in ids[0].tolist()[1:]]
    assert text["lead"] == [base.tokenizer.decode([int(ids[0, 0])])], "the text's first token is shown, not compared"


def test_wide_models_keep_their_second_moments_once(planted, monkeypatch):
    base, other, _ = planted
    monkeypatch.setattr(budget, "HALVES_BYTES", 0)
    run = diff.layers(base, other, "l4", NAMES, _texts())
    assert run["halves"]["cka"] is None and run["halves"]["procrustes"] is None
    assert len(run["halves"]["distance"]) == 2 and run["cka"][-1] < 1


# Directions

def test_components_are_the_differences_principal_directions():
    generator = torch.Generator().manual_seed(0)
    axes = torch.linalg.qr(torch.randn(16, 16, generator=generator)).Q
    # A constant offset along one axis, and less varying along two others
    differences = 3.0 * axes[0] + torch.randn(500, 1, generator=generator) * axes[1] + 0.3 * torch.randn(500, 1, generator=generator) * axes[2]
    components, shares, mean_square = difference_components(differences, 4)
    assert components.shape == (4, 16) and torch.allclose(components.norm(dim=1), torch.ones(4), atol=1e-5)
    assert abs(float(components[0] @ axes[0])) > 0.98 and float((differences @ components[0]).mean()) > 0, "it points the way the mean difference does"
    assert abs(float(components[1] @ axes[1])) > 0.98
    assert float(shares[0]) > float(shares[1]) > float(shares[2]) and float(shares.sum()) == pytest.approx(1.0, abs=1e-4)
    assert mean_square == pytest.approx(float(differences.pow(2).sum(dim=1).mean()), rel=1e-5)
    controls = control_directions(components[0], 3, generator)
    assert torch.allclose(controls @ components[0], torch.zeros(3), atol=1e-5) and torch.allclose(controls.norm(dim=1), torch.ones(3), atol=1e-5)
    with pytest.raises(budget.DiffInputError):
        difference_components(torch.zeros(10, 16))


def test_the_planted_direction_is_found_and_closes_the_gap(planted):
    base, other, direction = planted
    run = diff.directions(base, other, "l4", NAMES, _texts(), AFTER, seed=3)
    assert run["layer"] == AFTER and run["fit_texts"] == run["test_texts"] == 6 and run["seed"] == 3
    assert run["fit_tokens"] + run["test_tokens"] == run["total_tokens"] - 12 and run["massive_tokens"] == 0
    # The models differ by one fixed vector: one component holds all of the difference, the same for every token
    first = run["components"][0]
    assert first["index"] == 1 and first["share"] == pytest.approx(1.0, abs=1e-4) and first["mean_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert run["mean_share"] == pytest.approx(1.0, abs=1e-4) and run["relative_difference"] > 0
    assert len(first["high"]) == budget.EXTREME_TOKENS and first["high"][0]["value"] == pytest.approx(PLANT, abs=1e-3)
    context = first["high"][0]
    assert context["token"] and len(context["before"]) <= budget.CONTEXT_BEFORE and len(context["after"]) <= budget.CONTEXT_AFTER

    test = run["test"]
    assert test["kl_base"] > 0.01 and test["kl_full"] == pytest.approx(0.0, abs=1e-5), "nothing after the layer differs"
    assert test["full_closed"] == pytest.approx(1.0, abs=1e-3) and test["agreement_full"] == 1.0
    # The component gets all of it back but what each text's first token carries (it isn't compared, so it isn't edited)
    top = test["components"][0]
    assert 0.85 < top["closed"] <= 1.0 and top["agreement"] > test["agreement_base"]
    assert len(top["controls"]) == budget.CONTROL_DIRECTIONS and all(c["closed"] < 0.3 for c in top["controls"])
    assert len(test["components"]) == budget.TESTED_COMPONENTS and test["joint"]["count"] == budget.TESTED_COMPONENTS
    assert test["joint"]["closed"] == pytest.approx(top["closed"], abs=1e-3), "the other components hold nothing"

    # Before the changed layer the models are the same, and there is nothing to decompose
    with pytest.raises(budget.DiffInputError) as e:
        diff.directions(base, other, "l4", NAMES, _texts(), CHANGED)
    assert "identical" in str(e.value)
    with pytest.raises(budget.DiffInputError):
        diff.directions(base, other, "l4", NAMES, _texts(), 9)

    # At the last position the edit is read straight by the unembedding
    last = diff.directions(base, other, "l4", NAMES, _texts(), run["num_layers"])
    assert last["test"]["full_closed"] == pytest.approx(1.0, abs=1e-3)
    # Entering the first layer (position 0) the models are the same too
    with pytest.raises(budget.DiffInputError):
        diff.directions(base, other, "l4", NAMES, _texts(), 0)


def test_edits_at_the_embeddings_reach_the_model():
    """Position 0 is edited on the way into the first layer."""
    model, config = _model()
    changed = copy.deepcopy(model)
    # The positional embeddings (the token embeddings are tied to the unembedding, which would differ too)
    positional = changed.pos_embedding.weight if config.learned_pos_emb else changed.pos_embedding
    with torch.no_grad():
        positional += 0.5 * torch.randn(positional.shape, generator=torch.Generator().manual_seed(2))
    tokenizer = _WordTokenizer()
    base, other = OzeraTarget(model, config, tokenizer), OzeraTarget(changed, config, tokenizer)
    run = diff.directions(base, other, "l4", NAMES, _texts(), 0)
    # Only what enters the first layer differs: the whole difference closes the gap
    assert run["test"]["kl_base"] > 0 and run["test"]["full_closed"] == pytest.approx(1.0, abs=1e-3)
    assert 0 < run["test"]["joint"]["closed"] <= 1.0 + 1e-6
    # A position's difference is the same in every text: a handful of directions hold most of it
    assert sum(c["share"] for c in run["components"]) > 0.5


# Patch

def test_patching_the_changed_layer_restores_the_donor(planted):
    base, other, _ = planted
    prompt = "the cat sat on a mat while rain fell"
    layers = len(base.layers)
    run = diff.patch(base, other, "l4", NAMES, prompt, donor="other", kind="mlp")
    assert run["donor"] == "other" and run["kind"] == "mlp" and run["prompt_tokens"] == len(run["tokens"]) == 9
    assert run["kl_base"] > 0.01 and len(run["grid"]) == layers and len(run["grid"][0]) == 9
    # Only layer 1's MLP differs: with its output patched everywhere the base model is the other model
    assert run["all_positions"][CHANGED] == pytest.approx(0.0, abs=1e-5)
    # Before the change the two models' outputs are the same, and patching them does nothing
    assert run["all_positions"][0] == pytest.approx(run["kl_base"], rel=1e-3)
    assert run["grid"][0] == pytest.approx([run["kl_base"]] * 9, rel=1e-3)
    # After it the donor's MLPs read a different stream: their outputs carry part of the change
    assert all(0 <= run["all_positions"][layer] != run["kl_base"] for layer in (2, 3))
    assert min(run["grid"][CHANGED]) < run["kl_base"], "one position's patch closes part of it"
    assert len(run["predictions"]["donor"]) == 5 and run["predictions"]["donor"][0]["prob"] >= run["predictions"]["donor"][1]["prob"]

    # The residual stream: the donor's stream entering a layer after the change carries all of it
    stream = diff.patch(base, other, "l4", NAMES, prompt, donor="other", kind="resid")
    assert stream["all_positions"][:AFTER] == pytest.approx([stream["kl_base"]] * AFTER, rel=1e-3)
    assert stream["all_positions"][AFTER:] == pytest.approx([0.0] * (layers - AFTER), abs=1e-5)
    # Patching the last position entering the last layer leaves only what its attention reads elsewhere
    assert stream["grid"][layers - 1][-1] < stream["kl_base"]

    # The other way round, the KL is measured from the base model's prediction
    back = diff.patch(base, other, "l4", NAMES, prompt, donor="base", kind="mlp")
    assert back["donor"] == "base" and back["all_positions"][CHANGED] == pytest.approx(0.0, abs=1e-5)
    assert back["predictions"]["donor"] == run["predictions"]["recipient"]

    for bad in ({"kind": "head"}, {"donor": "neither"}):
        with pytest.raises(budget.DiffInputError):
            diff.patch(base, other, "l4", NAMES, prompt, **bad)
    with pytest.raises(budget.DiffInputError) as e:
        diff.patch(base, other, "l4", NAMES, "word " * 70)
    assert "more than 64 tokens" in str(e.value)
    with pytest.raises(budget.DiffInputError):
        diff.patch(base, other, "l4", NAMES, "word")


# Budget

def test_budget():
    sizes = dict(num_texts=48, total_tokens=2400)
    small = budget.estimate_layers_seconds("l4", 135_000_000, 30, 576, **sizes)
    assert small < budget.estimate_layers_seconds("l4", 135_000_000, 30, 576, num_texts=96, total_tokens=4800)
    assert small < budget.estimate_layers_seconds("l4", 1_700_000_000, 28, 2048, **sizes)
    assert budget.similarity_halves(30, 576) and not budget.similarity_halves(36, 2560)
    assert budget.similarity_bytes(36, 2560, False) < 3.2e9
    assert budget.direction_conditions() == 2 + budget.TESTED_COMPONENTS * (1 + budget.CONTROL_DIRECTIONS) + 1
    assert budget.estimate_directions_seconds("l4", 135_000_000, 576, **sizes) > budget.SETUP_SECONDS
    assert budget.patch_token_passes(30, 20) == 2 * 20 + 30 * 21 * 20
    assert budget.estimate_patch_seconds("a10g", 4_000_000_000, 36, 64) > budget.estimate_patch_seconds("a10g", 4_000_000_000, 36, 16)
    assert budget.estimate_tokens(4000, 10, False) == 1010 and budget.estimate_tokens(10 ** 7, 10, True) == 10 * budget.MAX_TEXT_TOKENS
    budget.check_run_size("l4", "tiny", 10)
    with pytest.raises(budget.DiffInputError):
        budget.check_run_size("l4", "tiny", 10_000)
    # The built-in samples at their default size fit every pair's worker
    from core.open_source import OPEN_SOURCE_MODELS
    from core.probes.budget import MAX_RUN_SECONDS
    for config in OPEN_SOURCE_MODELS.values():
        if config.is_instruct:
            continue
        tokens = budget.estimate_tokens(48 * 220, 48, False)
        for seconds in (
            budget.estimate_layers_seconds(config.gpu_tier, config.parameters, config.num_layers, config.hidden_dim, 48, tokens),
            budget.estimate_directions_seconds(config.gpu_tier, config.parameters, config.hidden_dim, 48, tokens),
            budget.estimate_patch_seconds(config.gpu_tier, config.parameters, config.num_layers, 24),
        ):
            assert seconds < MAX_RUN_SECONDS[config.gpu_tier], (config.model_id, seconds)


# Open-weight architectures

def _open_weight_pair(family: str, dtype=torch.float32):
    """A small random open-weight model, and a copy with one layer's MLP changed."""
    from core.probes.runner import OpenSourceTarget
    from tests.test_attribution import _open_source_target

    base = _open_source_target(family, dtype)
    loader = copy.copy(base.loader)
    loader.model = copy.deepcopy(base.loader.model)
    other = OpenSourceTarget(loader)
    weight = other.layers[CHANGED].mlp.down_proj.weight
    with torch.no_grad():
        weight += (0.5 * torch.randn(weight.shape, generator=torch.Generator().manual_seed(4))).to(weight.dtype)
    return base, other


def _letters(count: int = 10) -> list[str]:
    generator = torch.Generator().manual_seed(2)
    return ["".join(chr(97 + int(i)) for i in torch.randint(26, (8 + n % 4,), generator=generator)) for n in range(count)]


@pytest.mark.parametrize("family", ["llama", "qwen", "gemma"])
def test_open_weight_architectures(family):
    base, other = _open_weight_pair(family)
    hooks = lambda: sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for t in (base, other) for m in t.loader.model.modules())  # noqa: E731
    before = hooks()
    texts = _letters()
    run = diff.layers(base, other, "l4", NAMES, texts)
    positions = run["num_layers"] + 1
    # The tokenizer's BOS isn't compared
    assert run["total_tokens"] == run["num_tokens"] + len(texts) and len(run["texts"][0]["tokens"]) == len(texts[0])
    assert run["texts"][0]["lead"] == [], "after a BOS every token of the text is compared"
    assert run["distance"][:AFTER] == [0.0] * AFTER and all(v > 0 for v in run["distance"][AFTER:])
    assert all(v is not None and 0 <= v <= 1 for key in ("cka", "procrustes") for v in run[key])
    assert run["kl"] > 0 and run["token_distance"].shape == (positions, run["num_tokens"])

    found = diff.directions(base, other, "l4", NAMES, texts, AFTER)
    # Only that layer's MLP differs: its whole difference, added back, leaves nothing between the models
    assert found["test"]["kl_base"] > 0 and found["test"]["full_closed"] == pytest.approx(1.0, abs=2e-3)
    assert 0.99 < sum(c["share"] for c in found["components"]) <= 1.0 + 1e-4 or len(found["components"]) == budget.COMPONENTS
    assert found["components"][0]["high"][0]["token"] != "<bos>"

    for kind in budget.PATCH_KINDS:
        patched = diff.patch(base, other, "l4", NAMES, "abcdefgh", kind=kind)
        assert patched["tokens"][0] == "<bos>" and len(patched["grid"]) == run["num_layers"] and len(patched["grid"][0]) == 9
    assert diff.patch(base, other, "l4", NAMES, "abcdefgh", kind="mlp")["all_positions"][CHANGED] == pytest.approx(0.0, abs=1e-5)
    assert hooks() == before, "no hooks left behind"


def test_half_precision_models():
    """The workers load open-weight models in bfloat16: the statistics are taken in float32 from them."""
    base, other = _open_weight_pair("llama", torch.bfloat16)
    run = diff.layers(base, other, "l4", NAMES, _letters())
    assert run["distance"][0] == 0.0 and run["distance"][-1] > 0 and bool(torch.isfinite(run["token_distance"]).all())
    assert all(v is not None for v in run["cka"] + run["procrustes"])
    found = diff.directions(base, other, "l4", NAMES, _letters(), AFTER)
    assert found["test"]["full_closed"] == pytest.approx(1.0, abs=0.05)
    patched = diff.patch(base, other, "l4", NAMES, "abcdefgh")
    assert patched["all_positions"][-1] < patched["kl_base"]


# The API over a worker on the CPU, and the assistant's tools

CUSTOM = "mine"


def _local_worker(user_id: int, version: str):
    """The inference worker's own methods on the CPU: a random nano, and a custom model that is nano with one layer changed."""
    import services.modal_inference as mi
    from services.model_specs import BASE_MODEL_CONFIGS

    worker = object.__new__(mi._InferenceWorker)
    mi._InferenceWorker.__dict__["setup"]._get_raw_f()(worker)
    torch.manual_seed(0)
    config = BASE_MODEL_CONFIGS["nano"]
    base = TransformerLM(config).eval()
    custom = copy.deepcopy(base)
    with torch.no_grad():
        custom.blocks[2].feed_forward.fc2.weight += 0.2 * torch.randn(custom.blocks[2].feed_forward.fc2.weight.shape)
    worker._models[("nano", None, None)] = (base, config)
    # A custom model's context window is the sequence length it was trained with
    worker._models[(CUSTOM, user_id, version)] = (custom, dataclasses.replace(config, max_seq_len=64))

    class Local:
        def __getattr__(self, name):
            raw = mi._InferenceWorker.__dict__[name]._get_raw_f()

            async def aio(**kwargs):
                return await asyncio.to_thread(raw, worker, **kwargs)

            return SimpleNamespace(remote=SimpleNamespace(aio=aio))

    return Local


@pytest.fixture()
def api_db(monkeypatch):
    from datetime import datetime

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.diff
    from models.database import Base, CreditBalance, JobStatus, TrainingJob, User
    from services import assistant_runs, assistant_tools, credit_service

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(assistant_runs, "SessionLocal", Session)
    monkeypatch.setattr(assistant_tools, "SessionLocal", Session)
    # SQLite has no row locks (Postgres's FOR UPDATE and lock_timeout)
    monkeypatch.setattr(credit_service, "get_credit_balance_for_update",
                        lambda db, user_id: db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first())
    db = Session()
    user = User(email="diff@example.com", is_active=True)
    db.add(user)
    db.commit()
    db.add(CreditBalance(user_id=user.id, balance_usd=10, reserved_usd=0))
    # The user's custom model: nano trained on their own text
    db.add(TrainingJob(
        job_id="job-1", user_id=user.id, dataset_id="d", model_config="nano", model_name=CUSTOM, epochs=1, batch_size=8,
        learning_rate=1e-3, seq_len=64, status=JobStatus.COMPLETED, estimated_cost_usd=1, estimated_minutes=1, reserved_credits_usd=0,
        completed_at=datetime.utcnow(),
    ))
    db.commit()
    worker = _local_worker(user.id, "job-1")
    monkeypatch.setattr(api.diff, "_get_inference_worker", lambda model_id: worker)
    yield db, user
    db.close()


def _charges(db, user) -> list:
    from models.database import Transaction

    return db.query(Transaction).filter(Transaction.user_id == user.id).order_by(Transaction.id).all()


def test_pairs_are_siblings_or_a_custom_model_and_its_base(api_db):
    from api.diff import list_pairs_for, pair_of, resolve_pair
    from api.schemas.diff import PairInfo
    from core.open_source import OPEN_SOURCE_MODELS

    db, user = api_db
    pairs = [PairInfo.model_validate(p) for p in asyncio.run(list_pairs_for(db, user))]
    assert (pairs[0].base, pairs[0].other, pairs[0].kind, pairs[0].chat_template) == ("nano", CUSTOM, "custom", False)
    assert pairs[0].base_name == "Ozera Nano" and pairs[0].num_layers == 6 and pairs[0].hidden_dim == 192
    siblings = pairs[1:]
    assert len(siblings) == sum(1 for c in OPEN_SOURCE_MODELS.values() if not c.is_instruct) == 8
    assert all(p.kind == "instruct" and p.chat_template and p.other == f"{p.base}-it" for p in siblings)
    assert {p.gpu_tier for p in siblings} == {"l4", "a10g"}
    assert len(asyncio.run(list_pairs_for(db, None))) == 8, "signed out, only the shared pairs"

    pair = asyncio.run(resolve_pair(db, user, "nano", CUSTOM))
    assert pair.kind == "custom" and pair.other_ref.owner_id == user.id and pair.other_ref.version == "job-1" and pair.tier == "l4"
    assert asyncio.run(resolve_pair(db, user, "qwen3-4b", "qwen3-4b-it")).tier == "a10g"
    for base, other, status in (
        ("smollm-135m", "qwen3-0.6b-it", 400), ("smollm-135m-it", "smollm-135m", 400), ("nano", "smollm-135m", 400),
        ("nano", "nano", 400), ("nano", "nope", 404), ("mini", CUSTOM, 400), (CUSTOM, "nano", 400),
    ):
        with pytest.raises(HTTPException) as e:
            asyncio.run(resolve_pair(db, user, base, other))
        assert e.value.status_code == status, (base, other, e.value.detail)

    # One model's ID names its pair
    for model, expected in ((CUSTOM, ("nano", CUSTOM)), ("nano", ("nano", CUSTOM)), ("gemma-3-270m-it", ("gemma-3-270m", "gemma-3-270m-it")),
                            ("gemma-3-270m", ("gemma-3-270m", "gemma-3-270m-it"))):
        assert asyncio.run(pair_of(db, user, model)) == expected
    for model, status in (("mini", 400), ("nope", 404)):
        with pytest.raises(HTTPException) as e:
            asyncio.run(pair_of(db, user, model))
        assert e.value.status_code == status


def test_api_runs_the_three_views(api_db):
    from api.diff import estimate_for, run_directions_for, run_layers_for, run_patch_for
    from api.schemas.diff import (
        DirectionsRequest, DirectionsResponse, EstimateRequest, LayersRequest, LayersResponse, PatchRequest, PatchResponse, TextSpec,
    )
    from core.tensor_codec import to_float32
    from models.database import TransactionType
    from services.credit_service import min_gpu_request_charge

    db, user = api_db
    texts = TextSpec(sample="prose", n=8, seed=0)
    estimate = asyncio.run(estimate_for(db, user, EstimateRequest(kind="layers", base="nano", other=CUSTOM, texts=texts)))
    assert estimate["checked"] and estimate["within_limit"] and estimate["num_texts"] == 8
    assert estimate["estimated_cost"] >= min_gpu_request_charge("nano")

    run = LayersResponse.model_validate(asyncio.run(run_layers_for(db, user, LayersRequest(base="nano", other=CUSTOM, texts=texts)))).model_dump(mode="json")
    assert run["total_tokens"] == estimate["estimated_tokens"], "an Ozera pair's tokens are counted before the GPU starts"
    assert run["num_layers"] == 6 and run["num_texts"] == 8 and run["texts_name"] == "Prose" and run["other_name"] == CUSTOM
    assert run["token_distance"]["encoding"] == "base64_float32" and to_float32(run["token_distance"]).shape == (7, run["num_tokens"])
    assert run["distance"][:3] == [0.0, 0.0, 0.0] and run["distance"][3] > 0, "the custom model's layer 2 was changed"
    assert all(len(t["tokens"]) <= 64 for t in run["texts"]), "texts are cut to the custom model's context window"

    found = DirectionsResponse.model_validate(asyncio.run(run_directions_for(db, user, DirectionsRequest(
        base="nano", other=CUSTOM, texts=texts, layer=3, seed=7,
    )))).model_dump()
    assert found["layer"] == 3 and found["control_seed"] == 7 and found["seed"] == 0 and len(found["components"]) == 8
    assert found["test"]["full_closed"] == pytest.approx(1.0, abs=1e-3) and len(found["test"]["components"]) == 4

    patched = PatchResponse.model_validate(asyncio.run(run_patch_for(db, user, PatchRequest(
        base="nano", other=CUSTOM, prompt="The ferry left the harbor a little after six ", donor="base", kind="attn",
    )))).model_dump()
    assert patched["donor"] == "base" and patched["kind"] == "attn" and patched["prompt"].endswith("six")
    assert len(patched["grid"]) == 6 and len(patched["grid"][0]) == len(patched["tokens"])

    charges = _charges(db, user)
    assert [c.transaction_type for c in charges] == [TransactionType.PATCHING_CHARGE] * 3
    assert all(float(-c.amount_usd) >= min_gpu_request_charge("nano") and c.description.startswith("Model diff, ") for c in charges)
    assert float(-charges[0].amount_usd) == pytest.approx(run["cost"])

    # The user's own texts
    own = asyncio.run(run_layers_for(db, user, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec(texts=["One two.", "Three four five.", "Six.", "Seven eight nine ten."], name="Mine"))))
    assert own["sample"] is None and own["texts_name"] == "Mine" and own["num_texts"] == 4


def test_api_checks_before_a_gpu_starts(api_db, monkeypatch):
    import api.diff
    from api.diff import estimate_for, run_directions_for, run_layers_for, run_patch_for
    from api.schemas.diff import DirectionsRequest, EstimateRequest, LayersRequest, PatchRequest, TextSpec
    from models.database import CreditBalance

    db, user = api_db

    def no_worker(model_id):
        raise AssertionError("a GPU was started")

    monkeypatch.setattr(api.diff, "_get_inference_worker", no_worker)
    good = TextSpec(sample="prose", n=8)
    cases = [
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec(sample="nope")), 404),
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec(sample="prose", texts=["a", "b", "c", "d"])), 400),
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec()), 400),
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec(texts=["a", "b"])), 400),
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=TextSpec(sample="prose", n=65)), 400),
        # Only a pair with an instruct model has a chat template
        (run_layers_for, LayersRequest(base="nano", other=CUSTOM, texts=good, chat_template=True), 400),
        (run_layers_for, LayersRequest(base="mini", other=CUSTOM, texts=good), 400),
        (run_directions_for, DirectionsRequest(base="nano", other=CUSTOM, texts=good, layer=7), 400),
        # An Ozera pair's prompt is tokenized on the backend
        (run_patch_for, PatchRequest(base="nano", other=CUSTOM, prompt="word " * 70), 400),
        (run_patch_for, PatchRequest(base="nano", other=CUSTOM, prompt="   "), 400),
        (estimate_for, EstimateRequest(kind="patch", base="nano", other=CUSTOM), 400),
        (estimate_for, EstimateRequest(kind="layers", base="nano", other=CUSTOM), 400),
    ]
    for route, body, status in cases:
        with pytest.raises(HTTPException) as e:
            asyncio.run(route(db, user, body))
        assert e.value.status_code == status, (body, e.value.detail)
    # An open-weight pair's sizes are estimates (its tokenizer is only on the workers), dearer on the larger tier
    small = asyncio.run(estimate_for(db, user, EstimateRequest(kind="layers", base="smollm-135m", other="smollm-135m-it", texts=good, chat_template=True)))
    large = asyncio.run(estimate_for(db, user, EstimateRequest(kind="directions", base="qwen3-4b", other="qwen3-4b-it", texts=good)))
    assert not small["checked"] and small["within_limit"] and large["within_limit"] and large["estimated_seconds"] > small["estimated_seconds"]
    patch = asyncio.run(estimate_for(db, user, EstimateRequest(kind="patch", base="gemma-3-1b", other="gemma-3-1b-it", prompt="The capital of France is")))
    assert patch["num_texts"] == 1 and patch["within_limit"]

    db.query(CreditBalance).filter(CreditBalance.user_id == user.id).update({"balance_usd": 0})
    db.commit()
    with pytest.raises(HTTPException) as e:
        asyncio.run(run_layers_for(db, user, LayersRequest(base="nano", other=CUSTOM, texts=good)))
    assert e.value.status_code == 402 and not _charges(db, user)


def test_the_assistant_runs_the_three_views(api_db):
    import json

    from services import assistant_runs, assistant_tools
    from services.assistant_tools import ToolContext
    from tests.test_attribution import _quote_and_run

    db, user = api_db
    args = {"model": CUSTOM, "sample": "instructions", "count": 8, "texts": [], "seed": 0, "chat_template": False}
    quoted, layers = _quote_and_run("diff_layers", user.id, args)
    assert dict(quoted.details)["Pair"] == f"Ozera Nano vs {CUSTOM}" and dict(quoted.details)["Texts"] == "Instructions (8)" and quoted.estimate > 0
    summary = layers.content
    assert summary["pair"] == {"base": "nano", "other": CUSTOM} and len(summary["distance"]) == 7 and summary["largest_distance"]["position"] >= 3
    assert summary["halves_differ_by_at_most"]["cka"] is not None and len(summary["most_divergent_tokens"]) == 8
    assert layers.charged == pytest.approx(layers.result["data"]["run"]["cost"]) and layers.result["page"] == "diff"
    assert layers.result["data"]["texts"]["sample"] == "instructions"
    assert len(json.dumps(summary)) < 6000, "the summary fits a tool result"

    quoted, found = _quote_and_run("diff_directions", user.id, {**args, "layer": 3})
    assert dict(quoted.details)["At"] == "layer 2's output"
    assert found.content["components"][0]["closed"] is not None and "beats_every_control" in found.content["components"][0]
    assert "closed" not in found.content["components"][-1], "only the largest components are tested"
    assert found.result["data"]["response"]["layer"] == 3 and len(json.dumps(found.content)) < 6000

    quoted, patched = _quote_and_run("diff_patch", user.id, {"model": "nano", "prompt": "Explain how the tides work", "donor": "other", "kind": "resid", "chat_template": False})
    assert dict(quoted.details)["From"] == CUSTOM and dict(quoted.details)["Into"] == "Ozera Nano"
    assert len(patched.content["lowest_kl_cells"]) == 10 and len(patched.content["all_positions_by_layer"]) == 6
    assert patched.result["data"]["response"]["kind"] == "resid"

    # The user's own texts, and what can't run
    _, own = _quote_and_run("diff_layers", user.id, {**args, "sample": "", "count": 0, "texts": ["One two.", "Three four five.", "Six.", "Seven eight."]})
    assert own.content["texts"] == "Your texts" and own.content["num_texts"] == 4
    ctx = ToolContext(user.id)
    for name, bad in (
        ("diff_layers", {**args, "texts": ["a", "b", "c", "d"]}),
        ("diff_layers", {**args, "sample": "nope"}),
        ("diff_layers", {**args, "model": "mini"}),
        ("diff_layers", {**args, "chat_template": True}),
        ("diff_layers", {**args, "count": 500}),
        ("diff_directions", {**args, "layer": 9}),
        ("diff_patch", {"model": CUSTOM, "prompt": "", "donor": "other", "kind": "resid", "chat_template": False}),
        ("diff_patch", {"model": CUSTOM, "prompt": "Hello there", "donor": "nobody", "kind": "resid", "chat_template": False}),
    ):
        outcome = asyncio.run(assistant_runs.quote(name, ctx, bad))
        assert getattr(outcome, "error", False), (name, bad)

    # Filling the page: a model that has a pair
    filled = asyncio.run(assistant_tools.execute("set_inputs", ctx, {"page": "diff", "fields": [
        {"control": "model", "value": CUSTOM}, {"control": "sample", "value": "prose"}, {"control": "layer", "value": "3"},
        {"control": "texts", "value": "One.\nTwo.\nThree.\nFour."},
    ]}))
    assert not filled.error and filled.ui["fields"] == {"model": CUSTOM, "sample": "prose", "layer": 3, "texts": "One.\nTwo.\nThree.\nFour."}
    refused = asyncio.run(assistant_tools.execute("set_inputs", ctx, {"page": "diff", "fields": [{"control": "model", "value": "mini"}]}))
    assert refused.error and "custom model" in refused.content["error"]


def test_a_layer_profile_is_drawn_as_lines():
    from api.export import DataFigureRequest, export_data_figure

    user = SimpleNamespace(id=1)
    series = [
        {"label": "All texts", "values": [0.0, 0.1, 0.3, 0.25]},
        {"label": "Half A", "values": [0.0, 0.11, 0.31, None], "style": "faint"},
        {"label": "Chance", "values": [0.05, 0.05, 0.05, 0.05], "style": "dashed"},
    ]
    figure = DataFigureRequest(kind="lines", series=series, column_labels=["emb", "0", "1", "2"], y_label="Distance")
    assert asyncio.run(export_data_figure(figure, user)).body[:4] == b"\x89PNG"
    vector = DataFigureRequest(kind="lines", series=series, column_labels=["emb", "0", "1", "2"], config={"format": "svg"})
    assert b"<svg" in asyncio.run(export_data_figure(vector, user)).body
    for bad in (DataFigureRequest(kind="lines"), DataFigureRequest(kind="lines", series=series, column_labels=["a"]),
                DataFigureRequest(kind="lines", series=[series[0], {"label": "x", "values": [1.0]}])):
        with pytest.raises(HTTPException):
            asyncio.run(export_data_figure(bad, user))


def test_massive_tokens_are_left_out():
    """A few tokens with activations many times the rest's don't decide the statistics."""
    from core.diff.profile import ordinary_tokens

    generator = torch.Generator().manual_seed(0)
    a = torch.randn(200, 32, generator=generator)
    b = a + 0.1 * torch.randn(200, 32, generator=generator)
    assert bool(ordinary_tokens(a, b).all())
    a[:5] *= 8
    assert ordinary_tokens(a, b).tolist() == [False] * 5 + [True] * 195, "massive in either model"
    # A distribution that is simply wide has no massive tokens, however far its tail reaches
    wide = a[5:] * torch.linspace(0.3, 2.6, 195)[:, None]
    assert bool(ordinary_tokens(wide, wide).all())
    assert bool(ordinary_tokens(a[:10], b[:10]).all()), "too few tokens to tell"

    stats = StreamSimilarity(1, 32, False, torch.device("cpu"))
    keep = ordinary_tokens(a, b)
    stats.add(0, 0, a[keep], b[keep])
    assert stats.alignment()["cka"][0] > 0.95
