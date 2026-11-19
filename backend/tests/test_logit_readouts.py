"""
The Generate page's logit lens and logit attribution (core.lens), as INSTRUMENTS_PLAN.md's
Phase 1 defines done: the lens's last row is the model's own top-k at every position, and each
position's parts sum to its logit. Run from backend/: python -m pytest tests/test_logit_readouts.py

The open-source architectures (Llama, Qwen3, Gemma 3) are tested on small random models when
transformers is installed (it's only in the Modal images).
"""

import dataclasses
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lens import Capture, LensError, LensModel, attribute, frozen_norm, ozera_readouts, read_rows, top_tokens  # noqa: E402
from core.tensor_codec import to_float32  # noqa: E402
from core.transformer.config import TransformerConfig  # noqa: E402
from core.transformer.model_torch import TransformerLM  # noqa: E402


class _Tokenizer:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def _ozera_model() -> TransformerLM:
    """A small Ozera model whose norms, biases and logits aren't near their initial values."""
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=300, d_model=48, num_layers=3, num_heads=4, d_ff=96, max_seq_len=64, initializer_range=0.3,
    )
    model = TransformerLM(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0, 0.3)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(0, 0.2)
    return model


@pytest.fixture(scope="module")
def ozera():
    model = _ozera_model()
    token_ids = torch.randint(0, 300, (1, 24))
    with torch.no_grad():
        logits, _, activations = model.forward(token_ids, return_attention=True, capture_activations=True)
    tokens = token_ids[0].tolist()
    lens_model = LensModel.for_ozera(model, _Tokenizer())
    capture = Capture.from_ozera(model, activations, activations["logits"], tokens)
    return model, tokens, logits[0], activations, lens_model, capture


def _check_lens(rows: dict, logits: torch.Tensor, tokens: list[int]):
    top = top_tokens(logits.float(), 20)
    k = rows["top_ids"].shape[-1]
    assert torch.equal(rows["top_ids"][-1], top[:, :k]), "the last row is the model's own top-k"
    probs = torch.softmax(logits.float(), dim=-1)
    assert torch.equal(rows["top_probs"][-1], probs.gather(1, top[:, :k]))
    # A position's target is the next token; the last position's is its final (top) token
    assert rows["target_ids"].tolist()[:-1] == tokens[1:]
    assert rows["target_ids"][-1] == rows["final_ids"][-1] == top[-1, 0]
    assert torch.equal(rows["final_ids"], top[:, 0])
    assert bool((rows["final_rank"][-1] == 1).all())
    assert torch.allclose(rows["target_prob"][-1], probs.gather(1, rows["target_ids"][:, None])[:, 0])


def _check_attribution(parts: dict, tolerance: float):
    total = parts["embed"] + parts["attn"].sum(dim=1) + parts["mlp"].sum(dim=1) + parts["bias"]
    assert (total - parts["logit"]).abs().max() < tolerance, "the parts sum to the logit"
    heads = parts["heads"].sum(dim=-1) + parts["attn_bias"]
    assert (heads - parts["attn"]).abs().max() < tolerance, "the heads (and the projection's bias) sum to the attention output"


def test_ozera_lens_last_row_is_the_models_top_k(ozera):
    model, tokens, logits, _, lens_model, capture = ozera
    # The lens's reading of the last layer is what the model computes
    assert torch.allclose(lens_model.logits(capture.resid_post[-1]), logits, atol=1e-5)
    rows = read_rows(lens_model, capture)
    assert rows["top_ids"].shape == (model.config.num_layers + 1, len(tokens), 5)
    _check_lens(rows, logits, tokens)


def test_ozera_parts_sum_to_the_logit(ozera):
    model, _, logits, _, lens_model, capture = ozera
    rows = read_rows(lens_model, capture)
    parts = attribute(lens_model, capture, rows["target_ids"])
    assert parts["heads"].shape == (len(capture.token_ids), model.config.num_layers, model.config.num_heads)
    assert torch.allclose(parts["logit"], logits.gather(1, rows["target_ids"][:, None])[:, 0])
    assert parts["logit"].abs().max() > 1, "the test's logits aren't trivially small"
    assert parts["bias"].abs().max() > 0.01 and parts["attn_bias"].abs().max() > 0.01
    _check_attribution(parts, 1e-4)


def test_ozera_heads_are_recomputed_exactly(ozera):
    model, _, _, _, _, capture = ozera
    for block, heads, attn_out in zip(model.blocks, capture.head_outputs, capture.attn_out):
        assert torch.allclose(block.attention.Wo(heads), attn_out, atol=1e-6)


def test_top_tokens_break_ties_by_token_id():
    torch.manual_seed(0)
    # Half-precision logits tie; argmax takes the lowest ID among them
    logits = (torch.randn(64, 2000).to(torch.bfloat16).float() * 16)
    ids = top_tokens(logits, 5)
    assert torch.equal(ids[:, 0], logits.argmax(dim=-1))
    for row in range(logits.shape[0]):
        expected = sorted(range(logits.shape[1]), key=lambda j: (-logits[row, j].item(), j))[:5]
        assert ids[row].tolist() == expected
    assert top_tokens(torch.tensor([[0.0, -0.0, 1.0, 1.0]]), 9).tolist() == [[2, 3, 0, 1]]


def test_frozen_norms():
    torch.manual_seed(0)
    x = torch.randn(5, 16) * 3

    layer_norm = torch.nn.LayerNorm(16)
    with torch.no_grad():
        layer_norm.weight.uniform_(0.5, 1.5)
        layer_norm.bias.normal_()
    norm = frozen_norm(layer_norm)
    assert norm.centers and torch.allclose(norm.apply(x), layer_norm(x), atol=1e-5)
    # A sum of parts passes through the frozen norm part by part
    directions = torch.randn(5, 16)
    parts = [torch.randn(5, 16) for _ in range(3)]
    whole = sum(parts)
    pulled = norm.pull_back(directions, whole)
    shares = sum((part * pulled).sum(dim=-1) for part in parts) + norm.bias_share(directions)
    assert torch.allclose(shares, (layer_norm(whole) * directions).sum(dim=-1), atol=1e-4)

    class Rms(torch.nn.Module):
        """RMSNorm as Llama and Qwen write it (plus=0) and as Gemma does (plus=1)."""

        def __init__(self, plus: float):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(16) + 0.5)
            self.eps = 1e-6
            self.plus = plus

        def forward(self, hidden):
            return hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps) * (self.plus + self.weight)

    for plus in (0.0, 1.0):
        module = Rms(plus)
        norm = frozen_norm(module)
        assert not norm.centers and norm.bias is None
        assert torch.allclose(norm.gain, plus + module.weight)
        assert torch.allclose(norm.apply(x), module(x), atol=1e-5)

    with pytest.raises(LensError):
        frozen_norm(torch.nn.Linear(16, 16))
    with pytest.raises(LensError):
        frozen_norm(torch.nn.Tanh())


def test_readouts_are_stored_and_served(ozera):
    from inference.activation_store import ActivationStore

    model, tokens, _, activations, _, _ = ozera
    readouts = ozera_readouts(model, _Tokenizer(), activations, activations["logits"], tokens)
    assert set(readouts) == {"logit_lens", "logit_attribution"}

    store = ActivationStore()
    activation_id = store.store_activations({**activations, **readouts}, tokens, "prompt", "nano", user_id=1)

    info = store.get_activation_summary(activation_id, 1)["tensor_info"]
    assert info["logit_lens"] == {"seq_len": len(tokens), "rows": 4, "k": 5}
    assert info["logit_attribution"] == {"seq_len": len(tokens), "num_layers": 3, "num_heads": 4}

    for name in ("logit_lens", "logit_attribution"):
        served = store.get_tensor_activation(activation_id, name, 1)
        json.dumps(served, allow_nan=False)  # No raw bytes or NaNs reach the client
        assert served["data"]["seq_len"] == len(tokens)
        assert store.get_tensor_activation(activation_id, name, 2) is None, "other users' don't exist"
    json.dumps(store.get_activations(activation_id, 1), allow_nan=False)

    lens = store.get_readout(activation_id, "logit_lens", 1)
    assert lens["top_ids"].shape == (4, len(tokens), 5)
    # Every token the lens names has its text
    assert {str(int(i)) for i in lens["top_ids"].reshape(-1)} | {str(i) for i in lens["target_ids"]} <= set(lens["tokens"])
    assert store.get_readout(activation_id, "top_k_logits", 1) is None

    attribution = store.get_readout(activation_id, "logit_attribution", 1)
    total = attribution["embed"] + attribution["attn"].sum(axis=1) + attribution["mlp"].sum(axis=1) + attribution["bias"]
    assert abs(total - attribution["logit"]).max() < 1e-4


def test_the_assistant_reads_both_where_the_model_generated(ozera):
    import asyncio
    from types import SimpleNamespace

    from inference.activation_store import get_activation_store
    from services.assistant_tools import DIRECT_PARTS, get_activation

    model, tokens, _, activations, lens_model, capture = ozera
    readouts = ozera_readouts(model, _Tokenizer(), activations, activations["logits"], tokens)
    prompt_tokens = 20
    activation_id = get_activation_store().store_activations(
        {**activations, **readouts}, tokens, "prompt", "nano", user_id=901,
        metadata={"prompt_tokens": prompt_tokens, "decoded_tokens": [f"<{t}>" for t in tokens]},
    )
    outcome = asyncio.run(get_activation(None, SimpleNamespace(id=901), None, {"activation_id": activation_id}))
    predictions = outcome.content["predictions"]
    assert len(predictions) == len(tokens)
    assert all("settles" not in p and "wrote" not in p for p in predictions[:prompt_tokens - 1])

    rows = read_rows(lens_model, capture)
    parts = attribute(lens_model, capture, rows["target_ids"])
    for position in predictions[prompt_tokens - 1:]:
        i = position["pos"]
        ranks = rows["target_rank"][:, i].tolist()
        if ranks[-1] != 1:
            assert position["settles"] is None
        else:
            row = 0 if position["settles"] == "emb" else int(position["settles"][1:]) + 1
            assert all(rank == 1 for rank in ranks[row:]) and (row == 0 or ranks[row - 1] != 1)
        assert len(position["wrote"]) == DIRECT_PARTS
        largest = max(parts["embed"][i].abs(), parts["mlp"][i].abs().max(), parts["heads"][i].abs().max())
        assert abs(abs(position["wrote"][0][1]) - float(largest)) < 0.006
    # The last position has no next token: its top prediction stands in, so it always settles
    assert predictions[-1]["next"] is None and predictions[-1]["settles"] is not None
    assert "settles" in outcome.content["note"] and "wrote" in outcome.content["note"]


def test_a_model_that_cant_be_read_leaves_the_readouts_out(ozera, capsys):
    model, tokens, _, activations, _, _ = ozera
    broken = {**activations, "layers": activations["layers"][:-1]}
    assert ozera_readouts(model, _Tokenizer(), broken, activations["logits"], tokens) == {}
    assert "skipped" in capsys.readouterr().out


# ---- Open-source architectures, on small random models ----


class _FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 1
    unk_token_id = 0

    def __call__(self, text, return_tensors=None, **kwargs):
        from transformers import BatchEncoding

        ids = [2] + [3 + (ord(c) % 90) for c in text]
        return BatchEncoding({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(f"<{i}>" for i in ids)

    def convert_tokens_to_ids(self, token):
        return 0


def _open_source_loader(family: str):
    transformers = pytest.importorskip("transformers")
    from core.open_source.gemma import GemmaLoader
    from core.open_source.qwen import QwenLoader
    from core.open_source.smollm import SmolLMLoader

    shape = dict(vocab_size=120, hidden_size=48, intermediate_size=96, num_hidden_layers=3, max_position_embeddings=128)
    if family == "llama":
        config = transformers.LlamaConfig(num_attention_heads=6, num_key_value_heads=2, **shape)
        model, loader = transformers.LlamaForCausalLM(config), SmolLMLoader("smollm-135m", device="cpu")
    elif family == "qwen":
        # Qwen3's head width isn't hidden_size / heads
        config = transformers.Qwen3Config(num_attention_heads=4, num_key_value_heads=2, head_dim=16, **shape)
        model, loader = transformers.Qwen3ForCausalLM(config), QwenLoader("qwen3-0.6b", device="cpu")
    else:
        # Gemma normalizes each branch's output before adding it to the residual stream
        config = transformers.Gemma3TextConfig(num_attention_heads=4, num_key_value_heads=1, head_dim=16, sliding_window=64, **shape)
        model, loader = transformers.Gemma3ForCausalLM(config), GemmaLoader("gemma-3-270m", device="cpu")

    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() > 1:
                parameter.mul_(8.0)
            else:
                parameter.add_(torch.randn_like(parameter) * 0.3)
    loader.model, loader.tokenizer = model.eval(), _FakeTokenizer()
    loader.config = dataclasses.replace(
        loader.config, num_layers=config.num_hidden_layers, num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads, hidden_dim=config.hidden_size, vocab_size=config.vocab_size,
        is_instruct=False,
    )
    return loader


@pytest.mark.parametrize("family", ["llama", "qwen", "gemma"])
def test_open_source_capture_has_both_readouts(family):
    loader = _open_source_loader(family)
    result = loader.generate_with_activations(
        "hello there world", max_new_tokens=6, temperature=0.0, do_sample=False, stop_at_eos=False,
    )
    activations = result["activations"]
    lens, attribution, top = activations["logit_lens"], activations["logit_attribution"], activations["top_k_logits"]

    # The last row is the model's own top-k, at every position
    k = lens["k"]
    assert to_float32(lens["top_ids"])[-1].astype(int).tolist() == [row[:k] for row in top["indices"]]
    assert abs(to_float32(lens["top_probs"])[-1] - torch.tensor(top["probabilities"])[:, :k].numpy()).max() < 1e-6
    # Greedy decoding: each generated token was the top prediction before it
    prompt_tokens = result["prompt_tokens"]
    assert lens["target_ids"][prompt_tokens - 1:-1] == result["tokens"][prompt_tokens:]
    assert to_float32(lens["target_rank"])[-1, prompt_tokens - 1:].tolist() == [1.0] * (len(result["tokens"]) - prompt_tokens + 1)

    parts = {
        key: torch.from_numpy(to_float32(attribution[key]).copy())
        for key in ("embed", "attn", "mlp", "heads", "attn_bias", "bias", "logit")
    }
    assert parts["logit"].abs().max() > 1
    _check_attribution(parts, 1e-4)

    # The capture pass's extra inputs don't reach the frontend, and aren't kept
    assert not any("attn_heads" in key or key == "resid_pre" for layer in activations["layers"] for key in layer)
    assert "resid_pre" not in activations and not loader._activations and not loader._hooks


def test_prompt_captures_dont_record_readout_inputs():
    loader = _open_source_loader("llama")
    captured = loader.capture_prompt_activations("hello there")
    assert not any(key.endswith("attn_heads") or key == "resid_pre" for key in captured["activations"])
