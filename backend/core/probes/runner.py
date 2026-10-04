"""
Running texts through a model for probes: pooled residual-stream features for training, and
per-token probe scores.

A target wraps a loaded model (Ozera or open-source) with what probing needs from it: its
decoder layers, how to tokenize a text and where its read span is, and a forward pass.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .budget import MAX_SEQUENCE_TOKENS, ProbeInputError
from .capture import ResidualReader
from .pooling import POOLINGS, pool_hidden, pool_scores

READ_SPANS = ("text", "prompt")

# Padded tokens per forward pass when collecting features
TOKENS_PER_BATCH = 8192


@dataclass
class EncodedText:
    """A text's token IDs and its read span [start, end): the positions probes pool over."""

    ids: list[int]
    start: int
    end: int


class ProbeTarget:
    """A model as probing sees it (see the subclasses)."""

    layers: Sequence[torch.nn.Module]
    hidden_dim: int
    parameters: int
    device: torch.device
    max_tokens: int
    pad_id: int
    # How core.patching's engine runs the model: its loader (or the model itself for Ozera
    # models), its type, and the tokenizer the engine needs for Ozera models
    model_type: str
    model_loader: object
    engine_tokenizer: object = None

    def lock(self):
        """Held while the model runs with probe hooks attached."""
        return nullcontext()

    def encode(self, text: str, chat_template: bool, read_span: str) -> EncodedText:
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        raise NotImplementedError

    def token_text(self, token_id: int) -> str:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        """A generated sequence's text (without special tokens)."""
        raise NotImplementedError

    def attention_branch(self, layer: int) -> torch.nn.Module:
        """The module whose output a decoder layer adds to the residual stream as its attention branch."""
        raise NotImplementedError

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-token logits from the last layer's output (applying the final norm and unembedding)."""
        raise NotImplementedError

    def token_logprobs(self, ids: Sequence[int], start: int) -> list[float]:
        """
        Log-probability of each token from position start on, given the tokens before it.

        Args:
            ids: A whole sequence's token IDs
            start: First position scored (>= 1)
        """
        raise NotImplementedError


def _gather_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Log-probabilities [n] of targets [n] under logits [n, vocab]."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return logprobs.gather(1, targets[:, None])[:, 0].tolist()


class OzeraTarget(ProbeTarget):
    """An Ozera model (base or custom). Texts are read as plain text over all their tokens."""

    model_type = "ozera"

    def __init__(self, model, config, tokenizer):
        self.model = model
        self.model_loader = model
        self.engine_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.layers = list(model.blocks)
        self.hidden_dim = config.d_model
        self.parameters = config.count_parameters()
        self.device = next(model.parameters()).device
        self.max_tokens = min(MAX_SEQUENCE_TOKENS, config.max_seq_len)
        self.pad_id = 0

    def encode(self, text: str, chat_template: bool, read_span: str) -> EncodedText:
        if chat_template:
            raise ProbeInputError("Ozera models have no chat template")
        ids = self.tokenizer.encode(text)
        return EncodedText(ids, 0, len(ids))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        # Attention is causal and padding is on the right, so padding never reaches real tokens
        self.model(input_ids)

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def attention_branch(self, layer: int) -> torch.nn.Module:
        return self.model.blocks[layer].attention

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.model.ln_f(hidden))

    def token_logprobs(self, ids: Sequence[int], start: int) -> list[float]:
        with torch.no_grad():
            logits, _, _ = self.model(torch.tensor([list(ids)], device=self.device))
        targets = torch.tensor(list(ids[start:]), device=self.device)
        return _gather_logprobs(logits[0, start - 1:-1], targets)


class OpenSourceTarget(ProbeTarget):
    """
    An open-source (HuggingFace) model, through its loader.

    Base models read raw text. Instruct models can also read the text as a user message in
    their chat template, where the read span is the text's own tokens ("text") or runs on to
    the end of the template, where the model starts its reply ("prompt").
    """

    model_type = "open_source"

    def __init__(self, loader):
        self.loader = loader
        self.model_loader = loader
        self.tokenizer = loader.tokenizer
        self.decoder = loader.model.model
        self.layers = list(self.decoder.layers)
        self.hidden_dim = loader.model.config.hidden_size
        self.parameters = loader.config.parameters
        self.device = next(loader.model.parameters()).device
        self.max_tokens = MAX_SEQUENCE_TOKENS
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else (self.tokenizer.eos_token_id or 0)

    def lock(self):
        # Hooks on the shared model must not see other requests' forward passes
        return self.loader.lock

    def encode(self, text: str, chat_template: bool, read_span: str) -> EncodedText:
        if chat_template:
            if not self.loader.config.is_instruct:
                raise ProbeInputError(f"{self.loader.config.display_name} has no chat template")
            rendered = self.tokenizer.apply_chat_template(
                self.loader.chat_messages(text),
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            # apply_chat_template tokenizes its rendering without adding special tokens
            encoding = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            char_start = rendered.rfind(text)
            char_end = char_start + len(text)
        else:
            encoding = self.tokenizer(text, return_offsets_mapping=True)
            char_start, char_end = 0, len(text)

        ids = list(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        in_text = [
            i for i, (a, b) in enumerate(offsets)
            if b > a and (char_start < 0 or (a < char_end and b > char_start))
        ]
        if not in_text:
            in_text = list(range(len(ids)))
        start, end = in_text[0], in_text[-1] + 1
        if read_span == "prompt":
            end = len(ids)
        return EncodedText(ids, start, end)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        # The decoder stack alone: the reader stops the pass after the last layer it reads
        self.decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def attention_branch(self, layer: int) -> torch.nn.Module:
        from core.patching.engine import attention_branch_end

        return attention_branch_end(self.layers[layer])

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        model = self.loader.model
        logits = model.lm_head(self.decoder.norm(hidden))
        cap = getattr(model.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def token_logprobs(self, ids: Sequence[int], start: int) -> list[float]:
        input_ids = torch.tensor([list(ids)], device=self.device)
        # Logits for the positions that predict tokens start.. (the last position predicts nothing)
        keep = len(ids) - start + 1
        with torch.no_grad():
            logits = self.loader.model(input_ids=input_ids, use_cache=False, logits_to_keep=keep).logits
        targets = torch.tensor(list(ids[start:]), device=self.device)
        return _gather_logprobs(logits[0, :-1], targets)


def encode_texts(target: ProbeTarget, texts: Sequence[str], chat_template: bool, read_span: str) -> list[EncodedText]:
    """
    Tokenize texts for a target.

    Raises:
        ProbeInputError: a sequence is longer than the target can read
    """
    if read_span not in READ_SPANS:
        raise ProbeInputError(f"Unknown read span: {read_span}")
    encoded = [target.encode(text, chat_template, read_span) for text in texts]
    too_long = [i for i, e in enumerate(encoded) if len(e.ids) > target.max_tokens]
    if too_long:
        raise ProbeInputError(
            f"{len(too_long)} example(s) are longer than {target.max_tokens} tokens "
            f"(the first is example {too_long[0] + 1}); shorten or remove them"
        )
    return encoded


def _batches(encoded: Sequence[EncodedText], tokens_per_batch: int) -> list[list[int]]:
    """Example indices grouped into batches of similar length, padded to at most tokens_per_batch."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids))
    batches: list[list[int]] = []
    for i in order:
        # Sorted by length, so i is the longest in any batch it joins
        if batches and (len(batches[-1]) + 1) * len(encoded[i].ids) <= tokens_per_batch:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


@dataclass
class Batch:
    """Some examples padded into one forward pass, on the target's device."""

    idx: torch.Tensor  # [batch] the examples' indices
    input_ids: torch.Tensor  # [batch, seq], padded on the right
    attention_mask: torch.Tensor  # [batch, seq]
    starts: torch.Tensor  # [batch] each read span's first position
    ends: torch.Tensor  # [batch] one past each read span's last position


def forward_batches(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    tokens_per_batch: int = TOKENS_PER_BATCH,
) -> Iterator[Batch]:
    """Batches covering every example once, grouped by length."""
    for idx in _batches(encoded, tokens_per_batch):
        length = max(len(encoded[i].ids) for i in idx)
        input_ids = torch.full((len(idx), length), target.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(idx), length), dtype=torch.long)
        for row, i in enumerate(idx):
            input_ids[row, :len(encoded[i].ids)] = torch.tensor(encoded[i].ids)
            attention_mask[row, :len(encoded[i].ids)] = 1
        yield Batch(
            idx=torch.tensor(idx, device=target.device),
            input_ids=input_ids.to(target.device),
            attention_mask=attention_mask.to(target.device),
            starts=torch.tensor([encoded[i].start for i in idx], device=target.device),
            ends=torch.tensor([encoded[i].end for i in idx], device=target.device),
        )


def collect_features(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    tokens_per_batch: int = TOKENS_PER_BATCH,
    poolings: tuple[str, ...] = POOLINGS,
) -> dict[str, torch.Tensor]:
    """
    Pooled residual-stream features of every example at every position, for each pooling.

    Caller holds target.lock(). Hidden states are pooled on the device as each layer
    produces them, so only [positions, n, d] per pooling is kept.

    Returns:
        {pooling: [layers + 1, n, d] float32}, position 0 being the embeddings
    """
    n = len(encoded)
    positions = len(target.layers) + 1
    features = {
        pooling: torch.empty(positions, n, target.hidden_dim, dtype=torch.float32, device=target.device)
        for pooling in poolings
    }

    current: dict = {}

    def on_hidden(position: int, hidden: torch.Tensor) -> None:
        batch = current["batch"]
        for pooling, values in pool_hidden(hidden, batch.starts, batch.ends, poolings).items():
            features[pooling][position, batch.idx] = values
        current["read"] += 1

    with ResidualReader(target.layers, None, on_hidden) as reader:
        for batch in forward_batches(target, encoded, tokens_per_batch):
            current.update(batch=batch, read=0)
            reader.run(lambda: target.forward(batch.input_ids, batch.attention_mask))
            if current["read"] != positions:
                raise RuntimeError(f"Read {current['read']} of the model's {positions} residual stream positions")

    return features


def score_texts(
    target: ProbeTarget,
    encoded: Sequence[EncodedText],
    layer: int,
    weights: Sequence[float],
    bias: float,
    pooling: str,
) -> list[dict]:
    """
    A probe's score at every token of each text, and its pooled score.

    Caller holds target.lock().

    Args:
        encoded: The texts, from encode_texts (with the probe's chat template and read span)
        layer: The decoder layer whose output the probe reads
        weights, bias: The probe, on raw activations
        pooling: How the probe pools ("last", "mean" or "max")

    Returns:
        Per text: {"tokens": [str], "scores": [float], "span": [start, end], "score": float}
    """
    if not 0 <= layer < len(target.layers):
        raise ProbeInputError(f"Layer {layer} is out of range for this model ({len(target.layers)} layers)")
    if pooling not in POOLINGS:
        raise ProbeInputError(f"Unknown pooling: {pooling}")
    if len(weights) != target.hidden_dim:
        raise ProbeInputError(f"The probe has {len(weights)} weights; this model's activations have {target.hidden_dim}")

    w = torch.tensor(weights, dtype=torch.float32, device=target.device)
    read = {}

    def on_hidden(position: int, hidden: torch.Tensor) -> None:
        read["hidden"] = hidden

    results = []
    with ResidualReader(target.layers, [layer + 1], on_hidden) as reader:
        for e in encoded:
            ids = torch.tensor([e.ids], device=target.device)
            reader.run(lambda: target.forward(ids, torch.ones_like(ids)))
            hidden = read.pop("hidden")[0].float()
            token_scores = hidden @ w + bias
            results.append({
                "tokens": [target.token_text(i) for i in e.ids],
                "scores": token_scores.tolist(),
                "span": [e.start, e.end],
                "score": pool_scores(token_scores, e.start, e.end, pooling, hidden, w, bias),
            })
    return results
