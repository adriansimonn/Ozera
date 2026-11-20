"""
Tracing a prompt on the circuits worker: the attribution graph of one target token at the
prompt's last position (circuit-tracer's attribute), pruned, with what the page shows of each
feature node: the tokens its decoder promotes and suppresses, and where it fires in the prompt.
"""

import time
from typing import Callable, Optional

import torch

from core.circuits import graph as graphs
from core.circuits.budget import (
    EDGE_THRESHOLD,
    MAX_FEATURE_NODES,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_PROMPT_TOKENS,
    NODE_THRESHOLD,
    PROMOTED_TOKENS,
    SUPPRESSED_TOKENS,
    TOP_PREDICTIONS,
    CircuitInputError,
    batch_size,
    traced_features,
)

Progress = Callable[[str, float], None]


def _no_progress(stage: str, fraction: float) -> None:
    pass


def encode_prompt(model, config, prompt: str) -> torch.Tensor:
    """
    A prompt's token IDs as the model is traced on them: a base model's text after a BOS, an
    instruct model's as the user's message in its chat template (as the other pages read it).

    Raises:
        CircuitInputError: the prompt has too many tokens
    """
    if config.is_instruct:
        messages = [{"role": "user", "content": prompt}]
        if config.system_prompt:
            messages.insert(0, {"role": "system", "content": config.system_prompt})
        ids = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        tokens = model.ensure_tokenized(torch.tensor(ids, dtype=torch.long))
    else:
        tokens = model.ensure_tokenized(prompt)
    if tokens.numel() > MAX_PROMPT_TOKENS:
        raise CircuitInputError(f"The prompt is {tokens.numel()} tokens on this model; graphs are of at most {MAX_PROMPT_TOKENS}")
    if tokens.numel() < 2:
        raise CircuitInputError("Write a prompt to trace")
    return tokens


def token_texts(tokenizer, ids) -> list[str]:
    return [tokenizer.decode([int(i)]) for i in ids]


def last_logits(logits: torch.Tensor) -> torch.Tensor:
    """The last position's logits from a model output of shape [1, positions, vocabulary] or [positions, vocabulary]."""
    return (logits[0, -1] if logits.ndim == 3 else logits[-1]).float()


def read_model(model, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(the last position's logits, every feature's activation as a sparse [layers, positions, features])."""
    logits, activations = model.get_activations(tokens, sparse=True)
    return last_logits(logits), activations


def reading(logits: torch.Tensor, token_id: int) -> dict:
    """A token's logit above the mean logit (what the graph attributes), its probability and its rank."""
    probabilities = torch.softmax(logits, dim=-1)
    return {
        "logit": float(logits[token_id] - logits.mean()),
        "prob": float(probabilities[token_id]),
        "rank": int((logits > logits[token_id]).sum()),
    }


def top_predictions(tokenizer, logits: torch.Tensor, count: int = TOP_PREDICTIONS) -> list[dict]:
    probabilities = torch.softmax(logits, dim=-1)
    values, ids = probabilities.topk(count)
    return [{"token": tokenizer.decode([int(i)]), "token_id": int(i), "prob": float(p)} for p, i in zip(values, ids)]


def choose_target(tokenizer, logits: torch.Tensor, target: Optional[str]) -> int:
    """The target token: the first token of the text given, or the model's top prediction."""
    if target is None or target == "":
        return int(logits.argmax())
    ids = tokenizer.encode(target, add_special_tokens=False)
    if not ids:
        raise CircuitInputError("The target has no tokens")
    return int(ids[0])


def _transcoders(model):
    """The model's transcoder set as a module (nnsight wraps the model's attributes)."""
    transcoders = model.transcoders
    return getattr(transcoders, "_module", transcoders)


def describe_features(model, nodes: list[dict], activations: torch.Tensor) -> None:
    """
    Add to each feature node the tokens its decoder writes toward ("promotes") and away from
    ("suppresses"), read through the unembedding, and its activation at every position ("fires").
    """
    features = [node for node in nodes if node["kind"] == graphs.FEATURE]
    if not features:
        return
    dense = activations.to_dense() if activations.is_sparse else activations
    unembed = model.unembed_weight
    transcoders = _transcoders(model)
    by_layer: dict[int, list[dict]] = {}
    for node in features:
        by_layer.setdefault(node["layer"], []).append(node)
    for layer, group in by_layer.items():
        ids = torch.tensor([node["feature"] for node in group], device=unembed.device)
        decoders = transcoders._get_decoder_vectors(layer, ids).to(unembed.dtype)
        fires = dense[layer][:, ids].T.float().cpu()
        for start in range(0, len(group), 64):
            written = (decoders[start:start + 64] @ unembed.T).float()
            top = written.topk(PROMOTED_TOKENS, dim=-1)
            bottom = (-written).topk(SUPPRESSED_TOKENS, dim=-1)
            for row, node in enumerate(group[start:start + 64]):
                node["promotes"] = token_texts(model.tokenizer, top.indices[row])
                node["suppresses"] = token_texts(model.tokenizer, bottom.indices[row])
                node["fires"] = [round(float(v), 3) for v in fires[start + row]]


def trace(model, config, transcoders, model_id: str, prompt: str, target: Optional[str] = None, progress: Progress = _no_progress) -> dict:
    """
    A prompt's attribution graph (see core.circuits.graph for its nodes and edges).

    Returns the graph as stored: the prompt's tokens, the target with the model's reading of it,
    the model's top predictions, the scores, the pruned nodes and edges.

    Raises:
        CircuitInputError: the prompt or target can't be traced
    """
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph

    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokens = encode_prompt(model, config, prompt)
    progress("Reading the prompt", 0.1)
    logits, activations = read_model(model, tokens)
    target_id = choose_target(model.tokenizer, logits, target)

    progress("Tracing", 0.2)
    positions = int(tokens.numel())
    active = int(activations._nnz()) if activations.is_sparse else int((activations != 0).sum())
    # Error nodes (one per layer and position), token nodes and the target
    others = (activations.shape[0] + 1) * positions + 1
    traced = attribute(
        tokens, model, attribution_targets=torch.tensor([target_id]),
        batch_size=batch_size(model_id, positions), max_feature_nodes=traced_features(active, others),
    )

    progress("Pruning", 0.8)
    adjacency = traced.adjacency_matrix.float().cpu()
    selected = traced.selected_features.cpu()
    layout = graphs.Layout(len(selected), traced.cfg.n_layers, traced.n_pos, len(traced.logit_targets))
    weights = torch.zeros(layout.size)
    weights[layout.logit_start:] = 1.0 / layout.n_logits
    pruned = prune_graph(traced, NODE_THRESHOLD, EDGE_THRESHOLD)
    packed = graphs.package(
        adjacency, layout, weights,
        features=traced.active_features.cpu()[selected],
        activations=traced.activation_values.float().cpu()[selected],
        node_mask=pruned.node_mask.cpu(), edge_mask=pruned.edge_mask.cpu(),
        max_nodes=MAX_GRAPH_NODES, max_edges=MAX_GRAPH_EDGES,
    )

    progress("Reading features", 0.9)
    describe_features(model, packed["nodes"], activations)
    token_ids = [int(t) for t in tokens]
    texts = token_texts(model.tokenizer, token_ids)
    target_reading = {"token": model.tokenizer.decode([target_id]), "token_id": target_id, **reading(logits, target_id)}
    for node in packed["nodes"]:
        if node["kind"] == graphs.TOKEN:
            node["token"] = texts[node["position"]]
        elif node["kind"] == graphs.LOGIT:
            node.update(id=graphs.node_id(graphs.LOGIT, feature=target_id), token=target_reading["token"], token_id=target_id,
                        prob=target_reading["prob"], position=len(token_ids) - 1)

    return {
        "version": graphs.GRAPH_VERSION,
        "model": model_id,
        "transcoders": {"id": transcoders.id, "name": transcoders.name, "reference": transcoders.reference, "width": transcoders.width},
        "prompt": prompt,
        "chat_template": bool(config.is_instruct),
        "tokens": texts,
        "token_ids": token_ids,
        "num_layers": layout.n_layers,
        "target": target_reading,
        "target_given": bool(target),
        "predictions": top_predictions(model.tokenizer, logits),
        "scores": graphs.scores(adjacency, layout, weights),
        "counts": {
            "active_features": int(traced.active_features.shape[0]),
            "traced_features": layout.n_features,
            **packed["pruned"],
        },
        "settings": {"node_threshold": NODE_THRESHOLD, "edge_threshold": EDGE_THRESHOLD, "max_feature_nodes": MAX_FEATURE_NODES},
        "nodes": packed["nodes"],
        "edges": packed["edges"],
        "seconds": round(time.time() - started, 2),
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
    }
