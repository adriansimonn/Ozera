"""
An attribution graph as Ozera stores and draws it, made from circuit-tracer's.

circuit-tracer's graph is a dense matrix of direct effects between nodes, in the order: the
traced feature nodes, an error node per layer and position (what a layer's transcoder failed to
reconstruct there), a token node per position (its embedding), then the target logits. Row t,
column s is the direct effect of source s on target t, with attention patterns and norm scales
frozen, so each node's inputs add up to its value (a feature's pre-activation, a logit minus the
mean logit).

A node's influence is the share of the target's attribution that flows through it: every
edge is taken as its share of its target's absolute inputs, and shares are multiplied along
paths and summed (circuit-tracer's compute_node_influence). Paths start at token and error
nodes, so those two kinds' influences add up to 1: the tokens' share is the replacement score.

Influence has no sign: a node that holds the target down counts as much as one that pushes it
up. A node's support is the same sum with each edge's sign kept, so paths that raise the target
and paths that lower it cancel: positive for a node that on balance pushes the target up,
negative for one that holds it down, and never larger in size than its influence.

Nothing here needs circuit-tracer: the functions take tensors.
"""

from typing import Optional

import torch

GRAPH_VERSION = 1

FEATURE, ERROR, TOKEN, LOGIT = "feature", "error", "token", "logit"


def node_id(kind: str, layer: Optional[int] = None, position: Optional[int] = None, feature: Optional[int] = None) -> str:
    """A node's ID: f:<layer>:<position>:<feature>, e:<layer>:<position>, t:<position>, or l:<token ID>."""
    if kind == FEATURE:
        return f"f:{layer}:{position}:{feature}"
    if kind == ERROR:
        return f"e:{layer}:{position}"
    if kind == TOKEN:
        return f"t:{position}"
    return f"l:{feature}"


def parse_feature_id(text: str) -> Optional[tuple[int, int, int]]:
    """(layer, position, feature) of a feature node's ID, or None if it isn't one."""
    parts = (text or "").split(":")
    if len(parts) != 4 or parts[0] != "f" or not all(p.isdigit() for p in parts[1:]):
        return None
    return int(parts[1]), int(parts[2]), int(parts[3])


class Layout:
    """Where each kind of node sits in a circuit-tracer adjacency matrix."""

    def __init__(self, n_features: int, n_layers: int, n_positions: int, n_logits: int):
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_positions = n_positions
        self.n_logits = n_logits
        self.error_start = n_features
        self.token_start = n_features + n_layers * n_positions
        self.logit_start = self.token_start + n_positions
        self.size = self.logit_start + n_logits

    def kind(self, index: int) -> str:
        if index < self.error_start:
            return FEATURE
        if index < self.token_start:
            return ERROR
        return TOKEN if index < self.logit_start else LOGIT


def normalized(adjacency: torch.Tensor) -> torch.Tensor:
    """Every edge as its share of its target's absolute inputs."""
    absolute = adjacency.abs()
    return absolute / absolute.sum(dim=1, keepdim=True).clamp(min=1e-10)


def node_influence(shares: torch.Tensor, logit_weights: torch.Tensor, max_steps: int = 1000) -> torch.Tensor:
    """
    Each node's influence on the weighted targets through every path (see the module docstring).

    Args:
        shares: normalized(adjacency)
        logit_weights: [nodes], zero except at the logit nodes
    """
    step = logit_weights @ shares
    total = step.clone()
    for _ in range(max_steps):
        if not step.any():
            return total
        step = step @ shares
        total += step
    raise RuntimeError("The graph's influence didn't settle: it has a cycle")


def signed(adjacency: torch.Tensor) -> torch.Tensor:
    """Every edge as its share of its target's absolute inputs, with its sign."""
    return adjacency / adjacency.abs().sum(dim=1, keepdim=True).clamp(min=1e-10)


def node_support(adjacency: torch.Tensor, logit_weights: torch.Tensor) -> torch.Tensor:
    """
    Each node's net push on the weighted targets through every path (see the module docstring):
    node_influence with the edges' signs kept.
    """
    return node_influence(signed(adjacency), logit_weights)


def scores(adjacency: torch.Tensor, layout: Layout, logit_weights: torch.Tensor) -> dict:
    """
    How much of the graph is features rather than error:

    - replacement: of the influence that reaches the targets from the graph's sources (token
      and error nodes), the share that starts at tokens;
    - completeness: over all nodes, weighted by influence, the share of a node's inputs that
      doesn't come from error nodes;
    - target_inputs: the target logits' direct inputs split by kind, as shares of their
      absolute sum and of their signed sum.
    """
    shares = normalized(adjacency)
    influence = node_influence(shares, logit_weights)
    errors = slice(layout.error_start, layout.token_start)
    tokens = slice(layout.token_start, layout.logit_start)
    token_influence = float(influence[tokens].sum())
    error_influence = float(influence[errors].sum())
    weight = influence + logit_weights
    non_error = 1 - shares[:, errors].sum(dim=1)

    rows = adjacency[layout.logit_start:]
    row_weights = logit_weights[layout.logit_start:]
    row_weights = row_weights / row_weights.sum().clamp(min=1e-10)
    parts = {FEATURE: slice(0, layout.error_start), ERROR: errors, TOKEN: tokens}
    absolute = {kind: float((rows[:, part].abs().sum(dim=1) * row_weights).sum()) for kind, part in parts.items()}
    signed = {kind: float((rows[:, part].sum(dim=1) * row_weights).sum()) for kind, part in parts.items()}
    total_absolute = sum(absolute.values()) or 1.0
    total_signed = sum(signed.values())
    return {
        "replacement": token_influence / max(token_influence + error_influence, 1e-10),
        "completeness": float((non_error * weight).sum() / weight.sum().clamp(min=1e-10)),
        "target_inputs": {
            "sum": total_signed,
            "absolute": {kind: value / total_absolute for kind, value in absolute.items()},
            "signed": {kind: (value / total_signed if abs(total_signed) > 1e-10 else 0.0) for kind, value in signed.items()},
        },
    }


def package(
    adjacency: torch.Tensor,
    layout: Layout,
    logit_weights: torch.Tensor,
    features: torch.Tensor,
    activations: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    max_nodes: int,
    max_edges: int,
) -> dict:
    """
    A pruned graph's nodes and edges.

    Args:
        features: [traced features, 3] of (layer, position, feature index), in the matrix's order
        activations: [traced features]
        node_mask, edge_mask: what pruning kept (circuit-tracer's prune_graph), over the matrix
        max_nodes: Feature and error nodes kept at most, the most influential first (token and
            logit nodes are always kept)
        max_edges: Edges kept at most, by their share of the targets' attribution

    Returns:
        {"nodes": [...], "edges": {"source", "target", "weight", "share"}, "pruned": {...}}:
        every node has its influence and its support; edges name nodes by their index in
        "nodes"; a weight is the direct effect, a share the part of the targets' attribution
        that flows through the edge.
    """
    shares = normalized(adjacency)
    influence = node_influence(shares, logit_weights)
    support = node_support(adjacency, logit_weights)

    keep = node_mask.clone().bool()
    inner = torch.arange(layout.size) < layout.token_start
    candidates = (keep & inner).nonzero().squeeze(-1)
    if len(candidates) > max_nodes:
        ranked = candidates[torch.argsort(influence[candidates], descending=True)]
        keep[ranked[max_nodes:]] = False
    kept = keep.nonzero().squeeze(-1)
    index_of = torch.full((layout.size,), -1, dtype=torch.long)
    index_of[kept] = torch.arange(len(kept))

    nodes = []
    for index in kept.tolist():
        kind = layout.kind(index)
        node: dict = {"kind": kind, "influence": float(influence[index]), "support": float(support[index])}
        if kind == FEATURE:
            layer, position, feature = (int(v) for v in features[index])
            node.update(id=node_id(FEATURE, layer, position, feature), layer=layer, position=position, feature=feature,
                        activation=float(activations[index]))
        elif kind == ERROR:
            layer, position = divmod(index - layout.error_start, layout.n_positions)
            node.update(id=node_id(ERROR, layer, position), layer=layer, position=position)
        elif kind == TOKEN:
            position = index - layout.token_start
            node.update(id=node_id(TOKEN, position=position), position=position)
        else:
            node.update(index=index - layout.logit_start, influence=float(logit_weights[index]), support=float(logit_weights[index]))
        nodes.append(node)

    mask = edge_mask.clone().bool()
    mask[~keep] = False
    mask[:, ~keep] = False
    targets, sources = mask.nonzero(as_tuple=True)
    edge_shares = shares[targets, sources] * (influence + logit_weights)[targets]
    total_edges = len(targets)
    if total_edges > max_edges:
        top = torch.argsort(edge_shares, descending=True)[:max_edges]
        targets, sources, edge_shares = targets[top], sources[top], edge_shares[top]
    order = torch.argsort(edge_shares, descending=True)
    targets, sources, edge_shares = targets[order], sources[order], edge_shares[order]

    kinds = [layout.kind(i) for i in node_mask.bool().nonzero().squeeze(-1).tolist()]
    return {
        "nodes": nodes,
        "edges": {
            "source": index_of[sources].tolist(),
            "target": index_of[targets].tolist(),
            "weight": [round(float(v), 5) for v in adjacency[targets, sources]],
            "share": [round(float(v), 6) for v in edge_shares],
        },
        "pruned": {
            # What pruning kept, before the caps above
            "feature_nodes": kinds.count(FEATURE),
            "error_nodes": kinds.count(ERROR),
            "edges": int(edge_mask.sum()),
            "kept_edges": len(order),
        },
    }


def summary(graph: dict) -> dict:
    """A stored graph in brief: what a job's row and the lists of graphs keep."""
    kinds = [node["kind"] for node in graph["nodes"]]
    return {
        "target": graph["target"],
        "tokens": len(graph["tokens"]),
        "replacement": graph["scores"]["replacement"],
        "completeness": graph["scores"]["completeness"],
        "error_inputs": graph["scores"]["target_inputs"]["absolute"][ERROR],
        "feature_nodes": kinds.count(FEATURE),
        "error_nodes": kinds.count(ERROR),
        "edges": len(graph["edges"]["source"]),
        "active_features": graph["counts"]["active_features"],
        "traced_features": graph["counts"]["traced_features"],
        "seconds": graph["seconds"],
    }
