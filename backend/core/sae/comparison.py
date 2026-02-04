"""
Feature alignment algorithms for comparing SAE features across models and layers.

Supports two comparison modes:
1. Same-space comparison: When SAEs share the same activation space (same model,
   different layers with same d_model), compare decoder directions directly.
2. Cross-space comparison: When SAEs have different activation spaces (different
   models, different dimensions), compare activation patterns on shared input.

Research questions addressed:
- Which concepts are shared between nano and mini?
- Which features are unique to the larger model?
- How do features emerge across layers?
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import torch

from .similarity_metrics import (
    cosine_similarity_matrix,
    activation_correlation_matrix,
    linear_cka,
)


@dataclass
class FeatureMatch:
    """A matched pair of features between two SAEs."""
    feature_a: int
    feature_b: int
    similarity: float
    method: str  # "decoder_cosine" or "activation_correlation"
    shared_tokens: List[str] = field(default_factory=list)
    label_a: str = ""
    label_b: str = ""


@dataclass
class ComparisonResult:
    """Full comparison result between two SAEs."""
    overall_similarity: float
    cka_score: float
    matched_features: int
    unmatched_a: int
    unmatched_b: int
    top_matches: List[FeatureMatch]
    divergent_features_a: List[int]
    divergent_features_b: List[int]
    similarity_matrix_sample: Optional[List[List[float]]] = None
    feature_indices_a: Optional[List[int]] = None
    feature_indices_b: Optional[List[int]] = None

    def to_dict(self) -> Dict:
        """Serialize for API response."""
        return {
            "overall_similarity": self.overall_similarity,
            "cka_score": self.cka_score,
            "matched_features": self.matched_features,
            "unmatched_a": self.unmatched_a,
            "unmatched_b": self.unmatched_b,
            "top_matches": [
                {
                    "feature_a": m.feature_a,
                    "feature_b": m.feature_b,
                    "similarity": m.similarity,
                    "shared_tokens": m.shared_tokens,
                    "label_a": m.label_a,
                    "label_b": m.label_b,
                }
                for m in self.top_matches
            ],
            "divergent_features_a": self.divergent_features_a,
            "divergent_features_b": self.divergent_features_b,
            "similarity_matrix_sample": self.similarity_matrix_sample,
            "feature_indices_a": self.feature_indices_a,
            "feature_indices_b": self.feature_indices_b,
        }


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def greedy_match(
    similarity_matrix: np.ndarray,
    threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """
    Greedy 1-to-1 matching using a similarity matrix.

    Iteratively selects the highest-similarity pair, removes both from
    consideration, and repeats.

    Args:
        similarity_matrix: (n_a, n_b) matrix of similarity scores
        threshold: Minimum similarity to count as a match

    Returns:
        List of (idx_a, idx_b, similarity) tuples
    """
    sim = similarity_matrix.copy()
    n_a, n_b = sim.shape
    matches = []
    used_a = set()
    used_b = set()

    while True:
        # Find the maximum remaining similarity
        if len(used_a) >= n_a or len(used_b) >= n_b:
            break

        # Mask out used indices
        masked = sim.copy()
        for a in used_a:
            masked[a, :] = -1
        for b in used_b:
            masked[:, b] = -1

        max_val = masked.max()
        if max_val < threshold:
            break

        idx = np.unravel_index(np.argmax(masked), masked.shape)
        i, j = int(idx[0]), int(idx[1])

        matches.append((i, j, float(max_val)))
        used_a.add(i)
        used_b.add(j)

    return matches


def compare_by_decoder_similarity(
    decoder_a: Union[np.ndarray, torch.Tensor],
    decoder_b: Union[np.ndarray, torch.Tensor],
    top_k: int = 100,
    match_threshold: float = 0.3,
) -> Tuple[List[FeatureMatch], np.ndarray, List[int], List[int]]:
    """
    Compare SAE features by decoder weight cosine similarity.

    Only works when both SAEs operate in the same activation space (same d_model).

    Args:
        decoder_a: Decoder weights from SAE A (d_hidden_a, d_model)
        decoder_b: Decoder weights from SAE B (d_hidden_b, d_model)
        top_k: Number of top features to consider from each SAE
        match_threshold: Minimum cosine similarity for a match

    Returns:
        Tuple of (matches, similarity_matrix, indices_a, indices_b)
    """
    decoder_a = _to_numpy(decoder_a)
    decoder_b = _to_numpy(decoder_b)

    # Select top features by decoder norm (most prominent directions)
    norms_a = np.linalg.norm(decoder_a, axis=1)
    norms_b = np.linalg.norm(decoder_b, axis=1)

    top_k_a = min(top_k, decoder_a.shape[0])
    top_k_b = min(top_k, decoder_b.shape[0])

    indices_a = np.argsort(norms_a)[-top_k_a:][::-1].copy()
    indices_b = np.argsort(norms_b)[-top_k_b:][::-1].copy()

    # Compute cosine similarity matrix for top features
    sim_matrix = cosine_similarity_matrix(
        decoder_a[indices_a],
        decoder_b[indices_b],
    )

    # Use absolute similarity (directions can be flipped)
    abs_sim = np.abs(sim_matrix)

    # Greedy matching
    raw_matches = greedy_match(abs_sim, threshold=match_threshold)

    matches = [
        FeatureMatch(
            feature_a=int(indices_a[i]),
            feature_b=int(indices_b[j]),
            similarity=sim,
            method="decoder_cosine",
        )
        for i, j, sim in raw_matches
    ]

    return matches, sim_matrix, indices_a.tolist(), indices_b.tolist()


def compare_by_activation_overlap(
    hidden_a: Union[np.ndarray, torch.Tensor],
    hidden_b: Union[np.ndarray, torch.Tensor],
    tokens: List[str],
    top_k: int = 100,
    match_threshold: float = 0.3,
) -> Tuple[List[FeatureMatch], np.ndarray, List[int], List[int]]:
    """
    Compare SAE features by activation pattern overlap on shared input.

    Works across different activation spaces (different d_model).
    Features are matched by how similarly they activate across shared tokens.

    Args:
        hidden_a: SAE A hidden activations (n_tokens, d_hidden_a)
        hidden_b: SAE B hidden activations (n_tokens, d_hidden_b)
        tokens: Token strings (for shared_tokens in matches)
        top_k: Number of top features to consider from each SAE
        match_threshold: Minimum correlation for a match

    Returns:
        Tuple of (matches, correlation_matrix, indices_a, indices_b)
    """
    corr_matrix, indices_a, indices_b = activation_correlation_matrix(
        hidden_a, hidden_b,
        top_k_a=top_k, top_k_b=top_k,
    )

    abs_corr = np.abs(corr_matrix)
    raw_matches = greedy_match(abs_corr, threshold=match_threshold)

    hidden_a_np = _to_numpy(hidden_a)
    hidden_b_np = _to_numpy(hidden_b)

    matches = []
    for i, j, sim in raw_matches:
        feat_a_idx = indices_a[i]
        feat_b_idx = indices_b[j]

        # Find shared activating tokens
        active_a = set(np.where(hidden_a_np[:, feat_a_idx] > 0)[0])
        active_b = set(np.where(hidden_b_np[:, feat_b_idx] > 0)[0])
        shared_positions = active_a & active_b
        shared_token_strs = [tokens[p] for p in sorted(shared_positions) if p < len(tokens)]

        matches.append(FeatureMatch(
            feature_a=feat_a_idx,
            feature_b=feat_b_idx,
            similarity=sim,
            method="activation_correlation",
            shared_tokens=shared_token_strs[:20],
        ))

    return matches, corr_matrix, indices_a, indices_b


def compare_saes(
    hidden_a: Union[np.ndarray, torch.Tensor],
    hidden_b: Union[np.ndarray, torch.Tensor],
    tokens: List[str],
    decoder_a: Optional[Union[np.ndarray, torch.Tensor]] = None,
    decoder_b: Optional[Union[np.ndarray, torch.Tensor]] = None,
    top_k: int = 100,
    match_threshold: float = 0.3,
) -> ComparisonResult:
    """
    Full comparison between two SAEs.

    Automatically selects comparison method:
    - If decoder weights are provided and same dimension: use decoder cosine similarity
    - Otherwise: use activation-based correlation

    Args:
        hidden_a: SAE A hidden activations (n_tokens, d_hidden_a)
        hidden_b: SAE B hidden activations (n_tokens, d_hidden_b)
        tokens: Token strings for the shared input
        decoder_a: Optional decoder weights (d_hidden_a, d_model)
        decoder_b: Optional decoder weights (d_hidden_b, d_model)
        top_k: Number of top features to compare
        match_threshold: Minimum similarity for a match

    Returns:
        ComparisonResult with full comparison metrics
    """
    hidden_a_np = _to_numpy(hidden_a)
    hidden_b_np = _to_numpy(hidden_b)

    # Compute CKA between the full hidden representations
    cka_score = linear_cka(hidden_a_np, hidden_b_np)

    # Check if we can use decoder similarity (same activation space dimension)
    use_decoder = (
        decoder_a is not None
        and decoder_b is not None
        and _to_numpy(decoder_a).shape[1] == _to_numpy(decoder_b).shape[1]
    )

    if use_decoder:
        matches, sim_matrix, indices_a, indices_b = compare_by_decoder_similarity(
            decoder_a, decoder_b,
            top_k=top_k,
            match_threshold=match_threshold,
        )
        # Also add shared_tokens from activations
        for match in matches:
            active_a = set(np.where(hidden_a_np[:, match.feature_a] > 0)[0])
            active_b = set(np.where(hidden_b_np[:, match.feature_b] > 0)[0])
            shared_positions = active_a & active_b
            match.shared_tokens = [
                tokens[p] for p in sorted(shared_positions) if p < len(tokens)
            ][:20]
    else:
        matches, sim_matrix, indices_a, indices_b = compare_by_activation_overlap(
            hidden_a_np, hidden_b_np, tokens,
            top_k=top_k,
            match_threshold=match_threshold,
        )

    # Compute statistics
    matched_a = {m.feature_a for m in matches}
    matched_b = {m.feature_b for m in matches}

    # Active features (features that activate at all on this input)
    active_a = set(np.where((hidden_a_np > 0).any(axis=0))[0])
    active_b = set(np.where((hidden_b_np > 0).any(axis=0))[0])

    unmatched_a = len(active_a - matched_a)
    unmatched_b = len(active_b - matched_b)

    # Overall similarity: weighted combination of CKA and match quality
    if matches:
        avg_match_sim = np.mean([m.similarity for m in matches])
        match_coverage = len(matches) / max(len(active_a), len(active_b), 1)
        overall_similarity = 0.5 * cka_score + 0.3 * avg_match_sim + 0.2 * match_coverage
    else:
        overall_similarity = cka_score * 0.5

    overall_similarity = float(min(max(overall_similarity, 0.0), 1.0))

    # Divergent features: active in one but not matched
    divergent_a = sorted(active_a - matched_a)[:20]
    divergent_b = sorted(active_b - matched_b)[:20]

    # Sample of similarity matrix for visualization (top 30x30)
    sample_size = min(30, sim_matrix.shape[0], sim_matrix.shape[1])
    sim_sample = sim_matrix[:sample_size, :sample_size].tolist()

    # Sort matches by similarity
    matches.sort(key=lambda m: m.similarity, reverse=True)

    return ComparisonResult(
        overall_similarity=overall_similarity,
        cka_score=cka_score,
        matched_features=len(matches),
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        top_matches=matches[:30],
        divergent_features_a=divergent_a,
        divergent_features_b=divergent_b,
        similarity_matrix_sample=sim_sample,
        feature_indices_a=indices_a[:sample_size],
        feature_indices_b=indices_b[:sample_size],
    )
