"""
Feature interpretability tools for Sparse Autoencoders.

Provides tools for understanding what concepts individual features represent,
finding top activating examples, and generating automatic labels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import torch


@dataclass
class TokenActivationExample:
    """An example of a token that activates a feature."""

    token: str
    token_id: int
    activation_value: float
    position: int
    context: List[str]  # Surrounding tokens for context
    prompt: str  # Full prompt for reference


@dataclass
class FeatureInterpretation:
    """Interpretation data for a single feature."""

    feature_idx: int
    top_activating_tokens: List[TokenActivationExample]
    token_frequency_distribution: Dict[str, float]  # Token -> activation frequency
    suggested_label: str
    confidence: float
    activation_statistics: Dict[str, float]
    semantic_clusters: List[List[str]]  # Groups of related tokens


@dataclass
class FeatureCatalog:
    """Collection of feature interpretations for an SAE."""

    sae_id: str
    num_features: int
    interpretations: Dict[int, FeatureInterpretation]
    dead_features: List[int]
    polysemantic_features: List[int]  # Features that activate on diverse concepts
    monosemantic_features: List[int]  # Features with clear single concept


class FeatureInterpreter:
    """
    Interpreter for understanding SAE feature meanings.

    Collects examples of feature activations and generates
    interpretability data including suggested labels.
    """

    def __init__(self, sae, max_examples_per_feature: int = 100):
        """
        Initialize the feature interpreter.

        Args:
            sae: Trained SparseAutoencoderTorch model
            max_examples_per_feature: Maximum examples to store per feature
        """
        self.sae = sae
        self.d_hidden = sae.d_hidden
        self.max_examples = max_examples_per_feature

        # Storage for top activating examples per feature
        self.feature_examples: Dict[int, List[TokenActivationExample]] = defaultdict(list)

        # Token frequency counts per feature
        self.feature_token_counts: Dict[int, Counter] = defaultdict(Counter)
        self.feature_activation_counts: Dict[int, int] = defaultdict(int)

        # Activation statistics per feature
        self.feature_activation_sums: np.ndarray = np.zeros(self.d_hidden)
        self.feature_activation_max: np.ndarray = np.zeros(self.d_hidden)

    def process_example(
        self,
        hidden_activations: Union[np.ndarray, torch.Tensor],
        tokens: List[str],
        token_ids: List[int],
        prompt: str,
        context_window: int = 3,
    ) -> None:
        """
        Process an example and update feature interpretations.

        Args:
            hidden_activations: SAE hidden activations (seq_len, d_hidden)
            tokens: Token strings
            token_ids: Token IDs
            prompt: Original prompt
            context_window: Number of tokens to include as context
        """
        if isinstance(hidden_activations, torch.Tensor):
            hidden_activations = hidden_activations.detach().cpu().numpy()

        seq_len = hidden_activations.shape[0]

        for pos in range(seq_len):
            pos_activations = hidden_activations[pos]
            active_mask = pos_activations > 0
            active_indices = np.where(active_mask)[0]

            for feat_idx in active_indices:
                feat_val = float(pos_activations[feat_idx])

                # Update statistics
                self.feature_activation_sums[feat_idx] += feat_val
                self.feature_activation_max[feat_idx] = max(
                    self.feature_activation_max[feat_idx], feat_val
                )
                self.feature_activation_counts[feat_idx] += 1

                # Update token counts
                if pos < len(tokens):
                    self.feature_token_counts[feat_idx][tokens[pos]] += 1

                # Create example
                context_start = max(0, pos - context_window)
                context_end = min(len(tokens), pos + context_window + 1)
                context = tokens[context_start:context_end]

                example = TokenActivationExample(
                    token=tokens[pos] if pos < len(tokens) else "",
                    token_id=token_ids[pos] if pos < len(token_ids) else 0,
                    activation_value=feat_val,
                    position=pos,
                    context=context,
                    prompt=prompt[:200],  # Truncate for storage
                )

                # Add to examples, keeping top activations
                examples = self.feature_examples[feat_idx]
                examples.append(example)

                # Keep only top examples by activation value
                if len(examples) > self.max_examples:
                    examples.sort(key=lambda x: x.activation_value, reverse=True)
                    self.feature_examples[feat_idx] = examples[: self.max_examples]

    def get_top_tokens_for_feature(
        self, feature_idx: int, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get the top tokens that activate a feature.

        Args:
            feature_idx: Feature index
            top_k: Number of top tokens to return

        Returns:
            List of (token, frequency) tuples
        """
        token_counts = self.feature_token_counts[feature_idx]
        total = sum(token_counts.values())

        if total == 0:
            return []

        # Normalize to frequencies
        frequencies = [(token, count / total) for token, count in token_counts.items()]
        frequencies.sort(key=lambda x: x[1], reverse=True)

        return frequencies[:top_k]

    def get_top_examples_for_feature(
        self, feature_idx: int, top_k: int = 10
    ) -> List[TokenActivationExample]:
        """
        Get the top activating examples for a feature.

        Args:
            feature_idx: Feature index
            top_k: Number of examples to return

        Returns:
            List of TokenActivationExample objects
        """
        examples = self.feature_examples.get(feature_idx, [])
        examples.sort(key=lambda x: x.activation_value, reverse=True)
        return examples[:top_k]

    def suggest_label(self, feature_idx: int) -> Tuple[str, float]:
        """
        Generate a suggested label for a feature based on its activations.

        Args:
            feature_idx: Feature index

        Returns:
            Tuple of (suggested_label, confidence)
        """
        top_tokens = self.get_top_tokens_for_feature(feature_idx, top_k=10)

        if not top_tokens:
            return "dead_feature", 0.0

        # Check if feature is monosemantic (dominated by one token type)
        if top_tokens[0][1] > 0.5:
            # Single token dominates - high confidence label
            return f"token:{top_tokens[0][0]}", top_tokens[0][1]

        # Check for common patterns
        all_tokens = [t for t, _ in top_tokens[:5]]

        # Punctuation pattern
        if all(self._is_punctuation(t) for t in all_tokens):
            return "punctuation", 0.8

        # Whitespace/formatting pattern
        if all(self._is_whitespace(t) for t in all_tokens):
            return "whitespace", 0.8

        # Numeric pattern
        if all(self._is_numeric(t) for t in all_tokens):
            return "numeric", 0.8

        # Capital letters pattern
        if all(self._starts_with_capital(t) for t in all_tokens):
            return "capitalized", 0.7

        # Common word categories
        categories = self._categorize_tokens(all_tokens)
        if categories:
            most_common_category, count = categories.most_common(1)[0]
            if count >= 3:
                return most_common_category, count / len(all_tokens)

        # Default: use top tokens as label
        label_tokens = "_".join(all_tokens[:3])
        return f"mixed:{label_tokens}", 0.3

    def _is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return all(c in ".,;:!?\"'`()[]{}/-@#$%^&*+=<>" for c in token.strip())

    def _is_whitespace(self, token: str) -> bool:
        """Check if token is whitespace-related."""
        return token.strip() == "" or token in [" ", "\n", "\t", "Ġ", "▁"]

    def _is_numeric(self, token: str) -> bool:
        """Check if token is numeric."""
        return token.strip().replace(".", "").replace(",", "").isdigit()

    def _starts_with_capital(self, token: str) -> bool:
        """Check if token starts with capital letter."""
        stripped = token.strip().lstrip("Ġ▁ ")
        return stripped and stripped[0].isupper()

    def _categorize_tokens(self, tokens: List[str]) -> Counter:
        """Categorize tokens into semantic groups."""
        categories = Counter()

        for token in tokens:
            clean = token.strip().lower().lstrip("ġ▁ ")

            # Simple categorization based on common patterns
            if clean in ["the", "a", "an", "this", "that", "these", "those"]:
                categories["determiner"] += 1
            elif clean in ["and", "or", "but", "so", "yet", "for", "nor"]:
                categories["conjunction"] += 1
            elif clean in ["is", "are", "was", "were", "be", "been", "being"]:
                categories["be_verb"] += 1
            elif clean in ["have", "has", "had", "having"]:
                categories["have_verb"] += 1
            elif clean in ["to", "of", "in", "on", "at", "by", "for", "with"]:
                categories["preposition"] += 1
            elif clean in ["i", "you", "he", "she", "it", "we", "they"]:
                categories["pronoun"] += 1
            elif clean in ["not", "no", "never", "none"]:
                categories["negation"] += 1

        return categories

    def compute_polysemanticity_score(self, feature_idx: int) -> float:
        """
        Compute how polysemantic (multi-meaning) a feature is.

        Lower score = more monosemantic (single clear meaning)
        Higher score = more polysemantic (multiple meanings)

        Args:
            feature_idx: Feature index

        Returns:
            Polysemanticity score (0-1)
        """
        token_counts = self.feature_token_counts[feature_idx]

        if not token_counts:
            return 0.0

        total = sum(token_counts.values())
        if total == 0:
            return 0.0

        # Compute entropy of token distribution
        probs = np.array([c / total for c in token_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(token_counts)) if len(token_counts) > 1 else 1.0

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def get_feature_interpretation(
        self, feature_idx: int
    ) -> FeatureInterpretation:
        """
        Get complete interpretation data for a feature.

        Args:
            feature_idx: Feature index

        Returns:
            FeatureInterpretation object
        """
        top_examples = self.get_top_examples_for_feature(feature_idx, top_k=20)
        top_tokens = self.get_top_tokens_for_feature(feature_idx, top_k=50)
        suggested_label, confidence = self.suggest_label(feature_idx)

        # Compute statistics
        count = self.feature_activation_counts[feature_idx]
        stats = {
            "total_activations": count,
            "mean_activation": float(
                self.feature_activation_sums[feature_idx] / max(count, 1)
            ),
            "max_activation": float(self.feature_activation_max[feature_idx]),
            "unique_tokens": len(self.feature_token_counts[feature_idx]),
            "polysemanticity": self.compute_polysemanticity_score(feature_idx),
        }

        return FeatureInterpretation(
            feature_idx=feature_idx,
            top_activating_tokens=top_examples,
            token_frequency_distribution=dict(top_tokens),
            suggested_label=suggested_label,
            confidence=confidence,
            activation_statistics=stats,
            semantic_clusters=[],  # Would require more sophisticated clustering
        )

    def build_catalog(self, sae_id: str = "default") -> FeatureCatalog:
        """
        Build a complete catalog of feature interpretations.

        Args:
            sae_id: Identifier for this SAE

        Returns:
            FeatureCatalog object
        """
        interpretations = {}
        dead_features = []
        polysemantic_features = []
        monosemantic_features = []

        for feat_idx in range(self.d_hidden):
            interp = self.get_feature_interpretation(feat_idx)
            interpretations[feat_idx] = interp

            # Classify feature
            if interp.activation_statistics["total_activations"] == 0:
                dead_features.append(feat_idx)
            elif interp.activation_statistics["polysemanticity"] > 0.7:
                polysemantic_features.append(feat_idx)
            elif interp.activation_statistics["polysemanticity"] < 0.3:
                monosemantic_features.append(feat_idx)

        return FeatureCatalog(
            sae_id=sae_id,
            num_features=self.d_hidden,
            interpretations=interpretations,
            dead_features=dead_features,
            polysemantic_features=polysemantic_features,
            monosemantic_features=monosemantic_features,
        )


def find_semantically_similar_features(
    feature_interpretations: Dict[int, FeatureInterpretation],
    feature_idx: int,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Find features that are semantically similar to a target feature.

    Similarity is based on overlap in top activating tokens.

    Args:
        feature_interpretations: Dictionary of feature interpretations
        feature_idx: Target feature index
        top_k: Number of similar features to return

    Returns:
        List of (feature_idx, similarity_score) tuples
    """
    target = feature_interpretations.get(feature_idx)
    if not target:
        return []

    target_tokens = set(target.token_frequency_distribution.keys())
    if not target_tokens:
        return []

    similarities = []

    for other_idx, other_interp in feature_interpretations.items():
        if other_idx == feature_idx:
            continue

        other_tokens = set(other_interp.token_frequency_distribution.keys())
        if not other_tokens:
            continue

        # Jaccard similarity
        intersection = len(target_tokens & other_tokens)
        union = len(target_tokens | other_tokens)
        similarity = intersection / union if union > 0 else 0

        if similarity > 0:
            similarities.append((other_idx, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def cluster_features_by_tokens(
    feature_interpretations: Dict[int, FeatureInterpretation],
    min_similarity: float = 0.3,
) -> List[List[int]]:
    """
    Cluster features by their token activation patterns.

    Args:
        feature_interpretations: Dictionary of feature interpretations
        min_similarity: Minimum Jaccard similarity to consider as cluster

    Returns:
        List of feature index clusters
    """
    # Build adjacency based on token overlap
    feature_indices = list(feature_interpretations.keys())
    n = len(feature_indices)

    # Get token sets
    token_sets = {}
    for idx in feature_indices:
        interp = feature_interpretations[idx]
        token_sets[idx] = set(interp.token_frequency_distribution.keys())

    # Simple greedy clustering
    clusters = []
    used = set()

    for idx in feature_indices:
        if idx in used:
            continue

        if not token_sets[idx]:
            continue

        cluster = [idx]
        used.add(idx)

        for other_idx in feature_indices:
            if other_idx in used:
                continue
            if not token_sets[other_idx]:
                continue

            # Compute Jaccard similarity
            intersection = len(token_sets[idx] & token_sets[other_idx])
            union = len(token_sets[idx] | token_sets[other_idx])
            similarity = intersection / union if union > 0 else 0

            if similarity >= min_similarity:
                cluster.append(other_idx)
                used.add(other_idx)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
