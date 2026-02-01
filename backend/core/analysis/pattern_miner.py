"""
Pattern mining algorithms for attention analysis.

Implements cross-layer pattern aggregation, head importance scoring,
and circuit detection for multi-head patterns.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from .head_classifier import HeadClassifier, ClassificationResult, HeadType


@dataclass
class HeadImportance:
    """Importance metrics for a single head."""
    layer: int
    head: int
    entropy_score: float  # Lower entropy = more focused
    max_attention_score: float  # Higher = more concentrated
    variance_score: float  # Higher = more dynamic
    overall_importance: float  # Combined score


@dataclass
class CrossLayerPattern:
    """Pattern detected across multiple layers."""
    pattern_type: str
    description: str
    involved_heads: list[tuple[int, int]]  # List of (layer, head) tuples
    confidence: float
    evidence: dict[str, float]


@dataclass
class AttentionComparison:
    """Comparison of attention patterns across two prompts."""
    prompt1: str
    prompt2: str
    activation_id1: str
    activation_id2: str
    layer_similarities: list[dict]  # Similarity per layer
    head_differences: list[dict]  # Most different heads
    common_patterns: list[str]  # Patterns found in both
    divergent_patterns: list[str]  # Patterns that differ
    length_warning: Optional[str] = None  # Warning if sequence lengths differ significantly
    comparison_method: str = "direct"  # "direct" or "statistical" based on length match


@dataclass
class PatternMiningResult:
    """Complete result of pattern mining analysis."""
    model_id: str
    activation_ids: list[str]
    head_importance: list[HeadImportance]
    cross_layer_patterns: list[CrossLayerPattern]
    classification_summary: dict[str, int]  # Count of each head type
    top_heads: list[tuple[int, int, float]]  # Top N most important heads
    circuit_candidates: list[dict]  # Potential multi-head circuits


class PatternMiner:
    """
    Pattern mining for attention analysis.

    Aggregates patterns across layers, computes head importance,
    and detects potential circuits.
    """

    def __init__(self, head_classifier: Optional[HeadClassifier] = None):
        """
        Initialize pattern miner.

        Args:
            head_classifier: Optional HeadClassifier instance. If not provided,
                           creates one with default settings.
        """
        self.head_classifier = head_classifier or HeadClassifier()

    def mine_patterns(
        self,
        activations: dict[str, torch.Tensor],
        tokens: list[int],
        decoded_tokens: list[str],
        num_layers: int,
        model_id: str,
        activation_id: str,
        prompt: str,
    ) -> PatternMiningResult:
        """
        Run comprehensive pattern mining on captured activations.

        Args:
            activations: Dictionary of captured activations.
            tokens: List of token IDs.
            decoded_tokens: List of decoded token strings.
            num_layers: Number of layers in the model.
            model_id: Model identifier.
            activation_id: ID of captured activations.
            prompt: Original prompt.

        Returns:
            PatternMiningResult with all analysis results.
        """
        # First, classify all heads
        classification_result = self.head_classifier.classify_all_heads(
            activations, tokens, decoded_tokens, num_layers,
            model_id, activation_id, prompt
        )

        # Compute head importance
        head_importance = self._compute_head_importance(activations, num_layers)

        # Find cross-layer patterns
        cross_layer_patterns = self._find_cross_layer_patterns(
            classification_result, activations, num_layers
        )

        # Generate classification summary
        classification_summary = self._summarize_classifications(classification_result)

        # Find top important heads
        top_heads = sorted(
            [(h.layer, h.head, h.overall_importance) for h in head_importance],
            key=lambda x: x[2],
            reverse=True
        )[:10]

        # Detect potential circuits
        circuit_candidates = self._detect_circuits(
            classification_result, head_importance, activations, num_layers
        )

        return PatternMiningResult(
            model_id=model_id,
            activation_ids=[activation_id],
            head_importance=head_importance,
            cross_layer_patterns=cross_layer_patterns,
            classification_summary=classification_summary,
            top_heads=top_heads,
            circuit_candidates=circuit_candidates,
        )

    def _compute_head_importance(
        self,
        activations: dict[str, torch.Tensor],
        num_layers: int,
    ) -> list[HeadImportance]:
        """
        Compute importance metrics for each attention head.

        Args:
            activations: Dictionary of captured activations.
            num_layers: Number of layers.

        Returns:
            List of HeadImportance for each head.
        """
        importance_list = []

        for layer_idx in range(num_layers):
            attn_key = f"layer_{layer_idx}_attn_weights"
            attn_weights = activations.get(attn_key)

            if attn_weights is None:
                continue

            if isinstance(attn_weights, torch.Tensor):
                attn_weights = attn_weights.cpu().numpy()

            if len(attn_weights.shape) == 4:
                attn_weights = attn_weights[0]

            for head_idx in range(attn_weights.shape[0]):
                head_attn = attn_weights[head_idx]

                # Compute entropy (lower = more focused attention)
                eps = 1e-10
                entropy_per_row = -np.sum(head_attn * np.log(head_attn + eps), axis=-1)
                mean_entropy = float(np.mean(entropy_per_row))
                # Normalize: lower entropy is better, scale to 0-1
                entropy_score = max(0.0, 1.0 - mean_entropy / np.log(head_attn.shape[1]))

                # Max attention score (higher = more concentrated)
                max_attention_score = float(np.max(head_attn))

                # Variance score (higher = more dynamic/interesting patterns)
                variance_score = float(np.std(head_attn))
                variance_score = min(1.0, variance_score * 5)  # Scale up

                # Combined importance
                overall_importance = (
                    entropy_score * 0.3 +
                    max_attention_score * 0.4 +
                    variance_score * 0.3
                )

                importance_list.append(HeadImportance(
                    layer=layer_idx,
                    head=head_idx,
                    entropy_score=entropy_score,
                    max_attention_score=max_attention_score,
                    variance_score=variance_score,
                    overall_importance=overall_importance,
                ))

        return importance_list

    def _find_cross_layer_patterns(
        self,
        classification: ClassificationResult,
        activations: dict[str, torch.Tensor],
        num_layers: int,
    ) -> list[CrossLayerPattern]:
        """
        Find patterns that span multiple layers.

        Args:
            classification: Head classification results.
            activations: Dictionary of captured activations.
            num_layers: Number of layers.

        Returns:
            List of cross-layer patterns.
        """
        patterns = []

        # Pattern 1: Induction head emergence
        # Induction heads typically appear in middle-to-late layers
        induction_heads = classification.get_heads_by_type(HeadType.INDUCTION)
        if induction_heads:
            layers = [h.layer for h in induction_heads]
            avg_layer = np.mean(layers)
            patterns.append(CrossLayerPattern(
                pattern_type="induction_emergence",
                description=f"Induction heads found primarily in layers {min(layers)}-{max(layers)} (avg: {avg_layer:.1f})",
                involved_heads=[(h.layer, h.head) for h in induction_heads],
                confidence=np.mean([h.confidence for h in induction_heads]),
                evidence={
                    'min_layer': min(layers),
                    'max_layer': max(layers),
                    'avg_layer': float(avg_layer),
                    'count': len(induction_heads),
                },
            ))

        # Pattern 2: Previous token head distribution
        prev_token_heads = classification.get_heads_by_type(HeadType.PREVIOUS_TOKEN)
        if prev_token_heads:
            layers = [h.layer for h in prev_token_heads]
            patterns.append(CrossLayerPattern(
                pattern_type="previous_token_distribution",
                description=f"Previous token heads across {len(set(layers))} layers",
                involved_heads=[(h.layer, h.head) for h in prev_token_heads],
                confidence=np.mean([h.confidence for h in prev_token_heads]),
                evidence={
                    'unique_layers': len(set(layers)),
                    'total_heads': len(prev_token_heads),
                },
            ))

        # Pattern 3: Layer specialization
        # Check if different layers have different head type distributions
        layer_specializations = {}
        for c in classification.classifications:
            if c.layer not in layer_specializations:
                layer_specializations[c.layer] = {}
            layer_specializations[c.layer][c.primary_type.value] = \
                layer_specializations[c.layer].get(c.primary_type.value, 0) + 1

        # Find layers with strong specialization
        for layer, types in layer_specializations.items():
            total = sum(types.values())
            for type_name, count in types.items():
                ratio = count / total
                if ratio >= 0.5 and total >= 2:  # At least half the heads of one type
                    patterns.append(CrossLayerPattern(
                        pattern_type="layer_specialization",
                        description=f"Layer {layer} specializes in {type_name} ({ratio*100:.0f}% of heads)",
                        involved_heads=[(layer, h.head) for h in classification.classifications
                                       if h.layer == layer and h.primary_type.value == type_name],
                        confidence=ratio,
                        evidence={
                            'layer': layer,
                            'type': type_name,
                            'ratio': ratio,
                            'count': count,
                        },
                    ))

        return patterns

    def _summarize_classifications(
        self,
        classification: ClassificationResult,
    ) -> dict[str, int]:
        """
        Generate summary statistics of head classifications.

        Args:
            classification: Classification results.

        Returns:
            Dictionary mapping head type to count.
        """
        summary = {t.value: 0 for t in HeadType}

        for c in classification.classifications:
            summary[c.primary_type.value] += 1

        return summary

    def _detect_circuits(
        self,
        classification: ClassificationResult,
        importance: list[HeadImportance],
        activations: dict[str, torch.Tensor],
        num_layers: int,
    ) -> list[dict]:
        """
        Detect potential multi-head circuits.

        Circuits are combinations of heads that work together to implement
        a computation (e.g., induction circuits with previous token + induction heads).

        Args:
            classification: Head classification results.
            importance: Head importance scores.
            activations: Captured activations.
            num_layers: Number of layers.

        Returns:
            List of potential circuit candidates.
        """
        circuits = []

        # Circuit type 1: Induction circuit
        # Previous token head (early layer) + Induction head (later layer)
        prev_token_heads = classification.get_heads_by_type(HeadType.PREVIOUS_TOKEN)
        induction_heads = classification.get_heads_by_type(HeadType.INDUCTION)

        if prev_token_heads and induction_heads:
            early_prev = [h for h in prev_token_heads if h.layer < num_layers // 2]
            late_induction = [h for h in induction_heads if h.layer >= num_layers // 2]

            if early_prev and late_induction:
                circuits.append({
                    'type': 'induction_circuit',
                    'description': 'Previous token heads feeding into induction heads',
                    'components': {
                        'previous_token_heads': [(h.layer, h.head) for h in early_prev],
                        'induction_heads': [(h.layer, h.head) for h in late_induction],
                    },
                    'confidence': min(
                        np.mean([h.confidence for h in early_prev]),
                        np.mean([h.confidence for h in late_induction])
                    ),
                })

        # Circuit type 2: Copying circuit
        # Copying heads that might work together
        copying_heads = classification.get_heads_by_type(HeadType.COPYING)
        if len(copying_heads) >= 2:
            # Group by layer
            layers_with_copying = set(h.layer for h in copying_heads)
            if len(layers_with_copying) >= 2:
                circuits.append({
                    'type': 'copying_circuit',
                    'description': 'Multiple copying heads across layers',
                    'components': {
                        'copying_heads': [(h.layer, h.head) for h in copying_heads],
                    },
                    'confidence': np.mean([h.confidence for h in copying_heads]),
                })

        return circuits

    def compare_attention_patterns(
        self,
        activations1: dict[str, torch.Tensor],
        activations2: dict[str, torch.Tensor],
        tokens1: list[str],
        tokens2: list[str],
        activation_id1: str,
        activation_id2: str,
        prompt1: str,
        prompt2: str,
        num_layers: int,
    ) -> AttentionComparison:
        """
        Compare attention patterns between two prompts.

        Args:
            activations1: Activations from first prompt.
            activations2: Activations from second prompt.
            tokens1: Decoded tokens from first prompt.
            tokens2: Decoded tokens from second prompt.
            activation_id1: ID of first activation capture.
            activation_id2: ID of second activation capture.
            prompt1: First prompt text.
            prompt2: Second prompt text.
            num_layers: Number of layers.

        Returns:
            AttentionComparison with similarity and difference analysis.
        """
        len1 = len(tokens1)
        len2 = len(tokens2)
        min_len = min(len1, len2)
        max_len = max(len1, len2)

        # Check for significant length mismatch
        length_warning = None
        comparison_method = "direct"
        length_ratio = min_len / max_len if max_len > 0 else 1.0

        # Determine if we need statistical comparison
        if min_len <= 2:
            # Extremely short sequence - attention is trivial
            length_warning = (
                f"Very short sequence detected: {len1} vs {len2} tokens. "
                f"Sequences with 1-2 tokens have trivial attention patterns (only self/BOS attention). "
                f"Similarity scores are heavily penalized and may not be meaningful. "
                f"For better comparison, use longer prompts (4+ tokens)."
            )
            comparison_method = "statistical"
        elif length_ratio < 0.5 or min_len < 4:
            length_warning = (
                f"Significant sequence length mismatch: {len1} vs {len2} tokens. "
                f"Using statistical comparison instead of direct matrix comparison. "
                f"Results may be less reliable for mismatched lengths."
            )
            comparison_method = "statistical"

        layer_similarities = []
        head_differences = []

        for layer_idx in range(num_layers):
            attn_key = f"layer_{layer_idx}_attn_weights"
            attn1 = activations1.get(attn_key)
            attn2 = activations2.get(attn_key)

            if attn1 is None or attn2 is None:
                continue

            # Convert to numpy
            if isinstance(attn1, torch.Tensor):
                attn1 = attn1.cpu().numpy()
            if isinstance(attn2, torch.Tensor):
                attn2 = attn2.cpu().numpy()

            # Remove batch dimension if present
            if len(attn1.shape) == 4:
                attn1 = attn1[0]
            if len(attn2.shape) == 4:
                attn2 = attn2[0]

            num_heads = min(attn1.shape[0], attn2.shape[0])
            layer_head_sims = []

            for head_idx in range(num_heads):
                h1_full = attn1[head_idx]
                h2_full = attn2[head_idx]

                if comparison_method == "statistical":
                    # Use statistical comparison for mismatched lengths
                    similarity = self._compute_statistical_similarity(h1_full, h2_full)
                else:
                    # Direct comparison for similar-length sequences
                    seq_len = min(h1_full.shape[0], h1_full.shape[1],
                                  h2_full.shape[0], h2_full.shape[1])
                    h1 = h1_full[:seq_len, :seq_len]
                    h2 = h2_full[:seq_len, :seq_len]

                    # Compute cosine similarity of flattened patterns
                    h1_flat = h1.flatten()
                    h2_flat = h2.flatten()
                    similarity = float(np.dot(h1_flat, h2_flat) / (
                        np.linalg.norm(h1_flat) * np.linalg.norm(h2_flat) + 1e-10
                    ))

                layer_head_sims.append(similarity)

                # Track significant differences
                if similarity < 0.7:
                    head_differences.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'similarity': similarity,
                        'difference': 1.0 - similarity,
                    })

            layer_similarities.append({
                'layer': layer_idx,
                'mean_similarity': float(np.mean(layer_head_sims)),
                'min_similarity': float(np.min(layer_head_sims)),
                'max_similarity': float(np.max(layer_head_sims)),
            })

        # Sort head differences by magnitude
        head_differences.sort(key=lambda x: x['difference'], reverse=True)
        head_differences = head_differences[:10]  # Top 10 most different

        # Analyze common vs divergent patterns
        common_patterns = []
        divergent_patterns = []

        # Overall similarity assessment
        overall_similarity = np.mean([l['mean_similarity'] for l in layer_similarities]) if layer_similarities else 0.0
        if overall_similarity > 0.8:
            common_patterns.append(f"High overall similarity ({overall_similarity:.2f})")
        elif overall_similarity < 0.5:
            divergent_patterns.append(f"Low overall similarity ({overall_similarity:.2f})")

        # Check for layer-specific patterns
        for ls in layer_similarities:
            if ls['mean_similarity'] < 0.5:
                divergent_patterns.append(f"Layer {ls['layer']} shows divergent patterns")
            elif ls['mean_similarity'] > 0.9:
                common_patterns.append(f"Layer {ls['layer']} shows similar patterns")

        return AttentionComparison(
            prompt1=prompt1,
            prompt2=prompt2,
            activation_id1=activation_id1,
            activation_id2=activation_id2,
            layer_similarities=layer_similarities,
            head_differences=head_differences,
            common_patterns=common_patterns,
            divergent_patterns=divergent_patterns,
            length_warning=length_warning,
            comparison_method=comparison_method,
        )

    def _compute_statistical_similarity(
        self,
        attn1: np.ndarray,
        attn2: np.ndarray,
    ) -> float:
        """
        Compute similarity between attention matrices using statistical features.

        This method is used when sequences have significantly different lengths,
        making direct matrix comparison unreliable. Instead, we compare:
        - Entropy of attention distributions
        - BOS attention strength
        - Local vs global attention ratios
        - Attention concentration metrics

        Args:
            attn1: Attention matrix for first prompt [seq_len1, seq_len1].
            attn2: Attention matrix for second prompt [seq_len2, seq_len2].

        Returns:
            Similarity score between 0 and 1.
        """
        eps = 1e-10
        len1, len2 = attn1.shape[0], attn2.shape[0]
        min_len = min(len1, len2)
        max_len = max(len1, len2)

        # For extremely short sequences (1-2 tokens), attention is trivial
        # and cannot be meaningfully compared - apply a length penalty
        if min_len <= 2:
            # Base penalty: very short sequences have degenerate attention patterns
            # The shorter the sequence, the less meaningful the comparison
            length_penalty = min_len / max(max_len, 3)  # Ranges from ~0.33 to 0.67

            # If both are very short, they're trivially similar but uninformative
            if max_len <= 2:
                return 0.5  # Neutral - can't determine meaningful similarity

            # One is short, one is long - likely very different in practice
            # Return a low similarity reflecting the fundamental incomparability
            return length_penalty * 0.5  # Will be 0.17 - 0.33 range

        def extract_features(attn: np.ndarray) -> tuple[np.ndarray, int]:
            """Extract statistical features from an attention matrix."""
            seq_len = attn.shape[0]
            features = []

            # Feature 1: Mean entropy (normalized)
            entropies = []
            for i in range(1, seq_len):
                row = attn[i, :i+1]
                row = row / (np.sum(row) + eps)
                entropy = -np.sum(row * np.log(row + eps))
                entropies.append(entropy / (np.log(i + 1) + eps))
            mean_entropy = np.mean(entropies) if entropies else 0.0
            features.append(mean_entropy)

            # Feature 2: BOS attention strength (attention to first token)
            if seq_len > 1:
                bos_attn = np.mean(attn[1:, 0])
            else:
                bos_attn = 0.0
            features.append(bos_attn)

            # Feature 3: Self-attention strength (diagonal)
            diag = np.diag(attn)
            self_attn = np.mean(diag)
            features.append(self_attn)

            # Feature 4: Previous token attention (sub-diagonal)
            if seq_len > 1:
                prev_attn = np.mean([attn[i, i-1] for i in range(1, seq_len)])
            else:
                prev_attn = 0.0
            features.append(prev_attn)

            # Feature 5: Local window attention (within 3 tokens)
            local_scores = []
            for i in range(seq_len):
                local_start = max(0, i - 3)
                local_end = min(seq_len, i + 1)
                local_attn = np.sum(attn[i, local_start:local_end])
                local_scores.append(local_attn)
            features.append(np.mean(local_scores))

            # Feature 6: Max attention (concentration)
            max_attns = []
            for i in range(seq_len):
                max_attns.append(np.max(attn[i, :i+1]) if i > 0 else attn[i, 0])
            features.append(np.mean(max_attns))

            # Feature 7: Attention variance
            variances = []
            for i in range(1, seq_len):
                variances.append(np.var(attn[i, :i+1]))
            features.append(np.mean(variances) if variances else 0.0)

            # Feature 8: Far attention (attention to positions > 5 tokens away)
            far_scores = []
            for i in range(6, seq_len):
                far_attn = np.sum(attn[i, :i-5])
                far_scores.append(far_attn)
            features.append(np.mean(far_scores) if far_scores else 0.0)

            return np.array(features)

        # Extract features from both matrices
        f1 = extract_features(attn1)
        f2 = extract_features(attn2)

        # Use Euclidean distance-based similarity instead of cosine
        # This better captures magnitude differences between feature vectors
        diff = f1 - f2
        euclidean_dist = np.sqrt(np.sum(diff ** 2))

        # Normalize: max possible distance is sqrt(8) ≈ 2.83 (8 features, each ranging 0-1)
        max_dist = np.sqrt(len(f1))
        normalized_dist = euclidean_dist / max_dist

        # Convert distance to similarity (0 = very different, 1 = identical)
        base_similarity = 1.0 - normalized_dist

        # Apply length ratio penalty for significant length mismatches
        length_ratio = min_len / max_len
        if length_ratio < 0.3:
            # Very different lengths - apply stronger penalty
            length_penalty = 0.5 + (length_ratio / 0.3) * 0.5  # 0.5 to 1.0
            base_similarity *= length_penalty

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_similarity))
