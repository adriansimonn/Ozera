"""
Head classification algorithms for attention pattern analysis.

Implements detection of common attention head types:
- Induction heads (A...B...A -> B pattern)
- Previous token heads (sub-diagonal attention)
- Positional heads (fixed offset patterns)
- Copying heads (attention to repeated tokens)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import torch


class HeadType(str, Enum):
    """Types of attention heads that can be detected."""
    INDUCTION = "induction"
    PREVIOUS_TOKEN = "previous_token"
    POSITIONAL = "positional"
    COPYING = "copying"
    BOS_ATTENTION = "bos_attention"  # Attends to first token
    DELIMITER = "delimiter"  # Attends to punctuation/structural tokens
    LOCAL_WINDOW = "local_window"  # Attends to nearby tokens (broader than previous)
    DISTRIBUTED = "distributed"  # Spreads attention broadly across sequence
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class HeadClassification:
    """Classification result for a single attention head."""
    layer: int
    head: int
    primary_type: HeadType
    confidence: float
    scores: dict[str, float]  # Scores for each head type
    pattern_summary: str


@dataclass
class ClassificationResult:
    """Complete classification result for all heads."""
    model_id: str
    activation_id: str
    prompt: str
    tokens: list[str]
    num_layers: int
    num_heads: int
    classifications: list[HeadClassification]

    def get_heads_by_type(self, head_type: HeadType) -> list[HeadClassification]:
        """Get all heads classified as a specific type."""
        return [c for c in self.classifications if c.primary_type == head_type]

    def get_head(self, layer: int, head: int) -> Optional[HeadClassification]:
        """Get classification for a specific head."""
        for c in self.classifications:
            if c.layer == layer and c.head == head:
                return c
        return None


class HeadClassifier:
    """
    Classifier for attention head patterns.

    Analyzes attention weight matrices to identify common head types
    based on their attention patterns.
    """

    def __init__(
        self,
        induction_threshold: float = 0.3,
        previous_token_threshold: float = 0.5,
        positional_threshold: float = 0.4,
        copying_threshold: float = 0.3,
        bos_attention_threshold: float = 0.25,
        delimiter_threshold: float = 0.3,
        local_window_threshold: float = 0.4,
        distributed_threshold: float = 0.7,  # High entropy threshold
    ):
        """
        Initialize the classifier with detection thresholds.

        Args:
            induction_threshold: Minimum score for induction head detection.
            previous_token_threshold: Minimum score for previous token head detection.
            positional_threshold: Minimum score for positional head detection.
            copying_threshold: Minimum score for copying head detection.
            bos_attention_threshold: Minimum score for BOS attention head detection.
            delimiter_threshold: Minimum score for delimiter head detection.
            local_window_threshold: Minimum score for local window head detection.
            distributed_threshold: Minimum entropy ratio for distributed head detection.
        """
        self.induction_threshold = induction_threshold
        self.previous_token_threshold = previous_token_threshold
        self.positional_threshold = positional_threshold
        self.copying_threshold = copying_threshold
        self.bos_attention_threshold = bos_attention_threshold
        self.delimiter_threshold = delimiter_threshold
        self.local_window_threshold = local_window_threshold
        self.distributed_threshold = distributed_threshold

    def classify_all_heads(
        self,
        activations: dict[str, torch.Tensor],
        tokens: list[int],
        decoded_tokens: list[str],
        num_layers: int,
        model_id: str,
        activation_id: str,
        prompt: str,
    ) -> ClassificationResult:
        """
        Classify all attention heads in the model.

        Args:
            activations: Dictionary of captured activations with keys like
                        'layer_{idx}_attn_weights'.
            tokens: List of token IDs.
            decoded_tokens: List of decoded token strings.
            num_layers: Number of layers in the model.
            model_id: Model identifier.
            activation_id: ID of the captured activations.
            prompt: Original prompt text.

        Returns:
            ClassificationResult with classifications for all heads.
        """
        classifications = []
        num_heads = None

        for layer_idx in range(num_layers):
            attn_key = f"layer_{layer_idx}_attn_weights"
            attn_weights = activations.get(attn_key)

            if attn_weights is None:
                continue

            # Convert to numpy for analysis
            if isinstance(attn_weights, torch.Tensor):
                attn_weights = attn_weights.cpu().numpy()

            # Shape: [batch, num_heads, seq_len, seq_len]
            if len(attn_weights.shape) == 4:
                attn_weights = attn_weights[0]  # Remove batch dimension

            if num_heads is None:
                num_heads = attn_weights.shape[0]

            # Classify each head
            for head_idx in range(attn_weights.shape[0]):
                head_attn = attn_weights[head_idx]  # [seq_len, seq_len]
                classification = self._classify_head(
                    head_attn, tokens, decoded_tokens, layer_idx, head_idx
                )
                classifications.append(classification)

        return ClassificationResult(
            model_id=model_id,
            activation_id=activation_id,
            prompt=prompt,
            tokens=decoded_tokens,
            num_layers=num_layers,
            num_heads=num_heads or 0,
            classifications=classifications,
        )

    def _classify_head(
        self,
        attn: np.ndarray,
        tokens: list[int],
        decoded_tokens: list[str],
        layer: int,
        head: int,
    ) -> HeadClassification:
        """
        Classify a single attention head.

        Args:
            attn: Attention weights matrix [seq_len, seq_len].
            tokens: List of token IDs.
            decoded_tokens: List of decoded token strings.
            layer: Layer index.
            head: Head index.

        Returns:
            HeadClassification with scores and primary type.
        """
        scores = {
            'induction': self._compute_induction_score(attn, tokens, decoded_tokens),
            'previous_token': self._compute_previous_token_score(attn),
            'positional': self._compute_positional_score(attn),
            'copying': self._compute_copying_score(attn, tokens, decoded_tokens),
            'bos_attention': self._compute_bos_attention_score(attn),
            'delimiter': self._compute_delimiter_score(attn, decoded_tokens),
            'local_window': self._compute_local_window_score(attn),
            'distributed': self._compute_distributed_score(attn),
        }

        # Determine primary type based on scores and thresholds
        # Check pattern-based types first (more specific), then fallback types
        primary_type = HeadType.UNKNOWN
        max_score = 0.0

        if scores['induction'] >= self.induction_threshold and scores['induction'] > max_score:
            primary_type = HeadType.INDUCTION
            max_score = scores['induction']

        if scores['previous_token'] >= self.previous_token_threshold and scores['previous_token'] > max_score:
            primary_type = HeadType.PREVIOUS_TOKEN
            max_score = scores['previous_token']

        if scores['positional'] >= self.positional_threshold and scores['positional'] > max_score:
            primary_type = HeadType.POSITIONAL
            max_score = scores['positional']

        if scores['copying'] >= self.copying_threshold and scores['copying'] > max_score:
            primary_type = HeadType.COPYING
            max_score = scores['copying']

        if scores['bos_attention'] >= self.bos_attention_threshold and scores['bos_attention'] > max_score:
            primary_type = HeadType.BOS_ATTENTION
            max_score = scores['bos_attention']

        if scores['delimiter'] >= self.delimiter_threshold and scores['delimiter'] > max_score:
            primary_type = HeadType.DELIMITER
            max_score = scores['delimiter']

        if scores['local_window'] >= self.local_window_threshold and scores['local_window'] > max_score:
            primary_type = HeadType.LOCAL_WINDOW
            max_score = scores['local_window']

        # Distributed is a fallback - only classify as distributed if no other pattern found
        # and the attention is sufficiently spread out
        if primary_type == HeadType.UNKNOWN and scores['distributed'] >= self.distributed_threshold:
            primary_type = HeadType.DISTRIBUTED
            max_score = scores['distributed']

        # Check for mixed types (multiple high scores among non-distributed types)
        pattern_scores = {k: v for k, v in scores.items() if k != 'distributed'}
        high_scores = sum(1 for s in pattern_scores.values() if s >= 0.25)
        if high_scores >= 2 and primary_type not in (HeadType.UNKNOWN, HeadType.DISTRIBUTED):
            primary_type = HeadType.MIXED

        # Generate pattern summary
        pattern_summary = self._generate_pattern_summary(attn, primary_type, scores)

        return HeadClassification(
            layer=layer,
            head=head,
            primary_type=primary_type,
            confidence=max_score,
            scores=scores,
            pattern_summary=pattern_summary,
        )

    def _compute_induction_score(
        self,
        attn: np.ndarray,
        tokens: list[int],
        decoded_tokens: list[str],
    ) -> float:
        """
        Compute induction head score.

        Induction heads implement the A...B...A -> B pattern:
        - They attend to the token that followed a previous occurrence of the current token.
        - For repeated bigrams tokens[i:i+2] == tokens[j:j+2], check if attn[j+1, i+1] is high.

        Args:
            attn: Attention weights [seq_len, seq_len].
            tokens: Token IDs.
            decoded_tokens: Decoded tokens.

        Returns:
            Induction score between 0 and 1.
        """
        seq_len = len(tokens)
        if seq_len < 4:
            return 0.0

        induction_scores = []

        # Find repeated tokens and check induction pattern
        for j in range(2, seq_len):
            current_token = tokens[j - 1]

            # Find earlier occurrences of the same token
            for i in range(1, j - 1):
                if tokens[i - 1] == current_token:
                    # Check if head attends to position i (token after earlier occurrence)
                    if i < seq_len and j < seq_len:
                        attention_value = attn[j, i]
                        induction_scores.append(attention_value)

        if not induction_scores:
            return 0.0

        return float(np.mean(induction_scores))

    def _compute_previous_token_score(self, attn: np.ndarray) -> float:
        """
        Compute previous token head score.

        Previous token heads attend primarily to the immediately preceding token,
        creating a strong sub-diagonal pattern.

        Args:
            attn: Attention weights [seq_len, seq_len].

        Returns:
            Previous token score between 0 and 1.
        """
        seq_len = attn.shape[0]
        if seq_len < 2:
            return 0.0

        # Extract the sub-diagonal (previous token attention)
        sub_diagonal = []
        for i in range(1, seq_len):
            sub_diagonal.append(attn[i, i - 1])

        if not sub_diagonal:
            return 0.0

        # Score is the mean attention to the previous token
        return float(np.mean(sub_diagonal))

    def _compute_positional_score(self, attn: np.ndarray) -> float:
        """
        Compute positional head score.

        Positional heads attend to tokens at fixed relative offsets,
        creating diagonal or stripe patterns in the attention matrix.

        Args:
            attn: Attention weights [seq_len, seq_len].

        Returns:
            Positional score between 0 and 1.
        """
        seq_len = attn.shape[0]
        if seq_len < 3:
            return 0.0

        # Check various offset patterns (not just -1 which is previous token)
        offset_scores = []

        for offset in [-2, -3, -4, 0]:  # Check specific positions (0 = self-attention)
            diagonal_values = []
            for i in range(seq_len):
                j = i + offset
                if 0 <= j < seq_len:
                    diagonal_values.append(attn[i, j])

            if diagonal_values:
                offset_scores.append(np.mean(diagonal_values))

        if not offset_scores:
            return 0.0

        # Return the maximum offset score (strongest positional pattern)
        return float(max(offset_scores))

    def _compute_copying_score(
        self,
        attn: np.ndarray,
        tokens: list[int],
        decoded_tokens: list[str],
    ) -> float:
        """
        Compute copying head score.

        Copying heads attend to positions with the same or similar tokens,
        potentially for copying or lookup operations.

        Args:
            attn: Attention weights [seq_len, seq_len].
            tokens: Token IDs.
            decoded_tokens: Decoded tokens.

        Returns:
            Copying score between 0 and 1.
        """
        seq_len = len(tokens)
        if seq_len < 2:
            return 0.0

        copying_scores = []

        # For each position, check attention to identical tokens
        for i in range(seq_len):
            for j in range(i):  # Only look at earlier positions
                if tokens[i] == tokens[j]:
                    copying_scores.append(attn[i, j])

        if not copying_scores:
            return 0.0

        return float(np.mean(copying_scores))

    def _compute_bos_attention_score(self, attn: np.ndarray) -> float:
        """
        Compute BOS (beginning of sequence) attention score.

        BOS attention heads strongly attend to the first token, which often
        serves as a "no-op" or information aggregation position.

        Args:
            attn: Attention weights [seq_len, seq_len].

        Returns:
            BOS attention score between 0 and 1.
        """
        seq_len = attn.shape[0]
        if seq_len < 2:
            return 0.0

        # Attention to the first token from all other positions
        bos_attention = attn[1:, 0]  # Exclude self-attention at position 0
        return float(np.mean(bos_attention))

    def _compute_delimiter_score(
        self,
        attn: np.ndarray,
        decoded_tokens: list[str],
    ) -> float:
        """
        Compute delimiter attention score.

        Delimiter heads attend to structural tokens like punctuation,
        which often mark syntactic boundaries.

        Args:
            attn: Attention weights [seq_len, seq_len].
            decoded_tokens: Decoded token strings.

        Returns:
            Delimiter attention score between 0 and 1.
        """
        seq_len = len(decoded_tokens)
        if seq_len < 2:
            return 0.0

        # Common delimiter patterns (handles various tokenizer formats)
        delimiter_chars = {'.', ',', '!', '?', ';', ':', '\n', '(', ')', '[', ']', '{', '}'}

        # Find delimiter positions
        delimiter_positions = []
        for i, token in enumerate(decoded_tokens):
            # Check if token contains delimiter characters
            token_stripped = token.strip()
            if any(c in token_stripped for c in delimiter_chars) or token_stripped in delimiter_chars:
                delimiter_positions.append(i)

        if not delimiter_positions:
            return 0.0

        # Compute mean attention to delimiter positions from non-delimiter positions
        delimiter_attention = []
        for i in range(seq_len):
            if i not in delimiter_positions:
                for j in delimiter_positions:
                    if j < i:  # Only attend to earlier delimiters (causal)
                        delimiter_attention.append(attn[i, j])

        if not delimiter_attention:
            return 0.0

        return float(np.mean(delimiter_attention))

    def _compute_local_window_score(self, attn: np.ndarray) -> float:
        """
        Compute local window attention score.

        Local window heads attend to a range of nearby tokens, not just
        the immediately previous one. This captures broader local context.

        Args:
            attn: Attention weights [seq_len, seq_len].

        Returns:
            Local window score between 0 and 1.
        """
        seq_len = attn.shape[0]
        if seq_len < 4:
            return 0.0

        window_size = min(5, seq_len - 1)  # Look at last 5 tokens or fewer
        local_scores = []

        for i in range(window_size, seq_len):
            # Sum attention to the local window (excluding immediate previous, which is separate)
            window_start = max(0, i - window_size)
            window_end = i - 1  # Exclude immediate previous token
            if window_end > window_start:
                local_attn = np.sum(attn[i, window_start:window_end])
                local_scores.append(local_attn)

        if not local_scores:
            return 0.0

        return float(np.mean(local_scores))

    def _compute_distributed_score(self, attn: np.ndarray) -> float:
        """
        Compute distributed attention score based on entropy.

        Distributed heads spread attention across many tokens rather than
        focusing on specific positions. High entropy indicates distributed attention.

        Args:
            attn: Attention weights [seq_len, seq_len].

        Returns:
            Distributed score between 0 and 1 (higher = more distributed).
        """
        seq_len = attn.shape[0]
        if seq_len < 2:
            return 0.0

        # Compute normalized entropy for each query position
        eps = 1e-10
        max_entropy = np.log(seq_len)  # Maximum possible entropy (uniform distribution)

        if max_entropy < eps:
            return 0.0

        entropy_scores = []
        for i in range(1, seq_len):  # Skip first position (only attends to itself)
            # Get attention distribution for this query
            attn_dist = attn[i, :i+1]  # Only consider causal positions
            attn_dist = attn_dist / (np.sum(attn_dist) + eps)  # Renormalize

            # Compute entropy
            entropy = -np.sum(attn_dist * np.log(attn_dist + eps))
            local_max_entropy = np.log(i + 1)  # Max entropy for this position

            if local_max_entropy > eps:
                normalized_entropy = entropy / local_max_entropy
                entropy_scores.append(normalized_entropy)

        if not entropy_scores:
            return 0.0

        return float(np.mean(entropy_scores))

    def _generate_pattern_summary(
        self,
        attn: np.ndarray,
        head_type: HeadType,
        scores: dict[str, float],
    ) -> str:
        """
        Generate a human-readable summary of the attention pattern.

        Args:
            attn: Attention weights.
            head_type: Classified head type.
            scores: Score dictionary.

        Returns:
            Summary string.
        """
        seq_len = attn.shape[0]

        # Compute basic statistics
        mean_attn = float(np.mean(attn))
        max_attn = float(np.max(attn))

        # Find dominant attention positions
        flat_attn = attn.flatten()
        top_k = min(3, len(flat_attn))
        top_indices = np.argsort(flat_attn)[-top_k:][::-1]

        if head_type == HeadType.INDUCTION:
            return f"Induction pattern (score={scores['induction']:.2f}): Attends to tokens following repeated sequences"
        elif head_type == HeadType.PREVIOUS_TOKEN:
            return f"Previous token pattern (score={scores['previous_token']:.2f}): Strong sub-diagonal attention"
        elif head_type == HeadType.POSITIONAL:
            return f"Positional pattern (score={scores['positional']:.2f}): Fixed offset attention pattern"
        elif head_type == HeadType.COPYING:
            return f"Copying pattern (score={scores['copying']:.2f}): Attends to identical tokens"
        elif head_type == HeadType.BOS_ATTENTION:
            return f"BOS attention pattern (score={scores['bos_attention']:.2f}): Attends to first token"
        elif head_type == HeadType.DELIMITER:
            return f"Delimiter pattern (score={scores['delimiter']:.2f}): Attends to punctuation/structural tokens"
        elif head_type == HeadType.LOCAL_WINDOW:
            return f"Local window pattern (score={scores['local_window']:.2f}): Attends to nearby context"
        elif head_type == HeadType.DISTRIBUTED:
            return f"Distributed pattern (score={scores['distributed']:.2f}): Spreads attention broadly"
        elif head_type == HeadType.MIXED:
            top_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
            return f"Mixed pattern: {top_types[0][0]}={top_types[0][1]:.2f}, {top_types[1][0]}={top_types[1][1]:.2f}"
        else:
            return f"Unknown pattern (max_attn={max_attn:.2f}, mean_attn={mean_attn:.3f})"

    def get_head_importance(
        self,
        activations: dict[str, torch.Tensor],
        num_layers: int,
    ) -> dict[str, float]:
        """
        Compute importance scores for each head based on attention entropy
        and concentration.

        Args:
            activations: Dictionary of captured activations.
            num_layers: Number of layers.

        Returns:
            Dictionary mapping "layer_head" to importance score.
        """
        importance = {}

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

                # Compute entropy of attention distribution
                # Lower entropy = more focused attention = potentially more important
                eps = 1e-10
                entropy = -np.sum(head_attn * np.log(head_attn + eps), axis=-1)
                mean_entropy = float(np.mean(entropy))

                # Compute max attention (higher = more focused)
                max_attn = float(np.max(head_attn))

                # Importance = combination of low entropy and high max attention
                # Normalize to 0-1 range
                importance_score = (1.0 - mean_entropy / 10.0) * 0.5 + max_attn * 0.5
                importance_score = max(0.0, min(1.0, importance_score))

                importance[f"{layer_idx}_{head_idx}"] = importance_score

        return importance
