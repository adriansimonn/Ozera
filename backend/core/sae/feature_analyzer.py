"""
Feature activation analysis for Sparse Autoencoders.

Provides tools for analyzing which features activate for given inputs,
computing activation patterns, and identifying important features.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch


@dataclass
class FeatureActivation:
    """Information about a single feature's activation."""

    feature_idx: int
    activation_value: float
    rank: int  # Rank by activation magnitude
    percentile: float  # Percentile of this activation compared to feature's history


@dataclass
class TokenFeatureActivations:
    """Feature activations for a single token position."""

    position: int
    token: str
    token_id: int
    top_features: List[FeatureActivation]
    total_active_features: int
    l0_sparsity: int
    l1_norm: float


@dataclass
class SequenceFeatureActivations:
    """Feature activations for an entire sequence."""

    tokens: List[str]
    token_ids: List[int]
    per_token_activations: List[TokenFeatureActivations]
    feature_activation_matrix: List[List[float]]  # (seq_len, d_hidden) as nested list
    active_features_per_position: List[int]  # L0 per position
    most_active_features: List[int]  # Feature indices sorted by total activation


@dataclass
class FeatureActivationStats:
    """Statistics about a feature's activation patterns."""

    feature_idx: int
    activation_frequency: float  # Fraction of inputs where this feature activates
    mean_activation: float  # Mean activation when active
    max_activation: float  # Maximum activation observed
    std_activation: float  # Standard deviation when active
    top_activating_positions: List[int]  # Positions where this feature activates most
    position_distribution: List[float]  # Activation frequency per position


class FeatureAnalyzer:
    """
    Analyzer for SAE feature activations.

    Provides methods to analyze which features activate for given inputs,
    identify patterns, and track feature behavior across examples.
    """

    def __init__(self, sae, device: str = "cpu"):
        """
        Initialize the feature analyzer.

        Args:
            sae: Trained SparseAutoencoderTorch model
            device: Device to run inference on
        """
        self.sae = sae
        self.device = device
        self.d_hidden = sae.d_hidden

        # Move SAE to device
        self.sae = self.sae.to(device)
        self.sae.eval()

        # Statistics accumulated over examples
        self.feature_activation_counts = np.zeros(self.d_hidden)
        self.feature_activation_sums = np.zeros(self.d_hidden)
        self.feature_activation_max = np.zeros(self.d_hidden)
        self.feature_activation_sq_sums = np.zeros(self.d_hidden)
        self.total_examples = 0

    def analyze_sequence(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        tokens: List[str],
        token_ids: List[int],
        top_k: int = 10,
    ) -> SequenceFeatureActivations:
        """
        Analyze feature activations for a sequence.

        Args:
            activations: Model activations (seq_len, d_input)
            tokens: Token strings
            token_ids: Token IDs
            top_k: Number of top features to return per token

        Returns:
            SequenceFeatureActivations with detailed analysis
        """
        # Ensure tensor
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).float()

        activations = activations.to(self.device)

        # Handle 3D input (batch, seq, hidden) by squeezing batch
        if activations.dim() == 3 and activations.shape[0] == 1:
            activations = activations.squeeze(0)

        seq_len = activations.shape[0]

        # Get SAE hidden activations
        with torch.no_grad():
            hidden = self.sae.encode(activations)

        hidden_np = hidden.cpu().numpy()

        # Per-token analysis
        per_token_activations = []
        active_features_per_position = []

        for pos in range(seq_len):
            token_hidden = hidden_np[pos]
            active_mask = token_hidden > 0
            active_indices = np.where(active_mask)[0]
            active_values = token_hidden[active_mask]

            # Sort by activation value
            sorted_order = np.argsort(active_values)[::-1]
            sorted_indices = active_indices[sorted_order]
            sorted_values = active_values[sorted_order]

            # Top-k features
            top_features = []
            for rank, (feat_idx, feat_val) in enumerate(
                zip(sorted_indices[:top_k], sorted_values[:top_k])
            ):
                percentile = self._compute_percentile(feat_idx, feat_val)
                top_features.append(
                    FeatureActivation(
                        feature_idx=int(feat_idx),
                        activation_value=float(feat_val),
                        rank=rank,
                        percentile=percentile,
                    )
                )

            token_activations = TokenFeatureActivations(
                position=pos,
                token=tokens[pos] if pos < len(tokens) else "",
                token_id=token_ids[pos] if pos < len(token_ids) else 0,
                top_features=top_features,
                total_active_features=len(active_indices),
                l0_sparsity=len(active_indices),
                l1_norm=float(np.abs(token_hidden).sum()),
            )
            per_token_activations.append(token_activations)
            active_features_per_position.append(len(active_indices))

        # Find most active features across sequence
        total_activation_per_feature = np.abs(hidden_np).sum(axis=0)
        most_active_features = np.argsort(total_activation_per_feature)[::-1].tolist()

        return SequenceFeatureActivations(
            tokens=tokens,
            token_ids=token_ids,
            per_token_activations=per_token_activations,
            feature_activation_matrix=hidden_np.tolist(),
            active_features_per_position=active_features_per_position,
            most_active_features=most_active_features[:100],  # Top 100
        )

    def _compute_percentile(self, feature_idx: int, value: float) -> float:
        """Compute percentile of activation value for a feature."""
        if self.total_examples == 0:
            return 50.0  # No history, return median

        # Simple approximation using mean and std
        mean = self.feature_activation_sums[feature_idx] / max(
            self.feature_activation_counts[feature_idx], 1
        )
        if self.feature_activation_counts[feature_idx] < 2:
            return 50.0

        # Use z-score as proxy for percentile
        variance = (
            self.feature_activation_sq_sums[feature_idx]
            / self.feature_activation_counts[feature_idx]
        ) - mean**2
        std = np.sqrt(max(variance, 1e-8))

        z_score = (value - mean) / max(std, 1e-8)
        # Convert z-score to approximate percentile using normal CDF
        percentile = 50 * (1 + np.tanh(z_score * 0.7))  # Approximate
        return float(np.clip(percentile, 0, 100))

    def update_statistics(
        self, hidden_activations: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Update running statistics with new batch of activations.

        Args:
            hidden_activations: Hidden activations (batch_size, d_hidden)
        """
        if isinstance(hidden_activations, torch.Tensor):
            hidden_activations = hidden_activations.detach().cpu().numpy()

        # Flatten if 3D
        if hidden_activations.ndim == 3:
            hidden_activations = hidden_activations.reshape(-1, hidden_activations.shape[-1])

        batch_size = hidden_activations.shape[0]
        active_mask = hidden_activations > 0

        # Update counts (number of times each feature activated)
        self.feature_activation_counts += active_mask.sum(axis=0)

        # Update sums (total activation per feature)
        self.feature_activation_sums += (hidden_activations * active_mask).sum(axis=0)

        # Update squared sums (for variance calculation)
        self.feature_activation_sq_sums += (
            (hidden_activations**2) * active_mask
        ).sum(axis=0)

        # Update max
        batch_max = hidden_activations.max(axis=0)
        self.feature_activation_max = np.maximum(
            self.feature_activation_max, batch_max
        )

        self.total_examples += batch_size

    def get_feature_stats(self, feature_idx: int) -> FeatureActivationStats:
        """
        Get statistics for a specific feature.

        Args:
            feature_idx: Index of the feature

        Returns:
            FeatureActivationStats for the feature
        """
        if self.total_examples == 0:
            return FeatureActivationStats(
                feature_idx=feature_idx,
                activation_frequency=0.0,
                mean_activation=0.0,
                max_activation=0.0,
                std_activation=0.0,
                top_activating_positions=[],
                position_distribution=[],
            )

        count = self.feature_activation_counts[feature_idx]
        activation_frequency = count / self.total_examples

        if count == 0:
            return FeatureActivationStats(
                feature_idx=feature_idx,
                activation_frequency=0.0,
                mean_activation=0.0,
                max_activation=0.0,
                std_activation=0.0,
                top_activating_positions=[],
                position_distribution=[],
            )

        mean_activation = self.feature_activation_sums[feature_idx] / count
        max_activation = self.feature_activation_max[feature_idx]

        variance = (self.feature_activation_sq_sums[feature_idx] / count) - mean_activation**2
        std_activation = float(np.sqrt(max(variance, 0)))

        return FeatureActivationStats(
            feature_idx=feature_idx,
            activation_frequency=float(activation_frequency),
            mean_activation=float(mean_activation),
            max_activation=float(max_activation),
            std_activation=std_activation,
            top_activating_positions=[],  # Would need position tracking
            position_distribution=[],
        )

    def get_all_feature_stats(self) -> List[FeatureActivationStats]:
        """Get statistics for all features."""
        return [self.get_feature_stats(i) for i in range(self.d_hidden)]

    def find_top_features_for_input(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        """
        Find the top-k most active features for an input.

        Args:
            activations: Input activations (d_input,) or (batch, d_input)
            top_k: Number of top features to return

        Returns:
            List of (feature_idx, activation_value) tuples
        """
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).float()

        activations = activations.to(self.device)

        # Handle different input shapes
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        with torch.no_grad():
            hidden = self.sae.encode(activations)

        hidden_np = hidden.cpu().numpy()

        # Average over batch if needed
        if hidden_np.shape[0] > 1:
            hidden_avg = hidden_np.mean(axis=0)
        else:
            hidden_avg = hidden_np[0]

        # Get top-k
        top_indices = np.argsort(hidden_avg)[-top_k:][::-1]
        top_values = hidden_avg[top_indices]

        return [(int(idx), float(val)) for idx, val in zip(top_indices, top_values)]

    def compute_feature_activation_heatmap(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        feature_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute activation heatmap for features across positions.

        Args:
            activations: Input activations (seq_len, d_input)
            feature_indices: Optional list of feature indices to include

        Returns:
            Heatmap array (seq_len, num_features)
        """
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).float()

        activations = activations.to(self.device)

        with torch.no_grad():
            hidden = self.sae.encode(activations)

        hidden_np = hidden.cpu().numpy()

        if feature_indices is not None:
            return hidden_np[:, feature_indices]
        return hidden_np

    def find_features_by_pattern(
        self,
        hidden_activations: Union[np.ndarray, torch.Tensor],
        tokens: List[str],
        pattern: str = "first_token",
    ) -> List[int]:
        """
        Find features that match a specific activation pattern.

        Patterns:
        - "first_token": Features that activate strongly on first token
        - "last_token": Features that activate strongly on last token
        - "uniform": Features that activate uniformly across positions
        - "position_specific": Features with strong position preference

        Args:
            hidden_activations: Hidden activations (seq_len, d_hidden)
            tokens: Token strings
            pattern: Pattern type to search for

        Returns:
            List of matching feature indices
        """
        if isinstance(hidden_activations, torch.Tensor):
            hidden_activations = hidden_activations.detach().cpu().numpy()

        seq_len, d_hidden = hidden_activations.shape
        matching_features = []

        for feat_idx in range(d_hidden):
            feat_acts = hidden_activations[:, feat_idx]

            if feat_acts.max() < 1e-6:
                continue  # Skip dead features

            if pattern == "first_token":
                # Strong activation on first token
                if feat_acts[0] > feat_acts.mean() * 2:
                    matching_features.append(feat_idx)

            elif pattern == "last_token":
                # Strong activation on last token
                if feat_acts[-1] > feat_acts.mean() * 2:
                    matching_features.append(feat_idx)

            elif pattern == "uniform":
                # Low variance relative to mean
                if feat_acts.mean() > 0:
                    cv = feat_acts.std() / feat_acts.mean()  # Coefficient of variation
                    if cv < 0.5:
                        matching_features.append(feat_idx)

            elif pattern == "position_specific":
                # High variance, indicating position-specific activation
                if feat_acts.mean() > 0:
                    cv = feat_acts.std() / feat_acts.mean()
                    if cv > 1.5:
                        matching_features.append(feat_idx)

        return matching_features


def analyze_feature_activations_batch(
    sae,
    activations_batch: Union[np.ndarray, torch.Tensor],
    tokens_batch: Optional[List[List[str]]] = None,
    device: str = "cpu",
) -> Dict:
    """
    Analyze feature activations for a batch of inputs.

    Args:
        sae: Trained SparseAutoencoderTorch model
        activations_batch: Batch of activations (batch, seq_len, d_input)
        tokens_batch: Optional batch of token strings
        device: Device to run on

    Returns:
        Dictionary with batch-level analysis
    """
    analyzer = FeatureAnalyzer(sae, device)

    if isinstance(activations_batch, np.ndarray):
        activations_batch = torch.from_numpy(activations_batch).float()

    activations_batch = activations_batch.to(device)

    # Flatten for global statistics
    batch_size, seq_len, d_input = activations_batch.shape
    flat_activations = activations_batch.view(-1, d_input)

    with torch.no_grad():
        hidden_flat = sae.encode(flat_activations)

    # Update statistics
    analyzer.update_statistics(hidden_flat)

    # Reshape back
    hidden_reshaped = hidden_flat.view(batch_size, seq_len, -1).cpu().numpy()

    # Compute aggregate metrics
    active_mask = hidden_reshaped > 0
    avg_l0_per_token = active_mask.sum(axis=2).mean()
    feature_frequencies = active_mask.mean(axis=(0, 1))

    # Find most frequently active features
    top_features = np.argsort(feature_frequencies)[-50:][::-1].tolist()

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_l0_per_token": float(avg_l0_per_token),
        "total_active_features": int((feature_frequencies > 0).sum()),
        "top_features": top_features,
        "feature_frequencies": feature_frequencies.tolist(),
    }
