"""
SAE quality metrics for evaluation and monitoring.

Provides comprehensive metrics for evaluating SAE reconstruction quality,
sparsity characteristics, and feature health.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch

from .config import SAEConfig


@dataclass
class SparsityMetrics:
    """Sparsity-related metrics for SAE activations."""

    avg_l0: float  # Average number of active features per input
    l0_std: float  # Standard deviation of L0
    sparsity_fraction: float  # Fraction of hidden units active on average
    avg_l1: float  # Average L1 norm of hidden activations
    max_activation: float  # Maximum activation value observed


@dataclass
class FeatureHealthMetrics:
    """Health metrics for SAE features."""

    num_features: int  # Total number of features
    dead_features: int  # Features that never activate
    dead_feature_fraction: float  # Fraction of dead features
    low_frequency_features: int  # Features activating rarely
    high_frequency_features: int  # Features activating very often
    feature_frequency_distribution: List[float]  # Activation frequency per feature
    feature_magnitude_distribution: List[float]  # Mean activation magnitude per feature


@dataclass
class ReconstructionMetrics:
    """Reconstruction quality metrics."""

    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    normalized_mse: float  # MSE normalized by input variance
    explained_variance: float  # Fraction of variance explained
    cosine_similarity: float  # Average cosine similarity between input and reconstruction
    relative_reconstruction_error: float  # ||x - x_hat|| / ||x||


@dataclass
class SAEQualityMetrics:
    """Complete quality metrics for an SAE."""

    sparsity: SparsityMetrics
    feature_health: FeatureHealthMetrics
    reconstruction: ReconstructionMetrics
    config: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sparsity": {
                "avg_l0": self.sparsity.avg_l0,
                "l0_std": self.sparsity.l0_std,
                "sparsity_fraction": self.sparsity.sparsity_fraction,
                "avg_l1": self.sparsity.avg_l1,
                "max_activation": self.sparsity.max_activation,
            },
            "feature_health": {
                "num_features": self.feature_health.num_features,
                "dead_features": self.feature_health.dead_features,
                "dead_feature_fraction": self.feature_health.dead_feature_fraction,
                "low_frequency_features": self.feature_health.low_frequency_features,
                "high_frequency_features": self.feature_health.high_frequency_features,
                "feature_frequency_distribution": self.feature_health.feature_frequency_distribution,
                "feature_magnitude_distribution": self.feature_health.feature_magnitude_distribution,
            },
            "reconstruction": {
                "mse": self.reconstruction.mse,
                "rmse": self.reconstruction.rmse,
                "normalized_mse": self.reconstruction.normalized_mse,
                "explained_variance": self.reconstruction.explained_variance,
                "cosine_similarity": self.reconstruction.cosine_similarity,
                "relative_reconstruction_error": self.reconstruction.relative_reconstruction_error,
            },
            "config": self.config,
        }


def compute_sparsity_metrics(
    hidden_activations: Union[np.ndarray, torch.Tensor],
) -> SparsityMetrics:
    """
    Compute sparsity metrics for hidden activations.

    Args:
        hidden_activations: Hidden layer activations (batch_size, d_hidden)

    Returns:
        SparsityMetrics object
    """
    # Convert to numpy if needed
    if isinstance(hidden_activations, torch.Tensor):
        hidden_activations = hidden_activations.detach().cpu().numpy()

    # L0: number of active (non-zero) features per sample
    active_mask = hidden_activations > 0
    l0_per_sample = active_mask.sum(axis=1)
    avg_l0 = float(l0_per_sample.mean())
    l0_std = float(l0_per_sample.std())

    # Sparsity fraction
    d_hidden = hidden_activations.shape[1]
    sparsity_fraction = avg_l0 / d_hidden

    # L1: sum of absolute values per sample
    l1_per_sample = np.abs(hidden_activations).sum(axis=1)
    avg_l1 = float(l1_per_sample.mean())

    # Maximum activation
    max_activation = float(hidden_activations.max())

    return SparsityMetrics(
        avg_l0=avg_l0,
        l0_std=l0_std,
        sparsity_fraction=sparsity_fraction,
        avg_l1=avg_l1,
        max_activation=max_activation,
    )


def compute_feature_health_metrics(
    hidden_activations: Union[np.ndarray, torch.Tensor],
    low_frequency_threshold: float = 0.001,
    high_frequency_threshold: float = 0.9,
) -> FeatureHealthMetrics:
    """
    Compute feature health metrics.

    Args:
        hidden_activations: Hidden layer activations (batch_size, d_hidden)
        low_frequency_threshold: Fraction below which a feature is "low frequency"
        high_frequency_threshold: Fraction above which a feature is "high frequency"

    Returns:
        FeatureHealthMetrics object
    """
    # Convert to numpy if needed
    if isinstance(hidden_activations, torch.Tensor):
        hidden_activations = hidden_activations.detach().cpu().numpy()

    batch_size, d_hidden = hidden_activations.shape

    # Activation mask
    active_mask = hidden_activations > 0

    # Per-feature activation frequency
    feature_frequencies = active_mask.mean(axis=0)

    # Dead features (never activate)
    dead_mask = feature_frequencies == 0
    dead_features = int(dead_mask.sum())
    dead_feature_fraction = dead_features / d_hidden

    # Low and high frequency features
    low_freq_mask = (feature_frequencies > 0) & (feature_frequencies < low_frequency_threshold)
    high_freq_mask = feature_frequencies > high_frequency_threshold
    low_frequency_features = int(low_freq_mask.sum())
    high_frequency_features = int(high_freq_mask.sum())

    # Per-feature mean magnitude (only when active)
    # Avoid division by zero for dead features
    active_counts = active_mask.sum(axis=0)
    active_counts = np.where(active_counts == 0, 1, active_counts)  # Prevent div by zero
    feature_magnitudes = (hidden_activations * active_mask).sum(axis=0) / active_counts
    feature_magnitudes = np.where(dead_mask, 0, feature_magnitudes)

    return FeatureHealthMetrics(
        num_features=d_hidden,
        dead_features=dead_features,
        dead_feature_fraction=dead_feature_fraction,
        low_frequency_features=low_frequency_features,
        high_frequency_features=high_frequency_features,
        feature_frequency_distribution=feature_frequencies.tolist(),
        feature_magnitude_distribution=feature_magnitudes.tolist(),
    )


def compute_reconstruction_metrics(
    inputs: Union[np.ndarray, torch.Tensor],
    reconstructions: Union[np.ndarray, torch.Tensor],
) -> ReconstructionMetrics:
    """
    Compute reconstruction quality metrics.

    Args:
        inputs: Original input activations (batch_size, d_input)
        reconstructions: Reconstructed activations (batch_size, d_input)

    Returns:
        ReconstructionMetrics object
    """
    # Convert to numpy if needed
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()

    # MSE and RMSE
    errors = inputs - reconstructions
    mse = float((errors ** 2).mean())
    rmse = float(np.sqrt(mse))

    # Normalized MSE (by input variance)
    input_variance = float(inputs.var())
    normalized_mse = mse / max(input_variance, 1e-8)

    # Explained variance
    total_variance = float(((inputs - inputs.mean()) ** 2).sum())
    residual_variance = float((errors ** 2).sum())
    explained_variance = 1.0 - (residual_variance / max(total_variance, 1e-8))
    explained_variance = max(0.0, min(1.0, explained_variance))  # Clamp to [0, 1]

    # Cosine similarity
    input_norms = np.linalg.norm(inputs, axis=1, keepdims=True)
    recon_norms = np.linalg.norm(reconstructions, axis=1, keepdims=True)
    input_normalized = inputs / np.maximum(input_norms, 1e-8)
    recon_normalized = reconstructions / np.maximum(recon_norms, 1e-8)
    cosine_sims = (input_normalized * recon_normalized).sum(axis=1)
    cosine_similarity = float(cosine_sims.mean())

    # Relative reconstruction error
    input_norm = float(np.linalg.norm(inputs))
    error_norm = float(np.linalg.norm(errors))
    relative_reconstruction_error = error_norm / max(input_norm, 1e-8)

    return ReconstructionMetrics(
        mse=mse,
        rmse=rmse,
        normalized_mse=normalized_mse,
        explained_variance=explained_variance,
        cosine_similarity=cosine_similarity,
        relative_reconstruction_error=relative_reconstruction_error,
    )


def compute_sae_quality_metrics(
    inputs: Union[np.ndarray, torch.Tensor],
    reconstructions: Union[np.ndarray, torch.Tensor],
    hidden_activations: Union[np.ndarray, torch.Tensor],
    config: Optional[SAEConfig] = None,
) -> SAEQualityMetrics:
    """
    Compute complete SAE quality metrics.

    Args:
        inputs: Original input activations (batch_size, d_input)
        reconstructions: Reconstructed activations (batch_size, d_input)
        hidden_activations: Hidden layer activations (batch_size, d_hidden)
        config: Optional SAE configuration

    Returns:
        SAEQualityMetrics object
    """
    sparsity = compute_sparsity_metrics(hidden_activations)
    feature_health = compute_feature_health_metrics(hidden_activations)
    reconstruction = compute_reconstruction_metrics(inputs, reconstructions)

    return SAEQualityMetrics(
        sparsity=sparsity,
        feature_health=feature_health,
        reconstruction=reconstruction,
        config=config.to_dict() if config else None,
    )


def compute_feature_utilization(
    hidden_activations: Union[np.ndarray, torch.Tensor],
    num_bins: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute feature utilization distribution.

    Returns histogram of feature activation frequencies.

    Args:
        hidden_activations: Hidden layer activations (batch_size, d_hidden)
        num_bins: Number of bins for histogram

    Returns:
        Dictionary with bin_edges and counts
    """
    if isinstance(hidden_activations, torch.Tensor):
        hidden_activations = hidden_activations.detach().cpu().numpy()

    # Compute per-feature activation frequency
    active_mask = hidden_activations > 0
    feature_frequencies = active_mask.mean(axis=0)

    # Create histogram
    counts, bin_edges = np.histogram(feature_frequencies, bins=num_bins, range=(0, 1))

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
    }


def compute_activation_magnitude_distribution(
    hidden_activations: Union[np.ndarray, torch.Tensor],
    num_bins: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute distribution of activation magnitudes.

    Args:
        hidden_activations: Hidden layer activations (batch_size, d_hidden)
        num_bins: Number of bins for histogram

    Returns:
        Dictionary with bin_edges and counts
    """
    if isinstance(hidden_activations, torch.Tensor):
        hidden_activations = hidden_activations.detach().cpu().numpy()

    # Get only non-zero activations
    active_values = hidden_activations[hidden_activations > 0]

    if len(active_values) == 0:
        return {"bin_edges": [0.0, 1.0], "counts": [0]}

    # Create histogram
    counts, bin_edges = np.histogram(active_values, bins=num_bins)

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
    }


def compute_feature_correlation_matrix(
    hidden_activations: Union[np.ndarray, torch.Tensor],
    top_k: int = 100,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute correlation matrix between most active features.

    Args:
        hidden_activations: Hidden layer activations (batch_size, d_hidden)
        top_k: Number of top features to include

    Returns:
        Tuple of (correlation_matrix, feature_indices)
    """
    if isinstance(hidden_activations, torch.Tensor):
        hidden_activations = hidden_activations.detach().cpu().numpy()

    # Find top-k most frequently active features
    active_mask = hidden_activations > 0
    feature_frequencies = active_mask.mean(axis=0)
    top_indices = np.argsort(feature_frequencies)[-top_k:][::-1]

    # Extract top features
    top_features = hidden_activations[:, top_indices]

    # Compute correlation matrix
    # Avoid issues with constant features
    stds = top_features.std(axis=0)
    valid_mask = stds > 1e-8

    if valid_mask.sum() < 2:
        return np.eye(len(top_indices)), top_indices.tolist()

    # Only compute correlations for valid features
    valid_features = top_features[:, valid_mask]
    correlation_matrix = np.corrcoef(valid_features.T)

    # Handle NaN values
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    return correlation_matrix, top_indices.tolist()


def evaluate_sae_on_batch(
    sae,
    batch: Union[np.ndarray, torch.Tensor],
    device: str = "cpu",
) -> SAEQualityMetrics:
    """
    Evaluate SAE quality on a batch of inputs.

    Args:
        sae: SparseAutoencoderTorch model
        batch: Input activations (batch_size, d_input)
        device: Device to run on

    Returns:
        SAEQualityMetrics object
    """
    import torch

    # Ensure tensor
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch).float()

    batch = batch.to(device)
    sae = sae.to(device)
    sae.eval()

    with torch.no_grad():
        reconstructions, hidden = sae(batch, return_hidden=True)

    return compute_sae_quality_metrics(
        inputs=batch,
        reconstructions=reconstructions,
        hidden_activations=hidden,
        config=sae.config if hasattr(sae, 'config') else None,
    )
