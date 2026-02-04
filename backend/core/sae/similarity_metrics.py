"""
Representation similarity metrics for comparing SAE features and layers.

Implements research-standard metrics for comparing neural network representations:
- Linear CKA (Centered Kernel Alignment)
- Cosine similarity matrices
- Activation correlation
- Representation similarity via PWCCA-inspired approach
"""

import numpy as np
from typing import Union, Tuple, List, Dict
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def linear_cka(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment) between two representations.

    CKA measures similarity between representations in a way that is invariant
    to orthogonal transformations and isotropic scaling. This makes it ideal
    for comparing representations across layers or models.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited" (ICML 2019)

    Args:
        X: First representation matrix (n_samples, d_x)
        Y: Second representation matrix (n_samples, d_y)

    Returns:
        CKA similarity score in [0, 1]
    """
    X = _to_numpy(X).astype(np.float64)
    Y = _to_numpy(Y).astype(np.float64)

    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute HSIC (Hilbert-Schmidt Independence Criterion) with linear kernel
    # HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    XtX = X.T @ X  # (d_x, d_x)
    YtY = Y.T @ Y  # (d_y, d_y)
    XtY = X.T @ Y  # (d_x, d_y)

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-12:
        return 0.0

    return float(hsic_xy / denominator)


def cosine_similarity_matrix(
    A: Union[np.ndarray, torch.Tensor],
    B: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between feature vectors.

    Used for comparing decoder directions of SAE features within the same
    activation space.

    Args:
        A: Feature vectors from first SAE (n_features_a, d)
        B: Feature vectors from second SAE (n_features_b, d)

    Returns:
        Cosine similarity matrix (n_features_a, n_features_b) in [-1, 1]
    """
    A = _to_numpy(A).astype(np.float64)
    B = _to_numpy(B).astype(np.float64)

    assert A.shape[1] == B.shape[1], "Feature dimensions must match"

    # Normalize rows
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)

    A_normalized = A / np.maximum(A_norms, 1e-8)
    B_normalized = B / np.maximum(B_norms, 1e-8)

    return A_normalized @ B_normalized.T


def activation_correlation_matrix(
    hidden_a: Union[np.ndarray, torch.Tensor],
    hidden_b: Union[np.ndarray, torch.Tensor],
    top_k_a: int = 100,
    top_k_b: int = 100,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute Pearson correlation between feature activation patterns.

    For cross-model comparison where decoder directions live in different spaces,
    we compare features by how similarly they activate across shared input tokens.

    Args:
        hidden_a: SAE A hidden activations (n_tokens, d_hidden_a)
        hidden_b: SAE B hidden activations (n_tokens, d_hidden_b)
        top_k_a: Number of top features to consider from SAE A
        top_k_b: Number of top features to consider from SAE B

    Returns:
        Tuple of (correlation_matrix, feature_indices_a, feature_indices_b)
    """
    hidden_a = _to_numpy(hidden_a).astype(np.float64)
    hidden_b = _to_numpy(hidden_b).astype(np.float64)

    assert hidden_a.shape[0] == hidden_b.shape[0], "Must have same number of tokens"

    # Select top features by activation frequency
    freq_a = (hidden_a > 0).mean(axis=0)
    freq_b = (hidden_b > 0).mean(axis=0)

    top_k_a = min(top_k_a, hidden_a.shape[1])
    top_k_b = min(top_k_b, hidden_b.shape[1])

    indices_a = np.argsort(freq_a)[-top_k_a:][::-1].copy()
    indices_b = np.argsort(freq_b)[-top_k_b:][::-1].copy()

    # Extract top features
    feats_a = hidden_a[:, indices_a]  # (n_tokens, top_k_a)
    feats_b = hidden_b[:, indices_b]  # (n_tokens, top_k_b)

    # Compute correlation matrix
    n = feats_a.shape[0]
    mean_a = feats_a.mean(axis=0, keepdims=True)
    mean_b = feats_b.mean(axis=0, keepdims=True)
    std_a = feats_a.std(axis=0, keepdims=True)
    std_b = feats_b.std(axis=0, keepdims=True)

    # Normalize
    normed_a = (feats_a - mean_a) / np.maximum(std_a, 1e-8)
    normed_b = (feats_b - mean_b) / np.maximum(std_b, 1e-8)

    # Pearson correlation = (1/n) * normed_a.T @ normed_b
    corr_matrix = (normed_a.T @ normed_b) / n

    return corr_matrix, indices_a.tolist(), indices_b.tolist()


def compute_representation_similarity(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute multiple representation similarity metrics between two sets of activations.

    Args:
        X: First representation (n_samples, d_x)
        Y: Second representation (n_samples, d_y)

    Returns:
        Dictionary of similarity metrics
    """
    X = _to_numpy(X).astype(np.float64)
    Y = _to_numpy(Y).astype(np.float64)

    assert X.shape[0] == Y.shape[0], "Must have same number of samples"

    cka_score = linear_cka(X, Y)

    # Mean activation correlation (using top features)
    corr_mat, _, _ = activation_correlation_matrix(X, Y, top_k_a=50, top_k_b=50)
    # For each feature in A, find its best match in B
    max_corrs = np.max(np.abs(corr_mat), axis=1)
    mean_max_correlation = float(max_corrs.mean())

    # Representational similarity via shared variance
    # Project both into shared space via SVD
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Singular values of cross-covariance
    cross_cov = X_centered.T @ Y_centered / X.shape[0]
    try:
        singular_values = np.linalg.svd(cross_cov, compute_uv=False)
        # Fraction of shared variance
        total_var_x = np.sum(np.var(X_centered, axis=0))
        total_var_y = np.sum(np.var(Y_centered, axis=0))
        shared_variance = float(
            np.sum(singular_values ** 2) / max(np.sqrt(total_var_x * total_var_y), 1e-8)
        )
    except np.linalg.LinAlgError:
        shared_variance = 0.0

    return {
        "cka": cka_score,
        "mean_max_correlation": mean_max_correlation,
        "shared_variance": min(shared_variance, 1.0),
    }


def compute_layer_similarity_matrix(
    layer_activations_a: List[Union[np.ndarray, torch.Tensor]],
    layer_activations_b: List[Union[np.ndarray, torch.Tensor]],
) -> np.ndarray:
    """
    Compute CKA similarity matrix between all pairs of layers.

    Args:
        layer_activations_a: List of activation matrices, one per layer of model A
        layer_activations_b: List of activation matrices, one per layer of model B

    Returns:
        CKA similarity matrix (num_layers_a, num_layers_b)
    """
    n_a = len(layer_activations_a)
    n_b = len(layer_activations_b)
    matrix = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            matrix[i, j] = linear_cka(layer_activations_a[i], layer_activations_b[j])

    return matrix
