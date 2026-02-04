"""
Sparse Autoencoder (SAE) implementation for mechanistic interpretability.

This module provides tools for training and analyzing SAEs on transformer activations.
SAEs decompose model activations into sparse, interpretable features.
"""

from .config import SAEConfig, SAEActivationType, ActivationSource
from .model import SparseAutoencoder
from .model_torch import SparseAutoencoderTorch
from .loss import sae_loss, reconstruction_loss, sparsity_loss
from .activation_buffer import ActivationBuffer
from .trainer import SAETrainer, SAETrainingConfig
from .checkpoints import save_sae_checkpoint, load_sae_checkpoint

# Phase 3.2: Analysis tools
from .metrics import (
    SparsityMetrics,
    FeatureHealthMetrics,
    ReconstructionMetrics,
    SAEQualityMetrics,
    compute_sparsity_metrics,
    compute_feature_health_metrics,
    compute_reconstruction_metrics,
    compute_sae_quality_metrics,
    compute_feature_utilization,
    compute_activation_magnitude_distribution,
    compute_feature_correlation_matrix,
    evaluate_sae_on_batch,
)
from .feature_analyzer import (
    FeatureActivation,
    TokenFeatureActivations,
    SequenceFeatureActivations,
    FeatureActivationStats,
    FeatureAnalyzer,
    analyze_feature_activations_batch,
)
from .interpretability import (
    TokenActivationExample,
    FeatureInterpretation,
    FeatureCatalog,
    FeatureInterpreter,
    find_semantically_similar_features,
    cluster_features_by_tokens,
)

# Phase 3.5: External SAE loading
from .loaders import (
    load_external_sae,
    list_external_saes,
    save_loaded_sae,
    ExternalSAESource,
    ExternalSAEMetadata,
    LoadedSAE,
    HuggingFaceLoader,
    GemmaScopeLoader,
    UploadLoader,
)

__all__ = [
    # Config
    "SAEConfig",
    "SAEActivationType",
    "ActivationSource",
    # Models
    "SparseAutoencoder",
    "SparseAutoencoderTorch",
    # Loss
    "sae_loss",
    "reconstruction_loss",
    "sparsity_loss",
    # Training
    "ActivationBuffer",
    "SAETrainer",
    "SAETrainingConfig",
    "save_sae_checkpoint",
    "load_sae_checkpoint",
    # Metrics (Phase 3.2)
    "SparsityMetrics",
    "FeatureHealthMetrics",
    "ReconstructionMetrics",
    "SAEQualityMetrics",
    "compute_sparsity_metrics",
    "compute_feature_health_metrics",
    "compute_reconstruction_metrics",
    "compute_sae_quality_metrics",
    "compute_feature_utilization",
    "compute_activation_magnitude_distribution",
    "compute_feature_correlation_matrix",
    "evaluate_sae_on_batch",
    # Feature Analysis (Phase 3.2)
    "FeatureActivation",
    "TokenFeatureActivations",
    "SequenceFeatureActivations",
    "FeatureActivationStats",
    "FeatureAnalyzer",
    "analyze_feature_activations_batch",
    # Interpretability (Phase 3.2)
    "TokenActivationExample",
    "FeatureInterpretation",
    "FeatureCatalog",
    "FeatureInterpreter",
    "find_semantically_similar_features",
    "cluster_features_by_tokens",
    # External SAE Loading (Phase 3.5)
    "load_external_sae",
    "list_external_saes",
    "save_loaded_sae",
    "ExternalSAESource",
    "ExternalSAEMetadata",
    "LoadedSAE",
    "HuggingFaceLoader",
    "GemmaScopeLoader",
    "UploadLoader",
]
