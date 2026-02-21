#!/usr/bin/env python3
"""
Test script for SAE Feature Analysis Tools (Phase 3, Step 2).

Verifies:
- analyze_activations() returns sensible feature activations
- Metrics are computed correctly (L0 ~10-100 for realistic SAEs)
- Interpretability tools work correctly
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/adrian/Documents/ozera/backend")

from core.sae.config import SAEConfig, SAEActivationType
from core.sae.model_torch import SparseAutoencoderTorch
from core.sae.metrics import (
    compute_sparsity_metrics,
    compute_feature_health_metrics,
    compute_reconstruction_metrics,
    compute_sae_quality_metrics,
    evaluate_sae_on_batch,
)
from core.sae.feature_analyzer import (
    FeatureAnalyzer,
    analyze_feature_activations_batch,
)
from core.sae.interpretability import (
    FeatureInterpreter,
    find_semantically_similar_features,
)


def test_metrics():
    """Test SAE quality metrics computation."""
    print("\n=== Testing Metrics ===\n")

    # Create test data
    batch_size = 256
    d_input = 192
    d_hidden = 1536

    # Random inputs
    inputs = np.random.randn(batch_size, d_input).astype(np.float32)

    # Simulated SAE outputs (reconstruction close to input)
    noise = np.random.randn(batch_size, d_input).astype(np.float32) * 0.1
    reconstructions = inputs + noise

    # Simulated sparse hidden activations (L0 around 30-50)
    hidden = np.zeros((batch_size, d_hidden), dtype=np.float32)
    for i in range(batch_size):
        # Randomly activate 30-50 features per sample
        n_active = np.random.randint(30, 51)
        active_indices = np.random.choice(d_hidden, n_active, replace=False)
        hidden[i, active_indices] = np.random.exponential(1.0, n_active)

    # Test sparsity metrics
    sparsity = compute_sparsity_metrics(hidden)
    print(f"Sparsity Metrics:")
    print(f"  avg_l0: {sparsity.avg_l0:.1f} (expected ~40)")
    print(f"  l0_std: {sparsity.l0_std:.1f}")
    print(f"  sparsity_fraction: {sparsity.sparsity_fraction:.4f}")
    print(f"  avg_l1: {sparsity.avg_l1:.2f}")
    print(f"  max_activation: {sparsity.max_activation:.2f}")

    assert 25 < sparsity.avg_l0 < 55, f"L0 out of range: {sparsity.avg_l0}"
    print("  L0 in expected range (25-55)")

    # Test feature health metrics
    health = compute_feature_health_metrics(hidden)
    print(f"\nFeature Health Metrics:")
    print(f"  num_features: {health.num_features}")
    print(f"  dead_features: {health.dead_features}")
    print(f"  dead_feature_fraction: {health.dead_feature_fraction:.4f}")
    print(f"  low_frequency_features: {health.low_frequency_features}")
    print(f"  high_frequency_features: {health.high_frequency_features}")

    # Test reconstruction metrics
    recon = compute_reconstruction_metrics(inputs, reconstructions)
    print(f"\nReconstruction Metrics:")
    print(f"  mse: {recon.mse:.6f}")
    print(f"  rmse: {recon.rmse:.6f}")
    print(f"  normalized_mse: {recon.normalized_mse:.6f}")
    print(f"  explained_variance: {recon.explained_variance:.4f}")
    print(f"  cosine_similarity: {recon.cosine_similarity:.4f}")

    assert recon.explained_variance > 0.8, f"Low explained variance: {recon.explained_variance}"
    print("  Explained variance > 0.8")

    # Test combined metrics
    quality = compute_sae_quality_metrics(inputs, reconstructions, hidden)
    print(f"\nSAE Quality Metrics (combined):")
    quality_dict = quality.to_dict()
    print(f"  sparsity.avg_l0: {quality_dict['sparsity']['avg_l0']:.1f}")
    print(f"  reconstruction.mse: {quality_dict['reconstruction']['mse']:.6f}")
    print("  Combined metrics computed successfully")

    print("\nAll metrics tests passed!")


def test_feature_analyzer():
    """Test feature analyzer functionality."""
    print("\n=== Testing Feature Analyzer ===\n")

    # Create a test SAE
    config = SAEConfig(
        d_input=128,
        expansion_factor=4,  # 512 hidden features
        activation=SAEActivationType.RELU,
        use_encoder_bias=True,
        use_decoder_bias=True,
    )
    sae = SparseAutoencoderTorch(config)
    sae.eval()

    # Create analyzer
    analyzer = FeatureAnalyzer(sae, device="cpu")

    # Test analyze_sequence
    seq_len = 10
    activations = torch.randn(seq_len, config.d_input)
    tokens = [f"token_{i}" for i in range(seq_len)]
    token_ids = list(range(seq_len))

    result = analyzer.analyze_sequence(activations, tokens, token_ids, top_k=5)

    print(f"Sequence Analysis:")
    print(f"  tokens: {len(result.tokens)}")
    print(f"  per_token_activations: {len(result.per_token_activations)}")
    print(f"  most_active_features: {len(result.most_active_features)}")

    # Check first token
    tok0 = result.per_token_activations[0]
    print(f"\n  First token analysis:")
    print(f"    position: {tok0.position}")
    print(f"    token: {tok0.token}")
    print(f"    l0_sparsity: {tok0.l0_sparsity}")
    print(f"    top_features count: {len(tok0.top_features)}")

    if tok0.top_features:
        top_feat = tok0.top_features[0]
        print(f"    top feature: idx={top_feat.feature_idx}, val={top_feat.activation_value:.4f}")

    assert len(result.per_token_activations) == seq_len
    print("  Per-token activations correct length")

    # Test find_top_features_for_input
    single_activation = torch.randn(config.d_input)
    top_features = analyzer.find_top_features_for_input(single_activation, top_k=10)
    print(f"\n  Top features for single input:")
    print(f"    count: {len(top_features)}")
    if top_features:
        print(f"    top: idx={top_features[0][0]}, val={top_features[0][1]:.4f}")

    # Test heatmap
    heatmap = analyzer.compute_feature_activation_heatmap(activations, feature_indices=[0, 1, 2])
    print(f"\n  Feature heatmap shape: {heatmap.shape}")
    assert heatmap.shape == (seq_len, 3)
    print("  Heatmap shape correct")

    # Test update_statistics
    batch_activations = torch.randn(32, config.d_input)
    with torch.no_grad():
        hidden_batch = sae.encode(batch_activations)
    analyzer.update_statistics(hidden_batch)
    print(f"\n  Updated statistics with {analyzer.total_examples} examples")

    # Test get_feature_stats
    stats = analyzer.get_feature_stats(0)
    print(f"  Feature 0 stats:")
    print(f"    activation_frequency: {stats.activation_frequency:.4f}")
    print(f"    mean_activation: {stats.mean_activation:.4f}")

    print("\nAll feature analyzer tests passed!")


def test_interpretability():
    """Test interpretability tools."""
    print("\n=== Testing Interpretability ===\n")

    # Create a test SAE
    config = SAEConfig(
        d_input=128,
        expansion_factor=4,
        activation=SAEActivationType.RELU,
    )
    sae = SparseAutoencoderTorch(config)
    sae.eval()

    # Create interpreter
    interpreter = FeatureInterpreter(sae, max_examples_per_feature=50)

    # Process some examples
    print("Processing examples...")
    for i in range(10):
        activations = torch.randn(8, config.d_input)  # 8 tokens
        tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        token_ids = list(range(8))

        with torch.no_grad():
            hidden = sae.encode(activations)

        interpreter.process_example(hidden, tokens, token_ids, "The quick brown fox jumps over lazy dog")

    # Test get_top_tokens_for_feature
    top_tokens = interpreter.get_top_tokens_for_feature(0, top_k=5)
    print(f"\nTop tokens for feature 0:")
    for tok, freq in top_tokens[:3]:
        print(f"  '{tok}': {freq:.4f}")

    # Test get_top_examples_for_feature
    top_examples = interpreter.get_top_examples_for_feature(0, top_k=3)
    print(f"\nTop examples for feature 0: {len(top_examples)} found")
    if top_examples:
        ex = top_examples[0]
        print(f"  Best: token='{ex.token}', val={ex.activation_value:.4f}")
        print(f"  Context: {ex.context}")

    # Test suggest_label
    label, confidence = interpreter.suggest_label(0)
    print(f"\nSuggested label for feature 0: '{label}' (confidence: {confidence:.2f})")

    # Test polysemanticity score
    poly_score = interpreter.compute_polysemanticity_score(0)
    print(f"Polysemanticity score: {poly_score:.4f}")

    # Test get_feature_interpretation
    interp = interpreter.get_feature_interpretation(0)
    print(f"\nFeature interpretation:")
    print(f"  suggested_label: {interp.suggested_label}")
    print(f"  confidence: {interp.confidence:.2f}")
    print(f"  total_activations: {interp.activation_statistics['total_activations']}")

    # Test find_semantically_similar_features
    # First build interpretations for multiple features
    interpretations = {i: interpreter.get_feature_interpretation(i) for i in range(10)}
    similar = find_semantically_similar_features(interpretations, 0, top_k=3)
    print(f"\nSimilar features to feature 0: {similar}")

    print("\nAll interpretability tests passed!")


def test_sae_end_to_end():
    """End-to-end test with actual SAE forward pass."""
    print("\n=== Testing SAE End-to-End ===\n")

    # Create SAE
    config = SAEConfig(
        d_input=192,  # Ozera-Nano dimension
        expansion_factor=8,  # 1536 hidden
        activation=SAEActivationType.RELU,
    )
    sae = SparseAutoencoderTorch(config)
    sae.eval()

    # Generate test batch
    batch_size = 64
    batch = torch.randn(batch_size, config.d_input)

    # Run evaluation
    metrics = evaluate_sae_on_batch(sae, batch, device="cpu")

    print(f"End-to-end evaluation:")
    print(f"  avg_l0: {metrics.sparsity.avg_l0:.1f}")
    print(f"  dead_features: {metrics.feature_health.dead_features}")
    print(f"  mse: {metrics.reconstruction.mse:.6f}")
    print(f"  explained_variance: {metrics.reconstruction.explained_variance:.4f}")

    # L0 for untrained SAE might be high, but should still work
    assert metrics.sparsity.avg_l0 >= 0, "L0 should be non-negative"
    assert 0 <= metrics.reconstruction.explained_variance <= 1, "Explained variance should be in [0,1]"

    print("\nEnd-to-end test passed!")


def test_batch_analysis():
    """Test batch analysis function."""
    print("\n=== Testing Batch Analysis ===\n")

    # Create SAE
    config = SAEConfig(
        d_input=128,
        expansion_factor=4,
    )
    sae = SparseAutoencoderTorch(config)
    sae.eval()

    # Generate batch of sequences
    batch_size = 4
    seq_len = 8
    activations_batch = torch.randn(batch_size, seq_len, config.d_input)

    # Run batch analysis
    result = analyze_feature_activations_batch(sae, activations_batch, device="cpu")

    print(f"Batch analysis result:")
    print(f"  batch_size: {result['batch_size']}")
    print(f"  seq_len: {result['seq_len']}")
    print(f"  avg_l0_per_token: {result['avg_l0_per_token']:.1f}")
    print(f"  total_active_features: {result['total_active_features']}")
    print(f"  top_features count: {len(result['top_features'])}")

    assert result["batch_size"] == batch_size
    assert result["seq_len"] == seq_len
    print("\nBatch analysis test passed!")


def main():
    print("=" * 60)
    print("SAE Feature Analysis Tools Verification")
    print("=" * 60)

    test_metrics()
    test_feature_analyzer()
    test_interpretability()
    test_sae_end_to_end()
    test_batch_analysis()

    print("\n" + "=" * 60)
    print("All tests passed! Step 2 implementation verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
