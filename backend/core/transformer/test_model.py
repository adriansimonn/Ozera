"""
Test script for transformer model components.

Verifies that all pieces work together correctly.
"""

import numpy as np
import sys
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.transformer.config import TransformerConfig, get_config
from core.transformer.model import TransformerLM


def test_config():
    """Test configuration classes."""
    print("Testing configuration...")

    configs = ['dev', 'nano', 'mini']
    for name in configs:
        config = get_config(name)
        params = config.count_parameters()
        print(f"  {name}: {params:,} parameters")
        print(f"    d_model={config.d_model}, layers={config.num_layers}, heads={config.num_heads}")

    print("✓ Configuration tests passed\n")


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")

    config = get_config('dev')
    model = TransformerLM(config)

    # Check model structure
    assert len(model.blocks) == config.num_layers
    assert model.token_embedding.vocab_size == config.vocab_size
    assert model.token_embedding.d_model == config.d_model

    print(f"  Created model with {model.count_parameters():,} parameters")
    print("✓ Model initialization tests passed\n")


def test_forward_pass():
    """Test forward pass through model."""
    print("Testing forward pass...")

    config = get_config('dev')
    model = TransformerLM(config)

    # Create dummy input
    batch_size = 2
    seq_len = 16
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, _ = model.forward(token_ids, training=False)

    # Check output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

    print(f"  Input shape: {token_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print("✓ Forward pass tests passed\n")


def test_forward_pass_with_attention():
    """Test forward pass with attention weights."""
    print("Testing forward pass with attention...")

    config = get_config('dev')
    model = TransformerLM(config)

    # Create dummy input
    batch_size = 2
    seq_len = 8
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass with attention
    logits, attention_weights = model.forward(
        token_ids,
        training=False,
        return_attention=True
    )

    # Check attention weights
    assert len(attention_weights) == config.num_layers
    expected_attn_shape = (batch_size, config.num_heads, seq_len, seq_len)
    assert attention_weights[0].shape == expected_attn_shape

    print(f"  Number of attention layers: {len(attention_weights)}")
    print(f"  Attention shape per layer: {attention_weights[0].shape}")
    print("✓ Attention tests passed\n")


def test_generation():
    """Test autoregressive generation."""
    print("Testing autoregressive generation...")

    config = get_config('dev')
    model = TransformerLM(config)

    # Create prompt
    batch_size = 1
    prompt_len = 5
    max_new_tokens = 10
    prompt_ids = np.random.randint(0, config.vocab_size, (batch_size, prompt_len))

    # Generate
    generated_ids, _ = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0
    )

    # Check output shape
    expected_len = prompt_len + max_new_tokens
    assert generated_ids.shape == (batch_size, expected_len)

    print(f"  Prompt length: {prompt_len}")
    print(f"  Generated length: {generated_ids.shape[1]}")
    print(f"  Sample generated IDs: {generated_ids[0, :10]}")
    print("✓ Generation tests passed\n")


def test_generation_with_sampling():
    """Test generation with different sampling strategies."""
    print("Testing generation with sampling strategies...")

    config = get_config('dev')
    model = TransformerLM(config)

    prompt_ids = np.random.randint(0, config.vocab_size, (1, 5))

    # Test temperature
    gen_temp, _ = model.generate(prompt_ids, max_new_tokens=5, temperature=0.5)
    print(f"  Temperature 0.5: {gen_temp.shape}")

    # Test top-k
    gen_topk, _ = model.generate(prompt_ids, max_new_tokens=5, top_k=10)
    print(f"  Top-k=10: {gen_topk.shape}")

    # Test top-p
    gen_topp, _ = model.generate(prompt_ids, max_new_tokens=5, top_p=0.9)
    print(f"  Top-p=0.9: {gen_topp.shape}")

    print("✓ Sampling strategy tests passed\n")


def test_parameter_counts():
    """Test parameter counting."""
    print("Testing parameter counts...")

    config = get_config('dev')
    model = TransformerLM(config)

    # Get detailed counts
    detailed_counts = model.get_num_params()

    print(f"  Token embedding: {detailed_counts['token_embedding']:,}")
    print(f"  Positional encoding: {detailed_counts['pos_encoding']:,}")
    print(f"  Transformer blocks: {detailed_counts['transformer_blocks']:,}")
    print(f"  Final layer norm: {detailed_counts['final_ln']:,}")
    print(f"  Total: {detailed_counts['total']:,}")

    # Verify matches config count
    assert detailed_counts['total'] == config.count_parameters()

    print("✓ Parameter count tests passed\n")


def test_different_configs():
    """Test all predefined configurations."""
    print("Testing all configurations...")

    for config_name in ['dev', 'nano', 'mini']:
        config = get_config(config_name)
        model = TransformerLM(config)

        # Forward pass
        token_ids = np.random.randint(0, config.vocab_size, (1, 10))
        logits, _ = model.forward(token_ids)

        print(f"  {config_name}: {logits.shape} output shape, {model.count_parameters():,} params")

    print("✓ All configuration tests passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("TRANSFORMER MODEL TEST SUITE")
    print("="*60 + "\n")

    np.random.seed(42)  # For reproducibility

    test_config()
    test_model_initialization()
    test_forward_pass()
    test_forward_pass_with_attention()
    test_generation()
    test_generation_with_sampling()
    test_parameter_counts()
    test_different_configs()

    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
