"""
Example usage of the Ozera transformer models.

Demonstrates how to initialize and use the models for inference.
"""

import numpy as np
import sys
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.transformer import TransformerLM, get_config


def example_basic_usage():
    """Basic model initialization and forward pass."""
    print("Example 1: Basic Usage")
    print("-" * 40)

    # Initialize model with dev config
    config = get_config('dev')
    model = TransformerLM(config)

    print(f"Model: Ozera Dev")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Config: {config.num_layers} layers, {config.num_heads} heads")

    # Create sample input
    batch_size = 1
    seq_len = 10
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, _ = model.forward(token_ids)
    print(f"\nInput shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print()


def example_generation():
    """Autoregressive text generation."""
    print("Example 2: Autoregressive Generation")
    print("-" * 40)

    # Initialize model
    config = get_config('dev')
    model = TransformerLM(config)

    # Create prompt
    prompt_ids = np.array([[100, 200, 300, 400, 500]])
    print(f"Prompt: {prompt_ids[0].tolist()}")

    # Generate with default settings
    generated, _ = model.generate(prompt_ids, max_new_tokens=20)
    print(f"Generated ({generated.shape[1]} tokens): {generated[0].tolist()}")
    print()


def example_sampling_strategies():
    """Different sampling strategies for generation."""
    print("Example 3: Sampling Strategies")
    print("-" * 40)

    config = get_config('dev')
    model = TransformerLM(config)
    prompt_ids = np.array([[42, 123, 456]])

    # Low temperature (more deterministic)
    gen1, _ = model.generate(prompt_ids, max_new_tokens=10, temperature=0.5)
    print(f"Temperature 0.5: {gen1[0, 3:].tolist()}")

    # High temperature (more random)
    gen2, _ = model.generate(prompt_ids, max_new_tokens=10, temperature=1.5)
    print(f"Temperature 1.5: {gen2[0, 3:].tolist()}")

    # Top-k sampling
    gen3, _ = model.generate(prompt_ids, max_new_tokens=10, top_k=50)
    print(f"Top-k=50: {gen3[0, 3:].tolist()}")

    # Nucleus (top-p) sampling
    gen4, _ = model.generate(prompt_ids, max_new_tokens=10, top_p=0.9)
    print(f"Top-p=0.9: {gen4[0, 3:].tolist()}")
    print()


def example_attention_visualization():
    """Extract attention weights for visualization."""
    print("Example 4: Attention Weights")
    print("-" * 40)

    config = get_config('dev')
    model = TransformerLM(config)

    # Forward pass with attention
    token_ids = np.array([[1, 2, 3, 4, 5]])
    logits, attention_weights = model.forward(
        token_ids,
        return_attention=True
    )

    print(f"Number of layers: {len(attention_weights)}")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    print(f"  (batch_size, num_heads, seq_len, seq_len)")

    # Example: Get attention from first head of first layer
    first_head_attn = attention_weights[0][0, 0, :, :]
    print(f"\nFirst head attention matrix:")
    print(first_head_attn)
    print()


def example_model_sizes():
    """Compare different model configurations."""
    print("Example 5: Model Configurations")
    print("-" * 40)

    for config_name in ['dev', 'nano', 'mini']:
        config = get_config(config_name)
        model = TransformerLM(config)
        param_counts = model.get_num_params()

        print(f"\n{config_name.upper()}:")
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Token embeddings: {param_counts['token_embedding']:,}")
        print(f"  Transformer blocks: {param_counts['transformer_blocks']:,}")
        print(f"  Architecture: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("OZERA TRANSFORMER MODEL EXAMPLES")
    print("=" * 60)
    print()

    np.random.seed(42)

    example_basic_usage()
    example_generation()
    example_sampling_strategies()
    example_attention_visualization()
    example_model_sizes()

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
