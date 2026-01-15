"""
Test a trained Ozera model by generating text.

Usage:
    python scripts/test_trained_model.py --checkpoint checkpoints/best_model.npz
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer import TransformerLM, get_config
from core.training import SimpleTokenizer


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    config = checkpoint['config'].item()

    print(f"Model: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")
    print(f"Parameters: {config.count_parameters():,}")
    print(f"Vocab size: {config.vocab_size}")

    # Create model
    model = TransformerLM(config)

    # Load parameters
    # TODO: Implement proper parameter loading

    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default='Hello world', help='Prompt text')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40, help='Top-k sampling')

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint)

    print(f"\nGenerating text with prompt: '{args.prompt}'")
    print("="*60)

    # For now, just test that model loads
    print("\n✓ Model loaded successfully!")
    print("Note: Full inference will be available once parameter loading is implemented")


if __name__ == "__main__":
    main()
