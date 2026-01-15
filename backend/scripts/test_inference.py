"""
Test inference with loaded models.

Usage:
    python scripts/test_inference.py --model nano
    python scripts/test_inference.py --model mini --prompt "The future of AI"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference import load_model, TextGenerator


def main():
    parser = argparse.ArgumentParser(description="Test Ozera model inference")
    parser.add_argument('--model', type=str, default='nano', choices=['nano', 'mini'],
                        help='Model to use')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto if not specified)')

    args = parser.parse_args()

    print("="*60)
    print(f"Testing {args.model.upper()} model")
    print("="*60)

    # Load model
    print("\n[1/3] Loading model...")
    model, config = load_model(args.model, models_dir="models", device=args.device)

    # Create generator
    print("\n[2/3] Initializing generator...")
    device = next(model.parameters()).device
    generator = TextGenerator(model, device=str(device))

    # Generate text
    print("\n[3/3] Generating text...")
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Settings: max_tokens={args.max_tokens}, temperature={args.temperature}, top_k={args.top_k}")
    print("\n" + "="*60)

    result = generator.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        return_metadata=True
    )

    print(result['text'])
    print("\n" + "="*60)
    print(f"\nMetadata:")
    print(f"  Prompt tokens: {result['prompt_tokens']}")
    print(f"  Generated tokens: {result['generated_tokens']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Temperature: {result['temperature']}")
    print("="*60)


if __name__ == "__main__":
    main()
