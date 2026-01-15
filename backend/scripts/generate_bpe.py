"""
Generate text using BPE-trained model.

Usage:
    python scripts/generate_bpe.py --checkpoint checkpoints_bpe/best_model.pt --prompt "Once upon a time"
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.model_torch import TransformerLM
from core.tokenizer import get_tokenizer


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']

    print(f"  Model: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")
    print(f"  Parameters: {config.count_parameters():,}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    # Create model
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda'
):
    """Generate text from prompt."""

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_tokens} tokens with temperature={temperature}, top_k={top_k}...")
    print("="*60)

    # Generate
    with torch.no_grad():
        generated_ids, _ = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=None
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}\n")

    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Generate
    generated_text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )

    print(generated_text)
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
