"""
Generate text using a trained Ozera model.

Usage:
    python scripts/generate_text.py --checkpoint checkpoints/best_model.npz --prompt "Hello world"
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer import TransformerLM
from core.training import SimpleTokenizer


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    # Extract config
    config = checkpoint['config'].item()
    print(f"  Model: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")
    print(f"  Parameters: {config.count_parameters():,}")
    print(f"  Vocab size: {config.vocab_size}")

    return checkpoint, config


def load_model_parameters(model, checkpoint):
    """Load trained parameters into model."""
    model_params = checkpoint['model_params'].item()

    # Load token embeddings
    if 'token_embedding' in model_params:
        token_emb = model_params['token_embedding']
        if 'embedding' in token_emb:
            model.token_embedding.embedding = token_emb['embedding']

    # Load positional encoding
    if 'pos_encoding' in model_params:
        pos_enc = model_params['pos_encoding']
        if 'pos_embedding' in pos_enc:
            model.pos_encoding.pos_embedding = pos_enc['pos_embedding']

    # Load transformer blocks
    for i in range(len(model.blocks)):
        block_key = f'block_{i}'
        if block_key in model_params:
            block_params = model_params[block_key]

            # Load layer norms
            if 'ln1_gamma' in block_params:
                model.blocks[i].ln1_gamma = block_params['ln1_gamma']
            if 'ln1_beta' in block_params:
                model.blocks[i].ln1_beta = block_params['ln1_beta']
            if 'ln2_gamma' in block_params:
                model.blocks[i].ln2_gamma = block_params['ln2_gamma']
            if 'ln2_beta' in block_params:
                model.blocks[i].ln2_beta = block_params['ln2_beta']

            # Load attention parameters
            for param_name in ['attention.Wq', 'attention.Wk', 'attention.Wv', 'attention.Wo',
                             'attention.bq', 'attention.bk', 'attention.bv', 'attention.bo']:
                if param_name in block_params:
                    attr_path = param_name.split('.')
                    if attr_path[0] == 'attention':
                        setattr(model.blocks[i].attention, attr_path[1], block_params[param_name])

            # Load feedforward parameters
            for param_name in ['feed_forward.W1', 'feed_forward.W2',
                             'feed_forward.b1', 'feed_forward.b2']:
                if param_name in block_params:
                    attr_path = param_name.split('.')
                    if attr_path[0] == 'feed_forward':
                        setattr(model.blocks[i].feed_forward, attr_path[1], block_params[param_name])

    # Load final layer norm
    if 'final_ln' in model_params:
        final_ln = model_params['final_ln']
        if 'gamma' in final_ln:
            model.final_ln_gamma = final_ln['gamma']
        if 'beta' in final_ln:
            model.final_ln_beta = final_ln['beta']

    print("✓ Parameters loaded successfully")


def create_tokenizer_from_vocab(vocab_size: int, sample_text: str = None):
    """
    Create a tokenizer. Ideally we'd save the tokenizer with the checkpoint,
    but for now we'll recreate it from sample text.
    """
    if sample_text is None:
        # Use default character set if no sample text provided
        # This should match the training data
        sample_text = " !.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;:?'-\n"

    return SimpleTokenizer(sample_text)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9
):
    """Generate text from prompt."""

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = np.array([prompt_ids])  # Add batch dimension

    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_ids[0].tolist()}")
    print(f"\nGenerating {max_tokens} tokens...")
    print("="*60)

    # Generate
    generated_ids, _ = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained Ozera model")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                       help='Text prompt to continue')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus (top-p) sampling')
    parser.add_argument('--vocab-text', type=str, default=None,
                       help='Sample text to build vocabulary (should match training data)')

    args = parser.parse_args()

    # Load checkpoint
    checkpoint, config = load_checkpoint(args.checkpoint)

    # Create model
    print("\nInitializing model...")
    model = TransformerLM(config)

    # Load trained parameters
    print("Loading trained parameters...")
    load_model_parameters(model, checkpoint)

    # Create tokenizer
    print("\nCreating tokenizer...")
    if args.vocab_text and os.path.exists(args.vocab_text):
        with open(args.vocab_text, 'r', encoding='utf-8') as f:
            vocab_text = f.read()
        tokenizer = SimpleTokenizer(vocab_text)
    else:
        # Try to load from training data if available
        if os.path.exists('/tmp/tinyshakespeare.txt'):
            with open('/tmp/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
                vocab_text = f.read()
            tokenizer = SimpleTokenizer(vocab_text)
        else:
            print("  Warning: Using default character set. Results may be poor.")
            print("  For best results, provide --vocab-text with training data")
            tokenizer = create_tokenizer_from_vocab(config.vocab_size)

    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Generate text
    generated_text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("\n" + generated_text)
    print("\n" + "="*60)
    print("Generation complete!")


if __name__ == "__main__":
    main()
