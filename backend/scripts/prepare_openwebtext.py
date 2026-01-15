"""
Prepare OpenWebText dataset for training.

Downloads and tokenizes OpenWebText subset for efficient training.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.tokenizer import get_tokenizer


def prepare_dataset(
    output_dir: str = './data/openwebtext',
    num_proc: int = 4,
    subset_size: int = None
):
    """
    Download and tokenize OpenWebText.

    Args:
        output_dir: Output directory for processed data
        num_proc: Number of processes for parallel processing
        subset_size: Optional - use only first N examples (for testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("OPENWEBTEXT DATASET PREPARATION")
    print("="*60)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load dataset
    print("\n📚 Loading OpenWebText dataset...")
    print("   This may take a few minutes on first run...")

    # Use openwebtext subset (10GB) instead of full dataset
    dataset = load_dataset("openwebtext", split="train", streaming=False)

    if subset_size:
        print(f"   Using subset of {subset_size:,} examples")
        dataset = dataset.select(range(subset_size))
    else:
        print(f"   Total examples: {len(dataset):,}")

    # Tokenize dataset
    print("\n🔤 Tokenizing dataset...")

    def tokenize_function(examples):
        """Tokenize batch of texts."""
        all_tokens = []
        for text in examples["text"]:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)  # Add separator
        return {"tokens": [all_tokens]}

    # Process dataset
    print("   Processing in parallel...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Concatenate all tokens
    print("\n💾 Saving processed data...")
    all_tokens = []

    for example in tokenized:
        all_tokens.extend(example["tokens"])

    # Convert to numpy array
    tokens_array = np.array(all_tokens, dtype=np.uint16)

    # Save train/val split
    split_idx = int(len(tokens_array) * 0.95)
    train_tokens = tokens_array[:split_idx]
    val_tokens = tokens_array[split_idx:]

    np.save(output_dir / 'train.npy', train_tokens)
    np.save(output_dir / 'val.npy', val_tokens)

    print(f"\n✓ Dataset prepared successfully!")
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")
    print(f"   Total tokens: {len(tokens_array):,}")
    print(f"   Saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./data/openwebtext')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Use only first N examples (for testing)')

    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        subset_size=args.subset_size
    )


if __name__ == "__main__":
    main()
