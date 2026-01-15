"""
Train Ozera transformer models.

Usage:
    python scripts/train_model.py --config nano --data path/to/data.txt
"""

import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer import TransformerLM, get_config
from core.training import (
    Trainer, TrainingConfig,
    SimpleTokenizer, TextDataset, DataLoader,
    perplexity, train_test_split
)


def download_sample_data(output_path: str):
    """Download sample training data if none provided."""
    print("Downloading sample dataset (TinyShakespeare)...")
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded to {output_path}")


def train(args):
    """Main training function."""

    print("="*60)
    print("OZERA TRANSFORMER TRAINING")
    print("="*60)

    # Load or download data
    if args.data is None:
        data_path = "/tmp/tinyshakespeare.txt"
        if not os.path.exists(data_path):
            download_sample_data(data_path)
        args.data = data_path

    print(f"\n📚 Loading data from: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"   Dataset size: {len(text):,} characters")

    # Split train/val
    train_text, val_text = train_test_split(text, train_ratio=0.9)
    print(f"   Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")

    # Create tokenizer
    print(f"\n🔤 Creating tokenizer...")
    tokenizer = SimpleTokenizer(text)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")

    # Get model config
    config = get_config(args.config)
    config.vocab_size = tokenizer.vocab_size  # Adjust to actual vocab
    config.max_seq_len = args.seq_len

    print(f"\n🏗️  Initializing model: Ozera {args.config.upper()}")
    print(f"   Parameters: {config.count_parameters():,}")
    print(f"   Architecture: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")
    print(f"   Sequence length: {config.max_seq_len}")

    # Create model
    model = TransformerLM(config)

    # Create datasets
    print(f"\n📊 Creating datasets...")
    train_dataset = TextDataset(train_text, tokenizer, seq_len=args.seq_len)
    val_dataset = TextDataset(val_text, tokenizer, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Training config
    total_steps = len(train_loader) * args.epochs
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=min(100, total_steps // 10),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )

    # Create trainer
    trainer = Trainer(model, train_config, use_numerical_grads=False)

    print(f"\n🚀 Starting training...")
    print(f"   Total epochs: {args.epochs}")
    print(f"   Steps per epoch: {len(train_loader)}")
    print(f"   Total steps: {total_steps}")
    print(f"   Learning rate: {args.lr}")
    print("="*60)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = []

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Training
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step_start = time.time()

            loss = trainer.train_step(inputs, targets)
            epoch_losses.append(loss)

            step_time = time.time() - step_start

            # Log progress
            if trainer.step % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval:])
                ppl = perplexity(avg_loss)
                print(f"  Step {trainer.step:5d} | Loss: {loss:.4f} | PPL: {ppl:7.2f} | {step_time:.2f}s/step")

            # Evaluate
            if trainer.step % args.eval_interval == 0:
                print("\n  Evaluating...")
                val_losses = []
                for val_inputs, val_targets in val_loader:
                    val_loss = trainer.evaluate(val_inputs, val_targets)
                    val_losses.append(val_loss)

                avg_val_loss = np.mean(val_losses)
                val_ppl = perplexity(avg_val_loss)
                print(f"  📊 Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = f"{args.checkpoint_dir}/best_model.npz"
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    trainer.save_checkpoint(save_path)
                    print(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")
                print()

            # Save checkpoint
            if args.save_interval > 0 and trainer.step % args.save_interval == 0:
                save_path = f"{args.checkpoint_dir}/checkpoint_step_{trainer.step}.npz"
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                trainer.save_checkpoint(save_path)
                print(f"  💾 Saved checkpoint at step {trainer.step}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n  Epoch {epoch + 1} complete in {epoch_time:.1f}s")
        print(f"  Avg train loss: {avg_epoch_loss:.4f} | PPL: {perplexity(avg_epoch_loss):.2f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f} | PPL: {perplexity(best_val_loss):.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Ozera transformer models")

    # Model config
    parser.add_argument('--config', type=str, default='nano', choices=['dev', 'nano', 'mini'],
                        help='Model configuration')

    # Data
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training text file (downloads sample if not provided)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping threshold')

    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(42)

    train(args)


if __name__ == "__main__":
    main()
