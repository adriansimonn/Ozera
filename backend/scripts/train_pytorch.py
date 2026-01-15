"""
Train Ozera transformer models with PyTorch and GPU acceleration.

Usage:
    python scripts/train_pytorch.py --config nano --data path/to/data.txt --epochs 20
"""

import argparse
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.config import TransformerConfig, get_config
from core.transformer.model_torch import TransformerLM


class TextDataset(Dataset):
    """Simple text dataset for character-level language modeling."""

    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

        # Tokenize
        self.tokens = [self.char_to_id[ch] for ch in text]
        self.num_tokens = len(self.tokens)

    def __len__(self):
        return max(1, (self.num_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        # Pad if necessary
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def download_sample_data(output_path: str):
    """Download sample training data."""
    print("Downloading TinyShakespeare dataset...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded to {output_path}")


def train_epoch(model, train_loader, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _ = model(x)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(args):
    """Main training function."""

    print("="*60)
    print("OZERA PYTORCH GPU TRAINING")
    print("="*60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
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
    split_idx = int(len(text) * 0.9)
    train_text, val_text = text[:split_idx], text[split_idx:]

    # Create datasets
    train_dataset = TextDataset(train_text, seq_len=args.seq_len)
    val_dataset = TextDataset(val_text, seq_len=args.seq_len)

    print(f"\n🔤 Vocabulary size: {train_dataset.vocab_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Lambda
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    config = get_config(args.config)
    config.vocab_size = train_dataset.vocab_size
    config.max_seq_len = args.seq_len

    print(f"\n🏗️  Model: Ozera {args.config.upper()}")
    print(f"   Architecture: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")

    model = TransformerLM(config).to(device)
    total_params = model.count_parameters()

    print(f"   Parameters: {total_params:,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )

    print(f"\n🚀 Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Calculate perplexity
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)

        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | PPL: {val_ppl:.2f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'char_to_id': train_dataset.char_to_id,
                'id_to_char': train_dataset.id_to_char,
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'char_to_id': train_dataset.char_to_id,
                'id_to_char': train_dataset.id_to_char,
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  💾 Saved checkpoint at epoch {epoch+1}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f} | PPL: {np.exp(best_val_loss):.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Ozera with PyTorch")

    parser.add_argument('--config', type=str, default='nano', choices=['dev', 'nano', 'mini'])
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_torch')
    parser.add_argument('--save-every', type=int, default=5)

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train(args)


if __name__ == "__main__":
    main()
