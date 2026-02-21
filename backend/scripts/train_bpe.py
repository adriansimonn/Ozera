"""
Train Ozera transformer with BPE tokenization and OpenWebText.

Usage:
    python scripts/train_bpe.py --config nano --epochs 10
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
from core.tokenizer import get_tokenizer


class TokenDataset(Dataset):
    """Dataset for pre-tokenized data."""

    def __init__(self, tokens: np.ndarray, seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y


def train_epoch(model, train_loader, optimizer, device, scheduler=None, log_interval=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    log_loss = 0

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
        log_loss += loss.item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = log_loss / log_interval
            ppl = np.exp(avg_loss)
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
            log_loss = 0

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model."""
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
    print("OZERA TRAINING WITH BPE TOKENIZATION")
    print("="*60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading BPE tokenizer...")
    tokenizer = get_tokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load data
    print(f"\nLoading preprocessed data...")
    data_dir = Path(args.data_dir)

    train_tokens = np.load(data_dir / 'train.npy')
    val_tokens = np.load(data_dir / 'val.npy')

    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = TokenDataset(train_tokens, seq_len=args.seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len=args.seq_len)

    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Val sequences: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    config = get_config(args.config)
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.seq_len

    print(f"\nModel: Ozera {args.config.upper()}")
    print(f"   Architecture: {config.num_layers}L x {config.num_heads}H x {config.d_model}D")

    model = TransformerLM(config).to(device)
    total_params = model.count_parameters()

    print(f"   Parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )

    print(f"\nTraining Configuration:")
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
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler, args.log_interval
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Calculate perplexity
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)

        print(f"\n  Epoch {epoch + 1} Summary ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | PPL: {val_ppl:.2f}")

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
                'tokenizer': 'gpt2',
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokenizer': 'gpt2',
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  Saved checkpoint at epoch {epoch+1}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f} | PPL: {np.exp(best_val_loss):.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='nano', choices=['dev', 'nano', 'mini'])
    parser.add_argument('--data-dir', type=str, default='./data/openwebtext')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_bpe')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train(args)


if __name__ == "__main__":
    main()
