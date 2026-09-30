"""
Device-agnostic training logic for Modal cloud execution.

This module contains the core training functions extracted from training_worker.py,
designed to run on any device (CPU/GPU) in the Modal cloud environment.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_model

# Import transformer components
import sys
import os

# Add parent directory to path for imports when running in Modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.config import get_config
from core.transformer.model_torch import TransformerLM
from core.tokenizer.bpe_tokenizer import get_tokenizer


@dataclass
class TrainingConfig:
    """Configuration for a training job."""
    job_id: str
    user_id: int
    dataset_path: str  # Path in Modal volume
    model_output_path: str  # Path in Modal volume
    model_config: str  # 'nano' or 'mini'
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seq_len: int
    dataset_name: str = ""


@dataclass
class TrainingProgress:
    """Training progress data."""
    job_id: str
    status: str
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_ppl: Optional[float] = None
    val_ppl: Optional[float] = None
    elapsed_seconds: int = 0
    error_message: Optional[str] = None


class BPETextDataset(Dataset):
    """Text dataset using BPE tokenization (GPT-2 tokenizer)."""

    def __init__(self, tokens: list[int], seq_len: int = 256):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_tokens = len(tokens)

    def __len__(self):
        return max(1, (self.num_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        # Pad if necessary (use 0 as pad token)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    out_of_time: Optional[Callable[[], bool]] = None,
) -> Optional[float]:
    """
    Train for one epoch.

    Args:
        out_of_time: Optional check, run before each batch, for whether to stop

    Returns:
        Average loss for the epoch, or None if it was stopped before finishing
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        if out_of_time is not None and out_of_time():
            return None

        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _, _ = model(x)

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

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        logits, _, _ = model(x)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_training(
    config: TrainingConfig,
    progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    deadline: Optional[float] = None,
) -> tuple[str, float, Optional[str]]:
    """
    Run the training job.

    Args:
        config: Training configuration
        progress_callback: Optional callback for progress updates
        should_stop: Optional check, run after each epoch, for whether the job was cancelled
        deadline: Optional time.time() by which training must stop, leaving time to save
            the model and report back before the container's timeout. Training stops
            before an epoch that wouldn't finish by then (or during one that runs over),
            and the best completed epoch is saved as usual.

    Returns:
        Tuple of (final_status, actual_minutes, error_message); final_status is
        "completed", "failed", or "cancelled". A job stopped at the deadline is
        "completed", with a message saying how many epochs it trained.
    """
    start_time = time.time()

    def out_of_time(seconds_needed: float = 0) -> bool:
        return deadline is not None and time.time() + seconds_needed > deadline

    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load dataset from volume path
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {config.dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize with BPE
        tokenizer = get_tokenizer()
        all_tokens = tokenizer.encode(text)
        print(f"Tokenized {len(all_tokens)} tokens")

        # Split train/val (90/10)
        split_idx = int(len(all_tokens) * 0.9)
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]

        # Create datasets
        train_dataset = BPETextDataset(train_tokens, seq_len=config.seq_len)
        val_dataset = BPETextDataset(val_tokens, seq_len=config.seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        # Create model
        model_config = get_config(config.model_config)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.max_seq_len = config.seq_len

        model = TransformerLM(model_config).to(device)
        print(f"Created model with {model.count_parameters():,} parameters")

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * config.epochs
        warmup_steps = min(100, total_steps // 10)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )

        # Training loop
        best_val_loss = float('inf')
        best_state = None
        output_dir = Path(config.model_output_path)
        epochs_trained = 0
        last_epoch_seconds = 0.0

        for epoch in range(config.epochs):
            # Don't start an epoch that won't finish before the deadline
            if epoch > 0 and out_of_time(last_epoch_seconds):
                break

            epoch_start = time.time()

            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, device, scheduler, out_of_time=out_of_time
            )
            if train_loss is None:
                # Ran past the deadline mid-epoch: keep the epochs already finished
                break

            # Evaluate
            val_loss = evaluate(model, val_loader, device)
            epochs_trained = epoch + 1
            last_epoch_seconds = time.time() - epoch_start

            # Calculate metrics
            elapsed = time.time() - start_time
            train_ppl = float(np.exp(train_loss))
            val_ppl = float(np.exp(val_loss))

            # Send progress update
            if progress_callback:
                progress = TrainingProgress(
                    job_id=config.job_id,
                    status="running",
                    current_epoch=epoch + 1,
                    total_epochs=config.epochs,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    train_ppl=train_ppl,
                    val_ppl=val_ppl,
                    elapsed_seconds=int(elapsed),
                )
                progress_callback(progress)

            print(f"Epoch {epoch + 1}/{config.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train PPL: {train_ppl:.2f}, Val PPL: {val_ppl:.2f}")

            if should_stop and should_stop():
                print("Job was cancelled, stopping")
                return "cancelled", (time.time() - start_time) / 60, None

            # Keep the best model's weights (on the CPU). They're only written to the model
            # folder once training finishes: retraining under an existing model's name writes
            # to that model's folder, which must stay as it was if this job fails or is
            # cancelled.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
                weights_metadata = {
                    "format": "ozera",
                    "epoch": str(epoch),
                    "train_loss": str(train_loss),
                    "val_loss": str(val_loss),
                    "vocab_size": str(model_config.vocab_size),
                    "max_seq_len": str(model_config.max_seq_len),
                    "d_model": str(model_config.d_model),
                    "num_layers": str(model_config.num_layers),
                    "num_heads": str(model_config.num_heads),
                    "d_ff": str(model_config.d_ff),
                    "dropout_rate": str(model_config.dropout_rate),
                }

        if best_state is None:
            if epochs_trained == 0:
                raise RuntimeError(
                    "Training didn't finish an epoch within the time limit; "
                    "use a smaller dataset or model, or a faster GPU"
                )
            raise RuntimeError("Training produced no usable model: the validation loss was never finite")

        stopped_early_message = None
        if epochs_trained < config.epochs:
            stopped_early_message = (
                f"Stopped after {epochs_trained} of {config.epochs} epochs to stay within "
                f"the training time limit; the best epoch was saved"
            )
            print(stopped_early_message)

        # Save the best model's weights as safetensors with config metadata
        model.load_state_dict(best_state)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, output_dir / 'model.safetensors', metadata=weights_metadata)

        # Save model metadata
        metadata = {
            "model_id": config.model_name,
            "name": config.model_name,
            "base_config": config.model_config,
            "dataset_id": Path(config.dataset_path).parent.name,
            "dataset_name": config.dataset_name,
            "trained_at": datetime.utcnow().isoformat(),
            "val_loss": float(best_val_loss),
            "parameters": model.count_parameters(),
            "epochs": epochs_trained,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "learning_rate": config.learning_rate,
            "user_id": config.user_id,
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Calculate actual training time
        actual_minutes = (time.time() - start_time) / 60

        return "completed", actual_minutes, stopped_early_message

    except Exception as e:
        actual_minutes = (time.time() - start_time) / 60
        error_message = str(e)
        print(f"Training failed: {error_message}")

        # Send error progress
        if progress_callback:
            progress = TrainingProgress(
                job_id=config.job_id,
                status="failed",
                current_epoch=0,
                total_epochs=config.epochs,
                elapsed_seconds=int(time.time() - start_time),
                error_message=error_message,
            )
            progress_callback(progress)

        return "failed", actual_minutes, error_message
