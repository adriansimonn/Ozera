"""
Background training worker with BPE tokenization.
"""

import os
import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.config import get_config
from core.transformer.model_torch import TransformerLM
from core.tokenizer.bpe_tokenizer import get_tokenizer
from services.job_manager import (
    get_job_dir,
    load_job_config,
    load_job_progress,
    save_job_progress,
)
from api.schemas.training import JobStatus


# Paths
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
MODELS_DIR = Path(__file__).parent.parent / "models" / "custom"


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
    cancel_event: Optional[Event] = None,
) -> tuple[float, bool]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average loss, was_cancelled)
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            return total_loss / max(num_batches, 1), True

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

    return total_loss / max(num_batches, 1), False


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

        logits, _ = model(x)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_training_job(job_id: str, cancel_event: Event) -> None:
    """
    Run a training job.

    This function is designed to be called in a background thread.
    It reads config from the job directory, trains the model, and
    writes progress updates periodically.
    """
    job_dir = get_job_dir(job_id)

    try:
        # Load job configuration
        config = load_job_config(job_id)
        if not config:
            raise ValueError(f"Job config not found: {job_id}")

        # Update status to running
        progress = load_job_progress(job_id)
        progress["status"] = JobStatus.RUNNING.value
        progress["last_update"] = datetime.utcnow().isoformat()
        save_job_progress(job_id, progress)

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        dataset_id = config["dataset_id"]
        dataset_path = DATASETS_DIR / dataset_id / "raw.txt"
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_id}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize with BPE
        tokenizer = get_tokenizer()
        all_tokens = tokenizer.encode(text)

        # Split train/val (90/10)
        split_idx = int(len(all_tokens) * 0.9)
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]

        # Create datasets
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        train_dataset = BPETextDataset(train_tokens, seq_len=seq_len)
        val_dataset = BPETextDataset(val_tokens, seq_len=seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        # Create model
        model_config = get_config(config["model_config"])
        # BPE tokenizer has fixed vocab size
        model_config.vocab_size = tokenizer.vocab_size
        model_config.max_seq_len = seq_len

        model = TransformerLM(model_config).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler
        epochs = config["epochs"]
        total_steps = len(train_loader) * epochs
        warmup_steps = min(100, total_steps // 10)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["learning_rate"],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )

        # Update progress with total steps
        progress["total_steps"] = total_steps
        save_job_progress(job_id, progress)

        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            # Check for cancellation
            if cancel_event.is_set():
                progress["status"] = JobStatus.CANCELLED.value
                progress["last_update"] = datetime.utcnow().isoformat()
                save_job_progress(job_id, progress)
                return

            epoch_start = time.time()

            # Train
            train_loss, was_cancelled = train_epoch(
                model, train_loader, optimizer, device, scheduler, cancel_event
            )

            if was_cancelled:
                progress["status"] = JobStatus.CANCELLED.value
                progress["last_update"] = datetime.utcnow().isoformat()
                save_job_progress(job_id, progress)
                return

            # Evaluate
            val_loss = evaluate(model, val_loader, device)

            # Calculate metrics
            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start
            epochs_remaining = epochs - (epoch + 1)
            estimated_remaining = epoch_time * epochs_remaining

            train_ppl = float(np.exp(train_loss))
            val_ppl = float(np.exp(val_loss))

            # Update progress
            progress["current_epoch"] = epoch + 1
            progress["current_step"] = (epoch + 1) * len(train_loader)
            progress["train_loss"] = float(train_loss)
            progress["val_loss"] = float(val_loss)
            progress["train_ppl"] = train_ppl
            progress["val_ppl"] = val_ppl
            progress["elapsed_seconds"] = int(elapsed)
            progress["estimated_remaining_seconds"] = int(estimated_remaining)
            progress["last_update"] = datetime.utcnow().isoformat()
            save_job_progress(job_id, progress)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save checkpoint to job directory
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, job_dir / 'best_model.pt')

            # Save periodic checkpoints every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, job_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Training complete - save final model to custom models directory
        model_name = config["model_name"]
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy best model
        shutil.copy(job_dir / 'best_model.pt', model_dir / 'model.pt')

        # Save model metadata
        metadata = {
            "model_id": model_name,
            "name": model_name,
            "base_config": config["model_config"],
            "dataset_id": config["dataset_id"],
            "dataset_name": config["dataset_name"],
            "trained_at": datetime.utcnow().isoformat(),
            "val_loss": float(best_val_loss),
            "parameters": model.count_parameters(),
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "learning_rate": config["learning_rate"],
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update progress to completed
        progress["status"] = JobStatus.COMPLETED.value
        progress["last_update"] = datetime.utcnow().isoformat()
        save_job_progress(job_id, progress)

    except Exception as e:
        # Update progress to failed
        progress = load_job_progress(job_id) or {"job_id": job_id}
        progress["status"] = JobStatus.FAILED.value
        progress["error_message"] = str(e)
        progress["last_update"] = datetime.utcnow().isoformat()
        save_job_progress(job_id, progress)
        raise
