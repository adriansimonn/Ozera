"""
SAE Trainer with training loop and monitoring.

Handles the training process including:
- Optimizer setup with warmup
- Loss computation and backprop
- Dead feature detection and resampling
- Progress tracking and logging
- Checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import math

from .config import SAEConfig, SAEActivationType
from .model_torch import SparseAutoencoderTorch
from .loss import sae_loss, compute_loss_metrics
from .activation_buffer import ActivationBuffer, DiskActivationBuffer


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training process."""

    # Training duration
    num_steps: int = 50_000
    batch_size: int = 4096

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 1000
    use_cosine_decay: bool = True
    min_lr_fraction: float = 0.1  # Final LR = initial LR * this fraction

    # Sparsity
    sparsity_coefficient: float = 0.01

    # Dead feature handling
    dead_feature_threshold: int = 10_000  # Steps without activation
    dead_feature_resample: bool = True
    resample_check_interval: int = 5_000

    # Logging and checkpoints
    log_interval: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 5_000

    # Decoder normalization
    normalize_decoder: bool = True
    normalize_decoder_interval: int = 100


@dataclass
class SAETrainingProgress:
    """Progress information during training."""
    step: int
    total_steps: int
    loss: float
    recon_loss: float
    sparse_loss: float
    avg_l0: float
    dead_features: int
    learning_rate: float
    elapsed_seconds: int = 0
    status: str = "running"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "recon_loss": self.recon_loss,
            "sparse_loss": self.sparse_loss,
            "avg_l0": self.avg_l0,
            "dead_features": self.dead_features,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
            "error_message": self.error_message,
        }


class SAETrainer:
    """
    Trainer for Sparse Autoencoders.

    Handles the complete training loop with monitoring and checkpointing.
    """

    def __init__(
        self,
        model: SparseAutoencoderTorch,
        config: SAETrainingConfig,
        device: str = "cuda",
    ):
        """
        Initialize SAE trainer.

        Args:
            model: SAE model to train
            config: Training configuration
            device: Device for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Training state
        self.step = 0
        self.best_loss = float("inf")
        self.train_history: List[Dict[str, float]] = []

        # Dead feature tracking
        self.feature_activation_counts = torch.zeros(
            model.d_hidden, device=device, dtype=torch.long
        )
        self.steps_since_last_activation = torch.zeros(
            model.d_hidden, device=device, dtype=torch.long
        )

    def get_lr(self) -> float:
        """Get current learning rate based on schedule."""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (self.step / self.config.warmup_steps)
        elif self.config.use_cosine_decay:
            # Cosine decay after warmup
            progress = (self.step - self.config.warmup_steps) / (
                self.config.num_steps - self.config.warmup_steps
            )
            progress = min(progress, 1.0)
            min_lr = self.config.learning_rate * self.config.min_lr_fraction
            return min_lr + (self.config.learning_rate - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            return self.config.learning_rate

    def _update_lr(self):
        """Update optimizer learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_dead_feature_tracking(self, hidden: torch.Tensor):
        """Update dead feature tracking based on batch activations."""
        with torch.no_grad():
            # Which features are active in this batch?
            active_mask = (hidden > 0).any(dim=0)

            # Update activation counts
            self.feature_activation_counts += active_mask.long()

            # Update steps since last activation
            self.steps_since_last_activation += 1
            self.steps_since_last_activation[active_mask] = 0

    def get_dead_feature_mask(self) -> torch.Tensor:
        """Get mask of dead features."""
        return self.steps_since_last_activation > self.config.dead_feature_threshold

    def train_step(
        self, activations: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            activations: Batch of activations (batch_size, d_input)

        Returns:
            Dictionary of metrics for this step
        """
        self.model.train()

        # Update learning rate
        self._update_lr()

        # Move to device
        activations = activations.to(self.device)

        # Forward pass
        x_hat, hidden = self.model(activations, return_hidden=True)

        # Compute loss
        total_loss, recon_loss, sparse_loss = sae_loss(
            activations, x_hat, hidden, self.config.sparsity_coefficient
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Normalize decoder if configured
        if (
            self.config.normalize_decoder
            and self.step % self.config.normalize_decoder_interval == 0
        ):
            self.model.normalize_decoder()

        # Update dead feature tracking
        self._update_dead_feature_tracking(hidden)

        # Check for dead feature resampling
        if (
            self.config.dead_feature_resample
            and self.step > 0
            and self.step % self.config.resample_check_interval == 0
        ):
            dead_mask = self.get_dead_feature_mask()
            if dead_mask.any():
                self.model.resample_dead_features(dead_mask, activations, self.optimizer)
                # Reset tracking for resampled features
                self.steps_since_last_activation[dead_mask] = 0

        self.step += 1

        # Compute metrics
        with torch.no_grad():
            l0 = (hidden > 0).float().sum(dim=1).mean()
            dead_count = self.get_dead_feature_mask().sum()

        metrics = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "sparse_loss": sparse_loss.item(),
            "avg_l0": l0.item(),
            "dead_features": dead_count.item(),
            "learning_rate": self.get_lr(),
        }

        self.train_history.append(metrics)
        return metrics

    @torch.no_grad()
    def evaluate(
        self, buffer: Union[ActivationBuffer, DiskActivationBuffer], num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate SAE on a set of activations.

        Args:
            buffer: Activation buffer to sample from
            num_batches: Number of batches to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_metrics = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "sparse_loss": 0.0,
            "avg_l0": 0.0,
            "dead_features": 0.0,
            "explained_variance": 0.0,
        }

        for _ in range(num_batches):
            activations = buffer.sample(self.config.batch_size).to(self.device)
            x_hat, hidden = self.model(activations, return_hidden=True)

            metrics = compute_loss_metrics(
                activations, x_hat, hidden, self.config.sparsity_coefficient
            )

            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]

        # Average
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics

    def train(
        self,
        buffer: Union[ActivationBuffer, DiskActivationBuffer],
        progress_callback: Optional[Callable[[SAETrainingProgress], None]] = None,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            buffer: Activation buffer to train on
            progress_callback: Optional callback for progress updates
            checkpoint_callback: Optional callback for checkpoint saving

        Returns:
            Dictionary with training results
        """
        import time
        start_time = time.time()

        try:
            for step in range(self.config.num_steps):
                # Sample batch
                activations = buffer.sample(self.config.batch_size)

                # Train step
                metrics = self.train_step(activations)

                # Log progress
                if self.step % self.config.log_interval == 0:
                    elapsed = int(time.time() - start_time)
                    progress = SAETrainingProgress(
                        step=self.step,
                        total_steps=self.config.num_steps,
                        loss=metrics["loss"],
                        recon_loss=metrics["recon_loss"],
                        sparse_loss=metrics["sparse_loss"],
                        avg_l0=metrics["avg_l0"],
                        dead_features=int(metrics["dead_features"]),
                        learning_rate=metrics["learning_rate"],
                        elapsed_seconds=elapsed,
                    )
                    if progress_callback:
                        progress_callback(progress)

                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(buffer)
                    if eval_metrics["loss"] < self.best_loss:
                        self.best_loss = eval_metrics["loss"]

                # Checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    if checkpoint_callback:
                        checkpoint_callback(self.step)

            # Final evaluation
            final_metrics = self.evaluate(buffer)
            elapsed = int(time.time() - start_time)

            return {
                "status": "completed",
                "final_metrics": final_metrics,
                "best_loss": self.best_loss,
                "total_steps": self.step,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            elapsed = int(time.time() - start_time)
            if progress_callback:
                progress = SAETrainingProgress(
                    step=self.step,
                    total_steps=self.config.num_steps,
                    loss=0.0,
                    recon_loss=0.0,
                    sparse_loss=0.0,
                    avg_l0=0.0,
                    dead_features=0,
                    learning_rate=0.0,
                    elapsed_seconds=elapsed,
                    status="failed",
                    error_message=str(e),
                )
                progress_callback(progress)

            return {
                "status": "failed",
                "error_message": str(e),
                "total_steps": self.step,
                "elapsed_seconds": elapsed,
            }

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for checkpointing."""
        return {
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "feature_activation_counts": self.feature_activation_counts.cpu(),
            "steps_since_last_activation": self.steps_since_last_activation.cpu(),
            "train_history": self.train_history,
        }

    def load_training_state(self, state: Dict[str, Any]):
        """Load training state from checkpoint."""
        self.step = state["step"]
        self.best_loss = state["best_loss"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.feature_activation_counts = state["feature_activation_counts"].to(self.device)
        self.steps_since_last_activation = state["steps_since_last_activation"].to(
            self.device
        )
        self.train_history = state.get("train_history", [])


def train_sae(
    sae_config: SAEConfig,
    training_config: SAETrainingConfig,
    buffer: Union[ActivationBuffer, DiskActivationBuffer],
    progress_callback: Optional[Callable[[SAETrainingProgress], None]] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    device: str = "cuda",
) -> tuple[SparseAutoencoderTorch, Dict[str, Any]]:
    """
    Convenience function to train an SAE from scratch.

    Args:
        sae_config: SAE architecture configuration
        training_config: Training configuration
        buffer: Activation buffer with training data
        progress_callback: Optional progress callback
        checkpoint_callback: Optional checkpoint callback
        device: Device for training

    Returns:
        Tuple of (trained model, training results)
    """
    # Create model
    model = SparseAutoencoderTorch(sae_config)

    # Create trainer
    trainer = SAETrainer(model, training_config, device=device)

    # Train
    results = trainer.train(
        buffer,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
    )

    return model, results
