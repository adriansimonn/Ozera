"""
Activation buffer for efficient SAE training.

Collects and stores activations from transformer models for SAE training.
Supports both in-memory and disk-based storage for large datasets.
"""

import numpy as np
import torch
from typing import Optional, Dict, Iterator, Tuple, List, Union
from pathlib import Path
import h5py
from dataclasses import dataclass


@dataclass
class BufferStats:
    """Statistics about the activation buffer."""
    num_samples: int
    d_input: int
    mean: float
    std: float
    min: float
    max: float
    memory_mb: float


class ActivationBuffer:
    """
    In-memory buffer for collecting activations for SAE training.

    Efficiently stores activations and provides random sampling for training.
    Uses a ring buffer design to limit memory usage.
    """

    def __init__(
        self,
        d_input: int,
        capacity: int = 100_000,
        device: str = "cpu",
    ):
        """
        Initialize activation buffer.

        Args:
            d_input: Dimension of activations
            capacity: Maximum number of activations to store
            device: Device to store tensors on ("cpu" or "cuda")
        """
        self.d_input = d_input
        self.capacity = capacity
        self.device = device

        # Pre-allocate storage
        self.buffer = torch.zeros(capacity, d_input, device=device)
        self.write_idx = 0
        self.num_stored = 0

    def add(self, activations: torch.Tensor):
        """
        Add activations to the buffer.

        Args:
            activations: Tensor of activations (batch_size, d_input)
                        or (batch_size, seq_len, d_input)
        """
        # Flatten if 3D (batch, seq, d_input) -> (batch * seq, d_input)
        if activations.dim() == 3:
            activations = activations.reshape(-1, self.d_input)

        # Move to buffer device
        activations = activations.to(self.device)

        num_new = activations.shape[0]

        # Handle wrap-around for ring buffer
        if self.write_idx + num_new <= self.capacity:
            # Simple case: no wrap
            self.buffer[self.write_idx : self.write_idx + num_new] = activations
            self.write_idx += num_new
        else:
            # Wrap around
            first_chunk = self.capacity - self.write_idx
            self.buffer[self.write_idx :] = activations[:first_chunk]
            remaining = num_new - first_chunk
            self.buffer[:remaining] = activations[first_chunk:]
            self.write_idx = remaining

        self.num_stored = min(self.num_stored + num_new, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample a random batch of activations.

        Args:
            batch_size: Number of samples to return

        Returns:
            Tensor of sampled activations (batch_size, d_input)
        """
        if self.num_stored == 0:
            raise ValueError("Buffer is empty")

        # Sample random indices
        indices = torch.randint(0, self.num_stored, (batch_size,), device=self.device)
        return self.buffer[indices]

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """Alias for sample() for compatibility."""
        return self.sample(batch_size)

    def get_all(self) -> torch.Tensor:
        """Get all stored activations."""
        return self.buffer[: self.num_stored]

    def clear(self):
        """Clear the buffer."""
        self.buffer.zero_()
        self.write_idx = 0
        self.num_stored = 0

    def __len__(self) -> int:
        """Number of activations stored."""
        return self.num_stored

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.num_stored >= self.capacity

    def get_stats(self) -> BufferStats:
        """Compute statistics about stored activations."""
        data = self.get_all()
        return BufferStats(
            num_samples=self.num_stored,
            d_input=self.d_input,
            mean=data.mean().item(),
            std=data.std().item(),
            min=data.min().item(),
            max=data.max().item(),
            memory_mb=data.element_size() * data.numel() / (1024 * 1024),
        )


class StreamingActivationBuffer:
    """
    Streaming activation buffer that collects from model forward passes.

    Designed for use during SAE training to continuously collect fresh
    activations from the target model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        texts: List[str],
        target_layer: int,
        activation_type: str = "residual",  # "residual", "mlp_output", "attn_output"
        batch_size: int = 32,
        seq_len: int = 128,
        buffer_size: int = 100_000,
        device: str = "cuda",
    ):
        """
        Initialize streaming buffer.

        Args:
            model: Transformer model to collect activations from
            tokenizer: Tokenizer for processing texts
            texts: List of text strings to process
            target_layer: Which layer to collect activations from
            activation_type: Type of activation to collect
            batch_size: Batch size for model forward passes
            seq_len: Sequence length for tokenization
            buffer_size: Size of internal activation buffer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.texts = texts
        self.target_layer = target_layer
        self.activation_type = activation_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        # Infer activation dimension from model config
        self.d_input = model.config.d_model

        # Internal buffer
        self.buffer = ActivationBuffer(
            d_input=self.d_input,
            capacity=buffer_size,
            device=device,
        )

        # Text iterator state
        self.text_idx = 0

        # Map activation type to key in activation dict
        self._activation_key_map = {
            "residual": "post_ff",
            "mlp_output": "ff_output",
            "attn_output": "attn_output",
        }

    def _tokenize_batch(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a batch of texts."""
        # Simple tokenization - pad/truncate to seq_len
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            else:
                # Pad with 0 (assuming 0 is pad token)
                tokens = tokens + [0] * (self.seq_len - len(tokens))
            all_tokens.append(tokens)

        return torch.tensor(all_tokens, dtype=torch.long, device=self.device)

    def _get_next_texts(self) -> List[str]:
        """Get next batch of texts, cycling if needed."""
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.texts[self.text_idx % len(self.texts)])
            self.text_idx += 1
        return batch

    @torch.no_grad()
    def collect_batch(self) -> int:
        """
        Collect one batch of activations from the model.

        Returns:
            Number of activations collected
        """
        # Get next batch of texts
        texts = self._get_next_texts()

        # Tokenize
        input_ids = self._tokenize_batch(texts)

        # Forward pass with activation capture
        self.model.eval()
        _, _, activations = self.model(
            input_ids,
            return_attention=False,
            capture_activations=True,
        )

        # Extract target activations
        layer_activations = activations["layers"][self.target_layer]
        activation_key = self._activation_key_map[self.activation_type]
        target_activations = layer_activations[activation_key]

        # Add to buffer (will be flattened from (batch, seq, d) to (batch*seq, d))
        self.buffer.add(target_activations)

        return target_activations.numel() // self.d_input

    def fill_buffer(self):
        """Fill the buffer to capacity."""
        while not self.buffer.is_full():
            self.collect_batch()

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample from the buffer, refilling if needed."""
        # Ensure we have enough data
        if len(self.buffer) < batch_size:
            self.collect_batch()

        return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DiskActivationBuffer:
    """
    Disk-based activation buffer using HDF5 for large-scale training.

    Stores activations on disk with memory-mapped access for efficient
    random sampling without loading everything into memory.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        d_input: int,
        max_samples: Optional[int] = None,
        chunk_size: int = 10_000,
    ):
        """
        Initialize disk-based buffer.

        Args:
            filepath: Path to HDF5 file for storage
            d_input: Dimension of activations
            max_samples: Maximum number of samples (None for unlimited)
            chunk_size: Chunk size for HDF5 storage
        """
        self.filepath = Path(filepath)
        self.d_input = d_input
        self.max_samples = max_samples
        self.chunk_size = chunk_size

        # Create or open HDF5 file
        self.h5file = h5py.File(self.filepath, "a")

        # Create dataset if it doesn't exist
        if "activations" not in self.h5file:
            maxshape = (max_samples, d_input) if max_samples else (None, d_input)
            self.h5file.create_dataset(
                "activations",
                shape=(0, d_input),
                maxshape=maxshape,
                chunks=(chunk_size, d_input),
                dtype=np.float32,
            )

        self.dataset = self.h5file["activations"]

    def add(self, activations: Union[np.ndarray, torch.Tensor]):
        """
        Add activations to disk storage.

        Args:
            activations: Activations to add (N, d_input)
        """
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()

        # Flatten if 3D
        if activations.ndim == 3:
            activations = activations.reshape(-1, self.d_input)

        # Resize dataset
        current_size = self.dataset.shape[0]
        new_size = current_size + activations.shape[0]

        if self.max_samples and new_size > self.max_samples:
            # Truncate to max
            space_left = self.max_samples - current_size
            if space_left > 0:
                activations = activations[:space_left]
                new_size = self.max_samples
            else:
                return  # No space left

        self.dataset.resize(new_size, axis=0)
        self.dataset[current_size:new_size] = activations

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample random activations from disk.

        Args:
            batch_size: Number of samples

        Returns:
            Tensor of activations (batch_size, d_input)
        """
        num_samples = len(self)
        if num_samples == 0:
            raise ValueError("Buffer is empty")

        # Sample random indices
        indices = np.random.choice(num_samples, size=batch_size, replace=True)
        indices = np.sort(indices)  # Sort for more efficient disk access

        # Load samples
        samples = self.dataset[indices]
        return torch.from_numpy(samples)

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def close(self):
        """Close the HDF5 file."""
        self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_stats(self) -> BufferStats:
        """Compute statistics about stored activations."""
        if len(self) == 0:
            return BufferStats(
                num_samples=0,
                d_input=self.d_input,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                memory_mb=0.0,
            )

        # Sample subset for stats to avoid loading everything
        sample_size = min(10_000, len(self))
        sample = self.sample(sample_size).numpy()

        file_size_mb = self.filepath.stat().st_size / (1024 * 1024)

        return BufferStats(
            num_samples=len(self),
            d_input=self.d_input,
            mean=float(sample.mean()),
            std=float(sample.std()),
            min=float(sample.min()),
            max=float(sample.max()),
            memory_mb=file_size_mb,
        )


def create_activation_iterator(
    buffer: Union[ActivationBuffer, DiskActivationBuffer],
    batch_size: int,
    num_batches: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    """
    Create an iterator over activation batches.

    Args:
        buffer: Activation buffer to sample from
        batch_size: Batch size
        num_batches: Number of batches (None for infinite)

    Yields:
        Tensor of activations (batch_size, d_input)
    """
    batch_count = 0
    while num_batches is None or batch_count < num_batches:
        yield buffer.sample(batch_size)
        batch_count += 1
