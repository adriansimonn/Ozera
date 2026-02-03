"""
Data loader for SAE activation collection.

Streams text from OpenWebText dataset, runs forward passes through
Ozera models, and collects activations for SAE training.

Note: Uses OpenWebText (same distribution as Ozera model training)
instead of The Pile, which uses deprecated dataset scripts.
"""

import torch
import numpy as np
from typing import Optional, Iterator, List, Dict, Any
from pathlib import Path
import sys
import os

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

import tiktoken


class PileActivationCollector:
    """
    Collects activations from Ozera models running on The Pile dataset.

    Streams text from The Pile, tokenizes with GPT-2 BPE (matching
    Ozera model training), and collects target layer activations.
    """

    def __init__(
        self,
        model,
        config,
        target_layer: int,
        activation_type: str = "residual",
        batch_size: int = 32,
        seq_len: int = 256,
        device: str = "cuda",
    ):
        """
        Initialize the collector.

        Args:
            model: Loaded Ozera transformer model
            config: Model configuration (TransformerConfig)
            target_layer: Layer index to collect activations from
            activation_type: "residual" or "mlp_output"
            batch_size: Batch size for forward passes
            seq_len: Sequence length for tokenization
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.target_layer = target_layer
        self.activation_type = activation_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        # Use tiktoken GPT-2 BPE (same as Ozera training)
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Map activation type to key in activation dict
        self._activation_key_map = {
            "residual": "post_ff",
            "mlp_output": "ff_output",
            "attn_output": "attn_output",
        }

        # Pile dataset iterator (lazy loaded)
        self._pile_iter = None
        self._texts_consumed = 0

    def _get_pile_iterator(self) -> Iterator[Dict[str, Any]]:
        """Get streaming iterator over a diverse text dataset."""
        from datasets import load_dataset

        # Use OpenWebText - same distribution as Ozera model training
        # This is in standard Parquet format (no deprecated dataset scripts)
        dataset = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
        )

        return iter(dataset)

    def _ensure_pile_iterator(self):
        """Ensure Pile iterator is initialized."""
        if self._pile_iter is None:
            self._pile_iter = self._get_pile_iterator()

    def _get_next_texts(self, num_texts: int) -> List[str]:
        """
        Get next batch of texts from The Pile.

        Args:
            num_texts: Number of texts to fetch

        Returns:
            List of text strings
        """
        self._ensure_pile_iterator()

        texts = []
        for _ in range(num_texts):
            try:
                sample = next(self._pile_iter)
                texts.append(sample["text"])
                self._texts_consumed += 1
            except StopIteration:
                # Restart iterator if exhausted (unlikely with streaming)
                self._pile_iter = self._get_pile_iterator()
                sample = next(self._pile_iter)
                texts.append(sample["text"])
                self._texts_consumed += 1

        return texts

    def _tokenize_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Tensor of token IDs (batch_size, seq_len)
        """
        all_tokens = []

        for text in texts:
            # Encode text
            tokens = self.tokenizer.encode(text)

            # Truncate or pad to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            else:
                # Pad with 0 (end of text token for GPT-2)
                tokens = tokens + [0] * (self.seq_len - len(tokens))

            all_tokens.append(tokens)

        return torch.tensor(all_tokens, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def collect_batch(self) -> torch.Tensor:
        """
        Collect one batch of activations from The Pile.

        Returns:
            Tensor of activations (batch_size * seq_len, d_model)
        """
        # Get texts from Pile
        texts = self._get_next_texts(self.batch_size)

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

        # Reshape from (batch, seq, d) to (batch * seq, d)
        batch_size, seq_len, d_model = target_activations.shape
        flattened = target_activations.reshape(-1, d_model)

        return flattened

    def collect_activations(
        self,
        buffer,
        num_samples: int,
        progress_callback=None,
    ) -> int:
        """
        Collect activations into a buffer.

        Args:
            buffer: ActivationBuffer or DiskActivationBuffer to fill
            num_samples: Target number of activation vectors to collect
            progress_callback: Optional callback(collected, total) for progress

        Returns:
            Number of activations collected
        """
        collected = 0
        samples_per_batch = self.batch_size * self.seq_len

        while collected < num_samples:
            # Collect batch
            activations = self.collect_batch()

            # Add to buffer
            buffer.add(activations.cpu())

            collected += activations.shape[0]

            if progress_callback:
                progress_callback(min(collected, num_samples), num_samples)

        return collected

    @property
    def d_model(self) -> int:
        """Get model hidden dimension."""
        return self.config.d_model

    @property
    def texts_consumed(self) -> int:
        """Number of Pile texts consumed so far."""
        return self._texts_consumed


def collect_activations_for_sae(
    model_name: str,
    target_layer: int,
    activation_type: str,
    output_path: str,
    num_samples: int = 2_000_000,
    batch_size: int = 32,
    seq_len: int = 256,
    device: str = "cuda",
    models_dir: str = None,
) -> Dict[str, Any]:
    """
    Convenience function to collect activations for SAE training.

    Args:
        model_name: "nano" or "mini"
        target_layer: Layer index
        activation_type: "residual" or "mlp_output"
        output_path: Path for HDF5 activation file
        num_samples: Number of activation vectors to collect
        batch_size: Batch size for forward passes
        seq_len: Sequence length
        device: Device for computation
        models_dir: Directory containing model checkpoints

    Returns:
        Dict with collection stats
    """
    from inference.model_loader import load_model
    from core.sae.activation_buffer import DiskActivationBuffer

    # Default models directory
    if models_dir is None:
        models_dir = str(backend_dir / "models")

    print(f"Loading {model_name} model...")
    model, config = load_model(model_name, models_dir=models_dir, device=device)

    print(f"Creating activation buffer at {output_path}...")
    buffer = DiskActivationBuffer(
        filepath=output_path,
        d_input=config.d_model,
        max_samples=num_samples,
    )

    print(f"Initializing Pile collector for layer {target_layer} {activation_type}...")
    collector = PileActivationCollector(
        model=model,
        config=config,
        target_layer=target_layer,
        activation_type=activation_type,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    def progress_callback(collected, total):
        pct = 100.0 * collected / total
        print(f"\rCollecting activations: {collected:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

    print(f"Collecting {num_samples:,} activations from The Pile...")
    actual_collected = collector.collect_activations(
        buffer=buffer,
        num_samples=num_samples,
        progress_callback=progress_callback,
    )
    print()  # Newline after progress

    stats = buffer.get_stats()
    buffer.close()

    return {
        "model_name": model_name,
        "target_layer": target_layer,
        "activation_type": activation_type,
        "d_model": config.d_model,
        "num_collected": actual_collected,
        "texts_consumed": collector.texts_consumed,
        "output_path": output_path,
        "stats": {
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "memory_mb": stats.memory_mb,
        },
    }


if __name__ == "__main__":
    # Test collection
    import argparse

    parser = argparse.ArgumentParser(description="Collect activations from The Pile")
    parser.add_argument("--model", choices=["nano", "mini"], default="nano")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--activation", choices=["residual", "mlp_output"], default="residual")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="/tmp/test_activations.h5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    result = collect_activations_for_sae(
        model_name=args.model,
        target_layer=args.layer,
        activation_type=args.activation,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device,
    )

    print("\nCollection complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")
