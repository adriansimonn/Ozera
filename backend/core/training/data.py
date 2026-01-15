"""
Data loading and processing for Ozera training.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, List


class SimpleTokenizer:
    """
    Simple character-level tokenizer for initial testing.

    For production, use a proper tokenizer (BPE, WordPiece, etc.)
    """

    def __init__(self, text: str):
        """
        Initialize tokenizer with vocabulary from text.

        Args:
            text: Training text to build vocabulary from
        """
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text."""
        return ''.join([self.id_to_char.get(id, '?') for id in token_ids])


class TextDataset:
    """
    Simple dataset for language modeling.

    Creates sliding window sequences from text.
    """

    def __init__(
        self,
        text: str,
        tokenizer: SimpleTokenizer,
        seq_len: int = 128
    ):
        """
        Initialize dataset.

        Args:
            text: Training text
            tokenizer: Tokenizer instance
            seq_len: Sequence length
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Tokenize text
        self.token_ids = np.array(tokenizer.encode(text))
        self.num_tokens = len(self.token_ids)

        # Calculate number of sequences
        self.num_sequences = max(1, (self.num_tokens - 1) // seq_len)

    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sequence.

        Args:
            idx: Sequence index

        Returns:
            input_ids: Input token IDs (seq_len,)
            target_ids: Target token IDs (seq_len,)
        """
        start_idx = idx * self.seq_len

        # Get input sequence
        input_ids = self.token_ids[start_idx:start_idx + self.seq_len]

        # Get target sequence (shifted by 1)
        target_ids = self.token_ids[start_idx + 1:start_idx + self.seq_len + 1]

        # Pad if necessary
        if len(input_ids) < self.seq_len:
            input_ids = np.pad(
                input_ids,
                (0, self.seq_len - len(input_ids)),
                constant_values=0
            )
        if len(target_ids) < self.seq_len:
            target_ids = np.pad(
                target_ids,
                (0, self.seq_len - len(target_ids)),
                constant_values=0
            )

        return input_ids, target_ids


class DataLoader:
    """
    Data loader with batching and shuffling.
    """

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        """
        Initialize data loader.

        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_batches = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.num_batches += 1

    def __len__(self) -> int:
        """Number of batches."""
        return self.num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches.

        Yields:
            input_batch: Input token IDs (batch_size, seq_len)
            target_batch: Target token IDs (batch_size, seq_len)
        """
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))

            batch_indices = indices[start_idx:end_idx]

            # Collect batch
            input_batch = []
            target_batch = []

            for idx in batch_indices:
                input_ids, target_ids = self.dataset[idx]
                input_batch.append(input_ids)
                target_batch.append(target_ids)

            yield np.array(input_batch), np.array(target_batch)


def load_text_file(file_path: str) -> str:
    """
    Load text from file.

    Args:
        file_path: Path to text file

    Returns:
        Text content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def train_test_split(
    text: str,
    train_ratio: float = 0.9
) -> Tuple[str, str]:
    """
    Split text into train and test sets.

    Args:
        text: Input text
        train_ratio: Proportion of data for training

    Returns:
        train_text: Training text
        test_text: Test text
    """
    split_idx = int(len(text) * train_ratio)
    return text[:split_idx], text[split_idx:]
