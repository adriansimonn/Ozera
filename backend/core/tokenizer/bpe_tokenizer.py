"""
BPE Tokenizer for Ozera models.

Uses GPT-2's tokenizer (via tiktoken) for efficient subword tokenization.
"""

import tiktoken
from typing import List, Union


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer using GPT-2's vocabulary.

    Much more efficient than character-level tokenization.
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Size of vocabulary (50257 for GPT-2)
        """
        # Use GPT-2's tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token_id = 50256  # <|endoftext|>
        self.eos_token_id = 50256  # <|endoftext|>

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs (can be list or list of lists)

        Returns:
            Decoded text
        """
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [self.tokenizer.decode(ids, errors='ignore') for ids in token_ids]
        else:
            # Filter out invalid token IDs and decode with error handling
            valid_ids = [tid for tid in token_ids if 0 <= tid < self.vocab_size]
            return self.tokenizer.decode(valid_ids, errors='ignore')

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


def get_tokenizer(vocab_size: int = 50257) -> BPETokenizer:
    """
    Get BPE tokenizer instance.

    Args:
        vocab_size: Vocabulary size

    Returns:
        BPETokenizer instance
    """
    return BPETokenizer(vocab_size)


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = get_tokenizer()

    test_text = "Hello world! This is a test of the BPE tokenizer."
    print(f"Original text: {test_text}")

    # Encode
    tokens = tokenizer.encode(test_text)
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Decode
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded}")

    assert decoded == test_text, "Encoding/decoding mismatch!"
    print("\n✓ Tokenizer test passed!")
