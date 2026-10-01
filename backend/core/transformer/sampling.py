"""
Token-by-token generation for Ozera models, shared by the GPU worker and the local generator.

Generation ends after max_new_tokens, at the end-of-sequence token when one is given, or
when the caller's should_stop() says so (a user pressed stop).
"""

from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F

# Why a generation ended (reported to the client)
FINISH_LENGTH = "length"  # Generated the requested number of tokens
FINISH_EOS = "eos"  # The model generated an end-of-sequence token
FINISH_STOP = "stop"  # Stopped on request


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    """
    Pick the next token from the last position's logits.

    Args:
        logits: Logits of shape (batch, vocab_size); modified in place when sampling
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering
        top_p: Nucleus (top-p) filtering

    Returns:
        Token IDs of shape (batch, 1)
    """
    if temperature < 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens above the threshold, keeping the first token that crosses it
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class TokenStream:
    """
    Generates one sequence's tokens one at a time.

    Iterating yields each new token ID. Afterwards input_ids holds the prompt and every
    generated token, and finish_reason says why generation ended. Once the sequence is
    longer than the model's context window, each step sees only its last max_seq_len tokens.
    """

    def __init__(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        vocab_size: int,
        eos_token_id: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        """
        Args:
            model: TransformerLM
            input_ids: Prompt token IDs, shape (1, prompt_len), on the model's device
            max_new_tokens: Most tokens to generate
            temperature, top_k, top_p: Sampling parameters (see sample_next_token)
            vocab_size: Tokenizer vocabulary size; sampled IDs are clamped below it
            eos_token_id: Stop after generating this token (None = don't stop at EOS)
            should_stop: Checked before each token; generation ends once it returns True
        """
        self.model = model
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.should_stop = should_stop
        self.finish_reason: Optional[str] = None

    def __iter__(self) -> Iterator[int]:
        with torch.no_grad():
            max_len = self.model.config.max_seq_len
            for _ in range(self.max_new_tokens):
                if self.should_stop is not None and self.should_stop():
                    self.finish_reason = FINISH_STOP
                    return

                idx_cond = self.input_ids if self.input_ids.size(1) <= max_len else self.input_ids[:, -max_len:]
                logits, _, _ = self.model.forward(idx_cond, return_attention=False, capture_activations=False)
                next_token = sample_next_token(logits[:, -1, :], self.temperature, self.top_k, self.top_p)
                next_token = torch.clamp(next_token, 0, self.vocab_size - 1)
                self.input_ids = torch.cat([self.input_ids, next_token], dim=1)

                token = next_token.item()
                yield token
                if token == self.eos_token_id:
                    self.finish_reason = FINISH_EOS
                    return

            self.finish_reason = FINISH_LENGTH


class TextDeltas:
    """
    Turns generated tokens into the text each one adds, for streaming.

    Decodes everything generated so far and emits what's new, so a character whose bytes
    span several tokens comes out once it's complete.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted = 0  # Leading tokens whose text has been emitted

    def push(self, token_id: int) -> str:
        """Add a generated token; returns the text it completes (may be empty)."""
        self._ids.append(token_id)
        current = self.tokenizer.decode(self._ids)
        previous = self.tokenizer.decode(self._ids[: self._emitted]) if self._emitted else ""
        new_text = current[len(previous):]
        if new_text:
            self._emitted = len(self._ids)
        return new_text
