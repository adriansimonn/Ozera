"""
Text generation utilities for Ozera models.
"""

import torch
from typing import Optional, Dict, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.model_torch import TransformerLM
from core.tokenizer import get_tokenizer


class TextGenerator:
    # Handles text generation from Ozera models.

    def __init__(self, model: TransformerLM, device: str = 'cpu'):
        """
        Initialize text generator.

        Args:
            model: Loaded TransformerLM model
            device: Device model is on
        """
        self.model = model
        self.device = device
        self.tokenizer = get_tokenizer()
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        return_metadata: bool = False
    ) -> str | Dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            prompt: Text prompt to start generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter (top-p)
            return_metadata: If True, return dict with text and metadata

        Returns:
            Generated text string, or dict with text and metadata
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        # Convert to tensor
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        # Generate
        generated_ids, _ = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_attention=False
        )

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())

        if return_metadata:
            return {
                'text': generated_text,
                'prompt': prompt,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': len(generated_ids[0]) - prompt_tokens,
                'total_tokens': len(generated_ids[0]),
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }

        return generated_text

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None
    ) -> list[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of text prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            List of generated text strings
        """
        results = []

        for prompt in prompts:
            generated = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_metadata=False
            )
            results.append(generated)

        return results

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
