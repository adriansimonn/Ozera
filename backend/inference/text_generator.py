"""
Text generation utilities for Ozera models.
"""

import torch
from typing import Optional, Dict, Any, Iterator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.model_torch import TransformerLM
from core.tokenizer import get_tokenizer
from inference.activation_store import get_activation_store


class TextGenerator:
    # Handles text generation from Ozera models.

    def __init__(self, model: TransformerLM, device: str = 'cpu', model_name: str = 'unknown'):
        """
        Initialize text generator.

        Args:
            model: Loaded TransformerLM model
            device: Device model is on
            model_name: Name of the model for activation tracking
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.tokenizer = get_tokenizer()
        self.model.eval()
        self.activation_store = get_activation_store()

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

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None
    ) -> Iterator[str]:
        """
        Generate text from a prompt with streaming (yields tokens as generated).

        Args:
            prompt: Text prompt to start generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Yields:
            Generated text token by token
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        self.model.eval()

        # Track the number of tokens we've already yielded text for
        num_yielded_tokens = len(prompt_ids)

        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Get logits for current sequence
            idx_cond = input_ids if input_ids.size(1) <= self.model.config.max_seq_len else input_ids[:, -self.model.config.max_seq_len:]

            logits, _, _ = self.model.forward(idx_cond, return_attention=False, capture_activations=False)
            logits = logits[:, -1, :]

            # Handle temperature
            if temperature == 0.0:
                # Greedy decoding - just pick the argmax
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Clamp token to valid range
            next_token = torch.clamp(next_token, 0, self.tokenizer.vocab_size - 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode only the new token(s) to get the delta text
            # We decode from the last yielded position to handle multi-byte UTF-8 properly
            current_ids = input_ids[0].cpu().tolist()
            current_text = self.tokenizer.decode(current_ids)
            previous_text = self.tokenizer.decode(current_ids[:num_yielded_tokens])

            # Yield only the new text delta
            new_text = current_text[len(previous_text):]
            if new_text:
                yield new_text
                num_yielded_tokens = len(current_ids)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    @torch.no_grad()
    def generate_with_activations(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text and capture activations from the final forward pass.

        Args:
            prompt: Text prompt to start generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary with generated text, activation ID, and metadata
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        self.model.eval()

        # Generate tokens (similar to generate_stream but without yielding)
        for _ in range(max_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.model.config.max_seq_len else input_ids[:, -self.model.config.max_seq_len:]

            logits, _, _ = self.model.forward(idx_cond, return_attention=False, capture_activations=False)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.clamp(next_token, 0, self.tokenizer.vocab_size - 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Now do one final forward pass with activation capture
        final_ids = input_ids if input_ids.size(1) <= self.model.config.max_seq_len else input_ids[:, -self.model.config.max_seq_len:]
        _, _, activations = self.model.forward(final_ids, return_attention=True, capture_activations=True)

        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0].cpu().tolist())

        # Decode individual tokens for visualization
        token_list = input_ids[0].cpu().tolist()
        decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in token_list]

        # Store activations
        activation_id = self.activation_store.store_activations(
            activations=activations,
            tokens=input_ids[0].cpu().tolist(),
            prompt=prompt,
            model_name=self.model_name,
            metadata={
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'prompt_tokens': len(prompt_ids),
                'generated_tokens': len(input_ids[0]) - len(prompt_ids),
                'total_tokens': len(input_ids[0]),
                'generated_text': generated_text,
                'decoded_tokens': decoded_tokens
            }
        )

        return {
            'text': generated_text,
            'activation_id': activation_id,
            'prompt': prompt,
            'prompt_tokens': len(prompt_ids),
            'generated_tokens': len(input_ids[0]) - len(prompt_ids),
            'total_tokens': len(input_ids[0]),
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }

    @torch.no_grad()
    def generate_with_patches(
        self,
        prompt: str,
        patches: Dict[str, Dict[str, Any]],
        max_tokens: int = 50,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate text with activation patches applied.

        Patches are applied during the forward pass to replace activations
        at specified layers with source activations from a different prompt.

        Args:
            prompt: Text prompt to generate from
            patches: Dict mapping activation keys to patch info:
                {
                    "layer_0_attn_output": {
                        "source": tensor,  # Source activation to patch in
                        "positions": [0, 1, 2] or None,  # Positions to patch
                        "blend_factor": 1.0,  # 1.0 = full replacement
                    },
                    ...
                }
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Dict with generated text, tokens, and patching info
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        self.model.eval()

        # Generate with patches
        generated_ids, _ = self.model.generate_with_patches(
            input_ids,
            patches=patches,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        token_list = generated_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(token_list)
        decoded_tokens = [self.tokenizer.decode([t]) for t in token_list]

        return {
            'text': generated_text,
            'prompt': prompt,
            'prompt_tokens': len(prompt_ids),
            'generated_tokens': len(token_list) - len(prompt_ids),
            'total_tokens': len(token_list),
            'tokens': token_list,
            'decoded_tokens': decoded_tokens,
            'patched': True,
            'num_patches': len(patches),
        }

    @torch.no_grad()
    def capture_activations(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Capture activations for a prompt without generating new tokens.

        This is used to capture source activations for patching experiments.

        Args:
            prompt: Text prompt to capture activations for

        Returns:
            Dict with activation tensors and metadata
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        self.model.eval()

        # Forward pass with activation capture
        _, _, activations = self.model.forward(
            input_ids,
            return_attention=True,
            capture_activations=True,
        )

        # Flatten activations to dict format
        flat_activations = {}

        # Top-level activations
        for key in ['token_embeddings', 'positional_embeddings', 'combined_embeddings', 'final_layer_norm', 'logits']:
            if key in activations and activations[key] is not None:
                flat_activations[key] = activations[key]

        # Layer activations
        if 'layers' in activations:
            for layer_idx, layer_data in enumerate(activations['layers']):
                if layer_data is None:
                    continue
                for key, tensor in layer_data.items():
                    if tensor is not None:
                        flat_activations[f"layer_{layer_idx}_{key}"] = tensor

        # Decode tokens
        token_list = input_ids[0].cpu().tolist()
        decoded_tokens = [self.tokenizer.decode([t]) for t in token_list]

        return {
            'prompt': prompt,
            'tokens': token_list,
            'decoded_tokens': decoded_tokens,
            'activations': flat_activations,
            'num_layers': self.model.config.num_layers,
            'model_name': self.model_name,
        }

    def get_patchable_keys(self) -> list[str]:
        """
        Get list of all patchable activation keys.

        Returns:
            List of activation key strings that can be patched
        """
        return self.model.get_patchable_activation_keys()
