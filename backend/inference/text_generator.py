"""
Text generation utilities for Ozera models.
"""

import torch
from typing import Callable, Optional, Dict, Any, Iterator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transformer.model_torch import TransformerLM
from core.transformer.sampling import FINISH_EOS, FINISH_STOP, TextDeltas, TokenStream
from core.activation_limits import check_fits_context
from core.probes.monitor import ProbeMonitor
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

    def _token_stream(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        stop_at_eos: bool,
        should_stop: Optional[Callable[[], bool]] = None,
        monitor: Optional[ProbeMonitor] = None,
    ) -> TokenStream:
        """
        A generation from the prompt (see core.transformer.sampling.TokenStream), read by
        the monitor if one is given.
        """
        self.model.eval()
        model = self.model
        if monitor is not None:
            model = monitor.ozera_model(self.model, lambda: stream.input_ids.size(1))
        stream = TokenStream(
            model,
            torch.tensor([prompt_ids], dtype=torch.long).to(self.device),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            vocab_size=self.tokenizer.vocab_size,
            eos_token_id=self.tokenizer.eos_token_id if stop_at_eos else None,
            should_stop=should_stop,
        )
        return stream

    def _monitor(self, probe: Optional[Dict[str, Any]]) -> Optional[ProbeMonitor]:
        """A monitor for a probe ({"layer", "weights", "bias"}), or None without one."""
        if not probe:
            return None
        return ProbeMonitor(probe["layer"], probe["weights"], probe["bias"], lambda i: self.tokenizer.decode([i]))

    def _text(self, token_list: list[int], finish_reason: Optional[str]) -> str:
        """A sequence's text, without the EOS that ended it."""
        return self.tokenizer.decode(token_list[:-1] if finish_reason == FINISH_EOS else token_list)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        return_metadata: bool = False,
        stop_at_eos: bool = False,
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
            stop_at_eos: End at the end-of-sequence token; otherwise generate exactly max_tokens

        Returns:
            Generated text string, or dict with text and metadata
        """
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        stream = self._token_stream(prompt_ids, max_tokens, temperature, top_k, top_p, stop_at_eos)
        for _ in stream:
            pass

        token_list = stream.input_ids[0].cpu().tolist()
        generated_text = self._text(token_list, stream.finish_reason)

        if return_metadata:
            return {
                'text': generated_text,
                'prompt': prompt,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': len(token_list) - prompt_tokens,
                'total_tokens': len(token_list),
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'finish_reason': stream.finish_reason,
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

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        report_usage: bool = False,
        stop_at_eos: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
        probe: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str | Dict[str, Any]]:
        """
        Generate text from a prompt with streaming (yields tokens as generated).

        Args:
            prompt: Text prompt to start generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            report_usage: Finish with a dict of the real token counts
                ({"prompt_tokens", "generated_tokens"}) and why generation ended
                ("finish_reason"); requests are billed by it
            stop_at_eos: End at the end-of-sequence token (not streamed); otherwise generate
                exactly max_tokens
            should_stop: Checked before each token; generation ends once it returns True
            probe: {"layer", "weights", "bias"} of a probe to score each token with

        Yields:
            Generated text token by token (and with a probe, its {"event": "probe", ...}
            scores; see core.probes.monitor)
        """
        prompt_ids = self.tokenizer.encode(prompt)
        monitor = self._monitor(probe)
        stream = self._token_stream(prompt_ids, max_tokens, temperature, top_k, top_p, stop_at_eos, should_stop, monitor)

        deltas = TextDeltas(self.tokenizer)
        for token in stream:
            if monitor is not None:
                yield from monitor.take_events()
            if stream.eos_token_id is not None and token == stream.eos_token_id:
                continue  # The EOS that ends the generation isn't part of its text
            new_text = deltas.push(token)
            if new_text:
                yield new_text
        if monitor is not None:
            yield from monitor.take_events()

        if report_usage:
            yield {
                'prompt_tokens': len(prompt_ids),
                'generated_tokens': stream.input_ids.size(1) - len(prompt_ids),
                'finish_reason': stream.finish_reason,
            }

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def generate_with_activations(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        user_id: Optional[int] = None,
        stop_at_eos: bool = False,
        fit_to_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text and capture activations from the final forward pass.

        See generate_with_activations_stream; this returns only its result.
        """
        for item in self.generate_with_activations_stream(
            prompt, max_tokens, temperature, top_k, top_p, user_id, stop_at_eos, fit_to_context,
        ):
            if isinstance(item, dict) and 'event' not in item:
                return item
        raise RuntimeError("Generation ended without a result")

    def generate_with_activations_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        user_id: Optional[int] = None,
        stop_at_eos: bool = False,
        fit_to_context: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
        probe: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str | Dict[str, Any]]:
        """
        Generate text, streaming it, and capture activations from a final forward pass.

        Args:
            prompt: Text prompt to start generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            user_id: User the stored activations belong to
            stop_at_eos: End at the end-of-sequence token; otherwise generate exactly max_tokens
            fit_to_context: Lower max_tokens to what fits the context window instead of
                refusing the request
            should_stop: Checked before each token; generation ends once it returns True,
                and the tokens generated so far are captured
            probe: {"layer", "weights", "bias"} of a probe to score each token with

        Yields:
            Text chunks, then {"event": "generated"} when the capture starts, then a dict
            with the generated text, activation ID (none if stopped before generating
            anything), and metadata (and with a probe, its {"event": "probe", ...} scores)

        Raises:
            ActivationLimitError: the prompt plus max_tokens doesn't fit the model's context window
        """
        prompt_ids = self.tokenizer.encode(prompt)
        context_len = self.model.config.max_seq_len
        if fit_to_context and len(prompt_ids) < context_len:
            max_tokens = min(max_tokens, context_len - len(prompt_ids))
        # The whole sequence has to fit the context window for the capture to match the tokens
        check_fits_context(self.model_name, len(prompt_ids), max_tokens, context_len)

        monitor = self._monitor(probe)
        stream = self._token_stream(prompt_ids, max_tokens, temperature, top_k, top_p, stop_at_eos, should_stop, monitor)
        deltas = TextDeltas(self.tokenizer)
        for token in stream:
            if monitor is not None:
                yield from monitor.take_events()
            if stream.eos_token_id is not None and token == stream.eos_token_id:
                continue  # The EOS that ends the generation isn't part of its text
            new_text = deltas.push(token)
            if new_text:
                yield new_text
        yield {'event': 'generated'}

        input_ids = stream.input_ids
        token_list = input_ids[0].cpu().tolist()
        generated_text = self._text(token_list, stream.finish_reason)
        result = {
            'text': generated_text,
            'prompt': prompt,
            'prompt_tokens': len(prompt_ids),
            'generated_tokens': len(token_list) - len(prompt_ids),
            'total_tokens': len(token_list),
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'finish_reason': stream.finish_reason,
        }
        if stream.finish_reason == FINISH_STOP and len(token_list) == len(prompt_ids):
            # Stopped before generating anything: nothing to capture
            yield result
            return

        # Now do one final forward pass over the whole sequence with activation capture
        # (it also gives the monitor the last token)
        capture_model = self.model if monitor is None else monitor.ozera_model(self.model, lambda: input_ids.size(1))
        with torch.no_grad():
            _, _, activations = capture_model.forward(input_ids, return_attention=True, capture_activations=True)
        if monitor is not None:
            yield from monitor.take_events()

        # Decode individual tokens for visualization
        decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in token_list]

        # Store activations
        result['activation_id'] = self.activation_store.store_activations(
            user_id=user_id,
            activations=activations,
            tokens=token_list,
            prompt=prompt,
            model_name=self.model_name,
            metadata={
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'prompt_tokens': len(prompt_ids),
                'generated_tokens': len(token_list) - len(prompt_ids),
                'total_tokens': len(token_list),
                'generated_text': generated_text,
                'decoded_tokens': decoded_tokens
            }
        )
        yield result

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
