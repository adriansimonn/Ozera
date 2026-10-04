"""
Abstract base class for open-source model loaders.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import os
import threading
import torch

from core.activation_limits import ActivationLimitError, max_capture_tokens
from core.tensor_codec import encode_tensor
from core.transformer.sampling import FINISH_EOS, FINISH_LENGTH, FINISH_STOP
from .registry import OPEN_SOURCE_MODELS, OpenSourceModelConfig


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variables."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _stop_criteria(should_stop: Callable[[], bool]):
    """
    A generate() stopping criterion that ends generation once should_stop() returns True.

    Its `fired` attribute says whether it did. (Defined here because transformers is only
    installed on the GPU workers.)
    """
    from transformers import StoppingCriteria

    class _ShouldStop(StoppingCriteria):
        fired = False

        def __call__(self, input_ids, scores, **kwargs):
            if should_stop():
                self.fired = True
            return torch.full((input_ids.shape[0],), self.fired, dtype=torch.bool, device=input_ids.device)

    return _ShouldStop()


class OpenSourceModelLoader(ABC):
    """
    Abstract base class for loading and running inference on open-source models.

    All model family loaders (SmolLM, Gemma, Qwen) inherit from this class and
    implement the abstract methods for model-specific loading and hook registration.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Initialize the loader.

        Args:
            model_id: Internal model ID (e.g., "smollm-135m")
            device: Device to load model on ("cuda" or "cpu")
        """
        if model_id not in OPEN_SOURCE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        self.model_id = model_id
        self.config: OpenSourceModelConfig = OPEN_SOURCE_MODELS[model_id]
        self.device = device
        self.model = None
        self.tokenizer = None
        self._hooks: list = []
        self._activations: dict[str, torch.Tensor] = {}
        # Forward hooks and self._activations are shared by every caller of this model,
        # so concurrent requests on one worker container must take turns
        self.lock = threading.RLock()

    @abstractmethod
    def load(self, cache_dir: str, token: Optional[str] = None) -> None:
        """
        Load model and tokenizer from cache directory.

        Args:
            cache_dir: Path to HuggingFace cache directory
            token: Optional HuggingFace token for gated models. If None,
                   will try to get from HF_TOKEN environment variable.
        """
        pass

    @abstractmethod
    def _register_hooks(self) -> None:
        """
        Register forward hooks for activation capture.

        Hooks are only attached for the capture forward pass (see
        generate_with_activations) so they add no overhead to generation.

        Must register hooks for:
        - Token embeddings
        - Per-layer: attention input, attention output, attention weights,
          post-attention residual, FFN input, FFN output, post-FFN residual
        - Final layer norm
        """
        pass

    def _clear_activations(self) -> None:
        """Clear captured activations before a new forward pass."""
        self._activations.clear()

    def _normalize_attention_weights(self, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Normalize attention weights, handling GQA if needed.

        Subclasses can override this to apply model-specific normalization
        (e.g., expanding grouped query attention to full head count).

        Args:
            attn_weights: Raw attention weights tensor
            layer_idx: Layer index (for model-specific handling)

        Returns:
            Normalized attention weights tensor
        """
        # Default implementation: no normalization
        # Subclasses override this for GQA normalization
        return attn_weights

    def encode_prompt(self, prompt: str):
        """
        Tokenize a prompt for this model.

        Instruct models get the prompt as a user message wrapped in the tokenizer's
        chat template (with the generation prompt appended); base models get raw text.

        Args:
            prompt: Input text

        Returns:
            BatchEncoding with input_ids and attention_mask on the model device
        """
        if not self.config.is_instruct:
            return self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Tokenizing via the template avoids a duplicate BOS (Gemma's template includes it)
        return self.tokenizer.apply_chat_template(
            self.chat_messages(prompt),
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 / SmolLM3 reasoning mode; ignored by other templates
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def chat_messages(self, prompt: str) -> list[dict]:
        """An instruct model's chat messages for a prompt: the prompt as a user message, after any system prompt."""
        messages = [{"role": "user", "content": prompt}]
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        return messages

    @property
    def eos_token_ids(self) -> list[int]:
        """
        IDs of the tokens that end a generation for this model.

        The registry's EOS tokens, looked up in the model's own tokenizer, plus any the
        checkpoint's generation config or tokenizer declares.
        """
        if getattr(self, "_eos_token_ids", None) is None:
            ids: list[int] = []
            unk_id = getattr(self.tokenizer, "unk_token_id", None)
            for token in self.config.eos_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id != unk_id:
                    ids.append(token_id)
            declared = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
            if isinstance(declared, int):
                declared = [declared]
            ids.extend(declared or [])
            if self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)
            self._eos_token_ids = list(dict.fromkeys(ids))
        return self._eos_token_ids

    def _generate(
        self,
        inputs,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        stop_at_eos: bool,
        should_stop: Optional[Callable[[], bool]],
        streamer=None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, str]:
        """
        Run model.generate() for one prompt (caller holds self.lock).

        Args:
            inputs: The prompt's BatchEncoding (from encode_prompt)
            stop_at_eos: End at the model's EOS tokens; otherwise generate max_new_tokens
                regardless of them
            should_stop: Checked after each token; generation ends once it returns True
            streamer: Optional transformers streamer that receives tokens as they're generated

        Returns:
            (the sequence's token IDs including the prompt, the finish reason)
        """
        # Temperature <= 0 or very close to 0 should use greedy decoding
        if temperature <= 0.01:
            temperature = 1.0
            do_sample = False

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": use_cache,
            # None overrides the checkpoint's generation config, so EOS doesn't end generation
            "eos_token_id": self.eos_token_ids if stop_at_eos else None,
        }

        # Add attention mask to avoid unexpected behavior when pad_token == eos_token
        if hasattr(inputs, "attention_mask"):
            gen_kwargs["attention_mask"] = inputs.attention_mask

        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        stop = None
        if should_stop is not None:
            from transformers import StoppingCriteriaList

            stop = _stop_criteria(should_stop)
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])

        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_ids, **gen_kwargs)[0]

        prompt_tokens = inputs.input_ids.shape[1]
        if stop is not None and stop.fired:
            finish_reason = FINISH_STOP
        elif (
            stop_at_eos
            and generated_ids.shape[0] > prompt_tokens
            and generated_ids[-1].item() in self.eos_token_ids
        ):
            finish_reason = FINISH_EOS
        else:
            finish_reason = FINISH_LENGTH
        return generated_ids, finish_reason

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
        stop_at_eos: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
        streamer=None,
    ) -> dict:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            use_cache: Whether to use the KV cache. Without it, every step recomputes the
                whole sequence (slower; patching needs it for some interventions).
            stop_at_eos: End at the model's EOS tokens (see eos_token_ids); otherwise
                generate exactly max_new_tokens
            should_stop: Checked before and during generation; generation ends once it
                returns True
            streamer: Optional transformers streamer that receives tokens as they're generated

        Returns:
            Dict with generated text, token info, and why generation ended ("finish_reason")
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.encode_prompt(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        with self.lock:
            if should_stop is not None and should_stop():
                # Stopped before it started (e.g. while the worker was starting up)
                generated_ids, finish_reason = inputs.input_ids[0], FINISH_STOP
            else:
                generated_ids, finish_reason = self._generate(
                    inputs, max_new_tokens, temperature, top_k, top_p, do_sample,
                    stop_at_eos, should_stop, streamer=streamer, use_cache=use_cache,
                )

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_tokens = generated_ids.shape[0]

        return {
            "text": generated_text,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens,
            "tokens": generated_ids.tolist(),
            "finish_reason": finish_reason,
        }

    def generate_with_activations(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        stop_at_eos: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
        streamer=None,
        fit_to_capture_limit: bool = False,
    ) -> dict:
        """
        Generate text and capture activations for visualization.

        Generation runs without capture hooks. A separate forward pass over the
        full sequence, with hooks attached only for its duration, captures layer
        activations (attention, FFN, etc.) and the logits used for top-k display.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            stop_at_eos: End at the model's EOS tokens; otherwise generate max_new_tokens
            should_stop: Checked before and during generation; generation ends once it
                returns True, and the tokens generated so far are captured
            streamer: Optional transformers streamer that receives tokens as they're generated
            fit_to_capture_limit: Generate at most as many tokens as can be captured,
                instead of refusing a max_new_tokens past the limit (for open-ended
                generation, where max_new_tokens is only a cap)

        Returns:
            Dict with generated text, token info, activations, and why generation ended
            ("finish_reason"). A request stopped before generating anything has no
            "activations".

        Raises:
            ActivationLimitError: the prompt plus max_new_tokens would capture more
                activations than MAX_CAPTURE_BYTES (checked before generating)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.encode_prompt(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        token_limit = self._capture_token_limit()
        if fit_to_capture_limit and prompt_tokens < token_limit:
            max_new_tokens = min(max_new_tokens, token_limit - prompt_tokens)
        if prompt_tokens + max_new_tokens > token_limit:
            raise ActivationLimitError(
                f"Visualizing {self.config.display_name} is limited to {token_limit} tokens "
                f"(prompt plus generated). This prompt has {prompt_tokens}, so generate at most "
                f"{max(token_limit - prompt_tokens, 0)} tokens."
            )

        base_result = {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
        }

        with self.lock:
            if should_stop is not None and should_stop():
                # Stopped before it started: nothing generated, so nothing to capture
                return {
                    **base_result,
                    "text": "",
                    "generated_tokens": 0,
                    "total_tokens": prompt_tokens,
                    "finish_reason": FINISH_STOP,
                }

            # Step 1: Generate tokens.
            generated_ids, finish_reason = self._generate(
                inputs, max_new_tokens, temperature, top_k, top_p, do_sample,
                stop_at_eos, should_stop, streamer=streamer,
            )

            total_tokens = generated_ids.shape[0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Step 2: Forward pass on complete sequence to capture layer activations
            # (attention weights, FFN outputs, embeddings, etc.) via hooks attached
            # just for this pass, and to obtain logits for top-k token display.
            forward_outputs = self._capture_forward(generated_ids.unsqueeze(0))

            # Use forward pass logits for top-k display. These are computed from a
            # full-sequence forward pass (no KV cache), so logits[0][i] gives the
            # model's prediction for token i+1 given tokens 0..i.
            self._activations["logits"] = forward_outputs.logits.detach()

            try:
                result = {
                    **base_result,
                    "text": generated_text,
                    "generated_tokens": total_tokens - prompt_tokens,
                    "total_tokens": total_tokens,
                    "tokens": generated_ids.tolist(),
                    "activations": self.get_activations_for_frontend(),
                    "decoded_tokens": [
                        self.tokenizer.decode([tok]) for tok in generated_ids.tolist()
                    ],
                    "finish_reason": finish_reason,
                }
            finally:
                # The result holds encoded copies; don't keep the GPU tensors until the next capture
                self._clear_activations()

        return result

    def _capture_token_limit(self) -> int:
        """Longest sequence whose captured activations fit in MAX_CAPTURE_BYTES."""
        return max_capture_tokens(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            bytes_per_value=next(self.model.parameters()).element_size(),
        )

    def _capture_forward(self, input_ids: torch.Tensor, **model_kwargs):
        """
        Run one forward pass with the capture hooks attached, filling self._activations.

        Caller holds self.lock. Attention weights the hooks didn't capture are taken from
        the model's outputs (with GQA normalization).

        Returns:
            The model's outputs
        """
        self._clear_activations()
        self._register_hooks()
        try:
            with torch.no_grad():
                forward_outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    return_dict=True,
                    **model_kwargs,
                )
        finally:
            self._remove_hooks()

        # Store attention weights if available (hooks may have already captured them
        # with GQA normalization, so only store if not already present)
        if hasattr(forward_outputs, "attentions") and forward_outputs.attentions is not None:
            for i, attn in enumerate(forward_outputs.attentions):
                if f"layer_{i}_attn_weights" not in self._activations:
                    # Apply GQA normalization if needed
                    attn_weights = self._normalize_attention_weights(attn.detach(), i)
                    self._activations[f"layer_{i}_attn_weights"] = attn_weights

        return forward_outputs

    def capture_prompt_activations(self, prompt: str) -> dict:
        """
        Capture the activations of one forward pass over a prompt, without generating.

        Used for patching sources, whose positions have to be the prompt's tokens.

        Returns:
            Dict with the prompt's tokens, decoded tokens, and activations (tensors on the
            model's device, keyed like "layer_0_attn_output"; the loader keeps no reference)

        Raises:
            ActivationLimitError: the prompt's activations would exceed MAX_CAPTURE_BYTES
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.encode_prompt(prompt)
        tokens = inputs.input_ids[0].tolist()

        token_limit = self._capture_token_limit()
        if len(tokens) > token_limit:
            raise ActivationLimitError(
                f"Capturing activations from {self.config.display_name} is limited to "
                f"{token_limit} prompt tokens; this prompt has {len(tokens)}."
            )

        with self.lock:
            try:
                # Only the last position's logits are computed: nothing uses them
                self._capture_forward(inputs.input_ids, use_cache=False, logits_to_keep=1)
                activations = dict(self._activations)
            finally:
                self._clear_activations()

        return {
            "tokens": tokens,
            "decoded_tokens": [self.tokenizer.decode([tok]) for tok in tokens],
            "activations": activations,
        }

    def get_activations_for_frontend(self) -> dict:
        """
        Format captured activations for the frontend visualization.

        Returns:
            Dict matching the ActivationData interface expected by frontend
        """
        layers = []
        num_layers = self.config.num_layers

        for i in range(num_layers):
            layer_data = {}

            # Map captured keys to frontend expected keys
            key_mapping = {
                "attn_input": f"layer_{i}_attn_input",
                "attn_output": f"layer_{i}_attn_output",
                "attn_weights": f"layer_{i}_attn_weights",
                "post_attn": f"layer_{i}_post_attn",
                "ff_input": f"layer_{i}_ff_input",
                "ff_output": f"layer_{i}_ff_output",
                "post_ff": f"layer_{i}_post_ff",
            }

            for frontend_key, storage_key in key_mapping.items():
                if storage_key in self._activations:
                    tensor = self._activations[storage_key]
                    layer_data[frontend_key] = self._tensor_to_data(tensor)

            layers.append(layer_data)

        return {
            "token_embeddings": self._tensor_to_data(
                self._activations.get("token_embeddings")
            ),
            "positional_embeddings": self._tensor_to_data(
                self._activations.get("positional_embeddings")
            ),
            "combined_embeddings": self._tensor_to_data(
                self._activations.get("combined_embeddings")
            ),
            "layers": layers,
            "final_layer_norm": self._tensor_to_data(
                self._activations.get("final_layer_norm")
            ),
            "top_k_logits": self._logits_to_topk_data(
                self._activations.get("logits"), k=20
            ),
        }

    def _tensor_to_data(self, tensor: Optional[torch.Tensor]) -> Optional[dict]:
        """
        Encode a PyTorch tensor for transport to the backend.

        The backend expands it to the TensorData format expected by the frontend.

        Args:
            tensor: PyTorch tensor or None

        Returns:
            Compact tensor dict (raw bytes, shape, dtype, and statistics), or None
        """
        if tensor is None:
            return None

        return encode_tensor(tensor)

    def _logits_to_topk_data(self, logits_tensor: Optional[torch.Tensor], k: int = 20) -> Optional[dict]:
        """
        Convert full logits tensor to compact top-K format with decoded tokens.

        Instead of sending the entire [1, seq_len, vocab_size] tensor (millions of floats),
        compute top-K on GPU and send only ~K*seq_len values plus decoded token strings.

        Args:
            logits_tensor: Logits tensor of shape [1, seq_len, vocab_size] or None
            k: Number of top predictions per position

        Returns:
            Dict with indices, values, probabilities, and decoded tokens, or None
        """
        if logits_tensor is None or self.tokenizer is None:
            return None

        # logits_tensor shape: [1, seq_len, vocab_size]
        logits = logits_tensor.float()
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab_size]

        seq_len, vocab_size = logits.shape
        actual_k = min(k, vocab_size)

        # Top-K on GPU
        top_values, top_indices = torch.topk(logits, actual_k, dim=-1)  # [seq_len, k]

        # Softmax probabilities for the top-K values
        # Use full logits for accurate softmax, then gather top-K probs
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.gather(probs, 1, top_indices)  # [seq_len, k]

        # Move to CPU and convert
        top_indices_list = top_indices.cpu().tolist()
        top_values_list = top_values.cpu().tolist()
        top_probs_list = top_probs.cpu().tolist()

        # Decode all unique token IDs
        unique_ids = set()
        for pos_indices in top_indices_list:
            unique_ids.update(pos_indices)

        id_to_token = {}
        for token_id in unique_ids:
            id_to_token[token_id] = self.tokenizer.decode([token_id])

        # Build decoded tokens grid
        decoded_tokens = [
            [id_to_token[tid] for tid in pos_indices]
            for pos_indices in top_indices_list
        ]

        return {
            "indices": top_indices_list,
            "values": top_values_list,
            "probabilities": top_probs_list,
            "decoded_tokens": decoded_tokens,
            "k": actual_k,
            "seq_len": seq_len,
        }

    def __del__(self):
        """Clean up hooks on deletion."""
        self._remove_hooks()
