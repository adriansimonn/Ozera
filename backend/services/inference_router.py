"""
Inference router for routing requests to local or Modal workers.

Routes inference requests based on INFERENCE_MODE environment variable:
- "local": Use local model loading and inference (development)
- "modal": Use Modal cloud GPU inference (production)
"""

import os
from typing import AsyncIterator, Optional

# Inference mode configuration
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "local")

# Model size thresholds for GPU tier selection (in parameters)
TIER_THRESHOLDS = {
    "t4": 100_000_000,      # Up to 100M params
    "a10g": 1_000_000_000,  # Up to 1B params
    "a100": float("inf"),   # 1B+ params
}

# Base models are always available
BASE_MODELS = {"nano", "mini"}


class InferenceRouter:
    """Routes inference requests to appropriate backend."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize inference router.

        Args:
            models_dir: Local models directory for local mode
        """
        self.models_dir = models_dir
        self._local_loader = None
        self._local_generators = {}

    def get_inference_mode(self) -> str:
        """Get current inference mode."""
        return INFERENCE_MODE

    def is_modal_mode(self) -> bool:
        """Check if using Modal for inference."""
        return INFERENCE_MODE == "modal"

    def _get_local_loader(self):
        """Get or create local model loader."""
        if self._local_loader is None:
            from inference.model_loader import ModelLoader
            self._local_loader = ModelLoader(models_dir=self.models_dir)
        return self._local_loader

    def _get_local_generator(self, model_id: str):
        """Get or create local text generator for a model."""
        if model_id not in self._local_generators:
            from inference.text_generator import TextGenerator

            loader = self._get_local_loader()
            model, config = loader.load_model(model_id)
            device = "cuda" if self._is_cuda_available() else "cpu"

            self._local_generators[model_id] = TextGenerator(
                model=model,
                device=device,
                model_name=model_id,
            )
        return self._local_generators[model_id]

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_gpu_tier(self, model_id: str) -> str:
        """
        Determine GPU tier for a model.

        Args:
            model_id: Model identifier

        Returns:
            GPU tier ('t4', 'a10g', or 'a100')
        """
        # Base models use T4
        if model_id in BASE_MODELS:
            return "t4"

        # For custom models, we'd need to look up size
        # Default to T4 for now (can be enhanced with model registry)
        return "t4"

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """
        Generate text using the appropriate backend.

        Args:
            model_id: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generation result dict
        """
        if self.is_modal_mode():
            return await self._generate_modal(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            )
        else:
            return self._generate_local(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            )

    def _generate_local(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate using local inference."""
        generator = self._get_local_generator(model_id)
        result = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_metadata=True,
        )
        result["model"] = model_id
        return result

    async def _generate_modal(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model_id)
        worker = get_inference_worker(gpu_tier)

        result = await worker().generate.remote.aio(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return result

    async def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream text generation.

        Args:
            model_id: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Yields:
            Generated text tokens
        """
        if self.is_modal_mode():
            async for token in self._stream_modal(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            ):
                yield token
        else:
            for token in self._stream_local(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            ):
                yield token

    def _stream_local(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ):
        """Stream using local inference."""
        generator = self._get_local_generator(model_id)
        yield from generator.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    async def _stream_modal(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> AsyncIterator[str]:
        """Stream using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model_id)
        worker = get_inference_worker(gpu_tier)

        async for token in worker().generate_stream.remote_gen.aio(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            yield token

    async def generate_with_activations(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """
        Generate text with activation capture.

        Note: For Modal mode, activations are returned inline rather than
        stored in the activation store (since the store is per-process).

        Args:
            model_id: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generation result with activations or activation_id
        """
        if self.is_modal_mode():
            return await self._generate_with_activations_modal(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            )
        else:
            return self._generate_with_activations_local(
                model_id, prompt, max_tokens, temperature, top_k, top_p
            )

    def _generate_with_activations_local(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate with activations using local inference."""
        generator = self._get_local_generator(model_id)
        result = generator.generate_with_activations(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        result["model"] = model_id
        return result

    async def _generate_with_activations_modal(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate with activations using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model_id)
        worker = get_inference_worker(gpu_tier)

        result = await worker().generate_with_activations.remote.aio(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return result

    async def get_model_info(self, model_id: str) -> dict:
        """
        Get model information.

        Args:
            model_id: Model identifier

        Returns:
            Model info dict
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            gpu_tier = self.get_gpu_tier(model_id)
            worker = get_inference_worker(gpu_tier)
            return await worker().get_model_info.remote.aio(model_id)
        else:
            loader = self._get_local_loader()
            model, config = loader.load_model(model_id)
            return {
                "name": model_id,
                "parameters": config.count_parameters(),
                "layers": config.num_layers,
                "heads": config.num_heads,
                "hidden_dim": config.d_model,
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
            }

    async def list_models(self) -> list:
        """
        List available models.

        Returns:
            List of model info dicts
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            worker = get_inference_worker("t4")
            return await worker().list_models.remote.aio()
        else:
            loader = self._get_local_loader()
            model_names = loader.list_available_models(include_remote=True)
            return [
                {"id": name, "type": "base" if name in BASE_MODELS else "custom"}
                for name in model_names
            ]

    async def warmup_model(self, model_id: str) -> bool:
        """
        Pre-load a model.

        Args:
            model_id: Model to warm up

        Returns:
            True if successful
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            gpu_tier = self.get_gpu_tier(model_id)
            worker = get_inference_worker(gpu_tier)
            return await worker().warmup.remote.aio(model_id)
        else:
            try:
                self._get_local_generator(model_id)
                return True
            except Exception:
                return False


# Global router instance
_router: Optional[InferenceRouter] = None


def get_inference_router(models_dir: str = "models") -> InferenceRouter:
    """Get or create the global inference router."""
    global _router
    if _router is None:
        _router = InferenceRouter(models_dir=models_dir)
    return _router
