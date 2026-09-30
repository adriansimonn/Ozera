"""
Inference router for routing requests to local or Modal workers.

Routes inference requests based on INFERENCE_MODE environment variable:
- "local": Use local model loading and inference (development)
- "modal": Use Modal cloud GPU inference (production)
"""

import os
from typing import TYPE_CHECKING, AsyncIterator, Optional

from core.open_source import get_gpu_tier

if TYPE_CHECKING:
    from services.custom_models import ModelRef

# Inference mode configuration
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "local")

# Model size thresholds for GPU tier selection (in parameters)
TIER_THRESHOLDS = {
    "l4": 100_000_000,      # Up to 100M params
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

    def _get_local_generator(self, model: "ModelRef"):
        """Get or create local text generator for a model (keyed by owner and version too)."""
        key = (model.name, model.owner_id, model.version)
        if key not in self._local_generators:
            from inference.text_generator import TextGenerator

            loader = self._get_local_loader()
            lm, config = loader.load_model(model.name, owner_id=model.owner_id, version=model.version)
            device = "cuda" if self._is_cuda_available() else "cpu"

            # Drop generators for older versions of the same model
            for old_key in [k for k in self._local_generators if k[:2] == key[:2]]:
                del self._local_generators[old_key]
            self._local_generators[key] = TextGenerator(
                model=lm,
                device=device,
                model_name=model.name,
            )
        return self._local_generators[key]

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
            GPU tier ('l4' or 'a10g')
        """
        # Open-source models run on the tier registered for them (3-4B models need A10G);
        # base and custom Ozera models use L4
        return get_gpu_tier(model_id)

    async def generate(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> dict:
        """
        Generate text using the appropriate backend.

        Args:
            model: Model to use (resolved for the requesting user)
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
                model, prompt, max_tokens, temperature, top_k, top_p
            )
        else:
            return self._generate_local(
                model, prompt, max_tokens, temperature, top_k, top_p
            )

    def _generate_local(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate using local inference."""
        generator = self._get_local_generator(model)
        result = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_metadata=True,
        )
        result["model"] = model.name
        return result

    async def _generate_modal(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model.name)
        worker = get_inference_worker(gpu_tier)

        result = await worker().generate.remote.aio(
            model_id=model.name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **model.worker_kwargs(),
        )
        return result

    async def generate_stream(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream text generation.

        Args:
            model: Model to use (resolved for the requesting user)
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
                model, prompt, max_tokens, temperature, top_k, top_p
            ):
                yield token
        else:
            for token in self._stream_local(
                model, prompt, max_tokens, temperature, top_k, top_p
            ):
                yield token

    def _stream_local(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ):
        """Stream using local inference."""
        generator = self._get_local_generator(model)
        yield from generator.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    async def _stream_modal(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> AsyncIterator[str]:
        """Stream using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model.name)
        worker = get_inference_worker(gpu_tier)

        async for token in worker().generate_stream.remote_gen.aio(
            model_id=model.name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **model.worker_kwargs(),
        ):
            yield token

    async def generate_with_activations(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        user_id: Optional[int] = None,
    ) -> dict:
        """
        Generate text with activation capture.

        Note: For Modal mode, activations are returned inline rather than
        stored in the activation store (since the store is per-process).

        Args:
            model: Model to use (resolved for the requesting user)
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            user_id: Owner of the stored activations (local mode stores them here)

        Returns:
            Generation result with activations or activation_id
        """
        if self.is_modal_mode():
            return await self._generate_with_activations_modal(
                model, prompt, max_tokens, temperature, top_k, top_p
            )
        else:
            return self._generate_with_activations_local(
                model, prompt, max_tokens, temperature, top_k, top_p, user_id
            )

    def _generate_with_activations_local(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        user_id: Optional[int],
    ) -> dict:
        """Generate with activations using local inference."""
        generator = self._get_local_generator(model)
        result = generator.generate_with_activations(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            user_id=user_id,
        )
        result["model"] = model.name
        return result

    async def _generate_with_activations_modal(
        self,
        model: "ModelRef",
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> dict:
        """Generate with activations using Modal inference."""
        from services.modal_inference import get_inference_worker

        gpu_tier = self.get_gpu_tier(model.name)
        worker = get_inference_worker(gpu_tier)

        result = await worker().generate_with_activations.remote.aio(
            model_id=model.name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **model.worker_kwargs(),
        )
        return result

    async def get_model_info(self, model: "ModelRef") -> dict:
        """
        Get model information.

        Args:
            model: Model to inspect (resolved for the requesting user)

        Returns:
            Model info dict
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            gpu_tier = self.get_gpu_tier(model.name)
            worker = get_inference_worker(gpu_tier)
            return await worker().get_model_info.remote.aio(model.name, **model.worker_kwargs())
        else:
            loader = self._get_local_loader()
            _, config = loader.load_model(model.name, owner_id=model.owner_id, version=model.version)
            return {
                "name": model.name,
                "parameters": config.count_parameters(),
                "layers": config.num_layers,
                "heads": config.num_heads,
                "hidden_dim": config.d_model,
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
            }

    async def list_models(self) -> list:
        """
        List the shared models (custom models are listed per user, from the database).

        Returns:
            List of model info dicts
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            worker = get_inference_worker("l4")
            return await worker().list_models.remote.aio()
        else:
            loader = self._get_local_loader()
            return [{"id": name, "type": "base"} for name in loader.list_available_models()]

    async def warmup_model(self, model: "ModelRef") -> bool:
        """
        Pre-load a model.

        Args:
            model: Model to warm up (resolved for the requesting user)

        Returns:
            True if successful
        """
        if self.is_modal_mode():
            from services.modal_inference import get_inference_worker

            gpu_tier = self.get_gpu_tier(model.name)
            worker = get_inference_worker(gpu_tier)
            return await worker().warmup.remote.aio(model.name, **model.worker_kwargs())
        else:
            try:
                self._get_local_generator(model)
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
