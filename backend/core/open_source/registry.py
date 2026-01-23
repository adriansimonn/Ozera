"""
Registry of supported open-source models with metadata.
"""

from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import OpenSourceModelLoader


class ModelFamily(Enum):
    GEMMA = "gemma"
    QWEN = "qwen"
    SMOLLM = "smollm"


@dataclass
class OpenSourceModelConfig:
    """Configuration for an open-source model."""

    model_id: str           # Internal ID (e.g., "smollm-135m")
    hf_id: str              # HuggingFace ID
    family: ModelFamily
    display_name: str
    parameters: int         # Parameter count
    num_layers: int
    num_heads: int
    num_kv_heads: int       # For GQA (same as num_heads if not GQA)
    hidden_dim: int
    intermediate_dim: int   # FFN intermediate dimension
    vocab_size: int
    max_seq_len: int
    gpu_tier: str           # "t4" or "a10g"
    requires_auth: bool = False  # If HF token needed


# SmolLM models - uses standard Llama architecture
# Reference: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
SMOLLM_135M = OpenSourceModelConfig(
    model_id="smollm-135m",
    hf_id="HuggingFaceTB/SmolLM2-135M",
    family=ModelFamily.SMOLLM,
    display_name="SmolLM2 135M",
    parameters=135_000_000,
    num_layers=30,
    num_heads=9,
    num_kv_heads=3,  # GQA with 3 KV heads
    hidden_dim=576,
    intermediate_dim=1536,
    vocab_size=49152,
    max_seq_len=2048,
    gpu_tier="t4",
)

SMOLLM_360M = OpenSourceModelConfig(
    model_id="smollm-360m",
    hf_id="HuggingFaceTB/SmolLM2-360M",
    family=ModelFamily.SMOLLM,
    display_name="SmolLM2 360M",
    parameters=360_000_000,
    num_layers=32,
    num_heads=15,
    num_kv_heads=5,  # GQA
    hidden_dim=960,
    intermediate_dim=2560,
    vocab_size=49152,
    max_seq_len=2048,
    gpu_tier="t4",
)

SMOLLM_1_7B = OpenSourceModelConfig(
    model_id="smollm-1.7b",
    hf_id="HuggingFaceTB/SmolLM2-1.7B",
    family=ModelFamily.SMOLLM,
    display_name="SmolLM2 1.7B",
    parameters=1_700_000_000,
    num_layers=24,
    num_heads=32,
    num_kv_heads=32,  # No GQA
    hidden_dim=2048,
    intermediate_dim=8192,
    vocab_size=49152,
    max_seq_len=2048,
    gpu_tier="t4",
)

# Gemma models - uses GQA and GeGLU
# Reference: https://huggingface.co/google/gemma-2-2b
GEMMA_2_2B = OpenSourceModelConfig(
    model_id="gemma-2-2b",
    hf_id="google/gemma-2-2b",
    family=ModelFamily.GEMMA,
    display_name="Gemma 2 2B",
    parameters=2_600_000_000,
    num_layers=26,
    num_heads=8,
    num_kv_heads=4,  # GQA
    hidden_dim=2304,
    intermediate_dim=9216,
    vocab_size=256000,
    max_seq_len=8192,
    gpu_tier="a10g",
)

# Qwen models - uses GQA
# Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B
QWEN_0_5B = OpenSourceModelConfig(
    model_id="qwen-0.5b",
    hf_id="Qwen/Qwen2.5-0.5B",
    family=ModelFamily.QWEN,
    display_name="Qwen 2.5 0.5B",
    parameters=500_000_000,
    num_layers=24,
    num_heads=14,
    num_kv_heads=2,  # GQA
    hidden_dim=896,
    intermediate_dim=4864,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="t4",
)

QWEN_1_5B = OpenSourceModelConfig(
    model_id="qwen-1.5b",
    hf_id="Qwen/Qwen2.5-1.5B",
    family=ModelFamily.QWEN,
    display_name="Qwen 2.5 1.5B",
    parameters=1_500_000_000,
    num_layers=28,
    num_heads=12,
    num_kv_heads=2,  # GQA
    hidden_dim=1536,
    intermediate_dim=8960,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="t4",
)

QWEN_3B = OpenSourceModelConfig(
    model_id="qwen-3b",
    hf_id="Qwen/Qwen2.5-3B",
    family=ModelFamily.QWEN,
    display_name="Qwen 2.5 3B",
    parameters=3_000_000_000,
    num_layers=36,
    num_heads=16,
    num_kv_heads=2,  # GQA
    hidden_dim=2048,
    intermediate_dim=11008,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="a10g",
)


OPEN_SOURCE_MODELS: dict[str, OpenSourceModelConfig] = {
    "smollm-135m": SMOLLM_135M,
    "smollm-360m": SMOLLM_360M,
    "smollm-1.7b": SMOLLM_1_7B,
    "gemma-2-2b": GEMMA_2_2B,
    "qwen-0.5b": QWEN_0_5B,
    "qwen-1.5b": QWEN_1_5B,
    "qwen-3b": QWEN_3B,
}


def get_loader_for_model(model_id: str) -> "OpenSourceModelLoader":
    """
    Get the appropriate loader class instance for a model.

    Args:
        model_id: The internal model ID (e.g., "smollm-135m")

    Returns:
        An instance of the appropriate loader class

    Raises:
        ValueError: If model_id is not found in registry
    """
    if model_id not in OPEN_SOURCE_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(OPEN_SOURCE_MODELS.keys())}")

    config = OPEN_SOURCE_MODELS[model_id]

    # Import loaders here to avoid circular imports
    if config.family == ModelFamily.SMOLLM:
        from .smollm import SmolLMLoader
        return SmolLMLoader(model_id)
    elif config.family == ModelFamily.GEMMA:
        from .gemma import GemmaLoader
        return GemmaLoader(model_id)
    elif config.family == ModelFamily.QWEN:
        from .qwen import QwenLoader
        return QwenLoader(model_id)
    else:
        raise ValueError(f"No loader implemented for family: {config.family}")
