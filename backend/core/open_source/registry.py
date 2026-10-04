"""
Registry of supported open-source models with metadata.
"""

from enum import Enum
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

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
    gpu_tier: str           # "l4" or "a10g"
    # Tokens that end a generation, from each checkpoint's generation_config.json and
    # tokenizer (instruct models end a turn with their chat template's end-of-turn token)
    eos_tokens: tuple[str, ...]
    requires_auth: bool = False  # If HF token needed
    is_instruct: bool = False  # Prompts are wrapped in the tokenizer's chat template
    system_prompt: Optional[str] = None  # System message for the chat template (None = template default)


# SmolLM2 models - uses standard Llama architecture
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
    gpu_tier="l4",
    eos_tokens=("<|endoftext|>",),
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
    gpu_tier="l4",
    eos_tokens=("<|endoftext|>",),
)

# SmolLM3 - Llama-style layout with NoPE (no rotary embedding) on every 4th layer
# Reference: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
SMOLLM3_3B = OpenSourceModelConfig(
    model_id="smollm3-3b",
    hf_id="HuggingFaceTB/SmolLM3-3B-Base",
    family=ModelFamily.SMOLLM,
    display_name="SmolLM3 3B",
    parameters=3_000_000_000,
    num_layers=36,
    num_heads=16,
    num_kv_heads=4,  # GQA
    hidden_dim=2048,
    intermediate_dim=11008,
    vocab_size=128256,
    max_seq_len=65536,
    gpu_tier="a10g",
    eos_tokens=("<|end_of_text|>",),
)

# Gemma 3 models - GQA, GeGLU, 5:1 local (512-token sliding window) : global attention
# Reference: https://huggingface.co/google/gemma-3-1b-pt
GEMMA_3_270M = OpenSourceModelConfig(
    model_id="gemma-3-270m",
    hf_id="google/gemma-3-270m",
    family=ModelFamily.GEMMA,
    display_name="Gemma 3 270M",
    parameters=270_000_000,
    num_layers=18,
    num_heads=4,
    num_kv_heads=1,  # GQA
    hidden_dim=640,
    intermediate_dim=2048,
    vocab_size=262144,
    max_seq_len=32768,
    gpu_tier="l4",
    eos_tokens=("<eos>", "<end_of_turn>"),
    requires_auth=True,  # Gated: Gemma license must be accepted on HF
)

GEMMA_3_1B = OpenSourceModelConfig(
    model_id="gemma-3-1b",
    hf_id="google/gemma-3-1b-pt",
    family=ModelFamily.GEMMA,
    display_name="Gemma 3 1B",
    parameters=1_000_000_000,
    num_layers=26,
    num_heads=4,
    num_kv_heads=1,  # GQA
    hidden_dim=1152,
    intermediate_dim=6912,
    vocab_size=262144,
    max_seq_len=32768,
    gpu_tier="l4",
    eos_tokens=("<eos>", "<end_of_turn>"),
    requires_auth=True,  # Gated: Gemma license must be accepted on HF
)

# Qwen3 models - GQA with QK-norm, head_dim fixed at 128
# Reference: https://huggingface.co/Qwen/Qwen3-0.6B-Base
QWEN3_0_6B = OpenSourceModelConfig(
    model_id="qwen3-0.6b",
    hf_id="Qwen/Qwen3-0.6B-Base",
    family=ModelFamily.QWEN,
    display_name="Qwen3 0.6B",
    parameters=600_000_000,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,  # GQA
    hidden_dim=1024,
    intermediate_dim=3072,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="l4",
    eos_tokens=("<|endoftext|>",),
)

QWEN3_1_7B = OpenSourceModelConfig(
    model_id="qwen3-1.7b",
    hf_id="Qwen/Qwen3-1.7B-Base",
    family=ModelFamily.QWEN,
    display_name="Qwen3 1.7B",
    parameters=1_700_000_000,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,  # GQA
    hidden_dim=2048,
    intermediate_dim=6144,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="l4",
    eos_tokens=("<|endoftext|>",),
)

QWEN3_4B = OpenSourceModelConfig(
    model_id="qwen3-4b",
    hf_id="Qwen/Qwen3-4B-Base",
    family=ModelFamily.QWEN,
    display_name="Qwen3 4B",
    parameters=4_000_000_000,
    num_layers=36,
    num_heads=32,
    num_kv_heads=8,  # GQA
    hidden_dim=2560,
    intermediate_dim=9728,
    vocab_size=151936,
    max_seq_len=32768,
    gpu_tier="a10g",
    eos_tokens=("<|endoftext|>",),
)


# Instruction-tuned variants - same architecture as their base models, so they reuse
# the base config and loader; only the checkpoint, ID, and context length differ.
def _instruct(
    base: OpenSourceModelConfig,
    hf_id: str,
    end_of_turn: str,
    max_seq_len: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> OpenSourceModelConfig:
    return replace(
        base,
        model_id=f"{base.model_id}-it",
        hf_id=hf_id,
        display_name=f"{base.display_name} Instruct",
        max_seq_len=max_seq_len or base.max_seq_len,
        # The chat template's end-of-turn token first; the base model's end of text still ends it
        eos_tokens=tuple(dict.fromkeys((end_of_turn, *base.eos_tokens))),
        is_instruct=True,
        system_prompt=system_prompt,
    )


SMOLLM_135M_IT = _instruct(SMOLLM_135M, "HuggingFaceTB/SmolLM2-135M-Instruct", "<|im_end|>", max_seq_len=8192)
SMOLLM_360M_IT = _instruct(SMOLLM_360M, "HuggingFaceTB/SmolLM2-360M-Instruct", "<|im_end|>", max_seq_len=8192)
# SmolLM3's default system prompt injects today's date; /system_override keeps the
# default persona without it so the same prompt tokenizes identically every day.
SMOLLM3_3B_IT = _instruct(
    SMOLLM3_3B,
    "HuggingFaceTB/SmolLM3-3B",
    "<|im_end|>",
    system_prompt="You are a helpful AI assistant named SmolLM, trained by Hugging Face. /system_override",
)
GEMMA_3_270M_IT = _instruct(GEMMA_3_270M, "google/gemma-3-270m-it", "<end_of_turn>")
GEMMA_3_1B_IT = _instruct(GEMMA_3_1B, "google/gemma-3-1b-it", "<end_of_turn>")
# Qwen3 post-trained checkpoints (hybrid thinking; thinking is disabled when templating)
QWEN3_0_6B_IT = _instruct(QWEN3_0_6B, "Qwen/Qwen3-0.6B", "<|im_end|>", max_seq_len=40960)
QWEN3_1_7B_IT = _instruct(QWEN3_1_7B, "Qwen/Qwen3-1.7B", "<|im_end|>", max_seq_len=40960)
QWEN3_4B_IT = _instruct(QWEN3_4B, "Qwen/Qwen3-4B", "<|im_end|>", max_seq_len=40960)


OPEN_SOURCE_MODELS: dict[str, OpenSourceModelConfig] = {
    cfg.model_id: cfg
    for cfg in [
        SMOLLM_135M, SMOLLM_135M_IT,
        SMOLLM_360M, SMOLLM_360M_IT,
        SMOLLM3_3B, SMOLLM3_3B_IT,
        GEMMA_3_270M, GEMMA_3_270M_IT,
        GEMMA_3_1B, GEMMA_3_1B_IT,
        QWEN3_0_6B, QWEN3_0_6B_IT,
        QWEN3_1_7B, QWEN3_1_7B_IT,
        QWEN3_4B, QWEN3_4B_IT,
    ]
}


# Tier for everything that isn't a registered open-source model (Ozera base and custom models)
DEFAULT_GPU_TIER = "l4"


def get_gpu_tier(model_id: str) -> str:
    """
    Get the GPU tier ('l4' or 'a10g') whose inference worker serves a model.

    Args:
        model_id: Open-source model ID, or an Ozera base/custom model name

    Returns:
        GPU tier name
    """
    if model_id in OPEN_SOURCE_MODELS:
        return OPEN_SOURCE_MODELS[model_id].gpu_tier
    return DEFAULT_GPU_TIER


def instruct_sibling(model_id: str) -> Optional[str]:
    """
    The other half of a model's base/instruct pair (same architecture and shapes), or None.

    Args:
        model_id: An open-source model ID (e.g. "qwen3-0.6b" or "qwen3-0.6b-it")
    """
    config = OPEN_SOURCE_MODELS.get(model_id)
    if config is None:
        return None
    sibling = model_id[:-len("-it")] if config.is_instruct else f"{model_id}-it"
    return sibling if sibling in OPEN_SOURCE_MODELS else None


def model_for_hf_id(hf_id: str) -> Optional[str]:
    """
    The registered model with a HuggingFace ID (case-insensitive), or None.

    Gemma Scope 2 names Gemma 3 270M's checkpoint "google/gemma-3-270m-pt", which doesn't
    exist (the checkpoint is "google/gemma-3-270m"), so a "-pt" suffix is also tried without.
    """
    wanted = hf_id.strip().lower()
    by_hf = {config.hf_id.lower(): model_id for model_id, config in OPEN_SOURCE_MODELS.items()}
    if wanted in by_hf:
        return by_hf[wanted]
    if wanted.endswith("-pt"):
        return by_hf.get(wanted[:-len("-pt")])
    return None


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
