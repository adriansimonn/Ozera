/**
 * Static model data for all known models.
 * These are hardcoded so the UI renders instantly without waiting for Modal.
 * Only actual generation/experiments need to call Modal.
 */

import type { OpenSourceModelInfo } from '../types/model'
import type { ModelInfo } from '../api/client'

export const STATIC_BASE_MODELS: string[] = ['nano', 'mini']

export const STATIC_BASE_MODEL_INFO: Record<string, ModelInfo> = {
  nano: {
    name: 'nano',
    parameters: 12_412_608,
    layers: 6,
    heads: 6,
    hidden_dim: 192,
    vocab_size: 50257,
  },
  mini: {
    name: 'mini',
    parameters: 51_197_440,
    layers: 8,
    heads: 8,
    hidden_dim: 512,
    vocab_size: 50257,
  },
}

export const STATIC_OPEN_SOURCE_MODELS: OpenSourceModelInfo[] = [
  {
    id: 'smollm-135m',
    hf_id: 'HuggingFaceTB/SmolLM2-135M',
    display_name: 'SmolLM2 135M',
    family: 'smollm',
    parameters: 135_000_000,
    layers: 30,
    heads: 9,
    kv_heads: 3,
    hidden_dim: 576,
    intermediate_dim: 1536,
    vocab_size: 49152,
    max_seq_len: 2048,
    gpu_tier: 't4',
  },
  {
    id: 'smollm-360m',
    hf_id: 'HuggingFaceTB/SmolLM2-360M',
    display_name: 'SmolLM2 360M',
    family: 'smollm',
    parameters: 360_000_000,
    layers: 32,
    heads: 15,
    kv_heads: 5,
    hidden_dim: 960,
    intermediate_dim: 2560,
    vocab_size: 49152,
    max_seq_len: 2048,
    gpu_tier: 't4',
  },
  {
    id: 'smollm-1.7b',
    hf_id: 'HuggingFaceTB/SmolLM2-1.7B',
    display_name: 'SmolLM2 1.7B',
    family: 'smollm',
    parameters: 1_700_000_000,
    layers: 24,
    heads: 32,
    kv_heads: 32,
    hidden_dim: 2048,
    intermediate_dim: 8192,
    vocab_size: 49152,
    max_seq_len: 2048,
    gpu_tier: 't4',
  },
  {
    id: 'gemma-2-2b',
    hf_id: 'google/gemma-2-2b',
    display_name: 'Gemma 2 2B',
    family: 'gemma',
    parameters: 2_600_000_000,
    layers: 26,
    heads: 8,
    kv_heads: 4,
    hidden_dim: 2304,
    intermediate_dim: 9216,
    vocab_size: 256000,
    max_seq_len: 8192,
    gpu_tier: 'a10g',
  },
  {
    id: 'qwen-0.5b',
    hf_id: 'Qwen/Qwen2.5-0.5B',
    display_name: 'Qwen 2.5 0.5B',
    family: 'qwen',
    parameters: 500_000_000,
    layers: 24,
    heads: 14,
    kv_heads: 2,
    hidden_dim: 896,
    intermediate_dim: 4864,
    vocab_size: 151936,
    max_seq_len: 32768,
    gpu_tier: 't4',
  },
  {
    id: 'qwen-1.5b',
    hf_id: 'Qwen/Qwen2.5-1.5B',
    display_name: 'Qwen 2.5 1.5B',
    family: 'qwen',
    parameters: 1_500_000_000,
    layers: 28,
    heads: 12,
    kv_heads: 2,
    hidden_dim: 1536,
    intermediate_dim: 8960,
    vocab_size: 151936,
    max_seq_len: 32768,
    gpu_tier: 't4',
  },
  {
    id: 'qwen-3b',
    hf_id: 'Qwen/Qwen2.5-3B',
    display_name: 'Qwen 2.5 3B',
    family: 'qwen',
    parameters: 3_000_000_000,
    layers: 36,
    heads: 16,
    kv_heads: 2,
    hidden_dim: 2048,
    intermediate_dim: 11008,
    vocab_size: 151936,
    max_seq_len: 32768,
    gpu_tier: 'a10g',
  },
]

/** Lookup map from open-source model id → ModelInfo for useModelInfo */
export const STATIC_OPEN_SOURCE_MODEL_INFO: Record<string, ModelInfo> = Object.fromEntries(
  STATIC_OPEN_SOURCE_MODELS.map(m => [
    m.id,
    {
      name: m.id,
      parameters: m.parameters,
      layers: m.layers,
      heads: m.heads,
      hidden_dim: m.hidden_dim,
      vocab_size: m.vocab_size,
    },
  ])
)
