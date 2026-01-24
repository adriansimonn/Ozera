/**
 * Type definitions for transformer models and configurations.
 */

/**
 * Model family types for grouping models
 */
export type ModelFamily = 'ozera' | 'gemma' | 'qwen' | 'smollm'

/**
 * Information about an open-source model
 */
export interface OpenSourceModelInfo {
  id: string
  hf_id: string
  display_name: string
  family: ModelFamily
  parameters: number
  layers: number
  heads: number
  kv_heads: number
  hidden_dim: number
  intermediate_dim: number
  vocab_size: number
  max_seq_len: number
  gpu_tier: 't4' | 'a10g'
}

/**
 * Cache status for an open-source model
 */
export interface ModelCacheStatus {
  status: 'ready' | 'not_cached' | 'incomplete' | 'error'
  hf_id: string
  path?: string
  has_model?: boolean
  has_tokenizer?: boolean
  total_size_mb?: number
  file_count?: number
  error?: string
}

/**
 * Response from model download request
 */
export interface ModelDownloadResponse {
  status: 'downloading' | 'cached' | 'error'
  hf_id: string
  path?: string
  error?: string
}

/**
 * Model family metadata for UI grouping
 */
export interface ModelFamilyInfo {
  name: string
  display_name: string
  models: {
    id: string
    display_name: string
    parameters: number
    gpu_tier: string
  }[]
}

export interface TransformerConfig {
  vocab_size: number
  d_model: number
  num_layers: number
  num_heads: number
  d_ff: number
  max_seq_len: number
  dropout_rate: number
  attention_dropout: number
  residual_dropout: number
  layer_norm_eps: number
  use_bias: boolean
  learned_pos_emb: boolean
  initializer_range: number
  activation: 'gelu' | 'relu'
}

export interface ModelInfo {
  id: string
  name: string
  config: TransformerConfig
  parameter_count: number
  created_at: string
  is_pretrained: boolean
  training_status?: 'training' | 'completed' | 'failed'
}

export interface InferenceRequest {
  model_id: string
  prompt: string
  max_tokens: number
  temperature: number
  top_p?: number
  top_k?: number
  repetition_penalty?: number
}

export interface InferenceResponse {
  generated_text: string
  tokens: string[]
  logits: number[][]
  attention_weights: number[][][][]  // [layer][head][seq_len][seq_len]
  hidden_states: number[][][]  // [layer][seq_len][d_model]
}

export interface TrainingConfig {
  dataset_id: string
  model_config: TransformerConfig
  batch_size: number
  learning_rate: number
  num_epochs: number
  warmup_steps: number
  gradient_clip: number
  checkpoint_interval: number
}

export interface TrainingMetrics {
  epoch: number
  step: number
  loss: number
  perplexity: number
  learning_rate: number
  timestamp: string
}

/**
 * Activation data types for visualization
 */

export interface TensorData {
  values: number[] | number[][] | number[][][] | number[][][][]
  shape: number[]
  dtype: string
  mean: number
  std: number
  min: number
  max: number
}

export interface LayerActivations {
  attn_input?: TensorData
  attn_output?: TensorData
  attn_weights?: TensorData
  post_attn?: TensorData
  ff_input?: TensorData
  ff_output?: TensorData
  post_ff?: TensorData
}

export interface ActivationData {
  id: string
  activations: {
    token_embeddings?: TensorData
    positional_embeddings?: TensorData
    combined_embeddings?: TensorData
    layers?: LayerActivations[]
    final_layer_norm?: TensorData
    logits?: TensorData
  }
  tokens: number[]
  prompt: string
  model: string
  timestamp: string
  metadata: {
    temperature?: number
    top_k?: number
    top_p?: number
    max_tokens?: number
    prompt_tokens?: number
    generated_tokens?: number
    total_tokens?: number
    decoded_tokens?: string[]
    [key: string]: any
  }
}

export interface ActivationSummary {
  id: string
  prompt: string
  model: string
  timestamp: string
  num_tokens: number
  num_layers: number
  metadata: Record<string, any>
}

/**
 * Tensor statistics for lazy loading (shape and stats only, no values)
 */
export interface TensorStats {
  shape: number[]
  mean?: number
  std?: number
  min?: number
  max?: number
}

/**
 * Extended activation summary with layer info for lazy loading
 */
export interface ActivationSummaryWithInfo extends ActivationSummary {
  layer_info: Record<string, TensorStats>[]
  tensor_info: Record<string, TensorStats>
}

/**
 * Response from lazy loading a single layer
 */
export interface LayerActivationResponse {
  layer_idx: number
  activations: LayerActivations
}

/**
 * Response from lazy loading a single tensor
 */
export interface TensorActivationResponse {
  tensor_name: string
  data: TensorData
}

export interface GenerateWithActivationsResponse {
  text: string
  activation_id: string
  prompt: string
  prompt_tokens: number
  generated_tokens: number
  total_tokens: number
  temperature: number
  top_k?: number
  top_p?: number
}
