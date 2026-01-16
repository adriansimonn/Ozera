/**
 * Type definitions for transformer models and configurations.
 */

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
