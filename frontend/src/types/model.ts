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

/**
 * Pre-computed top-K logits from the backend.
 * Replaces the full logits tensor to avoid sending millions of floats.
 */
export interface TopKLogits {
  indices: number[][]      // [seq_len, k] - token IDs
  values: number[][]       // [seq_len, k] - logit values
  probabilities: number[][] // [seq_len, k] - softmax probabilities
  decoded_tokens: string[][] // [seq_len, k] - decoded token strings
  k: number
  seq_len: number
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
    top_k_logits?: TopKLogits
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

// ============= SAE Types (Phase 3.2) =============

/**
 * SAE activation type variants
 */
export type SAEActivationType = 'relu' | 'jumprelu' | 'topk'

/**
 * Activation source types for SAE training
 */
export type ActivationSource = 'residual' | 'mlp_output' | 'attn_output'

/**
 * SAE configuration
 */
export interface SAEConfig {
  d_input: number
  expansion_factor: number
  d_hidden: number
  activation: SAEActivationType
  sparsity_coefficient: number
  target_layer: number
  activation_source: ActivationSource
}

/**
 * SAE info for display
 */
export interface SAEInfo {
  id: string
  name: string
  model_id: string
  layer: number
  num_features: number
  activation_source: ActivationSource
  created_at: string
  config: SAEConfig
}

/**
 * Feature activation for a single feature
 */
export interface FeatureActivation {
  feature_idx: number
  activation_value: number
  rank: number
  percentile: number
}

/**
 * Token-level feature activations
 */
export interface TokenFeatureActivations {
  position: number
  token: string
  token_id: number
  top_features: FeatureActivation[]
  total_active_features: number
  l0_sparsity: number
  l1_norm: number
}

/**
 * Sequence-level feature activations
 */
export interface SequenceFeatureActivations {
  tokens: string[]
  token_ids: number[]
  per_token_activations: TokenFeatureActivations[]
  feature_activation_matrix: number[][]
  active_features_per_position: number[]
  most_active_features: number[]
}

/**
 * Statistics for a single feature
 */
export interface FeatureStats {
  feature_idx: number
  activation_frequency: number
  mean_activation: number
  max_activation: number
  polysemanticity: number
  suggested_label: string
  confidence: number
  unique_tokens: number
}

/**
 * Feature catalog containing all feature interpretations
 */
export interface FeatureCatalog {
  sae_id: string
  num_features: number
  features: FeatureStats[]
  dead_features: number[]
  polysemantic_features: number[]
  monosemantic_features: number[]
}

/**
 * Token activation example for interpretability
 */
export interface TokenActivationExample {
  token: string
  token_id: number
  activation_value: number
  position: number
  context: string[]
  prompt: string
}

/**
 * Feature interpretation data
 */
export interface FeatureInterpretation {
  feature_idx: number
  top_activating_tokens: TokenActivationExample[]
  token_frequency_distribution: Record<string, number>
  suggested_label: string
  confidence: number
  activation_statistics: {
    total_activations: number
    mean_activation: number
    max_activation: number
    unique_tokens: number
    polysemanticity: number
  }
}

/**
 * Sparsity metrics
 */
export interface SparsityMetrics {
  avg_l0: number
  l0_std: number
  sparsity_fraction: number
  avg_l1: number
  max_activation: number
}

/**
 * Feature health metrics
 */
export interface FeatureHealthMetrics {
  num_features: number
  dead_features: number
  dead_feature_fraction: number
  low_frequency_features: number
  high_frequency_features: number
  feature_frequency_distribution: number[]
  feature_magnitude_distribution: number[]
}

/**
 * Reconstruction metrics
 */
export interface ReconstructionMetrics {
  mse: number
  rmse: number
  normalized_mse: number
  explained_variance: number
  cosine_similarity: number
  relative_reconstruction_error: number
}

/**
 * Complete SAE quality metrics
 */
export interface SAEQualityMetrics {
  sparsity: SparsityMetrics
  feature_health: FeatureHealthMetrics
  reconstruction: ReconstructionMetrics
}

/**
 * SAE training progress
 */
export interface SAETrainingProgress {
  step: number
  total_steps: number
  loss: number
  reconstruction_loss: number
  sparsity_loss: number
  avg_l0: number
  dead_features: number
}

/**
 * Feature match for comparison
 */
export interface FeatureMatch {
  feature_a: number
  feature_b: number
  similarity: number
  shared_tokens: string[]
  label_a: string
  label_b: string
}

/**
 * SAE comparison metrics
 */
export interface SAEComparisonMetrics {
  overall_similarity: number
  matched_features: number
  unmatched_a: number
  unmatched_b: number
  top_matches: FeatureMatch[]
  divergent_features_a: number[]
  divergent_features_b: number[]
}
