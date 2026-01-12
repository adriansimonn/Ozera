/**
 * Type definitions for visualization components.
 */

export interface AttentionVisualizationData {
  layer: number
  head: number
  tokens: string[]
  attention_matrix: number[][]  // [query_pos][key_pos]
}

export interface ActivationData {
  layer: number
  position: number
  activations: number[]
  token: string
}

export interface EmbeddingVisualization {
  tokens: string[]
  embeddings: number[][]  // [token_idx][embedding_dim]
  reduced_embeddings?: number[][]  // For 2D/3D visualization
}

export interface ProbabilityDistribution {
  tokens: string[]
  probabilities: number[]
  top_k: number
}

export interface LayerActivationStats {
  layer: number
  mean: number
  std: number
  min: number
  max: number
  sparsity: number
}

export interface HeadBehavior {
  layer: number
  head: number
  classification: 'induction' | 'copying' | 'positional' | 'previous_token' | 'unknown'
  confidence: number
  patterns: string[]
}

export interface ActivationPatch {
  source_prompt: string
  target_prompt: string
  layer: number
  position: number
  patch_type: 'attention' | 'mlp' | 'residual'
}

export interface ComparisonMetrics {
  model_a_id: string
  model_b_id: string
  layer_similarities: number[]
  feature_alignment: number
  semantic_similarity: number
}
