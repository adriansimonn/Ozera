/**
 * Type definitions for attention pattern analysis.
 */

/**
 * Types of attention heads that can be detected.
 */
export type HeadType = 'induction' | 'previous_token' | 'positional' | 'copying' | 'mixed' | 'unknown'

/**
 * Classification result for a single attention head.
 */
export interface HeadClassification {
  layer: number
  head: number
  primary_type: HeadType
  confidence: number
  scores: Record<string, number>
  pattern_summary: string
}

/**
 * Request to classify attention heads.
 */
export interface ClassifyHeadsRequest {
  activation_id: string
}

/**
 * Response containing classification results for all heads.
 */
export interface ClassifyHeadsResponse {
  model_id: string
  activation_id: string
  prompt: string
  tokens: string[]
  num_layers: number
  num_heads: number
  classifications: HeadClassification[]
  summary: Record<HeadType, number>
}

/**
 * Request to compare attention patterns.
 */
export interface CompareAttentionRequest {
  activation_id_1: string
  activation_id_2: string
}

/**
 * Similarity metrics for a single layer.
 */
export interface LayerSimilarity {
  layer: number
  mean_similarity: number
  min_similarity: number
  max_similarity: number
}

/**
 * Information about a significantly different head.
 */
export interface HeadDifference {
  layer: number
  head: number
  similarity: number
  difference: number
}

/**
 * Response containing attention comparison results.
 */
export interface CompareAttentionResponse {
  prompt1: string
  prompt2: string
  activation_id1: string
  activation_id2: string
  layer_similarities: LayerSimilarity[]
  head_differences: HeadDifference[]
  common_patterns: string[]
  divergent_patterns: string[]
}

/**
 * Request to run pattern mining.
 */
export interface MinePatternRequest {
  activation_id: string
}

/**
 * Importance metrics for a single head.
 */
export interface HeadImportance {
  layer: number
  head: number
  entropy_score: number
  max_attention_score: number
  variance_score: number
  overall_importance: number
}

/**
 * Cross-layer pattern information.
 */
export interface CrossLayerPattern {
  pattern_type: string
  description: string
  involved_heads: [number, number][]
  confidence: number
  evidence: Record<string, number | string>
}

/**
 * Potential circuit candidate.
 */
export interface CircuitCandidate {
  type: string
  description: string
  components: Record<string, [number, number][]>
  confidence: number
}

/**
 * Response containing pattern mining results.
 */
export interface MinePatternResponse {
  model_id: string
  activation_ids: string[]
  head_importance: HeadImportance[]
  cross_layer_patterns: CrossLayerPattern[]
  classification_summary: Record<HeadType, number>
  top_heads: [number, number, number][]
  circuit_candidates: CircuitCandidate[]
}

/**
 * Request to get head importance scores.
 */
export interface HeadImportanceRequest {
  activation_id: string
}

/**
 * Response containing head importance scores.
 */
export interface HeadImportanceResponse {
  activation_id: string
  model_id: string
  importance: Record<string, number>
}

/**
 * Color mappings for head types.
 */
export const HEAD_TYPE_COLORS: Record<HeadType, string> = {
  induction: '#10B981',       // green
  previous_token: '#3B82F6',  // blue
  positional: '#8B5CF6',      // purple
  copying: '#F59E0B',         // amber
  mixed: '#EC4899',           // pink
  unknown: '#6B7280',         // gray
}

/**
 * Display names for head types.
 */
export const HEAD_TYPE_NAMES: Record<HeadType, string> = {
  induction: 'Induction',
  previous_token: 'Previous Token',
  positional: 'Positional',
  copying: 'Copying',
  mixed: 'Mixed',
  unknown: 'Unknown',
}
