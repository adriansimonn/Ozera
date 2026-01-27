/**
 * Type definitions for activation patching API.
 */

/**
 * Types of activation patches supported.
 */
export type PatchType = 'attention' | 'mlp' | 'residual' | 'attn_output' | 'ff_output' | 'post_attn' | 'post_ff'

/**
 * Types of interventions supported.
 * - 'patch': Replace with source activations (requires source prompt)
 * - 'zero_ablate': Zero out activations
 * - 'mean_ablate': Replace with mean activation
 * - 'noise_ablate': Replace with Gaussian noise matching activation statistics
 */
export type InterventionType = 'patch' | 'zero_ablate' | 'mean_ablate' | 'noise_ablate'

/**
 * Specification for a single activation patch.
 */
export interface PatchSpec {
  layer: number
  patch_type: PatchType
  positions?: number[] | null
  heads?: number[] | null
  neurons?: number[] | null
  blend_factor: number
  intervention_type: InterventionType
}

/**
 * Request to capture source activations.
 */
export interface CaptureActivationsRequest {
  prompt: string
  model: string
}

/**
 * Response from capturing activations.
 */
export interface CaptureActivationsResponse {
  activation_id: string
  prompt: string
  model_type: 'ozera' | 'open_source'
  model_id: string
  num_tokens: number
  num_layers: number
  decoded_tokens: string[]
}

/**
 * Summary of captured activations.
 */
export interface CapturedActivationSummary {
  id: string
  prompt: string
  model_type: 'ozera' | 'open_source'
  model_id: string
  num_tokens: number
  num_layers: number
}

/**
 * Detailed captured activation info including tokens.
 */
export interface CapturedActivationDetail extends CapturedActivationSummary {
  tokens: number[]
  decoded_tokens: string[]
  available_keys: string[]
}

/**
 * Request to run a patching experiment.
 * source_prompt is optional for ablation-only experiments.
 */
export interface RunPatchingRequest {
  source_prompt?: string | null
  target_prompt: string
  model: string
  patches: PatchSpec[]
  max_tokens?: number
  temperature?: number
}

/**
 * Request to run patching with pre-captured activations.
 */
export interface RunPatchingWithCapturedRequest {
  source_activation_id: string
  target_prompt: string
  model: string
  patches: PatchSpec[]
  max_tokens?: number
  temperature?: number
}

/**
 * Information about a changed token.
 */
export interface ChangedToken {
  position: number
  baseline_token: string
  patched_token: string
}

/**
 * Summary of patching effects.
 */
export interface EffectSummary {
  first_divergence_position: number | null
  token_changes: number
  changed_tokens: ChangedToken[]
  baseline_length: number
  patched_length: number
}

/**
 * Result of a patching experiment.
 * source_activation_id is null for ablation-only experiments.
 */
export interface PatchingResult {
  baseline_output: string
  patched_output: string
  baseline_tokens: number[]
  patched_tokens: number[]
  baseline_decoded: string[]
  patched_decoded: string[]
  source_activation_id: string | null
  patches_applied: PatchSpec[]
  effect_summary: EffectSummary
}

/**
 * Model info for patching.
 */
export interface PatchingModelInfo {
  model_id: string
  model_type: 'ozera' | 'open_source' | 'custom'
  display_name: string
  num_layers: number
  num_heads: number
}

/**
 * Layer info for patching configuration.
 */
export interface LayerPatchInfo {
  index: number
  patch_types: PatchType[]
  num_heads: number
}

/**
 * Model layer info response.
 */
export interface ModelLayerInfo {
  model_id: string
  model_type: 'ozera' | 'open_source'
  num_layers: number
  num_heads: number
  layers: LayerPatchInfo[]
  patch_types: PatchType[]
}

/**
 * Saved patching experiment.
 */
export interface PatchingExperiment {
  id: string
  name: string
  source_prompt?: string | null
  target_prompt: string
  model_id: string
  model_type: 'ozera' | 'open_source'
  patches: PatchSpec[]
  baseline_output?: string
  patched_output?: string
  baseline_decoded?: string[]
  patched_decoded?: string[]
  effect_summary?: EffectSummary
  created_at: string
  updated_at?: string
}

/**
 * Request to save a patching experiment.
 */
export interface SaveExperimentRequest {
  name: string
  source_prompt?: string | null
  target_prompt: string
  model_id: string
  patches: PatchSpec[]
  result?: PatchingResult
}

/**
 * Summary of a saved experiment for listing.
 */
export interface ExperimentSummary {
  id: string
  name: string
  model_id: string
  source_prompt_preview: string
  target_prompt_preview: string
  patch_count: number
  has_result: boolean
  created_at: string
}
