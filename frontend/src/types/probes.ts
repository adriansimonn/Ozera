/**
 * Types for the Probe Lab API (backend/api/probes.py).
 *
 * Sweep results have one entry per residual stream position: position 0 is the embeddings,
 * position i the residual stream after layer i - 1. A saved probe's `layer` is the decoder
 * layer whose output it reads (position layer + 1).
 */

export type Pooling = 'last' | 'mean' | 'max'
export type ProbeMethod = 'logreg' | 'diff_means'
export type ReadSpan = 'text' | 'prompt'

export const POOLINGS: Pooling[] = ['last', 'mean', 'max']
export const PROBE_METHODS: ProbeMethod[] = ['logreg', 'diff_means']

export interface ProbeRow {
  text: string
  label: 0 | 1
  group?: string | number | null
}

export interface ProbeDatasetSummary {
  id: string
  name: string
  description: string
  category: 'contrastive' | 'minimal_pairs'
  label_names: [string, string]
  source: string
  num_rows: number
  num_positive: number
  total_chars: number
  ood: { description: string; num_rows: number; total_chars: number } | null
}

export interface ProbeDatasetDetail extends ProbeDatasetSummary {
  rows: ProbeRow[]
  ood_rows: ProbeRow[]
}

export interface ParsedProbeDataset {
  filename: string
  rows: ProbeRow[]
  label_names: [string, string]
  warnings: string[]
}

export interface ProbeModelInfo {
  model_id: string
  display_name: string
  model_type: 'ozera' | 'open_source' | 'custom'
  num_layers: number | null
  hidden_dim: number | null
  parameters: number | null
  is_instruct: boolean
  gpu_tier: string
  /** The model's base or instruct counterpart (same shapes), for transfer tests */
  sibling?: string | null
}

export interface ProbeEstimate {
  estimated_cost: number
  estimated_tokens: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
}

export interface ProbeDatasetSpec {
  builtin_id?: string
  rows?: ProbeRow[]
  name?: string
  label_names?: [string, string]
  ood_rows?: ProbeRow[]
  ood_name?: string
  use_builtin_ood?: boolean
}

export interface TrainProbesRequest {
  model: string
  dataset: ProbeDatasetSpec
  chat_template: boolean
  read_span: ReadSpan
  test_fraction: number
  seed: number
}

/** A tensor decoded from the wire format: nested arrays of the tensor's shape. */
export interface DecodedTensor<T> {
  values: T
  shape: number[]
}

export type MetricName =
  | 'test_auroc'
  | 'test_acc'
  | 'train_acc'
  | 'control_test_auroc'
  | 'control_test_acc'
  | 'control_train_acc'
  | 'selectivity'
  | 'score_mean'
  | 'score_std'
  | 'ood_auroc'
  | 'ood_acc'

/** Per-position values; null where a metric is undefined (e.g. a class missing from a set). */
export type PerPosition = (number | null)[]

export interface MethodResult {
  metrics: Partial<Record<MetricName, PerPosition>>
  test_scores: DecodedTensor<number[][]>
  ood_scores: DecodedTensor<number[][]> | null
  weights: DecodedTensor<number[][]>
  biases: PerPosition
  l2?: PerPosition
}

export interface PoolingResult {
  methods: Record<ProbeMethod, MethodResult>
  pca: DecodedTensor<number[][][]>
  pca_ood: DecodedTensor<number[][][]> | null
  pca_variance: DecodedTensor<number[][]>
  act_norms: PerPosition
}

export interface ProbeExample {
  text: string
  label: number
  split?: 'train' | 'test' | 'ood' | null
}

export interface ProbeRunResult {
  model: string
  num_layers: number
  hidden_dim: number
  chat_template: boolean
  read_span: ReadSpan
  seed: number
  dataset: {
    name: string
    builtin_id: string | null
    label_names: [string, string]
    n_train: number
    n_test: number
    n_ood: number
    ood_name: string | null
  }
  total_tokens: number
  cost: number
  majority: { test_acc: number | null; ood_acc: number | null }
  poolings: Record<Pooling, PoolingResult>
  test_examples: ProbeExample[]
  ood_examples: ProbeExample[]
  pca_examples: ProbeExample[]
  pca_ood_examples: ProbeExample[]
}

export interface InlineProbe {
  layer: number
  pooling: Pooling
  weights: number[]
  bias: number
  chat_template: boolean
  read_span: ReadSpan
  method?: ProbeMethod
  /** How far apart the class means are along the weights' direction (steering's unit) */
  class_gap?: number | null
}

export interface ScoreProbeRequest {
  model?: string
  texts: string[]
  probe_id?: number
  probe?: InlineProbe
}

export interface TextScores {
  text: string
  tokens: string[]
  scores: number[]
  /** [start, end) of the positions the probe pools over */
  span: [number, number]
  score: number
}

export interface ScoreProbeResponse {
  model: string
  layer: number
  pooling: Pooling
  results: TextScores[]
  cost: number
}

export interface SaveProbeRequest {
  name: string
  model: string
  layer: number
  pooling: Pooling
  method: ProbeMethod
  weights: number[]
  bias: number
  chat_template: boolean
  read_span: ReadSpan
  normalization: Record<string, unknown>
  metrics: Record<string, unknown>
  dataset: Record<string, unknown>
}

export interface SavedProbe {
  id: number
  name: string
  model_id: string
  layer: number
  pooling: Pooling
  method: ProbeMethod
  chat_template: boolean
  read_span: ReadSpan
  hidden_dim: number
  bias: number
  normalization: {
    score_mean?: number | null
    score_std?: number | null
    act_norm?: number | null
    class_gap?: number | null
  }
  metrics: Record<string, unknown>
  dataset: Record<string, unknown>
  created_at: string
  model_available: boolean
  model_changed: boolean
  weights?: number[] | null
}

// Causal validation: steering and directional ablation

export interface SteerProbeRequest {
  model?: string
  probe_id?: number
  probe?: InlineProbe
  prompt: string
  /** Multiples of the gap between the class means along the direction */
  alphas: number[]
  ablate: boolean
  generated_only: boolean
  max_tokens: number
  temperature: number
  seed: number
}

export interface SteeredGeneration {
  kind: 'baseline' | 'steer' | 'ablate'
  /** 0 for the baseline, null for the ablated generation */
  alpha: number | null
  text: string
  generated_tokens: number
  /** The text's perplexity under the unsteered model */
  perplexity: number | null
  /** The probe's pooled score of the text, read by the unsteered model */
  probe_score: number | null
  scored: TextScores | null
  /** First generated token that differs from the baseline's (null: identical) */
  first_divergence: number | null
  tokens_changed: number
}

export interface SteerProbeResponse {
  model: string
  layer: number
  pooling: Pooling
  method: ProbeMethod
  prompt: string
  prompt_tokens: number
  /** Norm of the vector added at α = 1 */
  unit_norm: number
  generated_only: boolean
  max_tokens: number
  temperature: number
  generations: SteeredGeneration[]
  cost: number
}

export interface SteeringEstimate {
  estimated_cost: number
  generations: number
}

export interface AblateProbeRequest {
  model?: string
  probe_id?: number
  probe?: InlineProbe
  dataset: ProbeDatasetSpec
  test_fraction: number
  seed: number
}

export type AblationCurve = 'clean' | 'probe_frozen' | 'probe_retrained' | 'control_frozen' | 'control_retrained'

export interface AblationBehaviour {
  /** Mean KL divergence (nats per token) of next-token predictions from the clean model's */
  kl: number
  top1_agreement: number
  /** Share of the residual stream's mean squared norm along the direction at the probe's layer */
  energy_fraction: number
  /** Controls: [lowest, highest] KL of the control directions */
  kl_range?: [number, number] | null
}

export interface AblateProbeResponse {
  model: string
  num_layers: number
  hidden_dim: number
  layer: number
  pooling: Pooling
  method: ProbeMethod
  chat_template: boolean
  read_span: ReadSpan
  seed: number
  dataset: ProbeRunResult['dataset']
  total_tokens: number
  compared_tokens: number
  /** Test AUROC per residual stream position (num_layers + 1 values); control curves average the controls */
  curves: Record<AblationCurve, PerPosition>
  behaviour: Record<'probe' | 'control', AblationBehaviour>
  cost: number
}

/** A monitoring probe's scores of a generation's tokens, by position (streamed as it generates). */
export interface MonitorTrace {
  tokens: string[]
  /** null where a position hasn't been scored */
  scores: (number | null)[]
  promptTokens: number
}

/** What the Generate page shows about the probe monitoring a generation. */
export interface MonitorProbeInfo {
  id: number
  name: string
  layer: number
  labelNames: [string, string]
  /** Scores of ±scale fill the color scale (2 standard deviations of training scores, if known) */
  scale: number | null
}

// Generalization matrices

/** One dataset of a generalization run: a built-in set (its main rows or its OOD set), or uploaded rows. */
export interface MatrixDatasetSpec {
  builtin_id?: string
  part?: 'main' | 'ood'
  rows?: ProbeRow[]
  name?: string
  label_names?: [string, string]
}

export interface GeneralizeRequest {
  model: string
  transfer_model?: string | null
  datasets: MatrixDatasetSpec[]
  pooling: Pooling
  chat_template: boolean
  read_span: ReadSpan
  test_fraction: number
  seed: number
}

export interface MatrixDatasetInfo {
  name: string
  description: string | null
  builtin_id: string | null
  part: 'main' | 'ood'
  label_names: [string, string]
  n_train: number
  n_test: number
  n_test_positive: number
}

/** [train condition][test condition][value per residual stream position] */
export type ConditionMatrix = (number | null)[][][]

/**
 * A generalization matrix. Condition i is models[floor(i / datasets.length)] reading
 * datasets[i % datasets.length]; probes train on a condition's training split and are tested on
 * every condition's test split.
 */
export interface GeneralizeResponse {
  model: string
  transfer_model: string | null
  models: string[]
  pooling: Pooling
  chat_template: boolean
  read_span: ReadSpan
  seed: number
  num_layers: number
  hidden_dim: number
  datasets: MatrixDatasetInfo[]
  auroc: Record<ProbeMethod, ConditionMatrix>
  acc: Record<ProbeMethod, ConditionMatrix>
  /** [train dataset][test dataset]: test examples whose text is also in the training split */
  overlap: number[][]
  total_tokens: number
  cost: number
}

// SAE features

export type SaeRef = { kind: 'ozera'; model: 'nano' | 'mini'; layer: number } | { kind: 'external'; sae_id: string }

/** An SAE that reads a model's residual stream after one layer. */
export interface ProbeSaeInfo {
  ref: SaeRef
  name: string
  source: 'ozera' | 'gemma_scope' | 'huggingface'
  source_id?: string | null
  layer: number
  width: number | null
  /** The model it was trained on */
  trained_on: string
  /** exact: trained on this model; sibling: on its base or instruct counterpart */
  match: 'exact' | 'sibling'
  unusable_reason?: string | null
}

export interface SaeProbeRequest {
  model?: string
  probe_id?: number
  probe?: InlineProbe
  sae: SaeRef
  dataset: ProbeDatasetSpec
  test_fraction: number
  seed: number
}

export interface SaeFeature {
  feature: number
  /** Cosine between the feature's decoder direction and the probe's direction */
  cos: number
  /** (probe weights · decoder row) × class mean difference: its part of the probe's class gap */
  contribution: number
  /** Mean pooled activation over the positive / negative training examples */
  mean_pos: number
  mean_neg: number
  /** Share of positive / negative training examples it's active on */
  freq_pos: number
  freq_neg: number
  /** The example it's most active on (index into the response's examples) */
  top_example: number | null
  top_activation: number | null
}

export interface SaeProbeResponse {
  model: string
  layer: number
  pooling: Pooling
  method: ProbeMethod
  chat_template: boolean
  read_span: ReadSpan
  seed: number
  num_layers: number
  hidden_dim: number
  sae: {
    info: ProbeSaeInfo
    tokens: number
    /** Active features per token */
    l0: number
    /** Fraction of these activations' variance the SAE's reconstructions miss */
    fvu: number | null
  }
  dataset: ProbeRunResult['dataset']
  alignment: {
    /** Nearest the probe's direction by |cosine| */
    features: SaeFeature[]
    /** Share of the direction (squared norm) in the span of its k nearest features */
    captured: { k: number; fraction: number }[]
    /** A random direction's nearest feature's |cosine| (mean) */
    random_max_cos: number
    /** The same for shuffled-label difference-in-means directions */
    control_max_cos: number[]
    contributions: {
      features: SaeFeature[]
      /** The probe's mean score on positive minus negative training examples */
      gap: number
      /** Every feature's contribution summed (the gap on the SAE's reconstructions) */
      reconstructed: number
      cumulative: { k: number; share: number | null }[]
    }
  }
  /** Sparse probes on the k features whose class means differ most; null if none differ */
  sparse: {
    ks: number[]
    metrics: Partial<Record<'test_auroc' | 'test_acc' | 'train_acc' | 'ood_auroc' | 'ood_acc', (number | null)[]>>
    l2: number[]
    features: SaeFeature[]
  } | null
  /** A dense logistic regression probe on the same split */
  dense: Partial<Record<'test_auroc' | 'test_acc' | 'train_acc' | 'ood_auroc' | 'ood_acc', number | null>>
  examples: ProbeExample[]
  total_tokens: number
  cost: number
}
