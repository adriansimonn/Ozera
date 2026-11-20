/**
 * Types of the Attribution page (backend/api/schemas/attribution.py).
 */

export type MetricName = 'logit_diff' | 'kl'
export type CellKind = 'resid' | 'attn' | 'mlp' | 'head'
export type ComponentKind = 'head' | 'attn' | 'mlp'
export type AblationMethod = 'zero' | 'mean' | 'resample'
/** The kinds a sweep reports by position, in the order of its position arrays */
export type PositionKind = 'resid' | 'attn' | 'mlp'

export const POSITION_KINDS: PositionKind[] = ['resid', 'attn', 'mlp']
export const ABLATION_METHODS: AblationMethod[] = ['zero', 'mean', 'resample']

export interface PairRow {
  clean: string
  corrupted: string
  correct?: string | null
  incorrect?: string | null
  group?: string | null
}

export interface PairSpec {
  task?: string | null
  n?: number
  seed?: number
  rows?: PairRow[] | null
  name?: string | null
}

export interface AttributionTask {
  id: string
  name: string
  description: string
  kind: 'text' | 'tokens'
  default_pairs: number
  max_pairs: number
  examples: PairRow[]
}

export interface ParsedPairs {
  filename: string
  rows: PairRow[]
  has_answers: boolean
  warnings: string[]
}

export interface AttributionModelInfo {
  model_id: string
  display_name: string
  model_type: 'ozera' | 'open_source' | 'custom'
  num_layers: number | null
  num_heads: number | null
  hidden_dim: number | null
  parameters: number | null
  is_instruct: boolean
  gpu_tier: string
}

export interface DroppedPair {
  index: number
  reason: string
}

export interface AttributionEstimate {
  estimated_cost: number
  estimated_tokens: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  num_pairs: number
  /** Ozera models' pairs are checked before the run; open-weight models' when it starts */
  checked: boolean
  dropped: DroppedPair[]
}

export interface PairResult {
  clean_tokens: string[]
  corrupted_tokens: string[]
  correct: string | null
  incorrect: string | null
  group: string | null
  cls: number
  source: number
  clean_value: number
  corrupted_value: number
}

/** Pairs of one template and length: their tokens line up, so effects by position can be averaged over them. */
export interface AlignmentClass {
  id: number
  group: string | null
  tokens: number
  pairs: number[]
  /** The token every clean prompt has at each position, or null where they differ */
  labels: (string | null)[]
}

/**
 * An attribution patching sweep. Effects are on the scale where the corrupted run is 0 and the
 * clean run 1 when `normalized`, else in the metric's units.
 */
export interface AttributionRun {
  run_id: string
  model: string
  task: string | null
  task_name: string
  seed: number
  metric: MetricName
  normalized: boolean
  gap: number
  clean_mean: number
  corrupted_mean: number
  /** The share of pairs whose clean metric beats their corrupted one */
  solved: number
  num_layers: number
  num_heads: number
  hidden_dim: number
  num_pairs: number
  total_tokens: number
  pairs_key: string
  pairs: PairResult[]
  classes: AlignmentClass[]
  dropped: DroppedPair[]
  /** [layers][heads]: every head's estimated effect, mean and standard error over pairs */
  heads: number[][]
  heads_se: number[][]
  /** [pairs][layers][heads] */
  heads_pairs: number[][][]
  /** [attention, MLP][layers]: a layer's whole output */
  layers: number[][]
  layers_se: number[][]
  /** Per pair, [resid, attn, mlp][layers][tokens] */
  position_pairs: number[][][][]
  /** Per alignment class, the mean over its pairs */
  position_classes: number[][][][]
  /** [layers][heads]: heads' direct effects on the clean logit difference, in logits */
  dla: number[][] | null
  attention: { pairs: number[]; patterns: string[] } | null
  cost: number
}

export interface Cell {
  kind: CellKind
  layer: number
  head?: number | null
  /** null: every position */
  position?: number | null
  pair?: number | null
  cls?: number | null
  estimate?: number | null
}

export interface VerifiedCell extends Required<Cell> {
  effect: number
  se: number
  n: number
}

export interface VerifyResponse {
  model: string
  metric: MetricName
  normalized: boolean
  gap: number
  clean_mean: number
  corrupted_mean: number
  num_layers: number
  num_heads: number
  num_pairs: number
  pairs_key: string
  cells: VerifiedCell[]
  rank_correlation: number | null
  cost: number
}

export interface Component {
  kind: ComponentKind
  layer: number
  head?: number | null
}

export interface AblateResponse {
  model: string
  metric: MetricName
  normalized: boolean
  gap: number
  clean_mean: number
  corrupted_mean: number
  num_layers: number
  num_heads: number
  num_pairs: number
  pairs_key: string
  components: Component[]
  random_sets: Component[][]
  mean_over: 'position' | 'tokens' | null
  methods: Partial<Record<AblationMethod, { value: number; se: number; random: number[] }>>
  seed: number
  cost: number
}

/** A chart's data, for a figure drawn by the backend (POST /export/figure). */
export interface DataFigureRequest {
  kind: 'heatmap' | 'scatter' | 'bars' | 'graph' | 'lines'
  config?: import('./export').ExportConfig
  title?: string
  x_label?: string
  y_label?: string
  value_label?: string
  values?: number[][]
  row_labels?: string[]
  column_labels?: string[]
  x?: number[]
  y?: number[]
  errors?: number[]
  note?: string
  bars?: { label: string; value: number; error?: number | null; marks?: number[] }[]
  baseline?: number
  baseline_label?: string
  /** lines: each line over the points column_labels name; a faint line is a part of the last solid one, a dashed one a reference */
  series?: { label: string; values: (number | null)[]; style?: 'solid' | 'faint' | 'dashed' }[]
  /** graph: an attribution graph's nodes by layer and position, and its edges as [source, target, weight, share] */
  graph?: {
    tokens: string[]
    num_layers: number
    target: string
    nodes: { kind: string; layer: number | null; position: number; influence: number; support?: number; pinned?: boolean }[]
    edges: number[][]
  }
}
