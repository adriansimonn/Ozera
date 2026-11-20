/**
 * Circuit Tracing: attribution graphs of one prompt and one target token, built from a model's
 * transcoders (backend/core/circuits), and interventions on their feature nodes.
 */

export interface TranscoderSetInfo {
  id: string
  name: string
  /** The set on HuggingFace, as circuit-tracer names it */
  reference: string
  /** Features per layer */
  width: number
  l0: string
  affine: boolean
  source: string
}

export interface CircuitModelInfo {
  model_id: string
  display_name: string
  num_layers: number
  hidden_dim: number
  parameters: number
  is_instruct: boolean
  gpu_tier: string
  /** The model's transcoder sets, the default first */
  transcoders: TranscoderSetInfo[]
}

export interface TraceRequest {
  model: string
  transcoders?: string | null
  prompt: string
  /** The next word to trace (its first token); the model's top prediction when empty */
  target?: string | null
}

export interface TraceEstimate {
  estimated_cost: number
  estimated_seconds: number
  estimated_tokens: number
  max_seconds: number
  within_limit: boolean
}

/** The model's reading of a token at the prompt's last position */
export interface TokenReading {
  /** The token's logit above the mean logit (what a graph attributes) */
  logit: number
  prob: number
  /** 0 for the top prediction */
  rank: number
}

export interface TargetReading extends TokenReading {
  token: string
  token_id: number
}

/** A completed job's graph in brief */
export interface CircuitSummary {
  target: TargetReading
  tokens: number
  replacement: number
  completeness: number
  /** Error nodes' share of the target's direct inputs (absolute) */
  error_inputs: number
  feature_nodes: number
  error_nodes: number
  edges: number
  active_features: number
  traced_features: number
  seconds: number
}

export type CircuitJobStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface CircuitJob {
  job_id: string
  model: string
  transcoders: string
  transcoders_name: string
  prompt: string
  /** The target as asked for (null: the top prediction) */
  target: string | null
  status: CircuitJobStatus
  progress: { stage: string; fraction: number } | null
  cost: number
  refunded: number | null
  error: string | null
  estimated_seconds: number
  summary: CircuitSummary | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export type NodeKind = 'feature' | 'error' | 'token' | 'logit'

export interface GraphNode {
  kind: NodeKind
  /** f:<layer>:<position>:<feature>, e:<layer>:<position>, t:<position>, l:<token ID> */
  id: string
  /** The share of the target's attribution that flows through the node */
  influence: number
  /** The same with the edges' signs kept: positive pushes the target up, negative holds it down */
  support: number
  position: number
  /** Feature and error nodes */
  layer?: number
  /** Feature nodes */
  feature?: number
  activation?: number
  /** Tokens the feature's decoder writes toward and away from */
  promotes?: string[]
  suppresses?: string[]
  /** The feature's activation at every position of the prompt */
  fires?: number[]
  /** Token and logit nodes */
  token?: string
  token_id?: number
  prob?: number
}

/** Edges by column; source and target are indices into the graph's nodes */
export interface GraphEdges {
  source: number[]
  target: number[]
  /** The source's direct effect on the target */
  weight: number[]
  /** The share of the target token's attribution that flows through the edge */
  share: number[]
}

export interface CircuitScores {
  /** Of the influence reaching the target from token and error nodes, the tokens' share */
  replacement: number
  /** Over all nodes, weighted by influence, the share of inputs that isn't from error nodes */
  completeness: number
  /** The target logit's direct inputs by kind */
  target_inputs: {
    sum: number
    absolute: Record<'feature' | 'error' | 'token', number>
    signed: Record<'feature' | 'error' | 'token', number>
  }
}

export interface CircuitGraph {
  version: number
  job_id: string
  model: string
  transcoders: { id: string; name: string; reference: string; width: number }
  prompt: string
  chat_template: boolean
  tokens: string[]
  token_ids: number[]
  num_layers: number
  target: TargetReading
  target_given: boolean
  predictions: { token: string; token_id: number; prob: number }[]
  scores: CircuitScores
  counts: {
    active_features: number
    traced_features: number
    /** What pruning kept, before the stored graph's caps */
    feature_nodes: number
    error_nodes: number
    edges: number
    kept_edges: number
  }
  nodes: GraphNode[]
  edges: GraphEdges
  seconds: number
  gpu_peak_gib: number | null
}

export interface InterventionReading extends TokenReading {
  /** The logit minus the unchanged model's */
  change?: number | null
  nodes?: number | null
  /** The mean activation of the nodes changed */
  activation?: number | null
}

export type ControlKind = 'activation' | 'site'
export const CONTROL_KINDS: ControlKind[] = ['activation', 'site']

export interface InterveneRequest {
  job_id: string
  nodes: string[]
  /** The factor the nodes' activations are multiplied by (0 ablates) */
  scale: number
  control_sets?: number
  seed?: number
}

export interface InterveneEstimate {
  estimated_cost: number
  estimated_seconds: number
}

export interface InterveneResponse {
  job_id: string
  model: string
  target: { token: string; token_id: number }
  scale: number
  base: InterventionReading
  chosen: InterventionReading
  controls: Record<ControlKind, InterventionReading[]>
  predictions: Record<'base' | 'chosen', { token: string; token_id: number; prob: number }[]>
  nodes: { id: string; layer: number; position: number; feature: number; activation: number }[]
  seed: number
  seconds: number
  cost: number
}
