/**
 * Types for the Geometry Lab API (backend/api/geometry.py).
 *
 * Per-position lists have num_layers + 1 entries: position 0 is the embeddings, position i the
 * residual stream after layer i - 1. Examples are templates × values, template by template:
 * example i is template floor(i / k) with value i % k.
 */

import type { ProbeSaeInfo, SaeRef } from './probes'

export type ConceptStructure = 'cyclic' | 'ordinal' | 'categorical'
export type GeometryRead = 'value_last' | 'value_mean' | 'template_last'

export interface HypothesisInfo {
  id: string
  label: string
}

export interface ConceptSummary {
  id: string
  name: string
  description: string
  structure: ConceptStructure
  values: string[]
  groups: string[] | null
  /** Cyclic: angles in degrees; ordinal: numeric values (null: evenly spaced) */
  positions: number[] | null
  num_templates: number
  total_chars: number
  hypotheses: HypothesisInfo[]
  source: string
}

export interface ConceptDetail extends ConceptSummary {
  templates: string[]
  eval_templates: string[]
  helix_periods: number[]
}

export interface CustomConcept {
  name: string
  structure: ConceptStructure
  values: string[]
  templates: string[]
  /** Templates ending where the model would say a value, for steering */
  eval_templates?: string[]
  groups?: string[] | null
  positions?: number[] | null
  helix_periods?: number[]
}

export interface ConceptSpec {
  builtin_id?: string
  custom?: CustomConcept
}

export interface GeometryEstimate {
  estimated_cost: number
  estimated_tokens: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  num_examples: number
}

export interface GeometryRunRequest {
  model: string
  concept: ConceptSpec
  read: GeometryRead
  chat_template: boolean
  seed: number
}

/** A decoded tensor (see api/client decodeTensorData). */
export interface DecodedTensor<T> {
  values: T
  shape: number[]
}

export type PerPosition = (number | null)[]

export interface ShapeHypothesis {
  id: string
  label: string
  /** Dimensions of the hypothesis's ideal layout */
  dims: number
  /** [per subspace size][per position] held-out R² */
  r2: PerPosition[]
  shuffled_mean: PerPosition[]
  shuffled_max: PerPosition[]
}

export interface ShapeTests {
  /** Subspace sizes m: fits read the top m class-mean components */
  dims: number[]
  folds: number
  shuffles: number
  hypotheses: ShapeHypothesis[]
}

export interface RunConcept {
  builtin_id: string | null
  name: string
  structure: ConceptStructure
  values: string[]
  templates: string[]
  eval_templates: string[]
  positions: number[] | null
  groups: string[] | null
  helix_periods: number[]
}

export interface GeometryRun {
  model: string
  num_layers: number
  hidden_dim: number
  read: GeometryRead
  chat_template: boolean
  seed: number
  concept: RunConcept
  value_tokens: { tokens: string[]; counts: number[] }[]
  /** [positions, k, r] class means on all their components, RMS radius 1 per position */
  means: DecodedTensor<number[][][]>
  /** [positions, n, 3] every example on the top three components */
  points: DecodedTensor<number[][][]>
  /** [positions, ≤10] explained variance ratios of the class means' components */
  explained: DecodedTensor<number[][]>
  participation_ratio: PerPosition
  components_90: PerPosition
  scale: PerPosition
  between_fraction: PerPosition
  neighbors: PerPosition | null
  neighbor_chance: number | null
  shapes: ShapeTests | null
  total_tokens: number
  cost: number
}

export interface WireTensor {
  encoding: 'base64_float32'
  values: string
  shape: number[]
}

export interface CompareRequest {
  a: { values: string[]; coords: WireTensor }
  b: { values: string[]; coords: WireTensor }
  dims: number | null
  position?: number | null
  seed?: number
}

export interface CompareResult {
  /** Per position of b */
  r2: PerPosition
  cka: PerPosition
  shuffled_r2: { mean: PerPosition; max: PerPosition }
  shuffled_cka: { mean: PerPosition; max: PerPosition }
  shuffles: number
  dims: number
  position: number
  /** [k, 3] a aligned into b's frame, and b, on b's top three components */
  overlay: { a: number[][]; b: number[][] }
}

// Steer

export type SteerMethodId = 'manifold' | 'linear'

export interface SteerRequest {
  model: string
  concept: ConceptSpec
  read: GeometryRead
  chat_template: boolean
  /** Residual stream position: 0 the embeddings, i layer i - 1's output */
  position: number
  /** Subspace size: the class means' top components */
  dims: number
  source: number
  target: number
  steps: number
  eval_template: string
}

export interface SteerEstimate {
  estimated_cost: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  eval_passes: number
}

export interface TopToken {
  token: string
  prob: number
}

export interface SteerMethodResult {
  /** [steps + 1][readable] probability of each readable value's token */
  probs: (number | null)[][]
  mass: PerPosition
  /** Hellinger distance to the nearest unsteered value's distribution */
  hellinger: PerPosition
  /** That value's index */
  nearest: number[]
  /** Hellinger distance to the target value's unsteered distribution */
  to_target: PerPosition
  top: TopToken[][]
  /** ‖edit‖ / ‖activation‖ */
  edit_size: PerPosition
}

export interface SteerResult {
  model: string
  num_layers: number
  read: GeometryRead
  chat_template: boolean
  position: number
  dims: number
  source: number
  target: number
  steps: number
  eval_template: string
  values: string[]
  /** The eval prompt with the source value */
  prompt: string
  edited_tokens: string[]
  /** Tokens every value starts with, appended to the prompt before reading */
  prefix: string[]
  /** Values read (their first tokens are distinct) */
  readable: number[]
  /** Each readable value's token */
  tokens: string[]
  /** Value index at each step */
  path: number[]
  /** Share of the class means' variance in the subspace */
  explained: number | null
  /** Share of values whose eval prompt's activation is nearest its own class mean in the subspace */
  eval_decoding: number | null
  /** The eval prompt filled with each value, unedited */
  reference: {
    probs: (number | null)[][]
    mass: PerPosition
    top: TopToken[][]
    /** Hellinger distance to the nearest other value's distribution */
    nearest_other: PerPosition
  }
  methods: Record<SteerMethodId, SteerMethodResult>
  /** On the class means' top two components, scaled to their RMS radius */
  plot: {
    means: number[][]
    eval: number[][]
    curve: number[][]
    manifold: number[][]
    linear: number[][]
  }
  total_tokens: number
  cost: number
}

// SAE

export type SaeVerdict = 'capture' | 'shatter' | 'dilute' | 'missed'

export interface GeometrySaeRequest {
  model: string
  concept: ConceptSpec
  read: GeometryRead
  chat_template: boolean
  sae: SaeRef
  seed: number
}

export interface GeometrySaeEstimate {
  estimated_cost: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  /** The decoder layer whose output the SAE reads */
  layer: number
}

export interface SaeFields {
  /** Mean share of values in a receptive field (where a feature reaches half its peak) */
  width: number | null
  /** Share of values some field contains */
  coverage: number | null
  /** Share of fields that are one run of adjacent values (null for categorical concepts) */
  contiguous: number | null
  /** Mean number of the features active on an example, over examples any is active on */
  coactive: number | null
}

/**
 * Restricted-support R² on one view of the shape: the centered class means on their top `dims`
 * components (null: all of them), reconstructed from features' fixed contributions.
 */
export interface SaeShapeFit {
  dims: number | null
  /** Share of the class means' variance in these dimensions */
  variance: number | null
  /** R² with every feature */
  ceiling: number | null
  /** Features in greedy order */
  order: number[]
  /** R² of the first n greedy features */
  r2: PerPosition
  /** Each greedy feature's decoder cosine with the shape's subspace */
  cos: PerPosition
  /** R² of the first n varying features in random orders */
  random: { mean: PerPosition; max: PerPosition }
  /** The features nearest the shape's subspace by decoder cosine */
  cosine_order: number[]
  cosine_r2: PerPosition
  /** Greedy features reaching 90% (null if they never do) */
  n90: number | null
  /** Components holding 90% of the shape's variance here */
  spans: number
  /** The most features a compact group may have */
  compact: number
  fields: SaeFields
  verdict: SaeVerdict
}

export interface GeometrySaeResult {
  model: string
  num_layers: number
  hidden_dim: number
  read: GeometryRead
  chat_template: boolean
  seed: number
  layer: number
  /** The residual stream position read (layer + 1) */
  position: number
  values: string[]
  sae: { info: ProbeSaeInfo; tokens: number; l0: number; fvu: number | null }
  active_features: number
  varying_features: number
  features: {
    ids: number[]
    /** [F, k] mean activation per value */
    tuning: DecodedTensor<number[][]>
    /** [F, k] share of each value's examples it's active on */
    active: DecodedTensor<number[][]>
    /** The example it's most active on (example i: template floor(i / k), value i % k) */
    top_example: (number | null)[]
    top_activation: (number | null)[]
  }
  fits: SaeShapeFit[]
  total_tokens: number
  cost: number
}

// Atlas

export type AtlasCorpusId = 'openwebtext' | 'wikitext' | 'tinystories'
export type AtlasTokens = 1_000_000 | 2_000_000 | 5_000_000

export interface AtlasCorpus {
  id: AtlasCorpusId
  name: string
  description: string
  license: string
  url: string
}

/** What an atlas read */
export interface AtlasMeta {
  model: string
  sae: SaeRef
  sae_name: string
  layer: number
  hidden_dim: number
  width: number
  corpus: { id: string; name: string; license?: string | null; url?: string | null }
  /** Tokens read (fewer than requested when the SAE's latents filled the budget) */
  tokens: number
  requested_tokens: number
  sequences: number
  seq_len: number
  stopped_early: boolean
  created_at: string
  seconds: number
}

export type AtlasJobStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface AtlasJob {
  job_id: string
  model: string
  sae_name: string
  corpus: string
  tokens: number
  status: AtlasJobStatus
  progress: { stage: string; fraction: number } | null
  /** Whether the signed-in user started it (cost and errors are only shown to them) */
  mine: boolean
  cost: number | null
  refunded: number | null
  error: string | null
  estimated_seconds: number
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface AtlasStatus {
  sae: ProbeSaeInfo
  atlases: AtlasMeta[]
  jobs: AtlasJob[]
}

export interface AtlasJobRequest {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
  tokens: AtlasTokens
  seed: number
}

export interface AtlasEstimate {
  estimated_cost: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  corpus_available: boolean
}

export interface AtlasRef {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
}

/** A community's activations on the span of its decoder rows: PCA explained ratios, against random groups */
export interface AtlasGeometry {
  /** Stored tokens it was measured on */
  tokens: number
  /** Top 5 explained variance ratios */
  explained: number[]
  /** The same for frequency-matched random groups of as many features (mean) */
  baseline: number[] | null
  /** Top-3 share minus the random groups' */
  score: number | null
}

export interface AtlasCommunitySummary {
  id: number
  size: number
  features: number[]
  /** Corpus tokens any of its features is active on */
  tokens: number
  /** [text, count] of the tokens (or words) it fires on most */
  top_words: [string, number][]
  /** Position by geometry score (null: not measured) */
  rank: number | null
  geometry: AtlasGeometry | null
  concept: boolean
}

export interface AtlasSummary extends AtlasMeta {
  l0: number
  fvu: number | null
  alive: number
  stored_tokens: number
  /** tokens: those co-active pairs were counted on (fewer than read for a dense SAE; missing in older atlases) */
  graph: { nodes: number; edges: number; tokens?: number; min_firings: number; neighbors: number; min_pair_count: number; min_lift: number; resolution: number }
  geometry: { max_features: number; tokens: number; min_tokens: number; random_groups: number; measured: number }
  communities: AtlasCommunitySummary[]
  /** Community IDs by geometry score */
  ranked: number[]
  map: { method: 'umap' | 'pca'; features: number[]; xy: [number, number][] }
}

export interface AtlasContext {
  tokens: string[]
  /** Index of the token it fired on */
  at: number
  /** Its activation there */
  act: number
  /** Its activation on each token, percent of its maximum over the corpus */
  levels: number[]
}

export interface AtlasCommunityConcept extends CustomConcept {
  source?: string
  /** The built-in concept whose order and positions it took */
  matched?: string | null
}

export interface AtlasCommunity {
  id: number
  features: { id: number; tokens: number; mean: number; max: number; top: string | null }[]
  top_words: [string, number][]
  examples: (AtlasContext & { feature: number })[]
  concept: AtlasCommunityConcept | null
  concept_note: string | null
  geometry:
    | (AtlasGeometry & {
        /** [≤400, 3] on the top three PCs, RMS radius 1 */
        points: number[][]
        /** Index into features of the most active one at each point */
        point_feature: number[]
        /** [before, token, after] at each point */
        point_context: [string, string, string][]
      })
    | null
}

export interface AtlasFeatureRow {
  id: number
  tokens: number
  freq: number
  mean: number
  max: number
  community: number | null
  /** How varied the tokens its top contexts fire on are (0: all the same, 1: all different) */
  diversity: number
}

export interface AtlasFeaturesRequest extends AtlasRef {
  sort: 'tokens' | 'mean' | 'max' | 'id'
  descending: boolean
  offset: number
  limit: number
  community?: number | null
}

export interface AtlasFeaturesResponse {
  total: number
  alive: number
  width: number
  corpus_tokens: number
  rows: AtlasFeatureRow[]
  /** Features never active over the corpus (the first 10,000) */
  dead: number[]
}

export interface AtlasFeature extends AtlasFeatureRow {
  corpus_tokens: number
  examples: AtlasContext[]
  /** weight: sqrt(P(j|i) P(i|j)); conditional: P(j|i) */
  neighbors: { id: number; weight: number; conditional: number }[]
}
