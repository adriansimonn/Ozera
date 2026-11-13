/** Autointerp: Claude's explanations of SAE features and their scores (backend api/autointerp.py). */

import type { AtlasCorpusId } from './geometry'
import type { SaeRef } from './probes'

export interface AutointerpModel {
  id: string
  name: string
}

export interface AutointerpAtlas {
  model: string
  corpus: string
  corpus_name: string
  version: number
  tokens: number
  requested_tokens: number
  seed: number
  width: number
  l0: number | null
  created_at: string
}

export type AutointerpState = 'ready' | 'recompute' | 'dense' | 'no_atlas'

export interface AutointerpStatus {
  /** Whether Claude is configured on the server */
  available: boolean
  atlas: AutointerpAtlas | null
  state: AutointerpState
  reason: string | null
  explainer: string
  scorer: string
  /** Models a single feature can be explained and scored with */
  models: AutointerpModel[]
  /** Models a job can use */
  job_models: AutointerpModel[]
  /** Expected charge for explaining and scoring one feature with the default models */
  explain_price: number
  /** Expected charge for scoring a hypothesis with the default scorer */
  hypothesis_price: number
  /** Expected charge for a generation test (Claude's texts and the GPU call reading them) */
  generation_test_price: number | null
  /** The versions explanations are written and scored with now */
  prompt_version: number
  examples_version: number
  /** The atlas's communities and their names */
  communities: { total: number; nameable: number; named: number; to_name: number } | null
  explained: number
  controls: number
  /** Live features of the atlas */
  alive: number | null
  max_job_features: number
  score_sample: number
  /** A job running on this atlas (anyone's) and the user's recent ones */
  jobs: AutointerpJob[]
}

/** communities: name the atlas's communities from stored explanations (explains nothing) */
export type JobSelectionKind = 'top' | 'range' | 'list' | 'all' | 'communities'

export interface JobSelection {
  kind: JobSelectionKind
  n?: number
  start?: number
  end?: number
  features?: number[]
  scoring: 'all' | 'sample'
}

export interface JobRequest {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
  selection: JobSelection
  explainer: string
  scorer: string
}

export interface JobEstimate {
  features: number
  scored: number
  controls: number
  /** Selected features already explained with these models */
  skipped: number
  estimated_cost: number
  /** About what the job would hold (its upper bound, from estimated input) */
  hold: number
  max_features: number
  /** A naming job: communities it would name */
  communities: number
}

export type AutointerpJobStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface AutointerpJob {
  job_id: string
  model: string
  corpus: string
  sae_name: string
  explainer: string
  scorer: string
  selection: JobSelection & { skipped?: number; listed?: number }
  features: number
  scored: number
  controls: number
  status: AutointerpJobStatus
  stage: 'preparing' | 'explaining' | 'scoring' | 'importing' | null
  progress: { stage: string; fraction: number } | null
  mine: boolean
  estimated: number | null
  /** Charged as a hold so far */
  held: number | null
  bound: number | null
  settled: number | null
  refunded: number | null
  error: string | null
  explained: { explained: number; refused: number; failed: number; scoring: number } | null
  results: Record<string, number> | null
  /** A naming job: communities to name, and the names read */
  communities: number | null
  names: { named: number; refused: number; failed: number } | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export type FeatureSort = 'feature' | 'score' | 'detection' | 'fuzzing' | 'delta' | 'freq'

export interface FeaturesRequest {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
  sort: FeatureSort
  descending: boolean
  offset: number
  limit: number
  min_score?: number | null
  status?: 'scored' | 'unscored' | 'refused' | null
  token_features?: 'only' | 'exclude' | null
  search?: string | null
}

export interface FeatureRow {
  feature: number
  explanation: string | null
  status: 'scored' | 'unscored' | 'refused'
  note: string | null
  score: number | null
  detection: number | null
  detection_hard: number | null
  fuzzing: number | null
  token_detection: number | null
  /** Detection minus the token baseline's */
  delta: number | null
  token_feature: boolean | null
  freq: number | null
  explainer: string
  scorer: string
}

export interface FeaturesResponse {
  total: number
  explained: number
  rows: FeatureRow[]
}

export interface ScoreQuantiles {
  n: number
  mean: number
  p10: number
  median: number
  p90: number
}

export interface FrequencyBand {
  lo: number
  hi: number
  n: number
  mean: number | null
  median: number | null
  /** Share above the controls' 90th percentile */
  above_controls: number | null
}

export interface ModelPair {
  explainer: string
  scorer: string
  features: number
  controls: number
}

export interface SummaryResponse {
  pairs: ModelPair[]
  explainer: string | null
  scorer: string | null
  scores: number[]
  freqs: (number | null)[]
  deltas: (number | null)[]
  control_scores: number[]
  features: ScoreQuantiles | null
  controls: ScoreQuantiles | null
  token_baseline: ScoreQuantiles | null
  above_controls: number | null
  token_features: number
  statuses: Record<string, number>
  bands: FrequencyBand[]
}

export interface ExampleContext {
  tokens: string[]
  levels: number[]
  read: boolean[]
  centre: number
  act: number
  decile: number | null
  /** Fuzzing: the positions the scorer saw marked */
  marks: number[] | null
  /** Explain set: top or decile; negatives: random or token */
  kind: string | null
}

export interface ExamplesResponse {
  feature: number
  explain: ExampleContext[]
  mistakes: {
    missed: ExampleContext[]
    false_alarms: ExampleContext[]
    fuzzing: ExampleContext[]
    counts: Record<string, number>
  } | null
  note: string | null
}

export interface AutointerpFeatureRef {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
  feature: number
}

export interface Explanation {
  id: number
  feature: number
  explanation: string | null
  status: 'scored' | 'unscored' | 'refused'
  note: string | null
  /** Balanced accuracies (0.5 is chance) */
  detection: number | null
  detection_ci: number | null
  detection_hard: number | null
  fuzzing: number | null
  fuzzing_ci: number | null
  score: number | null
  /** How much the score moves when explained and scored again (95%, from calibration) */
  score_pm: number | null
  token_detection: number | null
  token_feature: boolean | null
  decile_recall: (number | null)[] | null
  control_percentile: number | null
  explainer: string
  scorer: string
  prompt_version: number
  examples_version: number
  created_at: string
}

export interface ExplainResponse extends Explanation {
  cost: number
  cached: boolean
}

export interface TokenWeight {
  token: string
  value: number
}

/** The user's own explanation of a feature, scored on the same held-out set as autointerp's */
export interface Hypothesis {
  id: number
  feature: number
  text: string
  status: 'scored' | 'unscored' | 'refused'
  note: string | null
  detection: number | null
  detection_ci: number | null
  detection_hard: number | null
  fuzzing: number | null
  fuzzing_ci: number | null
  score: number | null
  token_detection: number | null
  decile_recall: (number | null)[] | null
  control_percentile: number | null
  scorer: string
  prompt_version: number
  examples_version: number
  created_at: string
}

export interface HypothesisResponse {
  hypothesis: Hypothesis
  cost: number
  cached: boolean
}

export interface GenerationText {
  /** Written to make the feature fire, or as a near-miss */
  kind: 'firing' | 'near_miss'
  tokens: string[]
  /** The feature's activation on each token, percent of its maximum over the corpus */
  levels: number[]
  read: boolean[]
  peak: number
  /** The peak reaches the threshold */
  fired: boolean
}

/** Texts Claude wrote from an explanation, read through the model and SAE */
export interface GenerationTest {
  id: number
  feature: number
  explanation_id: number | null
  hypothesis_id: number | null
  text: string
  generator: string
  /** Share of the corpus maximum a text must reach to count as firing */
  threshold: number
  maximum: number
  firing_rate: number | null
  near_miss_rate: number | null
  texts: GenerationText[]
  mine: boolean
  created_at: string
}

export interface GenerationTestResponse {
  test: GenerationTest
  cost: number
  cached: boolean
}

/** What a generation test or the mistakes view is about: an explanation, or one of the user's hypotheses */
export interface ExplanationSource {
  explanation_id?: number
  hypothesis_id?: number
}

export interface CommunityName {
  community: number
  name: string
  /** Its features with an explanation when it was named */
  explained: number
  size: number
  namer: string
  created_at: string
}

export interface FeatureAutointerp {
  feature: number
  explanation: Explanation | null
  /** Every explanation of the feature, newest first (the first is the one shown) */
  versions: Explanation[]
  hypotheses: Hypothesis[]
  /** Generation tests of its explanations, and of the user's hypotheses */
  tests: GenerationTest[]
  top_tokens: TokenWeight[]
  promoted: TokenWeight[]
  suppressed: TokenWeight[]
  has_sets: boolean
  scorable: boolean
  unscored_note: string | null
  controls: number
}

export interface FeatureLabel {
  feature: number
  explanation: string
  status: 'scored' | 'unscored'
  score: number | null
  detection: number | null
  fuzzing: number | null
  token_feature: boolean | null
  explainer: string
}

export interface LabelsResponse {
  model: string | null
  corpus: string | null
  labels: FeatureLabel[]
}
