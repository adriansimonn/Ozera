/**
 * Types of the Model Diff page (backend/api/schemas/diff.py).
 */

export type PatchKind = 'resid' | 'attn' | 'mlp'
export type Donor = 'other' | 'base'
export type LayerMeasure = 'distance' | 'cka' | 'procrustes' | 'cosine'

export const PATCH_KINDS: PatchKind[] = ['resid', 'attn', 'mlp']
export const LAYER_MEASURES: LayerMeasure[] = ['distance', 'cka', 'procrustes', 'cosine']

/** Two models of the same shape: a shared base model and what it's compared with. */
export interface DiffPair {
  base: string
  other: string
  base_name: string
  other_name: string
  /** The other model is the base model's instruct sibling, or the user's custom model */
  kind: 'instruct' | 'custom'
  num_layers: number
  hidden_dim: number
  parameters: number
  gpu_tier: string
  /** Whether texts can be read in a chat template */
  chat_template: boolean
}

export interface DiffSample {
  id: string
  name: string
  description: string
  num_texts: number
  default_texts: number
  examples: string[]
}

export interface TextSpec {
  sample?: string | null
  n?: number
  seed?: number
  texts?: string[] | null
  name?: string | null
}

export interface DiffEstimate {
  estimated_cost: number
  estimated_tokens: number
  estimated_seconds: number
  max_seconds: number
  within_limit: boolean
  num_texts: number
  /** Ozera models' texts are tokenized before the run; open-weight models' when it starts */
  checked: boolean
}

export interface DroppedText {
  index: number
  reason: string
}

/** One text, token by token (its first token isn't compared). */
export interface DiffText {
  source: number
  half: number
  /** The text's tokens before the compared ones (its first, on a model without a BOS) */
  lead: string[]
  tokens: string[]
  /** KL(other ‖ base) of the next-token predictions, in nats */
  kl: number[]
  base_top: string[]
  other_top: string[]
  base_prob: number[]
  other_prob: number[]
}

type Profile = (number | null)[]

interface ResultBase {
  base: string
  other: string
  base_name: string
  other_name: string
  chat_template: boolean
  sample: string | null
  texts_name: string
  seed: number
  cost: number
}

/**
 * Two models compared at every residual stream position (0: the embeddings, i: layer i-1's
 * output) and at every token.
 */
export interface LayersRun extends ResultBase {
  num_layers: number
  hidden_dim: number
  num_texts: number
  /** Tokens compared (each text's first token aside) */
  num_tokens: number
  total_tokens: number
  half_tokens: number[]
  distance: Profile
  cosine: Profile
  cka: Profile
  cka_chance: Profile
  procrustes: Profile
  /** Each measure on each half of the texts alone; CKA and Procrustes only on narrow models */
  halves: {
    distance: Profile[]
    cosine: Profile[]
    cka: Profile[] | null
    procrustes: Profile[] | null
  }
  kl: number
  half_kl: (number | null)[]
  top1_agreement: number
  texts: DiffText[]
  /** [positions][tokens], tokens in the texts' order; −1 where a token was left out */
  token_distance: number[][]
  /** Per position, tokens left out for their massive activations */
  massive: number[]
  dropped: DroppedText[]
  cut: number
}

export interface TokenContext {
  text: number
  position: number
  value: number
  before: string[]
  token: string
  after: string[]
}

export interface DiffComponent {
  index: number
  share: number
  mean_cosine: number | null
  high: TokenContext[]
  low: TokenContext[]
}

export interface ComponentTest {
  index: number
  kl: number
  /** The share of the unchanged base model's KL the edit removes; null when the models already agree */
  closed: number | null
  agreement: number
  controls: { kl: number; closed: number | null }[]
}

export interface DirectionsRun extends ResultBase {
  /** Residual stream position */
  layer: number
  num_layers: number
  hidden_dim: number
  fit_texts: number
  test_texts: number
  fit_tokens: number
  test_tokens: number
  total_tokens: number
  massive_tokens: number
  relative_difference: number
  mean_share: number
  components: DiffComponent[]
  test: {
    kl_base: number
    kl_full: number
    agreement_base: number
    agreement_full: number
    full_closed: number | null
    components: ComponentTest[]
    joint: { count: number; kl: number; closed: number | null; agreement: number }
  }
  dropped: DroppedText[]
  cut: number
  control_seed: number
}

export interface Prediction {
  token: string
  prob: number
}

/** Values are KL(donor ‖ patched recipient) at the prompt's last position, in nats. */
export interface PatchRun {
  base: string
  other: string
  base_name: string
  other_name: string
  chat_template: boolean
  prompt: string
  donor: Donor
  kind: PatchKind
  num_layers: number
  tokens: string[]
  kl_base: number
  /** [layers][tokens] */
  grid: number[][]
  all_positions: number[]
  predictions: { donor: Prediction[]; recipient: Prediction[] }
  cost: number
}
