/**
 * Shared helpers of the Model Diff page: names and numbers as its views write them, a layers
 * run's profiles and their halves, and its tokens in order.
 */

import type { DiffEstimate, DiffPair, DiffText, Donor, LayerMeasure, LayersRun, PatchKind } from '../../types/diff'

export const MEASURE_LABELS: Record<LayerMeasure, string> = { distance: 'Distance', cka: 'CKA', procrustes: 'Procrustes', cosine: 'Cosine' }
export const MEASURE_TITLES: Record<LayerMeasure, string> = {
  distance: 'How far apart the two models’ activations of the same token are, over how far activations spread around their mean: 0 identical, about 1.4 unrelated',
  cka: 'Linear CKA: 1 when one model’s activations are a rotation and scaling of the other’s, weighted toward the directions that vary most',
  procrustes: 'How much of one model’s activations a scaled rotation of the other’s explains',
  cosine: 'The mean cosine between the two models’ activations of the same token',
}
export const KIND_LABELS: Record<PatchKind, string> = { resid: 'Residual', attn: 'Attention', mlp: 'MLP' }
export const KIND_TITLES: Record<PatchKind, string> = {
  resid: 'The residual stream entering each layer (layer 0: the embeddings)',
  attn: 'Each layer’s attention output',
  mlp: 'Each layer’s MLP output',
}

/** A pair as its picker writes it. */
export function pairLabel(pair: DiffPair): string {
  const shape = `${pair.num_layers}L · d=${pair.hidden_dim}`
  return pair.kind === 'custom' ? `${pair.other_name} vs ${pair.base_name} · ${shape}` : `${pair.base_name} vs Instruct · ${shape}`
}

/** The two models of a result, as its heading names them. */
export function pairName(result: { base_name: string; other_name: string }): string {
  return `${result.base_name} vs ${result.other_name}`
}

export function donorNames(result: { base_name: string; other_name: string }, donor: Donor): { donor: string; recipient: string } {
  return donor === 'other' ? { donor: result.other_name, recipient: result.base_name } : { donor: result.base_name, recipient: result.other_name }
}

export function estimateText(estimate: DiffEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

/** A residual stream position on an axis: the embeddings, then each layer's output. */
export function positionLabel(position: number): string {
  return position === 0 ? 'emb' : String(position - 1)
}

export function positionName(position: number): string {
  return position === 0 ? 'Embeddings' : `Layer ${position - 1}`
}

export function value(v: number | null | undefined, digits = 3): string {
  return v == null || !Number.isFinite(v) ? '–' : v.toFixed(digits)
}

/** A share as a signed percentage (it can be negative: an edit that moved predictions away). */
export function percent(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return '–'
  const text = Math.abs(v * 100).toFixed(digits)
  return `${v < 0 && Number(text) !== 0 ? '−' : ''}${text}%`
}

/** A KL in nats: two decimals, or two significant digits when it's small. */
export function nats(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '–'
  if (v === 0) return '0'
  return Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toPrecision(2)
}

/** A token as a label: whitespace made visible. */
export function tokenLabel(token: string | null | undefined): string {
  if (token == null) return '·'
  const text = token.replace(/\n/g, '↵').replace(/\t/g, '→')
  return text.trim() === '' ? '␣'.repeat(Math.max(1, text.length)) : text
}

/** The largest difference between a measure's profiles on the two halves of the texts, and where. */
export function halvesGap(run: LayersRun, measure: LayerMeasure): { gap: number; position: number } | null {
  const halves = run.halves[measure]
  if (!halves) return null
  let gap = -1
  let position = 0
  halves[0].forEach((a, p) => {
    const b = halves[1][p]
    if (a != null && b != null && Math.abs(a - b) > gap) {
      gap = Math.abs(a - b)
      position = p
    }
  })
  return gap < 0 ? null : { gap, position }
}

/** The span of a profile's values. */
export function profileRange(values: (number | null)[]): [number, number] {
  const known = values.filter((v): v is number => v != null)
  return known.length ? [Math.min(...known), Math.max(...known)] : [0, 1]
}

export function peak(values: (number | null)[], lowest = false): number {
  let best = 0
  values.forEach((v, p) => {
    const current = values[best]
    if (v != null && (current == null || (lowest ? v < current : v > current))) best = p
  })
  return best
}

/** Where each text's tokens start among all the run's compared tokens. */
export function tokenOffsets(texts: DiffText[]): number[] {
  const offsets: number[] = []
  let total = 0
  for (const text of texts) {
    offsets.push(total)
    total += text.tokens.length
  }
  return offsets
}

export interface DivergentToken {
  text: number
  index: number
  kl: number
}

/** The tokens where the two models' next-token predictions are furthest apart. */
export function mostDivergent(run: LayersRun, count: number): DivergentToken[] {
  const all: DivergentToken[] = []
  run.texts.forEach((text, t) => text.kl.forEach((kl, index) => all.push({ text: t, index, kl })))
  return all.sort((a, b) => b.kl - a.kl).slice(0, count)
}

/** The ranges of positions where tokens were left out for their massive activations, as text. */
export function massiveText(run: LayersRun): string | null {
  const most = Math.max(0, ...run.massive)
  if (most === 0) return null
  const where = run.massive.map((n, p) => (n > 0 ? p : -1)).filter((p) => p >= 0)
  const span = where.length === 1 ? positionName(where[0]).toLowerCase() : `${positionName(where[0]).toLowerCase()} to ${positionName(where[where.length - 1]).toLowerCase()}`
  return `Up to ${most} of ${run.num_tokens.toLocaleString()} tokens left out for massive activations (${span}).`
}
