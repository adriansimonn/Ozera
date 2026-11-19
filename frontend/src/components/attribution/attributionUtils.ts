/**
 * Shared helpers of the Attribution page: names and numbers as its views write them, the
 * layer × position grid a scope shows, a sweep's top cells, and component lists as text.
 */

import type {
  AblationMethod,
  AttributionEstimate,
  AttributionRun,
  Cell,
  Component,
  PositionKind,
} from '../../types/attribution'
import { POSITION_KINDS } from '../../types/attribution'

export const KIND_LABELS: Record<PositionKind, string> = { resid: 'Residual', attn: 'Attention', mlp: 'MLP' }
export const KIND_TITLES: Record<PositionKind, string> = {
  resid: 'The residual stream entering each layer (layer 0: the embeddings)',
  attn: "Each layer's attention output",
  mlp: "Each layer's MLP output",
}
export const METHOD_LABELS: Record<AblationMethod, string> = { zero: 'Zero', mean: 'Mean', resample: 'Resample' }
export const METHOD_TITLES: Record<AblationMethod, string> = {
  zero: 'Outputs set to zero',
  mean: "Outputs set to their mean over the pair set's prompts",
  resample: "Outputs taken from each pair's corrupted prompt",
}

/** A signed effect, to three digits. */
export function effect(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return '–'
  const text = Math.abs(value).toFixed(digits)
  return `${value < 0 && Number(text) !== 0 ? '−' : ''}${text}`
}

/** A metric's own value (logits or nats): two decimals, or two significant digits when it's small. */
export function metricValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '–'
  const size = Math.abs(value)
  if (size === 0) return '0'
  const text = size >= 0.1 ? size.toFixed(2) : size.toPrecision(2)
  return `${value < 0 ? '−' : ''}${text}`
}

export function metricName(metric: AttributionRun['metric']): string {
  return metric === 'kl' ? 'KL' : 'Logit diff'
}

/** What effects are measured in, for axis titles and tooltips. */
export function effectUnit(run: { normalized: boolean; metric: AttributionRun['metric'] }): string {
  if (run.normalized) return 'share of the clean − corrupted gap'
  return run.metric === 'kl' ? 'nats' : 'logits'
}

export function estimateText(estimate: AttributionEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

/** A token as a label: whitespace made visible. */
export function tokenLabel(token: string | null | undefined): string {
  if (token == null) return '·'
  const text = token.replace(/\n/g, '↵').replace(/\t/g, '→')
  return text.trim() === '' ? '␣'.repeat(Math.max(1, text.length)) : text
}

export function headName(layer: number, head: number): string {
  return `${layer}.${head}`
}

export function componentName(c: Component): string {
  if (c.kind === 'head') return headName(c.layer, c.head ?? 0)
  return `${c.kind === 'attn' ? 'a' : 'm'}${c.layer}`
}

export function cellName(cell: Cell): string {
  const where = cell.position == null ? '' : ` @${cell.position}`
  if (cell.kind === 'head') return `Head ${headName(cell.layer, cell.head ?? 0)}${where}`
  const kind = cell.kind === 'resid' ? 'Resid' : cell.kind === 'attn' ? 'Attn' : 'MLP'
  return `${kind} L${cell.layer}${where}`
}

/**
 * Components from text: a head as layer.head, a layer's attention as a<layer>, its MLP as
 * m<layer>, separated by commas or spaces. Returns the components, or what's wrong.
 */
export function parseComponents(text: string, layers: number, heads: number): { components: Component[]; error: string | null } {
  const components: Component[] = []
  const seen = new Set<string>()
  for (const part of text.split(/[\s,;]+/).filter(Boolean)) {
    const head = /^[lL]?(\d+)[.:hH](\d+)$/.exec(part)
    const layer = /^([amAM])(\d+)$/.exec(part)
    let component: Component
    if (head) component = { kind: 'head', layer: Number(head[1]), head: Number(head[2]) }
    else if (layer) component = { kind: layer[1].toLowerCase() === 'a' ? 'attn' : 'mlp', layer: Number(layer[2]), head: null }
    else return { components: [], error: `Can't read "${part}": write 9.6, a9 or m9` }
    if (component.layer >= layers) return { components: [], error: `${part}: layers run 0–${layers - 1}` }
    if (component.kind === 'head' && (component.head ?? 0) >= heads) return { components: [], error: `${part}: heads run 0–${heads - 1}` }
    const key = componentName(component)
    if (!seen.has(key)) {
      seen.add(key)
      components.push(component)
    }
  }
  return { components, error: null }
}

// Scopes: what a layer × position map is of. "cls:<id>" is the mean over an alignment class's
// pairs, "pair:<index>" one pair.

export interface ScopeOption {
  value: string
  label: string
}

export function scopeOptions(run: AttributionRun): ScopeOption[] {
  const means = run.classes
    .filter((c) => c.pairs.length > 1)
    .sort((a, b) => b.pairs.length - a.pairs.length)
    .map((c) => ({ value: `cls:${c.id}`, label: `Mean · ${c.group ?? `${c.tokens} tokens`} (${c.pairs.length})` }))
  const pairs = run.pairs.map((_, i) => ({ value: `pair:${i}`, label: `Pair ${i + 1}` }))
  return [...means, ...pairs]
}

export function defaultScope(run: AttributionRun): string {
  return scopeOptions(run)[0]?.value ?? 'pair:0'
}

export interface Scope {
  cls: number | null
  pair: number | null
  /** The pair whose tokens stand for the scope's (a class's first pair) */
  example: number
  /** Each position's label: the token, or null where the scope's prompts differ */
  labels: (string | null)[]
  /** Positions where the clean and corrupted prompts differ */
  changed: boolean[]
}

export function readScope(run: AttributionRun, scope: string): Scope | null {
  const [kind, raw] = scope.split(':')
  const index = Number(raw)
  if (kind === 'cls') {
    const group = run.classes.find((c) => c.id === index)
    if (!group) return null
    const changed = group.labels.map((_, p) => group.pairs.some((i) => run.pairs[i].clean_tokens[p] !== run.pairs[i].corrupted_tokens[p]))
    return { cls: index, pair: null, example: group.pairs[0], labels: group.labels, changed }
  }
  const pair = run.pairs[index]
  if (!pair) return null
  return {
    cls: null,
    pair: index,
    example: index,
    labels: pair.clean_tokens,
    changed: pair.clean_tokens.map((token, p) => token !== pair.corrupted_tokens[p]),
  }
}

/** A scope's [layers][tokens] grid of one kind's estimated effects. */
export function positionGrid(run: AttributionRun, scope: Scope, kind: PositionKind): number[][] {
  const grids = scope.cls != null ? run.position_classes[scope.cls] : run.position_pairs[scope.pair ?? 0]
  return grids?.[POSITION_KINDS.indexOf(kind)] ?? []
}

function largest<T extends { estimate?: number | null }>(cells: T[], count: number): T[] {
  return [...cells].sort((a, b) => Math.abs(b.estimate ?? 0) - Math.abs(a.estimate ?? 0)).slice(0, count)
}

/** Every head as a cell (patched at every position, on every pair), with its estimate. */
export function headCells(run: AttributionRun): Cell[] {
  return run.heads.flatMap((row, layer) => row.map((estimate, head) => ({ kind: 'head' as const, layer, head, estimate })))
}

/** A sweep's largest heads and whole attention and MLP layers. */
export function topComponents(run: AttributionRun, count: number): Cell[] {
  const layers = (['attn', 'mlp'] as const).flatMap((kind, k) =>
    run.layers[k].map((estimate, layer) => ({ kind, layer, estimate })),
  )
  return largest([...headCells(run), ...layers], count)
}

function scopeCell(scope: Scope): Pick<Cell, 'pair' | 'cls'> {
  return scope.cls != null ? { cls: scope.cls } : { pair: scope.pair }
}

/** The largest cells of a scope's map of one kind. */
export function topPositions(run: AttributionRun, scope: Scope, kind: PositionKind, count: number): Cell[] {
  const grid = positionGrid(run, scope, kind)
  const cells = grid.flatMap((row, layer) => row.map((estimate, position) => ({ kind, layer, position, estimate, ...scopeCell(scope) })))
  return largest(cells, count)
}

/** One pair's residual stream at every layer and position. */
export function residualCells(run: AttributionRun, pair: number): Cell[] {
  const grid = run.position_pairs[pair]?.[0] ?? []
  return grid.flatMap((row, layer) => row.map((estimate, position) => ({ kind: 'resid' as const, layer, position, pair, estimate })))
}

const patternCache = new WeakMap<AttributionRun, Map<number, Uint8Array>>()

/** A head's attention pattern on one of the pairs a sweep kept them for: [tokens][tokens], or null. */
export function attentionPattern(run: AttributionRun, pair: number, layer: number, head: number): number[][] | null {
  const slot = run.attention?.pairs.indexOf(pair) ?? -1
  if (!run.attention || slot < 0) return null
  let cache = patternCache.get(run)
  if (!cache) {
    cache = new Map()
    patternCache.set(run, cache)
  }
  let bytes = cache.get(pair)
  if (!bytes) {
    const binary = atob(run.attention.patterns[slot])
    bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
    cache.set(pair, bytes)
  }
  const tokens = run.pairs[pair].clean_tokens.length
  const start = (layer * run.num_heads + head) * tokens * tokens
  if (start + tokens * tokens > bytes.length) return null
  const out: number[][] = []
  for (let i = 0; i < tokens; i++) {
    const row = new Array<number>(tokens)
    for (let j = 0; j < tokens; j++) row[j] = bytes[start + i * tokens + j] / 255
    out.push(row)
  }
  return out
}

/** The largest absolute value of a grid (at least a small floor, so an all-zero grid has a scale). */
export function gridScale(...grids: number[][][]): number {
  let max = 0
  for (const grid of grids) for (const row of grid) for (const value of row) if (Math.abs(value) > max) max = Math.abs(value)
  return max > 1e-9 ? max : 1
}
