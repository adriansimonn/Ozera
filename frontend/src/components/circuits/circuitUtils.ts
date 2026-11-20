/** Reading a circuit graph: names, numbers as text, a node's edges, and which nodes a view shows. */

import type { CircuitGraph, GraphNode, InterveneEstimate, TraceEstimate } from '../../types/circuits'
import { percent, showToken } from '../visualization/tokenText'

export { percent, showToken }

/** How many of a graph's feature and error nodes the chart shows (the most influential first) */
export const SHOWN_OPTIONS = [40, 80, 160, 0] as const
/** Edges drawn when no node is selected: the ones carrying most of the target's attribution */
export const OVERVIEW_EDGES = 140
/** Feature nodes an intervention takes at most (backend/core/circuits/budget.py) */
export const MAX_INTERVENE_NODES = 64

export const SCALES: { value: number; label: string; title: string }[] = [
  { value: 0, label: 'Ablate', title: 'Set the nodes’ activations to zero' },
  { value: 0.5, label: '× 0.5', title: 'Halve the nodes’ activations' },
  { value: 2, label: '× 2', title: 'Double the nodes’ activations' },
  { value: -1, label: '× −1', title: 'Flip the sign of the nodes’ activations' },
]

/** A logit or a change in one, with its sign. */
export function logits(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '–'
  const text = Math.abs(value).toFixed(digits)
  return `${value < 0 && Number(text) !== 0 ? '−' : ''}${text}`
}

export function signedLogits(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '–'
  const text = Math.abs(value).toFixed(digits)
  if (Number(text) === 0) return text
  return `${value < 0 ? '−' : '+'}${text}`
}

export function money(value: number): string {
  return `$${value.toFixed(2)}`
}

export function traceEstimateText(estimate: TraceEstimate): string {
  return estimate.within_limit ? `≈ ${money(estimate.estimated_cost)} · ~${Math.ceil(estimate.estimated_seconds)}s GPU` : 'Too long to trace'
}

export function interveneEstimateText(estimate: InterveneEstimate): string {
  return `≈ ${money(estimate.estimated_cost)} · ~${Math.ceil(estimate.estimated_seconds)}s GPU`
}

/** A row's name: the embeddings, a layer, or the output. */
export function rowLabel(row: number, layers: number): string {
  return row < 0 ? 'Emb' : row >= layers ? 'Out' : `L${row}`
}

/** The row a node sits on: -1 for tokens, its layer for features and errors, the layer count for the target. */
export function nodeRow(node: GraphNode, layers: number): number {
  return node.kind === 'token' ? -1 : node.kind === 'logit' ? layers : (node.layer ?? 0)
}

/** A node in a few words: "L12 #3411", "L5 error", a token, the target. */
export function nodeName(node: GraphNode): string {
  if (node.kind === 'feature') return `L${node.layer} #${node.feature}`
  if (node.kind === 'error') return `L${node.layer} error`
  if (node.kind === 'token') return showToken(node.token)
  return `→ ${showToken(node.token)}`
}

/** A node with where it sits: "L12 #3411 at ·France". */
export function nodeTitle(node: GraphNode, graph: CircuitGraph): string {
  if (node.kind === 'token') return `Token ${showToken(node.token)} (position ${node.position})`
  if (node.kind === 'logit') return `Target ${showToken(node.token)}`
  const at = `at ${showToken(graph.tokens[node.position])}`
  return node.kind === 'feature' ? `Layer ${node.layer} · feature ${node.feature} ${at}` : `Layer ${node.layer} error ${at}`
}

export interface NodeEdge {
  /** The node at the other end */
  node: GraphNode
  index: number
  weight: number
  share: number
}

/** A node's inputs and outputs among the graph's stored edges, the largest shares first. */
export function nodeEdges(graph: CircuitGraph, index: number): { inputs: NodeEdge[]; outputs: NodeEdge[] } {
  const { source, target, weight, share } = graph.edges
  const inputs: NodeEdge[] = []
  const outputs: NodeEdge[] = []
  for (let i = 0; i < source.length; i++) {
    if (target[i] === index) inputs.push({ node: graph.nodes[source[i]], index: source[i], weight: weight[i], share: share[i] })
    else if (source[i] === index) outputs.push({ node: graph.nodes[target[i]], index: target[i], weight: weight[i], share: share[i] })
  }
  // The edges arrive sorted by share
  return { inputs, outputs }
}

/** The graph's feature nodes, the most influential first. */
export function rankedFeatures(graph: CircuitGraph): GraphNode[] {
  return graph.nodes.filter((n) => n.kind === 'feature').sort((a, b) => b.influence - a.influence)
}

/** The feature nodes that support the target most (the largest positive support first). */
export function supportingFeatures(graph: CircuitGraph): GraphNode[] {
  return graph.nodes.filter((n) => n.kind === 'feature' && n.support > 0).sort((a, b) => b.support - a.support)
}

/** A signed share: +3.2%, −0.4%. */
export function signedPercent(value: number): string {
  if (!Number.isFinite(value)) return '–'
  const text = percent(Math.abs(value))
  return /^0(\.0)?%$/.test(text) ? text : `${value < 0 ? '−' : '+'}${text}`
}

/**
 * The indices of the nodes a chart shows: every token, the target, the `count` most
 * influential feature and error nodes (all of them for 0), and whatever is pinned or selected.
 */
export function shownNodes(graph: CircuitGraph, count: number, keep: Iterable<string>, errors: boolean): Set<number> {
  const kept = new Set(keep)
  const inner = graph.nodes
    .map((node, index) => ({ node, index }))
    .filter(({ node }) => node.kind === 'feature' || (errors && node.kind === 'error'))
    .sort((a, b) => b.node.influence - a.node.influence)
  const shown = new Set<number>()
  graph.nodes.forEach((node, index) => {
    if (node.kind === 'token' || node.kind === 'logit' || kept.has(node.id)) shown.add(index)
  })
  for (const { index } of count > 0 ? inner.slice(0, count) : inner) shown.add(index)
  return shown
}

/** Error nodes' share of the influence that reaches the target (1 − the replacement score). */
export function errorShare(graph: CircuitGraph): number {
  return 1 - graph.scores.replacement
}
