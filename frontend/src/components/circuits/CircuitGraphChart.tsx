/**
 * The attribution graph: token positions across, layers up (the embeddings at the bottom, the
 * target at the top). Feature nodes are squares, stronger the more of the target's attribution
 * flows through them, red where they push the target up on balance and blue where they hold it
 * down; error nodes are dashed outlines. Edges are drawn for the node under the pointer or
 * selected, and otherwise the ones that carry the most: red raises its target, blue lowers it.
 */

import { useMemo, useState, type PointerEvent } from 'react'
import type { CircuitGraph, GraphNode } from '../../types/circuits'
import { ChartTooltip, type TooltipRow } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { OVERVIEW_EDGES, nodeRow, nodeTitle, percent, rowLabel, showToken, signedPercent } from './circuitUtils'

const NODE = 13
const GAP = 4
const PER_LINE = 6
const MIN_COLUMN = 62
const LEFT = 40
const TOP = 12
const EMPTY_ROW = 10
const TOKEN_ROW = 30
const TARGET_ROW = 30

interface Props {
  graph: CircuitGraph
  /** Indices of the nodes to draw */
  shown: Set<number>
  selected: string | null
  pinned: string[]
  onSelect: (nodeId: string | null) => void
}

interface Placed {
  index: number
  node: GraphNode
  x: number
  y: number
}

export function CircuitGraphChart({ graph, shown, selected, pinned, onSelect }: Props) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(720)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const layers = graph.num_layers
  const positions = graph.tokens.length

  const layout = useMemo(() => {
    // Nodes by cell, the most influential first
    const cells = new Map<string, number[]>()
    for (const index of shown) {
      const node = graph.nodes[index]
      if (node.kind !== 'feature' && node.kind !== 'error') continue
      const key = `${node.layer}:${node.position}`
      const list = cells.get(key)
      if (list) list.push(index)
      else cells.set(key, [index])
    }
    for (const list of cells.values()) list.sort((a, b) => graph.nodes[b].influence - graph.nodes[a].influence)

    const perLine = Array.from({ length: positions }, () => 1)
    const lines = Array.from({ length: layers }, () => 0)
    for (const [key, list] of cells) {
      const [layer, position] = key.split(':').map(Number)
      perLine[position] = Math.max(perLine[position], Math.min(PER_LINE, list.length))
      lines[layer] = Math.max(lines[layer], Math.ceil(list.length / PER_LINE))
    }
    // Long prompts get narrower columns (their token labels are cut shorter)
    const minColumn = positions > 20 ? 42 : positions > 12 ? 52 : MIN_COLUMN
    const natural = perLine.map((count) => Math.max(minColumn, count * (NODE + GAP) + GAP + 10))
    // Spread the columns over the width there is, when the graph is narrower than it
    const spare = Math.max(0, width - LEFT - 8 - natural.reduce((a, b) => a + b, 0))
    const columnWidth = natural.map((w) => w + Math.min(spare / positions, 60))
    const columnX: number[] = []
    let x = LEFT
    for (const w of columnWidth) {
      columnX.push(x)
      x += w
    }
    const totalWidth = x + 8

    // Rows from the top: the target, the layers from the last down, the tokens
    const rowHeight = (layer: number) => (lines[layer] === 0 ? EMPTY_ROW : lines[layer] * (NODE + GAP) + GAP + 6)
    const rowY: Record<number, number> = {}
    let y = TOP
    rowY[layers] = y
    y += TARGET_ROW
    for (let layer = layers - 1; layer >= 0; layer--) {
      rowY[layer] = y
      y += rowHeight(layer)
    }
    rowY[-1] = y
    const totalHeight = y + TOKEN_ROW + 4

    const placed = new Map<number, Placed>()
    for (const [key, list] of cells) {
      const [layer, position] = key.split(':').map(Number)
      const count = Math.min(PER_LINE, list.length)
      const blockWidth = count * (NODE + GAP) - GAP
      const startX = columnX[position] + (columnWidth[position] - blockWidth) / 2
      const cellLines = Math.ceil(list.length / PER_LINE)
      // The most influential nodes sit on the cell's top line
      list.forEach((index, i) => {
        const line = Math.floor(i / PER_LINE)
        placed.set(index, {
          index,
          node: graph.nodes[index],
          x: startX + (i % PER_LINE) * (NODE + GAP) + NODE / 2,
          y: rowY[layer] + rowHeight(layer) / 2 - (cellLines * (NODE + GAP) - GAP) / 2 + line * (NODE + GAP) + NODE / 2,
        })
      })
    }
    graph.nodes.forEach((node, index) => {
      if (!shown.has(index)) return
      if (node.kind === 'token') placed.set(index, { index, node, x: columnX[node.position] + columnWidth[node.position] / 2, y: rowY[-1] + 12 })
      else if (node.kind === 'logit') {
        placed.set(index, { index, node, x: columnX[positions - 1] + columnWidth[positions - 1] / 2, y: rowY[layers] + TARGET_ROW / 2 - 2 })
      }
    })
    const maxInfluence = Math.max(1e-9, ...[...placed.values()].filter((p) => p.node.kind === 'feature' || p.node.kind === 'error').map((p) => p.node.influence))
    return { placed, columnX, columnWidth, rowY, rowHeight, totalWidth, totalHeight, maxInfluence }
  }, [graph, shown, width, layers, positions])

  const focus = hover?.index ?? (selected ? graph.nodes.findIndex((n) => n.id === selected) : -1)

  // The edges to draw: the focused node's, or the strongest overall (they arrive sorted by share)
  const edges = useMemo(() => {
    const { source, target, weight, share } = graph.edges
    const out: { from: Placed; to: Placed; weight: number; share: number }[] = []
    for (let i = 0; i < source.length; i++) {
      if (focus >= 0 ? source[i] !== focus && target[i] !== focus : out.length >= OVERVIEW_EDGES) continue
      const from = layout.placed.get(source[i])
      const to = layout.placed.get(target[i])
      if (from && to) out.push({ from, to, weight: weight[i], share: share[i] })
    }
    return out
  }, [graph, layout, focus])
  const maxShare = Math.max(1e-9, ...edges.map((e) => e.share))

  // Rows from the embeddings (-1) to the target (the layer count)
  const rows = Array.from({ length: layers + 2 }, (_, i) => i - 1)
  const rowHeightOf = (row: number) => (row < 0 ? TOKEN_ROW : row >= layers ? TARGET_ROW : layout.rowHeight(row))
  const rowIsEmpty = (row: number) => row >= 0 && row < layers && layout.rowHeight(row) === EMPTY_ROW

  const pinnedSet = new Set(pinned)
  const hovered = hover ? graph.nodes[hover.index] : null
  const tooltipRows = (node: GraphNode): TooltipRow[] => {
    if (node.kind === 'logit') return [{ label: 'probability', value: percent(node.prob ?? 0) }]
    const rows: TooltipRow[] = [{ label: 'of the target’s attribution', value: percent(node.influence) }]
    if (node.kind !== 'token') rows.push({ label: node.support >= 0 ? 'supports the target' : 'opposes the target', value: signedPercent(node.support) })
    if (node.kind === 'feature') {
      rows.push({ label: 'activation', value: (node.activation ?? 0).toFixed(2) })
      if (node.promotes?.length) rows.push({ label: 'promotes', value: node.promotes.slice(0, 3).map(showToken).join(' ') })
    }
    return rows
  }

  const over = (index: number) => (e: PointerEvent<SVGElement>) => {
    const svg = e.currentTarget.ownerSVGElement
    if (!svg) return
    const pointer = pointerIn(e, svg)
    setHover({ index, px: pointer.x, py: pointer.y })
  }
  const pick = (node: GraphNode) => () => onSelect(node.id === selected ? null : node.id)

  return (
    <div ref={ref} className="pl-chart ct-graph">
      {/* The rows' names stay in view while a long prompt's graph scrolls sideways */}
      <div className="ct-graph__rows" style={{ width: LEFT, height: layout.totalHeight, marginBottom: -layout.totalHeight }}>
        <svg width={LEFT} height={layout.totalHeight} aria-hidden="true">
          {rows.map((row) =>
            rowIsEmpty(row) ? null : (
              <text key={row} x={LEFT - 8} y={layout.rowY[row] + rowHeightOf(row) / 2} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
                {rowLabel(row, layers)}
              </text>
            ),
          )}
        </svg>
      </div>
      <svg width={layout.totalWidth} height={layout.totalHeight} role="img" aria-label="Attribution graph: layers up, token positions across" onPointerLeave={() => setHover(null)}>
        {/* A rule under each row */}
        {rows.map((row) => (
          <line key={row} x1={LEFT} x2={layout.totalWidth - 8} y1={layout.rowY[row] + rowHeightOf(row)} y2={layout.rowY[row] + rowHeightOf(row)} stroke={colors.grid} />
        ))}
        {layout.columnX.slice(1).map((x) => (
          <line key={x} x1={x} x2={x} y1={TOP} y2={layout.totalHeight - 4} stroke={colors.grid} />
        ))}

        {edges.map((edge, i) => {
          const mid = (edge.from.y + edge.to.y) / 2
          const strength = Math.sqrt(edge.share / maxShare)
          return (
            <path
              key={i}
              d={`M${edge.from.x},${edge.from.y} C${edge.from.x},${mid} ${edge.to.x},${mid} ${edge.to.x},${edge.to.y}`}
              fill="none"
              stroke={edge.weight >= 0 ? colors.classes[1] : colors.classes[0]}
              strokeWidth={0.75 + 2.25 * strength}
              strokeOpacity={focus >= 0 ? 0.3 + 0.6 * strength : 0.14 + 0.5 * strength}
              pointerEvents="none"
            />
          )
        })}

        {[...layout.placed.values()].map(({ index, node, x, y }) => {
          const isSelected = node.id === selected
          const isPinned = pinnedSet.has(node.id)
          if (node.kind === 'token' || node.kind === 'logit') {
            const text = showToken(node.token)
            // As many characters as the column holds (the target's box may run wider)
            const room = node.kind === 'logit' ? 12 : Math.max(3, Math.floor((layout.columnWidth[node.position] - 6) / 6.6))
            const label = text.length > room ? `${text.slice(0, room - 1)}…` : text
            const boxWidth = Math.max(26, label.length * 6.4 + (node.kind === 'logit' ? 12 : 6))
            return (
              <g key={index} onPointerMove={over(index)} onClick={pick(node)} style={{ cursor: 'pointer' }}>
                <rect
                  x={x - boxWidth / 2}
                  y={y - 9}
                  width={boxWidth}
                  height={18}
                  fill={node.kind === 'logit' ? colors.text : isSelected ? colors.hover : 'transparent'}
                  stroke={isSelected ? colors.text : node.kind === 'logit' ? colors.text : 'none'}
                  strokeWidth={isSelected ? 2 : 1}
                />
                <text x={x} y={y} dy="0.34em" textAnchor="middle" fontSize={11} fontFamily="var(--oz-mono)" fill={node.kind === 'logit' ? colors.surface : colors.text}>
                  {label}
                </text>
              </g>
            )
          }
          const strength = Math.sqrt(node.influence / layout.maxInfluence)
          const feature = node.kind === 'feature'
          return (
            <g key={index} onPointerMove={over(index)} onClick={pick(node)} style={{ cursor: 'pointer' }}>
              {isPinned && <rect x={x - NODE / 2 - 3} y={y - NODE / 2 - 3} width={NODE + 6} height={NODE + 6} fill="none" stroke={colors.accent} strokeWidth={1.5} />}
              <rect
                x={x - NODE / 2}
                y={y - NODE / 2}
                width={NODE}
                height={NODE}
                fill={feature ? (node.support >= 0 ? colors.classes[1] : colors.classes[0]) : colors.surface}
                fillOpacity={feature ? 0.22 + 0.78 * strength : 1}
                stroke={isSelected ? colors.text : feature ? 'none' : colors.dim}
                strokeWidth={isSelected ? 2 : 1}
                strokeDasharray={feature || isSelected ? undefined : '2.5 2'}
              />
              {isSelected && <rect x={x - NODE / 2 - 2.5} y={y - NODE / 2 - 2.5} width={NODE + 5} height={NODE + 5} fill="none" stroke={colors.text} strokeWidth={1} />}
            </g>
          )
        })}
      </svg>
      {hover && hovered && (
        <ChartTooltip x={hover.px} y={hover.py} containerWidth={Math.max(width, layout.totalWidth)} title={nodeTitle(hovered, graph)} rows={tooltipRows(hovered)} />
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.classes[1] }} />
          Feature, supports the target
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.classes[0] }} />
          Opposes it
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ border: `1px dashed ${colors.dim}`, boxSizing: 'border-box' }} />
          Error
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-line" style={{ background: colors.classes[1] }} />
          Edge raises its target
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-line" style={{ background: colors.classes[0] }} />
          Lowers it
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ border: `1.5px solid ${colors.accent}`, boxSizing: 'border-box' }} />
          Pinned
        </span>
      </div>
    </div>
  )
}

/** The row a node is on, for callers that list nodes by row. */
export function rowOf(node: GraphNode, layers: number): string {
  return rowLabel(nodeRow(node, layers), layers)
}
