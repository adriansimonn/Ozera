/**
 * A heatmap of effects: layers down, positions or heads across, on the labs' diverging scale
 * (blue lowers the metric, red raises it), with a hover readout and an optional selection.
 */

import { useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { ChartTooltip, type TooltipRow } from '../lab/chartKit'
import { divergingColor, pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { effect } from './attributionUtils'

const MARGIN = { top: 10, right: 12, left: 44 }
const MIN_CELL_WIDTH = 16
const MAX_CELL_WIDTH = 56

export interface HeatmapSelection {
  row: number
  column: number
}

interface Props {
  /** [rows][columns] */
  values: number[][]
  /** The value the scale's ends stand for (the largest absolute value shown, usually) */
  scale: number
  columnLabels: string[]
  /** Columns marked under their label (the positions where a pair's prompts differ) */
  marked?: boolean[]
  rowTitle?: string
  columnTitle: string
  /** Rotate the column labels (tokens); numbers stay upright */
  slanted?: boolean
  selected?: HeatmapSelection | null
  onSelect?: (cell: HeatmapSelection) => void
  tooltip: (row: number, column: number) => { title: string; rows: TooltipRow[] }
  ariaLabel: string
  /** Shown in the legend, under the scale */
  unit?: ReactNode
}

export function EffectHeatmap({
  values,
  scale,
  columnLabels,
  marked,
  rowTitle = 'Layer',
  columnTitle,
  slanted = false,
  selected,
  onSelect,
  tooltip,
  ariaLabel,
  unit,
}: Props) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(560)
  const [hover, setHover] = useState<{ row: number; column: number; px: number; py: number } | null>(null)

  const rows = values.length
  const columns = values[0]?.length ?? 0
  if (rows === 0 || columns === 0) return null

  const bottom = slanted ? 70 : 40
  const cellH = Math.max(12, Math.min(22, Math.floor(520 / rows)))
  const available = Math.max(width - MARGIN.left - MARGIN.right, 80)
  const cellW = Math.max(MIN_CELL_WIDTH, Math.min(MAX_CELL_WIDTH, available / columns))
  const innerW = cellW * columns
  const innerH = cellH * rows
  const rowEvery = Math.max(1, Math.ceil(rows / (innerH / 13)))
  const columnEvery = slanted ? 1 : Math.max(1, Math.ceil(columns / (innerW / 22)))
  const color = (value: number) => divergingColor(colors, value / scale)

  const cellAt = (e: PointerEvent<SVGRectElement>) => {
    const pointer = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const column = Math.floor((pointer.x - MARGIN.left) / cellW)
    const row = Math.floor((pointer.y - MARGIN.top) / cellH)
    return row >= 0 && column >= 0 && row < rows && column < columns ? { row, column, px: pointer.x, py: pointer.y } : null
  }

  // Arrow keys move the selection; without one they start from the first cell
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!onSelect) return
    const step = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key]
    if (!step) return
    e.preventDefault()
    const from = selected ?? { row: 0, column: 0 }
    onSelect({
      row: Math.max(0, Math.min(rows - 1, from.row + step[0])),
      column: Math.max(0, Math.min(columns - 1, from.column + step[1])),
    })
  }

  const shown = hover ? tooltip(hover.row, hover.column) : null

  return (
    <div className="at-heatmap">
      <div
        ref={ref}
        className="pl-chart at-heatmap__scroll"
        tabIndex={onSelect ? 0 : undefined}
        onKeyDown={onKeyDown}
        role={onSelect ? 'grid' : undefined}
        aria-label={onSelect ? ariaLabel : undefined}
      >
        <svg width={innerW + MARGIN.left + MARGIN.right} height={innerH + MARGIN.top + bottom} role="img" aria-label={ariaLabel}>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {values.map((line, row) =>
              line.map((value, column) => (
                <rect key={`${row}-${column}`} x={column * cellW} y={row * cellH} width={cellW + 0.5} height={cellH + 0.5} fill={color(value)} />
              )),
            )}
            {selected && selected.row < rows && selected.column < columns && (
              <rect x={selected.column * cellW + 1} y={selected.row * cellH + 1} width={cellW - 2} height={cellH - 2} fill="none" stroke={colors.text} strokeWidth={2} />
            )}
            {hover && <rect x={hover.column * cellW} y={hover.row * cellH} width={cellW} height={cellH} fill="none" stroke={colors.text} strokeWidth={1.5} />}
            {values.map((_, row) =>
              row % rowEvery === 0 ? (
                <text key={row} x={-7} y={(row + 0.5) * cellH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
                  {row}
                </text>
              ) : null,
            )}
            {columnLabels.map((label, column) => {
              if (column % columnEvery !== 0) return null
              const x = (column + 0.5) * cellW
              const y = innerH + 9
              return (
                <g key={column}>
                  {marked?.[column] && <rect x={column * cellW + 2} y={innerH + 2} width={cellW - 4} height={2} fill={colors.text} />}
                  <text
                    x={x}
                    y={slanted ? y + 4 : y + 6}
                    textAnchor={slanted ? 'end' : 'middle'}
                    fontSize={10}
                    fill={marked?.[column] ? colors.text : colors.dim}
                    fontWeight={marked?.[column] ? 600 : 400}
                    transform={slanted ? `rotate(-45 ${x} ${y + 4})` : undefined}
                  >
                    {label.length > 9 ? `${label.slice(0, 8)}…` : label}
                  </text>
                </g>
              )
            })}
            <text x={-34} y={innerH / 2} textAnchor="middle" fontSize={11} fill={colors.dim} transform={`rotate(-90 -34 ${innerH / 2})`}>
              {rowTitle}
            </text>
            <text x={innerW / 2} y={innerH + bottom - 6} textAnchor="middle" fontSize={11} fill={colors.dim}>
              {columnTitle}
            </text>
          </g>
          <rect
            x={MARGIN.left}
            y={MARGIN.top}
            width={innerW}
            height={innerH}
            fill="transparent"
            style={onSelect ? { cursor: 'pointer' } : undefined}
            onPointerMove={(e) => setHover(cellAt(e))}
            onPointerLeave={() => setHover(null)}
            onClick={(e) => {
              const cell = cellAt(e as unknown as PointerEvent<SVGRectElement>)
              if (cell && onSelect) onSelect({ row: cell.row, column: cell.column })
            }}
          />
        </svg>
        {hover && shown && <ChartTooltip x={hover.px} y={hover.py} containerWidth={Math.max(width, innerW)} title={shown.title} rows={shown.rows} />}
      </div>
      <div className="at-ramp">
        <span>{effect(-scale, 2)}</span>
        <span
          className="at-ramp__bar"
          style={{ background: `linear-gradient(to right, ${color(-scale)}, ${color(-scale / 2)}, ${color(0)}, ${color(scale / 2)}, ${color(scale)})` }}
        />
        <span>+{effect(scale, 2)}</span>
        {unit && <span className="at-ramp__unit">{unit}</span>}
      </div>
    </div>
  )
}
