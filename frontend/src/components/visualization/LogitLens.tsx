/**
 * Logit lens: what the model would predict if it stopped after each layer.
 *
 * A grid of layers (down) by positions (across). Each cell shows the top token the residual
 * stream there decodes to, colored by the probability it gives one token: the one that came
 * next in the sequence, or the model's final prediction. The last row is the model's output.
 * Only the columns in view are drawn, so long generations stay light.
 */

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type { LogitLens as LogitLensData } from '../../types/model'
import { ExportButton } from '../export'
import { ChartTooltip, LabKitStyles, Segmented } from '../lab/chartKit'
import { pointerIn, useElementWidth } from '../lab/labTheme'
import { useTheme } from '../../hooks/useTheme'
import { inkOnSequential, sequentialScale } from '../../styles/appTheme'
import { lensRowLabel, percent, showToken } from './tokenText'

export type LensColorBy = 'target' | 'final'

interface LogitLensProps {
  lens: LogitLensData
  /** The sequence's decoded tokens */
  tokens: string[]
  /** Tokens before the first generated one */
  promptTokens: number
  selectedPosition: number
  onSelectPosition: (position: number) => void
  activationId?: string
}

const CELL_W = 78
const CELL_H = 26
const HEAD_H = 46
// The hover readout's height, to keep it inside the grid near its last rows (which matter most)
const TIP_H = 200
// Columns drawn past each edge of the view
const OVERSCAN = 3

interface Hover {
  row: number
  position: number
  x: number
  y: number
}

export const LogitLens = memo(function LogitLens({
  lens,
  tokens,
  promptTokens,
  selectedPosition,
  onSelectPosition,
  activationId,
}: LogitLensProps) {
  const { isLight } = useTheme()
  const [colorBy, setColorBy] = useState<LensColorBy>('target')
  const [hover, setHover] = useState<Hover | null>(null)
  const [scrollLeft, setScrollLeft] = useState(0)
  const gridRef = useRef<HTMLDivElement>(null)
  const { ref: scrollRef, width: viewWidth } = useElementWidth<HTMLDivElement>(900)

  const { rows, seq_len: seqLen, k } = lens
  const topIds = lens.top_ids.values as number[][][]
  const topProbs = lens.top_probs.values as number[][][]
  const probs = (colorBy === 'target' ? lens.target_prob : lens.final_prob).values as number[][]
  const text = (id: number) => showToken(lens.tokens[String(id)])

  // A new generation starts at its first column
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
    setScrollLeft(0)
    setHover(null)
  }, [lens, scrollRef])

  const first = Math.max(0, Math.floor(scrollLeft / CELL_W) - OVERSCAN)
  const last = Math.min(seqLen - 1, Math.ceil((scrollLeft + viewWidth) / CELL_W) + OVERSCAN)

  const cells = useMemo(() => {
    const scale = sequentialScale(isLight)
    const columns = []
    for (let position = first; position <= last; position++) {
      const selected = position === selectedPosition
      columns.push(
        <div
          key={position}
          className={`ll-column${selected ? ' is-selected' : ''}${position === promptTokens ? ' is-first-generated' : ''}`}
          style={{ left: position * CELL_W }}
        >
          <button
            type="button"
            className="ll-head"
            aria-pressed={selected}
            title={`Token ${position}: "${tokens[position] ?? ''}"`}
            onClick={() => onSelectPosition(position)}
          >
            <span className="ll-head-index">{position}</span>
            <span className="ll-head-token">{showToken(tokens[position])}</span>
          </button>
          {Array.from({ length: rows }, (_, row) => {
            const p = probs[row][position]
            return (
              <div
                key={row}
                className="ll-cell"
                data-row={row}
                data-position={position}
                style={{ background: scale(p), color: inkOnSequential(isLight, p) }}
              >
                {showToken(lens.tokens[String(topIds[row][position][0])])}
              </div>
            )
          })}
        </div>,
      )
    }
    return columns
  }, [first, last, rows, probs, topIds, lens.tokens, tokens, selectedPosition, promptTokens, isLight, onSelectPosition])

  // One handler for every cell: the cell under the pointer names its row and position
  const cellAt = (target: EventTarget | null): { row: number; position: number } | null => {
    const cell = target instanceof HTMLElement ? target.closest<HTMLElement>('.ll-cell') : null
    if (!cell) return null
    return { row: Number(cell.dataset.row), position: Number(cell.dataset.position) }
  }

  const scale = sequentialScale(isLight)
  const targetIsPrediction = hover != null && hover.position === seqLen - 1

  return (
    <div className="logit-lens lab-kit">
      <div className="ll-bar">
        <div className="ll-bar-group">
          <span className="ll-label">Color</span>
          <Segmented<LensColorBy>
            value={colorBy}
            onChange={setColorBy}
            ariaLabel="Color cells by"
            options={[
              { value: 'target', label: 'Next token', title: 'The probability each layer gives the token that came next' },
              { value: 'final', label: 'Final prediction', title: "The probability each layer gives the model's final top prediction" },
            ]}
          />
        </div>
        <div className="ll-bar-group">
          <div className="ll-ramp-legend">
            <span>0%</span>
            <span
              className="ll-ramp"
              style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map((t) => scale(t)).join(', ')})` }}
            />
            <span>100%</span>
          </div>
          {/* Outside the legend: the export dialog opens inside its button's parent */}
          {activationId && (
            <ExportButton
              exportType="logit-lens"
              activationId={activationId}
              colorBy={colorBy}
              position={selectedPosition}
              positions={seqLen}
            />
          )}
        </div>
      </div>

      <div className="ll-grid" ref={gridRef}>
        <div className="ll-rows" aria-hidden="true">
          <div className="ll-rows-head" style={{ height: HEAD_H }} />
          {Array.from({ length: rows }, (_, row) => (
            <div key={row} className={`ll-row-label${hover?.row === row ? ' is-hovered' : ''}`} style={{ height: CELL_H }}>
              {lensRowLabel(row)}
            </div>
          ))}
        </div>
        <div
          className="ll-scroll"
          ref={scrollRef}
          onScroll={(e) => {
            setScrollLeft(e.currentTarget.scrollLeft)
            setHover(null)
          }}
        >
          <div
            className="ll-sheet"
            style={{ width: seqLen * CELL_W, height: HEAD_H + rows * CELL_H }}
            onMouseMove={(e) => {
              const cell = cellAt(e.target)
              if (!cell || !gridRef.current) return setHover(null)
              // The pointer in the grid's own pixels (the page is zoomed, see pointerIn)
              setHover({ ...cell, ...pointerIn(e, gridRef.current) })
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const cell = cellAt(e.target)
              if (cell) onSelectPosition(cell.position)
            }}
          >
            {cells}
          </div>
        </div>

        {hover && (
          <ChartTooltip
            x={hover.x}
            y={hover.y + TIP_H > (gridRef.current?.offsetHeight ?? Infinity) ? hover.y - TIP_H : hover.y}
            containerWidth={gridRef.current?.offsetWidth ?? viewWidth}
            title={`${lensRowLabel(hover.row)} · token ${hover.position} ${showToken(tokens[hover.position])}`}
            rows={Array.from({ length: k }, (_, i) => ({
              value: percent(topProbs[hover.row][hover.position][i]),
              label: text(topIds[hover.row][hover.position][i]),
            }))}
          >
            <div className="ll-tip-rule" />
            <div className="pl-tooltip-row">
              <span className="pl-tooltip-value">{percent((lens.target_prob.values as number[][])[hover.row][hover.position])}</span>
              <span className="pl-tooltip-label">
                {targetIsPrediction ? 'final' : 'next'} {text(lens.target_ids[hover.position])} · rank{' '}
                {(lens.target_rank.values as number[][])[hover.row][hover.position]}
              </span>
            </div>
            {!targetIsPrediction && (
              <div className="pl-tooltip-row">
                <span className="pl-tooltip-value">{percent((lens.final_prob.values as number[][])[hover.row][hover.position])}</span>
                <span className="pl-tooltip-label">
                  final {text(lens.final_ids[hover.position])} · rank {(lens.final_rank.values as number[][])[hover.row][hover.position]}
                </span>
              </div>
            )}
          </ChartTooltip>
        )}
      </div>

      <LabKitStyles />
      <style>{`
        .logit-lens { padding: 0; }

        .logit-lens .ll-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 0.75rem 1.5rem;
          padding: 0.9rem var(--oz-gutter);
          border-bottom: 1px solid var(--oz-rule);
        }
        .logit-lens .ll-bar-group { display: flex; align-items: center; gap: 0.75rem; }
        .logit-lens .ll-label { color: var(--oz-fg-dim); font-size: 0.85rem; font-weight: 500; }
        .logit-lens .ll-ramp-legend {
          display: flex; align-items: center; gap: 0.5rem;
          color: var(--oz-fg-dim); font-family: var(--oz-mono); font-size: 0.75rem;
        }
        .logit-lens .ll-ramp { width: 140px; height: 8px; border: 1px solid var(--oz-rule-strong); }

        .logit-lens .ll-grid { position: relative; display: flex; padding: 1rem 0 1.25rem var(--oz-gutter); }

        .logit-lens .ll-rows { flex-shrink: 0; width: 3.25rem; }
        .logit-lens .ll-row-label {
          display: flex;
          align-items: center;
          color: var(--oz-fg-dim);
          font-family: var(--oz-mono);
          font-size: 0.75rem;
        }
        .logit-lens .ll-row-label.is-hovered { color: var(--oz-fg); }

        .logit-lens .ll-scroll { flex: 1; min-width: 0; overflow-x: auto; overflow-y: hidden; padding-bottom: 0.5rem; }
        .logit-lens .ll-sheet { position: relative; }
        .logit-lens .ll-column { position: absolute; top: 0; width: ${CELL_W}px; }

        /* The first generated token: everything left of this rule is the prompt */
        .logit-lens .ll-column.is-first-generated::before {
          content: '';
          position: absolute;
          top: 0;
          bottom: 0;
          left: -1px;
          border-left: 1px dashed var(--oz-fg-faint);
        }

        .logit-lens .ll-head {
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          gap: 1px;
          width: 100%;
          height: ${HEAD_H}px;
          padding: 0 5px 6px;
          background: none;
          border: none;
          border-bottom: 1px solid transparent;
          font-family: var(--oz-mono);
          text-align: left;
          cursor: pointer;
          transition: color 0.15s ease;
        }
        .logit-lens .ll-head-index { color: var(--oz-fg-faint); font-size: 0.68rem; }
        .logit-lens .ll-head-token { font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .logit-lens .ll-head:hover { color: var(--oz-mark); }
        .logit-lens .ll-column.is-selected .ll-head { border-bottom-color: var(--oz-fg); }
        .logit-lens .ll-column.is-selected .ll-head-index { color: var(--oz-fg); }

        .logit-lens .ll-cell {
          height: ${CELL_H}px;
          padding: 0 5px;
          border-right: 1px solid var(--oz-bg);
          border-bottom: 1px solid var(--oz-bg);
          font-family: var(--oz-mono);
          font-size: 0.75rem;
          line-height: ${CELL_H - 1}px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          cursor: pointer;
        }
        .logit-lens .ll-cell:hover { box-shadow: inset 0 0 0 1.5px var(--oz-fg); }
        .logit-lens .ll-column.is-selected .ll-cell { border-right-color: var(--oz-fg); box-shadow: inset 1px 0 0 var(--oz-fg); }
        .logit-lens .ll-column.is-selected .ll-cell:hover { box-shadow: inset 0 0 0 1.5px var(--oz-fg); }

        .logit-lens .ll-tip-rule { height: 1px; margin: 0.35rem 0; background: var(--oz-rule); }
        .logit-lens .pl-tooltip { max-width: 300px !important; }
        .logit-lens .pl-tooltip-label { font-family: var(--oz-mono); overflow: hidden; text-overflow: ellipsis; }
      `}</style>
    </div>
  )
})
