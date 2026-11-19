/**
 * The Verify view: exact activation patching of a sweep's cells, next to the sweep's
 * estimates. Attribution patching misses and misjudges cells, so nothing counts until it's
 * patched here.
 */

import { useState, type PointerEvent } from 'react'
import type { AttributionLab, VerifyMode, VerifyResult } from '../../hooks/useAttributionLab'
import { VERIFY_COUNTS } from '../../hooks/useAttributionLab'
import type { VerifiedCell } from '../../types/attribution'
import { ChartTooltip, Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { KIND_LABELS, cellName, effect, effectUnit, estimateText, gridScale, tokenLabel } from './attributionUtils'
import { EffectHeatmap } from './EffectHeatmap'
import { FigureExport } from './FigureExport'

const MODES: { value: VerifyMode; label: string; title: string }[] = [
  { value: 'top', label: 'Top components', title: 'The sweep’s largest heads and whole attention and MLP layers, on every pair' },
  { value: 'positions', label: 'Top positions', title: 'The largest cells of the Sweep view’s position map' },
  { value: 'heads', label: 'All heads', title: 'Every head, on every pair' },
  { value: 'resid', label: 'Residual stream', title: 'One pair’s residual stream at every layer and position' },
]

// A cell the estimate called small that moved the metric by this much of the gap (or more)
const MISSED = 0.1

function findings(result: VerifyResult): Finding[] {
  const { response } = result
  const out: Finding[] = []
  // Cells that matter and that the estimate put at half their size or less
  const missed = response.normalized
    ? response.cells.filter((c) => c.estimate != null && Math.abs(c.effect) >= MISSED && Math.abs(c.estimate) < Math.abs(c.effect) / 2)
    : []
  const rho = response.rank_correlation
  if (rho != null) {
    const text = `Rank correlation ${rho.toFixed(2)} over ${response.cells.length} cells.`
    if (rho < 0.5) out.push({ level: 'warning', text: `Estimates and exact effects disagree. ${text}` })
    else if (rho >= 0.8 && missed.length === 0) out.push({ level: 'good', text: `Estimates rank cells as exact patching does. ${text}` })
    else out.push({ level: 'info', text })
  }
  if (missed.length > 0) {
    const worst = [...missed].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))[0]
    out.push({
      level: 'warning',
      text: `${missed.length} cell${missed.length === 1 ? '' : 's'} underestimated by half or more (${cellName(worst)}: ${effect(worst.estimate, 2)} estimated, ${effect(worst.effect, 2)} exact).`,
    })
  }
  return out
}

const SIZE = 340
const PAD = { top: 12, right: 14, bottom: 40, left: 52 }

/** Each cell's estimate against its exact effect, with the line where they agree. */
function Scatter({ cells, unit }: { cells: VerifiedCell[]; unit: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(SIZE)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const points = cells.filter((c) => c.estimate != null)
  if (points.length === 0) return <div className="pl-empty">No estimates to compare.</div>

  const size = Math.max(220, Math.min(width, 440))
  const inner = size - PAD.left - PAD.right
  const all = points.flatMap((c) => [c.estimate as number, c.effect])
  let low = Math.min(0, ...all)
  let high = Math.max(0, ...all)
  const pad = (high - low || 1) * 0.06
  low -= pad
  high += pad
  const at = (v: number) => ((v - low) / (high - low)) * inner
  const x = (v: number) => PAD.left + at(v)
  const y = (v: number) => PAD.top + inner - at(v)
  const ticks = [low + pad, 0, high - pad].filter((t, i, list) => list.indexOf(t) === i)

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const pointer = pointerIn(e, e.currentTarget)
    let best = -1
    let distance = 14 ** 2
    points.forEach((c, i) => {
      const d = (x(c.estimate as number) - pointer.x) ** 2 + (y(c.effect) - pointer.y) ** 2
      if (d < distance) {
        best = i
        distance = d
      }
    })
    setHover(best >= 0 ? { index: best, px: pointer.x, py: pointer.y } : null)
  }
  const shown = hover ? points[hover.index] : null

  return (
    <div ref={ref} className="pl-chart">
      <svg width={size} height={inner + PAD.top + PAD.bottom} role="img" aria-label="Attribution estimates against exact patching effects" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <line x1={x(low)} y1={y(0)} x2={x(high)} y2={y(0)} stroke={colors.grid} />
        <line x1={x(0)} y1={y(low)} x2={x(0)} y2={y(high)} stroke={colors.grid} />
        <line x1={x(low)} y1={y(low)} x2={x(high)} y2={y(high)} stroke={colors.baseline} strokeDasharray="4 4" />
        {points.map((c, i) => (
          <g key={i}>
            {c.se > 0 && <line x1={x(c.estimate as number)} x2={x(c.estimate as number)} y1={y(c.effect - c.se)} y2={y(c.effect + c.se)} stroke={colors.series[0]} strokeOpacity={0.45} />}
            <circle cx={x(c.estimate as number)} cy={y(c.effect)} r={hover?.index === i ? 5 : 3.5} fill={colors.series[0]} stroke={colors.surface} strokeWidth={1} />
          </g>
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <text x={x(t)} y={PAD.top + inner + 14} textAnchor="middle" fontSize={10} fill={colors.dim}>
              {effect(t, 2)}
            </text>
            <text x={PAD.left - 7} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
              {effect(t, 2)}
            </text>
          </g>
        ))}
        <text x={PAD.left + inner / 2} y={PAD.top + inner + 32} textAnchor="middle" fontSize={11} fill={colors.dim}>
          Estimate
        </text>
        <text x={14} y={PAD.top + inner / 2} textAnchor="middle" fontSize={11} fill={colors.dim} transform={`rotate(-90 14 ${PAD.top + inner / 2})`}>
          Exact
        </text>
      </svg>
      {hover && shown && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          title={cellName(shown)}
          rows={[
            { label: 'estimate', value: effect(shown.estimate) },
            { label: `exact ± ${effect(shown.se)}`, value: effect(shown.effect) },
            { label: unit, value: `${shown.n} pair${shown.n === 1 ? '' : 's'}` },
          ]}
        />
      )}
    </div>
  )
}

function CellTable({ cells }: { cells: VerifiedCell[] }) {
  const rows = [...cells].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, 40)
  return (
    <div className="at-table-scroll">
      <table className="pl-table at-table">
        <thead>
          <tr>
            <th>Cell</th>
            <th>Estimate</th>
            <th>Exact</th>
            <th>±</th>
            <th>Pairs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c, i) => (
            <tr key={i}>
              <td>{cellName(c)}</td>
              <td>{effect(c.estimate)}</td>
              <td>{effect(c.effect)}</td>
              <td>{effect(c.se)}</td>
              <td>{c.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** A full sweep's cells as [rows][columns] grids of estimates and exact effects. */
function grids(cells: VerifiedCell[], column: (c: VerifiedCell) => number): { estimate: number[][]; exact: number[][] } {
  const estimate: number[][] = []
  const exact: number[][] = []
  for (const cell of cells) {
    ;(estimate[cell.layer] ??= [])[column(cell)] = cell.estimate ?? 0
    ;(exact[cell.layer] ??= [])[column(cell)] = cell.effect
  }
  return { estimate, exact }
}

function Result({ lab, result }: { lab: AttributionLab; result: VerifyResult }) {
  const { response, mode } = result
  const unit = effectUnit(response)
  const run = lab.run && lab.run.run_id === result.runId ? lab.run : null
  const stale = result.runId != null && lab.run != null && lab.run.run_id !== result.runId
  const pair = mode === 'resid' && run && result.pair != null ? run.pairs[result.pair] : null
  const full = mode === 'heads' || (mode === 'resid' && pair != null)
  const maps = full ? grids(response.cells, (c) => (mode === 'heads' ? (c.head ?? 0) : (c.position ?? 0))) : null
  const scale = maps ? gridScale(maps.estimate, maps.exact) : 1
  const columns = maps
    ? mode === 'heads'
      ? maps.exact[0].map((_, head) => String(head))
      : (pair as NonNullable<typeof pair>).clean_tokens.map(tokenLabel)
    : []
  const title = `Attribution vs. exact patching: ${run ? `${run.task_name} · ` : ''}${response.model}`
  const compared = response.cells.filter((c) => c.estimate != null)

  return (
    <div className="pl-result">
      <div className="pl-run-meta">
        {stale && <strong style={{ color: 'var(--oz-warn)' }}>From another sweep · </strong>}
        {MODES.find((m) => m.value === mode)?.label} · {response.model} · {response.num_pairs} pairs · ${response.cost.toFixed(2)}
      </div>
      <StatTiles
        tiles={[
          { label: 'Cells patched', value: String(response.cells.length) },
          {
            label: 'Rank correlation',
            value: response.rank_correlation != null ? response.rank_correlation.toFixed(2) : '–',
            title: 'Spearman’s, between the sweep’s estimates and the exact effects of these cells',
          },
          {
            label: 'Largest exact',
            value: effect(Math.max(...response.cells.map((c) => Math.abs(c.effect))), 2),
            hint: response.normalized ? 'of the gap' : unit,
          },
        ]}
      />
      <Findings items={findings(result)} />

      {maps ? (
        <div className="at-compare">
          {(['estimate', 'exact'] as const).map((which) => (
            <div key={which}>
              <div className="pl-card-title">
                <h3>{which === 'estimate' ? 'Estimate' : 'Exact'}</h3>
                <FigureExport
                  filename={`${which}-${mode}-${response.model}`}
                  figure={() => ({
                    kind: 'heatmap',
                    title: `${which === 'estimate' ? 'Attribution patching' : 'Activation patching'}: ${run ? `${run.task_name} · ` : ''}${response.model}`,
                    values: maps[which],
                    row_labels: maps[which].map((_, layer) => String(layer)),
                    column_labels: columns,
                    x_label: mode === 'heads' ? 'Head' : 'Position',
                    y_label: 'Layer',
                    value_label: `${which === 'estimate' ? 'Estimated' : 'Exact'} effect (${unit})`,
                  })}
                />
              </div>
              <EffectHeatmap
                values={maps[which]}
                scale={scale}
                columnLabels={columns}
                marked={pair ? pair.clean_tokens.map((token, p) => token !== pair.corrupted_tokens[p]) : undefined}
                columnTitle={mode === 'heads' ? 'Head' : 'Position'}
                slanted={mode === 'resid'}
                selected={mode === 'heads' && lab.selectedHead ? { row: lab.selectedHead.layer, column: lab.selectedHead.head } : null}
                onSelect={mode === 'heads' ? ({ row, column }) => lab.setSelectedHead({ layer: row, head: column }) : undefined}
                ariaLabel={`${which === 'estimate' ? 'Estimated' : 'Exact'} effects, layers by ${mode === 'heads' ? 'heads' : 'positions'}`}
                unit={unit}
                tooltip={(layer, column) => ({
                  title: mode === 'heads' ? `Head ${layer}.${column}` : `${KIND_LABELS.resid} L${layer} · ${columns[column]}`,
                  rows: [
                    { label: 'estimate', value: effect(maps.estimate[layer][column]) },
                    { label: 'exact', value: effect(maps.exact[layer][column]) },
                  ],
                })}
              />
            </div>
          ))}
        </div>
      ) : null}

      <div className="at-compare">
        <div>
          <div className="pl-card-title">
            <h3>Estimate against exact</h3>
            <FigureExport
              filename={`attribution-vs-patching-${response.model}`}
              figure={() => ({
                kind: 'scatter',
                title,
                x: compared.map((c) => c.estimate as number),
                y: compared.map((c) => c.effect),
                errors: compared.map((c) => c.se),
                x_label: `Attribution estimate (${unit})`,
                y_label: `Exact patching effect (${unit})`,
                note: response.rank_correlation != null ? `Spearman ${response.rank_correlation.toFixed(2)}, ${compared.length} cells` : undefined,
              })}
            />
          </div>
          <Scatter cells={response.cells} unit={unit} />
        </div>
        <div>
          <div className="pl-card-title">
            <h3>Largest exact effects</h3>
          </div>
          <CellTable cells={response.cells} />
        </div>
      </div>
    </div>
  )
}

export function VerifyView({ lab }: { lab: AttributionLab }) {
  const run = lab.run
  if (!run && !lab.verified) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="patching">Run a sweep, then patch its cells exactly.</Placeholder>
      </section>
    )
  }
  const estimate = lab.verifyEstimate
  const count = lab.verifyCells.length
  const counted = lab.verifyMode === 'top' || lab.verifyMode === 'positions'
  const note = !lab.isAuthenticated ? 'Sign in to run' : !run ? 'Run a sweep first' : estimate ? `${count} cells · ${estimateText(estimate)}` : `${count} cells`

  return (
    <section className="pl-panel" style={{ opacity: lab.running ? 0.55 : 1 }}>
      <div className="pl-controls-row">
        <div data-assist="attribution.verify_cells">
          <div className="pl-control-label">Cells</div>
          <Segmented ariaLabel="Cells to patch" value={lab.verifyMode} onChange={lab.setVerifyMode} disabled={lab.verifying} options={MODES} />
        </div>
        {counted && (
          <div data-assist="attribution.verify_count">
            <div className="pl-control-label">Top</div>
            <Segmented
              ariaLabel="How many cells"
              value={String(lab.verifyCount)}
              onChange={(value) => lab.setVerifyCount(Number(value))}
              disabled={lab.verifying}
              options={VERIFY_COUNTS.map((n) => ({ value: String(n), label: String(n) }))}
            />
          </div>
        )}
      </div>
      {run && lab.verifyMode === 'positions' && lab.scopeInfo && (
        <div className="pl-hint">
          {KIND_LABELS[lab.sites]} · {lab.scopeInfo.pair != null ? `pair ${lab.scopeInfo.pair + 1}` : 'mean of its pairs'} (set in the Sweep view)
        </div>
      )}
      {run && lab.verifyMode === 'resid' && <div className="pl-hint">Pair {lab.residPair + 1} (set in the Sweep view)</div>}
      <RunBar
        note={note}
        warn={estimate != null && !estimate.within_limit}
        label={lab.verifying ? 'Patching…' : 'Verify'}
        running={lab.verifying}
        disabled={!run || count === 0 || (estimate != null && !estimate.within_limit)}
        onRun={lab.verify}
        assist="attribution.verify_button"
      />
      {lab.verified && <Result lab={lab} result={lab.verified} />}
    </section>
  )
}
