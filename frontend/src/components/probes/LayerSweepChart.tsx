/**
 * Layer sweep: each method's AUROC (or accuracy) at every residual stream position, against
 * the baselines that say whether to believe it (the shuffled-label control task, the probe on
 * the embeddings, and chance or the majority class). Click or use arrow keys to pick a layer.
 */

import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { PROBE_METHODS, type PerPosition, type Pooling, type ProbeMethod, type ProbeRunResult } from '../../types/probes'
import { ChartTooltip, Segmented, type TooltipRow } from './chartKit'
import { useElementWidth, useProbeColors } from './probeTheme'
import {
  METHOD_LABELS,
  METHOD_SHORT,
  formatMetric,
  formatPercent,
  metricKey,
  metricValues,
  positionLabel,
  type EvalSet,
  type SweepMetric,
} from './probeUtils'

interface LayerSweepChartProps {
  run: ProbeRunResult
  pooling: Pooling
  method: ProbeMethod
  metric: SweepMetric
  evalSet: EvalSet
  position: number
  onSelectPosition: (position: number) => void
}

const HEIGHT = 300
const MARGIN = { top: 28, right: 20, bottom: 40, left: 48 }

function linePath(values: PerPosition, x: (i: number) => number, y: (v: number) => number): string {
  let path = ''
  let pen = false
  values.forEach((v, i) => {
    if (v == null) {
      pen = false
      return
    }
    path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
    pen = true
  })
  return path
}

export function LayerSweepChart({ run, pooling, method, metric, evalSet, position, onSelectPosition }: LayerSweepChartProps) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const [view, setView] = useState<'chart' | 'table'>('chart')

  const positions = run.num_layers + 1
  const key = metricKey(metric, evalSet)
  const series = PROBE_METHODS.map((m) => ({ method: m, values: metricValues(run, pooling, m, key) }))
  const selected = series.find((s) => s.method === method)!
  const control =
    evalSet === 'test'
      ? metricValues(run, pooling, method, metric === 'auroc' ? 'control_test_auroc' : 'control_test_acc')
      : null
  const embedding = selected.values[0]
  const chance = metric === 'auroc' ? 0.5 : evalSet === 'test' ? run.majority.test_acc : run.majority.ood_acc
  const chanceLabel = metric === 'auroc' ? 'chance' : 'majority class'

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 50)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => MARGIN.left + (positions > 1 ? i * step : innerW / 2)

  const all = [...series.flatMap((s) => s.values), ...(control ?? []), chance].filter((v): v is number => v != null)
  const yMin = Math.min(0.5, Math.floor(Math.min(...all) * 10) / 10)
  const yMax = 1
  const y = (v: number) => MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * innerH
  const tickStep = yMax - yMin > 0.6 ? 0.2 : 0.1
  const yTicks: number[] = []
  for (let t = yMax; t >= yMin - 1e-9; t -= tickStep) yTicks.push(Math.round(t * 100) / 100)
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))

  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - MARGIN.left) / step)))

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    setHover({ index: indexAt(px), px, py: e.clientY - box.top })
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft' && position > 0) onSelectPosition(position - 1)
    else if (e.key === 'ArrowRight' && position < positions - 1) onSelectPosition(position + 1)
    else return
    e.preventDefault()
  }

  // Labels drawn over lines get a ring of the surface color so they stay legible
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }
  const metricName = metric === 'auroc' ? 'AUROC' : 'Accuracy'
  const setName = evalSet === 'test' ? 'test' : 'OOD'
  const tooltipRows = (i: number): TooltipRow[] => [
    ...series.map((s) => ({ key: colors.methods[s.method], label: METHOD_LABELS[s.method], value: formatMetric(s.values[i]) })),
    ...(control ? [{ key: colors.baseline, label: `control (${METHOD_SHORT[method]})`, value: formatMetric(control[i]) }] : []),
  ]

  return (
    <div className="pl-sweep">
      <div className="pl-card-title">
        <h3>
          {metricName} by layer · {setName}
        </h3>
        <Segmented
          ariaLabel="Chart or table"
          value={view}
          onChange={setView}
          options={[
            { value: 'chart', label: 'Chart' },
            { value: 'table', label: 'Table' },
          ]}
        />
      </div>

      {view === 'chart' ? (
        <div
          ref={ref}
          className="pl-chart"
          tabIndex={0}
          role="group"
          aria-label={`Layer sweep chart. Selected: ${position === 0 ? 'embeddings' : `layer ${position - 1}`}. Use arrow keys to change layer.`}
          onKeyDown={onKeyDown}
        >
          <svg width={width} height={HEIGHT} role="img" aria-hidden="true">
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} strokeWidth={1} />
                <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={MARGIN.top + innerH} y2={MARGIN.top + innerH} stroke={colors.axis} />
            {Array.from({ length: positions }, (_, i) =>
              i % labelEvery === 0 || i === positions - 1 ? (
                <text key={i} x={x(i)} y={MARGIN.top + innerH + 16} textAnchor="middle" fontSize={11} fill={colors.text}>
                  {positionLabel(i)}
                </text>
              ) : null,
            )}
            <text x={MARGIN.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
              Layer
            </text>

            {/* Selected layer */}
            <rect
              x={x(position) - Math.max(step, 8) / 2}
              y={MARGIN.top}
              width={Math.max(step, 8)}
              height={innerH}
              fill={colors.hover}
            />
            <text x={x(position)} y={MARGIN.top - 10} textAnchor="middle" fontSize={11} fontWeight={600} fill={colors.text} {...halo}>
              {position === 0 ? 'embeddings' : `layer ${position - 1}`}
            </text>

            {/* Baselines */}
            {chance != null && (
              <g>
                <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(chance)} y2={y(chance)} stroke={colors.baseline} strokeWidth={1} />
                <text x={MARGIN.left + innerW} y={y(chance) - 4} textAnchor="end" fontSize={10} fill={colors.text} {...halo}>
                  {chanceLabel} {formatMetric(chance, 2)}
                </text>
              </g>
            )}
            {embedding != null && (
              <g>
                <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(embedding)} y2={y(embedding)} stroke={colors.baseline} strokeWidth={1} opacity={0.7} />
                <text
                  x={MARGIN.left + 4}
                  y={y(embedding) - MARGIN.top < 16 ? y(embedding) + 13 : y(embedding) - 4}
                  fontSize={10}
                  fill={colors.text}
                  {...halo}
                >
                  embeddings {formatMetric(embedding, 2)}
                </text>
              </g>
            )}
            {control && (
              <path d={linePath(control, x, y)} fill="none" stroke={colors.baseline} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
            )}

            {/* Methods: the selected one drawn last, on top */}
            {[...series.filter((s) => s.method !== method), selected].map((s) => (
              <path
                key={s.method}
                d={linePath(s.values, x, y)}
                fill="none"
                stroke={colors.methods[s.method]}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={s.method === method ? 1 : 0.55}
              />
            ))}
            {series.map((s) =>
              s.values[position] != null ? (
                <circle
                  key={s.method}
                  cx={x(position)}
                  cy={y(s.values[position] as number)}
                  r={4}
                  fill={colors.methods[s.method]}
                  stroke={colors.surface}
                  strokeWidth={2}
                />
              ) : null,
            )}

            {hover && (
              <line x1={x(hover.index)} x2={x(hover.index)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} strokeWidth={1} />
            )}
            <rect
              x={MARGIN.left - step / 2}
              y={MARGIN.top}
              width={innerW + step}
              height={innerH}
              fill="transparent"
              style={{ cursor: 'pointer' }}
              onPointerMove={onPointerMove}
              onPointerLeave={() => setHover(null)}
              onClick={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                onSelectPosition(indexAt(e.clientX - box.left))
              }}
            />
          </svg>
          {hover && (
            <ChartTooltip
              x={hover.px}
              y={hover.py}
              containerWidth={width}
              title={hover.index === 0 ? 'Embeddings' : `Layer ${hover.index - 1}`}
              rows={tooltipRows(hover.index)}
            />
          )}
        </div>
      ) : (
        <SweepTable run={run} pooling={pooling} method={method} evalSet={evalSet} position={position} onSelectPosition={onSelectPosition} />
      )}

      <div className="pl-legend">
        {PROBE_METHODS.map((m) => (
          <span key={m} className="pl-legend-item">
            <span className="pl-legend-line" style={{ background: colors.methods[m], opacity: m === method ? 1 : 0.55 }} />
            {METHOD_LABELS[m]}
          </span>
        ))}
        {control && (
          <span className="pl-legend-item">
            <span className="pl-legend-line" style={{ background: colors.baseline }} />
            Control (shuffled labels)
          </span>
        )}
      </div>
    </div>
  )
}

function SweepTable({
  run,
  pooling,
  method,
  evalSet,
  position,
  onSelectPosition,
}: {
  run: ProbeRunResult
  pooling: Pooling
  method: ProbeMethod
  evalSet: EvalSet
  position: number
  onSelectPosition: (position: number) => void
}) {
  const colors = useProbeColors()
  const metrics = (m: ProbeMethod) => run.poolings[pooling].methods[m].metrics
  const hasOod = run.dataset.n_ood > 0
  const columns: { label: string; values: PerPosition; percent?: boolean }[] = [
    { label: 'LR AUROC', values: metrics('logreg').test_auroc ?? [] },
    { label: 'DiM AUROC', values: metrics('diff_means').test_auroc ?? [] },
    ...(hasOod
      ? [
          { label: 'LR OOD', values: metrics('logreg').ood_auroc ?? [] },
          { label: 'DiM OOD', values: metrics('diff_means').ood_auroc ?? [] },
        ]
      : []),
    { label: `${METHOD_SHORT[method]} acc`, values: metrics(method).test_acc ?? [], percent: true },
    { label: 'Control acc', values: metrics(method).control_test_acc ?? [], percent: true },
    { label: 'Selectivity', values: metrics(method).selectivity ?? [], percent: true },
    { label: 'Ctrl train acc', values: metrics(method).control_train_acc ?? [], percent: true },
  ]
  return (
    <div style={{ maxHeight: 340, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
      <table className="pl-table" aria-label={`Per-layer metrics (${evalSet === 'test' ? 'test set' : 'OOD'})`}>
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th>Layer</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: run.num_layers + 1 }, (_, i) => (
            <tr
              key={i}
              onClick={() => onSelectPosition(i)}
              style={{ background: i === position ? colors.hover : undefined, borderTop: `1px solid ${colors.grid}` }}
            >
              <td style={{ fontWeight: i === position ? 600 : 400 }}>{i === 0 ? 'emb' : positionLabel(i)}</td>
              {columns.map((c) => (
                <td key={c.label}>
                  {c.percent ? formatPercent(c.values[i]) : formatMetric(c.values[i])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The controls above the sweep chart and every view below it. */
export function SweepControls({
  pooling,
  method,
  metric,
  evalSet,
  hasOod,
  onPooling,
  onMethod,
  onMetric,
  onEvalSet,
}: {
  pooling: Pooling
  method: ProbeMethod
  metric: SweepMetric
  evalSet: EvalSet
  hasOod: boolean
  onPooling: (p: Pooling) => void
  onMethod: (m: ProbeMethod) => void
  onMetric: (m: SweepMetric) => void
  onEvalSet: (e: EvalSet) => void
}) {
  return (
    <div className="pl-controls-row">
      <div>
        <div className="pl-control-label">Pooling</div>
        <Segmented
          ariaLabel="Pooling"
          value={pooling}
          onChange={onPooling}
          options={[
            { value: 'last', label: 'Last token' },
            { value: 'mean', label: 'Mean' },
            { value: 'max', label: 'Max' },
          ]}
        />
      </div>
      <div>
        <div className="pl-control-label">Method</div>
        <Segmented
          ariaLabel="Probe method"
          value={method}
          onChange={onMethod}
          options={[
            { value: 'logreg', label: 'Logistic regression' },
            { value: 'diff_means', label: 'Diff. in means' },
          ]}
        />
      </div>
      <div>
        <div className="pl-control-label">Metric</div>
        <Segmented
          ariaLabel="Metric"
          value={metric}
          onChange={onMetric}
          options={[
            { value: 'auroc', label: 'AUROC' },
            { value: 'acc', label: 'Accuracy' },
          ]}
        />
      </div>
      <div>
        <div className="pl-control-label">Set</div>
        <Segmented
          ariaLabel="Evaluation set"
          value={evalSet}
          onChange={onEvalSet}
          options={[
            { value: 'test', label: 'Test' },
            { value: 'ood', label: 'OOD', disabled: !hasOod, title: hasOod ? undefined : 'This run has no OOD test set' },
          ]}
        />
      </div>
    </div>
  )
}
