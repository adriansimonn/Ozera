/**
 * The Layers view: each hypothesis's held-out R² at every layer against its shuffled-value
 * baseline and the embeddings, like the Probe Lab's layer sweep, plus neighbor order, the class
 * means' dimension, and how much of the variance is between values. Click or use the arrow keys
 * to pick a layer (shared with the other views).
 */

import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { LineChart as LineChartIcon } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRun, PerPosition } from '../../types/geometry'
import { ChartTooltip, Placeholder, Segmented, type TooltipRow } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { formatPercent, formatValue, layerName, linePath, positionLabel } from './geometryUtils'

type LayerMetric = 'shape' | 'neighbors' | 'dimension' | 'between'

interface Series {
  key: string
  label: string
  values: PerPosition
  color: string
  dashed?: boolean
  faint?: boolean
}

interface ChartData {
  series: Series[]
  band: { lo: PerPosition; hi: PerPosition } | null
  refs: { value: number; label: string }[]
  domain: [number, number]
  format: (v: number | null | undefined) => string
  title: string
}

const HEIGHT = 320
const MARGIN = { top: 28, right: 96, bottom: 40, left: 48 }

function bandPath(lo: PerPosition, hi: PerPosition, x: (i: number) => number, y: (v: number) => number): string {
  const idx = lo.map((_, i) => i).filter((i) => lo[i] != null && hi[i] != null)
  if (idx.length < 2) return ''
  const top = idx.map((i, j) => `${j ? 'L' : 'M'}${x(i).toFixed(1)},${y(hi[i] as number).toFixed(1)}`).join('')
  const bottom = [...idx].reverse().map((i) => `L${x(i).toFixed(1)},${y(lo[i] as number).toFixed(1)}`).join('')
  return `${top}${bottom}Z`
}

function chartData(run: GeometryRun, metric: LayerMetric, dimsIndex: number, focus: string, seriesColors: string[], baseline: string): ChartData {
  const k = run.concept.values.length
  if (metric === 'shape' && run.shapes) {
    const hypotheses = run.shapes.hypotheses
    const focused = hypotheses.find((h) => h.id === focus) ?? hypotheses[0]
    const series: Series[] = hypotheses.map((h, i) => ({
      key: h.id,
      label: h.label,
      values: h.r2[dimsIndex],
      color: seriesColors[i],
      faint: h.id !== focused.id,
    }))
    series.push({ key: 'shuffled', label: `${focused.label} shuffled (mean)`, values: focused.shuffled_mean[dimsIndex], color: baseline, dashed: true })
    const all = [...series.flatMap((s) => s.values), ...focused.shuffled_max[dimsIndex]].filter((v): v is number => v != null)
    const low = Math.max(-0.5, Math.min(0, Math.floor(Math.min(...all) * 10) / 10))
    const embedding = focused.r2[dimsIndex][0]
    return {
      series,
      band: { lo: focused.shuffled_mean[dimsIndex], hi: focused.shuffled_max[dimsIndex] },
      refs: embedding != null ? [{ value: embedding, label: `embeddings ${formatValue(embedding)}` }] : [],
      domain: [low, 1],
      format: (v) => formatValue(v),
      title: `Held-out R² · top ${run.shapes.dims[dimsIndex]} components`,
    }
  }
  if (metric === 'neighbors' && run.neighbors) {
    return {
      series: [{ key: 'neighbors', label: 'Adjacent', values: run.neighbors, color: seriesColors[0] }],
      band: null,
      refs: run.neighbor_chance != null ? [{ value: run.neighbor_chance, label: `chance ${formatPercent(run.neighbor_chance)}` }] : [],
      domain: [0, 1],
      format: formatPercent,
      title: 'Values whose nearest class mean is adjacent',
    }
  }
  if (metric === 'dimension') {
    return {
      series: [
        { key: 'pr', label: 'Participation ratio', values: run.participation_ratio, color: seriesColors[0] },
        { key: 'n90', label: '90% of variance', values: run.components_90, color: seriesColors[1] },
      ],
      band: null,
      refs: [{ value: k - 1, label: `k − 1 = ${k - 1}` }],
      domain: [0, k - 1],
      format: (v) => formatValue(v, 1),
      title: 'Dimensions the class means use',
    }
  }
  return {
    series: [{ key: 'between', label: 'Between values', values: run.between_fraction, color: seriesColors[0] }],
    band: null,
    refs: [],
    domain: [0, 1],
    format: formatPercent,
    title: "Share of the examples' variance between values",
  }
}

export function LayerChart({ lab }: { lab: GeometryLab }) {
  if (!lab.run) {
    return (
      <section className="pl-panel">
        <Placeholder icon={LineChartIcon}>{lab.running ? 'Reading every layer…' : 'Run a concept to see it layer by layer.'}</Placeholder>
      </section>
    )
  }
  return <LayerChartBody lab={lab} run={lab.run} />
}

function LayerChartBody({ lab, run }: { lab: GeometryLab; run: GeometryRun }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [metric, setMetric] = useState<LayerMetric>('shape')
  const [focus, setFocus] = useState('')
  const [view, setView] = useState<'chart' | 'table'>('chart')
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const { position } = lab

  const metricOptions: { value: LayerMetric; label: string; disabled?: boolean; title?: string }[] = [
    { value: 'shape', label: 'Shape R²', disabled: !run.shapes, title: run.shapes ? undefined : 'No hypothesis applies to this concept' },
    { value: 'neighbors', label: 'Neighbors', disabled: !run.neighbors, title: run.neighbors ? undefined : 'Categorical values have no neighbors' },
    { value: 'dimension', label: 'Dimension' },
    { value: 'between', label: 'Between' },
  ]
  const active = metricOptions.some((o) => o.value === metric && !o.disabled) ? metric : 'dimension'
  const dimsIndex = run.shapes ? Math.max(run.shapes.dims.indexOf(lab.dims), 0) : 0
  const focusId = run.shapes?.hypotheses.some((h) => h.id === focus) ? focus : run.shapes?.hypotheses[0]?.id ?? ''
  const focused = run.shapes?.hypotheses.find((h) => h.id === focusId)
  const data = chartData(run, active, dimsIndex, focusId, colors.series, colors.baseline)

  const positions = run.num_layers + 1
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 50)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => MARGIN.left + (positions > 1 ? i * step : innerW / 2)
  const [y0, y1] = data.domain
  const y = (v: number) => MARGIN.top + (1 - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0 || 1)) * innerH
  const yTicks = Array.from({ length: 6 }, (_, i) => y0 + (i * (y1 - y0)) / 5)
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))
  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - MARGIN.left) / step)))
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: px, y: py } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    setHover({ index: indexAt(px), px, py })
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft' && position > 0) lab.setPosition(position - 1)
    else if (e.key === 'ArrowRight' && position < positions - 1) lab.setPosition(position + 1)
    else return
    e.preventDefault()
  }

  const tooltipRows = (i: number): TooltipRow[] => [
    ...data.series.map((s) => ({ key: s.color, label: s.label, value: data.format(s.values[i]) })),
    ...(active === 'shape' && focused
      ? [{ key: colors.baseline, label: `${focused.label} shuffled (max)`, value: data.format(focused.shuffled_max[dimsIndex][i]) }]
      : []),
  ]

  // Direct labels at the lines' right ends, nudged apart
  const endLabels = data.series
    .filter((s) => !s.dashed && s.values[positions - 1] != null)
    .map((s) => ({ s, py: y(s.values[positions - 1] as number) }))
    .sort((a, b) => a.py - b.py)
  for (let i = 1; i < endLabels.length; i++) endLabels[i].py = Math.max(endLabels[i].py, endLabels[i - 1].py + 12)

  return (
    <div className="gl-layers" style={{ opacity: lab.running ? 0.55 : 1 }}>
      <section className="pl-panel">
        <div className="pl-controls-row">
          <div>
            <div className="pl-control-label">Metric</div>
            <Segmented ariaLabel="Metric" value={active} onChange={setMetric} options={metricOptions} />
          </div>
          {active === 'shape' && run.shapes && (
            <>
              <div>
                <div className="pl-control-label" title="Shape tests read the top m class-mean components">
                  Fit dims
                </div>
                <Segmented
                  ariaLabel="Subspace size"
                  value={String(lab.dims)}
                  onChange={(v) => lab.setDims(Number(v))}
                  options={run.shapes.dims.map((m) => ({ value: String(m), label: String(m) }))}
                />
              </div>
              <div>
                <div className="pl-control-label" title="The hypothesis whose shuffled baseline and embeddings are drawn">
                  Baseline for
                </div>
                <Segmented
                  ariaLabel="Hypothesis"
                  value={focusId}
                  onChange={setFocus}
                  options={run.shapes.hypotheses.map((h) => ({ value: h.id, label: h.label }))}
                />
              </div>
            </>
          )}
        </div>

        <div className="pl-card-title">
          <h3>{data.title}</h3>
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

        <div ref={ref}>
          {view === 'chart' ? (
            <div
              className="pl-chart"
              tabIndex={0}
              role="group"
              aria-label={`${data.title} by layer. Selected: ${layerName(position)}. Use arrow keys to change layer.`}
              onKeyDown={onKeyDown}
            >
              <svg width={width} height={HEIGHT} role="img" aria-hidden="true">
                {yTicks.map((t) => (
                  <g key={t}>
                    <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
                    <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {Math.abs(y1 - y0) > 5 ? t.toFixed(0) : t.toFixed(1)}
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
                <rect x={x(position) - Math.max(step, 8) / 2} y={MARGIN.top} width={Math.max(step, 8)} height={innerH} fill={colors.hover} />
                <text x={x(position)} y={MARGIN.top - 10} textAnchor="middle" fontSize={11} fontWeight={600} fill={colors.text} {...halo}>
                  {layerName(position).toLowerCase()}
                </text>

                {/* Baselines */}
                {data.band && <path d={bandPath(data.band.lo, data.band.hi, x, y)} fill={colors.baseline} opacity={0.18} />}
                {data.refs.map((r) => (
                  <g key={r.label}>
                    <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(r.value)} y2={y(r.value)} stroke={colors.baseline} strokeDasharray="4 3" />
                    <text x={MARGIN.left + 4} y={y(r.value) - MARGIN.top < 16 ? y(r.value) + 13 : y(r.value) - 4} fontSize={10} fill={colors.text} {...halo}>
                      {r.label}
                    </text>
                  </g>
                ))}

                {/* Series: faint ones first, the focused one on top */}
                {[...data.series.filter((s) => s.faint), ...data.series.filter((s) => !s.faint)].map((s) => (
                  <path
                    key={s.key}
                    d={linePath(s.values, x, y)}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={s.dashed ? 1.5 : 2}
                    strokeDasharray={s.dashed ? '5 4' : undefined}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    opacity={s.faint ? 0.55 : 1}
                  />
                ))}
                {data.series
                  .filter((s) => !s.dashed && s.values[position] != null)
                  .map((s) => (
                    <circle key={s.key} cx={x(position)} cy={y(s.values[position] as number)} r={4} fill={s.color} stroke={colors.surface} strokeWidth={2} />
                  ))}
                {endLabels.map(({ s, py }) => (
                  <text key={s.key} x={MARGIN.left + innerW + 6} y={py} dy="0.32em" fontSize={10} fontWeight={600} fill={colors.text}>
                    {s.label}
                  </text>
                ))}

                {hover && <line x1={x(hover.index)} x2={x(hover.index)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} />}
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
                    const pointer = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
                    lab.setPosition(indexAt(pointer.x))
                  }}
                />
              </svg>
              {hover && <ChartTooltip x={hover.px} y={hover.py} containerWidth={width} title={layerName(hover.index)} rows={tooltipRows(hover.index)} />}
            </div>
          ) : (
            <LayerTable run={run} data={data} position={position} onSelect={lab.setPosition} />
          )}
        </div>

        <div className="pl-legend">
          {data.series.map((s) => (
            <span key={s.key} className="pl-legend-item">
              <span
                className="pl-legend-line"
                style={{
                  background: s.dashed ? `repeating-linear-gradient(to right, ${s.color} 0 5px, transparent 5px 9px)` : s.color,
                  opacity: s.faint ? 0.55 : 1,
                }}
              />
              {s.label}
            </span>
          ))}
          {data.band && (
            <span className="pl-legend-item">
              <span className="pl-legend-rect" style={{ background: colors.baseline, opacity: 0.35 }} />
              Shuffled mean–max
            </span>
          )}
        </div>
      </section>
    </div>
  )
}

function LayerTable({ run, data, position, onSelect }: { run: GeometryRun; data: ChartData; position: number; onSelect: (p: number) => void }) {
  const colors = useLabColors()
  return (
    <div style={{ maxHeight: 360, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
      <table className="pl-table" aria-label={`${data.title} per layer`}>
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th>Layer</th>
            {data.series.map((s) => (
              <th key={s.key}>{s.label}</th>
            ))}
            {data.band && <th>Shuffled max</th>}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: run.num_layers + 1 }, (_, i) => (
            <tr key={i} onClick={() => onSelect(i)} style={{ background: i === position ? colors.hover : undefined, borderTop: `1px solid ${colors.grid}` }}>
              <td style={{ fontWeight: i === position ? 600 : 400 }}>{positionLabel(i)}</td>
              {data.series.map((s) => (
                <td key={s.key}>{data.format(s.values[i])}</td>
              ))}
              {data.band && <td>{data.format(data.band.hi[i])}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
