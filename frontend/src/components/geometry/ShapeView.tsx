/**
 * The Shape view: a concept's class means and examples at one layer on the class means' top
 * principal components, in 3D and as an exportable SVG plane of two components, with the
 * layer's numbers next to their baselines and one-line findings.
 */

import { Component, Suspense, lazy, useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Download, Orbit } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRun } from '../../types/geometry'
import { ChartTooltip, Findings, Placeholder, Segmented, StatTiles, type Finding, type Tile } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import {
  READ_LABELS,
  closedPath,
  downloadSvg,
  exampleText,
  formatPercent,
  formatValue,
  layerName,
  valueColors,
} from './geometryUtils'

const ShapeScene = lazy(() => import('./ShapeScene'))

const PAIRS: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 2],
]

/** Shows a fallback instead of the 3D scene when WebGL can't start. */
class SceneBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function pad3(row: number[]): number[] {
  return [row[0] ?? 0, row[1] ?? 0, row[2] ?? 0]
}

/** The best-fitting hypothesis at a position (largest margin over its best shuffled fit). */
function bestHypothesis(run: GeometryRun, position: number, dimsIndex: number) {
  let best: { label: string; r2: number; max: number; mean: number; emb: number | null; embMax: number | null } | null = null
  for (const h of run.shapes?.hypotheses ?? []) {
    const r2 = h.r2[dimsIndex]?.[position]
    const max = h.shuffled_max[dimsIndex]?.[position]
    const mean = h.shuffled_mean[dimsIndex]?.[position]
    if (r2 == null || max == null || mean == null) continue
    if (!best || r2 - max > best.r2 - best.max) {
      best = { label: h.label, r2, max, mean, emb: h.r2[dimsIndex]?.[0] ?? null, embMax: h.shuffled_max[dimsIndex]?.[0] ?? null }
    }
  }
  return best
}

function shapeFindings(run: GeometryRun, position: number, dims: number): Finding[] {
  const findings: Finding[] = []
  const k = run.concept.values.length
  const dimsIndex = run.shapes ? run.shapes.dims.indexOf(dims) : -1
  const best = dimsIndex >= 0 ? bestHypothesis(run, position, dimsIndex) : null

  if (!run.shapes) {
    findings.push({ level: 'info', text: 'Nothing to test: give values groups (value | group) to test clusters.' })
  } else if (best) {
    if (best.r2 - best.max >= 0.1) {
      findings.push({
        level: 'good',
        text: `${best.label} fits best: R² ${formatValue(best.r2)} vs ≤ ${formatValue(best.max)} for shuffled values.`,
      })
    } else {
      findings.push({
        level: 'warning',
        text: `No shape beats shuffled values by 0.1 R² here (${best.label} ${formatValue(best.r2)} vs ${formatValue(best.max)}).`,
      })
    }
    if (position > 0 && best.emb != null && best.embMax != null && best.emb - best.embMax >= 0.1) {
      findings.push({
        level: 'warning',
        text: `${best.label} already fits at the embeddings (R² ${formatValue(best.emb)}): it may be token identity.`,
      })
    }
  }

  const neighbors = run.neighbors?.[position]
  if (neighbors != null && run.neighbor_chance != null) {
    findings.push({
      level: neighbors >= 2 * run.neighbor_chance && neighbors >= 0.5 ? 'good' : 'info',
      text: `Nearest class mean is an adjacent value for ${formatPercent(neighbors)} of values (chance ${formatPercent(run.neighbor_chance)}).`,
    })
  }

  const counts = run.value_tokens.flatMap((v) => v.counts)
  const low = Math.min(...counts)
  const high = Math.max(...counts)
  if (high > low) {
    findings.push({
      level: 'warning',
      text: `Values split into ${low}–${high} tokens${run.read === 'value_last' ? '; each is read at its last' : ''}.`,
    })
  }
  if (k <= 8) {
    findings.push({ level: 'info', text: `${k} values span at most ${k - 1} dimensions: shuffled orders fit well too.` })
  }
  return findings
}

function LayerSlider({ lab, run }: { lab: GeometryLab; run: GeometryRun }) {
  return (
    <div className="gl-slider">
      <div className="pl-control-label">{layerName(lab.position)}</div>
      <input
        type="range"
        min={0}
        max={run.num_layers}
        value={lab.position}
        onChange={(e) => lab.setPosition(Number(e.target.value))}
        aria-label="Layer"
        aria-valuetext={layerName(lab.position)}
      />
    </div>
  )
}

function ValueLegend({ run, fills }: { run: GeometryRun; fills: string[] }) {
  const colors = useLabColors()
  const { values, structure, groups } = run.concept
  if (structure === 'categorical' && groups) {
    const distinct = Array.from(new Set(groups))
    return (
      <div className="pl-legend">
        {distinct.map((g) => (
          <span key={g} className="pl-legend-item">
            <span className="pl-legend-dot" style={{ background: fills[groups.indexOf(g)] }} />
            {g}
          </span>
        ))}
      </div>
    )
  }
  if (structure === 'categorical') return null
  const stops = fills.map((c, i) => `${c} ${(i / Math.max(fills.length - 1, 1)) * 100}%`).join(', ')
  return (
    <div className="gl-ramp-legend">
      <span>{values[0]}</span>
      <span className="gl-ramp" style={{ background: `linear-gradient(to right, ${stops})`, borderColor: colors.border }} />
      <span>{values[values.length - 1]}</span>
      {structure === 'cyclic' && <span className="pl-hint">↺</span>}
    </div>
  )
}

interface PlanePoint {
  x: number
  y: number
  index: number
  mean: boolean
}

const PLANE_HEIGHT = 380
const PLANE_MARGIN = { top: 12, right: 12, bottom: 34, left: 40 }
const HOVER_RADIUS = 16

/** Two of the top three components as an SVG plot (what figure export saves). */
function PcPlane({ run, position, pair, showPoints, fills }: { run: GeometryRun; position: number; pair: [number, number]; showPoints: boolean; fills: string[] }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ point: PlanePoint; px: number; py: number } | null>(null)

  const k = run.concept.values.length
  const [a, b] = pair
  const means = run.means.values[position].map(pad3)
  const points = showPoints ? run.points.values[position] : []
  const explained = run.explained.values[position] ?? []

  const innerW = Math.max(width - PLANE_MARGIN.left - PLANE_MARGIN.right, 50)
  const innerH = PLANE_HEIGHT - PLANE_MARGIN.top - PLANE_MARGIN.bottom
  // Fit the class means, and most examples (a few far ones may fall outside)
  const meanExtent = Math.max(...means.flatMap((m) => [Math.abs(m[a]), Math.abs(m[b])]), 1e-6) * 1.2
  const pointExtents = points.flatMap((p) => [Math.abs(p[a]), Math.abs(p[b])]).sort((x, y) => x - y)
  const pointExtent = pointExtents.length ? pointExtents[Math.floor(pointExtents.length * 0.97)] * 1.05 : 0
  const extent = Math.max(meanExtent, pointExtent)
  const scale = Math.min(innerW, innerH) / (2 * extent)
  const px = (x: number) => PLANE_MARGIN.left + innerW / 2 + x * scale
  const py = (y: number) => PLANE_MARGIN.top + innerH / 2 - y * scale
  const inside = (x: number, y: number) =>
    px(x) >= PLANE_MARGIN.left && px(x) <= PLANE_MARGIN.left + innerW && py(y) >= PLANE_MARGIN.top && py(y) <= PLANE_MARGIN.top + innerH

  const order = closedPath(run.concept.structure) && k > 2 ? [...means, means[0]] : means
  const path = order.map((m, i) => `${i ? 'L' : 'M'}${px(m[a]).toFixed(1)},${py(m[b]).toFixed(1)}`).join('')
  const labelEvery = Math.max(1, Math.ceil(k / 24))

  const targets: PlanePoint[] = [
    ...means.map((m, v) => ({ x: m[a], y: m[b], index: v, mean: true })),
    ...points.map((p, i) => ({ x: p[a], y: p[b], index: i, mean: false })),
  ]
  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: mx, y: my } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    let best: PlanePoint | null = null
    let bestDist = HOVER_RADIUS ** 2
    for (const t of targets) {
      // Means win ties within their larger radius
      const d = (px(t.x) - mx) ** 2 + (py(t.y) - my) ** 2 - (t.mean ? 30 : 0)
      if (d < bestDist) {
        bestDist = d
        best = t
      }
    }
    setHover(best ? { point: best, px: mx, py: my } : null)
  }

  const axisName = (i: number) => `PC${i + 1} (${((explained[i] ?? 0) * 100).toFixed(1)}%)`
  const filename = `${run.concept.name}-${run.model}-${layerName(position)}-PC${a + 1}-PC${b + 1}`.replace(/[^\w.-]+/g, '_')

  return (
    <div ref={ref} className="pl-chart">
      <div className="pl-card-title">
        <h3>
          PC{a + 1} × PC{b + 1}
        </h3>
        <button
          type="button"
          className="pl-icon-button"
          title="Download SVG"
          aria-label="Download SVG"
          onClick={() => svgRef.current && downloadSvg(svgRef.current, `${filename}.svg`, colors.surface)}
          style={{ color: colors.text }}
        >
          <Download className="pl-icon" />
        </button>
      </div>
      <svg ref={svgRef} width={width} height={PLANE_HEIGHT} role="img" aria-label={`${run.concept.name} at ${layerName(position)}, PC${a + 1} against PC${b + 1}`}>
        <rect x={PLANE_MARGIN.left} y={PLANE_MARGIN.top} width={innerW} height={innerH} fill="none" stroke={colors.grid} />
        <line x1={px(0)} x2={px(0)} y1={PLANE_MARGIN.top} y2={PLANE_MARGIN.top + innerH} stroke={colors.grid} />
        <line x1={PLANE_MARGIN.left} x2={PLANE_MARGIN.left + innerW} y1={py(0)} y2={py(0)} stroke={colors.grid} />
        {points.map((p, i) =>
          inside(p[a], p[b]) ? <circle key={i} cx={px(p[a])} cy={py(p[b])} r={2.5} fill={fills[i % k]} opacity={0.45} /> : null,
        )}
        <path d={path} fill="none" stroke={colors.text} strokeWidth={1.5} strokeLinejoin="round" opacity={0.6} />
        {means.map((m, v) => (
          <circle key={v} cx={px(m[a])} cy={py(m[b])} r={5} fill={fills[v]} stroke={colors.surface} strokeWidth={2} />
        ))}
        {means.map((m, v) =>
          v % labelEvery === 0 || v === k - 1 ? (
            <text
              key={v}
              x={px(m[a]) + 7}
              y={py(m[b]) - 7}
              fontSize={10}
              fontWeight={600}
              fill={colors.text}
              stroke={colors.surface}
              strokeWidth={3}
              paintOrder="stroke"
            >
              {run.concept.values[v]}
            </text>
          ) : null,
        )}
        {hover && (
          <circle cx={px(hover.point.x)} cy={py(hover.point.y)} r={hover.point.mean ? 8 : 5} fill="none" stroke={colors.text} strokeWidth={1.5} />
        )}
        <text x={PLANE_MARGIN.left + innerW / 2} y={PLANE_HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
          {axisName(a)}
        </text>
        <text
          x={12}
          y={PLANE_MARGIN.top + innerH / 2}
          textAnchor="middle"
          fontSize={11}
          fill={colors.text}
          transform={`rotate(-90 12 ${PLANE_MARGIN.top + innerH / 2})`}
        >
          {axisName(b)}
        </text>
        <rect
          x={PLANE_MARGIN.left}
          y={PLANE_MARGIN.top}
          width={innerW}
          height={innerH}
          fill="transparent"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          rows={[
            {
              key: fills[hover.point.mean ? hover.point.index : hover.point.index % k],
              label: hover.point.mean ? 'class mean' : 'example',
              value: run.concept.values[hover.point.mean ? hover.point.index : hover.point.index % k],
            },
          ]}
        >
          {!hover.point.mean && (
            <div style={{ marginTop: '0.3rem', whiteSpace: 'normal' }}>{exampleText(run.concept, hover.point.index)}</div>
          )}
        </ChartTooltip>
      )}
    </div>
  )
}

export function ShapeView({ lab }: { lab: GeometryLab }) {
  const colors = useLabColors()
  const [showPoints, setShowPoints] = useState(true)
  const { run, position } = lab
  const fills = useMemo(() => (run ? valueColors(run.concept, colors) : []), [run, colors])

  if (!run) {
    return (
      <section className="pl-panel">
        <Placeholder icon={Orbit}>{lab.running ? 'Reading every layer…' : 'Choose a model and a concept, then run.'}</Placeholder>
      </section>
    )
  }

  const k = run.concept.values.length
  const dimsIndex = run.shapes ? run.shapes.dims.indexOf(lab.dims) : -1
  const best = dimsIndex >= 0 ? bestHypothesis(run, position, dimsIndex) : null
  const explained = run.explained.values[position] ?? []
  const top3 = explained.slice(0, 3).reduce((sum, v) => sum + v, 0)
  const neighbors = run.neighbors?.[position]

  const tiles: Tile[] = [
    ...(best
      ? [
          {
            label: `${best.label} R²`,
            value: formatValue(best.r2),
            hint: `shuffled ≤ ${formatValue(best.max)}`,
            title: `Held-out R² of the best-fitting hypothesis, read from the top ${lab.dims} class-mean components, against the best of the shuffled value orders`,
          },
        ]
      : []),
    ...(neighbors != null
      ? [
          {
            label: 'Neighbors',
            value: formatPercent(neighbors),
            hint: `chance ${formatPercent(run.neighbor_chance)}`,
            title: "Share of values whose nearest class mean is an adjacent value",
          },
        ]
      : []),
    {
      label: 'PC1–3',
      value: formatPercent(top3),
      hint: 'of class-mean variance',
      title: 'Variance of the class means on their top three principal components',
    },
    {
      label: 'Dimension',
      value: formatValue(run.participation_ratio[position], 1),
      hint: `${run.components_90[position] ?? '–'} for 90% · of ${k - 1}`,
      title: 'Participation ratio (Σλ)²/Σλ² of the class means, and the components holding 90% of their variance; k values span at most k − 1',
    },
    {
      label: 'Between',
      value: formatPercent(run.between_fraction[position]),
      hint: 'of variance',
      title: "Share of the examples' variance that lies between the values' class means",
    },
  ]

  return (
    <div className="gl-shape" style={{ opacity: lab.running ? 0.55 : 1 }} aria-busy={lab.running}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{run.concept.name}</strong> · {lab.modelName(run.model)} · {run.concept.templates.length} templates × {k} values · read at{' '}
          {READ_LABELS[run.read].toLowerCase()} · ${run.cost.toFixed(2)}
        </div>
        <div className="pl-controls-row">
          <LayerSlider lab={lab} run={run} />
          <div>
            <div className="pl-control-label">Plane</div>
            <Segmented
              ariaLabel="Principal components"
              value={`${lab.pcPair[0]}${lab.pcPair[1]}`}
              onChange={(v) => lab.setPcPair([Number(v[0]), Number(v[1])])}
              options={PAIRS.map(([a, b]) => ({ value: `${a}${b}`, label: `PC${a + 1}–${b + 1}` }))}
            />
          </div>
          {run.shapes && (
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
          )}
          <label className="pl-check-row" style={{ alignSelf: 'flex-end', fontSize: '0.75rem' }}>
            <input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} />
            Examples
          </label>
        </div>
        <StatTiles tiles={tiles} />
        <Findings items={shapeFindings(run, position, lab.dims)} />
        <ValueLegend run={run} fills={fills} />
      </section>

      <div className="pl-results-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>3D · {layerName(position)}</h3>
          </div>
          <SceneBoundary fallback={<Placeholder icon={Orbit}>3D needs WebGL; the plane view shows the same data.</Placeholder>}>
            <Suspense fallback={<div className="gl-scene-loading pl-hint">Loading 3D…</div>}>
              <ShapeScene
                points={run.points.values[position]}
                means={run.means.values[position].map(pad3)}
                colors={fills}
                labels={run.concept.values}
                closed={closedPath(run.concept.structure)}
                showPoints={showPoints}
                explained={explained}
                exampleText={(i) => exampleText(run.concept, i)}
                ink={colors.text}
                muted={colors.baseline}
                surface={colors.isLight ? '#ffffff' : '#141414'}
              />
            </Suspense>
          </SceneBoundary>
        </section>
        <section className="pl-panel">
          <PcPlane run={run} position={position} pair={lab.pcPair} showPoints={showPoints} fills={fills} />
        </section>
      </div>

      <style>{`
        .geometry-lab .gl-scene-loading { height: 380px; display: flex; align-items: center; justify-content: center; }
      `}</style>
    </div>
  )
}
