/**
 * The Compare view: the current run's class means at the selected layer, aligned by orthogonal
 * Procrustes onto another run's (or the same run's) at every layer, for the same values: another
 * layer, another model, or the base/instruct sibling. Aligned R² and linear CKA come with the
 * same comparison on shuffled value pairings; the overlay shows the two configurations on top of
 * each other. Computed on the backend's CPU from the runs' coordinates (free).
 */

import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Download, GitCompareArrows, Play } from 'lucide-react'
import { apiClient } from '../../api/client'
import type { GeometryLab, StoredRun } from '../../hooks/useGeometryLab'
import type { CompareResult, GeometryRun } from '../../types/geometry'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, Placeholder, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { closedPath, downloadSvg, encodeWire, formatValue, layerName, linePath, positionLabel, sameValues, valueColors } from './geometryUtils'

type CompareDims = number | 'all'

const CHART_HEIGHT = 280
const CHART_MARGIN = { top: 28, right: 56, bottom: 40, left: 48 }
const PLANE_HEIGHT = 340
const PLANE_MARGIN = { top: 12, right: 12, bottom: 30, left: 12 }

function columns(rows: number[][], m: number | null): number[][] {
  return m == null ? rows : rows.map((row) => row.slice(0, m))
}

export function CompareView({ lab }: { lab: GeometryLab }) {
  if (!lab.run || lab.runId == null) {
    return (
      <section className="pl-panel">
        <Placeholder icon={GitCompareArrows}>Run a concept, then compare it with another layer, model, or its base/instruct sibling.</Placeholder>
      </section>
    )
  }
  return <CompareBody lab={lab} run={lab.run} runId={lab.runId} />
}

function CompareBody({ lab, run, runId }: { lab: GeometryLab; run: GeometryRun; runId: number }) {
  const colors = useLabColors()
  const candidates = lab.runs.filter((r) => sameValues(r.run.concept.values, run.concept.values))
  const [otherId, setOtherId] = useState<number>(runId)
  const other: StoredRun = candidates.find((r) => r.id === otherId) ?? candidates.find((r) => r.id === runId)!
  const [dims, setDims] = useState<CompareDims>(lab.dims)
  const [overlayAt, setOverlayAt] = useState<number | null>(null)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const position = lab.position
  const rankA = run.means.values[position]?.[0]?.length ?? 0
  const rankB = other.run.means.values[0]?.[0]?.length ?? 0
  const dimOptions = [2, 3, 4, 6, 8].filter((m) => m <= Math.min(rankA, rankB))
  const m = dims === 'all' || !dimOptions.includes(dims) ? null : dims

  // A new pair of runs or layer starts from the best-aligned layer again
  useEffect(() => setOverlayAt(null), [runId, other.id, position, m])

  useEffect(() => {
    let cancelled = false
    const timer = setTimeout(() => {
      const a = columns(run.means.values[position], m)
      const b = other.run.means.values.map((rows) => columns(rows, m))
      setLoading(true)
      setError(null)
      apiClient
        .compareGeometry({
          a: { values: run.concept.values, coords: encodeWire(a, [a.length, a[0].length]) },
          b: { values: other.run.concept.values, coords: encodeWire(b, [b.length, b[0].length, b[0][0].length]) },
          dims: m,
          position: overlayAt,
        })
        .then((r) => !cancelled && setResult(r))
        .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Comparison failed'))
        .finally(() => !cancelled && setLoading(false))
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [run, other.run, position, m, overlayAt])

  const at = result?.position ?? 0
  const findings: Finding[] = []
  if (result) {
    const r2 = result.r2[at]
    const max = result.shuffled_r2.max[at]
    if (r2 != null && max != null) {
      findings.push(
        r2 - max >= 0.1
          ? { level: 'good', text: `${layerName(at)} of ${other.label} aligns at R² ${formatValue(r2)} vs ≤ ${formatValue(max)} for shuffled pairings.` }
          : { level: 'warning', text: `${layerName(at)} aligns no better than shuffled pairings (R² ${formatValue(r2)} vs ${formatValue(max)}).` },
      )
    }
    const shuffledMean = Math.max(...result.shuffled_r2.mean.filter((v): v is number => v != null))
    if (shuffledMean >= 0.5) {
      findings.push({ level: 'warning', text: `In ${result.dims} dimensions even shuffled pairings align at R² ${formatValue(shuffledMean)}: compare fewer.` })
    }
    if (other.id === runId) findings.push({ level: 'info', text: `Same run: ${layerName(position)} matches itself exactly.` })
  }

  const fills = valueColors(run.concept, colors)
  const sibling = lab.models.find((m) => m.model_id === run.model)?.sibling ?? null
  const siblingRun = candidates.find((r) => r.run.model === sibling && r.run.read === run.read)
  const compareSibling = async () => {
    if (!sibling) return
    if (siblingRun) {
      setOtherId(siblingRun.id)
      return
    }
    // The current run becomes the other one; the sibling's run becomes the current one
    setOtherId(runId)
    await lab.rerunOn(sibling)
  }
  const runOptions = candidates.map((r) => ({ value: String(r.id), label: r.id === runId ? `${r.label} (this run)` : r.label }))

  return (
    <div className="gl-compare">
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{layerName(position)}</strong> of {candidates.find((r) => r.id === runId)?.label} against every layer of:
        </div>
        <div className="pl-controls-row">
          <div style={{ minWidth: 260, flex: 1, maxWidth: 480 }}>
            <div className="pl-control-label" title="Runs of the same values, in the same order">
              Compare with
            </div>
            <Dropdown value={String(other.id)} onChange={(v) => setOtherId(Number(v))} options={runOptions} />
          </div>
          <div className="gl-slider">
            <div className="pl-control-label">This run: {layerName(position)}</div>
            <input type="range" min={0} max={run.num_layers} value={position} onChange={(e) => lab.setPosition(Number(e.target.value))} aria-label="Layer of this run" />
          </div>
          <div>
            <div className="pl-control-label" title="Both configurations are cut to their top components before aligning">
              Dims
            </div>
            <Segmented
              ariaLabel="Dimensions compared"
              value={m == null ? 'all' : String(m)}
              onChange={(v) => setDims(v === 'all' ? 'all' : Number(v))}
              options={[...dimOptions.map((d) => ({ value: String(d), label: String(d) })), { value: 'all', label: 'All', title: `All ${Math.max(rankA, rankB)}` }]}
            />
          </div>
        </div>
        {sibling && (
          <button
            type="button"
            className="pl-button"
            onClick={compareSibling}
            disabled={lab.running}
            style={{ borderColor: colors.border, color: colors.text, marginBottom: '0.75rem' }}
            title="Run this concept on the model's base or instruct counterpart and compare"
          >
            {lab.running ? <span className="pl-spinner" style={{ borderColor: colors.border, borderTopColor: colors.text }} /> : <Play className="pl-icon" aria-hidden="true" />}
            {siblingRun ? `Compare with ${lab.modelName(sibling)}` : `Run on ${lab.modelName(sibling)}`}
          </button>
        )}
        {!sibling && candidates.length < 2 && <div className="pl-hint" style={{ marginBottom: '0.75rem' }}>Run the same concept on another model to compare models.</div>}
        {error && <div className="pl-error-note">{error}</div>}
        {result && (
          <>
            <StatTiles
              tiles={[
                {
                  label: 'Aligned R²',
                  value: formatValue(result.r2[at]),
                  hint: `shuffled ≤ ${formatValue(result.shuffled_r2.max[at])}`,
                  title: 'Share of the other configuration explained by a rotated, scaled copy of this one (orthogonal Procrustes)',
                },
                {
                  label: 'CKA',
                  value: formatValue(result.cka[at]),
                  hint: `shuffled ≤ ${formatValue(result.shuffled_cka.max[at])}`,
                  title: 'Linear centered kernel alignment between the two configurations',
                },
                { label: 'Overlay', value: layerName(at), hint: other.run.model },
              ]}
            />
            <Findings items={findings} />
          </>
        )}
      </section>

      {result && (
        <div className="pl-results-grid" style={{ opacity: loading ? 0.6 : 1 }}>
          <section className="pl-panel">
            <div className="pl-card-title">
              <h3>By layer of the other run</h3>
            </div>
            <CompareChart result={result} positions={other.run.num_layers + 1} selected={at} onSelect={setOverlayAt} />
          </section>
          <section className="pl-panel">
            <Overlay result={result} run={run} fills={fills} labelA={`${layerName(position)} (aligned)`} labelB={layerName(at)} />
          </section>
        </div>
      )}
    </div>
  )
}

function CompareChart({ result, positions, selected, onSelect }: { result: CompareResult; positions: number; selected: number; onSelect: (p: number) => void }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const innerW = Math.max(width - CHART_MARGIN.left - CHART_MARGIN.right, 50)
  const innerH = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => CHART_MARGIN.left + (positions > 1 ? i * step : innerW / 2)
  const y = (v: number) => CHART_MARGIN.top + (1 - Math.max(0, Math.min(1, v))) * innerH
  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - CHART_MARGIN.left) / step)))
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))
  const series = [
    { key: 'r2', label: 'Aligned R²', values: result.r2, color: colors.series[0] },
    { key: 'cka', label: 'CKA', values: result.cka, color: colors.series[1] },
  ]
  const lo = result.shuffled_r2.mean
  const hi = result.shuffled_r2.max
  const band = lo
    .map((_, i) => i)
    .filter((i) => lo[i] != null && hi[i] != null)
  const bandPath = band.length > 1
    ? `${band.map((i, j) => `${j ? 'L' : 'M'}${x(i)},${y(hi[i] as number)}`).join('')}${[...band].reverse().map((i) => `L${x(i)},${y(lo[i] as number)}`).join('')}Z`
    : ''

  // Direct labels at the lines' right ends, nudged apart
  const endLabels = series
    .filter((s) => s.values[positions - 1] != null)
    .map((s) => ({ s, py: y(s.values[positions - 1] as number) }))
    .sort((a, b) => a.py - b.py)
  for (let i = 1; i < endLabels.length; i++) endLabels[i].py = Math.max(endLabels[i].py, endLabels[i - 1].py + 12)

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: px, y: py } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    setHover({ index: indexAt(px), px, py })
  }

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={CHART_HEIGHT} role="img" aria-label="Aligned R² and CKA at every layer of the other run, with shuffled pairings">
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
          <g key={t}>
            <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <text x={CHART_MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {Array.from({ length: positions }, (_, i) =>
          i % labelEvery === 0 || i === positions - 1 ? (
            <text key={i} x={x(i)} y={CHART_MARGIN.top + innerH + 16} textAnchor="middle" fontSize={11} fill={colors.text}>
              {positionLabel(i)}
            </text>
          ) : null,
        )}
        <text x={CHART_MARGIN.left + innerW / 2} y={CHART_HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
          Layer
        </text>
        <rect x={x(selected) - Math.max(step, 8) / 2} y={CHART_MARGIN.top} width={Math.max(step, 8)} height={innerH} fill={colors.hover} />
        {bandPath && <path d={bandPath} fill={colors.baseline} opacity={0.18} />}
        <path d={linePath(lo, x, y)} fill="none" stroke={colors.baseline} strokeWidth={1.5} strokeDasharray="5 4" />
        {series.map((s) => (
          <path key={s.key} d={linePath(s.values, x, y)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
        ))}
        {series.map((s) =>
          s.values[selected] != null ? (
            <circle key={s.key} cx={x(selected)} cy={y(s.values[selected] as number)} r={4} fill={s.color} stroke={colors.surface} strokeWidth={2} />
          ) : null,
        )}
        {endLabels.map(({ s, py }) => (
          <text key={s.key} x={CHART_MARGIN.left + innerW + 6} y={py} dy="0.32em" fontSize={10} fontWeight={600} fill={colors.text}>
            {s.label}
          </text>
        ))}
        {hover && <line x1={x(hover.index)} x2={x(hover.index)} y1={CHART_MARGIN.top} y2={CHART_MARGIN.top + innerH} stroke={colors.axis} />}
        <rect
          x={CHART_MARGIN.left - step / 2}
          y={CHART_MARGIN.top}
          width={innerW + step}
          height={innerH}
          fill="transparent"
          style={{ cursor: 'pointer' }}
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
          onClick={(e) => {
            const pointer = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
            onSelect(indexAt(pointer.x))
          }}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          title={layerName(hover.index)}
          rows={[
            ...series.map((s) => ({ key: s.color, label: s.label, value: formatValue(s.values[hover.index], 3) })),
            { key: colors.baseline, label: 'shuffled R² (mean)', value: formatValue(lo[hover.index], 3) },
            { key: colors.baseline, label: 'shuffled R² (max)', value: formatValue(hi[hover.index], 3) },
          ]}
        />
      )}
      <div className="pl-legend">
        {series.map((s) => (
          <span key={s.key} className="pl-legend-item">
            <span className="pl-legend-line" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.baseline, opacity: 0.35 }} />
          Shuffled R² mean–max
        </span>
      </div>
    </div>
  )
}

function Overlay({ result, run, fills, labelA, labelB }: { result: CompareResult; run: GeometryRun; fills: string[]; labelA: string; labelB: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ v: number; which: 'a' | 'b'; px: number; py: number } | null>(null)
  const { a, b } = result.overlay
  const k = b.length
  const values = run.concept.values
  const innerW = Math.max(width - PLANE_MARGIN.left - PLANE_MARGIN.right, 50)
  const innerH = PLANE_HEIGHT - PLANE_MARGIN.top - PLANE_MARGIN.bottom
  const extent = useMemo(() => Math.max(...[...a, ...b].flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]), 1e-6) * 1.15, [a, b])
  const scale = Math.min(innerW, innerH) / (2 * extent)
  const px = (v: number) => PLANE_MARGIN.left + innerW / 2 + v * scale
  const py = (v: number) => PLANE_MARGIN.top + innerH / 2 - v * scale
  const closed = closedPath(run.concept.structure) && k > 2
  const path = (pts: number[][]) => (closed ? [...pts, pts[0]] : pts).map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join('')
  const labelEvery = Math.max(1, Math.ceil(k / 24))

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: mx, y: my } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    let best: { v: number; which: 'a' | 'b' } | null = null
    let bestDist = 16 ** 2
    ;(['b', 'a'] as const).forEach((which) =>
      (which === 'a' ? a : b).forEach((p, v) => {
        const d = (px(p[0]) - mx) ** 2 + (py(p[1]) - my) ** 2
        if (d < bestDist) {
          bestDist = d
          best = { v, which }
        }
      }),
    )
    setHover(best ? { ...(best as { v: number; which: 'a' | 'b' }), px: mx, py: my } : null)
  }

  return (
    <div ref={ref} className="pl-chart">
      <div className="pl-card-title">
        <h3 title="Both on the other run's top two components; this run's class means rotated and scaled onto them">Overlay</h3>
        <button
          type="button"
          className="pl-icon-button"
          title="Download SVG"
          aria-label="Download SVG"
          onClick={() => svgRef.current && downloadSvg(svgRef.current, `${run.concept.name}-overlay.svg`.replace(/[^\w.-]+/g, '_'), colors.surface)}
          style={{ color: colors.text }}
        >
          <Download className="pl-icon" />
        </button>
      </div>
      <svg ref={svgRef} width={width} height={PLANE_HEIGHT} role="img" aria-label={`${labelA} aligned onto ${labelB}`}>
        <rect x={PLANE_MARGIN.left} y={PLANE_MARGIN.top} width={innerW} height={innerH} fill="none" stroke={colors.grid} />
        {b.map((p, v) => (
          <line key={v} x1={px(a[v][0])} y1={py(a[v][1])} x2={px(p[0])} y2={py(p[1])} stroke={colors.baseline} strokeWidth={1} />
        ))}
        <path d={path(b)} fill="none" stroke={colors.text} strokeWidth={1.5} opacity={0.6} />
        <path d={path(a)} fill="none" stroke={colors.text} strokeWidth={1.5} strokeDasharray="4 3" opacity={0.6} />
        {b.map((p, v) => (
          <circle key={`b${v}`} cx={px(p[0])} cy={py(p[1])} r={5} fill={fills[v]} stroke={colors.surface} strokeWidth={2} />
        ))}
        {a.map((p, v) => (
          <circle key={`a${v}`} cx={px(p[0])} cy={py(p[1])} r={4.5} fill={colors.surface} stroke={fills[v]} strokeWidth={2} />
        ))}
        {b.map((p, v) =>
          v % labelEvery === 0 || v === k - 1 ? (
            <text key={v} x={px(p[0]) + 7} y={py(p[1]) - 7} fontSize={10} fontWeight={600} fill={colors.text} stroke={colors.surface} strokeWidth={3} paintOrder="stroke">
              {values[v]}
            </text>
          ) : null,
        )}
        <text x={PLANE_MARGIN.left + innerW / 2} y={PLANE_HEIGHT - 8} textAnchor="middle" fontSize={11} fill={colors.text}>
          Other run's PC1 × PC2
        </text>
        <rect x={PLANE_MARGIN.left} y={PLANE_MARGIN.top} width={innerW} height={innerH} fill="transparent" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          rows={[{ key: fills[hover.v], label: hover.which === 'a' ? labelA : labelB, value: values[hover.v] }]}
        />
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.text }} />
          {labelB}
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-ring" style={{ borderColor: colors.text }} />
          {labelA}
        </span>
      </div>
    </div>
  )
}
