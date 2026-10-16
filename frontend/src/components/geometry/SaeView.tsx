/**
 * The SAE view: how does an SAE break up the shape? The run's concept is read through an SAE at
 * the layer it reads. Features are added greedily by how much of the centered class means their
 * fixed contributions (tuning curve × decoder row) write, against random orders and the ranking
 * by decoder cosine; each feature's tuning curve is drawn as a heat strip and on the shape; and a
 * verdict says whether the SAE captures, shatters, dilutes or misses the shape. Results are kept
 * per run and SAE, and any two on the same values can be compared side by side.
 */

import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Download, Puzzle } from 'lucide-react'
import { apiClient } from '../../api/client'
import { runConceptSpec, type GeometryLab } from '../../hooks/useGeometryLab'
import type { ProbeSaeInfo } from '../../types/probes'
import type { GeometryRun, GeometrySaeEstimate, GeometrySaeRequest, GeometrySaeResult, PerPosition, SaeShapeFit, SaeVerdict } from '../../types/geometry'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding, type Tile, type TooltipRow } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { FeatureLink } from '../lab/FeatureLink'
import { formatWidth, refKey, saeBrowserUrl } from '../lab/saeLinks'
import { READ_LABELS, closedPath, downloadSvg, exampleText, formatPercent, formatValue, layerName, linePath, sameValues, sequentialColor } from './geometryUtils'

const VERDICTS: Record<SaeVerdict, { label: string; title: string; level: Finding['level'] }> = {
  capture: {
    label: 'Capture',
    title: 'A compact group writes 90% of the shape: at most two features per dimension it spans, and at most half the values',
    level: 'good',
  },
  shatter: {
    label: 'Shatter',
    title: 'The features writing 90% of the shape are rarely active together: each tiles a piece of it',
    level: 'info',
  },
  dilute: {
    label: 'Dilute',
    title: 'Partly shared features write the shape, but no compact group writes 90% of it',
    level: 'info',
  },
  missed: {
    label: 'Missed',
    title: "Every feature together writes less than 90% of the shape; the rest is in the SAE's error",
    level: 'warning',
  },
}
// Results kept (across runs), heat-strip rows at most, shapes drawn per result
const MAX_RESULTS = 8
const STRIP_ROWS = 32
const FIELD_PLOTS = 8
// Features the baseline finding compares at
const BASELINE_AT = 8

interface StoredSae {
  id: number
  runId: number
  key: string
  label: string
  result: GeometrySaeResult
}

function dimsLabel(dims: number | null): string {
  return dims == null ? 'All' : String(dims)
}

function dimsText(dims: number | null): string {
  return dims == null ? 'all dims' : `top ${dims} PCs`
}

function fitFor(result: GeometrySaeResult, dims: number | null): SaeShapeFit {
  return result.fits.find((f) => f.dims === dims) ?? result.fits[result.fits.length - 1]
}

function at(values: PerPosition, n: number): number | null {
  return values.length ? values[Math.min(n, values.length) - 1] ?? null : null
}

function estimateNote(estimate: GeometrySaeEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

function saeLabel(s: ProbeSaeInfo): string {
  const parts = [`Layer ${s.layer}`, s.name, `${formatWidth(s.width)} features`]
  if (s.match === 'sibling') parts.push(`trained on ${s.trained_on}`)
  if (s.unusable_reason) parts.push(s.unusable_reason)
  return parts.join(' · ')
}

/** The values in a feature's receptive field (where it reaches half its peak), as runs of adjacent values. */
function fieldText(values: string[], tuning: number[], structure: GeometryRun['concept']['structure']): string {
  const k = values.length
  const peak = Math.max(...tuning)
  if (!(peak > 0)) return '–'
  const on = tuning.map((t) => t >= 0.5 * peak)
  if (on.every(Boolean)) return 'all values'
  if (structure === 'categorical') {
    const names = values.filter((_, i) => on[i])
    return names.length > 4 ? `${names.slice(0, 3).join(', ')}, +${names.length - 3}` : names.join(', ')
  }
  const runs: [number, number][] = []
  for (let i = 0; i < k; i++) {
    if (!on[i] || (i > 0 && on[i - 1])) continue
    let j = i
    while (j + 1 < k && on[j + 1]) j++
    runs.push([i, j])
  }
  if (structure === 'cyclic' && runs.length > 1 && on[0] && on[k - 1]) {
    const first = runs.shift() as [number, number]
    runs[runs.length - 1] = [runs[runs.length - 1][0], first[1]]
  }
  const text = runs.map(([a, b]) => (a === b ? values[a] : `${values[a]}–${values[b]}`))
  return text.length > 3 ? `${text.slice(0, 3).join(', ')}, …` : text.join(', ')
}

function argmax(values: number[]): number {
  let best = 0
  values.forEach((v, i) => {
    if (v > values[best]) best = i
  })
  return best
}

function saeFindings(result: GeometrySaeResult, fit: SaeShapeFit): Finding[] {
  const out: Finding[] = []
  const { fvu, info } = result.sae
  if (fvu != null && fvu > 0.5) {
    out.push({ level: 'warning', text: `Poor fit: ${formatPercent(fvu)} of variance unexplained at the read positions.` })
  }
  if (info.match === 'sibling') out.push({ level: 'info', text: `Trained on ${info.trained_on}, not ${result.model}.` })

  const verdict = VERDICTS[fit.verdict]
  const coactive = formatValue(fit.fields.coactive, 1)
  const steps = fit.r2.length
  const text =
    fit.verdict === 'capture'
      ? `${fit.n90} feature${fit.n90 === 1 ? '' : 's'} write 90% of the shape (compact: ≤ ${fit.compact}).`
      : fit.verdict === 'shatter'
        ? `${fit.n90} features write 90%, ${coactive} active per example: each tiles a piece.`
        : fit.verdict === 'dilute' && fit.n90 != null
          ? `${fit.n90} features write 90% (compact: ≤ ${fit.compact}), ${coactive} active per example.`
          : fit.verdict === 'dilute'
            ? `${steps} features write ${formatPercent(at(fit.r2, steps))}; all ${result.varying_features} write ${formatPercent(fit.ceiling)}.`
            : `every feature together writes ${formatPercent(fit.ceiling)} of the shape; the rest is in the SAE's error.`
  out.push({ level: verdict.level, text: `${verdict.label}: ${text}` })

  const n = Math.min(BASELINE_AT, steps)
  if (n > 0) {
    out.push({
      level: 'info',
      text: `First ${n}: greedy ${formatPercent(at(fit.r2, n))}, by decoder cosine ${formatPercent(at(fit.cosine_r2, n))}, random ${formatPercent(at(fit.random.mean, n))}.`,
    })
  }
  const all = result.fits[result.fits.length - 1]
  if (fit.dims != null && all.verdict !== fit.verdict) {
    out.push({ level: 'info', text: `On all dimensions (token identity included): ${VERDICTS[all.verdict].label.toLowerCase()}.` })
  }
  return out
}

export function SaeView({ lab }: { lab: GeometryLab }) {
  const [results, setResults] = useState<StoredSae[]>([])
  const [nextId, setNextId] = useState(1)
  const { run, runId } = lab
  if (!run || runId == null) {
    return (
      <section className="pl-panel">
        <Placeholder icon={Puzzle}>{lab.running ? 'Reading every layer…' : 'Run a concept, then read it through an SAE.'}</Placeholder>
      </section>
    )
  }
  const add = (result: GeometrySaeResult, key: string) => {
    const id = nextId
    setNextId(id + 1)
    const label = `${result.sae.info.name} · ${lab.modelName(result.model)}`
    setResults((current) => [{ id, runId, key, label, result }, ...current.filter((r) => !(r.runId === runId && r.key === key))].slice(0, MAX_RESULTS))
    return id
  }
  return <SaeBody key={runId} lab={lab} run={run} runId={runId} results={results} onResult={add} />
}

function SaeBody({
  lab,
  run,
  runId,
  results,
  onResult,
}: {
  lab: GeometryLab
  run: GeometryRun
  runId: number
  results: StoredSae[]
  onResult: (result: GeometrySaeResult, key: string) => number
}) {
  const colors = useLabColors()
  const [saes, setSaes] = useState<ProbeSaeInfo[] | null>(null)
  const [choice, setChoice] = useState('')
  const [estimate, setEstimate] = useState<GeometrySaeEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [shownId, setShownId] = useState<number | null>(null)
  const [compareId, setCompareId] = useState('')

  useEffect(() => {
    if (!lab.isAuthenticated) {
      setSaes([])
      return
    }
    let cancelled = false
    setSaes(null)
    apiClient
      .listProbeSaes(run.model)
      .then((list) => !cancelled && setSaes(list))
      .catch(() => !cancelled && setSaes([]))
    return () => {
      cancelled = true
    }
  }, [lab.isAuthenticated, run.model])

  const usable = (saes ?? []).filter((s) => !s.unusable_reason)
  const layer = lab.position - 1
  const nearest = [...usable].sort((a, b) => Math.abs(a.layer - layer) - Math.abs(b.layer - layer))[0] ?? null
  const sae = usable.find((s) => refKey(s.ref) === choice) ?? nearest

  const request = useMemo(
    (): GeometrySaeRequest | null =>
      sae
        ? { model: run.model, concept: runConceptSpec(run.concept), read: run.read, chat_template: run.chat_template, sae: sae.ref, seed: run.seed }
        : null,
    [run, sae],
  )
  const requestKey = JSON.stringify(request)

  useEffect(() => {
    if (!lab.isAuthenticated || !request) {
      setEstimate(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      apiClient
        .estimateGeometrySae(request)
        .then((e) => !cancelled && setEstimate(e))
        .catch(() => !cancelled && setEstimate(null))
    }, 350)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // requestKey stands for request
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lab.isAuthenticated, requestKey])

  const go = async () => {
    if (!request || !sae) return
    setRunning(true)
    setError(null)
    try {
      const result = await apiClient.saeGeometry(request)
      setShownId(onResult(result, refKey(sae.ref)))
      lab.setPosition(result.position)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') lab.showPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'SAE run failed')
    } finally {
      setRunning(false)
    }
  }

  const pick = (key: string) => {
    setChoice(key)
    const picked = usable.find((s) => refKey(s.ref) === key)
    if (picked) lab.setPosition(picked.layer + 1)
  }

  const mine = results.filter((r) => r.runId === runId)
  const shown = mine.find((r) => r.id === shownId) ?? mine[0] ?? null
  const comparable = shown ? results.filter((r) => r.id !== shown.id && sameValues(r.result.values, shown.result.values)) : []
  const other = comparable.find((r) => String(r.id) === compareId) ?? null
  const isGemma = run.model.startsWith('gemma-3')
  const note = !lab.isAuthenticated ? 'Sign in to read SAEs' : estimate ? estimateNote(estimate) : null

  return (
    <div className="gl-sae">
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{run.concept.name}</strong> · {lab.modelName(run.model)} · read at {READ_LABELS[run.read].toLowerCase()}
        </div>
        {!lab.isAuthenticated ? null : saes == null ? (
          <div className="pl-note">Looking for SAEs…</div>
        ) : usable.length === 0 ? (
          <div className="pl-empty" style={{ borderColor: colors.border }}>
            No SAE reads {lab.modelName(run.model)}&apos;s residual stream. {isGemma ? 'Load Gemma Scope 2 resid_post SAEs on the ' : 'Load one on the '}
            <a href="/sae" target="_blank" rel="noopener noreferrer" style={{ color: colors.accent }}>
              SAE page
            </a>
            .
          </div>
        ) : (
          <div className="gl-sae-pick">
            <div className="pl-control-label" title="SAEs that read this model's residual stream after a layer; choosing one moves the layer">
              SAE
            </div>
            <Dropdown
              value={sae ? refKey(sae.ref) : ''}
              onChange={pick}
              options={(saes ?? []).map((s) => ({ value: refKey(s.ref), label: saeLabel(s), disabled: !!s.unusable_reason }))}
              placeholder="Choose an SAE"
              disabled={running}
            />
          </div>
        )}
        <RunBar
          note={note}
          warn={estimate != null && !estimate.within_limit}
          label="Read with SAE"
          running={running}
          disabled={!lab.isAuthenticated || !sae || (estimate != null && !estimate.within_limit)}
          onRun={go}
        />
        {error && <div className="pl-error-note">{error}</div>}
      </section>

      {shown && (
        <SaeResults
          key={shown.id}
          lab={lab}
          run={run}
          shown={shown}
          mine={mine}
          onShow={setShownId}
          comparable={comparable}
          other={other}
          onCompare={setCompareId}
          running={running}
        />
      )}

      <style>{`
        .geometry-lab .gl-sae-pick { max-width: 640px; }
        .geometry-lab .gl-sae-head { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: flex-end; margin-bottom: 0.875rem; }
        .geometry-lab .gl-sae-head .gl-sae-result { min-width: 220px; max-width: 420px; flex: 1; }
        .geometry-lab .gl-fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 0.75rem; }
        .geometry-lab .gl-field-title { font-size: 0.72rem; margin-bottom: 0.2rem; display: flex; justify-content: space-between; gap: 0.4rem; white-space: nowrap; }
        .geometry-lab .gl-field-title > span { overflow: hidden; text-overflow: ellipsis; }
        .geometry-lab .gl-sae-table td { white-space: nowrap; }
        .geometry-lab .gl-sae-table .gl-sae-example { text-align: left; max-width: 18rem; overflow: hidden; text-overflow: ellipsis; font-size: 0.72rem; }
        .geometry-lab .gl-sae-compare th, .geometry-lab .gl-sae-compare td { white-space: nowrap; }
      `}</style>
    </div>
  )
}

function SaeResults({
  lab,
  run,
  shown,
  mine,
  onShow,
  comparable,
  other,
  onCompare,
  running,
}: {
  lab: GeometryLab
  run: GeometryRun
  shown: StoredSae
  mine: StoredSae[]
  onShow: (id: number) => void
  comparable: StoredSae[]
  other: StoredSae | null
  onCompare: (id: string) => void
  running: boolean
}) {
  const result = shown.result
  const options = result.fits.map((f) => f.dims)
  const [dims, setDims] = useState<number | null>(() => (options.includes(lab.dims) ? lab.dims : options.includes(3) ? 3 : options[0]))
  const fit = fitFor(result, dims)
  const otherFit = other ? fitFor(other.result, dims) : null
  const k = result.values.length
  const info = result.sae.info

  const tiles: Tile[] = [
    { label: 'Verdict', value: VERDICTS[fit.verdict].label, hint: dimsText(fit.dims), title: VERDICTS[fit.verdict].title },
    {
      label: 'To 90%',
      value: fit.n90 != null ? String(fit.n90) : `>${fit.r2.length}`,
      hint: `compact ≤ ${fit.compact}`,
      title: `Fewest greedy features whose fixed contributions write 90% of the shape. A compact group has at most two per dimension the shape spans (${fit.spans}) and at most half the values.`,
    },
    { label: 'All features', value: formatPercent(fit.ceiling), hint: 'of shape', title: "R² of every feature's contribution together; the rest is in the SAE's error" },
    {
      label: 'Co-active',
      value: formatValue(fit.fields.coactive, 1),
      hint: 'per example',
      title: 'Mean number of the features writing 90% that are active on an example: about 1 when they tile the shape',
    },
    {
      label: 'Field',
      value: fit.fields.width != null ? (fit.fields.width * k).toFixed(1) : '–',
      hint: `of ${k} values`,
      title: 'Mean number of values in those features’ receptive fields (where a feature reaches half its peak)',
    },
    { label: 'Unexplained', value: formatPercent(result.sae.fvu), hint: 'variance', title: "Share of the read positions' variance the SAE's reconstructions miss" },
    { label: 'L0', value: result.sae.l0.toFixed(1), hint: 'active per token' },
  ]

  return (
    <div style={{ opacity: running ? 0.55 : 1 }} aria-busy={running}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{info.name}</strong> · {layerName(result.position)} · {formatWidth(info.width)} features · {result.varying_features} vary across values · $
          {result.cost.toFixed(2)}
        </div>
        <div className="gl-sae-head">
          <div>
            <div className="pl-control-label" title="The shape: the class means on their top principal components, or all of their dimensions">
              Dims
            </div>
            <Segmented
              ariaLabel="Subspace size"
              value={dimsLabel(fit.dims)}
              onChange={(v) => setDims(v === 'All' ? null : Number(v))}
              options={options.map((m) => ({ value: dimsLabel(m), label: dimsLabel(m) }))}
            />
          </div>
          {mine.length > 1 && (
            <div className="gl-sae-result">
              <div className="pl-control-label">Result</div>
              <Dropdown value={String(shown.id)} onChange={(v) => onShow(Number(v))} options={mine.map((r) => ({ value: String(r.id), label: r.label }))} />
            </div>
          )}
          {comparable.length > 0 && (
            <div className="gl-sae-result">
              <div className="pl-control-label" title="Another SAE result on the same values (any model or layer)">
                Compare with
              </div>
              <Dropdown
                value={other ? String(other.id) : ''}
                onChange={onCompare}
                options={[{ value: '', label: 'None' }, ...comparable.map((r) => ({ value: String(r.id), label: `${r.label} · ${layerName(r.result.position)}` }))]}
              />
            </div>
          )}
        </div>
        <StatTiles tiles={tiles} />
        <Findings items={saeFindings(result, fit)} />
      </section>

      <div className="pl-results-grid">
        <section className="pl-panel">
          <SupportChart
            fit={fit}
            other={other && otherFit ? { fit: otherFit, label: other.label } : null}
            filename={`${run.concept.name}-${result.model}-L${result.layer}-sae-r2`.replace(/[^\w.-]+/g, '_')}
          />
        </section>
        <section className="pl-panel">
          <TuningStrip run={run} result={result} fit={fit} title={other ? 'Tuning · A' : 'Tuning'} />
        </section>
      </div>

      {other && otherFit && (
        <div className="pl-results-grid">
          <section className="pl-panel">
            <CompareTable a={{ stored: shown, fit }} b={{ stored: other, fit: otherFit }} k={k} />
          </section>
          <section className="pl-panel">
            <TuningStrip run={run} result={other.result} fit={otherFit} title="Tuning · B" />
          </section>
        </div>
      )}

      <section className="pl-panel">
        <FieldsOnShape run={run} result={result} fit={fit} />
      </section>

      <section className="pl-panel">
        <FeatureTable run={run} result={result} fit={fit} />
        <div className="pl-hint" style={{ marginTop: '0.5rem', opacity: 0.75 }}>
          {result.active_features} features active on the concept · {result.sae.tokens} read tokens
        </div>
      </section>
    </div>
  )
}

// Restricted-support R²

interface SupportSeries {
  key: string
  label: string
  values: PerPosition
  color: string
  dash?: string
}

const CHART_HEIGHT = 320
const CHART_MARGIN = { top: 14, right: 96, bottom: 34, left: 40 }

function SupportChart({ fit, other, filename }: { fit: SaeShapeFit; other: { fit: SaeShapeFit; label: string } | null; filename: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ n: number; px: number; py: number } | null>(null)

  const series: SupportSeries[] = [
    { key: 'greedy', label: other ? 'Greedy · A' : 'Greedy', values: fit.r2, color: colors.series[0] },
    { key: 'cosine', label: 'By cosine', values: fit.cosine_r2, color: colors.series[1], dash: '5 4' },
    { key: 'random', label: other ? 'Random · A' : 'Random', values: fit.random.mean, color: colors.baseline, dash: '3 3' },
    ...(other
      ? [
          { key: 'greedy-b', label: 'Greedy · B', values: other.fit.r2, color: colors.series[2] },
          { key: 'random-b', label: 'Random · B', values: other.fit.random.mean, color: colors.baseline, dash: '1 3' },
        ]
      : []),
  ]
  const N = Math.max(1, ...series.map((s) => s.values.length))
  const all = series.flatMap((s) => s.values).filter((v): v is number => v != null)
  const yMin = Math.max(-0.5, Math.min(0, Math.floor(Math.min(...all, 0) * 10) / 10))
  const innerW = Math.max(width - CHART_MARGIN.left - CHART_MARGIN.right, 60)
  const innerH = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom
  const logN = Math.max(Math.log2(N), 1)
  const x = (n: number) => CHART_MARGIN.left + (Math.log2(n) / logN) * innerW
  const y = (v: number) => CHART_MARGIN.top + (1 - (Math.max(yMin, Math.min(1, v)) - yMin) / (1 - yMin)) * innerH
  const yTicks: number[] = []
  for (let t = 1; t >= yMin - 1e-9; t -= 0.25) yTicks.push(Math.round(t * 100) / 100)
  const xTicks: number[] = []
  for (let n = 1; n <= N; n *= 2) xTicks.push(n)
  const path = (values: PerPosition) => linePath(values, (i) => x(i + 1), y)
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const ends = series
    .map((s) => ({ s, v: s.values[s.values.length - 1] }))
    .filter((e): e is { s: SupportSeries; v: number } => e.v != null)
    .map((e) => ({ ...e, py: y(e.v) }))
    .sort((a, b) => a.py - b.py)
  for (let i = 1; i < ends.length; i++) ends[i].py = Math.max(ends[i].py, ends[i - 1].py + 12)

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: px, y: py } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const n = Math.max(1, Math.min(N, Math.round(2 ** (((px - CHART_MARGIN.left) / innerW) * logN))))
    setHover({ n, px, py })
  }

  return (
    <div>
      <div className="pl-card-title">
        <h3 title="R² of the centered class means written by the first n features' fixed contributions (tuning curve × decoder row), in greedy order, by decoder cosine, and in random orders">
          Restricted-support R² · {dimsText(fit.dims)}
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
      <div ref={ref} className="pl-chart">
        <svg ref={svgRef} width={width} height={CHART_HEIGHT} role="img" aria-label="R² against the number of features, greedy, by cosine and random">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
              <text x={CHART_MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {xTicks.map((n) => (
            <text key={n} x={x(n)} y={CHART_MARGIN.top + innerH + 14} textAnchor="middle" fontSize={10} fill={colors.text}>
              {n}
            </text>
          ))}
          <text x={CHART_MARGIN.left + innerW / 2} y={CHART_HEIGHT - 4} textAnchor="middle" fontSize={10} fill={colors.text} opacity={0.7}>
            features
          </text>
          <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(0.9)} y2={y(0.9)} stroke={colors.baseline} strokeDasharray="2 3" />
          <text x={CHART_MARGIN.left + 4} y={y(0.9) - 4} fontSize={10} fill={colors.text} {...halo}>
            90%
          </text>
          {fit.ceiling != null && (
            <>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(fit.ceiling)} y2={y(fit.ceiling)} stroke={colors.series[0]} strokeOpacity={0.35} strokeDasharray="6 3" />
              <text x={CHART_MARGIN.left + innerW / 2} y={y(fit.ceiling) - 4} fontSize={10} fill={colors.text} textAnchor="middle" {...halo}>
                all features {formatPercent(fit.ceiling)}
              </text>
            </>
          )}
          {fit.n90 != null && (
            <line x1={x(fit.n90)} x2={x(fit.n90)} y1={CHART_MARGIN.top} y2={CHART_MARGIN.top + innerH} stroke={colors.series[0]} strokeOpacity={0.35} />
          )}
          {series.map((s) => (
            <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.key.startsWith('greedy') ? 2.25 : 1.5} strokeDasharray={s.dash} strokeLinejoin="round" />
          ))}
          {hover &&
            series.map((s) => {
              const v = s.values[hover.n - 1]
              return v != null ? <circle key={s.key} cx={x(hover.n)} cy={y(v)} r={3.5} fill={s.color} stroke={colors.surface} strokeWidth={1.5} /> : null
            })}
          {ends.map(({ s, py }) => (
            <text key={s.key} x={CHART_MARGIN.left + innerW + 6} y={py} dy="0.32em" fontSize={10} fontWeight={600} fill={colors.text}>
              {s.label}
            </text>
          ))}
          <rect
            x={CHART_MARGIN.left - 6}
            y={CHART_MARGIN.top}
            width={innerW + 12}
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
            title={`${hover.n} feature${hover.n === 1 ? '' : 's'}`}
            rows={series.map((s) => ({ key: s.color, label: s.label, value: formatValue(s.values[hover.n - 1] ?? null, 3) }))}
          />
        )}
      </div>
    </div>
  )
}

// Tuning curves

type StripOrder = 'greedy' | 'peak'
const STRIP_MARGIN = { top: 8, right: 8, bottom: 60, left: 64 }

function stripFeatures(fit: SaeShapeFit): number[] {
  const n = Math.max((fit.n90 ?? fit.order.length) + 4, 8)
  return fit.order.slice(0, Math.min(n, STRIP_ROWS))
}

function TuningStrip({ run, result, fit, title }: { run: GeometryRun; result: GeometrySaeResult; fit: SaeShapeFit; title: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [order, setOrder] = useState<StripOrder>('greedy')
  const [hover, setHover] = useState<{ i: number; j: number; px: number; py: number } | null>(null)
  const values = result.values
  const k = values.length
  const row = useMemo(() => new Map(result.features.ids.map((id, i) => [id, i])), [result])
  const greedy = stripFeatures(fit)
  const rank = new Map(greedy.map((f, i) => [f, i]))
  const features =
    order === 'greedy'
      ? greedy
      : [...greedy].sort((a, b) => argmax(result.features.tuning.values[row.get(a) ?? 0]) - argmax(result.features.tuning.values[row.get(b) ?? 0]) || (rank.get(a) ?? 0) - (rank.get(b) ?? 0))
  const tuning = (f: number) => result.features.tuning.values[row.get(f) ?? 0] ?? []
  const rows = features.length
  const innerW = Math.max(width - STRIP_MARGIN.left - STRIP_MARGIN.right, 60)
  const cellW = innerW / k
  const cellH = Math.max(8, Math.min(16, 380 / Math.max(rows, 1)))
  const innerH = cellH * rows
  const height = innerH + STRIP_MARGIN.top + STRIP_MARGIN.bottom
  const colEvery = Math.max(1, Math.ceil(k / (innerW / 13)))

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: mx, y: my } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const i = Math.floor((my - STRIP_MARGIN.top) / cellH)
    const j = Math.floor((mx - STRIP_MARGIN.left) / cellW)
    setHover(i >= 0 && j >= 0 && i < rows && j < k ? { i, j, px: mx, py: my } : null)
  }

  const hoverRows = (i: number, j: number): TooltipRow[] => {
    const f = features[i]
    const t = tuning(f)
    const peak = Math.max(...t)
    const g = rank.get(f) ?? 0
    const gain = (fit.r2[g] ?? 0) - (g > 0 ? fit.r2[g - 1] ?? 0 : 0)
    return [
      { label: 'mean activation', value: formatValue(t[j], 3) },
      { label: 'of peak', value: peak > 0 ? formatPercent(t[j] / peak) : '–' },
      { label: 'active on', value: formatPercent(result.features.active.values[row.get(f) ?? 0]?.[j]) },
      { label: `ΔR² (greedy #${g + 1})`, value: `+${formatValue(gain, 3)}` },
      { label: 'decoder cosine', value: formatValue(fit.cos[g], 2) },
    ]
  }
  const linkText = (f: number) => {
    const example = result.features.top_example[row.get(f) ?? 0]
    return example != null ? exampleText(run.concept, example) : undefined
  }
  const n90Row = order === 'greedy' && fit.n90 != null && fit.n90 < rows ? fit.n90 : null

  return (
    <div>
      <div className="pl-card-title">
        <h3 title="Each feature's mean activation per value, scaled to its own peak: its receptive field">{title}</h3>
        <div style={{ display: 'inline-flex', gap: '0.4rem', alignItems: 'center' }}>
          <Segmented
            ariaLabel="Order features by"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'greedy', label: 'Greedy', title: 'In the order the greedy fit adds them' },
              { value: 'peak', label: 'Peak', title: 'By the value each one peaks at' },
            ]}
          />
          <button
            type="button"
            className="pl-icon-button"
            title="Download SVG"
            aria-label="Download SVG"
            onClick={() => svgRef.current && downloadSvg(svgRef.current, `${run.concept.name}-L${result.layer}-tuning.svg`.replace(/[^\w.-]+/g, '_'), colors.surface)}
            style={{ color: colors.text }}
          >
            <Download className="pl-icon" />
          </button>
        </div>
      </div>
      <div ref={ref} className="pl-chart">
        <svg ref={svgRef} width={width} height={height} role="img" aria-label="Tuning curves: each feature's mean activation per value">
          <g transform={`translate(${STRIP_MARGIN.left},${STRIP_MARGIN.top})`}>
            {features.map((f, i) => {
              const t = tuning(f)
              const peak = Math.max(...t, 1e-12)
              return t.map((v, j) => (
                <rect key={`${f}-${j}`} x={j * cellW} y={i * cellH} width={cellW + 0.5} height={cellH + 0.5} fill={sequentialColor(colors, Math.max(0, v) / peak)} />
              ))
            })}
            {n90Row != null && (
              <g>
                <line x1={-STRIP_MARGIN.left + 4} x2={innerW} y1={n90Row * cellH} y2={n90Row * cellH} stroke={colors.text} strokeWidth={1.25} strokeDasharray="4 3" />
                <text x={innerW - 2} y={n90Row * cellH - 3} textAnchor="end" fontSize={9} fill={colors.text} stroke={colors.surface} strokeWidth={3} paintOrder="stroke">
                  90%
                </text>
              </g>
            )}
            {hover && <rect x={hover.j * cellW} y={hover.i * cellH} width={cellW} height={cellH} fill="none" stroke={colors.accent} strokeWidth={2} />}
            {features.map((f, i) => (
              <a key={f} href={saeBrowserUrl(result.sae.info.ref, f, linkText(f))} target="_blank" rel="noopener noreferrer">
                <title>Open #{f} in the SAE browser</title>
                <text x={-6} y={(i + 0.5) * cellH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.accent} style={{ fontVariantNumeric: 'tabular-nums' }}>
                  #{f}
                </text>
              </a>
            ))}
            {values.map((label, j) =>
              j % colEvery === 0 || j === k - 1 ? (
                <text key={j} x={(j + 0.5) * cellW} y={innerH + 8} textAnchor="end" fontSize={10} fill={colors.text} transform={`rotate(-45 ${(j + 0.5) * cellW} ${innerH + 8})`}>
                  {label}
                </text>
              ) : null,
            )}
          </g>
          <rect x={STRIP_MARGIN.left} y={STRIP_MARGIN.top} width={innerW} height={innerH} fill="transparent" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hover && (
          <ChartTooltip x={hover.px} y={hover.py} containerWidth={width} title={`#${features[hover.i]} · ${values[hover.j]}`} rows={hoverRows(hover.i, hover.j)} />
        )}
      </div>
      <div className="gl-ramp-legend">
        <span>0</span>
        <span
          className="gl-ramp"
          style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map((t) => sequentialColor(colors, t)).join(', ')})`, borderColor: colors.border }}
        />
        <span>peak</span>
        <span className="pl-hint">· each row on its own scale</span>
      </div>
    </div>
  )
}

// Receptive fields on the shape

const FIELD_SIZE = 150

function FieldsOnShape({ run, result, fit }: { run: GeometryRun; result: GeometrySaeResult; fit: SaeShapeFit }) {
  const colors = useLabColors()
  const means = run.means.values[result.position]
  const row = useMemo(() => new Map(result.features.ids.map((id, i) => [id, i])), [result])
  if (!means || means.length !== result.values.length || (means[0]?.length ?? 0) < 2) return null
  const features = fit.order.slice(0, Math.min(FIELD_PLOTS, Math.max(fit.n90 ?? FIELD_PLOTS, 4)))
  const extent = Math.max(...means.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]), 1e-6) * 1.2
  const pad = 10
  const scale = (FIELD_SIZE / 2 - pad) / extent
  const px = (v: number) => FIELD_SIZE / 2 + v * scale
  const py = (v: number) => FIELD_SIZE / 2 - v * scale
  const order = closedPath(run.concept.structure) ? [...means, means[0]] : means
  const line = run.concept.structure === 'categorical' ? '' : order.map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join('')

  return (
    <div>
      <div className="pl-card-title">
        <h3 title="The class means on their top two components at this layer (the Shape view's plane), shaded and sized by each feature's mean activation">
          Fields on the shape · PC1 × PC2
        </h3>
      </div>
      <div className="gl-fields">
        {features.map((f, n) => {
          const t = result.features.tuning.values[row.get(f) ?? 0] ?? []
          const peak = Math.max(...t, 1e-12)
          const example = result.features.top_example[row.get(f) ?? 0]
          return (
            <div key={f}>
              <div className="gl-field-title">
                <FeatureLink sae={result.sae.info.ref} feature={f} text={example != null ? exampleText(run.concept, example) : undefined} />
                <span style={{ opacity: 0.7 }} title={fieldText(result.values, t, run.concept.structure)}>
                  {fieldText(result.values, t, run.concept.structure)}
                </span>
              </div>
              <svg width={FIELD_SIZE} height={FIELD_SIZE} role="img" aria-label={`Feature ${f}'s mean activation on each value, on the shape`} style={{ display: 'block' }}>
                <rect x={0.5} y={0.5} width={FIELD_SIZE - 1} height={FIELD_SIZE - 1} fill="none" stroke={colors.grid} />
                {line && <path d={line} fill="none" stroke={colors.text} strokeOpacity={0.25} strokeWidth={1} />}
                {means.map((p, v) => {
                  const s = Math.max(0, t[v] ?? 0) / peak
                  return (
                    <circle key={v} cx={px(p[0])} cy={py(p[1])} r={2 + 4.5 * s} fill={sequentialColor(colors, s)} stroke={colors.text} strokeOpacity={0.35} strokeWidth={0.75}>
                      <title>
                        {result.values[v]}: {formatValue(t[v], 3)}
                      </title>
                    </circle>
                  )
                })}
                <text x={FIELD_SIZE - 4} y={FIELD_SIZE - 5} textAnchor="end" fontSize={9} fill={colors.text} opacity={0.6}>
                  #{n + 1}
                </text>
              </svg>
            </div>
          )
        })}
      </div>
      <div className="pl-hint" style={{ marginTop: '0.5rem', opacity: 0.75 }}>
        {layerName(result.position)} · first {features.length} greedy features
      </div>
    </div>
  )
}

// The features

function FeatureTable({ run, result, fit }: { run: GeometryRun; result: GeometrySaeResult; fit: SaeShapeFit }) {
  const colors = useLabColors()
  const row = useMemo(() => new Map(result.features.ids.map((id, i) => [id, i])), [result])
  const ref = result.sae.info.ref

  return (
    <div>
      <div className="pl-card-title">
        <h3 title="Features in the order the greedy fit adds them; the line marks where they reach 90%">Features · greedy</h3>
      </div>
      <div style={{ maxHeight: 360, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
        <table className="pl-table gl-sae-table" aria-label="SAE features in greedy order">
          <thead>
            <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
              <th>#</th>
              <th>Feature</th>
              <th title="R² the feature adds when the greedy fit adds it">ΔR²</th>
              <th title="Values where it reaches half its peak" style={{ textAlign: 'left' }}>
                Field
              </th>
              <th title="The value its mean activation peaks at, and that mean">Peak</th>
              <th title="Share of the peak value's examples it's active on">Active</th>
              <th title="Its decoder row's cosine with the shape's subspace">Cosine</th>
              <th style={{ textAlign: 'left' }}>Strongest on</th>
            </tr>
          </thead>
          <tbody>
            {fit.order.map((f, g) => {
              const i = row.get(f) ?? 0
              const t = result.features.tuning.values[i] ?? []
              const peakAt = argmax(t)
              const example = result.features.top_example[i]
              const text = example != null ? exampleText(run.concept, example) : undefined
              const gain = (fit.r2[g] ?? 0) - (g > 0 ? fit.r2[g - 1] ?? 0 : 0)
              const boundary = fit.n90 != null && g === fit.n90 - 1
              return (
                <tr key={f} style={{ borderTop: `1px solid ${colors.grid}`, borderBottom: boundary ? `2px dashed ${colors.text}` : undefined, cursor: 'default' }}>
                  <td style={{ opacity: 0.6 }}>{g + 1}</td>
                  <td>
                    <FeatureLink sae={ref} feature={f} text={text} />
                  </td>
                  <td>+{formatValue(gain, 3)}</td>
                  <td style={{ textAlign: 'left' }}>{fieldText(result.values, t, run.concept.structure)}</td>
                  <td>
                    {result.values[peakAt]} · {formatValue(t[peakAt], 2)}
                  </td>
                  <td>{formatPercent(result.features.active.values[i]?.[peakAt])}</td>
                  <td>{formatValue(fit.cos[g], 2)}</td>
                  <td className="gl-sae-example" title={text}>
                    {text ?? '–'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Two SAEs side by side

function CompareTable({ a, b, k }: { a: { stored: StoredSae; fit: SaeShapeFit }; b: { stored: StoredSae; fit: SaeShapeFit }; k: number }) {
  const colors = useLabColors()
  const rows: { label: string; title?: string; value: (side: { stored: StoredSae; fit: SaeShapeFit }) => string }[] = [
    { label: 'Model', value: (s) => s.stored.result.model },
    { label: 'Layer', value: (s) => String(s.stored.result.layer) },
    { label: 'Features', value: (s) => formatWidth(s.stored.result.sae.info.width) },
    { label: 'L0', value: (s) => s.stored.result.sae.l0.toFixed(1) },
    { label: 'Unexplained', value: (s) => formatPercent(s.stored.result.sae.fvu) },
    { label: 'Dims', value: (s) => dimsText(s.fit.dims) },
    { label: 'All features', title: 'R² of every feature together', value: (s) => formatPercent(s.fit.ceiling) },
    { label: 'To 90%', value: (s) => (s.fit.n90 != null ? String(s.fit.n90) : `>${s.fit.r2.length}`) },
    { label: 'First 8', title: 'R² of the first 8 greedy features', value: (s) => formatPercent(at(s.fit.r2, 8)) },
    { label: 'Co-active', value: (s) => formatValue(s.fit.fields.coactive, 1) },
    { label: 'Field', title: 'Mean values per receptive field', value: (s) => (s.fit.fields.width != null ? (s.fit.fields.width * k).toFixed(1) : '–') },
    { label: 'Verdict', value: (s) => VERDICTS[s.fit.verdict].label },
  ]
  return (
    <div>
      <div className="pl-card-title">
        <h3>Side by side</h3>
      </div>
      <table className="pl-table gl-sae-compare" aria-label="Two SAE results side by side">
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th />
            <th title={a.stored.label}>A</th>
            <th title={b.stored.label}>B</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{ cursor: 'default' }}>
            <td>SAE</td>
            <td title={a.stored.label}>{a.stored.result.sae.info.name}</td>
            <td title={b.stored.label}>{b.stored.result.sae.info.name}</td>
          </tr>
          {rows.map((r) => (
            <tr key={r.label} style={{ borderTop: `1px solid ${colors.grid}`, cursor: 'default' }}>
              <td title={r.title}>{r.label}</td>
              <td>{r.value(a)}</td>
              <td>{r.value(b)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
