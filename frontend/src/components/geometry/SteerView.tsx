/**
 * The Steer view: does the model use the shape? An eval prompt ("Today is {x}. Tomorrow is"),
 * filled with a source value, is edited at the selected layer at each step of a walk to a target
 * value: along the spline through the class means in their top components (manifold), and along
 * the straight line between the two class means (linear). Each step's next-token distribution is
 * read three ways: the values' token probabilities (a step × value heatmap per method, on one
 * color scale), their total, and the Hellinger distance to the nearest distribution the unedited
 * prompt gives for any value. Results are kept per run.
 */

import { useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Download, Spline } from 'lucide-react'
import { apiClient } from '../../api/client'
import { runConceptSpec, type GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRun, PerPosition, SteerEstimate, SteerMethodId, SteerMethodResult, SteerRequest, SteerResult, TopToken } from '../../types/geometry'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding, type Tile, type TooltipRow } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { READ_LABELS, downloadSvg, formatPercent, formatValue, layerName, linePath, sequentialColor, valueColors } from './geometryUtils'

const METHODS: { id: SteerMethodId; label: string; title: string }[] = [
  { id: 'manifold', label: 'Manifold', title: "Sets the activation's coordinates in the subspace to the spline's point and leaves the rest alone" },
  { id: 'linear', label: 'Linear', title: "Adds t × (target's class mean − source's class mean)" },
]
const STEP_OPTIONS = [8, 16, 32]
const DIM_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 16]
const MAX_DIMS = 16
// Values named in a finding's path before it's cut short
const PATH_NAMES = 8
// One method's peak Hellinger distance counts as lower than the other's below this share of it
const CLEARLY_CLOSER = 0.85

function nonNull(values: (number | null | undefined)[]): number[] {
  return values.filter((v): v is number => v != null && !Number.isNaN(v))
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function formatProb(p: number | null | undefined): string {
  if (p == null) return '–'
  return p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`
}

function topText(top: TopToken[]): string {
  return top.map((t) => `${JSON.stringify(t.token)} ${formatProb(t.prob)}`).join(' · ')
}

function dimOptions(k: number, ordinal: boolean): number[] {
  const most = Math.min(k - 1, MAX_DIMS)
  const options = DIM_OPTIONS.filter((m) => m <= most && (m > 1 || ordinal))
  if (!options.includes(most)) options.push(most)
  return options
}

function defaultTarget(run: GeometryRun): number {
  const k = run.concept.values.length
  return run.concept.structure === 'cyclic' ? Math.floor(k / 2) : k - 1
}

function estimateNote(estimate: SteerEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ${estimate.eval_passes} passes · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

/** The value a step's walk is at: a value's name, or the two it's between. */
function pathLabel(values: string[], s: number): string {
  const k = values.length
  const near = Math.round(s)
  if (Math.abs(s - near) < 1e-6) return values[((near % k) + k) % k]
  const lo = Math.floor(s)
  return `${values[((lo % k) + k) % k]}–${values[(((lo + 1) % k) + k) % k]}`
}

/** The values a walk passes, in order (its path rounded to whole values). */
function passedValues(path: number[], k: number): number[] {
  const passed: number[] = []
  for (const s of path) {
    const v = ((Math.round(s) % k) + k) % k
    if (passed[passed.length - 1] !== v) passed.push(v)
  }
  return passed
}

/** A method's nearest unsteered values in order, repeats merged. */
function nearestSequence(method: SteerMethodResult): number[] {
  const sequence: number[] = []
  for (const v of method.nearest) if (sequence[sequence.length - 1] !== v) sequence.push(v)
  return sequence
}

function sequenceText(sequence: number[], values: string[]): string {
  const names = sequence.map((v) => values[v])
  if (names.length <= PATH_NAMES) return names.join(' → ')
  return [...names.slice(0, 3), '…', ...names.slice(-3)].join(' → ')
}

function steerFindings(result: SteerResult): Finding[] {
  const findings: Finding[] = []
  const { values, methods } = result
  const k = values.length
  const manifold = methods.manifold
  const linear = methods.linear
  const last = result.path.length - 1

  const unchanged = Math.max(...nonNull(manifold.hellinger), ...nonNull(linear.hellinger)) < 1e-3
  if (unchanged) {
    findings.push({
      level: 'warning',
      text:
        result.position === result.num_layers
          ? "Edits after the last layer never reach the prompt's last token: steer at an earlier layer."
          : "The edits don't change the readout at this layer.",
    })
    return findings
  }

  const passed = passedValues(result.path, k)
  const visits = (method: SteerMethodResult) => passed.filter((v) => method.nearest.includes(v)).length
  const manifoldVisits = visits(manifold)
  const linearVisits = visits(linear)
  findings.push({
    level: manifoldVisits > linearVisits && manifoldVisits >= Math.min(3, passed.length) ? 'good' : 'info',
    text: `Nearest unsteered behavior — manifold: ${sequenceText(nearestSequence(manifold), values)}; linear: ${sequenceText(nearestSequence(linear), values)}.`,
  })

  // An end reaches the target's behavior when it's as close to it as values' own distributions are to their nearest neighbor's
  const target = values[result.target]
  const spread = median(nonNull(result.reference.nearest_other))
  const manifoldEnd = manifold.to_target[last]
  const linearEnd = linear.to_target[last]
  const reaches = (end: number | null) => spread != null && end != null && end <= spread
  if (spread != null && !reaches(manifoldEnd) && !reaches(linearEnd)) {
    findings.push({
      level: 'warning',
      text: `Neither walk reaches ${target}'s behavior: they end ${formatValue(manifoldEnd)} (manifold) and ${formatValue(linearEnd)} (linear) from it; values sit ${formatValue(spread)} apart.`,
    })
  } else if (spread != null && !reaches(manifoldEnd)) {
    findings.push({
      level: 'warning',
      text: `Manifold steering ends ${formatValue(manifoldEnd)} from ${target}'s behavior (values sit ${formatValue(spread)} apart): the subspace misses part of the value; try more dims.`,
    })
  }

  const manifoldPeak = Math.max(...nonNull(manifold.hellinger))
  const linearPeak = Math.max(...nonNull(linear.hellinger))
  const peaks = `Hellinger ≤ ${formatValue(manifoldPeak)} (manifold) vs ${formatValue(linearPeak)} (linear); values sit ${formatValue(spread)} apart`
  if (manifoldPeak <= CLEARLY_CLOSER * linearPeak) {
    findings.push({ level: 'good', text: `Manifold steering stays closer to real behavior: ${peaks}.` })
  } else if (linearPeak <= CLEARLY_CLOSER * manifoldPeak) {
    findings.push({ level: 'info', text: `Linear steering stays closer to real behavior: ${peaks}.` })
  } else {
    findings.push({ level: 'info', text: `Both stay about as close to real behavior: ${peaks}.` })
  }

  const manifoldMass = Math.min(...nonNull(manifold.mass))
  const linearMass = Math.min(...nonNull(linear.mass))
  const unsteered = result.reference.mass[result.source]
  if (unsteered != null && Math.min(manifoldMass, linearMass) < 0.75 * unsteered) {
    findings.push({
      level: 'info',
      text: `Value-token mass falls to ${formatPercent(manifoldMass)} (manifold) and ${formatPercent(linearMass)} (linear), from ${formatPercent(unsteered)} unsteered.`,
    })
  }

  if (result.eval_decoding != null && result.eval_decoding < 0.5) {
    findings.push({
      level: 'warning',
      text: `In the eval prompt, the subspace reads ${formatPercent(result.eval_decoding)} of values as themselves (chance ${formatPercent(1 / k)}): it may not carry the value in this context.`,
    })
  }
  const dropped = k - result.readable.length
  if (dropped > 0) {
    findings.push({ level: 'warning', text: `${dropped} of ${k} values share their first token with another and aren't read.` })
  }
  return findings
}

export function SteerView({ lab }: { lab: GeometryLab }) {
  const [results, setResults] = useState<Record<number, SteerResult>>({})
  const { run, runId } = lab
  if (!run || runId == null) {
    return (
      <section className="pl-panel">
        <Placeholder icon={Spline}>{lab.running ? 'Reading every layer…' : 'Run a concept, then steer along its shape.'}</Placeholder>
      </section>
    )
  }
  if (run.concept.structure === 'categorical') {
    return (
      <section className="pl-panel">
        <Placeholder icon={Spline}>Steering follows the values' order; categorical concepts have none.</Placeholder>
      </section>
    )
  }
  return (
    <SteerBody
      key={runId}
      lab={lab}
      run={run}
      result={results[runId] ?? null}
      onResult={(r) => setResults((current) => ({ ...current, [runId]: r }))}
    />
  )
}

function SteerBody({ lab, run, result, onResult }: { lab: GeometryLab; run: GeometryRun; result: SteerResult | null; onResult: (r: SteerResult) => void }) {
  const colors = useLabColors()
  const values = run.concept.values
  const k = values.length
  const evalTemplates = run.concept.eval_templates
  const options = dimOptions(k, run.concept.structure === 'ordinal')
  const [source, setSource] = useState(0)
  const [target, setTarget] = useState(() => defaultTarget(run))
  const [steps, setSteps] = useState(16)
  const [dims, setDims] = useState(() => (options.includes(lab.dims) ? lab.dims : Math.min(3, k - 1)))
  const [evalTemplate, setEvalTemplate] = useState(evalTemplates[0] ?? '')
  const [estimate, setEstimate] = useState<SteerEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const position = lab.position

  const templateOk = evalTemplate.split('{x}').length === 2 && evalTemplate.replace('{x}', '').trim().length > 0
  const problem = !evalTemplate.trim()
    ? 'Write an eval prompt'
    : !templateOk
      ? 'The eval prompt needs exactly one {x}'
      : source === target
        ? 'Choose two different values'
        : null

  const request = useMemo(
    (): SteerRequest => ({
      model: run.model,
      concept: runConceptSpec(run.concept),
      read: run.read,
      chat_template: run.chat_template,
      position,
      dims,
      source,
      target,
      steps,
      eval_template: evalTemplate.trim(),
    }),
    [run, position, dims, source, target, steps, evalTemplate],
  )
  const requestKey = JSON.stringify(request)

  useEffect(() => {
    if (!lab.isAuthenticated || problem) {
      setEstimate(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      apiClient
        .estimateGeometrySteer(request)
        .then((e) => !cancelled && setEstimate(e))
        .catch(() => !cancelled && setEstimate(null))
    }, 350)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // requestKey stands for request
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lab.isAuthenticated, requestKey, problem])

  const steer = async () => {
    setRunning(true)
    setError(null)
    try {
      onResult(await apiClient.steerGeometry(request))
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') lab.showPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Steering failed')
    } finally {
      setRunning(false)
    }
  }

  const valueOptions = values.map((v, i) => ({ value: String(i), label: v }))
  const field = { borderColor: colors.border, background: colors.panel, color: colors.text }
  const note = !lab.isAuthenticated ? 'Sign in to steer' : problem ?? (estimate ? estimateNote(estimate) : null)

  return (
    <div className="gl-steer">
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{run.concept.name}</strong> · {lab.modelName(run.model)} · read at {READ_LABELS[run.read].toLowerCase()}
        </div>
        <div className="pl-controls-row">
          <div className="gl-slider">
            <div className="pl-control-label">{layerName(position)}</div>
            <input
              type="range"
              min={0}
              max={run.num_layers}
              value={position}
              onChange={(e) => lab.setPosition(Number(e.target.value))}
              aria-label="Layer"
              aria-valuetext={layerName(position)}
            />
          </div>
          <div>
            <div className="pl-control-label" title="The class means' top components the manifold lives in">
              Dims
            </div>
            <Segmented ariaLabel="Subspace size" value={String(dims)} onChange={(v) => setDims(Number(v))} options={options.map((m) => ({ value: String(m), label: String(m) }))} />
          </div>
          <div>
            <div className="pl-control-label" title="Steps of the walk; each method edits the prompt at steps + 1 points">
              Steps
            </div>
            <Segmented ariaLabel="Steps" value={String(steps)} onChange={(v) => setSteps(Number(v))} options={STEP_OPTIONS.map((n) => ({ value: String(n), label: String(n) }))} />
          </div>
        </div>
        <div className="pl-controls-row">
          <div className="gl-value-pick">
            <div className="pl-control-label" title="The value filled into the eval prompt, where the walk starts">
              From
            </div>
            <Dropdown value={String(source)} onChange={(v) => setSource(Number(v))} options={valueOptions} />
          </div>
          <div className="gl-value-pick">
            <div className="pl-control-label" title={run.concept.structure === 'cyclic' ? 'Where the walk ends, the shorter way round' : 'Where the walk ends'}>
              To
            </div>
            <Dropdown value={String(target)} onChange={(v) => setTarget(Number(v))} options={valueOptions} />
          </div>
          <div className="gl-eval">
            <div className="pl-control-label" title="Ends where the model would say a value; {x} is the source value. The readout is the next token.">
              Eval prompt
            </div>
            <input
              className="pl-input"
              value={evalTemplate}
              onChange={(e) => setEvalTemplate(e.target.value)}
              placeholder="Today is {x}. Tomorrow is"
              maxLength={300}
              aria-invalid={!templateOk}
              style={{ ...field, borderColor: evalTemplate && !templateOk ? '#ef4444' : colors.border }}
            />
            {evalTemplates.length > 1 && (
              <div className="gl-eval-picks">
                {evalTemplates.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className="pl-link"
                    onClick={() => setEvalTemplate(t)}
                    style={{ color: t === evalTemplate ? colors.text : colors.accent, fontWeight: t === evalTemplate ? 600 : 400 }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <RunBar
          note={note}
          warn={problem != null || (estimate != null && !estimate.within_limit)}
          label="Steer"
          running={running}
          disabled={!lab.isAuthenticated || problem != null || (estimate != null && !estimate.within_limit)}
          onRun={steer}
        />
        {error && <div className="pl-error-note">{error}</div>}
      </section>

      {result && <SteerResults run={run} result={result} running={running} />}

      <style>{`
        .geometry-lab .gl-value-pick { min-width: 150px; }
        .geometry-lab .gl-eval { flex: 1; min-width: 260px; }
        .geometry-lab .gl-eval-picks { display: flex; flex-wrap: wrap; gap: 0.25rem 0.9rem; margin-top: 0.35rem; }
        .geometry-lab .gl-steer-prompt { font-size: 0.75rem; margin: -0.4rem 0 0.875rem; }
        .geometry-lab .gl-steer-prompt code { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
      `}</style>
    </div>
  )
}

function SteerResults({ run, result, running }: { run: GeometryRun; result: SteerResult; running: boolean }) {
  const colors = useLabColors()
  const fills = useMemo(() => valueColors(run.concept, colors), [run, colors])
  const { values, methods, reference } = result
  const k = values.length
  const columns = result.readable.map((v) => values[v])
  const largest = Math.max(1e-6, ...nonNull([reference.probs, methods.manifold.probs, methods.linear.probs].flat(2)))
  const steps = result.path.length
  const stepLabels = result.path.map((s, i) => (i === 0 || i === steps - 1 || Math.abs(s - Math.round(s)) < 1e-6 ? pathLabel(values, s) : ''))
  const methodColor: Record<SteerMethodId, string> = { manifold: colors.series[0], linear: colors.series[1] }
  const spread = median(nonNull(reference.nearest_other))
  const unsteeredMass = reference.mass[result.source]

  const tiles: Tile[] = [
    ...METHODS.map(({ id, label }) => ({
      label: `Mass · ${label}`,
      value: formatPercent(Math.min(...nonNull(methods[id].mass))),
      hint: `lowest · unsteered ${formatPercent(unsteeredMass)}`,
      title: "Lowest total probability on the values' tokens along the walk",
    })),
    ...METHODS.map(({ id, label }) => ({
      label: `Hellinger · ${label}`,
      value: formatValue(Math.max(...nonNull(methods[id].hellinger))),
      hint: `peak · values ${formatValue(spread)} apart`,
      title: "Largest Hellinger distance along the walk to the nearest distribution the unedited prompt gives for any value; the hint is how far values' own distributions typically are from each other",
    })),
    {
      label: 'Eval reads',
      value: formatPercent(result.eval_decoding),
      hint: `chance ${formatPercent(1 / k)}`,
      title: "Share of values whose activation in the eval prompt is nearest its own class mean in the subspace",
    },
  ]

  const stepInfo = (id: SteerMethodId) => (i: number): { title: string; rows: TooltipRow[]; extra: ReactNode } => {
    const method = methods[id]
    return {
      title: `${METHODS.find((m) => m.id === id)?.label} · t = ${(i / (steps - 1)).toFixed(2)} · at ${pathLabel(values, result.path[i])}`,
      rows: [
        { label: 'on values', value: formatProb(method.mass[i]) },
        { label: `Hellinger to ${values[method.nearest[i]]} (nearest)`, value: formatValue(method.hellinger[i]) },
        { label: `Hellinger to ${values[result.target]}`, value: formatValue(method.to_target[i]) },
        { label: 'edit / activation', value: formatValue(method.edit_size[i]) },
      ],
      extra: <div className="pl-tooltip-text" style={{ marginTop: '0.3rem', whiteSpace: 'normal' }}>Top: {topText(method.top[i])}</div>,
    }
  }
  const referenceInfo = (i: number) => ({
    title: `Unsteered · ${values[i]}`,
    rows: [
      { label: 'on values', value: formatProb(reference.mass[i]) },
      { label: 'Hellinger to nearest other', value: formatValue(reference.nearest_other[i]) },
    ],
    extra: <div className="pl-tooltip-text" style={{ marginTop: '0.3rem', whiteSpace: 'normal' }}>Top: {topText(reference.top[i])}</div>,
  })
  const base = `${run.concept.name}-${result.model}-${layerName(result.position)}-${values[result.source]}-${values[result.target]}`.replace(/[^\w.-]+/g, '_')

  return (
    <div style={{ opacity: running ? 0.55 : 1 }} aria-busy={running}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>
            {values[result.source]} → {values[result.target]}
          </strong>{' '}
          · {layerName(result.position)} · {result.dims} dims ({formatPercent(result.explained)} of class-mean variance) · ${result.cost.toFixed(2)}
        </div>
        <div className="gl-steer-prompt">
          Edits <code>{result.edited_tokens.map((t) => JSON.stringify(t)).join(' ')}</code> in <code>{JSON.stringify(result.prompt)}</code>
          {result.prefix.length > 0 && (
            <>
              {' '}
              + <code>{result.prefix.map((t) => JSON.stringify(t)).join(' ')}</code>
            </>
          )}
          ; reads the next token.
        </div>
        <StatTiles tiles={tiles} />
        <Findings items={steerFindings(result)} />
      </section>

      <div className="pl-results-grid">
        {METHODS.map(({ id, label, title }) => (
          <section key={id} className="pl-panel">
            <ProbHeatmap
              title={label}
              titleAttr={title}
              color={methodColor[id]}
              rowLabels={stepLabels}
              columns={columns}
              probs={methods[id].probs}
              largest={largest}
              info={stepInfo(id)}
              filename={`${base}-${id}`}
              ariaLabel={`${label} steering: probability of each value's token at each step`}
            />
          </section>
        ))}
      </div>

      <div className="pl-results-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title="Total probability on the values' tokens at each step">Value mass</h3>
          </div>
          <StepChart
            result={result}
            series={METHODS.map(({ id, label }) => ({ key: id, label, values: methods[id].mass, color: methodColor[id] }))}
            refs={unsteeredMass != null ? [{ value: unsteeredMass, label: `unsteered ${formatPercent(unsteeredMass)}` }] : []}
            format={formatPercent}
            top={1}
          />
          <div className="pl-card-title" style={{ marginTop: '1rem' }}>
            <h3 title="Hellinger distance from each step's next-token distribution to the nearest one the unedited prompt gives for any value">
              Hellinger to nearest value
            </h3>
          </div>
          <StepChart
            result={result}
            series={METHODS.map(({ id, label }) => ({ key: id, label, values: methods[id].hellinger, color: methodColor[id] }))}
            refs={spread != null ? [{ value: spread, label: `values ${formatValue(spread)} apart` }] : []}
            format={(v) => formatValue(v)}
            top={Math.min(1, Math.max(0.1, Math.ceil(Math.max(...nonNull(methods.manifold.hellinger), ...nonNull(methods.linear.hellinger), spread ?? 0) * 12) / 10))}
          />
        </section>
        <section className="pl-panel">
          <PathPlane run={run} result={result} fills={fills} methodColor={methodColor} filename={`${base}-path`} />
        </section>
      </div>

      <section className="pl-panel">
        <ProbHeatmap
          title="Unsteered"
          titleAttr="The eval prompt filled with each value, unedited: what the model does by itself"
          rowLabels={values}
          columns={columns}
          probs={reference.probs}
          largest={largest}
          info={referenceInfo}
          filename={`${base}-unsteered`}
          ariaLabel="Unsteered: probability of each value's token with the prompt filled with each value"
          rowAxis="Prompt value"
        />
        <div className="gl-ramp-legend">
          <span>0</span>
          <span
            className="gl-ramp"
            style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map((t) => sequentialColor(colors, t)).join(', ')})`, borderColor: colors.border }}
          />
          <span>{formatProb(largest)}</span>
          <span className="pl-hint">· one scale for all three heatmaps · outlined: each row's most likely value</span>
        </div>
      </section>
    </div>
  )
}

// Heatmaps

const HEAT_MARGIN = { top: 8, right: 8, bottom: 60, left: 92 }

function ProbHeatmap({
  title,
  titleAttr,
  color,
  rowLabels,
  columns,
  probs,
  largest,
  info,
  filename,
  ariaLabel,
  rowAxis = 'Step',
}: {
  title: string
  titleAttr: string
  color?: string
  rowLabels: string[]
  columns: string[]
  probs: (number | null)[][]
  largest: number
  info: (row: number) => { title: string; rows: TooltipRow[]; extra: ReactNode }
  filename: string
  ariaLabel: string
  rowAxis?: string
}) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ i: number; j: number; px: number; py: number } | null>(null)
  const rows = probs.length
  const cols = columns.length
  const innerW = Math.max(width - HEAT_MARGIN.left - HEAT_MARGIN.right, 60)
  const cellW = innerW / cols
  const cellH = Math.max(6, Math.min(18, 340 / rows))
  const innerH = cellH * rows
  const height = innerH + HEAT_MARGIN.top + HEAT_MARGIN.bottom
  const colEvery = Math.max(1, Math.ceil(cols / (innerW / 13)))
  const rowEvery = Math.max(1, Math.ceil(11 / cellH))
  const argmax = probs.map((row) => {
    let best = -1
    let bestValue = -Infinity
    row.forEach((v, j) => {
      if (v != null && v > bestValue) {
        bestValue = v
        best = j
      }
    })
    return best
  })

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: mx, y: my } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const i = Math.floor((my - HEAT_MARGIN.top) / cellH)
    const j = Math.floor((mx - HEAT_MARGIN.left) / cellW)
    setHover(i >= 0 && j >= 0 && i < rows && j < cols ? { i, j, px: mx, py: my } : null)
  }
  const hovered = hover ? info(hover.i) : null

  return (
    <div>
      <div className="pl-card-title">
        <h3 title={titleAttr} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          {color && <span className="pl-legend-line" style={{ background: color, display: 'inline-block' }} />}
          {title}
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
        <svg ref={svgRef} width={width} height={height} role="img" aria-label={ariaLabel}>
          <g transform={`translate(${HEAT_MARGIN.left},${HEAT_MARGIN.top})`}>
            {probs.map((row, i) =>
              row.map((v, j) => (
                <rect key={`${i}-${j}`} x={j * cellW} y={i * cellH} width={cellW + 0.5} height={cellH + 0.5} fill={sequentialColor(colors, (v ?? 0) / largest)} />
              )),
            )}
            {argmax.map((j, i) =>
              j >= 0 ? (
                <rect key={i} x={j * cellW + 0.75} y={i * cellH + 0.75} width={Math.max(cellW - 1.5, 1)} height={Math.max(cellH - 1.5, 1)} fill="none" stroke={colors.text} strokeWidth={1.25} />
              ) : null,
            )}
            {hover && <rect x={hover.j * cellW} y={hover.i * cellH} width={cellW} height={cellH} fill="none" stroke={colors.accent} strokeWidth={2} />}
            {rowLabels.map((label, i) =>
              label && (i % rowEvery === 0 || i === rows - 1) ? (
                <text key={i} x={-6} y={(i + 0.5) * cellH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
                  {label}
                </text>
              ) : null,
            )}
            {columns.map((label, j) =>
              j % colEvery === 0 || j === cols - 1 ? (
                <text
                  key={j}
                  x={(j + 0.5) * cellW}
                  y={innerH + 8}
                  textAnchor="end"
                  fontSize={10}
                  fill={colors.text}
                  transform={`rotate(-45 ${(j + 0.5) * cellW} ${innerH + 8})`}
                >
                  {label}
                </text>
              ) : null,
            )}
            <text x={-HEAT_MARGIN.left + 2} y={-2} fontSize={10} fill={colors.text} opacity={0.7}>
              {rowAxis} ↓
            </text>
          </g>
          <rect x={HEAT_MARGIN.left} y={HEAT_MARGIN.top} width={innerW} height={innerH} fill="transparent" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hover && hovered && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={hovered.title}
            rows={[{ label: `P(${columns[hover.j]})`, value: formatProb(probs[hover.i][hover.j]) }, ...hovered.rows]}
          >
            {hovered.extra}
          </ChartTooltip>
        )}
      </div>
    </div>
  )
}

// Curves

interface StepSeries {
  key: string
  label: string
  values: PerPosition
  color: string
}

const STEP_HEIGHT = 190
const STEP_MARGIN = { top: 12, right: 72, bottom: 34, left: 44 }

function StepChart({
  result,
  series,
  refs,
  format,
  top,
}: {
  result: SteerResult
  series: StepSeries[]
  refs: { value: number; label: string }[]
  format: (v: number | null | undefined) => string
  top: number
}) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const steps = result.path.length
  const innerW = Math.max(width - STEP_MARGIN.left - STEP_MARGIN.right, 60)
  const innerH = STEP_HEIGHT - STEP_MARGIN.top - STEP_MARGIN.bottom
  const x = (i: number) => STEP_MARGIN.left + (i / (steps - 1)) * innerW
  const y = (v: number) => STEP_MARGIN.top + (1 - Math.max(0, Math.min(top, v)) / top) * innerH
  const ticks = Array.from({ length: 5 }, (_, i) => (i * top) / 4)
  const indexAt = (px: number) => Math.max(0, Math.min(steps - 1, Math.round(((px - STEP_MARGIN.left) / innerW) * (steps - 1))))
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const endLabels = series
    .filter((s) => s.values[steps - 1] != null)
    .map((s) => ({ s, py: y(s.values[steps - 1] as number) }))
    .sort((a, b) => a.py - b.py)
  for (let i = 1; i < endLabels.length; i++) endLabels[i].py = Math.max(endLabels[i].py, endLabels[i - 1].py + 12)

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={STEP_HEIGHT} role="img" aria-label="Each method at every step of the walk">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={STEP_MARGIN.left} x2={STEP_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <text x={STEP_MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
              {format(t)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={STEP_MARGIN.left + t * innerW} y={STEP_MARGIN.top + innerH + 14} textAnchor="middle" fontSize={10} fill={colors.text}>
            {t === 0 ? result.values[result.source] : t === 1 ? result.values[result.target] : `t ${t}`}
          </text>
        ))}
        {refs.map((r) => (
          <g key={r.label}>
            <line x1={STEP_MARGIN.left} x2={STEP_MARGIN.left + innerW} y1={y(r.value)} y2={y(r.value)} stroke={colors.baseline} strokeDasharray="4 3" />
            <text x={STEP_MARGIN.left + 4} y={y(r.value) - 4} fontSize={10} fill={colors.text} {...halo}>
              {r.label}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <path key={s.key} d={linePath(s.values, x, y)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {hover && (
          <>
            <line x1={x(hover.index)} x2={x(hover.index)} y1={STEP_MARGIN.top} y2={STEP_MARGIN.top + innerH} stroke={colors.axis} />
            {series.map((s) =>
              s.values[hover.index] != null ? (
                <circle key={s.key} cx={x(hover.index)} cy={y(s.values[hover.index] as number)} r={4} fill={s.color} stroke={colors.surface} strokeWidth={2} />
              ) : null,
            )}
          </>
        )}
        {endLabels.map(({ s, py }) => (
          <text key={s.key} x={STEP_MARGIN.left + innerW + 6} y={py} dy="0.32em" fontSize={10} fontWeight={600} fill={colors.text}>
            {s.label}
          </text>
        ))}
        <rect
          x={STEP_MARGIN.left - 6}
          y={STEP_MARGIN.top}
          width={innerW + 12}
          height={innerH}
          fill="transparent"
          onPointerMove={(e) => {
            const { x: px, y: py } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
            setHover({ index: indexAt(px), px, py })
          }}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          title={`t = ${(hover.index / (steps - 1)).toFixed(2)} · at ${pathLabel(result.values, result.path[hover.index])}`}
          rows={series.map((s) => ({ key: s.color, label: s.label, value: format(s.values[hover.index]) }))}
        />
      )}
    </div>
  )
}

// The walk on the class means' plane

const PLANE_HEIGHT = 360
const PLANE_MARGIN = { top: 12, right: 12, bottom: 30, left: 12 }

type PlaneTarget = { kind: 'mean' | 'eval' | SteerMethodId; index: number; x: number; y: number }

function PathPlane({
  run,
  result,
  fills,
  methodColor,
  filename,
}: {
  run: GeometryRun
  result: SteerResult
  fills: string[]
  methodColor: Record<SteerMethodId, string>
  filename: string
}) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ target: PlaneTarget; px: number; py: number } | null>(null)
  const values = result.values
  const k = values.length
  const steps = result.path.length

  // Orient the plane like the Shape view's at this layer (components' signs are arbitrary)
  const signs = useMemo(() => {
    const shape = run.means.values[result.position]
    return [0, 1].map((c) => {
      let dot = 0
      result.plot.means.forEach((p, v) => (dot += (p[c] ?? 0) * (shape?.[v]?.[c] ?? 0)))
      return dot < 0 ? -1 : 1
    })
  }, [run, result])
  const orient = (points: number[][]) => points.map((p) => [(p[0] ?? 0) * signs[0], (p[1] ?? 0) * signs[1]])
  const plot = useMemo(
    () => ({
      means: orient(result.plot.means),
      eval: orient(result.plot.eval),
      curve: orient(result.plot.curve),
      manifold: orient(result.plot.manifold),
      linear: orient(result.plot.linear),
    }),
    // orient depends only on signs
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [result, signs],
  )

  const innerW = Math.max(width - PLANE_MARGIN.left - PLANE_MARGIN.right, 60)
  const innerH = PLANE_HEIGHT - PLANE_MARGIN.top - PLANE_MARGIN.bottom
  const extent = Math.max(...[...plot.means, ...plot.eval, ...plot.manifold, ...plot.linear].flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]), 1e-6) * 1.15
  const scale = Math.min(innerW, innerH) / (2 * extent)
  const px = (v: number) => PLANE_MARGIN.left + innerW / 2 + v * scale
  const py = (v: number) => PLANE_MARGIN.top + innerH / 2 - v * scale
  const path = (points: number[][]) => points.map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join('')
  const labelEvery = Math.max(1, Math.ceil(k / 24))

  const targets: PlaneTarget[] = [
    ...plot.means.map((p, index) => ({ kind: 'mean' as const, index, x: p[0], y: p[1] })),
    ...plot.eval.map((p, index) => ({ kind: 'eval' as const, index, x: p[0], y: p[1] })),
    ...plot.manifold.map((p, index) => ({ kind: 'manifold' as const, index, x: p[0], y: p[1] })),
    ...plot.linear.map((p, index) => ({ kind: 'linear' as const, index, x: p[0], y: p[1] })),
  ]
  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: mx, y: my } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    let best: PlaneTarget | null = null
    let bestDist = 14 ** 2
    for (const t of targets) {
      const d = (px(t.x) - mx) ** 2 + (py(t.y) - my) ** 2
      if (d < bestDist) {
        bestDist = d
        best = t
      }
    }
    setHover(best ? { target: best, px: mx, py: my } : null)
  }
  const hoverRows = (t: PlaneTarget): { title: string; rows: TooltipRow[] } => {
    if (t.kind === 'mean') return { title: values[t.index], rows: [{ key: fills[t.index], label: 'class mean', value: values[t.index] }] }
    if (t.kind === 'eval') return { title: values[t.index], rows: [{ key: fills[t.index], label: 'eval prompt, unedited', value: values[t.index] }] }
    const method = result.methods[t.kind]
    return {
      title: `${METHODS.find((m) => m.id === t.kind)?.label} · t = ${(t.index / (steps - 1)).toFixed(2)}`,
      rows: [
        { key: methodColor[t.kind], label: 'at', value: pathLabel(values, result.path[t.index]) },
        { label: 'nearest behavior', value: values[method.nearest[t.index]] },
      ],
    }
  }
  const hovered = hover ? hoverRows(hover.target) : null

  return (
    <div ref={ref} className="pl-chart">
      <div className="pl-card-title">
        <h3 title="The class means and the edited activations on the class means' top two components (the Shape view's plane at this layer)">
          Path · PC1 × PC2
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
      <svg ref={svgRef} width={width} height={PLANE_HEIGHT} role="img" aria-label={`The walk from ${values[result.source]} to ${values[result.target]} on the class means' top two components`}>
        <rect x={PLANE_MARGIN.left} y={PLANE_MARGIN.top} width={innerW} height={innerH} fill="none" stroke={colors.grid} />
        <path d={path(plot.curve)} fill="none" stroke={colors.text} strokeWidth={1.25} opacity={0.35} />
        {plot.eval.map((p, v) => (
          <circle key={`e${v}`} cx={px(p[0])} cy={py(p[1])} r={3.5} fill={colors.surface} stroke={fills[v]} strokeWidth={1.5} />
        ))}
        {plot.means.map((p, v) => (
          <circle key={`m${v}`} cx={px(p[0])} cy={py(p[1])} r={5} fill={fills[v]} stroke={colors.surface} strokeWidth={2} />
        ))}
        {METHODS.map(({ id }) => (
          <g key={id}>
            <path
              d={path(plot[id])}
              fill="none"
              stroke={methodColor[id]}
              strokeWidth={2}
              strokeDasharray={id === 'linear' ? '5 4' : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {plot[id].map((p, i) => (
              <circle key={i} cx={px(p[0])} cy={py(p[1])} r={i === 0 || i === steps - 1 ? 3.5 : 2} fill={methodColor[id]} />
            ))}
          </g>
        ))}
        {plot.means.map((p, v) =>
          v % labelEvery === 0 || v === k - 1 || v === result.source || v === result.target ? (
            <text
              key={v}
              x={px(p[0]) + 7}
              y={py(p[1]) - 7}
              fontSize={10}
              fontWeight={v === result.source || v === result.target ? 700 : 600}
              fill={colors.text}
              stroke={colors.surface}
              strokeWidth={3}
              paintOrder="stroke"
            >
              {values[v]}
            </text>
          ) : null,
        )}
        {hover && <circle cx={px(hover.target.x)} cy={py(hover.target.y)} r={7} fill="none" stroke={colors.text} strokeWidth={1.5} />}
        <text x={PLANE_MARGIN.left + innerW / 2} y={PLANE_HEIGHT - 8} textAnchor="middle" fontSize={11} fill={colors.text}>
          PC1 × PC2 · {layerName(result.position)}
        </text>
        <rect x={PLANE_MARGIN.left} y={PLANE_MARGIN.top} width={innerW} height={innerH} fill="transparent" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && hovered && <ChartTooltip x={hover.px} y={hover.py} containerWidth={width} title={hovered.title} rows={hovered.rows} />}
      <div className="pl-legend">
        {METHODS.map(({ id, label }) => (
          <span key={id} className="pl-legend-item">
            <span
              className="pl-legend-line"
              style={{ background: id === 'linear' ? `repeating-linear-gradient(to right, ${methodColor[id]} 0 5px, transparent 5px 9px)` : methodColor[id] }}
            />
            {label}
          </span>
        ))}
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.text }} />
          Class means
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-ring" style={{ borderColor: colors.text }} />
          Eval prompt
        </span>
      </div>
    </div>
  )
}
