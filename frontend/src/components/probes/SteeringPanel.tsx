/**
 * Steering: generate with α × the probe's direction added to the residual stream after its
 * layer, next to an unsteered baseline (and optionally the direction ablated everywhere).
 *
 * Each generation is read back by the unsteered model: the probe's score of the text (does it
 * now read as the concept?) and its perplexity (did steering break fluency?), plotted against
 * α as two small charts, with the texts side by side below.
 */

import { useEffect, useMemo, useState, type PointerEvent } from 'react'
import type { SteeredGeneration, SteeringEstimate, SteerProbeResponse } from '../../types/probes'
import { apiClient } from '../../api/client'
import type { CausalProbe } from './CausalPanel'
import { ChartTooltip, RunBar, Segmented } from './chartKit'
import { divergingColor, inkOn, useElementWidth, useProbeColors } from './probeTheme'
import { formatScore } from './probeUtils'

const DEFAULT_ALPHAS = '-4, -2, 2, 4'
const MAX_ALPHAS = 6
const MAX_ALPHA = 64
const MAX_TOKENS = 200

function parseAlphas(text: string): { alphas: number[]; error: string | null } {
  const parts = text.split(/[\s,]+/).filter(Boolean)
  const alphas = parts.map(Number)
  if (alphas.some((a) => !Number.isFinite(a))) return { alphas: [], error: 'Strengths must be numbers, e.g. -4, -2, 2, 4' }
  if (alphas.some((a) => Math.abs(a) > MAX_ALPHA)) return { alphas: [], error: `Strengths can be at most ±${MAX_ALPHA}` }
  const unique = [...new Set(alphas.filter((a) => a !== 0))]
  if (unique.length > MAX_ALPHAS) return { alphas: [], error: `At most ${MAX_ALPHAS} strengths per run` }
  return { alphas: unique, error: null }
}

/** Round tick values covering [lo, hi]: steps of 1, 2 or 5 times a power of ten. */
function niceTicks(lo: number, hi: number, count = 4): number[] {
  const raw = (hi - lo) / count
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power
  const ticks: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Math.round(t / step) * step)
  return ticks
}

/** Round tick values for a log axis covering [10^lo, 10^hi]. */
function logTicks(lo: number, hi: number): number[] {
  const ticks: number[] = []
  const multiples = hi - lo > 2 ? [1] : hi - lo > 1 ? [1, 3] : [1, 1.5, 2, 3, 5, 7]
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    for (const m of multiples) {
      const t = m * 10 ** e
      if (Math.log10(t) >= lo && Math.log10(t) <= hi) ticks.push(t)
    }
  }
  return ticks
}

function formatTick(value: number): string {
  return Number.isInteger(value) ? String(value) : formatScore(value)
}

function alphaLabel(alpha: number): string {
  return alpha > 0 ? `+${alpha}` : String(alpha)
}

function generationTitle(g: SteeredGeneration): string {
  if (g.kind === 'baseline') return 'Baseline'
  if (g.kind === 'ablate') return 'Direction ablated'
  return `α = ${alphaLabel(g.alpha ?? 0)}`
}

/** Generations in display order: by α (baseline at 0), the ablated one last. */
function ordered(generations: SteeredGeneration[]): SteeredGeneration[] {
  const onAxis = generations.filter((g) => g.kind !== 'ablate').sort((a, b) => (a.alpha ?? 0) - (b.alpha ?? 0))
  return [...onAxis, ...generations.filter((g) => g.kind === 'ablate')]
}

interface AlphaChartProps {
  title: string
  generations: SteeredGeneration[]
  value: (g: SteeredGeneration) => number | null
  format: (v: number) => string
  log?: boolean
  color: string
  /** Probe score charts: label the threshold and the sides it separates */
  labelNames?: [string, string]
}

const CHART_HEIGHT = 210
const CHART_MARGIN = { top: 16, right: 16, bottom: 34, left: 48 }

function AlphaChart({ title, generations, value, format, log, color, labelNames }: AlphaChartProps) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(360)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)

  const points = generations
    .filter((g) => g.kind !== 'ablate')
    .map((g) => ({ g, alpha: g.alpha ?? 0, v: value(g) }))
    .filter((p): p is { g: SteeredGeneration; alpha: number; v: number } => p.v != null && (!log || p.v > 0))
  const ablated = generations.find((g) => g.kind === 'ablate')
  const ablatedValue = ablated ? value(ablated) : null

  const innerW = Math.max(width - CHART_MARGIN.left - CHART_MARGIN.right, 60)
  const innerH = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom
  const alphas = points.map((p) => p.alpha)
  const aMin = Math.min(0, ...alphas)
  const aMax = Math.max(0, ...alphas)
  const x = (a: number) => CHART_MARGIN.left + (aMax > aMin ? ((a - aMin) / (aMax - aMin)) * innerW : innerW / 2)

  const values = [...points.map((p) => p.v), ...(ablatedValue != null && (!log || ablatedValue > 0) ? [ablatedValue] : [])]
  if (labelNames) values.push(0)
  const tf = (v: number) => (log ? Math.log10(v) : v)
  let lo = Math.min(...values.map(tf))
  let hi = Math.max(...values.map(tf))
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0
    hi = 1
  }
  if (hi - lo < 1e-9) {
    lo -= 0.5
    hi += 0.5
  }
  const pad = (hi - lo) * 0.08
  lo -= pad
  hi += pad
  const y = (v: number) => CHART_MARGIN.top + (1 - (tf(v) - lo) / (hi - lo)) * innerH

  const ticks = log ? logTicks(lo, hi) : niceTicks(lo, hi)

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.alpha).toFixed(1)},${y(p.v).toFixed(1)}`).join('')
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    if (points.length === 0) return
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    let index = 0
    points.forEach((p, i) => {
      if (Math.abs(x(p.alpha) - px) < Math.abs(x(points[index].alpha) - px)) index = i
    })
    setHover({ index, px, py: e.clientY - box.top })
  }

  return (
    <div className="pl-alpha-chart">
      <div className="pl-alpha-chart-title">{title}</div>
      <div ref={ref} className="pl-chart">
        <svg width={width} height={CHART_HEIGHT} role="img" aria-label={`${title} at each steering strength`}>
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
              <text x={CHART_MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                {formatTick(t)}
              </text>
            </g>
          ))}
          <line
            x1={CHART_MARGIN.left}
            x2={CHART_MARGIN.left + innerW}
            y1={CHART_MARGIN.top + innerH}
            y2={CHART_MARGIN.top + innerH}
            stroke={colors.axis}
          />
          {labelNames && (
            <g>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(0)} y2={y(0)} stroke={colors.baseline} />
              <text x={CHART_MARGIN.left + 4} y={CHART_MARGIN.top + 10} fontSize={10} fill={colors.text} {...halo}>
                ↑ {labelNames[1]}
              </text>
              <text x={CHART_MARGIN.left + 4} y={CHART_MARGIN.top + innerH - 4} fontSize={10} fill={colors.text} {...halo}>
                ↓ {labelNames[0]}
              </text>
            </g>
          )}
          {ablatedValue != null && (!log || ablatedValue > 0) && (
            <g>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(ablatedValue)} y2={y(ablatedValue)} stroke={colors.ablated} strokeWidth={1.5} />
              <text x={CHART_MARGIN.left + innerW} y={y(ablatedValue) - 4} textAnchor="end" fontSize={10} fill={colors.text} {...halo}>
                ablated {format(ablatedValue)}
              </text>
            </g>
          )}
          {alphas.map((a) => (
            <text key={a} x={x(a)} y={CHART_MARGIN.top + innerH + 15} textAnchor="middle" fontSize={10} fill={colors.text} fontWeight={a === 0 ? 600 : 400}>
              {a === 0 ? '0' : alphaLabel(a)}
            </text>
          ))}
          <text x={CHART_MARGIN.left + innerW / 2} y={CHART_HEIGHT - 3} textAnchor="middle" fontSize={11} fill={colors.text}>
            α
          </text>
          {hover && points[hover.index] && (
            <line
              x1={x(points[hover.index].alpha)}
              x2={x(points[hover.index].alpha)}
              y1={CHART_MARGIN.top}
              y2={CHART_MARGIN.top + innerH}
              stroke={colors.axis}
            />
          )}
          <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {points.map((p) => (
            <circle
              key={p.alpha}
              cx={x(p.alpha)}
              cy={y(p.v)}
              r={p.alpha === 0 ? 5 : 4}
              fill={p.alpha === 0 ? colors.surface : color}
              stroke={p.alpha === 0 ? color : colors.surface}
              strokeWidth={2}
            />
          ))}
          <rect
            x={CHART_MARGIN.left - 12}
            y={CHART_MARGIN.top}
            width={innerW + 24}
            height={innerH}
            fill="transparent"
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hover && points[hover.index] && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={generationTitle(points[hover.index].g)}
            rows={[{ key: color, label: title.toLowerCase(), value: format(points[hover.index].v) }]}
          >
            <div className="pl-tooltip-text">{points[hover.index].g.text.slice(0, 140) || '(nothing generated)'}</div>
          </ChartTooltip>
        )}
      </div>
    </div>
  )
}

function GenerationCard({
  g,
  probe,
  colorTokens,
}: {
  g: SteeredGeneration
  probe: CausalProbe
  colorTokens: boolean
}) {
  const colors = useProbeColors()
  const scored = g.scored
  const scale =
    probe.scale ||
    (scored ? Math.max(...scored.scores.slice(scored.span[0], scored.span[1]).map(Math.abs), 1e-6) : 1)
  const side = g.probe_score != null ? (g.probe_score > 0 ? 1 : 0) : null
  return (
    <div
      className="pl-gen-card"
      style={{
        borderColor: g.kind === 'baseline' ? colors.text : colors.border,
        background: colors.panel,
        borderTopColor: g.kind === 'ablate' ? colors.ablated : undefined,
      }}
    >
      <div className="pl-gen-head">
        <strong>{generationTitle(g)}</strong>
        {g.kind !== 'baseline' && (
          <span className="pl-note">
            {g.first_divergence == null ? 'same as baseline' : `diverges at token ${g.first_divergence + 1}`}
          </span>
        )}
      </div>
      <div className="pl-gen-text">
        {!g.text.trim() ? (
          <span className="pl-note">(nothing generated)</span>
        ) : colorTokens && scored ? (
          scored.tokens.map((token, i) => {
            if (i < scored.span[0] || i >= scored.span[1]) return null
            const background = divergingColor(colors, scored.scores[i] / scale)
            return (
              <span key={i} className="pl-token" title={formatScore(scored.scores[i])} style={{ background, color: inkOn(background) }}>
                {token}
              </span>
            )
          })
        ) : (
          g.text
        )}
      </div>
      <div className="pl-gen-stats">
        <span title="The probe's pooled score of this text, read by the unsteered model">
          {side != null && <span className="pl-legend-dot" style={{ background: colors.classes[side] }} />}
          score <strong>{g.probe_score != null ? formatScore(g.probe_score) : '–'}</strong>
          {side != null && ` → ${probe.labelNames[side]}`}
        </span>
        <span title="Perplexity of this text under the unsteered model">
          ppl <strong>{g.perplexity != null ? formatScore(g.perplexity) : '–'}</strong>
        </span>
        <span>{g.generated_tokens} tokens</span>
      </div>
    </div>
  )
}

export function SteeringPanel({
  probe,
  isAuthenticated,
  onShowPurchaseCredits,
}: {
  probe: CausalProbe
  isAuthenticated: boolean
  onShowPurchaseCredits?: () => void
}) {
  const colors = useProbeColors()
  const [prompt, setPrompt] = useState('')
  const [alphaText, setAlphaText] = useState(DEFAULT_ALPHAS)
  const [maxTokens, setMaxTokens] = useState(40)
  const [generatedOnly, setGeneratedOnly] = useState(false)
  const [ablate, setAblate] = useState(true)
  const [temperature, setTemperature] = useState(0)
  const [estimate, setEstimate] = useState<SteeringEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ key: string; response: SteerProbeResponse } | null>(null)
  const [colorTokens, setColorTokens] = useState(false)

  const { alphas, error: alphaError } = useMemo(() => parseAlphas(alphaText), [alphaText])
  const canSteer = probe.classGap != null && probe.classGap > 0

  useEffect(() => {
    if (!isAuthenticated || alphaError) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateSteering({ model: probe.model, prompt_chars: prompt.trim().length, num_alphas: alphas.length, ablate, max_tokens: maxTokens })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, probe.model, prompt, alphas.length, ablate, maxTokens, alphaError])

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const response = await apiClient.steerProbe({
        ...probe.ref,
        prompt: prompt.trim(),
        alphas,
        ablate,
        generated_only: generatedOnly,
        max_tokens: maxTokens,
        temperature,
        seed: 0,
      })
      setResult({ key: probe.key, response })
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Steering failed')
    } finally {
      setRunning(false)
    }
  }

  const response = result?.response ?? null
  const stale = result != null && result.key !== probe.key
  const generations = response ? ordered(response.generations) : []
  const lineColor = response ? colors.methods[response.method] : colors.methods[probe.method]

  return (
    <div className="pl-steering">
      <div className="pl-card-title">
        <h3>Steering · layer {probe.layer}</h3>
      </div>

      {!canSteer ? (
        <div className="pl-empty" style={{ borderColor: colors.border }}>
          {probe.classGap == null
            ? 'Saved without a class gap. Save it again from a training run to steer.'
            : "The class means don't differ along this direction."}
        </div>
      ) : (
        <>
          <textarea
            className="pl-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Prompt to continue…"
            maxLength={2000}
            rows={2}
            style={{ borderColor: colors.border, background: colors.panel, color: colors.text }}
          />
          <div className="pl-steer-controls">
            <label>
              <div className="pl-control-label">
                Strengths <span style={{ textTransform: 'none' }}>α</span>
              </div>
              <input
                className="pl-input"
                title={`In class gaps: α = 1 adds ${formatScore(probe.classGap ?? 0)} along the direction`}
                value={alphaText}
                onChange={(e) => setAlphaText(e.target.value)}
                aria-invalid={alphaError != null}
                style={{ width: '11rem', borderColor: alphaError ? '#ef4444' : colors.border, background: colors.panel, color: colors.text }}
              />
            </label>
            <label>
              <div className="pl-control-label">Max tokens</div>
              <input
                className="pl-input"
                type="number"
                min={1}
                max={MAX_TOKENS}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Math.max(1, Math.min(MAX_TOKENS, Math.floor(Number(e.target.value) || 1))))}
                style={{ width: '5.5rem', borderColor: colors.border, background: colors.panel, color: colors.text }}
              />
            </label>
            <label>
              <div className="pl-control-label">Temperature</div>
              <input
                className="pl-input"
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                title="0 decodes greedily; above 0, every column samples from the same seed"
                onChange={(e) => setTemperature(Math.max(0, Math.min(2, Number(e.target.value) || 0)))}
                style={{ width: '5.5rem', borderColor: colors.border, background: colors.panel, color: colors.text }}
              />
            </label>
            <div>
              <div className="pl-control-label">Steer</div>
              <Segmented
                ariaLabel="Positions to steer"
                value={generatedOnly ? 'generated' : 'all'}
                onChange={(v) => setGeneratedOnly(v === 'generated')}
                options={[
                  { value: 'all', label: 'All tokens', title: 'Add the vector at the prompt’s tokens too' },
                  { value: 'generated', label: 'Generated only', title: 'Add the vector only at generated tokens' },
                ]}
              />
            </div>
            <label className="pl-check-row" style={{ alignSelf: 'center' }}>
              <input type="checkbox" checked={ablate} onChange={(e) => setAblate(e.target.checked)} />
              Also ablate
            </label>
          </div>
          {alphaError && <div className="pl-error-note">{alphaError}</div>}
          <RunBar
            note={
              !isAuthenticated
                ? 'Sign in to steer'
                : estimate
                  ? `${estimate.generations} generations · ≈ $${estimate.estimated_cost.toFixed(2)}`
                  : null
            }
            label="Steer"
            running={running}
            disabled={!isAuthenticated || !prompt.trim() || alphaError != null || (alphas.length === 0 && !ablate)}
            onRun={run}
          />
        </>
      )}

      {error && <div className="pl-error-note">{error}</div>}

      {response && (
        <div className="pl-result" style={{ borderColor: colors.border, opacity: running ? 0.55 : 1 }}>
          <div className="pl-run-meta">
            {stale && <strong style={{ color: colors.warning }}>From a different probe · </strong>}
            “{response.prompt.length > 80 ? `${response.prompt.slice(0, 80)}…` : response.prompt}” · layer {response.layer} ·{' '}
            {response.generated_only ? 'generated tokens only' : 'all tokens'} ·{' '}
            {response.temperature > 0 ? `temperature ${response.temperature}` : 'greedy'} · ${response.cost.toFixed(2)}
          </div>
          {generations.some((g) => g.kind === 'steer') && (
            <div className="pl-alpha-charts">
              <AlphaChart
                title="Probe score"
                generations={generations}
                value={(g) => g.probe_score}
                format={formatScore}
                color={lineColor}
                labelNames={probe.labelNames}
              />
              <AlphaChart
                title="Perplexity (log)"
                generations={generations}
                value={(g) => g.perplexity}
                format={formatScore}
                log
                color={lineColor}
              />
            </div>
          )}
          <label className="pl-check-row" style={{ fontSize: '0.75rem', margin: '0.25rem 0 0.6rem' }}>
            <input type="checkbox" checked={colorTokens} onChange={(e) => setColorTokens(e.target.checked)} />
            Color tokens by probe score
          </label>
          <div className="pl-gen-grid">
            {generations.map((g) => (
              <GenerationCard key={`${g.kind}:${g.alpha}`} g={g} probe={probe} colorTokens={colorTokens} />
            ))}
          </div>
        </div>
      )}

      <style>{`
        .probe-lab .pl-steer-controls { display: flex; flex-wrap: wrap; gap: 0.75rem 1.1rem; align-items: flex-end; margin-top: 0.6rem; }
        .probe-lab .pl-alpha-charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-bottom: 0.5rem; }
        .probe-lab .pl-alpha-chart-title { font-size: 0.75rem; font-weight: 600; margin-bottom: 0.25rem; }
        .probe-lab .pl-tooltip-text { margin-top: 0.3rem; font-size: 0.72rem; white-space: normal; max-width: 220px; opacity: 0.9; }
        .probe-lab .pl-gen-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 0.6rem; }
        .probe-lab .pl-gen-card { border: 1px solid; border-top-width: 3px; padding: 0.6rem 0.7rem; display: flex; flex-direction: column; gap: 0.45rem; min-width: 0; }
        .probe-lab .pl-gen-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; font-size: 0.8rem; }
        .probe-lab .pl-gen-text {
          font-size: 0.8rem;
          line-height: 1.55;
          white-space: pre-wrap;
          word-break: break-word;
          max-height: 220px;
          overflow-y: auto;
          flex: 1;
        }
        .probe-lab .pl-gen-text .pl-token { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.75rem; }
        .probe-lab .pl-gen-stats { display: flex; flex-wrap: wrap; gap: 0.35rem 0.8rem; font-size: 0.72rem; font-variant-numeric: tabular-nums; }
        .probe-lab .pl-gen-stats .pl-legend-dot { display: inline-block; margin-right: 0.3rem; vertical-align: 0; }
        @media (max-width: 900px) {
          .probe-lab .pl-alpha-charts { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  )
}
