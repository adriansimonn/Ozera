/**
 * Directional ablation: project the probe's direction out of the residual stream at every
 * layer, then check (1) whether the model's behaviour changes, against control directions that
 * remove as much of the stream, and (2) whether probes at every layer still read the concept,
 * both the clean probes ("frozen") and probes refit without the direction ("retrained").
 */

import { useEffect, useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { AblateProbeResponse, AblationCurve, PerPosition, ProbeEstimate } from '../../types/probes'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { apiClient } from '../../api/client'
import type { CausalProbe } from './CausalPanel'
import { ChartTooltip, Findings, RunBar, Segmented, StatTiles, type Finding, type TooltipRow } from './chartKit'
import { useElementWidth, useProbeColors, type ProbeColors } from './probeTheme'
import { estimateText, formatMetric, formatPercent, positionLabel } from './probeUtils'

const CURVE_LABELS: Record<AblationCurve, string> = {
  clean: 'Clean model',
  probe_frozen: 'Same probes, direction ablated',
  probe_retrained: 'Refit probes, direction ablated',
  control_frozen: 'Same probes, controls ablated',
  control_retrained: 'Refit probes, controls ablated',
}

const CURVE_SHORT: Record<AblationCurve, string> = {
  clean: 'clean',
  probe_frozen: 'same probes',
  probe_retrained: 'refit',
  control_frozen: 'controls (same)',
  control_retrained: 'controls',
}

function best(values: PerPosition, from: number): { value: number; position: number } | null {
  let found: { value: number; position: number } | null = null
  values.forEach((v, i) => {
    if (i >= from && v != null && (found == null || v > found.value)) found = { value: v, position: i }
  })
  return found
}

function findings(result: AblateProbeResponse): Finding[] {
  const out: Finding[] = []
  const { probe, control } = result.behaviour
  const [lo, hi] = control.kl_range ?? [control.kl, control.kl]
  const kl = (v: number) => formatMetric(v, 3)
  const vs = `KL ${kl(probe.kl)} vs controls ${kl(lo)}–${kl(hi)}`
  if (probe.kl > hi) {
    out.push({ level: 'good', text: `Changes predictions more than any control (${vs}).` })
  } else if (probe.kl < lo) {
    out.push({ level: 'warning', text: `Changes predictions less than every control (${vs}).` })
  } else {
    out.push({ level: 'info', text: `Changes predictions about as much as the controls (${vs}).` })
  }

  const from = result.layer + 1
  const refit = best(result.curves.probe_retrained, from)
  const clean = best(result.curves.clean, from)
  const controls = best(result.curves.control_retrained, from)
  if (refit && clean) {
    const scores = `AUROC ${formatMetric(refit.value, 2)} vs ${formatMetric(clean.value, 2)} clean${
      controls ? `, ${formatMetric(controls.value, 2)} controls` : ''
    }`
    if (refit.value >= Math.max(0.75, clean.value - 0.05)) {
      out.push({ level: 'warning', text: `Refit probes still read the concept (${scores}): it's also carried elsewhere.` })
    } else if (refit.value <= 0.6) {
      out.push({ level: 'good', text: `Refit probes can't read the concept (${scores}): this direction carries it.` })
    } else {
      out.push({ level: 'info', text: `Refit probes read less of the concept (${scores}): partly carried elsewhere.` })
    }
  }
  return out
}

const HEIGHT = 280
const MARGIN = { top: 26, right: 132, bottom: 40, left: 48 }

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

function seriesStyle(colors: ProbeColors, result: AblateProbeResponse): Record<'clean' | 'probe_frozen' | 'probe_retrained' | 'control_retrained', { color: string; width: number; dash?: string }> {
  return {
    clean: { color: colors.methods[result.method], width: 2 },
    probe_retrained: { color: colors.ablated, width: 2 },
    probe_frozen: { color: colors.ablated, width: 1.5, dash: '5 4' },
    control_retrained: { color: colors.baseline, width: 1.5 },
  }
}

function AblationChart({ result }: { result: AblateProbeResponse }) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const [focus, setFocus] = useState<number | null>(null)

  const positions = result.num_layers + 1
  const styles = seriesStyle(colors, result)
  const shown = Object.keys(styles) as (keyof typeof styles)[]
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 60)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => MARGIN.left + (positions > 1 ? i * step : innerW / 2)
  const all = shown.flatMap((k) => result.curves[k]).filter((v): v is number => v != null)
  const yMin = Math.min(0.4, Math.floor(Math.min(...all) * 10) / 10)
  const y = (v: number) => MARGIN.top + (1 - (v - yMin) / (1 - yMin)) * innerH
  const tickStep = 1 - yMin > 0.6 ? 0.2 : 0.1
  const ticks: number[] = []
  for (let t = 1; t >= yMin - 1e-9; t -= tickStep) ticks.push(Math.round(t * 100) / 100)
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))
  const probePosition = result.layer + 1
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  // Direct labels at the right end, nudged apart so they don't collide
  const ends = shown
    .map((k) => {
      const values = result.curves[k]
      let last = values.length - 1
      while (last >= 0 && values[last] == null) last--
      return { k, y: last >= 0 ? y(values[last] as number) : null }
    })
    .filter((e): e is { k: (typeof shown)[number]; y: number } => e.y != null)
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12)

  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - MARGIN.left) / step)))
  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    setHover({ index: indexAt(px), px, py: e.clientY - box.top })
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const current = focus ?? probePosition
    if (e.key === 'ArrowLeft') setFocus(Math.max(0, current - 1))
    else if (e.key === 'ArrowRight') setFocus(Math.min(positions - 1, current + 1))
    else return
    e.preventDefault()
  }
  const active = hover?.index ?? focus
  const tooltipRows = (i: number): TooltipRow[] =>
    (['clean', 'probe_retrained', 'probe_frozen', 'control_retrained', 'control_frozen'] as AblationCurve[]).map((k) => ({
      key: k in styles ? styles[k as keyof typeof styles].color : undefined,
      label: CURVE_LABELS[k],
      value: formatMetric(result.curves[k][i]),
    }))

  return (
    <div
      ref={ref}
      className="pl-chart"
      tabIndex={0}
      role="group"
      aria-label="Test AUROC at every layer, clean and with the direction ablated. Use arrow keys to read layers."
      onKeyDown={onKeyDown}
      onBlur={() => setFocus(null)}
    >
      <svg width={width} height={HEIGHT} role="img" aria-hidden="true">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
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

        <rect x={x(probePosition) - Math.max(step, 8) / 2} y={MARGIN.top} width={Math.max(step, 8)} height={innerH} fill={colors.hover} />
        <text x={x(probePosition)} y={MARGIN.top - 10} textAnchor="middle" fontSize={11} fontWeight={600} fill={colors.text} {...halo}>
          probe&apos;s layer {result.layer}
        </text>
        {yMin < 0.5 && (
          <g>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(0.5)} y2={y(0.5)} stroke={colors.baseline} opacity={0.7} />
            <text x={MARGIN.left + 4} y={y(0.5) - 4} fontSize={10} fill={colors.text} {...halo}>
              chance
            </text>
          </g>
        )}

        {shown.map((k) => (
          <path
            key={k}
            d={linePath(result.curves[k], x, y)}
            fill="none"
            stroke={styles[k].color}
            strokeWidth={styles[k].width}
            strokeDasharray={styles[k].dash}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        {ends.map((e) => (
          <text key={e.k} x={MARGIN.left + innerW + 8} y={e.y} dy="0.32em" fontSize={10.5} fill={colors.text}>
            <tspan fill={styles[e.k].color} fontWeight={700}>
              —{' '}
            </tspan>
            {CURVE_SHORT[e.k]}
          </text>
        ))}

        {active != null && (
          <line x1={x(active)} x2={x(active)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} />
        )}
        <rect
          x={MARGIN.left - step / 2}
          y={MARGIN.top}
          width={innerW + step}
          height={innerH}
          fill="transparent"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {active != null && (
        <ChartTooltip
          x={hover?.px ?? x(active)}
          y={hover?.py ?? MARGIN.top + 20}
          containerWidth={width}
          title={active === 0 ? 'Embeddings' : `Layer ${active - 1}`}
          rows={tooltipRows(active)}
        />
      )}
    </div>
  )
}

function AblationTable({ result }: { result: AblateProbeResponse }) {
  const colors = useProbeColors()
  const keys: AblationCurve[] = ['clean', 'probe_frozen', 'probe_retrained', 'control_frozen', 'control_retrained']
  return (
    <div style={{ maxHeight: 320, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
      <table className="pl-table" aria-label="Test AUROC per layer, clean and ablated">
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th>Layer</th>
            {keys.map((k) => (
              <th key={k}>{CURVE_LABELS[k]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: result.num_layers + 1 }, (_, i) => (
            <tr key={i} style={{ borderTop: `1px solid ${colors.grid}`, background: i === result.layer + 1 ? colors.hover : undefined, cursor: 'default' }}>
              <td style={{ fontWeight: i === result.layer + 1 ? 600 : 400 }}>{positionLabel(i)}</td>
              {keys.map((k) => (
                <td key={k}>{formatMetric(result.curves[k][i])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function AblationPanel({
  probe,
  lab,
  onShowPurchaseCredits,
}: {
  probe: CausalProbe
  lab: ProbeLab
  onShowPurchaseCredits?: () => void
}) {
  const colors = useProbeColors()
  const [estimate, setEstimate] = useState<ProbeEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ key: string; response: AblateProbeResponse } | null>(null)
  const [view, setView] = useState<'chart' | 'table'>('chart')

  const stats = lab.stats
  const instruct = lab.models.find((m) => m.model_id === probe.model)?.is_instruct ?? false
  const chatMismatch = probe.chatTemplate && !instruct

  useEffect(() => {
    if (!lab.isAuthenticated || !stats || chatMismatch) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateProbeRun({
          model: probe.model,
          num_examples: stats.numRows,
          total_chars: stats.chars,
          chat_template: probe.chatTemplate,
          kind: 'ablate',
          method: probe.method,
        })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [lab.isAuthenticated, stats, probe.model, probe.chatTemplate, probe.method, chatMismatch])

  const run = async () => {
    const dataset = lab.datasetSpec()
    if (!dataset) return
    setRunning(true)
    setError(null)
    try {
      const response = await apiClient.ablateProbe({
        ...probe.ref,
        dataset,
        test_fraction: lab.testFraction,
        seed: lab.seed,
      })
      setResult({ key: probe.key, response })
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Ablation failed')
    } finally {
      setRunning(false)
    }
  }

  const response = result?.response ?? null
  const stale = result != null && result.key !== probe.key
  const styles = response ? seriesStyle(colors, response) : null
  const otherDataset = stats && probe.datasetName && probe.datasetName !== stats.name

  return (
    <div className="pl-ablation">
      <div className="pl-card-title">
        <h3>Directional ablation</h3>
      </div>

      <div className="pl-note">
        Dataset: {stats ? `${stats.name} · ${stats.numRows} examples` : 'none chosen'}
      </div>
      {otherDataset && (
        <div className="pl-note" style={{ marginTop: '0.4rem', color: colors.warning }}>
          Probe trained on {probe.datasetName}
        </div>
      )}
      {chatMismatch && (
        <div className="pl-error-note">Needs a chat template, which {probe.model} doesn&apos;t have</div>
      )}

      <RunBar
        note={!lab.isAuthenticated ? 'Sign in to run ablations' : estimate ? estimateText(estimate) : null}
        warn={estimate != null && !estimate.within_limit}
        label="Run ablation"
        running={running}
        disabled={!lab.isAuthenticated || !stats || chatMismatch || (estimate != null && !estimate.within_limit)}
        onRun={run}
      />

      {error && <div className="pl-error-note">{error}</div>}

      {response && styles && (
        <div className="pl-result" style={{ borderColor: colors.border, opacity: running ? 0.55 : 1 }}>
          <div className="pl-run-meta">
            {stale && <strong style={{ color: colors.warning }}>From a different probe · </strong>}
            <strong>{response.dataset.name}</strong> · {response.model} · {response.dataset.n_train} train / {response.dataset.n_test}{' '}
            test · ${response.cost.toFixed(2)}
          </div>

          <StatTiles
            tiles={[
              {
                label: 'Next-token KL',
                value: formatMetric(response.behaviour.probe.kl, 3),
                hint: response.behaviour.control.kl_range
                  ? `controls ${formatMetric(response.behaviour.control.kl_range[0], 3)}–${formatMetric(response.behaviour.control.kl_range[1], 3)}`
                  : `controls ${formatMetric(response.behaviour.control.kl, 3)}`,
              },
              {
                label: 'Top-1 kept',
                value: formatPercent(response.behaviour.probe.top1_agreement),
                hint: `controls ${formatPercent(response.behaviour.control.top1_agreement)}`,
              },
              {
                label: 'Stream removed',
                value: formatPercent(response.behaviour.probe.energy_fraction),
                hint: 'same for controls',
              },
            ]}
          />
          <Findings items={findings(response)} />

          <div className="pl-card-title">
            <h3>Test AUROC by layer</h3>
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
          {view === 'chart' ? <AblationChart result={response} /> : <AblationTable result={response} />}
          <div className="pl-legend">
            {(['clean', 'probe_retrained', 'probe_frozen', 'control_retrained'] as const).map((k) => (
              <span key={k} className="pl-legend-item">
                <span
                  className="pl-legend-line"
                  style={{
                    background: styles[k].dash
                      ? `repeating-linear-gradient(to right, ${styles[k].color} 0 5px, transparent 5px 9px)`
                      : styles[k].color,
                  }}
                />
                {CURVE_LABELS[k]}
              </span>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}
