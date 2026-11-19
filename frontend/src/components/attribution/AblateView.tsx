/**
 * The Ablate view: remove chosen heads and layers on the clean prompts (zeroed, set to their
 * mean over the pair set, or resampled from the corrupted prompt) and read the metric, next to
 * random sets of as many components.
 */

import type { AttributionLab } from '../../hooks/useAttributionLab'
import type { AblateResponse, AblationMethod } from '../../types/attribution'
import { ABLATION_METHODS } from '../../types/attribution'
import { Findings, Placeholder, RunBar, type Finding } from '../lab/chartKit'
import { useElementWidth, useLabColors } from '../lab/labTheme'
import { METHOD_LABELS, METHOD_TITLES, componentName, effect, effectUnit, estimateText, headName } from './attributionUtils'
import { FigureExport } from './FigureExport'

const QUICK_HEADS = 5

function range(values: number[]): [number, number] | null {
  return values.length > 0 ? [Math.min(...values), Math.max(...values)] : null
}

function findings(response: AblateResponse): Finding[] {
  const out: Finding[] = []
  for (const method of ABLATION_METHODS) {
    const r = response.methods[method]
    if (!r) continue
    const random = range(r.random)
    const name = METHOD_LABELS[method]
    if (!random) {
      out.push({ level: 'info', text: `${name}: ${effect(r.value, 2)}. Too few other components for random sets.` })
    } else if (r.value < random[0]) {
      out.push({ level: 'good', text: `${name}: ${effect(r.value, 2)}, below every random set (${effect(random[0], 2)} to ${effect(random[1], 2)}).` })
    } else {
      out.push({ level: 'warning', text: `${name}: ${effect(r.value, 2)}, no lower than random sets (${effect(random[0], 2)} to ${effect(random[1], 2)}).` })
    }
  }
  return out
}

const HEIGHT = 260
const MARGIN = { top: 16, right: 16, bottom: 34, left: 44 }

function Bars({ response }: { response: AblateResponse }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const methods = ABLATION_METHODS.filter((m) => response.methods[m])
  const all = methods.flatMap((m) => {
    const r = response.methods[m] as NonNullable<AblateResponse['methods'][AblationMethod]>
    return [r.value - r.se, r.value + r.se, ...r.random]
  })
  const top = response.normalized ? 1 : 0
  const low = Math.min(0, ...all)
  const high = Math.max(top, ...all)
  const span = high - low || 1
  const innerW = Math.max(Math.min(width, 560) - MARGIN.left - MARGIN.right, 120)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const y = (v: number) => MARGIN.top + (1 - (v - low) / span) * innerH
  const band = innerW / methods.length
  const barW = Math.min(46, band * 0.32)
  const ticks = [low, 0, ...(response.normalized ? [1] : []), high].filter((t, i, list) => list.findIndex((o) => Math.abs(o - t) < span * 0.04) === i)

  return (
    <div ref={ref} className="pl-chart">
      <svg width={innerW + MARGIN.left + MARGIN.right} height={HEIGHT} role="img" aria-label="The metric with the chosen components ablated, next to random sets">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
            <text x={MARGIN.left - 7} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
              {effect(t, 2)}
            </text>
          </g>
        ))}
        {response.normalized && (
          <text x={MARGIN.left + innerW} y={y(1) - 5} textAnchor="end" fontSize={10} fill={colors.dim}>
            clean
          </text>
        )}
        {methods.map((method, i) => {
          const r = response.methods[method] as NonNullable<AblateResponse['methods'][AblationMethod]>
          const center = MARGIN.left + band * (i + 0.5)
          const chosenX = center - barW - 4
          const randomX = center + 4 + barW / 2
          return (
            <g key={method}>
              <rect x={chosenX} y={Math.min(y(r.value), y(0))} width={barW} height={Math.max(Math.abs(y(r.value) - y(0)), 1)} fill={colors.ablated}>
                <title>{`${METHOD_LABELS[method]}, chosen: ${effect(r.value)} ± ${effect(r.se)}`}</title>
              </rect>
              {r.se > 0 && <line x1={chosenX + barW / 2} x2={chosenX + barW / 2} y1={y(r.value - r.se)} y2={y(r.value + r.se)} stroke={colors.text} strokeWidth={1.5} />}
              {r.random.map((value, k) => (
                <circle key={k} cx={randomX + (k - (r.random.length - 1) / 2) * 5} cy={y(value)} r={3.5} fill={colors.baseline} stroke={colors.surface} strokeWidth={1}>
                  <title>{`${METHOD_LABELS[method]}, random set ${k + 1}: ${effect(value)}`}</title>
                </circle>
              ))}
              <text x={center} y={HEIGHT - 12} textAnchor="middle" fontSize={11} fill={colors.text}>
                {METHOD_LABELS[method]}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.ablated }} />
          Chosen components
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.baseline }} />
          Random sets of as many
        </span>
      </div>
    </div>
  )
}

function Result({ lab, response, runId }: { lab: AttributionLab; response: AblateResponse; runId: string | null }) {
  const stale = runId != null && lab.run != null && lab.run.run_id !== runId
  const unit = effectUnit(response)
  const methods = ABLATION_METHODS.filter((m) => response.methods[m])
  const names = response.components.map(componentName).join(', ')
  return (
    <div className="pl-result">
      <div className="pl-run-meta">
        {stale && <strong style={{ color: 'var(--oz-warn)' }}>From another sweep · </strong>}
        <strong>{names}</strong> · {response.model} · {response.num_pairs} pairs · ${response.cost.toFixed(2)}
      </div>
      <Findings items={findings(response)} />
      <div className="at-compare">
        <div>
          <div className="pl-card-title">
            <h3 title={response.normalized ? 'The metric on the clean prompts: 1 is the clean run, 0 the corrupted one' : `The metric's change from the corrupted run (${unit})`}>
              Metric when ablated
            </h3>
            <FigureExport
              filename={`ablation-${response.model}`}
              figure={() => ({
                kind: 'bars',
                title: `Ablating ${names}: ${response.model}`,
                y_label: response.normalized ? 'Metric (clean 1, corrupted 0)' : `Metric (${unit})`,
                baseline: response.normalized ? 1 : undefined,
                baseline_label: 'Clean run',
                bars: methods.flatMap((m) => {
                  const r = response.methods[m] as NonNullable<AblateResponse['methods'][AblationMethod]>
                  const mean = r.random.length ? r.random.reduce((a, b) => a + b, 0) / r.random.length : null
                  return [
                    { label: `${METHOD_LABELS[m]}: chosen`, value: r.value, error: r.se },
                    ...(mean != null ? [{ label: `${METHOD_LABELS[m]}: random`, value: mean, marks: r.random }] : []),
                  ]
                }),
              })}
            />
          </div>
          <Bars response={response} />
        </div>
        <div>
          <div className="pl-card-title">
            <h3>By ablation</h3>
          </div>
          <table className="pl-table at-table">
            <thead>
              <tr>
                <th>Ablation</th>
                <th>Chosen</th>
                <th>±</th>
                <th>Random sets</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((m) => {
                const r = response.methods[m] as NonNullable<AblateResponse['methods'][AblationMethod]>
                const random = range(r.random)
                return (
                  <tr key={m} title={METHOD_TITLES[m] + (m === 'mean' && response.mean_over ? ` (${response.mean_over === 'position' ? 'position by position' : 'over all tokens'})` : '')}>
                    <td>{METHOD_LABELS[m]}</td>
                    <td>{effect(r.value)}</td>
                    <td>{effect(r.se)}</td>
                    <td>{random ? `${effect(random[0], 2)} to ${effect(random[1], 2)}` : '–'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export function AblateView({ lab }: { lab: AttributionLab }) {
  const run = lab.run
  if (!run && !lab.ablated) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="patching">Run a sweep, then ablate the components it points to.</Placeholder>
      </section>
    )
  }
  const { components, error } = lab.parsedComponents
  const estimate = lab.ablateEstimate
  const note = !lab.isAuthenticated
    ? 'Sign in to run'
    : !run
      ? 'Run a sweep first'
      : error
        ? error
        : components.length === 0
          ? 'Name components to ablate'
          : estimate
            ? `${components.length} component${components.length === 1 ? '' : 's'} · ${estimateText(estimate)}`
            : `${components.length} component${components.length === 1 ? '' : 's'}`

  // The sweep's largest heads, by exact effect where every one of them has been patched
  const fillTop = () => {
    if (!run) return
    const heads = run.heads.flatMap((row, layer) => row.map((estimated, head) => ({ layer, head, estimated })))
    const exact = (h: { layer: number; head: number }) => lab.exactHeads[headName(h.layer, h.head)]?.effect
    const allExact = heads.every((h) => exact(h) != null)
    const size = (h: (typeof heads)[number]) => Math.abs(allExact ? (exact(h) as number) : h.estimated)
    const top = [...heads].sort((a, b) => size(b) - size(a)).slice(0, QUICK_HEADS)
    lab.setComponentsText(top.map((h) => headName(h.layer, h.head)).join(', '))
  }

  const toggle = (method: AblationMethod) => {
    const next = lab.methods.includes(method) ? lab.methods.filter((m) => m !== method) : [...lab.methods, method]
    if (next.length > 0) lab.setMethods(ABLATION_METHODS.filter((m) => next.includes(m)))
  }

  return (
    <section className="pl-panel" style={{ opacity: lab.running ? 0.55 : 1 }}>
      <div className="at-ablate-form">
        <div data-assist="attribution.components" className="at-ablate-components">
          <div className="pl-control-label" title="A head as layer.head, a layer’s attention as a<layer>, its MLP as m<layer>">
            Components
          </div>
          <input
            className="pl-input"
            type="text"
            value={lab.componentsText}
            onChange={(e) => lab.setComponentsText(e.target.value)}
            placeholder="9.6, 9.9, a10, m10"
            disabled={lab.ablating}
            aria-invalid={error != null}
          />
          {run && (
            <button type="button" className="pl-link" onClick={fillTop} disabled={lab.ablating}>
              Top {QUICK_HEADS} heads
            </button>
          )}
        </div>
        <div>
          <div className="pl-control-label">Ablations</div>
          <div className="pl-segmented" role="group" aria-label="Ablations">
            {ABLATION_METHODS.map((method) => (
              <button
                key={method}
                type="button"
                className="oz-chip"
                aria-pressed={lab.methods.includes(method)}
                title={METHOD_TITLES[method]}
                disabled={lab.ablating}
                onClick={() => toggle(method)}
              >
                {METHOD_LABELS[method]}
              </button>
            ))}
          </div>
        </div>
      </div>
      <RunBar
        note={note}
        warn={error != null || (estimate != null && !estimate.within_limit)}
        label={lab.ablating ? 'Ablating…' : 'Run ablation'}
        running={lab.ablating}
        disabled={!run || error != null || components.length === 0 || (estimate != null && !estimate.within_limit)}
        onRun={lab.ablate}
        assist="attribution.ablate_button"
      />
      {lab.ablated && <Result lab={lab} response={lab.ablated.response} runId={lab.ablated.runId} />}
    </section>
  )
}
