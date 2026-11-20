/**
 * The Directions view: the principal components of the difference between two models'
 * activations at one layer, the tokens at each one's ends, and what adding each to the base
 * model does to its predictions next to random directions carrying as much.
 */

import type { DiffLab } from '../../hooks/useDiffLab'
import type { ComponentTest, DirectionsRun, TokenContext } from '../../types/diff'
import { Dropdown } from '../common/Dropdown'
import { FigureExport } from '../attribution/FigureExport'
import { Findings, Placeholder, RunBar, StatTiles, type Finding } from '../lab/chartKit'
import { useElementWidth, useLabColors } from '../lab/labTheme'
import { estimateText, pairName, percent, positionName, tokenLabel, value } from './diffUtils'

function range(test: ComponentTest): [number, number] | null {
  const controls = test.controls.map((c) => c.closed).filter((v): v is number => v != null)
  return controls.length ? [Math.min(...controls), Math.max(...controls)] : null
}

/** Whether a component closes more of the KL than every random direction carrying as much. */
function beats(test: ComponentTest): boolean {
  const controls = range(test)
  return test.closed != null && controls != null && test.closed > controls[1]
}

function findings(run: DirectionsRun): Finding[] {
  const out: Finding[] = []
  const test = run.test
  if (test.full_closed == null) {
    out.push({ level: 'warning', text: 'The two models already predict the same on the held-out texts: nothing to close.' })
    return out
  }
  const first = test.components[0]
  const controls = first ? range(first) : null
  if (first && controls) {
    const against = `random directions ${percent(controls[0])} to ${percent(controls[1])}`
    if (beats(first) && (first.closed ?? 0) > 0) {
      out.push({ level: 'good', text: `Component 1 closes ${percent(first.closed)} of the KL to ${run.other_name}: more than every control (${against}).` })
    } else if (beats(first)) {
      out.push({ level: 'warning', text: `Component 1 moves predictions away (${percent(first.closed)}), though less than every control (${against}).` })
    } else {
      out.push({ level: 'warning', text: `Component 1 closes ${percent(first.closed)} of the KL: no more than the controls (${against}).` })
    }
  }
  out.push({
    level: 'info',
    text: `The ${test.joint.count} largest together close ${percent(test.joint.closed)}; the whole difference at this layer, ${percent(test.full_closed)}.`,
  })
  if (run.massive_tokens > 0) out.push({ level: 'info', text: `${run.massive_tokens} tokens with massive activations left out.` })
  if (run.dropped.length > 0) out.push({ level: 'info', text: `${run.dropped.length} text${run.dropped.length === 1 ? '' : 's'} dropped: ${run.dropped[0].reason}.` })
  return out
}

const HEIGHT = 250
const MARGIN = { top: 16, right: 16, bottom: 34, left: 48 }

/** Each component's share of the mean squared difference; click one to see its tokens. */
function ShareBars({ lab, run }: { lab: DiffLab; run: DirectionsRun }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const top = Math.max(...run.components.map((c) => c.share), 0.01)
  const innerW = Math.max(Math.min(width, 560) - MARGIN.left - MARGIN.right, 120)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const band = innerW / run.components.length
  const barW = Math.min(40, band * 0.62)
  const y = (v: number) => MARGIN.top + (1 - v / top) * innerH
  const ticks = [0, top / 2, top]
  return (
    <div ref={ref} className="pl-chart" data-assist="diff.component">
      <svg width={innerW + MARGIN.left + MARGIN.right} height={HEIGHT} role="img" aria-label="Each component's share of the difference between the two models">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
            <text x={MARGIN.left - 7} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
              {percent(t, top < 0.1 ? 1 : 0)}
            </text>
          </g>
        ))}
        {run.components.map((c, i) => {
          const center = MARGIN.left + band * (i + 0.5)
          const active = c.index === lab.component
          return (
            <g key={c.index} style={{ cursor: 'pointer' }} onClick={() => lab.setComponent(c.index)}>
              <rect x={center - band / 2} y={MARGIN.top} width={band} height={innerH} fill={active ? colors.hover : 'transparent'} />
              <rect x={center - barW / 2} y={y(c.share)} width={barW} height={Math.max(y(0) - y(c.share), 1)} fill={colors.series[0]}>
                <title>{`Component ${c.index}: ${percent(c.share)} of the difference`}</title>
              </rect>
              <text x={center} y={HEIGHT - 12} textAnchor="middle" fontSize={11} fontWeight={active ? 600 : 400} fill={colors.text}>
                {c.index}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/**
 * The share of the held-out KL each edit closes: a bar per tested component, a dot per random
 * direction carrying as much. (The whole difference's share, the ceiling, is in the tiles: on
 * this scale it would flatten what the chart compares.)
 */
function ClosedChart({ run }: { run: DirectionsRun }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const test = run.test
  const groups = [
    ...test.components.map((t) => ({ label: String(t.index), value: t.closed ?? 0, controls: t.controls.map((c) => c.closed ?? 0), title: `Component ${t.index}` })),
    { label: `Top ${test.joint.count}`, value: test.joint.closed ?? 0, controls: [] as number[], title: `The ${test.joint.count} largest together` },
  ]
  const all = groups.flatMap((g) => [g.value, ...g.controls])
  const low = Math.min(0, ...all)
  const high = Math.max(0.05, ...all)
  const span = high - low || 1
  const innerW = Math.max(Math.min(width, 560) - MARGIN.left - MARGIN.right, 160)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const y = (v: number) => MARGIN.top + (1 - (v - low) / span) * innerH
  const band = innerW / groups.length
  const barW = Math.min(34, band * 0.36)
  const ticks = [low, 0, high].filter((t, i, list) => list.findIndex((o) => Math.abs(o - t) < span * 0.06) === i)
  return (
    <div ref={ref} className="pl-chart">
      <svg width={innerW + MARGIN.left + MARGIN.right} height={HEIGHT} role="img" aria-label="The share of the KL between the two models each edit closes, next to random directions">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
            <text x={MARGIN.left - 7} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
              {percent(t, 0)}
            </text>
          </g>
        ))}
        {groups.map((group, i) => {
          const center = MARGIN.left + band * (i + 0.5)
          const reference = group.controls.length === 0
          const barX = reference ? center - barW / 2 : center - barW - 3
          return (
            <g key={group.label}>
              <rect x={barX} y={Math.min(y(group.value), y(0))} width={barW} height={Math.max(Math.abs(y(group.value) - y(0)), 1)} fill={reference ? colors.baseline : colors.series[0]}>
                <title>{`${group.title}: ${percent(group.value)}`}</title>
              </rect>
              {group.controls.map((v, k) => (
                <circle key={k} cx={center + 6 + k * 7} cy={y(v)} r={3.5} fill={colors.baseline} stroke={colors.surface} strokeWidth={1}>
                  <title>{`${group.title}, random direction ${k + 1}: ${percent(v)}`}</title>
                </circle>
              ))}
              <text x={center} y={HEIGHT - 12} textAnchor="middle" fontSize={11} fill={colors.text}>
                {group.label}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.series[0] }} />
          Component added
        </span>
        <span className="pl-legend-item" title="Random directions carrying the same amounts, token by token">
          <span className="pl-legend-dot" style={{ background: colors.baseline }} />
          Random directions
        </span>
        <span className="pl-legend-item" title="The tested components added together">
          <span className="pl-legend-rect" style={{ background: colors.baseline }} />
          Together
        </span>
      </div>
    </div>
  )
}

function Contexts({ title, contexts }: { title: string; contexts: TokenContext[] }) {
  return (
    <div>
      <div className="pl-control-label">{title}</div>
      <ul className="df-contexts">
        {contexts.map((c, i) => (
          <li key={i}>
            <span className="df-context">
              <span className="df-context__before">{c.before.join('')}</span>
              <b>{tokenLabel(c.token)}</b>
              <span className="df-context__before">{c.after.join('')}</span>
            </span>
            <span className="df-context__value">{value(c.value, 2)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Result({ lab, run }: { lab: DiffLab; run: DirectionsRun }) {
  const name = pairName(run)
  const shown = run.components.find((c) => c.index === lab.component) ?? run.components[0]
  const tested = run.test.components.find((t) => t.index === shown.index)
  const first = run.test.components[0]
  const controls = first ? range(first) : null
  const listed = run.components.reduce((sum, c) => sum + c.share, 0)
  return (
    <div className="pl-result">
      <div className="pl-run-meta">
        <strong>{positionName(run.layer)}</strong> · {name} · {run.texts_name} · fit on {run.fit_texts} texts, tested on {run.test_texts} ·{' '}
        {run.chat_template ? 'chat template' : 'plain text'} · ${run.cost.toFixed(2)}
      </div>
      <StatTiles
        tiles={[
          { label: 'Component 1', value: percent(run.components[0]?.share), hint: 'of the difference', title: 'Its share of the mean squared difference between the two models’ activations' },
          { label: `Largest ${run.components.length}`, value: percent(listed), hint: 'of the difference' },
          { label: 'Constant offset', value: percent(run.mean_share), hint: 'of the difference', title: 'The share of the difference that is the same for every token' },
          {
            label: 'KL closed',
            value: percent(first?.closed),
            hint: controls ? `controls ${percent(controls[0])} to ${percent(controls[1])}` : 'component 1',
            title: 'The share of the held-out KL between the two models that adding component 1 to the base model removes',
          },
          { label: 'Ceiling', value: percent(run.test.full_closed), hint: 'whole difference', title: 'The share closed by adding the whole difference at this layer: what the later layers do differently is left' },
        ]}
      />
      <Findings items={findings(run)} />
      <div className="df-compare">
        <div>
          <div className="pl-card-title">
            <h3 title="Each principal component’s share of the mean squared difference, on the fitted half of the texts">Share of the difference</h3>
            <FigureExport
              filename={`diff-components-${run.base}-${run.layer}`}
              figure={() => ({
                kind: 'bars',
                title: `Difference components at ${positionName(run.layer).toLowerCase()}: ${name}`,
                y_label: 'Share of the mean squared difference',
                bars: run.components.map((c) => ({ label: String(c.index), value: c.share })),
              })}
            />
          </div>
          <ShareBars lab={lab} run={run} />
        </div>
        <div>
          <div className="pl-card-title">
            <h3 title={`The share of the held-out KL to ${run.other_name} removed by adding each to ${run.base_name}`}>KL closed</h3>
            <FigureExport
              filename={`diff-closed-${run.base}-${run.layer}`}
              figure={() => ({
                kind: 'bars',
                title: `KL closed by each component at ${positionName(run.layer).toLowerCase()}: ${name}`,
                y_label: 'Share of the KL closed',
                baseline: run.test.full_closed ?? undefined,
                baseline_label: 'Whole difference',
                bars: [
                  ...run.test.components.flatMap((t) => {
                    const values = t.controls.map((c) => c.closed ?? 0)
                    return [
                      { label: `${t.index}`, value: t.closed ?? 0 },
                      ...(values.length ? [{ label: `${t.index}: random`, value: values.reduce((a, b) => a + b, 0) / values.length, marks: values }] : []),
                    ]
                  }),
                  { label: `Top ${run.test.joint.count}`, value: run.test.joint.closed ?? 0 },
                ],
              })}
            />
          </div>
          <ClosedChart run={run} />
        </div>
      </div>

      <div className="df-component">
        <div className="pl-card-title">
          <h3 title="The tokens whose difference between the two models lies furthest along the component, at either end">Component {shown.index}</h3>
          <span className="pl-note">
            {percent(shown.share)} of the difference
            {shown.mean_cosine != null && ` · cosine ${value(shown.mean_cosine, 2)} with the mean difference`}
            {tested && ` · closes ${percent(tested.closed)}`}
          </span>
        </div>
        <div className="df-compare">
          <Contexts title="Highest" contexts={shown.high} />
          <Contexts title="Lowest" contexts={shown.low} />
        </div>
      </div>
    </div>
  )
}

export function DirectionsView({ lab }: { lab: DiffLab }) {
  if (!lab.pair && !lab.directions) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="diff">Choose a pair to find the directions its models differ along.</Placeholder>
      </section>
    )
  }
  const estimate = lab.directionsEstimate
  const texts = lab.textSpec()
  const note = !lab.isAuthenticated ? 'Sign in to run' : !lab.pair ? 'Choose a pair' : !texts ? 'Choose texts' : estimate ? estimateText(estimate) : null
  const options = Array.from({ length: lab.layers + 1 }, (_, p) => ({ value: String(p), label: positionName(p) }))

  return (
    <section className="pl-panel">
      <div className="pl-controls-row">
        <div className="df-layer" data-assist="diff.layer">
          <div className="pl-control-label" title="The residual stream position whose difference is decomposed: the embeddings, or a layer’s output">
            Layer
          </div>
          <Dropdown value={String(Math.min(lab.position, lab.layers))} onChange={(v) => lab.setPosition(Number(v))} options={options} disabled={lab.finding} />
        </div>
      </div>
      <RunBar
        note={note}
        warn={estimate != null && !estimate.within_limit}
        label={lab.finding ? 'Finding…' : 'Find directions'}
        running={lab.finding}
        disabled={!lab.pair || !texts || (estimate != null && !estimate.within_limit)}
        onRun={lab.findDirections}
        assist="diff.directions_button"
      />
      {lab.directions && <Result lab={lab} run={lab.directions} />}
    </section>
  )
}
