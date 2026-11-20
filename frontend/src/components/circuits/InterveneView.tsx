/**
 * The Intervene view: the test of a graph. The pinned feature nodes' activations are multiplied
 * by a factor in the real model (0 ablates them) and the target is read again, next to control
 * sets of as many other active features: matched in activation anywhere in the prompt, and at
 * the pinned nodes' own layers and positions.
 */

import { X } from 'lucide-react'
import type { CircuitLab } from '../../hooks/useCircuitLab'
import type { CircuitGraph, ControlKind, InterveneResponse } from '../../types/circuits'
import { CONTROL_KINDS } from '../../types/circuits'
import { FigureExport } from '../attribution/FigureExport'
import { Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { useElementWidth, useLabColors } from '../lab/labTheme'
import { SCALES, interveneEstimateText, logits, money, nodeName, percent, showToken, signedLogits } from './circuitUtils'

const QUICK = [5, 10, 20]

const CONTROL_LABELS: Record<ControlKind, string> = { activation: 'Matched activation', site: 'Same site' }
const CONTROL_TITLES: Record<ControlKind, string> = {
  activation: 'As many other active features, each nearest in activation to a pinned node, anywhere in the prompt',
  site: 'As many other active features at the pinned nodes’ own layers and positions',
}

function scaleWords(scale: number): string {
  return scale === 0 ? 'Ablating' : `Scaling (× ${scale}) `
}

function changes(response: InterveneResponse, kind: ControlKind): number[] {
  return (response.controls[kind] ?? []).map((c) => c.change ?? 0)
}

function findings(response: InterveneResponse): Finding[] {
  const out: Finding[] = []
  const change = response.chosen.change ?? 0
  const count = response.nodes.length
  const what = `${scaleWords(response.scale).trim()} ${count} node${count === 1 ? '' : 's'}`
  const moved = `${what} moves ${showToken(response.target.token)} by ${signedLogits(change)} logits (${percent(response.base.prob)} → ${percent(response.chosen.prob)})`
  const larger = (kind: ControlKind) => {
    const values = changes(response, kind)
    return values.length > 0 && values.every((v) => Math.abs(change) > Math.abs(v))
  }
  const tested = CONTROL_KINDS.filter((kind) => changes(response, kind).length > 0)
  if (tested.length === 0) {
    out.push({ level: 'info', text: `${moved}. No control sets to compare with.` })
    return out
  }
  if (tested.every(larger)) out.push({ level: 'good', text: `${moved}: more than every control set.` })
  else if (tested.some(larger)) {
    const weak = tested.filter((kind) => !larger(kind)).map((kind) => CONTROL_LABELS[kind].toLowerCase())
    out.push({ level: 'warning', text: `${moved}: more than some controls, but not every ${weak.join(' or ')} set.` })
  } else out.push({ level: 'warning', text: `${moved}: no more than the control sets.` })
  for (const kind of tested) {
    const values = changes(response, kind)
    out.push({ level: 'info', text: `${CONTROL_LABELS[kind]}: ${signedLogits(Math.min(...values))} to ${signedLogits(Math.max(...values))} over ${values.length} sets.` })
  }
  return out
}

const HEIGHT = 250
const MARGIN = { top: 16, right: 16, bottom: 34, left: 48 }

/** The target's change in logits: the pinned nodes as a bar, each control set as a dot. */
function ChangeChart({ response }: { response: InterveneResponse }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const chosen = response.chosen.change ?? 0
  const groups = CONTROL_KINDS.map((kind) => ({ kind, values: changes(response, kind) })).filter((g) => g.values.length > 0)
  const all = [0, chosen, ...groups.flatMap((g) => g.values)]
  const low = Math.min(...all)
  const high = Math.max(...all)
  const span = high - low || 1
  const innerW = Math.max(Math.min(width, 560) - MARGIN.left - MARGIN.right, 160)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const y = (v: number) => MARGIN.top + (1 - (v - low) / span) * innerH
  const band = innerW / (groups.length + 1)
  const barW = Math.min(52, band * 0.4)
  const ticks = [low, 0, high].filter((t, i, list) => list.findIndex((o) => Math.abs(o - t) < span * 0.06) === i)
  return (
    <div ref={ref} className="pl-chart">
      <svg width={innerW + MARGIN.left + MARGIN.right} height={HEIGHT} role="img" aria-label="Change in the target's logit: the pinned nodes next to control sets">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
            <text x={MARGIN.left - 7} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
              {signedLogits(t)}
            </text>
          </g>
        ))}
        <rect x={MARGIN.left + band / 2 - barW / 2} y={Math.min(y(chosen), y(0))} width={barW} height={Math.max(Math.abs(y(chosen) - y(0)), 1)} fill={colors.ablated}>
          <title>{`Pinned nodes: ${signedLogits(chosen, 3)} logits`}</title>
        </rect>
        <text x={MARGIN.left + band / 2} y={HEIGHT - 12} textAnchor="middle" fontSize={11} fill={colors.text}>
          Pinned
        </text>
        {groups.map((group, i) => {
          const center = MARGIN.left + band * (i + 1.5)
          return (
            <g key={group.kind}>
              {group.values.map((value, k) => (
                <circle key={k} cx={center + (k - (group.values.length - 1) / 2) * 9} cy={y(value)} r={4} fill={colors.baseline} stroke={colors.surface} strokeWidth={1}>
                  <title>{`${CONTROL_LABELS[group.kind]}, set ${k + 1}: ${signedLogits(value, 3)} logits`}</title>
                </circle>
              ))}
              <text x={center} y={HEIGHT - 12} textAnchor="middle" fontSize={11} fill={colors.text}>
                {CONTROL_LABELS[group.kind]}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.ablated }} />
          Pinned nodes
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.baseline }} />
          Control sets of as many
        </span>
      </div>
    </div>
  )
}

function Result({ response, graph }: { response: InterveneResponse; graph: CircuitGraph | null }) {
  const stale = graph != null && graph.job_id !== response.job_id
  const names = response.nodes.map((n) => `L${n.layer} #${n.feature}`)
  const shownNames = names.slice(0, 8).join(', ') + (names.length > 8 ? ` +${names.length - 8}` : '')
  return (
    <div className="pl-result">
      <div className="pl-run-meta">
        {stale && <strong style={{ color: 'var(--oz-warn)' }}>From another graph · </strong>}
        <strong>{shownNames}</strong> · {response.scale === 0 ? 'ablated' : `× ${response.scale}`} · {response.model} · {money(response.cost)}
      </div>
      <StatTiles
        tiles={[
          { label: 'Target', value: showToken(response.target.token), hint: `was ${percent(response.base.prob)}` },
          { label: 'Logit change', value: signedLogits(response.chosen.change), hint: `${logits(response.base.logit)} → ${logits(response.chosen.logit)}`, title: 'The target’s logit above the mean logit, before and after' },
          { label: 'Probability', value: percent(response.chosen.prob), hint: `rank ${response.chosen.rank + 1}`, title: 'The target’s probability and rank after the intervention' },
        ]}
      />
      <Findings items={findings(response)} />
      <div className="ct-compare">
        <div>
          <div className="pl-card-title">
            <h3 title="The target’s logit above the mean logit, minus the unchanged model’s">Change in the target’s logit</h3>
            <FigureExport
              filename={`intervention-${response.model}`}
              figure={() => ({
                kind: 'bars',
                title: `${scaleWords(response.scale).trim()} ${response.nodes.length} nodes: ${showToken(response.target.token)} (${response.model})`,
                y_label: 'Change in the target’s logit',
                baseline: 0,
                baseline_label: 'Unchanged',
                bars: [
                  { label: 'Pinned nodes', value: response.chosen.change ?? 0 },
                  ...CONTROL_KINDS.flatMap((kind) => {
                    const values = changes(response, kind)
                    return values.length ? [{ label: CONTROL_LABELS[kind], value: values.reduce((a, b) => a + b, 0) / values.length, marks: values }] : []
                  }),
                ],
              })}
            />
          </div>
          <ChangeChart response={response} />
        </div>
        <div>
          <div className="pl-card-title">
            <h3>Top predictions</h3>
          </div>
          <table className="pl-table ct-table ct-predictions">
            <thead>
              <tr>
                <th>Before</th>
                <th></th>
                <th>After</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {response.predictions.base.map((before, i) => {
                const after = response.predictions.chosen[i]
                return (
                  <tr key={i}>
                    <td className="ct-mono">{showToken(before.token)}</td>
                    <td>{percent(before.prob)}</td>
                    <td className="ct-mono">{after ? showToken(after.token) : ''}</td>
                    <td>{after ? percent(after.prob) : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <table className="pl-table ct-table" style={{ marginTop: '1rem' }}>
            <thead>
              <tr>
                <th>Nodes</th>
                <th>Logit change</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Pinned</td>
                <td>{signedLogits(response.chosen.change)}</td>
                <td>{percent(response.chosen.prob)}</td>
              </tr>
              {CONTROL_KINDS.map((kind) => {
                const sets = response.controls[kind] ?? []
                if (sets.length === 0) return null
                const values = sets.map((c) => c.change ?? 0)
                const probs = sets.map((c) => c.prob)
                return (
                  <tr key={kind} title={CONTROL_TITLES[kind]}>
                    <td>{CONTROL_LABELS[kind]}</td>
                    <td>
                      {signedLogits(Math.min(...values))} to {signedLogits(Math.max(...values))}
                    </td>
                    <td>
                      {percent(Math.min(...probs))} to {percent(Math.max(...probs))}
                    </td>
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

export function InterveneView({ lab }: { lab: CircuitLab }) {
  const graph = lab.graph
  if (!graph && !lab.intervention) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="circuits">Trace a prompt, then test its graph by changing the nodes it points to.</Placeholder>
      </section>
    )
  }
  const estimate = lab.interveneEstimate
  const count = lab.pinned.length
  const note = !lab.isAuthenticated
    ? 'Sign in to run'
    : !graph
      ? 'Trace a prompt first'
      : count === 0
        ? 'Pin feature nodes to change'
        : `${count} node${count === 1 ? '' : 's'}${estimate ? ` · ${interveneEstimateText(estimate)}` : ''}`
  const byId = new Map((graph?.nodes ?? []).map((n) => [n.id, n]))

  return (
    <section className="pl-panel">
      <div className="ct-intervene-form">
        <div className="ct-pinned" data-assist="circuits.nodes">
          <div className="pl-control-label" title="The feature nodes whose activations are changed (pin them in the graph or its table)">
            Pinned nodes
          </div>
          {count > 0 ? (
            <div className="ct-chips">
              {lab.pinned.map((id) => {
                const node = byId.get(id)
                return (
                  <span key={id} className="ct-pin">
                    <button type="button" className="ct-pin__name" onClick={() => { lab.setSelected(id); lab.setView('graph') }} title={node && graph ? `${showToken(graph.tokens[node.position])} · ${percent(node.influence)}` : undefined}>
                      {node ? nodeName(node) : id}
                    </button>
                    <button type="button" className="ct-pin__remove" aria-label={`Unpin ${node ? nodeName(node) : id}`} onClick={() => lab.togglePin(id)} disabled={lab.intervening}>
                      <X />
                    </button>
                  </span>
                )
              })}
            </div>
          ) : (
            <div className="pl-hint">None yet</div>
          )}
          {graph && (
            <div className="ct-quick">
              {QUICK.map((n) => (
                <button key={n} type="button" className="pl-link" onClick={() => lab.pinTop(n)} disabled={lab.intervening} title={`The ${n} feature nodes that support the target most`}>
                  Top {n}
                </button>
              ))}
              {count > 0 && (
                <button type="button" className="pl-link" onClick={() => lab.setPinned([])} disabled={lab.intervening}>
                  Clear
                </button>
              )}
            </div>
          )}
        </div>
        <div data-assist="circuits.scale">
          <div className="pl-control-label" title="What the pinned nodes’ activations are multiplied by">
            Change
          </div>
          <Segmented
            ariaLabel="Change"
            value={String(lab.scale)}
            onChange={(value) => lab.setScale(Number(value))}
            disabled={lab.intervening}
            options={SCALES.map((s) => ({ value: String(s.value), label: s.label, title: s.title }))}
          />
        </div>
      </div>
      <RunBar
        note={note}
        label={lab.intervening ? 'Intervening…' : 'Run intervention'}
        running={lab.intervening}
        disabled={!graph || count === 0}
        onRun={lab.intervene}
        assist="circuits.intervene_button"
      />
      {lab.intervention && <Result response={lab.intervention} graph={graph} />}
    </section>
  )
}
