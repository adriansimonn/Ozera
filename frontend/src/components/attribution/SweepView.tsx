/**
 * The Sweep view: attribution patching's estimates. Every head's effect as a layer × head
 * map, each layer's whole attention and MLP output, and a layer × position map of the
 * residual stream, attention or MLP output for one pair or for the mean of pairs that line up.
 */

import type { AttributionLab } from '../../hooks/useAttributionLab'
import type { AttributionRun, PositionKind } from '../../types/attribution'
import { POSITION_KINDS } from '../../types/attribution'
import { Dropdown } from '../common/Dropdown'
import { Findings, Placeholder, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import {
  KIND_LABELS,
  KIND_TITLES,
  effect,
  effectUnit,
  gridScale,
  headName,
  metricName,
  metricValue,
  positionGrid,
  scopeOptions,
  tokenLabel,
} from './attributionUtils'
import { EffectHeatmap } from './EffectHeatmap'
import { FigureExport } from './FigureExport'

function findings(run: AttributionRun, verifiedHeads: number): Finding[] {
  const out: Finding[] = []
  if (!run.normalized) {
    out.push({ level: 'warning', text: 'Clean and corrupted runs score the same: the model doesn’t do this task. Effects are in the metric’s units.' })
  } else if (run.metric === 'logit_diff' && run.solved < 0.8) {
    out.push({ level: 'warning', text: `The clean prompt beats the corrupted one on ${Math.round(run.solved * 100)}% of pairs.` })
  }
  if (run.dropped.length > 0) {
    out.push({ level: 'info', text: `${run.dropped.length} pair${run.dropped.length === 1 ? '' : 's'} dropped: ${run.dropped[0].reason}.` })
  }
  if (verifiedHeads === 0) out.push({ level: 'info', text: 'Estimates only: nothing here is verified yet.' })
  return out
}

/** Each layer's whole attention and MLP output: two bars per layer, either side of zero. */
function LayerBars({ run }: { run: AttributionRun }) {
  const colors = useLabColors()
  const layers = run.num_layers
  const scale = gridScale(run.layers)
  const rowH = Math.max(10, Math.min(18, Math.floor(420 / layers)))
  const width = 300
  const left = 34
  const half = (width - left - 8) / 2
  const zero = left + half
  const series = [
    { name: 'Attention', color: colors.series[0], values: run.layers[0], errors: run.layers_se[0] },
    { name: 'MLP', color: colors.series[1], values: run.layers[1], errors: run.layers_se[1] },
  ]
  const labelEvery = Math.max(1, Math.ceil(layers / (layers * rowH / 13)))
  return (
    <div>
      <svg width={width} height={layers * rowH + 22} role="img" aria-label="Estimated effect of each layer's attention and MLP output" style={{ display: 'block', maxWidth: '100%' }}>
        <line x1={zero} x2={zero} y1={0} y2={layers * rowH} stroke={colors.axis} />
        {Array.from({ length: layers }, (_, layer) => (
          <g key={layer}>
            {layer % labelEvery === 0 && (
              <text x={left - 7} y={(layer + 0.5) * rowH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
                {layer}
              </text>
            )}
            {series.map((s, i) => {
              const value = s.values[layer]
              const length = (Math.abs(value) / scale) * half
              const barH = rowH / 2 - 1
              return (
                <rect key={s.name} x={value < 0 ? zero - length : zero} y={layer * rowH + 1 + i * barH} width={Math.max(length, 0.5)} height={barH} fill={s.color}>
                  <title>{`${s.name} L${layer}: ${effect(value)} ± ${effect(s.errors[layer])}`}</title>
                </rect>
              )
            })}
          </g>
        ))}
        <text x={left} y={layers * rowH + 15} fontSize={10} fill={colors.dim}>
          {effect(-scale, 2)}
        </text>
        <text x={zero} y={layers * rowH + 15} textAnchor="middle" fontSize={10} fill={colors.dim}>
          0
        </text>
        <text x={width - 8} y={layers * rowH + 15} textAnchor="end" fontSize={10} fill={colors.dim}>
          +{effect(scale, 2)}
        </text>
      </svg>
      <div className="pl-legend">
        {series.map((s) => (
          <span key={s.name} className="pl-legend-item">
            <span className="pl-legend-rect" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}

export function SweepView({ lab }: { lab: AttributionLab }) {
  const run = lab.run
  if (!run) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="patching">{lab.running ? 'Running the sweep…' : 'Choose a model and pairs, then run a sweep.'}</Placeholder>
      </section>
    )
  }

  const scope = lab.scopeInfo
  const grid = scope ? positionGrid(run, scope, lab.sites) : []
  const unit = effectUnit(run)
  const verifiedHeads = Object.keys(lab.exactHeads).length
  const headScale = gridScale(run.heads)
  const selected = lab.selectedHead
  const name = `${run.task_name} · ${run.model}`
  const scopeLabel = scopeOptions(run).find((o) => o.value === lab.scope)?.label ?? ''
  const columnLabels = scope ? scope.labels.map((label, p) => (label == null ? `#${p}` : tokenLabel(label))) : []

  return (
    <div style={{ opacity: lab.running ? 0.55 : 1 }} aria-busy={lab.running}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{run.task_name}</strong> · {run.model} · {run.num_pairs} pairs · {metricName(run.metric)} · ${run.cost.toFixed(2)}
        </div>
        <StatTiles
          tiles={
            run.metric === 'kl'
              ? [
                  {
                    label: 'KL, corrupted from clean',
                    value: metricValue(run.gap),
                    hint: run.normalized ? 'nats · effects are shares of it' : 'too small to divide by',
                    title: 'How far the corrupted run’s next-token prediction is from the clean run’s, mean over pairs',
                  },
                ]
              : [
                  { label: 'Clean', value: metricValue(run.clean_mean), title: 'Mean logit difference (correct − incorrect) on the clean prompts' },
                  { label: 'Corrupted', value: metricValue(run.corrupted_mean), title: 'Mean logit difference on the corrupted prompts' },
                  { label: 'Gap', value: metricValue(run.gap), hint: run.normalized ? 'effects are shares of it' : 'too small to divide by' },
                  { label: 'Pairs solved', value: `${Math.round(run.solved * 100)}%`, title: 'Pairs whose clean prompt scores above their corrupted one' },
                ]
          }
        />
        <Findings items={findings(run, verifiedHeads)} />
      </section>

      <div className="pl-results-grid at-sweep-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title="Estimated effect of patching each head's output at every position, mean over pairs">Heads</h3>
            <FigureExport
              filename={`attribution-heads-${run.model}`}
              figure={() => ({
                kind: 'heatmap',
                title: `Attribution patching, heads: ${name}`,
                values: run.heads,
                row_labels: run.heads.map((_, layer) => String(layer)),
                column_labels: run.heads[0].map((_, head) => String(head)),
                x_label: 'Head',
                y_label: 'Layer',
                value_label: `Estimated effect (${unit})`,
              })}
            />
          </div>
          <EffectHeatmap
            values={run.heads}
            scale={headScale}
            columnLabels={run.heads[0].map((_, head) => String(head))}
            columnTitle="Head"
            selected={selected ? { row: selected.layer, column: selected.head } : null}
            onSelect={({ row, column }) => lab.setSelectedHead({ layer: row, head: column })}
            ariaLabel="Estimated effect of each head, layers by heads"
            unit={unit}
            tooltip={(layer, head) => {
              const exact = lab.exactHeads[headName(layer, head)]
              return {
                title: `Head ${headName(layer, head)}`,
                rows: [
                  { label: `estimate ± ${effect(run.heads_se[layer][head])}`, value: effect(run.heads[layer][head]) },
                  ...(exact ? [{ label: `exact ± ${effect(exact.se)}`, value: effect(exact.effect) }] : []),
                  ...(run.dla ? [{ label: 'direct (logits)', value: effect(run.dla[layer][head]) }] : []),
                ],
              }
            }}
          />
          {selected && (
            <button type="button" className="pl-link at-open-head" onClick={() => lab.setView('heads')}>
              Open head {headName(selected.layer, selected.head)}
            </button>
          )}
        </section>

        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title="Estimated effect of patching a layer's whole attention or MLP output, mean over pairs">Layers</h3>
          </div>
          <LayerBars run={run} />
        </section>
      </div>

      <section className="pl-panel">
        <div className="pl-card-title">
          <h3 title={KIND_TITLES[lab.sites]}>By position</h3>
          <FigureExport
            filename={`attribution-${lab.sites}-${run.model}`}
            figure={() => ({
              kind: 'heatmap',
              title: `Attribution patching, ${KIND_LABELS[lab.sites].toLowerCase()}: ${name} (${scopeLabel})`,
              values: grid,
              row_labels: grid.map((_, layer) => String(layer)),
              column_labels: columnLabels,
              x_label: 'Position',
              y_label: 'Layer',
              value_label: `Estimated effect (${unit})`,
            })}
          />
        </div>
        <div className="pl-controls-row">
          <div data-assist="attribution.sites">
            <div className="pl-control-label">Show</div>
            <Segmented
              ariaLabel="Activation shown"
              value={lab.sites}
              onChange={(value: PositionKind) => lab.setSites(value)}
              options={POSITION_KINDS.map((kind) => ({ value: kind, label: KIND_LABELS[kind], title: KIND_TITLES[kind] }))}
            />
          </div>
          <div data-assist="attribution.scope" className="at-scope">
            <div className="pl-control-label">Positions of</div>
            <Dropdown value={lab.scope} onChange={lab.setScope} options={scopeOptions(run)} />
          </div>
        </div>
        {scope && grid.length > 0 ? (
          <EffectHeatmap
            values={grid}
            scale={gridScale(grid)}
            columnLabels={columnLabels}
            marked={scope.changed}
            columnTitle="Position (marked where the prompts differ)"
            slanted
            ariaLabel={`Estimated effect of the ${KIND_LABELS[lab.sites].toLowerCase()} by layer and position`}
            unit={unit}
            tooltip={(layer, position) => ({
              title: `${KIND_LABELS[lab.sites]} L${layer} · ${columnLabels[position]}`,
              rows: [
                { label: 'estimate', value: effect(grid[layer][position]) },
                { label: 'position', value: String(position) },
              ],
            })}
          />
        ) : (
          <div className="pl-empty">No pairs to show.</div>
        )}
        {scope && (
          <PairText run={run} pair={scope.example} />
        )}
      </section>
    </div>
  )
}

/** A pair's two prompts, token by token, with the tokens that differ marked and their answers. */
export function PairText({ run, pair }: { run: AttributionRun; pair: number }) {
  const p = run.pairs[pair]
  if (!p) return null
  const line = (tokens: string[], other: string[], answer: string | null) => (
    <>
      {tokens.map((token, i) => (
        <span key={i} className={`at-token${token !== other[i] ? ' is-changed' : ''}`}>
          {tokenLabel(token)}
        </span>
      ))}
      {answer != null && <span className="at-token at-token--answer">{tokenLabel(answer)}</span>}
    </>
  )
  return (
    <div className="at-pairtext">
      <div>
        <span className="at-pairtext__name">Clean</span>
        {line(p.clean_tokens, p.corrupted_tokens, p.correct)}
        <span className="at-pairtext__value">{metricValue(p.clean_value)}</span>
      </div>
      <div>
        <span className="at-pairtext__name">Corrupted</span>
        {line(p.corrupted_tokens, p.clean_tokens, p.incorrect)}
        <span className="at-pairtext__value">{metricValue(p.corrupted_value)}</span>
      </div>
    </div>
  )
}
