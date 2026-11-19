/**
 * The Heads view: every head's estimated effect, its exact patching effect where it's been
 * measured, and its direct effect on the logit difference; and one head's attention pattern
 * on a pair.
 */

import { useEffect, useMemo, useState } from 'react'
import type { AttributionLab } from '../../hooks/useAttributionLab'
import type { AttributionRun } from '../../types/attribution'
import type { TensorData } from '../../types/model'
import { Dropdown } from '../common/Dropdown'
import { Placeholder, Segmented, StatTiles } from '../lab/chartKit'
import { AttentionHeatmap } from '../visualization/AttentionHeatmap'
import { attentionPattern, effect, effectUnit, gridScale, headName, tokenLabel } from './attributionUtils'
import { EffectHeatmap } from './EffectHeatmap'
import { FigureExport } from './FigureExport'
import { PairText } from './SweepView'

type Quantity = 'estimate' | 'exact' | 'direct'

const QUANTITY_LABELS: Record<Quantity, string> = { estimate: 'Estimate', exact: 'Exact', direct: 'Direct' }
const QUANTITY_TITLES: Record<Quantity, string> = {
  estimate: 'Attribution patching’s estimate of patching the head, mean over pairs',
  exact: 'The metric with the head patched from the clean run into the corrupted run (verified heads only)',
  direct: 'What the head wrote toward the correct answer over the incorrect one on the clean run, in logits',
}
const TABLE_ROWS = 30

function HeadDetail({ lab, run, layer, head }: { lab: AttributionLab; run: AttributionRun; layer: number; head: number }) {
  const kept = useMemo(() => run.attention?.pairs ?? [], [run])
  const [pair, setPair] = useState(kept[0] ?? 0)
  useEffect(() => {
    if (kept.length > 0 && !kept.includes(pair)) setPair(kept[0])
  }, [kept, pair])

  const pattern = useMemo(() => attentionPattern(run, pair, layer, head), [run, pair, layer, head])
  const weights: TensorData | null = useMemo(() => {
    if (!pattern) return null
    // AttentionHeatmap reads [batch][head]: only this head's slot is filled
    const heads: number[][][] = []
    heads[head] = pattern
    return { values: [heads], shape: [1, run.num_heads, pattern.length, pattern.length], dtype: 'float32', mean: 0, std: 0, min: 0, max: 1 }
  }, [pattern, head, run.num_heads])
  const exact = lab.exactHeads[headName(layer, head)]
  const tokens = run.pairs[pair]?.clean_tokens.map(tokenLabel)

  return (
    <section className="pl-panel">
      <div className="pl-card-title">
        <h3>Head {headName(layer, head)}</h3>
      </div>
      <StatTiles
        tiles={[
          { label: 'Estimate', value: effect(run.heads[layer][head]), hint: `± ${effect(run.heads_se[layer][head])}`, title: QUANTITY_TITLES.estimate },
          { label: 'Exact', value: exact ? effect(exact.effect) : '–', hint: exact ? `± ${effect(exact.se)}` : 'not verified', title: QUANTITY_TITLES.exact },
          ...(run.dla ? [{ label: 'Direct', value: effect(run.dla[layer][head]), hint: 'logits', title: QUANTITY_TITLES.direct }] : []),
        ]}
      />
      {weights ? (
        <>
          <div className="pl-controls-row">
            <div className="at-scope">
              <div className="pl-control-label">Attention on</div>
              <Dropdown value={String(pair)} onChange={(value) => setPair(Number(value))} options={kept.map((i) => ({ value: String(i), label: `Pair ${i + 1} · clean` }))} />
            </div>
          </div>
          <div className="at-attention">
            <AttentionHeatmap attentionWeights={weights} layerIndex={layer} headIndex={head} tokens={tokens} showTextLabels />
          </div>
          <PairText run={run} pair={pair} />
        </>
      ) : (
        <div className="pl-empty">Attention patterns weren’t kept for this sweep (too large).</div>
      )}
    </section>
  )
}

export function HeadsView({ lab }: { lab: AttributionLab }) {
  const [quantity, setQuantity] = useState<Quantity>('estimate')
  const run = lab.run
  if (!run) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="attention">Run a sweep to see its heads.</Placeholder>
      </section>
    )
  }

  const verified = Object.keys(lab.exactHeads).length
  const total = run.num_layers * run.num_heads
  const shown: Quantity = quantity === 'direct' && !run.dla ? 'estimate' : quantity
  const value = (layer: number, head: number): number | null => {
    if (shown === 'estimate') return run.heads[layer][head]
    if (shown === 'direct') return run.dla?.[layer][head] ?? null
    return lab.exactHeads[headName(layer, head)]?.effect ?? null
  }
  const grid = run.heads.map((row, layer) => row.map((_, head) => value(layer, head) ?? 0))
  const unit = shown === 'direct' ? 'logits' : effectUnit(run)
  const ranked = run.heads
    .flatMap((row, layer) => row.map((_, head) => ({ layer, head, value: value(layer, head) })))
    .filter((h) => h.value != null)
    .sort((a, b) => Math.abs(b.value as number) - Math.abs(a.value as number))
    .slice(0, TABLE_ROWS)
  const selected = lab.selectedHead ?? (ranked[0] ? { layer: ranked[0].layer, head: ranked[0].head } : null)

  return (
    <div style={{ opacity: lab.running ? 0.55 : 1 }}>
      <section className="pl-panel">
        <div className="pl-controls-row">
          <div>
            <div className="pl-control-label">Show</div>
            <Segmented
              ariaLabel="Quantity shown"
              value={shown}
              onChange={setQuantity}
              options={(['estimate', 'exact', 'direct'] as Quantity[]).map((q) => ({
                value: q,
                label: QUANTITY_LABELS[q],
                title: QUANTITY_TITLES[q],
                disabled: q === 'direct' && !run.dla,
              }))}
            />
          </div>
          <div className="at-verified">
            <div className="pl-control-label">Verified</div>
            <span className="pl-note">
              {verified} of {total} heads
            </span>
            {verified < total && (
              <button
                type="button"
                className="pl-link"
                onClick={() => {
                  lab.setVerifyMode('heads')
                  lab.setView('verify')
                }}
              >
                Verify all
              </button>
            )}
          </div>
        </div>
      </section>

      <div className="pl-results-grid at-sweep-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title={QUANTITY_TITLES[shown]}>{QUANTITY_LABELS[shown]} by head</h3>
            <FigureExport
              filename={`heads-${shown}-${run.model}`}
              figure={() => ({
                kind: 'heatmap',
                title: `Heads, ${QUANTITY_LABELS[shown].toLowerCase()}: ${run.task_name} · ${run.model}`,
                values: grid,
                row_labels: grid.map((_, layer) => String(layer)),
                column_labels: grid[0].map((_, head) => String(head)),
                x_label: 'Head',
                y_label: 'Layer',
                value_label: `${QUANTITY_LABELS[shown]} (${unit})`,
              })}
            />
          </div>
          {shown === 'exact' && verified === 0 ? (
            <div className="pl-empty">No head has been patched yet.</div>
          ) : (
            <EffectHeatmap
              values={grid}
              scale={gridScale(grid)}
              columnLabels={grid[0].map((_, head) => String(head))}
              columnTitle="Head"
              selected={selected ? { row: selected.layer, column: selected.head } : null}
              onSelect={({ row, column }) => lab.setSelectedHead({ layer: row, head: column })}
              ariaLabel={`${QUANTITY_LABELS[shown]} of each head, layers by heads`}
              unit={unit}
              tooltip={(layer, head) => {
                const exact = lab.exactHeads[headName(layer, head)]
                return {
                  title: `Head ${headName(layer, head)}`,
                  rows: [
                    { label: 'estimate', value: effect(run.heads[layer][head]) },
                    { label: 'exact', value: exact ? effect(exact.effect) : 'not verified' },
                    ...(run.dla ? [{ label: 'direct (logits)', value: effect(run.dla[layer][head]) }] : []),
                  ],
                }
              }}
            />
          )}
        </section>

        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>Largest by {QUANTITY_LABELS[shown].toLowerCase()}</h3>
          </div>
          <div className="at-table-scroll" data-assist="attribution.head">
            <table className="pl-table at-table">
              <thead>
                <tr>
                  <th>Head</th>
                  <th title={QUANTITY_TITLES.estimate}>Estimate</th>
                  <th title={QUANTITY_TITLES.exact}>Exact</th>
                  {run.dla && <th title={QUANTITY_TITLES.direct}>Direct</th>}
                </tr>
              </thead>
              <tbody>
                {ranked.map(({ layer, head }) => {
                  const exact = lab.exactHeads[headName(layer, head)]
                  const active = selected?.layer === layer && selected?.head === head
                  return (
                    <tr key={headName(layer, head)} className={active ? 'is-active' : undefined} onClick={() => lab.setSelectedHead({ layer, head })}>
                      <td>{headName(layer, head)}</td>
                      <td>{effect(run.heads[layer][head])}</td>
                      <td>{exact ? effect(exact.effect) : '–'}</td>
                      {run.dla && <td>{effect(run.dla[layer][head])}</td>}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {selected && <HeadDetail lab={lab} run={run} layer={selected.layer} head={selected.head} />}
    </div>
  )
}
