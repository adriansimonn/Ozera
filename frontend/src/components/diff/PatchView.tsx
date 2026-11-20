/**
 * The Patch view: one model's activations patched into the other on one prompt, layer by layer
 * and position by position, read as the share of the KL between their next-token predictions
 * that each patch closes.
 */

import type { DiffLab } from '../../hooks/useDiffLab'
import type { PatchRun } from '../../types/diff'
import { PATCH_KINDS } from '../../types/diff'
import { EffectHeatmap } from '../attribution/EffectHeatmap'
import { FigureExport } from '../attribution/FigureExport'
import { Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { KIND_LABELS, KIND_TITLES, donorNames, estimateText, nats, percent, tokenLabel } from './diffUtils'

// Below this KL (nats) the two models already predict the same on the prompt
const MIN_KL = 0.01

/** The share of the unpatched KL a patch closes (below −1 it's drawn as −1: the patch moved the prediction far away). */
function closed(kl: number, base: number): number {
  return Math.max(-1, 1 - kl / base)
}

function best(run: PatchRun): { layer: number; position: number; kl: number } {
  let found = { layer: 0, position: 0, kl: Infinity }
  run.grid.forEach((row, layer) =>
    row.forEach((kl, position) => {
      if (kl < found.kl) found = { layer, position, kl }
    }),
  )
  return found
}

function findings(run: PatchRun): Finding[] {
  const { donor, recipient } = donorNames(run, run.donor)
  if (run.kl_base < MIN_KL) return [{ level: 'warning', text: `${donor} and ${recipient} already predict the same after this prompt: nothing to close.` }]
  const out: Finding[] = []
  const cell = best(run)
  out.push({
    level: 'info',
    text: `The best single patch closes ${percent(closed(cell.kl, run.kl_base), 0)}: layer ${cell.layer} at “${tokenLabel(run.tokens[cell.position])}”.`,
  })
  const whole = run.all_positions.map((kl) => closed(kl, run.kl_base))
  const reached = whole.findIndex((share) => share >= 0.5)
  const what = run.kind === 'resid' ? `${donor}’s stream entering` : `${donor}’s ${KIND_LABELS[run.kind].toLowerCase()} output at`
  if (reached >= 0) out.push({ level: 'info', text: `Patching every position of ${what} layer ${reached} closes ${percent(whole[reached], 0)}.` })
  else out.push({ level: 'info', text: `No layer patched at every position closes half of it (at most ${percent(Math.max(...whole), 0)}).` })
  return out
}

/** The share closed with a whole layer patched: a bar per layer, either side of zero. */
function LayerBars({ run }: { run: PatchRun }) {
  const colors = useLabColors()
  const layers = run.num_layers
  const rowH = Math.max(10, Math.min(18, Math.floor(420 / layers)))
  const width = 260
  const left = 34
  const half = (width - left - 8) / 2
  const zero = left + half
  const labelEvery = Math.max(1, Math.ceil(13 / rowH))
  return (
    <svg width={width} height={layers * rowH + 22} role="img" aria-label="Share of the KL closed with each layer patched at every position" style={{ display: 'block', maxWidth: '100%' }}>
      <line x1={zero} x2={zero} y1={0} y2={layers * rowH} stroke={colors.axis} />
      {run.all_positions.map((kl, layer) => {
        const share = closed(kl, run.kl_base)
        const length = Math.abs(share) * half
        return (
          <g key={layer}>
            {layer % labelEvery === 0 && (
              <text x={left - 7} y={(layer + 0.5) * rowH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.dim}>
                {layer}
              </text>
            )}
            <rect x={share < 0 ? zero - length : zero} y={layer * rowH + 1} width={Math.max(length, 0.5)} height={rowH - 2} fill={colors.series[0]}>
              <title>{`Layer ${layer}, every position: ${percent(share)} closed (KL ${nats(kl)})`}</title>
            </rect>
          </g>
        )
      })}
      <text x={left} y={layers * rowH + 15} fontSize={10} fill={colors.dim}>
        −100%
      </text>
      <text x={zero} y={layers * rowH + 15} textAnchor="middle" fontSize={10} fill={colors.dim}>
        0
      </text>
      <text x={width - 8} y={layers * rowH + 15} textAnchor="end" fontSize={10} fill={colors.dim}>
        100%
      </text>
    </svg>
  )
}

function Result({ run }: { run: PatchRun }) {
  const { donor, recipient } = donorNames(run, run.donor)
  const agree = run.kl_base < MIN_KL
  const shares = run.grid.map((row) => row.map((kl) => (agree ? 0 : closed(kl, run.kl_base))))
  const labels = run.tokens.map(tokenLabel)
  // The scale ends at the largest share shown, so a prompt whose patches close little is still readable
  const scale = Math.max(0.05, Math.min(1, Math.max(...shares.flat().map(Math.abs))))
  const cell = best(run)
  const name = `${donor} into ${recipient}`
  return (
    <div className="pl-result">
      <div className="pl-run-meta">
        <strong>{name}</strong> · {KIND_LABELS[run.kind].toLowerCase()} · {run.tokens.length} tokens · {run.chat_template ? 'chat template' : 'plain text'} · $
        {run.cost.toFixed(2)}
      </div>
      <StatTiles
        tiles={[
          { label: 'KL unpatched', value: nats(run.kl_base), hint: 'nats', title: `KL(${donor} ‖ ${recipient}) of the next-token predictions after the prompt` },
          { label: 'Donor predicts', value: tokenLabel(run.predictions.donor[0]?.token), hint: percent(run.predictions.donor[0]?.prob, 0), title: donor },
          { label: 'Recipient predicts', value: tokenLabel(run.predictions.recipient[0]?.token), hint: percent(run.predictions.recipient[0]?.prob, 0), title: recipient },
          {
            label: 'Best patch',
            value: agree ? '–' : percent(closed(cell.kl, run.kl_base), 0),
            hint: `layer ${cell.layer} · ${tokenLabel(run.tokens[cell.position])}`,
            title: 'The largest share of the unpatched KL one patched activation closes',
          },
        ]}
      />
      <Findings items={findings(run)} />
      {!agree && (
        <div className="df-patch-grid">
          <div>
            <div className="pl-card-title">
              <h3 title={`${KIND_TITLES[run.kind]}, patched from ${donor} into ${recipient} at one position`}>By layer and position</h3>
              <FigureExport
                filename={`diff-patch-${run.kind}-${run.base}`}
                figure={() => ({
                  kind: 'heatmap',
                  title: `Patching ${name}, ${KIND_LABELS[run.kind].toLowerCase()}`,
                  values: shares,
                  row_labels: shares.map((_, layer) => String(layer)),
                  column_labels: labels,
                  x_label: 'Position',
                  y_label: 'Layer',
                  value_label: 'Share of the KL closed',
                })}
              />
            </div>
            <EffectHeatmap
              values={shares}
              scale={scale}
              columnLabels={labels}
              columnTitle="Position"
              slanted
              ariaLabel={`Share of the KL closed by patching ${donor}'s ${KIND_LABELS[run.kind].toLowerCase()} into ${recipient}, by layer and position`}
              unit="share of the unpatched KL closed"
              tooltip={(layer, position) => ({
                title: `${KIND_LABELS[run.kind]} L${layer} · ${labels[position]}`,
                rows: [
                  { label: 'closed', value: percent(1 - run.grid[layer][position] / run.kl_base) },
                  { label: 'KL, nats', value: nats(run.grid[layer][position]) },
                ],
              })}
            />
          </div>
          <div>
            <div className="pl-card-title">
              <h3 title={run.kind === 'resid' ? `${donor} runs up to the layer, ${recipient} from there on` : `The layer’s whole output taken from ${donor}`}>Every position</h3>
              <FigureExport
                filename={`diff-patch-layers-${run.kind}-${run.base}`}
                figure={() => ({
                  kind: 'lines',
                  title: `Patching ${name}, ${KIND_LABELS[run.kind].toLowerCase()} at every position`,
                  x_label: 'Layer',
                  y_label: 'Share of the KL closed',
                  column_labels: run.all_positions.map((_, layer) => String(layer)),
                  series: [{ label: 'Every position patched', values: run.all_positions.map((kl) => 1 - kl / run.kl_base) }],
                })}
              />
            </div>
            <LayerBars run={run} />
          </div>
        </div>
      )}
    </div>
  )
}

export function PatchView({ lab }: { lab: DiffLab }) {
  const pair = lab.pair
  if (!pair && !lab.patched) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="diff">Choose a pair to patch one model into the other.</Placeholder>
      </section>
    )
  }
  const estimate = lab.patchEstimate
  const note = !lab.isAuthenticated ? 'Sign in to run' : !pair ? 'Choose a pair' : !lab.prompt.trim() ? 'Write a prompt' : estimate ? estimateText(estimate) : null

  return (
    <section className="pl-panel">
      <div className="df-patch-form">
        <div className="df-prompt" data-assist="diff.prompt">
          <div className="pl-control-label" title="Both models read it; their predictions of what comes next are compared">
            Prompt
          </div>
          <textarea
            className="pl-textarea"
            value={lab.prompt}
            onChange={(e) => lab.setPrompt(e.target.value)}
            placeholder="A short prompt, ending where the next word would come"
            maxLength={600}
            rows={3}
            disabled={lab.patching}
          />
        </div>
        <div data-assist="diff.donor">
          <div className="pl-control-label" title="The model whose activations are patched into the other one">
            Donor
          </div>
          <Segmented
            ariaLabel="Donor"
            value={lab.donor}
            onChange={lab.setDonor}
            disabled={lab.patching}
            options={[
              { value: 'other', label: pair?.other_name ?? 'Other', title: 'Into the base model' },
              { value: 'base', label: pair?.base_name ?? 'Base', title: 'Into the other model' },
            ]}
          />
        </div>
        <div data-assist="diff.sites">
          <div className="pl-control-label">Patch</div>
          <Segmented
            ariaLabel="Activation patched"
            value={lab.kind}
            onChange={lab.setKind}
            disabled={lab.patching}
            options={PATCH_KINDS.map((kind) => ({ value: kind, label: KIND_LABELS[kind], title: KIND_TITLES[kind] }))}
          />
        </div>
      </div>
      <RunBar
        note={note}
        warn={estimate != null && !estimate.within_limit}
        label={lab.patching ? 'Patching…' : 'Run patching'}
        running={lab.patching}
        disabled={!pair || !lab.prompt.trim() || (estimate != null && !estimate.within_limit)}
        onRun={lab.runPatch}
        assist="diff.patch_button"
      />
      {lab.patched && <Result run={lab.patched} />}
    </section>
  )
}
