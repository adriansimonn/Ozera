/**
 * A probe's direction: does the model actually use it, and which SAE features is it? A probe
 * picker shared by steering (add the direction during generation), directional ablation (remove
 * it everywhere, then see what changes), and SAE features, each in its own view.
 */

import { Eraser, Layers, Navigation } from 'lucide-react'
import type { ProbeLab } from '../../hooks/useProbeLab'
import type { InlineProbe, Pooling, ProbeMethod, ReadSpan } from '../../types/probes'
import { Dropdown } from '../common/Dropdown'
import { AblationPanel } from './AblationPanel'
import { Placeholder, Segmented } from './chartKit'
import { SaePanel } from './SaePanel'
import { SteeringPanel } from './SteeringPanel'
import { useProbeColors } from './probeTheme'
import { METHOD_LABELS, POOLING_LABELS, formatScore, inlineProbe } from './probeUtils'

/** A probe, as the steering and ablation panels use it. */
export interface CausalProbe {
  /** Changes whenever the probe does */
  key: string
  model: string
  layer: number
  pooling: Pooling
  method: ProbeMethod
  chatTemplate: boolean
  readSpan: ReadSpan
  labelNames: [string, string]
  /** Scores of ±scale fill the color scale (2 standard deviations of training scores) */
  scale: number | null
  /** Mean norm of the pooled activations at the probe's layer */
  actNorm: number | null
  /** How far apart the class means are along the direction (α = 1 steers by this much) */
  classGap: number | null
  /** The dataset the probe was trained on */
  datasetName: string | null
  /** How requests name the probe */
  ref: { probe_id: number } | { model: string; probe: InlineProbe }
}

function labelNamesOf(dataset: Record<string, unknown>): [string, string] {
  const names = dataset.label_names
  return Array.isArray(names) && names.length === 2 ? [String(names[0]), String(names[1])] : ['Negative', 'Positive']
}

function chosenProbe(lab: ProbeLab): CausalProbe | null {
  const { run, pooling, position, causalSource, causalMethod } = lab
  if (causalSource === 'sweep') {
    if (!run || position < 1) return null
    const probe = inlineProbe(run, pooling, causalMethod, position)
    const std = run.poolings[pooling].methods[causalMethod].metrics.score_std?.[position] ?? null
    return {
      key: `sweep:${run.model}:${run.seed}:${run.total_tokens}:${pooling}:${causalMethod}:${position}`,
      model: run.model,
      layer: position - 1,
      pooling,
      method: causalMethod,
      chatTemplate: run.chat_template,
      readSpan: run.read_span,
      labelNames: run.dataset.label_names,
      scale: std != null ? 2 * std : null,
      actNorm: run.poolings[pooling].act_norms[position] ?? null,
      classGap: probe.class_gap ?? null,
      datasetName: run.dataset.name,
      ref: { model: run.model, probe },
    }
  }
  const id = Number(causalSource.slice(6))
  const saved = lab.savedProbes.find((p) => p.id === id)
  if (!saved || !saved.model_available) return null
  const std = saved.normalization.score_std
  const name = saved.dataset.name
  return {
    key: `saved:${saved.id}`,
    model: saved.model_id,
    layer: saved.layer,
    pooling: saved.pooling,
    method: saved.method,
    chatTemplate: saved.chat_template,
    readSpan: saved.read_span,
    labelNames: labelNamesOf(saved.dataset),
    scale: std != null ? 2 * std : null,
    actNorm: saved.normalization.act_norm ?? null,
    classGap: saved.normalization.class_gap ?? null,
    datasetName: typeof name === 'string' ? name : null,
    ref: { probe_id: saved.id },
  }
}

export type CausalTool = 'steer' | 'ablate' | 'sae'

const TOOL_ICONS = { steer: Navigation, ablate: Eraser, sae: Layers } as const

/** One of the tools that test a probe's direction, under a picker for the probe (shared by all three). */
export function ProbeCausal({
  lab,
  tool,
  onShowPurchaseCredits,
}: {
  lab: ProbeLab
  tool: CausalTool
  onShowPurchaseCredits?: () => void
}) {
  const colors = useProbeColors()
  const { run, position } = lab
  const sweepAvailable = run != null && position > 0
  const options = [
    ...(sweepAvailable
      ? [{ value: 'sweep', label: `Sweep selection · ${run!.model} L${position - 1} ${POOLING_LABELS[lab.pooling].toLowerCase()}` }]
      : []),
    ...lab.savedProbes.map((p) => ({
      value: `saved:${p.id}`,
      label: `${p.name} · ${p.model_id} L${p.layer}`,
      disabled: !p.model_available,
    })),
  ]
  const probe = chosenProbe(lab)
  const saved = lab.causalSource.startsWith('saved:')
    ? lab.savedProbes.find((p) => p.id === Number(lab.causalSource.slice(6))) ?? null
    : null

  if (options.length === 0) {
    return (
      <section className="pl-panel">
        <Placeholder icon={TOOL_ICONS[tool]}>Train or save a probe to test its direction.</Placeholder>
      </section>
    )
  }

  return (
    <div className="pl-causal">
      <section className="pl-panel">
        <div className="pl-causal-pick">
          <div style={{ flex: '1 1 320px', minWidth: 0 }}>
            <div className="pl-control-label">Probe</div>
            <Dropdown value={probe ? lab.causalSource : ''} onChange={lab.setCausalSource} options={options} placeholder="Choose a probe" />
          </div>
          {lab.causalSource === 'sweep' && sweepAvailable && (
            <div>
              <div className="pl-control-label">Direction</div>
              <Segmented
                ariaLabel="Direction"
                value={lab.causalMethod}
                onChange={lab.setCausalMethod}
                options={[
                  { value: 'diff_means', label: 'Diff. in means', title: METHOD_LABELS.diff_means },
                  { value: 'logreg', label: 'Logistic regression', title: METHOD_LABELS.logreg },
                ]}
              />
            </div>
          )}
        </div>
        {probe && probe.classGap != null && probe.classGap > 0 && (
          <div className="pl-note" style={{ marginTop: '0.6rem' }}>
            Class gap {formatScore(probe.classGap)}
            {probe.actNorm ? ` · ${((probe.classGap / probe.actNorm) * 100).toFixed(1)}% of activation norm` : ''}
          </div>
        )}
        {saved?.model_changed && (
          <div className="pl-note" style={{ color: colors.warning, marginTop: '0.4rem' }}>
            Model changed since this probe was saved
          </div>
        )}
      </section>

      {probe && (
        <section className="pl-panel">
          {tool === 'steer' ? (
            <SteeringPanel probe={probe} isAuthenticated={lab.isAuthenticated} onShowPurchaseCredits={onShowPurchaseCredits} />
          ) : tool === 'ablate' ? (
            <AblationPanel probe={probe} lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />
          ) : (
            <SaePanel probe={probe} lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />
          )}
        </section>
      )}

      <style>{`
        .probe-lab .pl-causal-pick { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: flex-end; }
      `}</style>
    </div>
  )
}
