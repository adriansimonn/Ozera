/**
 * The Probe Lab's sweep and score views, laid out as sections (.pl-panel) that each page styles
 * for its interface: the layer sweep, the selected probe, its score views, the token heatmap,
 * and saved probes.
 */

import { useState } from 'react'
import { Microscope } from 'lucide-react'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { Placeholder } from './chartKit'
import { ErrorList } from './ErrorList'
import { LayerSummary } from './LayerSummary'
import { LayerSweepChart, SweepControls } from './LayerSweepChart'
import { PcaScatter } from './PcaScatter'
import { SavedProbes } from './SavedProbes'
import { RocCurve, ScoreHistogram } from './ScoreViews'
import { TokenHeatmap } from './TokenHeatmap'
import { useProbeColors } from './probeTheme'
import { evalScores } from './probeUtils'

export function ProbeResults({ lab }: { lab: ProbeLab }) {
  const colors = useProbeColors()
  const [showOodInPca, setShowOodInPca] = useState(true)
  const { run, pooling, method, position, evalSet } = lab

  if (!run) {
    return (
      <section className="pl-panel">
        <Placeholder icon={Microscope}>
          {lab.training ? 'Training probes…' : 'Choose a model and a dataset, then train probes to see results.'}
        </Placeholder>
      </section>
    )
  }

  const hasOod = run.dataset.n_ood > 0
  const set = evalSet === 'ood' && hasOod ? 'ood' : 'test'
  const { scores, labels, texts } = evalScores(run, pooling, method, position, set)

  return (
    <div className="pl-results" style={{ opacity: lab.training ? 0.55 : 1 }} aria-busy={lab.training}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{run.dataset.name}</strong> · {run.model} · {run.dataset.n_train} train / {run.dataset.n_test} test
          {hasOod ? ` / ${run.dataset.n_ood} OOD` : ''} · ${run.cost.toFixed(2)}
        </div>
        <SweepControls
          pooling={pooling}
          method={method}
          metric={lab.metric}
          evalSet={set}
          hasOod={hasOod}
          onPooling={lab.setPooling}
          onMethod={lab.setMethod}
          onMetric={lab.setMetric}
          onEvalSet={lab.setEvalSet}
        />
        <LayerSweepChart
          run={run}
          pooling={pooling}
          method={method}
          metric={lab.metric}
          evalSet={set}
          position={position}
          onSelectPosition={lab.setPosition}
        />
      </section>

      <section className="pl-panel">
        <LayerSummary
          run={run}
          pooling={pooling}
          method={method}
          position={position}
          testFraction={lab.runTestFraction}
          onSaved={lab.addSaved}
        />
      </section>

      <div className="pl-results-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>Score distribution</h3>
          </div>
          <ScoreHistogram scores={scores} labels={labels} labelNames={run.dataset.label_names} color={colors.methods[method]} />
        </section>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>ROC curve</h3>
          </div>
          <RocCurve scores={scores} labels={labels} labelNames={run.dataset.label_names} color={colors.methods[method]} />
        </section>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>PCA</h3>
            {hasOod && (
              <label className="pl-check-row" style={{ fontSize: '0.72rem' }}>
                <input type="checkbox" checked={showOodInPca} onChange={(e) => setShowOodInPca(e.target.checked)} />
                OOD
              </label>
            )}
          </div>
          <PcaScatter run={run} pooling={pooling} position={position} showOod={showOodInPca} />
        </section>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>Worst mistakes</h3>
          </div>
          <ErrorList scores={scores} labels={labels} texts={texts} labelNames={run.dataset.label_names} />
        </section>
      </div>
    </div>
  )
}

export function ProbeTools({ lab, onShowPurchaseCredits }: { lab: ProbeLab; onShowPurchaseCredits?: () => void }) {
  const savedId = lab.heatmapSource.startsWith('saved:') ? Number(lab.heatmapSource.slice(6)) : null
  return (
    <div className="pl-tools">
      <section className="pl-panel">
        <TokenHeatmap
          run={lab.run}
          pooling={lab.pooling}
          method={lab.method}
          position={lab.position}
          savedProbes={lab.savedProbes}
          source={lab.heatmapSource}
          onSourceChange={lab.setHeatmapSource}
          onShowPurchaseCredits={onShowPurchaseCredits}
        />
      </section>
      <section className="pl-panel">
        <SavedProbes
          probes={lab.savedProbes}
          loading={lab.loadingSaved}
          activeId={savedId}
          onUse={(probe) => lab.setHeatmapSource(`saved:${probe.id}`)}
          onDeleted={lab.removeSaved}
        />
      </section>
    </div>
  )
}
