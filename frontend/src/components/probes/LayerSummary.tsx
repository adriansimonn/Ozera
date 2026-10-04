/**
 * The selected probe (pooling, method, layer): its metrics next to its baselines, checks that
 * flag when the probe may be fooling you, and saving it.
 */

import { useEffect, useState } from 'react'
import { Save } from 'lucide-react'
import type { Pooling, ProbeMethod, ProbeRunResult, SavedProbe } from '../../types/probes'
import { apiClient } from '../../api/client'
import { Findings, StatTiles, type Finding, type Tile } from './chartKit'
import { useProbeColors } from './probeTheme'
import {
  METHOD_LABELS,
  METHOD_SHORT,
  POOLING_LABELS,
  formatMetric,
  formatPercent,
  saveRequest,
} from './probeUtils'

interface LayerSummaryProps {
  run: ProbeRunResult
  pooling: Pooling
  method: ProbeMethod
  position: number
  testFraction: number
  onSaved: (probe: SavedProbe) => void
}

function trustChecks(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod, position: number): Finding[] {
  const m = run.poolings[pooling].methods[method].metrics
  const at = (values: (number | null)[] | undefined) => values?.[position] ?? null
  const checks: Finding[] = []

  const testAuroc = at(m.test_auroc)
  const testAcc = at(m.test_acc)
  const trainAcc = at(m.train_acc)
  const embedding = m.test_auroc?.[0] ?? null
  const controlTrain = at(m.control_train_acc)
  const selectivity = at(m.selectivity)
  const oodAuroc = at(m.ood_auroc)

  if (position > 0 && embedding != null && embedding >= 0.85) {
    checks.push({
      level: 'warning',
      text: `Embeddings already reach AUROC ${formatMetric(embedding, 2)}: the tokens alone may give the label away.`,
    })
  } else if (position > 0 && embedding != null && testAuroc != null && testAuroc - embedding < 0.05) {
    checks.push({
      level: 'info',
      text: `Within ${formatMetric(Math.max(testAuroc - embedding, 0), 2)} AUROC of the embeddings.`,
    })
  }
  if (oodAuroc != null && testAuroc != null) {
    if (oodAuroc < 0.5) {
      checks.push({ level: 'warning', text: `Below chance on OOD (AUROC ${formatMetric(oodAuroc, 2)}).` })
    } else if (testAuroc - oodAuroc >= 0.1) {
      checks.push({
        level: 'warning',
        text: `AUROC drops from ${formatMetric(testAuroc, 2)} to ${formatMetric(oodAuroc, 2)} on OOD.`,
      })
    }
  }
  if (selectivity != null && selectivity < 0.1) {
    checks.push({ level: 'warning', text: `Low selectivity (${formatPercent(selectivity)}): barely beats shuffled labels.` })
  }
  if (trainAcc != null && testAcc != null && trainAcc - testAcc >= 0.15) {
    checks.push({ level: 'warning', text: `Overfitting: ${formatPercent(trainAcc)} train vs ${formatPercent(testAcc)} test accuracy.` })
  }
  if (controlTrain != null && controlTrain >= 0.9) {
    checks.push({ level: 'info', text: `Fits ${formatPercent(controlTrain)} of shuffled training labels: train accuracy says little.` })
  }
  if (run.dataset.n_test < 50 && testAcc != null) {
    const margin = 1.96 * Math.sqrt((testAcc * (1 - testAcc) + 0.25 / run.dataset.n_test) / run.dataset.n_test)
    checks.push({ level: 'info', text: `Only ${run.dataset.n_test} test examples (accuracy ±${formatPercent(margin)}).` })
  }
  return checks
}

export function LayerSummary({ run, pooling, method, position, testFraction, onSaved }: LayerSummaryProps) {
  const colors = useProbeColors()
  const result = run.poolings[pooling].methods[method]
  const m = result.metrics
  const at = (values: (number | null)[] | undefined) => values?.[position] ?? null
  const layerName = position === 0 ? 'Embeddings' : `Layer ${position - 1}`
  const defaultName = `${run.dataset.name} · L${position - 1} ${POOLING_LABELS[pooling].toLowerCase()} ${METHOD_SHORT[method]}`

  const [name, setName] = useState(defaultName)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [savedAs, setSavedAs] = useState<string | null>(null)
  useEffect(() => {
    setName(defaultName)
    setSaveError(null)
    setSavedAs(null)
  }, [defaultName])

  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      const saved = await apiClient.saveProbe(saveRequest(run, pooling, method, position, name.trim() || defaultName, testFraction))
      setSavedAs(saved.name)
      onSaved(saved)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save the probe')
    } finally {
      setSaving(false)
    }
  }

  const tiles: Tile[] = [
    { label: 'Test AUROC', value: formatMetric(at(m.test_auroc)) },
    { label: 'Test accuracy', value: formatPercent(at(m.test_acc)), hint: `train ${formatPercent(at(m.train_acc))}` },
    ...(run.dataset.n_ood > 0
      ? [{ label: 'OOD AUROC', value: formatMetric(at(m.ood_auroc)), hint: `accuracy ${formatPercent(at(m.ood_acc))}` }]
      : []),
    { label: 'Selectivity', value: formatPercent(at(m.selectivity)) },
    { label: 'Control', value: formatPercent(at(m.control_test_acc)), hint: `train ${formatPercent(at(m.control_train_acc))}` },
    { label: 'Embeddings', value: formatMetric(m.test_auroc?.[0] ?? null), hint: 'AUROC' },
    { label: 'Majority class', value: formatPercent(run.majority.test_acc) },
    ...(method === 'logreg' ? [{ label: 'L2 strength', value: String(result.l2?.[position] ?? '–') }] : []),
  ]
  const checks: Finding[] =
    position === 0
      ? [{ level: 'info', text: 'Embeddings are the baseline. Pick a layer to save a probe.' }]
      : trustChecks(run, pooling, method, position)
  if (checks.length === 0) checks.push({ level: 'good', text: 'No red flags.' })

  return (
    <div className="pl-summary">
      <div className="pl-card-title">
        <h3>
          {layerName} · {METHOD_LABELS[method]} · {POOLING_LABELS[pooling]}
          {run.chat_template ? ' · chat template' : ''}
        </h3>
      </div>

      <StatTiles tiles={tiles} />
      <Findings items={checks} />

      {position > 0 && (
        <div className="pl-save">
          <input
            className="pl-input"
            value={name}
            maxLength={255}
            onChange={(e) => setName(e.target.value)}
            aria-label="Probe name"
            style={{ borderColor: colors.border, background: colors.panel, color: colors.text }}
          />
          <button
            type="button"
            className="pl-button"
            onClick={handleSave}
            disabled={saving}
            style={{ borderColor: colors.border, color: colors.text }}
          >
            {saving ? (
              <span className="pl-spinner" style={{ borderColor: colors.border, borderTopColor: colors.text }} />
            ) : (
              <Save className="pl-icon" aria-hidden="true" />
            )}
            Save probe
          </button>
        </div>
      )}
      {savedAs && <div className="pl-save-note">Saved as “{savedAs}”.</div>}
      {saveError && (
        <div className="pl-save-note" style={{ color: '#ef4444' }}>
          {saveError}
        </div>
      )}

      <style>{`
        .probe-lab .pl-save { display: flex; gap: 0.5rem; }
        .probe-lab .pl-save .pl-input { flex: 1; min-width: 0; }
        .probe-lab .pl-save-note { font-size: 0.75rem; margin-top: 0.4rem; }
      `}</style>
    </div>
  )
}
