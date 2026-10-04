/**
 * The user's saved probes: what each was trained on and how it scored, and using, exporting or
 * deleting it.
 */

import { useState } from 'react'
import { Download, Crosshair, Trash2 } from 'lucide-react'
import type { SavedProbe } from '../../types/probes'
import { apiClient } from '../../api/client'
import { ConfirmModal } from '../common/ConfirmModal'
import { useProbeColors } from './probeTheme'
import { METHOD_SHORT, POOLING_LABELS, downloadJson, formatMetric } from './probeUtils'

interface SavedProbesProps {
  probes: SavedProbe[]
  loading: boolean
  activeId: number | null
  onUse: (probe: SavedProbe) => void
  onDeleted: (probeId: number) => void
}

function metric(probe: SavedProbe, key: string): number | null {
  const value = probe.metrics[key]
  return typeof value === 'number' ? value : null
}

export function SavedProbes({ probes, loading, activeId, onUse, onDeleted }: SavedProbesProps) {
  const colors = useProbeColors()
  const [deleting, setDeleting] = useState<SavedProbe | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleExport = async (probe: SavedProbe) => {
    setError(null)
    try {
      const full = await apiClient.getProbe(probe.id, true)
      downloadJson(`${probe.name.replace(/[^\w.-]+/g, '_')}.probe.json`, {
        name: full.name,
        model: full.model_id,
        layer: full.layer,
        reads: `residual stream after decoder layer ${full.layer}`,
        pooling: full.pooling,
        method: full.method,
        chat_template: full.chat_template,
        read_span: full.read_span,
        score: 'weights · activation + bias (raw activations; > 0 means the positive class)',
        weights: full.weights,
        bias: full.bias,
        normalization: full.normalization,
        metrics: full.metrics,
        dataset: full.dataset,
        created_at: full.created_at,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    }
  }

  const handleDelete = async () => {
    if (!deleting) return
    setBusy(true)
    setError(null)
    try {
      await apiClient.deleteProbe(deleting.id)
      onDeleted(deleting.id)
      setDeleting(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="pl-saved">
      <div className="pl-card-title">
        <h3>Saved probes ({probes.length})</h3>
      </div>

      {error && <div className="pl-error-note" style={{ margin: '0 0 0.5rem' }}>{error}</div>}

      {loading && probes.length === 0 ? (
        <div className="pl-note">Loading…</div>
      ) : probes.length === 0 ? (
        <div className="pl-empty" style={{ borderColor: colors.border }}>
          No saved probes yet
        </div>
      ) : (
        <div className="pl-saved-list">
          {probes.map((probe) => {
            const ood = metric(probe, 'ood_auroc')
            return (
              <div
                key={probe.id}
                className="pl-saved-item"
                style={{
                  borderColor: probe.id === activeId ? colors.text : colors.border,
                  background: colors.panel,
                }}
              >
                <div className="pl-saved-main">
                  <div className="pl-saved-name">{probe.name}</div>
                  <div className="pl-saved-meta">
                    {probe.model_id} · layer {probe.layer} · {POOLING_LABELS[probe.pooling].toLowerCase()} ·{' '}
                    {METHOD_SHORT[probe.method]}
                    {probe.chat_template ? ' · chat' : ''}
                  </div>
                  <div className="pl-saved-meta">
                    AUROC {formatMetric(metric(probe, 'test_auroc'))}
                    {ood != null ? ` · OOD ${formatMetric(ood)}` : ''} · {new Date(probe.created_at).toLocaleDateString()}
                  </div>
                  {(!probe.model_available || probe.model_changed) && (
                    <div className="pl-saved-meta" style={{ color: colors.warning }}>
                      {probe.model_available ? 'Model changed since saving' : 'Model unavailable'}
                    </div>
                  )}
                </div>
                <div className="pl-saved-actions">
                  <button
                    type="button"
                    className="pl-icon-button"
                    title="Score text with this probe"
                    aria-label={`Use ${probe.name} in the token heatmap`}
                    disabled={!probe.model_available}
                    onClick={() => onUse(probe)}
                    style={{ color: colors.text }}
                  >
                    <Crosshair className="pl-icon" />
                  </button>
                  <button
                    type="button"
                    className="pl-icon-button"
                    title="Export as JSON (with weights)"
                    aria-label={`Export ${probe.name}`}
                    onClick={() => handleExport(probe)}
                    style={{ color: colors.text }}
                  >
                    <Download className="pl-icon" />
                  </button>
                  <button
                    type="button"
                    className="pl-icon-button"
                    title="Delete"
                    aria-label={`Delete ${probe.name}`}
                    onClick={() => setDeleting(probe)}
                    style={{ color: colors.text }}
                  >
                    <Trash2 className="pl-icon" />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <ConfirmModal
        isOpen={deleting != null}
        title="Delete probe"
        message={`Delete “${deleting?.name ?? ''}”? This can't be undone.`}
        onConfirm={handleDelete}
        onCancel={() => setDeleting(null)}
        isLoading={busy}
      />

      <style>{`
        .probe-lab .pl-saved-list { display: flex; flex-direction: column; gap: 0.4rem; max-height: 420px; overflow-y: auto; }
        .probe-lab .pl-saved-item { display: flex; gap: 0.5rem; align-items: flex-start; padding: 0.55rem 0.65rem; border: 1px solid; }
        .probe-lab .pl-saved-main { flex: 1; min-width: 0; }
        .probe-lab .pl-saved-name { font-size: 0.82rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .probe-lab .pl-saved-meta { font-size: 0.72rem; margin-top: 0.1rem; }
        .probe-lab .pl-saved-actions { display: flex; gap: 0.15rem; }
      `}</style>
    </div>
  )
}
