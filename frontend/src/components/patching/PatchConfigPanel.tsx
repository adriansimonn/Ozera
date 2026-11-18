/**
 * Panel for configuring activation patches.
 * Allows users to select layers, patch types, positions, heads, blend factors, and intervention types.
 */

import { useState } from 'react'
import {  X } from 'lucide-react'
import type { PatchSpec, PatchType, InterventionType } from '../../types/patching'
import { Dropdown } from '../common/Dropdown'

interface PatchConfigPanelProps {
  modelId: string
  numLayers: number
  numHeads: number
  /** Neurons per MLP layer (null if unknown) */
  mlpNeurons: number | null
  patches: PatchSpec[]
  onAddPatch: (patch: PatchSpec) => void
  onRemovePatch: (index: number) => void
  onClearPatches: () => void
  disabled?: boolean
}

const PATCH_TYPES: { value: PatchType; label: string; description: string }[] = [
  { value: 'residual', label: 'Residual Stream', description: 'Full residual stream activation' },
  { value: 'attention', label: 'Attention', description: 'Attention layer output' },
  { value: 'mlp', label: 'MLP', description: 'MLP/feedforward layer output' },
  { value: 'attn_output', label: 'Attention Output', description: 'Attention output projection' },
  { value: 'ff_output', label: 'FF Output', description: 'Feedforward output' },
  { value: 'post_attn', label: 'Post-Attention', description: 'After attention residual connection' },
  { value: 'post_ff', label: 'Post-FF', description: 'After feedforward residual connection' },
]

const INTERVENTION_TYPES: { value: InterventionType; label: string; description: string }[] = [
  { value: 'patch', label: 'Patch', description: 'Replace with source activations (requires source prompt)' },
  { value: 'zero_ablate', label: 'Zero Ablation', description: 'Zero out activations to measure importance' },
  { value: 'mean_ablate', label: 'Mean Ablation', description: 'Replace with mean activation value' },
  { value: 'noise_ablate', label: 'Noise Ablation', description: 'Replace with Gaussian noise' },
]

/** Parse a comma-separated list of indices ('' = null, meaning all) */
function parseIndices(text: string): number[] | null {
  return text.trim() ? text.split(',').map(p => parseInt(p.trim())).filter(n => !isNaN(n)) : null
}

export function PatchConfigPanel({
  numLayers,
  numHeads,
  mlpNeurons,
  patches,
  onAddPatch,
  onRemovePatch,
  onClearPatches,
  disabled = false,
}: PatchConfigPanelProps) {
  const [layer, setLayer] = useState(0)
  const [patchType, setPatchType] = useState<PatchType>('residual')
  const [interventionType, setInterventionType] = useState<InterventionType>('patch')
  const [blendFactor, setBlendFactor] = useState(1.0)
  const [positions, setPositions] = useState<string>('')
  const [heads, setHeads] = useState<string>('')
  const [neurons, setNeurons] = useState<string>('')

  // Heads can be chosen for attention patches, neurons for MLP patches
  const isAttentionType = patchType === 'attention' || patchType === 'attn_output'
  const isMlpType = patchType === 'mlp' || patchType === 'ff_output'
  const isAblationType = interventionType !== 'patch'

  const handleAddPatch = () => {
    const patch: PatchSpec = {
      layer,
      patch_type: patchType,
      intervention_type: interventionType,
      blend_factor: blendFactor,
      positions: parseIndices(positions),
      heads: isAttentionType ? parseIndices(heads) : null,
      neurons: isMlpType ? parseIndices(neurons) : null,
    }
    onAddPatch(patch)
    // Reset inputs
    setPositions('')
    setHeads('')
    setNeurons('')
  }

  return (
    <div className="patch-config-panel">
      <h2 className="oz-title">Intervention configuration</h2>

      <div className="config-form">
        <div className="form-row">
          <div className="form-group">
            <label>Intervention type</label>
            <Dropdown
              value={interventionType}
              onChange={v => setInterventionType(v as InterventionType)}
              disabled={disabled}
              options={INTERVENTION_TYPES.map(it => ({ value: it.value, label: it.label }))}
            />
            <span className="intervention-hint">
              {INTERVENTION_TYPES.find(it => it.value === interventionType)?.description}
            </span>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Layer</label>
            <Dropdown
              value={String(layer)}
              onChange={v => setLayer(parseInt(v))}
              disabled={disabled}
              options={Array.from({ length: numLayers }, (_, i) => ({ value: String(i), label: `Layer ${i}` }))}
            />
          </div>

          <div className="form-group">
            <label>Activation type</label>
            <Dropdown
              value={patchType}
              onChange={v => setPatchType(v as PatchType)}
              disabled={disabled}
              options={PATCH_TYPES.map(pt => ({ value: pt.value, label: pt.label }))}
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Positions (comma-separated, empty = all)</label>
            <input
              type="text"
              className="oz-input"
              value={positions}
              onChange={e => setPositions(e.target.value)}
              placeholder="e.g., 0,1,2 or leave empty"
              disabled={disabled}
            />
          </div>

          {isAttentionType && (
            <div className="form-group">
              <label>Heads (0-{numHeads - 1}, empty = all)</label>
              <input
                type="text"
                className="oz-input"
                value={heads}
                onChange={e => setHeads(e.target.value)}
                placeholder="e.g., 0,1 or leave empty"
                disabled={disabled}
              />
            </div>
          )}

          {isMlpType && (
            <div className="form-group">
              <label>Neurons ({mlpNeurons !== null ? `0-${mlpNeurons - 1}, ` : ''}empty = all)</label>
              <input
                type="text"
                className="oz-input"
                value={neurons}
                onChange={e => setNeurons(e.target.value)}
                placeholder="e.g., 12,305 or leave empty"
                disabled={disabled}
              />
            </div>
          )}
        </div>

        <div className="form-row">
          <div className="form-group blend-group">
            <label>
              Blend factor <span className="blend-value">{blendFactor.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={blendFactor}
              onChange={e => setBlendFactor(parseFloat(e.target.value))}
              disabled={disabled}
            />
            <div className="blend-labels">
              <span>Original</span>
              <span>{isAblationType ? 'Full ablation' : 'Full replace'}</span>
            </div>
          </div>
        </div>

        <button
          className="oz-btn oz-btn--ghost oz-btn--block add-patch-btn"
          onClick={handleAddPatch}
          disabled={disabled}
        >
          Add Intervention
        </button>
      </div>

      {patches.length > 0 && (
        <div className="patches-list">
          <div className="list-header">
            <span>Active patches</span>
            <span className="list-count">{patches.length}</span>
            <button
              className="oz-link clear-btn"
              onClick={onClearPatches}
              disabled={disabled}
            >
              Clear all
            </button>
          </div>

          {patches.map((patch, index) => (
            <div key={index} className={`patch-item ${patch.intervention_type !== 'patch' ? 'ablation-item' : ''}`}>
              <div className="patch-info">
                <span className={`oz-tag ${patch.intervention_type === 'patch' ? 'oz-tag--solid' : ''}`}>
                  {patch.intervention_type === 'patch' ? 'Patch' :
                   patch.intervention_type === 'zero_ablate' ? 'Zero' :
                   patch.intervention_type === 'mean_ablate' ? 'Mean' : 'Noise'}
                </span>
                <span className="patch-layer">L{patch.layer}</span>
                <span className="patch-type">{patch.patch_type}</span>
                {patch.positions && (
                  <span className="patch-detail">pos: [{patch.positions.join(',')}]</span>
                )}
                {patch.heads && (
                  <span className="patch-detail">heads: [{patch.heads.join(',')}]</span>
                )}
                {patch.neurons && (
                  <span className="patch-detail">neurons: [{patch.neurons.join(',')}]</span>
                )}
                <span className="patch-blend">{(patch.blend_factor * 100).toFixed(0)}%</span>
              </div>
              <button
                className="oz-icon-btn"
                onClick={() => onRemovePatch(index)}
                disabled={disabled}
                aria-label="Remove intervention"
                title="Remove"
              >
                <X />
              </button>
            </div>
          ))}
        </div>
      )}

      <style>{`
        .patch-config-panel .config-form {
          display: flex;
          flex-direction: column;
          gap: 1.1rem;
        }

        .patch-config-panel .form-row {
          display: flex;
          gap: 0.9rem;
        }

        .patch-config-panel .form-group {
          display: flex;
          flex-direction: column;
          flex: 1;
          min-width: 0;
        }

        .patch-config-panel .form-group label {
          margin-bottom: 0.45rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }

        .patch-config-panel .blend-value {
          margin-left: 0.3rem;
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }

        .patch-config-panel .intervention-hint {
          margin-top: 0.45rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
        }

        .patch-config-panel .blend-group input[type="range"] {
          width: 100%;
        }

        .patch-config-panel .blend-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 0.15rem;
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
        }

        /* The interventions added so far: a ruled list */
        .patch-config-panel .patches-list {
          margin-top: 1.5rem;
        }

        .patch-config-panel .list-header {
          display: flex;
          align-items: baseline;
          gap: 0.6rem;
          margin-bottom: 0.6rem;
          font-size: 0.9375rem;
          font-weight: 500;
        }

        .patch-config-panel .list-count {
          color: var(--oz-fg-dim);
          font-weight: 400;
          font-variant-numeric: tabular-nums;
        }

        .patch-config-panel .clear-btn {
          margin-left: auto;
          font-size: 0.85rem;
        }

        .patch-config-panel .patch-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          padding: 0.45rem 0;
          border-top: 1px solid var(--oz-rule);
        }

        .patch-config-panel .patch-item:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .patch-config-panel .patch-info {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: 0.35rem 0.75rem;
          min-width: 0;
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }

        .patch-config-panel .patch-info .oz-tag {
          font-family: Inter, sans-serif;
        }

        .patch-config-panel .patch-layer {
          font-weight: 500;
        }

        .patch-config-panel .patch-detail,
        .patch-config-panel .patch-blend {
          color: var(--oz-fg-dim);
        }

        @media (max-width: 640px) {
          .patch-config-panel .form-row { flex-direction: column; }
        }
      `}</style>
    </div>
  )
}

export default PatchConfigPanel
