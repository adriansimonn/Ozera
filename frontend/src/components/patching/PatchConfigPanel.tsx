/**
 * Panel for configuring activation patches.
 * Allows users to select layers, patch types, positions, heads, blend factors, and intervention types.
 */

import { useState } from 'react'
import { Plus, X, Layers, Settings2 } from 'lucide-react'
import type { PatchSpec, PatchType, InterventionType } from '../../types/patching'

interface PatchConfigPanelProps {
  modelId: string
  numLayers: number
  numHeads: number
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

export function PatchConfigPanel({
  numLayers,
  numHeads,
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

  const handleAddPatch = () => {
    const patch: PatchSpec = {
      layer,
      patch_type: patchType,
      intervention_type: interventionType,
      blend_factor: blendFactor,
      positions: positions.trim() ? positions.split(',').map(p => parseInt(p.trim())).filter(n => !isNaN(n)) : null,
      heads: heads.trim() ? heads.split(',').map(h => parseInt(h.trim())).filter(n => !isNaN(n)) : null,
    }
    onAddPatch(patch)
    // Reset inputs
    setPositions('')
    setHeads('')
  }

  const isAttentionType = patchType === 'attention' || patchType === 'attn_output'
  const isAblationType = interventionType !== 'patch'

  return (
    <div className="patch-config-panel">
      <div className="panel-header">
        <Settings2 className="header-icon" />
        <h3>Intervention Configuration</h3>
      </div>

      <div className="config-form">
        <div className="form-row">
          <div className="form-group">
            <label>Intervention Type</label>
            <select
              value={interventionType}
              onChange={e => setInterventionType(e.target.value as InterventionType)}
              disabled={disabled}
              className={isAblationType ? 'ablation-select' : ''}
            >
              {INTERVENTION_TYPES.map(it => (
                <option key={it.value} value={it.value}>
                  {it.label}
                </option>
              ))}
            </select>
            <span className="intervention-hint">
              {INTERVENTION_TYPES.find(it => it.value === interventionType)?.description}
            </span>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Layer</label>
            <select
              value={layer}
              onChange={e => setLayer(parseInt(e.target.value))}
              disabled={disabled}
            >
              {Array.from({ length: numLayers }, (_, i) => (
                <option key={i} value={i}>
                  Layer {i}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Activation Type</label>
            <select
              value={patchType}
              onChange={e => setPatchType(e.target.value as PatchType)}
              disabled={disabled}
            >
              {PATCH_TYPES.map(pt => (
                <option key={pt.value} value={pt.value}>
                  {pt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Positions (comma-separated, empty = all)</label>
            <input
              type="text"
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
                value={heads}
                onChange={e => setHeads(e.target.value)}
                placeholder="e.g., 0,1 or leave empty"
                disabled={disabled}
              />
            </div>
          )}
        </div>

        <div className="form-row">
          <div className="form-group blend-group">
            <label>Blend Factor: {blendFactor.toFixed(2)}</label>
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
              <span>{isAblationType ? 'Full Ablation' : 'Full Replace'}</span>
            </div>
          </div>
        </div>

        <button
          className="add-patch-btn"
          onClick={handleAddPatch}
          disabled={disabled}
        >
          <Plus className="btn-icon" />
          Add Patch
        </button>
      </div>

      {patches.length > 0 && (
        <div className="patches-list">
          <div className="list-header">
            <Layers className="header-icon" />
            <span>Active Patches ({patches.length})</span>
            <button
              className="clear-btn"
              onClick={onClearPatches}
              disabled={disabled}
            >
              Clear All
            </button>
          </div>

          {patches.map((patch, index) => (
            <div key={index} className={`patch-item ${patch.intervention_type !== 'patch' ? 'ablation-item' : ''}`}>
              <div className="patch-info">
                <span className={`patch-intervention ${patch.intervention_type !== 'patch' ? 'ablation' : ''}`}>
                  {patch.intervention_type === 'patch' ? 'PATCH' :
                   patch.intervention_type === 'zero_ablate' ? 'ZERO' :
                   patch.intervention_type === 'mean_ablate' ? 'MEAN' : 'NOISE'}
                </span>
                <span className="patch-layer">L{patch.layer}</span>
                <span className="patch-type">{patch.patch_type}</span>
                {patch.positions && (
                  <span className="patch-detail">pos: [{patch.positions.join(',')}]</span>
                )}
                {patch.heads && (
                  <span className="patch-detail">heads: [{patch.heads.join(',')}]</span>
                )}
                <span className="patch-blend">{(patch.blend_factor * 100).toFixed(0)}%</span>
              </div>
              <button
                className="remove-patch-btn"
                onClick={() => onRemovePatch(index)}
                disabled={disabled}
              >
                <X className="remove-icon" />
              </button>
            </div>
          ))}
        </div>
      )}

      <style>{`
        .patch-config-panel {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1.25rem;
        }

        .panel-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 1rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .panel-header h3 {
          margin: 0;
          font-size: 0.9rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
          letter-spacing: 0.025em;
        }

        .header-icon {
          width: 16px;
          height: 16px;
          color: rgba(255, 255, 255, 0.5);
        }

        .config-form {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .form-row {
          display: flex;
          gap: 1rem;
        }

        .form-group {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 0.375rem;
        }

        .form-group label {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .form-group select,
        .form-group input[type="text"] {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
          padding: 0.625rem 0.75rem;
          font-size: 0.875rem;
          transition: border-color 0.2s;
        }

        .form-group select:focus,
        .form-group input[type="text"]:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .form-group select:disabled,
        .form-group input:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .blend-group {
          flex: 2;
        }

        .blend-group input[type="range"] {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          cursor: pointer;
        }

        .blend-group input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          background: #3b82f6;
          border-radius: 50%;
          cursor: pointer;
        }

        .blend-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
          margin-top: 0.25rem;
        }

        .add-patch-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          background: rgba(59, 130, 246, 0.2);
          border: 1px solid rgba(59, 130, 246, 0.3);
          color: rgba(59, 130, 246, 1);
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .add-patch-btn:hover:not(:disabled) {
          background: rgba(59, 130, 246, 0.3);
          border-color: rgba(59, 130, 246, 0.5);
        }

        .add-patch-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-icon {
          width: 16px;
          height: 16px;
        }

        .patches-list {
          margin-top: 1.25rem;
          padding-top: 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .list-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.75rem;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.7);
        }

        .clear-btn {
          margin-left: auto;
          padding: 0.25rem 0.5rem;
          background: transparent;
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.8);
          font-size: 0.7rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .clear-btn:hover:not(:disabled) {
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.5);
        }

        .patch-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.625rem 0.75rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          margin-bottom: 0.5rem;
        }

        .patch-info {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex-wrap: wrap;
        }

        .intervention-hint {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
          margin-top: 0.25rem;
        }

        .ablation-select {
          border-color: rgba(168, 85, 247, 0.4) !important;
        }

        .patch-intervention {
          font-size: 0.65rem;
          font-weight: 600;
          padding: 0.125rem 0.375rem;
          background: rgba(59, 130, 246, 0.2);
          border: 1px solid rgba(59, 130, 246, 0.3);
          color: rgba(59, 130, 246, 0.9);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .patch-intervention.ablation {
          background: rgba(168, 85, 247, 0.2);
          border-color: rgba(168, 85, 247, 0.3);
          color: rgba(168, 85, 247, 0.9);
        }

        .patch-item.ablation-item {
          border-color: rgba(168, 85, 247, 0.2);
        }

        .patch-layer {
          font-weight: 600;
          color: rgba(59, 130, 246, 0.9);
          font-size: 0.85rem;
        }

        .patch-type {
          background: rgba(255, 255, 255, 0.1);
          padding: 0.125rem 0.5rem;
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.8);
        }

        .patch-detail {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          font-family: monospace;
        }

        .patch-blend {
          font-size: 0.75rem;
          color: rgba(34, 197, 94, 0.8);
          font-weight: 500;
        }

        .remove-patch-btn {
          padding: 0.25rem;
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.4);
          cursor: pointer;
          transition: color 0.2s;
        }

        .remove-patch-btn:hover:not(:disabled) {
          color: rgba(239, 68, 68, 0.8);
        }

        .remove-icon {
          width: 14px;
          height: 14px;
        }
      `}</style>
    </div>
  )
}

export default PatchConfigPanel
