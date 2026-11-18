/**
 * SAE Selector component for choosing model, layer, and activation type.
 * Fetches available SAEs from the API and provides dropdown selection.
 */

import { useState, useEffect, useCallback } from 'react'
import { saeClient, type SAEListResponse, type ExternalSAEInfo } from '../../api/client'
import { DEFAULT_SAE_LIST } from '../../data/defaultSAEs'
import { Dropdown } from '../common/Dropdown'

export interface SAESelection {
  model: 'nano' | 'mini'
  layer: number
  activationType: 'residual' | 'mlp_output'
  // For external SAEs
  externalId?: string
}

interface SAESelectorProps {
  selection: SAESelection | null
  onSelectionChange: (selection: SAESelection) => void
  className?: string
  refreshKey?: number
}

export function SAESelector({
  selection,
  onSelectionChange,
  className = '',
  refreshKey = 0,
}: SAESelectorProps) {
  // Use static default SAE list immediately - no loading needed
  const [saeList] = useState<SAEListResponse>(DEFAULT_SAE_LIST)
  const [externalSAEs, setExternalSAEs] = useState<ExternalSAEInfo[]>([])

  // Set default selection immediately on mount
  useEffect(() => {
    if (!selection) {
      const firstModel = Object.keys(saeList.models)[0] as 'nano' | 'mini'
      const firstSae = saeList.models[firstModel]?.saes[0]
      if (firstSae) {
        onSelectionChange({
          model: firstModel,
          layer: firstSae.layer,
          activationType: firstSae.activation_type,
        })
      }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch external SAEs in background (non-blocking)
  const fetchExternalSAEs = useCallback(async () => {
    try {
      const result = await saeClient.listLoadedExternalSAEs()
      setExternalSAEs(result.external_saes)
    } catch {
      // External SAEs are optional
    }
  }, [])

  useEffect(() => {
    fetchExternalSAEs()
  }, [refreshKey, fetchExternalSAEs])

  // Get available layers for selected model
  const getAvailableLayers = (): number[] => {
    if (!selection) return []
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return []

    const layers = new Set<number>()
    modelInfo.saes.forEach(sae => layers.add(sae.layer))
    return Array.from(layers).sort((a, b) => a - b)
  }

  // Get available activation types for selected model and layer
  const getAvailableActivationTypes = (): ('residual' | 'mlp_output')[] => {
    if (!selection) return []
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return []

    const types = new Set<'residual' | 'mlp_output'>()
    modelInfo.saes
      .filter(sae => sae.layer === selection.layer)
      .forEach(sae => types.add(sae.activation_type))
    return Array.from(types)
  }

  // Get current SAE info
  const getCurrentSAEInfo = () => {
    if (!selection) return null
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return null

    return modelInfo.saes.find(
      sae => sae.layer === selection.layer && sae.activation_type === selection.activationType
    )
  }

  const handleModelChange = (model: 'nano' | 'mini') => {
    const modelInfo = saeList.models[model]
    if (!modelInfo || modelInfo.saes.length === 0) return

    // Reset to first available layer and type for new model, clear external selection
    const firstSae = modelInfo.saes[0]
    onSelectionChange({
      model,
      layer: firstSae.layer,
      activationType: firstSae.activation_type,
    })
  }

  const handleLayerChange = (layer: number) => {
    if (!selection) return

    const modelInfo = saeList.models[selection.externalId ? 'nano' : selection.model]
    if (!modelInfo) return

    // Check if current activation type is available for new layer
    const availableTypes = modelInfo.saes
      .filter(sae => sae.layer === layer)
      .map(sae => sae.activation_type)

    const newType = availableTypes.includes(selection.activationType)
      ? selection.activationType
      : availableTypes[0]

    onSelectionChange({
      model: selection.externalId ? 'nano' : selection.model,
      layer,
      activationType: newType,
    })
  }

  const handleActivationTypeChange = (activationType: 'residual' | 'mlp_output') => {
    if (!selection) return
    onSelectionChange({
      model: selection.model,
      layer: selection.layer,
      activationType,
    })
  }

  const currentSaeInfo = getCurrentSAEInfo()
  const availableLayers = getAvailableLayers()
  const availableTypes = getAvailableActivationTypes()
  const isExternalSelected = !!selection?.externalId
  const selectedExternalSae = isExternalSelected
    ? externalSAEs.find(e => e.id === selection.externalId)
    : null

  return (
    <div className={`sae-selector ${className ?? ''}`}>
      {/* Selectors */}
      <div className={`sae-pickers${isExternalSelected ? ' is-off' : ''}`}>
        <div>
          <label className="oz-label">Model</label>
          <Dropdown
            value={selection?.model || ''}
            onChange={(v) => handleModelChange(v as 'nano' | 'mini')}
            options={Object.keys(saeList.models).map(model => ({ value: model, label: `ozera-${model}` }))}
          />
        </div>

        <div>
          <label className="oz-label">Layer</label>
          <Dropdown
            value={String(selection?.layer ?? '')}
            onChange={(v) => handleLayerChange(parseInt(v))}
            options={availableLayers.map(layer => ({ value: String(layer), label: `Layer ${layer}` }))}
          />
        </div>

        <div className="sae-picker-wide">
          <label className="oz-label">Activation type</label>
          <Dropdown
            value={selection?.activationType || ''}
            onChange={(v) => handleActivationTypeChange(v as 'residual' | 'mlp_output')}
            options={availableTypes.map(type => ({ value: type, label: type === 'residual' ? 'Residual Stream' : 'MLP Output' }))}
          />
        </div>
      </div>

      {/* SAE Info - Built-in */}
      {currentSaeInfo && !isExternalSelected && (
        <dl className="sae-specs">
          <div>
            <dt>d_input</dt>
            <dd>{currentSaeInfo.d_input || '-'}</dd>
          </div>
          <div>
            <dt>d_hidden</dt>
            <dd>{currentSaeInfo.d_hidden || '-'}</dd>
          </div>
          <div>
            <dt>Activation</dt>
            <dd>{currentSaeInfo.activation || 'relu'}</dd>
          </div>
          {currentSaeInfo.training ? (
            <div>
              <dt>Final L0</dt>
              <dd>{currentSaeInfo.training.final_l0?.toFixed(1) || '-'}</dd>
            </div>
          ) : (
            <div>
              <dt>Params</dt>
              <dd>
                {currentSaeInfo.num_parameters
                  ? `${(currentSaeInfo.num_parameters / 1e6).toFixed(1)}M`
                  : '-'}
              </dd>
            </div>
          )}
        </dl>
      )}

      {/* SAE Info - External */}
      {selectedExternalSae && (
        <dl className="sae-specs">
          <div>
            <dt>d_input</dt>
            <dd>{selectedExternalSae.d_input || '-'}</dd>
          </div>
          <div>
            <dt>d_hidden</dt>
            <dd>{selectedExternalSae.d_hidden?.toLocaleString() || '-'}</dd>
          </div>
          <div>
            <dt>Activation</dt>
            <dd>{selectedExternalSae.activation_type || '-'}</dd>
          </div>
          <div>
            <dt>Base model</dt>
            <dd>{selectedExternalSae.base_model || '-'}</dd>
          </div>
        </dl>
      )}

      <p className="oz-hint sae-count">{saeList.total_saes + externalSAEs.length} SAEs available</p>

      {/* External SAEs */}
      {externalSAEs.length > 0 && (
        <div className="sae-external">
          <div className="oz-label">External SAEs ({externalSAEs.length})</div>
          {externalSAEs.map(ext => {
            const isSelected = selection?.externalId === ext.id
            return (
              <button
                key={ext.id}
                className={`sae-external-item${isSelected ? ' is-active' : ''}`}
                aria-pressed={isSelected}
                onClick={() => onSelectionChange({
                  model: 'nano',
                  layer: 0,
                  activationType: 'residual',
                  externalId: ext.id,
                })}
                title={`${ext.source_id || ext.id} - ${ext.base_model || 'unknown model'}`}
              >
                <span className="sae-external-name">{ext.display_name || ext.id}</span>
                <span className="sae-external-meta">
                  {ext.base_model || 'unknown model'}
                  {ext.hookpoint ? ` · ${ext.hookpoint}` : ''}
                  {ext.d_hidden ? ` · ${ext.d_hidden.toLocaleString()} features` : ''}
                  {ext.activation_type ? ` · ${ext.activation_type}` : ''}
                </span>
              </button>
            )
          })}
        </div>
      )}

      <style>{`
        .sae-selector .sae-pickers {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.9rem 6px;
          transition: opacity 0.2s ease;
        }
        .sae-selector .sae-pickers.is-off {
          opacity: 0.4;
        }
        .sae-selector .sae-picker-wide {
          grid-column: 1 / -1;
        }
        /* The SAE's numbers: a name and its value to a row */
        .sae-selector .sae-specs {
          margin: 1.1rem 0 0;
        }
        .sae-selector .sae-specs > div {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.5rem 0;
          border-top: 1px solid var(--oz-rule);
        }
        .sae-selector .sae-specs > div:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }
        .sae-selector .sae-specs dt {
          color: var(--oz-fg-dim);
          font-size: 0.875rem;
        }
        .sae-selector .sae-specs dd {
          margin: 0;
          overflow: hidden;
          font-family: var(--oz-mono);
          font-size: 0.875rem;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .sae-selector .sae-count {
          margin: 0.7rem 0 0;
          font-variant-numeric: tabular-nums;
        }
        .sae-selector .sae-external {
          margin-top: 1.4rem;
        }
        /* SAEs the user has loaded: a ruled list, the chosen one marked */
        .sae-selector .sae-external-item {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 0.1rem;
          width: 100%;
          padding: 0.6rem 0;
          border-top: 1px solid var(--oz-rule);
          color: var(--oz-fg-dim);
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }
        .sae-selector .sae-external-item:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }
        .sae-selector .sae-external-item:hover,
        .sae-selector .sae-external-item.is-active {
          color: var(--oz-fg);
        }
        .sae-selector .sae-external-item.is-active::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 100%;
          height: 1px;
          background: var(--oz-mark);
        }
        .sae-selector .sae-external-name {
          font-size: 0.9375rem;
          font-weight: 500;
          letter-spacing: -0.01em;
        }
        .sae-selector .sae-external-meta {
          font-size: 0.8rem;
        }
      `}</style>
    </div>
  )
}
