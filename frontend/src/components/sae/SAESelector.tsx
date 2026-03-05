/**
 * SAE Selector component for choosing model, layer, and activation type.
 * Fetches available SAEs from the API and provides dropdown selection.
 */

import { useState, useEffect, useCallback } from 'react'
import { Loader2, AlertCircle, RefreshCw, Database, ExternalLink } from 'lucide-react'
import { saeClient, type SAEListResponse } from '../../api/client'
import { useThemeColors } from '../../hooks/useTheme'
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
  const tc = useThemeColors()
  const [saeList, setSaeList] = useState<SAEListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch available SAEs
  const fetchSAEs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await saeClient.listSAEs()
      setSaeList(data)

      // If no selection, set default
      if (!selection && data.total_saes > 0) {
        const firstModel = Object.keys(data.models)[0] as 'nano' | 'mini'
        const firstSae = data.models[firstModel]?.saes[0]
        if (firstSae) {
          onSelectionChange({
            model: firstModel,
            layer: firstSae.layer,
            activationType: firstSae.activation_type,
          })
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch SAEs')
    } finally {
      setLoading(false)
    }
  }, [selection, onSelectionChange])

  useEffect(() => {
    fetchSAEs()
  }, [refreshKey]) // Fetch on mount and when refreshKey changes

  // Get available layers for selected model
  const getAvailableLayers = (): number[] => {
    if (!saeList || !selection) return []
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return []

    const layers = new Set<number>()
    modelInfo.saes.forEach(sae => layers.add(sae.layer))
    return Array.from(layers).sort((a, b) => a - b)
  }

  // Get available activation types for selected model and layer
  const getAvailableActivationTypes = (): ('residual' | 'mlp_output')[] => {
    if (!saeList || !selection) return []
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
    if (!saeList || !selection) return null
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return null

    return modelInfo.saes.find(
      sae => sae.layer === selection.layer && sae.activation_type === selection.activationType
    )
  }

  const handleModelChange = (model: 'nano' | 'mini') => {
    const modelInfo = saeList?.models[model]
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
    if (!selection || !saeList) return

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
    ? saeList?.external_saes?.find(e => e.id === selection.externalId)
    : null

  if (loading) {
    return (
      <div style={{ background: tc.surface, backdropFilter: 'blur(20px)', border: `1px solid ${tc.border}`, padding: '1rem' }} className={className}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: tc.textMuted }}>
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Loading available SAEs...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ background: tc.surface, backdropFilter: 'blur(20px)', border: '1px solid rgba(239,68,68,0.3)', padding: '1rem' }} className={className}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'rgba(239,68,68,0.9)' }}>
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchSAEs}
            style={{ padding: '0.25rem 0.75rem', fontSize: '0.875rem', border: `1px solid ${tc.border}`, background: 'transparent', color: tc.textMuted, cursor: 'pointer', transition: 'all 0.2s' }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = tc.borderHover; e.currentTarget.style.color = tc.textMid }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = tc.border; e.currentTarget.style.color = tc.textMuted }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  const hasExternalSAEs = saeList?.external_saes && saeList.external_saes.length > 0
  if (!saeList || (saeList.total_saes === 0 && !hasExternalSAEs)) {
    return (
      <div style={{ background: tc.surface, backdropFilter: 'blur(20px)', border: `1px solid ${tc.border}`, padding: '1rem' }} className={className}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: tc.textFaint }}>
          <Database className="w-5 h-5" />
          <span>No SAEs available. Deploy SAEs to Modal first.</span>
        </div>
      </div>
    )
  }

  return (
    <div style={{ background: tc.surface, backdropFilter: 'blur(20px)', border: `1px solid ${tc.border}` }} className={className}>
      {/* Header */}
      <div style={{ padding: '1rem', borderBottom: `1px solid ${tc.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database className="w-4 h-4" style={{ color: tc.textMid }} />
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: tc.textStrong, textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
              SAE Selection
            </h3>
          </div>
          <div style={{ fontSize: '0.75rem', color: tc.textSub }}>
            {saeList.total_saes} SAEs available
          </div>
        </div>
      </div>

      {/* Selectors */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', background: tc.surfaceActive, opacity: isExternalSelected ? 0.4 : 1, transition: 'opacity 0.2s' }}>
        {/* Model Selector */}
        <div style={{ background: tc.deepBg, padding: '1rem' }}>
          <label style={{ display: 'block', fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>
            Model
          </label>
          <Dropdown
            value={selection?.model || ''}
            onChange={(v) => handleModelChange(v as 'nano' | 'mini')}
            options={Object.keys(saeList.models).map(model => ({ value: model, label: `ozera-${model}` }))}
          />
        </div>

        {/* Layer Selector */}
        <div style={{ background: tc.deepBg, padding: '1rem' }}>
          <label style={{ display: 'block', fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>
            Layer
          </label>
          <Dropdown
            value={String(selection?.layer ?? '')}
            onChange={(v) => handleLayerChange(parseInt(v))}
            options={availableLayers.map(layer => ({ value: String(layer), label: `Layer ${layer}` }))}
          />
        </div>

        {/* Activation Type Selector */}
        <div style={{ background: tc.deepBg, padding: '1rem' }}>
          <label style={{ display: 'block', fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>
            Activation Type
          </label>
          <Dropdown
            value={selection?.activationType || ''}
            onChange={(v) => handleActivationTypeChange(v as 'residual' | 'mlp_output')}
            options={availableTypes.map(type => ({ value: type, label: type === 'residual' ? 'Residual Stream' : 'MLP Output' }))}
          />
        </div>
      </div>

      {/* SAE Info - Built-in */}
      {currentSaeInfo && !isExternalSelected && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: tc.surfaceActive, borderTop: `1px solid ${tc.border}` }}>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_input</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{currentSaeInfo.d_input || '-'}</div>
          </div>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_hidden</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{currentSaeInfo.d_hidden || '-'}</div>
          </div>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Activation</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{currentSaeInfo.activation || 'relu'}</div>
          </div>
          {currentSaeInfo.training && (
            <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Final L0</div>
              <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: 'rgba(34,197,94,0.9)' }}>
                {currentSaeInfo.training.final_l0?.toFixed(1) || '-'}
              </div>
            </div>
          )}
          {!currentSaeInfo.training && (
            <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Params</div>
              <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>
                {currentSaeInfo.num_parameters
                  ? `${(currentSaeInfo.num_parameters / 1e6).toFixed(1)}M`
                  : '-'}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SAE Info - External */}
      {selectedExternalSae && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: tc.surfaceActive, borderTop: `1px solid ${tc.border}` }}>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_input</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{selectedExternalSae.d_input || '-'}</div>
          </div>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_hidden</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{selectedExternalSae.d_hidden?.toLocaleString() || '-'}</div>
          </div>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Activation</div>
            <div style={{ fontSize: '0.875rem', fontFamily: 'monospace', color: tc.text }}>{selectedExternalSae.activation_type || '-'}</div>
          </div>
          <div style={{ background: tc.deepBg, padding: '0.75rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Base Model</div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: tc.textStrong }}>{selectedExternalSae.base_model || '-'}</div>
          </div>
        </div>
      )}

      {/* External SAEs */}
      {saeList.external_saes && saeList.external_saes.length > 0 && (
        <div style={{ borderTop: `1px solid ${tc.border}` }}>
          <div style={{ padding: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ExternalLink className="w-3 h-3" style={{ color: tc.textSub }} />
            <span style={{ fontSize: '0.75rem', color: tc.textSub, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              External SAEs ({saeList.external_saes.length})
            </span>
          </div>
          <div style={{ padding: '0 0.75rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {saeList.external_saes.map(ext => {
              const isSelected = selection?.externalId === ext.id
              return (
                <button
                  key={ext.id}
                  onClick={() => onSelectionChange({
                    model: 'nano',
                    layer: 0,
                    activationType: 'residual',
                    externalId: ext.id,
                  })}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    gap: '0.75rem', padding: '0.5rem 0.75rem', width: '100%', textAlign: 'left',
                    background: isSelected ? tc.surfaceActive : tc.surface,
                    border: `1px solid ${isSelected ? tc.borderHover : tc.border}`,
                    cursor: 'pointer', transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.background = tc.surfaceHover
                      e.currentTarget.style.borderColor = tc.borderHover
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.background = tc.surface
                      e.currentTarget.style.borderColor = tc.border
                    }
                  }}
                  title={`${ext.source_id || ext.id} - ${ext.base_model || 'unknown model'}`}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.125rem' }}>
                    <span style={{ color: isSelected ? tc.textStrong : tc.textStrong, fontSize: '0.8rem', fontWeight: 500 }}>
                      {ext.display_name || ext.id}
                    </span>
                    <span style={{ color: tc.textFaint, fontSize: '0.7rem' }}>
                      {ext.base_model || 'unknown model'} {ext.hookpoint ? `· ${ext.hookpoint}` : ''}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.7rem', color: tc.textFaint, flexShrink: 0 }}>
                    {ext.d_hidden && <span>{ext.d_hidden.toLocaleString()} features</span>}
                    {ext.activation_type && <span>{ext.activation_type}</span>}
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
