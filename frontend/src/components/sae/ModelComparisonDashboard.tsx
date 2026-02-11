/**
 * Model Comparison Dashboard - Research feature for comparing SAE representations.
 *
 * Enables researchers to:
 * - Compare features between two SAEs (same or different models/layers)
 * - View layer-by-layer CKA similarity matrices
 * - Identify shared vs unique features between models
 * - Analyze how representations emerge across layers
 * - Select external SAEs for comparison (up to 2)
 */

import { useState, useCallback, useEffect } from 'react'
import {
  GitCompare,
  Play,
  Loader2,
  AlertCircle,
  ChevronDown,
  Database,
  Grid3x3,
  Zap,
  ExternalLink,
  X,
} from 'lucide-react'
import {
  saeClient,
  type SAEListResponse,
  type SAECompareResponse,
  type SAECompareLayersResponse,
  type ExternalSAEInfo,
} from '../../api/client'
import { FeatureComparison } from './FeatureComparison'
import { SimilarityMatrix } from './SimilarityMatrix'

interface SAECompareSelection {
  model: 'nano' | 'mini'
  layer: number
  activationType: 'residual' | 'mlp_output'
  externalId?: string
}

interface ModelComparisonDashboardProps {
  onFeatureSelect?: (saeId: string, featureIdx: number) => void
  className?: string
}

type CompareMode = 'features' | 'layers'

export function ModelComparisonDashboard({
  onFeatureSelect,
  className = '',
}: ModelComparisonDashboardProps) {
  const [saeList, setSaeList] = useState<SAEListResponse | null>(null)
  const [loadingList, setLoadingList] = useState(true)

  // Selections
  const [selectionA, setSelectionA] = useState<SAECompareSelection | null>(null)
  const [selectionB, setSelectionB] = useState<SAECompareSelection | null>(null)
  const [compareText, setCompareText] = useState('')
  const [compareMode, setCompareMode] = useState<CompareMode>('features')

  // Results
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [featureResult, setFeatureResult] = useState<SAECompareResponse | null>(null)
  const [layerResult, setLayerResult] = useState<SAECompareLayersResponse | null>(null)

  // Fetch available SAEs
  useEffect(() => {
    const fetchSAEs = async () => {
      try {
        const data = await saeClient.listSAEs()
        setSaeList(data)

        // Set default selections
        const models = Object.keys(data.models) as ('nano' | 'mini')[]
        if (models.length > 0) {
          const firstModel = models[0]
          const saes = data.models[firstModel]?.saes || []
          if (saes.length > 0) {
            setSelectionA({
              model: firstModel,
              layer: saes[0].layer,
              activationType: saes[0].activation_type,
            })
            // Default B to a different layer or the second model
            const secondSae = saes.length > 2 ? saes[2] : saes[saes.length - 1]
            if (models.length > 1) {
              const secondModel = models[1]
              const secondSaes = data.models[secondModel]?.saes || []
              if (secondSaes.length > 0) {
                setSelectionB({
                  model: secondModel,
                  layer: secondSaes[0].layer,
                  activationType: secondSaes[0].activation_type,
                })
              }
            } else {
              setSelectionB({
                model: firstModel,
                layer: secondSae.layer,
                activationType: secondSae.activation_type,
              })
            }
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load SAEs')
      } finally {
        setLoadingList(false)
      }
    }
    fetchSAEs()
  }, [])

  // Run comparison
  const runComparison = useCallback(async () => {
    if (!selectionA || !selectionB || !compareText.trim()) return

    setComparing(true)
    setError(null)

    try {
      if (compareMode === 'features') {
        const result = await saeClient.compareSAEs({
          model_a: selectionA.model,
          layer_a: selectionA.layer,
          activation_type_a: selectionA.activationType,
          model_b: selectionB.model,
          layer_b: selectionB.layer,
          activation_type_b: selectionB.activationType,
          external_id_a: selectionA.externalId,
          external_id_b: selectionB.externalId,
          text: compareText,
          top_k: 100,
        })

        if (result.error) {
          throw new Error(result.error)
        }

        setFeatureResult(result)
        setLayerResult(null)
      } else {
        const result = await saeClient.compareLayers({
          model_a: selectionA.model,
          activation_type_a: selectionA.activationType,
          model_b: selectionB.model,
          activation_type_b: selectionB.activationType,
          text: compareText,
        })

        if (result.error) {
          throw new Error(result.error)
        }

        setLayerResult(result)
        setFeatureResult(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setComparing(false)
    }
  }, [selectionA, selectionB, compareText, compareMode])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runComparison()
    }
  }

  // Helper to get available layers for a model
  const getAvailableLayers = (model: string): number[] => {
    if (!saeList) return []
    const modelInfo = saeList.models[model]
    if (!modelInfo) return []
    const layers = new Set<number>()
    modelInfo.saes.forEach((sae) => layers.add(sae.layer))
    return Array.from(layers).sort((a, b) => a - b)
  }

  const getAvailableTypes = (
    model: string,
    layer: number
  ): ('residual' | 'mlp_output')[] => {
    if (!saeList) return []
    const modelInfo = saeList.models[model]
    if (!modelInfo) return []
    const types = new Set<'residual' | 'mlp_output'>()
    modelInfo.saes
      .filter((sae) => sae.layer === layer)
      .forEach((sae) => types.add(sae.activation_type))
    return Array.from(types)
  }

  // Get SAE info for built-in SAE
  const getSaeInfo = (selection: SAECompareSelection | null) => {
    if (!selection || !saeList || selection.externalId) return null
    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return null
    return modelInfo.saes.find(
      sae => sae.layer === selection.layer && sae.activation_type === selection.activationType
    )
  }

  // Get external SAE info
  const getExternalSaeInfo = (selection: SAECompareSelection | null): ExternalSAEInfo | null => {
    if (!selection?.externalId || !saeList?.external_saes) return null
    return saeList.external_saes.find(e => e.id === selection.externalId) || null
  }

  const handleSelectionChange = (
    which: 'a' | 'b',
    field: 'model' | 'layer' | 'activationType',
    value: string | number
  ) => {
    const setter = which === 'a' ? setSelectionA : setSelectionB
    const current = which === 'a' ? selectionA : selectionB
    if (!current || !saeList) return

    if (field === 'model') {
      const modelName = value as 'nano' | 'mini'
      const modelInfo = saeList.models[modelName]
      if (!modelInfo || modelInfo.saes.length === 0) return
      const firstSae = modelInfo.saes[0]
      setter({
        model: modelName,
        layer: firstSae.layer,
        activationType: firstSae.activation_type,
      })
    } else if (field === 'layer') {
      const layer = value as number
      const types = getAvailableTypes(current.model, layer)
      setter({
        ...current,
        layer,
        activationType: types.includes(current.activationType)
          ? current.activationType
          : types[0],
        externalId: undefined,
      })
    } else {
      setter({
        ...current,
        activationType: value as 'residual' | 'mlp_output',
        externalId: undefined,
      })
    }
  }

  const handleExternalSelect = (which: 'a' | 'b', ext: ExternalSAEInfo) => {
    const setter = which === 'a' ? setSelectionA : setSelectionB
    setter({
      model: 'nano',
      layer: 0,
      activationType: 'residual',
      externalId: ext.id,
    })
    // Layer comparison doesn't support external SAEs, switch to features mode
    if (compareMode === 'layers') {
      setCompareMode('features')
    }
  }

  const handleClearExternal = (which: 'a' | 'b') => {
    const setter = which === 'a' ? setSelectionA : setSelectionB
    if (!saeList) return
    const models = Object.keys(saeList.models) as ('nano' | 'mini')[]
    if (models.length > 0) {
      const model = models[0]
      const saes = saeList.models[model]?.saes || []
      if (saes.length > 0) {
        setter({
          model,
          layer: saes[0].layer,
          activationType: saes[0].activation_type,
        })
      }
    }
  }

  const saeAName = selectionA
    ? selectionA.externalId
      ? getExternalSaeInfo(selectionA)?.display_name || selectionA.externalId
      : `${selectionA.model}-L${selectionA.layer}-${selectionA.activationType}`
    : ''
  const saeBName = selectionB
    ? selectionB.externalId
      ? getExternalSaeInfo(selectionB)?.display_name || selectionB.externalId
      : `${selectionB.model}-L${selectionB.layer}-${selectionB.activationType}`
    : ''

  const externalSAEs = saeList?.external_saes || []
  const saeInfoA = getSaeInfo(selectionA)
  const saeInfoB = getSaeInfo(selectionB)
  const extInfoA = getExternalSaeInfo(selectionA)
  const extInfoB = getExternalSaeInfo(selectionB)
  const hasExternalSelection = !!(selectionA?.externalId || selectionB?.externalId)

  if (loadingList) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'rgba(255,255,255,0.4)' }} />
        <span className="ml-3 text-gray-400">Loading SAEs...</span>
      </div>
    )
  }

  const renderSaePanel = (
    which: 'a' | 'b',
    selection: SAECompareSelection | null,
    saeInfo: typeof saeInfoA,
    extInfo: ExternalSAEInfo | null
  ) => {
    const isExternal = !!selection?.externalId
    const label = which === 'a' ? 'SAE A' : 'SAE B'

    return (
      <div style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.1)' }}>
        {/* Header */}
        <div className="p-3 border-b border-gray-800 flex items-center gap-2">
          <Database className="w-4 h-4" style={{ color: 'rgba(255,255,255,0.6)' }} />
          <span className="text-sm font-semibold uppercase tracking-wide" style={{ color: 'rgba(255,255,255,0.7)' }}>
            {label}
          </span>
          {isExternal && (
            <span style={{ fontSize: '0.65rem', padding: '0.125rem 0.375rem', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)', color: 'rgba(255,255,255,0.5)' }}>
              EXTERNAL
            </span>
          )}
        </div>

        {/* Ozera SAE Dropdowns */}
        <div className="grid grid-cols-3 gap-px bg-gray-800" style={{ opacity: isExternal ? 0.35 : 1, pointerEvents: isExternal ? 'none' : 'auto', transition: 'opacity 0.2s' }}>
          <div className="bg-black/40 p-3">
            <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
              Model
            </label>
            <div className="relative">
              <select
                value={selection?.model || ''}
                onChange={(e) =>
                  handleSelectionChange(which, 'model', e.target.value)
                }
                disabled={isExternal}
                className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm cursor-pointer"
                style={{ outline: 'none' }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
              >
                {saeList &&
                  Object.keys(saeList.models).map((model) => (
                    <option key={model} value={model}>
                      ozera-{model}
                    </option>
                  ))}
              </select>
              <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none" />
            </div>
          </div>
          <div className="bg-black/40 p-3">
            <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
              Layer
            </label>
            <div className="relative">
              <select
                value={selection?.layer ?? ''}
                onChange={(e) =>
                  handleSelectionChange(which, 'layer', parseInt(e.target.value))
                }
                disabled={isExternal}
                className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm cursor-pointer"
                style={{ outline: 'none' }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
              >
                {selection && !isExternal &&
                  getAvailableLayers(selection.model).map((layer) => (
                    <option key={layer} value={layer}>
                      Layer {layer}
                    </option>
                  ))}
              </select>
              <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none" />
            </div>
          </div>
          <div className="bg-black/40 p-3">
            <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
              Type
            </label>
            <div className="relative">
              <select
                value={selection?.activationType || ''}
                onChange={(e) =>
                  handleSelectionChange(which, 'activationType', e.target.value)
                }
                disabled={isExternal}
                className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm cursor-pointer"
                style={{ outline: 'none' }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
              >
                {selection && !isExternal &&
                  getAvailableTypes(selection.model, selection.layer).map(
                    (type) => (
                      <option key={type} value={type}>
                        {type === 'residual' ? 'Residual' : 'MLP'}
                      </option>
                    )
                  )}
              </select>
              <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500 pointer-events-none" />
            </div>
          </div>
        </div>

        {/* SAE Info - Built-in */}
        {saeInfo && !isExternal && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'rgba(255,255,255,0.1)', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_input</div>
              <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{saeInfo.d_input || '-'}</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_hidden</div>
              <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{saeInfo.d_hidden || '-'}</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Activation</div>
              <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{saeInfo.activation || 'relu'}</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Base Model</div>
              <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'rgba(255,255,255,0.8)' }}>ozera-{selection?.model}</div>
            </div>
          </div>
        )}

        {/* SAE Info - External */}
        {extInfo && isExternal && (
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'rgba(255,255,255,0.1)' }}>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_input</div>
                <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{extInfo.d_input || '-'}</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>d_hidden</div>
                <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{extInfo.d_hidden?.toLocaleString() || '-'}</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Activation</div>
                <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#fff' }}>{extInfo.activation_type || '-'}</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Base Model</div>
                <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'rgba(255,255,255,0.8)' }}>{extInfo.base_model || '-'}</div>
              </div>
            </div>
          </div>
        )}

        {/* External SAEs list */}
        {externalSAEs.length > 0 && (
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ padding: '0.5rem 0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                <ExternalLink className="w-3 h-3" style={{ color: 'rgba(255,255,255,0.5)' }} />
                <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  External SAEs ({externalSAEs.length})
                </span>
              </div>
              {isExternal && (
                <button
                  onClick={() => handleClearExternal(which)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.25rem',
                    padding: '0.125rem 0.5rem', fontSize: '0.65rem',
                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)',
                    color: 'rgba(255,255,255,0.5)', cursor: 'pointer', transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)'
                    e.currentTarget.style.color = 'rgba(255,255,255,0.8)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)'
                    e.currentTarget.style.color = 'rgba(255,255,255,0.5)'
                  }}
                >
                  <X className="w-2.5 h-2.5" />
                  Clear
                </button>
              )}
            </div>
            <div style={{ padding: '0 0.5rem 0.5rem', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {externalSAEs.map(ext => {
                const isSelected = selection?.externalId === ext.id
                return (
                  <button
                    key={ext.id}
                    onClick={() => handleExternalSelect(which, ext)}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      gap: '0.5rem', padding: '0.375rem 0.625rem', width: '100%', textAlign: 'left',
                      background: isSelected ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${isSelected ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.08)'}`,
                      cursor: 'pointer', transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.background = 'rgba(255,255,255,0.06)'
                        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
                        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'
                      }
                    }}
                    title={`${ext.source_id || ext.id} - ${ext.base_model || 'unknown model'}`}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.125rem', minWidth: 0, flex: 1 }}>
                      <span style={{ color: isSelected ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.7)', fontSize: '0.75rem', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {ext.display_name || ext.id}
                      </span>
                      <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.65rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {ext.base_model || 'unknown model'} {ext.hookpoint ? `· ${ext.hookpoint}` : ''}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.65rem', color: 'rgba(255,255,255,0.3)', flexShrink: 0 }}>
                      {ext.d_hidden && <span>{ext.d_hidden.toLocaleString()}f</span>}
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

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Dual SAE Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {renderSaePanel('a', selectionA, saeInfoA, extInfoA)}
        {renderSaePanel('b', selectionB, saeInfoB, extInfoB)}
      </div>

      {/* Compare mode toggle + text input + run button */}
      <div className="bg-black/40 border border-gray-800 p-4">
        {/* Mode toggle */}
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={() => setCompareMode('features')}
            style={{
              background: compareMode === 'features' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.3)',
              border: `1px solid ${compareMode === 'features' ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
              color: compareMode === 'features' ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.4)',
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors"
            onMouseEnter={(e) => {
              if (compareMode !== 'features') {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'
              }
            }}
            onMouseLeave={(e) => {
              if (compareMode !== 'features') {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'
              }
            }}
          >
            <Zap className="w-3.5 h-3.5" />
            Feature Alignment
          </button>
          <button
            onClick={() => !hasExternalSelection && setCompareMode('layers')}
            disabled={hasExternalSelection}
            title={hasExternalSelection ? 'Layer comparison requires built-in Ozera SAEs on both sides' : undefined}
            style={{
              background: compareMode === 'layers' && !hasExternalSelection ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.3)',
              border: `1px solid ${compareMode === 'layers' && !hasExternalSelection ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
              color: hasExternalSelection ? 'rgba(255,255,255,0.2)' : compareMode === 'layers' ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.4)',
              cursor: hasExternalSelection ? 'not-allowed' : 'pointer',
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors"
            onMouseEnter={(e) => {
              if (compareMode !== 'layers' && !hasExternalSelection) {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'
              }
            }}
            onMouseLeave={(e) => {
              if (compareMode !== 'layers' && !hasExternalSelection) {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'
              }
            }}
          >
            <Grid3x3 className="w-3.5 h-3.5" />
            Layer Similarity
          </button>
        </div>

        <p className="text-xs text-gray-500 mb-3">
          {compareMode === 'features'
            ? 'Run shared text through both SAEs and align features by activation overlap or decoder similarity.'
            : 'Compute CKA similarity across all layers. Uses model/activation type from each SAE selection.'}
        </p>

        {/* Text input + run */}
        <div className="flex gap-3">
          <div className="flex-1">
            <textarea
              value={compareText}
              onChange={(e) => setCompareText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter shared text for comparison..."
              className="w-full h-full bg-black/50 border border-gray-700 text-white px-3 py-2 text-sm resize-none"
              style={{ outline: 'none' }}
              onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
              onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
              rows={2}
              disabled={comparing}
            />
          </div>
          <button
            onClick={runComparison}
            disabled={comparing || !selectionA || !selectionB || !compareText.trim()}
            style={{
              background: (comparing || !selectionA || !selectionB || !compareText.trim()) ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.1)',
              border: `1px solid ${(comparing || !selectionA || !selectionB || !compareText.trim()) ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.25)'}`,
              color: (comparing || !selectionA || !selectionB || !compareText.trim()) ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.95)',
              cursor: (comparing || !selectionA || !selectionB || !compareText.trim()) ? 'not-allowed' : 'pointer',
            }}
            className="px-5 py-2 font-medium text-sm transition-colors flex items-center gap-2 self-stretch"
            onMouseEnter={(e) => {
              if (!comparing && selectionA && selectionB && compareText.trim()) {
                e.currentTarget.style.background = 'rgba(255,255,255,0.15)'
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.4)'
              }
            }}
            onMouseLeave={(e) => {
              if (!comparing && selectionA && selectionB && compareText.trim()) {
                e.currentTarget.style.background = 'rgba(255,255,255,0.1)'
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'
              }
            }}
          >
            {comparing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Comparing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Compare
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-800/50 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Loading state */}
      {comparing && (
        <div className="flex flex-col items-center justify-center p-12 bg-black/20 border border-dashed border-gray-800">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'rgba(255,255,255,0.4)' }} />
          <p className="mt-4 text-gray-400">
            {compareMode === 'features'
              ? 'Running SAE comparison on GPU...'
              : 'Computing CKA across all layers...'}
          </p>
        </div>
      )}

      {/* Feature comparison results */}
      {!comparing && featureResult && compareMode === 'features' && (
        <div className="space-y-4">
          {/* CKA Score banner */}
          <div className="bg-black/40 border border-gray-800 p-4">
            <div className="grid grid-cols-5 gap-4 text-center">
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  CKA Score
                </div>
                <div className="text-2xl font-mono" style={{ color: 'rgba(255,255,255,0.95)' }}>
                  {featureResult.cka_score.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Overall Similarity
                </div>
                <div className="text-2xl font-mono text-white">
                  {(featureResult.overall_similarity * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Matched Features
                </div>
                <div className="text-2xl font-mono text-green-400">
                  {featureResult.matched_features}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Unique to A
                </div>
                <div className="text-2xl font-mono" style={{ color: 'rgba(255,255,255,0.7)' }}>
                  {featureResult.unmatched_a}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Unique to B
                </div>
                <div className="text-2xl font-mono" style={{ color: 'rgba(255,255,255,0.7)' }}>
                  {featureResult.unmatched_b}
                </div>
              </div>
            </div>
          </div>

          {/* Feature alignment visualization */}
          <FeatureComparison
            saeA={{
              id: saeAName,
              name: saeAName,
              num_features: featureResult.sae_a.d_hidden,
              layer: featureResult.sae_a.layer,
              model: featureResult.sae_a.model,
            }}
            saeB={{
              id: saeBName,
              name: saeBName,
              num_features: featureResult.sae_b.d_hidden,
              layer: featureResult.sae_b.layer,
              model: featureResult.sae_b.model,
            }}
            comparison={{
              overall_similarity: featureResult.overall_similarity,
              matched_features: featureResult.matched_features,
              unmatched_a: featureResult.unmatched_a,
              unmatched_b: featureResult.unmatched_b,
              top_matches: featureResult.top_matches,
              divergent_features_a: featureResult.divergent_features_a,
              divergent_features_b: featureResult.divergent_features_b,
            }}
            onFeatureSelect={onFeatureSelect}
          />

          {/* Similarity matrix sample */}
          {featureResult.similarity_matrix_sample &&
            featureResult.feature_indices_a &&
            featureResult.feature_indices_b && (
              <SimilarityMatrix
                ckaMatrix={featureResult.similarity_matrix_sample}
                layersA={featureResult.feature_indices_a.slice(0, 30)}
                layersB={featureResult.feature_indices_b.slice(0, 30)}
                modelA={saeAName}
                modelB={saeBName}
                activationTypeA={featureResult.sae_a.activation_type}
                activationTypeB={featureResult.sae_b.activation_type}
                title="Feature Similarity Matrix"
                subtitle="Pairwise similarity between top features of each SAE. Brighter = more similar."
                labelPrefix="F"
              />
            )}

          {/* Research insights */}
          <div className="bg-black/40 border border-gray-800 p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              Research Insights
            </h3>
            <div className="space-y-2 text-xs text-gray-400">
              {featureResult.cka_score > 0.7 && (
                <p>
                  High CKA ({featureResult.cka_score.toFixed(3)}) indicates
                  these SAEs learn very similar representations, suggesting the
                  underlying model features are well-captured at both locations.
                </p>
              )}
              {featureResult.cka_score < 0.3 && (
                <p>
                  Low CKA ({featureResult.cka_score.toFixed(3)}) suggests
                  substantially different representations. These SAEs likely
                  capture different aspects of the model's computation.
                </p>
              )}
              {featureResult.matched_features > 0 && (
                <p>
                  {featureResult.matched_features} features matched between the
                  two SAEs. The best match has{' '}
                  {(featureResult.top_matches[0]?.similarity * 100).toFixed(1)}%
                  similarity.
                </p>
              )}
              {hasExternalSelection && (
                <p>
                  Cross-architecture comparison uses activation-based correlation
                  to align features across different model families and training procedures.
                </p>
              )}
              {!hasExternalSelection && selectionA?.model !== selectionB?.model && (
                <p>
                  Cross-model comparison uses activation-based correlation since
                  the models have different activation space dimensions.
                </p>
              )}
              {!hasExternalSelection && selectionA?.model === selectionB?.model &&
                selectionA?.layer !== selectionB?.layer && (
                  <p>
                    Same-model comparison across layers reveals how features
                    transform through the network depth.
                  </p>
                )}
            </div>
          </div>
        </div>
      )}

      {/* Layer similarity results */}
      {!comparing && layerResult && compareMode === 'layers' && (
        <div className="space-y-4">
          {/* Summary stats */}
          <div className="bg-black/40 border border-gray-800 p-4">
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {layerResult.model_a} Layers
                </div>
                <div className="text-xl font-mono" style={{ color: 'rgba(255,255,255,0.7)' }}>
                  {layerResult.layers_a.length}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {layerResult.model_b} Layers
                </div>
                <div className="text-xl font-mono" style={{ color: 'rgba(255,255,255,0.7)' }}>
                  {layerResult.layers_b.length}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Mean CKA
                </div>
                <div className="text-xl font-mono text-white">
                  {(
                    layerResult.cka_matrix.flat().reduce((a, b) => a + b, 0) /
                    layerResult.cka_matrix.flat().length
                  ).toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Tokens
                </div>
                <div className="text-xl font-mono text-gray-300">
                  {layerResult.num_tokens}
                </div>
              </div>
            </div>
          </div>

          {/* CKA Matrix */}
          <SimilarityMatrix
            ckaMatrix={layerResult.cka_matrix}
            layersA={layerResult.layers_a}
            layersB={layerResult.layers_b}
            modelA={layerResult.model_a}
            modelB={layerResult.model_b}
            activationTypeA={layerResult.activation_type_a}
            activationTypeB={layerResult.activation_type_b}
          />

          {/* Layer-wise insights */}
          <div className="bg-black/40 border border-gray-800 p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              Research Insights
            </h3>
            <div className="space-y-2 text-xs text-gray-400">
              {layerResult.model_a === layerResult.model_b && (
                <p>
                  Self-comparison: the diagonal shows how each layer's SAE features
                  compare to themselves (should be ~1.0). Off-diagonal values
                  reveal which layers develop similar or different representations.
                </p>
              )}
              {layerResult.model_a !== layerResult.model_b && (
                <p>
                  Cross-model comparison: reveals which layers in {layerResult.model_a}{' '}
                  correspond to which layers in {layerResult.model_b}. A strong
                  diagonal suggests layer-aligned representations despite
                  different model sizes.
                </p>
              )}
              {(() => {
                // Find highest off-diagonal CKA
                let maxOffDiag = 0
                let maxI = 0
                let maxJ = 0
                layerResult.cka_matrix.forEach((row, i) => {
                  row.forEach((val, j) => {
                    if (i !== j && val > maxOffDiag) {
                      maxOffDiag = val
                      maxI = i
                      maxJ = j
                    }
                  })
                })
                if (maxOffDiag > 0.5) {
                  return (
                    <p>
                      Strongest cross-layer similarity:{' '}
                      <span style={{ color: 'rgba(255,255,255,0.8)' }}>
                        L{layerResult.layers_a[maxI]}
                      </span>{' '}
                      and{' '}
                      <span style={{ color: 'rgba(255,255,255,0.8)' }}>
                        L{layerResult.layers_b[maxJ]}
                      </span>{' '}
                      (CKA = {maxOffDiag.toFixed(3)}), suggesting these layers
                      develop similar feature representations.
                    </p>
                  )
                }
                return null
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!comparing && !featureResult && !layerResult && !error && (
        <div className="flex flex-col items-center justify-center p-12 bg-black/20 border border-dashed border-gray-800 text-center">
          <GitCompare className="w-12 h-12 text-gray-700 mb-4" />
          <h3 className="text-gray-500 font-medium mb-1">
            Select SAEs and Run Comparison
          </h3>
          <p className="text-gray-600 text-sm max-w-md">
            Choose two SAEs to compare, enter shared text, and click Compare to
            analyze feature alignment and representation similarity.
          </p>
        </div>
      )}
    </div>
  )
}
