/**
 * SAE Selector component for choosing model, layer, and activation type.
 * Fetches available SAEs from the API and provides dropdown selection.
 */

import { useState, useEffect, useCallback } from 'react'
import { ChevronDown, Loader2, AlertCircle, RefreshCw, Database, ExternalLink } from 'lucide-react'
import { saeClient, type SAEListResponse } from '../../api/client'

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
}

export function SAESelector({
  selection,
  onSelectionChange,
  className = '',
}: SAESelectorProps) {
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
  }, []) // Only fetch on mount

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

    // Reset to first available layer and type for new model
    const firstSae = modelInfo.saes[0]
    onSelectionChange({
      model,
      layer: firstSae.layer,
      activationType: firstSae.activation_type,
    })
  }

  const handleLayerChange = (layer: number) => {
    if (!selection || !saeList) return

    const modelInfo = saeList.models[selection.model]
    if (!modelInfo) return

    // Check if current activation type is available for new layer
    const availableTypes = modelInfo.saes
      .filter(sae => sae.layer === layer)
      .map(sae => sae.activation_type)

    const newType = availableTypes.includes(selection.activationType)
      ? selection.activationType
      : availableTypes[0]

    onSelectionChange({
      ...selection,
      layer,
      activationType: newType,
    })
  }

  const handleActivationTypeChange = (activationType: 'residual' | 'mlp_output') => {
    if (!selection) return
    onSelectionChange({
      ...selection,
      activationType,
    })
  }

  const currentSaeInfo = getCurrentSAEInfo()
  const availableLayers = getAvailableLayers()
  const availableTypes = getAvailableActivationTypes()

  if (loading) {
    return (
      <div className={`bg-black/40 border border-gray-800 p-4 ${className}`}>
        <div className="flex items-center gap-3 text-gray-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Loading available SAEs...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`bg-black/40 border border-red-800/50 p-4 ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchSAEs}
            className="px-3 py-1 text-sm border border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  if (!saeList || saeList.total_saes === 0) {
    return (
      <div className={`bg-black/40 border border-gray-800 p-4 ${className}`}>
        <div className="flex items-center gap-3 text-gray-500">
          <Database className="w-5 h-5" />
          <span>No SAEs available. Deploy SAEs to Modal first.</span>
        </div>
      </div>
    )
  }

  return (
    <div className={`bg-black/40 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wide">
              SAE Selection
            </h3>
          </div>
          <div className="text-xs text-gray-500">
            {saeList.total_saes} SAEs available
          </div>
        </div>
      </div>

      {/* Selectors */}
      <div className="grid grid-cols-3 gap-px bg-gray-800">
        {/* Model Selector */}
        <div className="bg-black/40 p-4">
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-2">
            Model
          </label>
          <div className="relative">
            <select
              value={selection?.model || ''}
              onChange={(e) => handleModelChange(e.target.value as 'nano' | 'mini')}
              className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-3 py-2 pr-8 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
            >
              {Object.keys(saeList.models).map(model => (
                <option key={model} value={model}>
                  ozera-{model}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
          </div>
        </div>

        {/* Layer Selector */}
        <div className="bg-black/40 p-4">
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-2">
            Layer
          </label>
          <div className="relative">
            <select
              value={selection?.layer ?? ''}
              onChange={(e) => handleLayerChange(parseInt(e.target.value))}
              className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-3 py-2 pr-8 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
            >
              {availableLayers.map(layer => (
                <option key={layer} value={layer}>
                  Layer {layer}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
          </div>
        </div>

        {/* Activation Type Selector */}
        <div className="bg-black/40 p-4">
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-2">
            Activation Type
          </label>
          <div className="relative">
            <select
              value={selection?.activationType || ''}
              onChange={(e) => handleActivationTypeChange(e.target.value as 'residual' | 'mlp_output')}
              className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-3 py-2 pr-8 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
            >
              {availableTypes.map(type => (
                <option key={type} value={type}>
                  {type === 'residual' ? 'Residual Stream' : 'MLP Output'}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* SAE Info */}
      {currentSaeInfo && (
        <div className="grid grid-cols-4 gap-px bg-gray-800 border-t border-gray-800">
          <div className="bg-black/40 p-3 text-center">
            <div className="text-xs text-gray-500 uppercase tracking-wide">d_input</div>
            <div className="text-sm font-mono text-white">{currentSaeInfo.d_input || '-'}</div>
          </div>
          <div className="bg-black/40 p-3 text-center">
            <div className="text-xs text-gray-500 uppercase tracking-wide">d_hidden</div>
            <div className="text-sm font-mono text-purple-300">{currentSaeInfo.d_hidden || '-'}</div>
          </div>
          <div className="bg-black/40 p-3 text-center">
            <div className="text-xs text-gray-500 uppercase tracking-wide">Activation</div>
            <div className="text-sm font-mono text-white">{currentSaeInfo.activation || 'relu'}</div>
          </div>
          {currentSaeInfo.training && (
            <div className="bg-black/40 p-3 text-center">
              <div className="text-xs text-gray-500 uppercase tracking-wide">Final L0</div>
              <div className="text-sm font-mono text-green-400">
                {currentSaeInfo.training.final_l0?.toFixed(1) || '-'}
              </div>
            </div>
          )}
          {!currentSaeInfo.training && (
            <div className="bg-black/40 p-3 text-center">
              <div className="text-xs text-gray-500 uppercase tracking-wide">Params</div>
              <div className="text-sm font-mono text-white">
                {currentSaeInfo.num_parameters
                  ? `${(currentSaeInfo.num_parameters / 1e6).toFixed(1)}M`
                  : '-'}
              </div>
            </div>
          )}
        </div>
      )}

      {/* External SAEs */}
      {saeList.external_saes && saeList.external_saes.length > 0 && (
        <div className="border-t border-gray-800">
          <div className="p-3 flex items-center gap-2">
            <ExternalLink className="w-3 h-3 text-blue-400" />
            <span className="text-xs text-gray-500 uppercase tracking-wide">
              External SAEs ({saeList.external_saes.length})
            </span>
          </div>
          <div className="px-3 pb-3 flex flex-wrap gap-2">
            {saeList.external_saes.map(ext => (
              <div
                key={ext.id}
                className="inline-flex items-center gap-1.5 px-2 py-1 bg-blue-500/10 border border-blue-500/20 text-xs"
                title={`${ext.source_id || ext.id} - ${ext.base_model || 'unknown model'}`}
              >
                <span className="text-blue-300">
                  {ext.display_name || ext.id}
                </span>
                {ext.d_hidden && (
                  <span className="text-gray-500">
                    {ext.d_hidden.toLocaleString()} features
                  </span>
                )}
                {ext.activation_type && (
                  <span className="text-gray-500">
                    {ext.activation_type}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
