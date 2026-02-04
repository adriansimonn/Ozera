/**
 * Model Comparison Dashboard - Research feature for comparing SAE representations.
 *
 * Enables researchers to:
 * - Compare features between two SAEs (same or different models/layers)
 * - View layer-by-layer CKA similarity matrices
 * - Identify shared vs unique features between models
 * - Analyze how representations emerge across layers
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
} from 'lucide-react'
import {
  saeClient,
  type SAEListResponse,
  type SAECompareResponse,
  type SAECompareLayersResponse,
} from '../../api/client'
import { FeatureComparison } from './FeatureComparison'
import { SimilarityMatrix } from './SimilarityMatrix'

interface SAECompareSelection {
  model: 'nano' | 'mini'
  layer: number
  activationType: 'residual' | 'mlp_output'
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
      })
    } else {
      setter({
        ...current,
        activationType: value as 'residual' | 'mlp_output',
      })
    }
  }

  const saeAName = selectionA
    ? `${selectionA.model}-L${selectionA.layer}-${selectionA.activationType}`
    : ''
  const saeBName = selectionB
    ? `${selectionB.model}-L${selectionB.layer}-${selectionB.activationType}`
    : ''

  if (loadingList) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
        <span className="ml-3 text-gray-400">Loading SAEs...</span>
      </div>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Dual SAE Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* SAE A */}
        <div className="bg-black/40 border border-purple-800/40">
          <div className="p-3 border-b border-gray-800 flex items-center gap-2">
            <Database className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-semibold text-purple-400 uppercase tracking-wide">
              SAE A
            </span>
          </div>
          <div className="grid grid-cols-3 gap-px bg-gray-800">
            <div className="bg-black/40 p-3">
              <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
                Model
              </label>
              <div className="relative">
                <select
                  value={selectionA?.model || ''}
                  onChange={(e) =>
                    handleSelectionChange('a', 'model', e.target.value)
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
                >
                  {saeList &&
                    Object.keys(saeList.models).map((model) => (
                      <option key={model} value={model}>
                        {model}
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
                  value={selectionA?.layer ?? ''}
                  onChange={(e) =>
                    handleSelectionChange('a', 'layer', parseInt(e.target.value))
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
                >
                  {selectionA &&
                    getAvailableLayers(selectionA.model).map((layer) => (
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
                  value={selectionA?.activationType || ''}
                  onChange={(e) =>
                    handleSelectionChange('a', 'activationType', e.target.value)
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-purple-500 cursor-pointer"
                >
                  {selectionA &&
                    getAvailableTypes(selectionA.model, selectionA.layer).map(
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
        </div>

        {/* SAE B */}
        <div className="bg-black/40 border border-cyan-800/40">
          <div className="p-3 border-b border-gray-800 flex items-center gap-2">
            <Database className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-cyan-400 uppercase tracking-wide">
              SAE B
            </span>
          </div>
          <div className="grid grid-cols-3 gap-px bg-gray-800">
            <div className="bg-black/40 p-3">
              <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
                Model
              </label>
              <div className="relative">
                <select
                  value={selectionB?.model || ''}
                  onChange={(e) =>
                    handleSelectionChange('b', 'model', e.target.value)
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  {saeList &&
                    Object.keys(saeList.models).map((model) => (
                      <option key={model} value={model}>
                        {model}
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
                  value={selectionB?.layer ?? ''}
                  onChange={(e) =>
                    handleSelectionChange('b', 'layer', parseInt(e.target.value))
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  {selectionB &&
                    getAvailableLayers(selectionB.model).map((layer) => (
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
                  value={selectionB?.activationType || ''}
                  onChange={(e) =>
                    handleSelectionChange('b', 'activationType', e.target.value)
                  }
                  className="w-full appearance-none bg-black/50 border border-gray-700 text-white px-2 py-1.5 pr-7 text-sm focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  {selectionB &&
                    getAvailableTypes(selectionB.model, selectionB.layer).map(
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
        </div>
      </div>

      {/* Compare mode toggle + text input + run button */}
      <div className="bg-black/40 border border-gray-800 p-4">
        {/* Mode toggle */}
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={() => setCompareMode('features')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm border transition-colors ${
              compareMode === 'features'
                ? 'bg-purple-600/20 border-purple-500/50 text-purple-300'
                : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Feature Alignment
          </button>
          <button
            onClick={() => setCompareMode('layers')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm border transition-colors ${
              compareMode === 'layers'
                ? 'bg-purple-600/20 border-purple-500/50 text-purple-300'
                : 'bg-black/30 border-gray-700 text-gray-400 hover:border-gray-600'
            }`}
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
              className="w-full bg-black/50 border border-gray-700 text-white px-3 py-2 text-sm focus:outline-none focus:border-purple-500 resize-none"
              rows={2}
              disabled={comparing}
            />
          </div>
          <button
            onClick={runComparison}
            disabled={comparing || !selectionA || !selectionB || !compareText.trim()}
            className="px-5 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors flex items-center gap-2 self-start"
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
          <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
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
                <div className="text-2xl font-mono text-purple-400">
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
                <div className="text-2xl font-mono text-purple-400">
                  {featureResult.unmatched_a}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  Unique to B
                </div>
                <div className="text-2xl font-mono text-cyan-400">
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
              layer: selectionA!.layer,
              model: selectionA!.model,
            }}
            saeB={{
              id: saeBName,
              name: saeBName,
              num_features: featureResult.sae_b.d_hidden,
              layer: selectionB!.layer,
              model: selectionB!.model,
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
              <div className="bg-black/40 border border-gray-800">
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center gap-2">
                    <Grid3x3 className="w-4 h-4 text-purple-400" />
                    <h3 className="text-sm font-semibold text-gray-200">
                      Feature Similarity Matrix (Top Features)
                    </h3>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Pairwise similarity between top features of each SAE.
                    Brighter = more similar.
                  </p>
                </div>
                <div className="p-4">
                  <SimilarityMatrix
                    ckaMatrix={featureResult.similarity_matrix_sample}
                    layersA={featureResult.feature_indices_a.slice(0, 30)}
                    layersB={featureResult.feature_indices_b.slice(0, 30)}
                    modelA={`${selectionA!.model} L${selectionA!.layer}`}
                    modelB={`${selectionB!.model} L${selectionB!.layer}`}
                    activationTypeA={selectionA!.activationType}
                    activationTypeB={selectionB!.activationType}
                  />
                </div>
              </div>
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
              {selectionA?.model !== selectionB?.model && (
                <p>
                  Cross-model comparison uses activation-based correlation since
                  the models have different activation space dimensions.
                </p>
              )}
              {selectionA?.model === selectionB?.model &&
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
                <div className="text-xl font-mono text-purple-400">
                  {layerResult.layers_a.length}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {layerResult.model_b} Layers
                </div>
                <div className="text-xl font-mono text-cyan-400">
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
                      <span className="text-purple-300">
                        L{layerResult.layers_a[maxI]}
                      </span>{' '}
                      and{' '}
                      <span className="text-cyan-300">
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
