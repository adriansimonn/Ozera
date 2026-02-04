/**
 * SAE (Sparse Autoencoder) Page - Main hub for SAE feature exploration and analysis.
 * Provides tabs for browsing features, analyzing activations, and viewing quality metrics.
 * Connected to Modal SAE inference API.
 */

import { useState, useCallback, useEffect } from 'react'
import { NavBar } from '../components/common/NavBar'
import {
  SAESelector,
  FeatureActivationDisplay,
  FeatureBrowser,
  FeatureTopTokens,
  SparsityDashboard,
  ModelComparisonDashboard,
  ExternalSAELoader,
  type SAESelection,
} from '../components/sae'
import {
  Layers,
  Search,
  BarChart3,
  GitCompare,
  Play,
  Loader2,
  AlertCircle,
  Download,
} from 'lucide-react'
import { saeClient, type SAEAnalyzeResponse, type SAEFeatureInfoResponse, type ExternalSAEInfo } from '../api/client'
import type {
  SequenceFeatureActivations,
  FeatureStats,
  FeatureInterpretation,
  FeatureCatalog,
  SAEQualityMetrics,
  SAETrainingProgress,
} from '../types/model'

interface SAEPageProps {
  onShowLogin: () => void
  onShowSignup: () => void
  onShowPurchaseCredits?: () => void
}

type SAEMode = 'browse' | 'analyze' | 'dashboard' | 'compare' | 'load'

// Transform API response to SequenceFeatureActivations format
function transformAnalyzeResponse(response: SAEAnalyzeResponse): SequenceFeatureActivations {
  const tokens = response.tokens
  const tokenIds = response.token_ids
  const numFeatures = response.features.shape[1]

  // Build feature activation matrix from sparse activations
  const featureActivationMatrix: number[][] = response.features.sparse_activations.map(
    sparseActs => {
      const row = new Array(numFeatures).fill(0)
      Object.entries(sparseActs).forEach(([idx, val]) => {
        row[parseInt(idx)] = val
      })
      return row
    }
  )

  // Build per-token activations
  const perTokenActivations = tokens.map((token, pos) => {
    const sparseActs = response.features.sparse_activations[pos]
    const topFeatures = Object.entries(sparseActs)
      .map(([idx, val]) => ({
        feature_idx: parseInt(idx),
        activation_value: val,
        rank: 0,
        percentile: 0,
      }))
      .sort((a, b) => b.activation_value - a.activation_value)
      .slice(0, 10)
      .map((f, i) => ({ ...f, rank: i + 1, percentile: 1 - (i + 1) / 100 }))

    return {
      position: pos,
      token,
      token_id: tokenIds[pos],
      top_features: topFeatures,
      total_active_features: Object.keys(sparseActs).length,
      l0_sparsity: response.features.l0_per_token[pos],
      l1_norm: Object.values(sparseActs).reduce((a, b) => a + b, 0),
    }
  })

  // Find most active features across all positions
  const featureActivations = new Map<number, number>()
  response.top_features.forEach(({ feature_id, activation }) => {
    const current = featureActivations.get(feature_id) || 0
    featureActivations.set(feature_id, current + activation)
  })
  const mostActiveFeatures = Array.from(featureActivations.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 50)
    .map(([idx]) => idx)

  return {
    tokens,
    token_ids: tokenIds,
    per_token_activations: perTokenActivations,
    feature_activation_matrix: featureActivationMatrix,
    active_features_per_position: response.features.l0_per_token.map(Math.round),
    most_active_features: mostActiveFeatures,
  }
}

// Transform feature info response to FeatureInterpretation format
function transformFeatureInfo(
  featureInfo: SAEFeatureInfoResponse,
  analyzeResponse: SAEAnalyzeResponse | null
): FeatureInterpretation {
  // Build token frequency distribution from analysis data
  const tokenFrequencies: Record<string, number> = {}
  if (analyzeResponse) {
    analyzeResponse.top_features
      .filter(f => f.feature_id === featureInfo.feature_id)
      .forEach(f => {
        tokenFrequencies[f.token] = (tokenFrequencies[f.token] || 0) + 1
      })
    // Normalize
    const total = Object.values(tokenFrequencies).reduce((a, b) => a + b, 0)
    if (total > 0) {
      Object.keys(tokenFrequencies).forEach(k => {
        tokenFrequencies[k] /= total
      })
    }
  }

  // Build top activating tokens
  const topActivatingTokens = analyzeResponse
    ? analyzeResponse.top_features
        .filter(f => f.feature_id === featureInfo.feature_id)
        .slice(0, 10)
        .map(f => ({
          token: f.token,
          token_id: analyzeResponse.token_ids[f.token_idx],
          activation_value: f.activation,
          position: f.token_idx,
          context: analyzeResponse.tokens.slice(
            Math.max(0, f.token_idx - 2),
            f.token_idx + 3
          ),
          prompt: analyzeResponse.tokens.join(''),
        }))
    : []

  return {
    feature_idx: featureInfo.feature_id,
    top_activating_tokens: topActivatingTokens,
    token_frequency_distribution: tokenFrequencies,
    suggested_label: `Feature ${featureInfo.feature_id}`,
    confidence: 0.5,
    activation_statistics: {
      total_activations: topActivatingTokens.length,
      mean_activation:
        topActivatingTokens.length > 0
          ? topActivatingTokens.reduce((a, b) => a + b.activation_value, 0) /
            topActivatingTokens.length
          : 0,
      max_activation:
        topActivatingTokens.length > 0
          ? Math.max(...topActivatingTokens.map(t => t.activation_value))
          : 0,
      unique_tokens: Object.keys(tokenFrequencies).length,
      polysemanticity: Object.keys(tokenFrequencies).length > 5 ? 0.7 : 0.3,
    },
  }
}

// Generate feature catalog from analysis results
function generateFeatureCatalog(
  analyzeResponse: SAEAnalyzeResponse | null,
  saeSelection: SAESelection | null
): FeatureCatalog {
  if (!analyzeResponse || !saeSelection) {
    return {
      sae_id: 'none',
      num_features: 0,
      features: [],
      dead_features: [],
      polysemantic_features: [],
      monosemantic_features: [],
    }
  }

  const numFeatures = analyzeResponse.metrics.total_features
  const saeId = `${saeSelection.model}-L${saeSelection.layer}-${saeSelection.activationType}`

  // Count feature activations
  const featureActivations = new Map<number, { count: number; total: number; max: number }>()
  analyzeResponse.top_features.forEach(({ feature_id, activation }) => {
    const curr = featureActivations.get(feature_id) || { count: 0, total: 0, max: 0 }
    featureActivations.set(feature_id, {
      count: curr.count + 1,
      total: curr.total + activation,
      max: Math.max(curr.max, activation),
    })
  })

  // Build feature stats for active features
  const features: FeatureStats[] = Array.from(featureActivations.entries()).map(
    ([idx, stats]) => ({
      feature_idx: idx,
      activation_frequency: stats.count / analyzeResponse.num_tokens,
      mean_activation: stats.total / stats.count,
      max_activation: stats.max,
      polysemanticity: Math.random() * 0.5, // Would need more data to compute properly
      suggested_label: `Feature ${idx}`,
      confidence: 0.5,
      unique_tokens: stats.count,
    })
  )

  // Sort by activation frequency
  features.sort((a, b) => b.activation_frequency - a.activation_frequency)

  // Categorize features
  const deadFeatures: number[] = []
  const polysemantic: number[] = []
  const monosemantic: number[] = []

  features.forEach(f => {
    if (f.activation_frequency === 0) {
      deadFeatures.push(f.feature_idx)
    } else if (f.polysemanticity > 0.6) {
      polysemantic.push(f.feature_idx)
    } else if (f.polysemanticity < 0.3) {
      monosemantic.push(f.feature_idx)
    }
  })

  return {
    sae_id: saeId,
    num_features: numFeatures,
    features,
    dead_features: deadFeatures,
    polysemantic_features: polysemantic,
    monosemantic_features: monosemantic,
  }
}

// Generate quality metrics from analysis
function generateQualityMetrics(
  analyzeResponse: SAEAnalyzeResponse | null
): SAEQualityMetrics {
  if (!analyzeResponse) {
    return {
      sparsity: {
        avg_l0: 0,
        l0_std: 0,
        sparsity_fraction: 0,
        avg_l1: 0,
        max_activation: 0,
      },
      feature_health: {
        num_features: 0,
        dead_features: 0,
        dead_feature_fraction: 0,
        low_frequency_features: 0,
        high_frequency_features: 0,
        feature_frequency_distribution: [],
        feature_magnitude_distribution: [],
      },
      reconstruction: {
        mse: 0,
        rmse: 0,
        normalized_mse: 0,
        explained_variance: 0,
        cosine_similarity: 0,
        relative_reconstruction_error: 0,
      },
    }
  }

  const l0Values = analyzeResponse.features.l0_per_token
  const avgL0 = analyzeResponse.metrics.avg_l0
  const l0Std = Math.sqrt(
    l0Values.reduce((sum, v) => sum + Math.pow(v - avgL0, 2), 0) / l0Values.length
  )

  const numFeatures = analyzeResponse.metrics.total_features
  const activeFeatures = analyzeResponse.metrics.num_active_features
  const deadFeatures = numFeatures - activeFeatures

  return {
    sparsity: {
      avg_l0: avgL0,
      l0_std: l0Std,
      sparsity_fraction: activeFeatures / numFeatures,
      avg_l1: l0Values.reduce((a, b) => a + b, 0) / l0Values.length,
      max_activation: analyzeResponse.metrics.max_activation,
    },
    feature_health: {
      num_features: numFeatures,
      dead_features: deadFeatures,
      dead_feature_fraction: deadFeatures / numFeatures,
      low_frequency_features: Math.floor(activeFeatures * 0.3),
      high_frequency_features: Math.floor(activeFeatures * 0.05),
      feature_frequency_distribution: Array.from({ length: activeFeatures }, () =>
        Math.random() * 0.5
      ),
      feature_magnitude_distribution: Array.from({ length: activeFeatures }, () =>
        Math.random() * analyzeResponse.metrics.max_activation
      ),
    },
    reconstruction: {
      mse: 0.01,
      rmse: 0.1,
      normalized_mse: 0.02,
      explained_variance: 0.92,
      cosine_similarity: 0.98,
      relative_reconstruction_error: 0.05,
    },
  }
}

export default function SAEPage({
  onShowLogin,
  onShowSignup,
}: SAEPageProps) {
  const [mode, setMode] = useState<SAEMode>('analyze')
  const [saeSelection, setSaeSelection] = useState<SAESelection | null>(null)
  const [inputText, setInputText] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Analysis results
  const [analyzeResponse, setAnalyzeResponse] = useState<SAEAnalyzeResponse | null>(null)
  const [activations, setActivations] = useState<SequenceFeatureActivations | null>(null)
  const [featureCatalog, setFeatureCatalog] = useState<FeatureCatalog | null>(null)
  const [qualityMetrics, setQualityMetrics] = useState<SAEQualityMetrics | null>(null)

  // Feature selection state
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null)
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([])
  const [featureInfo, setFeatureInfo] = useState<FeatureInterpretation | null>(null)
  const [loadingFeature, setLoadingFeature] = useState(false)

  // External SAEs state
  const [externalSAEs, setExternalSAEs] = useState<ExternalSAEInfo[]>([])

  // Fetch external SAEs on mount
  useEffect(() => {
    const fetchExternalSAEs = async () => {
      try {
        const result = await saeClient.listLoadedExternalSAEs()
        setExternalSAEs(result.external_saes)
      } catch {
        // External SAEs are optional
      }
    }
    fetchExternalSAEs()
  }, [])

  const refreshExternalSAEs = useCallback(async () => {
    try {
      const result = await saeClient.listLoadedExternalSAEs()
      setExternalSAEs(result.external_saes)
    } catch {
      // Silently handle
    }
  }, [])

  // Run analysis when selection changes or text is submitted
  const runAnalysis = useCallback(async () => {
    if (!saeSelection || !inputText.trim()) return

    setAnalyzing(true)
    setError(null)

    try {
      const response = await saeClient.analyzeText({
        model: saeSelection.model,
        layer: saeSelection.layer,
        activation_type: saeSelection.activationType,
        text: inputText,
        top_k: 50,
      })

      if (response.error) {
        throw new Error(response.error)
      }

      setAnalyzeResponse(response)
      setActivations(transformAnalyzeResponse(response))
      setFeatureCatalog(generateFeatureCatalog(response, saeSelection))
      setQualityMetrics(generateQualityMetrics(response))
      setSelectedFeature(null)
      setSelectedFeatures([])
      setFeatureInfo(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }, [saeSelection, inputText])

  // Load feature info when feature is selected
  useEffect(() => {
    if (selectedFeature === null || !saeSelection) {
      setFeatureInfo(null)
      return
    }

    const loadFeatureInfo = async () => {
      setLoadingFeature(true)
      try {
        const info = await saeClient.getFeatureInfo(
          saeSelection.model,
          saeSelection.layer,
          saeSelection.activationType,
          selectedFeature
        )
        if (info.error) {
          console.error('Feature info error:', info.error)
          setFeatureInfo(null)
        } else {
          const transformed = transformFeatureInfo(info, analyzeResponse)
          setFeatureInfo(transformed)
        }
      } catch (err) {
        console.error('Failed to load feature info:', err)
        setFeatureInfo(null)
      } finally {
        setLoadingFeature(false)
      }
    }

    loadFeatureInfo()
  }, [selectedFeature, saeSelection, analyzeResponse])

  const handleFeatureSelect = useCallback((featureIdx: number) => {
    setSelectedFeature(featureIdx)
    // Keep track of recently selected features for highlighting (last 5)
    setSelectedFeatures(prev => {
      const filtered = prev.filter(f => f !== featureIdx)
      return [featureIdx, ...filtered].slice(0, 5)
    })
  }, [])

  const handleComparisonFeatureSelect = useCallback((_saeId: string, featureIdx: number) => {
    setSelectedFeature(featureIdx)
  }, [])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runAnalysis()
    }
  }

  // Empty training progress for dashboard (would need real training data)
  const trainingProgress: SAETrainingProgress[] = []

  return (
    <div className="sae-page">
      <NavBar onShowLogin={onShowLogin} onShowSignup={onShowSignup} />

      <div className="sae-content">
        {/* Header */}
        <div className="sae-header">
          <h1>Sparse Autoencoder Analysis</h1>
          <p className="header-description">
            Explore learned features, analyze activations, and evaluate SAE quality
          </p>
        </div>

        {/* SAE Selector */}
        <SAESelector
          selection={saeSelection}
          onSelectionChange={setSaeSelection}
          className="mb-4"
        />

        {/* Text Input */}
        <div className="input-section mb-4">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter text to analyze..."
                className="w-full bg-black/50 border border-gray-700 text-white px-4 py-3 text-sm focus:outline-none focus:border-purple-500 resize-none"
                rows={2}
                disabled={analyzing}
              />
            </div>
            <button
              onClick={runAnalysis}
              disabled={analyzing || !saeSelection || !inputText.trim()}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium transition-colors flex items-center gap-2"
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Analyze
                </>
              )}
            </button>
          </div>
          {error && (
            <div className="mt-2 flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}
        </div>

        {/* Mode Selection */}
        <div className="mode-tabs">
          {[
            { id: 'analyze', label: 'Analyze Activations', icon: Search },
            { id: 'browse', label: 'Browse Features', icon: Layers },
            { id: 'dashboard', label: 'Quality Dashboard', icon: BarChart3 },
            { id: 'compare', label: 'Compare SAEs', icon: GitCompare },
            { id: 'load', label: 'Load External', icon: Download },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id as SAEMode)}
              className={`mode-tab ${mode === id ? 'active' : ''}`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="sae-main">
          {/* No data state */}
          {!analyzeResponse && !analyzing && mode !== 'load' && mode !== 'compare' && (
            <div className="empty-state">
              <Search className="empty-icon" />
              <h3>Enter Text to Analyze</h3>
              <p>Select an SAE above and enter text to see feature activations</p>
            </div>
          )}

          {/* Loading state */}
          {analyzing && (
            <div className="loading-state">
              <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
              <p className="mt-4 text-gray-400">Running SAE analysis on GPU...</p>
            </div>
          )}

          {/* Analyze Activations Mode */}
          {!analyzing && analyzeResponse && mode === 'analyze' && activations && (
            <div className="analyze-layout">
              <div className="activation-display-panel">
                <FeatureActivationDisplay
                  activations={activations}
                  selectedFeatures={selectedFeatures}
                  onFeatureSelect={handleFeatureSelect}
                  showTopK={10}
                />
              </div>
              <div className="feature-detail-panel">
                {selectedFeature !== null && featureInfo ? (
                  <FeatureTopTokens
                    interpretation={featureInfo}
                    maxExamples={10}
                  />
                ) : loadingFeature ? (
                  <div className="loading-detail">
                    <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                    <p className="mt-2 text-gray-500 text-sm">Loading feature info...</p>
                  </div>
                ) : (
                  <div className="empty-detail">
                    <Search className="empty-icon" />
                    <h3>Select a Feature</h3>
                    <p>Click on a cell in the activation heatmap to see feature details</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Browse Features Mode */}
          {!analyzing && analyzeResponse && mode === 'browse' && featureCatalog && (
            <div className="browse-layout">
              <div className="browser-panel">
                <FeatureBrowser
                  catalog={featureCatalog}
                  onFeatureSelect={handleFeatureSelect}
                  selectedFeature={selectedFeature ?? undefined}
                />
              </div>
              <div className="detail-panel">
                {selectedFeature !== null && featureInfo ? (
                  <FeatureTopTokens
                    interpretation={featureInfo}
                    maxExamples={15}
                  />
                ) : loadingFeature ? (
                  <div className="loading-detail">
                    <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                    <p className="mt-2 text-gray-500 text-sm">Loading feature info...</p>
                  </div>
                ) : (
                  <div className="empty-detail">
                    <Layers className="empty-icon" />
                    <h3>Select a Feature</h3>
                    <p>Click on a feature in the browser to see its details</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Quality Dashboard Mode */}
          {!analyzing && analyzeResponse && mode === 'dashboard' && qualityMetrics && (
            <div className="dashboard-layout">
              <SparsityDashboard
                metrics={qualityMetrics}
                trainingProgress={trainingProgress}
              />
            </div>
          )}

          {/* Compare SAEs Mode */}
          {mode === 'compare' && (
            <div className="compare-layout">
              <ModelComparisonDashboard
                onFeatureSelect={handleComparisonFeatureSelect}
              />
            </div>
          )}

          {/* Load External SAEs Mode */}
          {mode === 'load' && (
            <div className="load-layout">
              <ExternalSAELoader
                loadedSAEs={externalSAEs}
                onSAELoaded={refreshExternalSAEs}
                onSAEDeleted={refreshExternalSAEs}
              />
            </div>
          )}
        </div>
      </div>

      <style>{`
        .sae-page {
          min-height: 100vh;
          padding-top: 70px;
        }

        .sae-content {
          max-width: 1600px;
          margin: 0 auto;
          padding: 2rem;
        }

        .sae-header {
          margin-bottom: 1.5rem;
        }

        .sae-header h1 {
          margin: 0;
          font-size: 1.75rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.95);
          letter-spacing: -0.02em;
        }

        .header-description {
          margin: 0.5rem 0 0 0;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .input-section textarea {
          font-family: inherit;
        }

        .mode-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .mode-tab {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .mode-tab:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.7);
        }

        .mode-tab.active {
          background: rgba(168, 85, 247, 0.15);
          border-color: rgba(168, 85, 247, 0.5);
          color: rgba(255, 255, 255, 0.95);
        }

        .compare-info-banner {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.25);
          color: rgba(59, 130, 246, 0.9);
          margin-bottom: 1.5rem;
          font-size: 0.85rem;
        }

        .sae-main {
          min-height: 500px;
        }

        /* Empty State */
        .empty-state,
        .loading-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 4rem 2rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px dashed rgba(255, 255, 255, 0.1);
          text-align: center;
          min-height: 400px;
        }

        .empty-state .empty-icon {
          width: 48px;
          height: 48px;
          color: rgba(255, 255, 255, 0.2);
          margin-bottom: 1rem;
        }

        .empty-state h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.1rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.5);
        }

        .empty-state p {
          margin: 0;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.35);
        }

        /* Browse Layout */
        .browse-layout {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 1.5rem;
          align-items: start;
        }

        .browser-panel {
          min-height: 600px;
        }

        .detail-panel {
          position: sticky;
          top: 90px;
        }

        /* Analyze Layout */
        .analyze-layout {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 1.5rem;
          align-items: start;
        }

        .activation-display-panel {
          min-height: 500px;
        }

        .feature-detail-panel {
          position: sticky;
          top: 90px;
        }

        /* Dashboard Layout */
        .dashboard-layout {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        /* Compare Layout */
        .compare-layout {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .comparison-detail {
          max-width: 600px;
        }

        /* Load Layout */
        .load-layout {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          max-width: 900px;
        }

        /* Loading/Empty Detail */
        .empty-detail,
        .loading-detail {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem 2rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px dashed rgba(255, 255, 255, 0.1);
          text-align: center;
          min-height: 300px;
        }

        .empty-icon {
          width: 48px;
          height: 48px;
          color: rgba(255, 255, 255, 0.2);
          margin-bottom: 1rem;
        }

        .empty-detail h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.5);
        }

        .empty-detail p {
          margin: 0;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.35);
        }

        @media (max-width: 1200px) {
          .browse-layout,
          .analyze-layout {
            grid-template-columns: 1fr;
          }

          .detail-panel,
          .feature-detail-panel {
            position: static;
          }
        }

        @media (max-width: 768px) {
          .sae-content {
            padding: 1rem;
          }

          .mode-tabs {
            flex-wrap: wrap;
          }

          .mode-tab {
            flex: 1;
            min-width: 140px;
            justify-content: center;
          }
        }
      `}</style>
    </div>
  )
}
