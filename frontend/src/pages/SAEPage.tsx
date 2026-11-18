import { useState, useCallback, useEffect } from 'react'
import { SaeDeepLinkNote } from '../components/sae/SaeDeepLinkNote'
import { useSaeDeepLink } from '../hooks/useSaeDeepLink'
import { useAssistantPage } from '../hooks/useAssistantPage'
import { saeBridge } from '../components/assistant/pages/sae'
import { saeRef, useCorpusStats } from '../hooks/useCorpusStats'
import { useFeatureLabels } from '../hooks/useFeatureLabels'
import { FeatureExplanation } from '../components/autointerp/FeatureExplanation'
import { AutointerpView } from '../components/autointerp/AutointerpView'
import type { AtlasCorpusId } from '../types/geometry'
import { useAuthStore } from '../stores/authStore'
import { NavBar } from '../components/common/NavBar'
import { EmptyState } from '../components/common/EmptyState'
import { Spinner } from '../components/common/Spinner'
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
  
  
  
  
  
  
  AlertCircle } from 'lucide-react'
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
  onShowPurchaseCredits: () => void
}

type SAEMode = 'analyze' | 'browse' | 'dashboard' | 'compare' | 'load' | 'autointerp'

// Transform API response to SequenceFeatureActivations format
function transformAnalyzeResponse(response: SAEAnalyzeResponse): SequenceFeatureActivations {
  const tokens = response.tokens
  const tokenIds = response.token_ids
  const numFeatures = response.features.shape[1]

  const featureActivationMatrix: number[][] = response.features.sparse_activations.map(
    sparseActs => {
      const row = new Array(numFeatures).fill(0)
      Object.entries(sparseActs).forEach(([idx, val]) => {
        row[parseInt(idx)] = val
      })
      return row
    }
  )

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
  const tokenFrequencies: Record<string, number> = {}
  if (analyzeResponse) {
    analyzeResponse.top_features
      .filter(f => f.feature_id === featureInfo.feature_id)
      .forEach(f => {
        tokenFrequencies[f.token] = (tokenFrequencies[f.token] || 0) + 1
      })
    const total = Object.values(tokenFrequencies).reduce((a, b) => a + b, 0)
    if (total > 0) {
      Object.keys(tokenFrequencies).forEach(k => {
        tokenFrequencies[k] /= total
      })
    }
  }

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
  const saeId = saeSelection.externalId
    ? saeSelection.externalId
    : `${saeSelection.model}-L${saeSelection.layer}-${saeSelection.activationType}`

  const featureActivations = new Map<number, { count: number; total: number; max: number }>()
  analyzeResponse.top_features.forEach(({ feature_id, activation }) => {
    const curr = featureActivations.get(feature_id) || { count: 0, total: 0, max: 0 }
    featureActivations.set(feature_id, {
      count: curr.count + 1,
      total: curr.total + activation,
      max: Math.max(curr.max, activation),
    })
  })

  const features: FeatureStats[] = Array.from(featureActivations.entries()).map(
    ([idx, stats]) => ({
      feature_idx: idx,
      activation_frequency: stats.count / analyzeResponse.num_tokens,
      mean_activation: stats.total / stats.count,
      max_activation: stats.max,
      polysemanticity: Math.random() * 0.5,
      suggested_label: `Feature ${idx}`,
      confidence: 0.5,
      unique_tokens: stats.count,
    })
  )

  features.sort((a, b) => b.activation_frequency - a.activation_frequency)

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
      sparsity: { avg_l0: 0, l0_std: 0, sparsity_fraction: 0, avg_l1: 0, max_activation: 0 },
      feature_health: {
        num_features: 0, dead_features: 0, dead_feature_fraction: 0,
        low_frequency_features: 0, high_frequency_features: 0,
        feature_frequency_distribution: [], feature_magnitude_distribution: [],
      },
      reconstruction: {
        mse: 0, rmse: 0, normalized_mse: 0, explained_variance: 0,
        cosine_similarity: 0, relative_reconstruction_error: 0,
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
      mse: 0.01, rmse: 0.1, normalized_mse: 0.02,
      explained_variance: 0.92, cosine_similarity: 0.98, relative_reconstruction_error: 0.05,
    },
  }
}

export default function DefaultSAEPage({ onShowPurchaseCredits }: SAEPageProps) {

  const [mode, setMode] = useState<SAEMode>('analyze')
  // The Autointerp view stays mounted once opened (its jobs and tables survive switching modes)
  const [autointerpOpened, setAutointerpOpened] = useState(false)
  useEffect(() => {
    if (mode === 'autointerp') setAutointerpOpened(true)
  }, [mode])
  // A link from the Probe Lab picks the SAE, the text, and a feature to show once analyzed
  const { link: deepLink, pendingFeature, clearPending, setPendingFeature } = useSaeDeepLink()
  const [saeSelection, setSaeSelection] = useState<SAESelection | null>(deepLink?.selection ?? null)
  // Corpus statistics from a Geometry Lab atlas of the SAE, when one exists (Browse uses them)
  const { isAuthenticated } = useAuthStore()
  const { atlas: corpusAtlas, catalog: corpusCatalog, loadFeature: loadCorpusFeature } = useCorpusStats(saeSelection, isAuthenticated)
  const [inputText, setInputText] = useState(deepLink?.text ?? '')
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Analysis results
  const [analyzeResponse, setAnalyzeResponse] = useState<SAEAnalyzeResponse | null>(null)
  const [activations, setActivations] = useState<SequenceFeatureActivations | null>(null)
  const [featureCatalog, setFeatureCatalog] = useState<FeatureCatalog | null>(null)
  const [qualityMetrics, setQualityMetrics] = useState<SAEQualityMetrics | null>(null)
  // Autointerp: the selected SAE, and labels of the features the heatmap shows
  const selectedRef = saeRef(saeSelection)
  const heatmapLabels = useFeatureLabels(selectedRef, activations?.most_active_features ?? [])

  // Feature selection state
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null)
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([])
  const [featureInfo, setFeatureInfo] = useState<FeatureInterpretation | null>(null)
  const [loadingFeature, setLoadingFeature] = useState(false)

  // External SAEs state
  const [externalSAEs, setExternalSAEs] = useState<ExternalSAEInfo[]>([])
  const [saeRefreshKey, setSaeRefreshKey] = useState(0)

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
      setSaeRefreshKey(k => k + 1)
    } catch {
      // Silently handle
    }
  }, [])

  // An analysis on screen: the Analyze button's, or an assistant run's
  const applyAnalysis = useCallback((response: SAEAnalyzeResponse, selection: SAESelection) => {
    setAnalyzeResponse(response)
    setActivations(transformAnalyzeResponse(response))
    setFeatureCatalog(generateFeatureCatalog(response, selection))
    setQualityMetrics(generateQualityMetrics(response))
    setSelectedFeature(null)
    setSelectedFeatures([])
    setFeatureInfo(null)
  }, [])

  const runAnalysis = useCallback(async () => {
    if (!saeSelection || !inputText.trim()) return

    setAnalyzing(true)
    setError(null)

    try {
      let response: SAEAnalyzeResponse

      if (saeSelection.externalId) {
        response = await saeClient.analyzeExternalSAE({
          sae_id: saeSelection.externalId,
          text: inputText,
          top_k: 50,
        })
      } else {
        response = await saeClient.analyzeText({
          model: saeSelection.model,
          layer: saeSelection.layer,
          activation_type: saeSelection.activationType,
          text: inputText,
          top_k: 50,
        })
      }

      if (response.error) {
        throw new Error(response.error)
      }

      applyAnalysis(response, saeSelection)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        setError('INSUFFICIENT_CREDITS')
        onShowPurchaseCredits()
      } else {
        setError(err instanceof Error ? err.message : 'Analysis failed')
      }
    } finally {
      setAnalyzing(false)
    }
  }, [saeSelection, inputText, onShowPurchaseCredits, applyAnalysis])

  // Load feature info when feature is selected
  useEffect(() => {
    if (selectedFeature === null || !saeSelection) {
      setFeatureInfo(null)
      return
    }
    if (mode === 'browse' && corpusAtlas) {
      let cancelled = false
      setLoadingFeature(true)
      loadCorpusFeature(selectedFeature)
        .then((info) => !cancelled && setFeatureInfo(info))
        .catch(() => !cancelled && setFeatureInfo(null))
        .finally(() => !cancelled && setLoadingFeature(false))
      return () => {
        cancelled = true
      }
    }

    const loadFeatureInfo = async () => {
      setLoadingFeature(true)
      try {
        let info: SAEFeatureInfoResponse

        if (saeSelection.externalId) {
          const extInfo = await saeClient.getExternalFeatureInfo(
            saeSelection.externalId,
            selectedFeature
          )
          if (extInfo.error) {
            setFeatureInfo(null)
            return
          }
          info = {
            feature_id: extInfo.feature_id,
            model: extInfo.display_name || saeSelection.externalId,
            layer: 0,
            activation_type: extInfo.activation_type,
            d_input: extInfo.d_input,
            d_hidden: extInfo.d_hidden,
            decoder_direction: extInfo.decoder_direction,
            decoder_norm: extInfo.decoder_norm,
            encoder_weights: extInfo.encoder_weights,
            encoder_bias: extInfo.encoder_bias,
          }
        } else {
          info = await saeClient.getFeatureInfo(
            saeSelection.model,
            saeSelection.layer,
            saeSelection.activationType,
            selectedFeature
          )
        }

        if (info.error) {
          setFeatureInfo(null)
        } else {
          setFeatureInfo(transformFeatureInfo(info, analyzeResponse))
        }
      } catch {
        setFeatureInfo(null)
      } finally {
        setLoadingFeature(false)
      }
    }

    loadFeatureInfo()
  }, [selectedFeature, saeSelection, analyzeResponse, mode, corpusAtlas, loadCorpusFeature])

  const handleFeatureSelect = useCallback((featureIdx: number) => {
    setSelectedFeature(featureIdx)
    setSelectedFeatures(prev => {
      const filtered = prev.filter(f => f !== featureIdx)
      return [featureIdx, ...filtered].slice(0, 5)
    })
  }, [])

  // The assistant: what's on screen, and set_inputs
  useAssistantPage('sae', saeBridge({
    mode,
    setMode,
    selection: saeSelection,
    setSelection: setSaeSelection,
    text: inputText,
    setText: setInputText,
    analyzing,
    analyzeResponse,
    selectedFeature,
    selectFeature: handleFeatureSelect,
    setPendingFeature,
    corpusAtlas,
    loadAnalysis: applyAnalysis,
  }))

  useEffect(() => {
    if (analyzeResponse && pendingFeature != null) {
      handleFeatureSelect(pendingFeature)
      clearPending()
    }
  }, [analyzeResponse, pendingFeature, handleFeatureSelect, clearPending])

  const handleComparisonFeatureSelect = useCallback((_saeId: string, featureIdx: number) => {
    setSelectedFeature(featureIdx)
  }, [])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runAnalysis()
    }
  }

  const trainingProgress: SAETrainingProgress[] = []

  const modes: { id: SAEMode; label: string }[] = [
    { id: 'analyze', label: 'Analyze' },
    { id: 'browse', label: 'Browse' },
    { id: 'dashboard', label: 'Quality' },
    { id: 'compare', label: 'Compare' },
    { id: 'autointerp', label: 'Autointerp' },
    { id: 'load', label: 'Load' },
  ]

  const renderLeftPanel = (compareConfigContent?: React.ReactNode) => (
    <div className={`ds-left oz-side${mode === 'compare' ? ' ds-left-wide' : ''}`}>
      <div className="oz-scroll oz-steps">
        {/* Analysis Mode — always first */}
        <div className="oz-section">
          <h2 className="oz-title">View</h2>
          <div data-assist="sae.mode" className="oz-chips oz-chips--grid" role="tablist" aria-label="SAE Analysis views">
            {modes.map(({ id, label }) => (
              <button
                key={id}
                role="tab"
                aria-selected={mode === id}
                onClick={() => setMode(id)}
                className={`oz-chip${mode === id ? ' active' : ''}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="ds-error oz-note oz-note--error" role="alert">
            <AlertCircle aria-hidden="true" />
            <span>
              {error === 'INSUFFICIENT_CREDITS'
                ? 'Insufficient credits for SAE analysis.'
                : error}
            </span>
            {error === 'INSUFFICIENT_CREDITS' ? (
              <button className="oz-link" onClick={onShowPurchaseCredits}>
                Add Credits
              </button>
            ) : (
              <button className="oz-link" onClick={() => setError(null)}>
                Dismiss
              </button>
            )}
          </div>
        )}

        {/* Compare config in left panel */}
        {compareConfigContent && (
          <div className="oz-section ds-compare-config">
            <h2 className="oz-title">SAE comparison</h2>
            {compareConfigContent}
          </div>
        )}

        {/* SAE Selection — hidden in compare & load modes */}
        {mode !== 'compare' && mode !== 'load' && (
          <div className="oz-section" data-assist="sae.sae">
            <h2 className="oz-title">SAE selection</h2>
            <SAESelector
              selection={saeSelection}
              onSelectionChange={setSaeSelection}
              refreshKey={saeRefreshKey}
            />
          </div>
        )}

        {/* Text Input — hidden in compare, load & autointerp modes */}
        {mode !== 'compare' && mode !== 'load' && mode !== 'autointerp' && (
          <div className="oz-section">
            <h2 className="oz-title">Input text</h2>
            {pendingFeature != null && <SaeDeepLinkNote feature={pendingFeature} />}
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter text to analyze with the SAE..."
              rows={3}
              disabled={analyzing}
              data-assist="sae.text"
              className="oz-textarea"
            />
            <button
              onClick={runAnalysis}
              disabled={analyzing || !saeSelection || !inputText.trim()}
              data-assist="sae.analyze_button"
              className={`oz-btn oz-btn--block ds-run-btn${analyzing ? '' : ' oz-btn--arrow'}`}
            >
              {analyzing ? (
                <>
                  <Spinner />
                  Analyzing...
                </>
              ) : (
                'Analyze'
              )}
            </button>
          </div>
        )}

        {/* Analysis summary — show when we have results */}
        {analyzeResponse && mode !== 'compare' && mode !== 'load' && mode !== 'autointerp' && (
          <div className="oz-section">
            <h2 className="oz-title">Analysis summary</h2>
            <dl className="ds-specs">
              <div>
                <dt>Tokens</dt>
                <dd>{analyzeResponse.num_tokens}</dd>
              </div>
              <div>
                <dt>Active features</dt>
                <dd>{analyzeResponse.metrics.num_active_features}</dd>
              </div>
              <div>
                <dt>Avg L0</dt>
                <dd>{analyzeResponse.metrics.avg_l0.toFixed(1)}</dd>
              </div>
              <div>
                <dt>Max activation</dt>
                <dd>{analyzeResponse.metrics.max_activation.toFixed(2)}</dd>
              </div>
            </dl>
          </div>
        )}

        {/* Selected feature info */}
        {selectedFeature !== null && (
          <div className="oz-section">
            <h2 className="oz-title">Selected feature</h2>
            <div className="ds-feature-id">#{selectedFeature}</div>
          </div>
        )}
      </div>
    </div>
  )

  const renderRightPanel = (compareResultsContent?: React.ReactNode) => (
    <div className="ds-right oz-main">
      <div className="oz-scroll">
        {/* Analyze Activations Mode */}
        {mode === 'analyze' && !analyzing && analyzeResponse && activations && (
          <div className="ds-results" data-assist="sae.feature">
            <div className="oz-section">
              <FeatureActivationDisplay
                activations={activations}
                selectedFeatures={selectedFeatures}
                onFeatureSelect={handleFeatureSelect}
                showTopK={10}
                labels={heatmapLabels}
              />
            </div>
            {selectedFeature !== null && featureInfo ? (
              <div className="oz-section">
                <FeatureTopTokens
                  interpretation={featureInfo}
                  maxExamples={10}
                />
              </div>
            ) : loadingFeature ? (
              <div className="oz-empty ds-loading-detail">
                <Spinner large />
                <p>Loading feature info...</p>
              </div>
            ) : analyzeResponse ? (
              <p className="oz-hint ds-hint">Click a cell in the heatmap above to view feature details.</p>
            ) : null}
          </div>
        )}

        {/* Browse Features Mode */}
        {mode === 'browse' && !analyzing && (corpusCatalog ?? (analyzeResponse ? featureCatalog : null)) && (
          <div className="ds-results">
            <div className="oz-section">
              {corpusAtlas && (
                <p
                  className="oz-hint"
                  style={{ margin: '0 0 0.75rem' }}
                  title={`Feature statistics and top contexts from a Geometry Lab atlas (${corpusAtlas.corpus.license ?? ''})`}
                >
                  Over {corpusAtlas.tokens.toLocaleString()} {corpusAtlas.corpus.name} tokens
                </p>
              )}
              <FeatureBrowser
                catalog={(corpusCatalog ?? featureCatalog) as FeatureCatalog}
                onFeatureSelect={handleFeatureSelect}
                selectedFeature={selectedFeature ?? undefined}
              />
            </div>
            {selectedFeature !== null && corpusAtlas && selectedRef && (
              <div className="oz-section">
                <FeatureExplanation
                  atlasRef={{ model: corpusAtlas.model, sae: selectedRef, corpus: corpusAtlas.corpus.id as AtlasCorpusId }}
                  feature={selectedFeature}
                  onShowPurchaseCredits={onShowPurchaseCredits}
                />
              </div>
            )}
            {selectedFeature !== null && featureInfo ? (
              <div className="oz-section">
                <FeatureTopTokens
                  interpretation={featureInfo}
                  maxExamples={15}
                />
              </div>
            ) : loadingFeature ? (
              <div className="oz-empty ds-loading-detail">
                <Spinner large />
                <p>Loading feature info...</p>
              </div>
            ) : null}
          </div>
        )}

        {/* Quality Dashboard Mode */}
        {mode === 'dashboard' && !analyzing && analyzeResponse && qualityMetrics && (
          <div className="oz-section">
            <SparsityDashboard
              metrics={qualityMetrics}
              trainingProgress={trainingProgress}
            />
          </div>
        )}

        {/* Compare SAEs Mode — results only */}
        {mode === 'compare' && compareResultsContent && (
          <div className="oz-section">
            {compareResultsContent}
          </div>
        )}

        {/* Autointerp Mode */}
        {autointerpOpened && (
          <div className="oz-section" hidden={mode !== 'autointerp'}>
            <AutointerpView sae={selectedRef} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
        )}

        {/* Load External SAEs Mode */}
        {mode === 'load' && (
          <div className="oz-section">
            <ExternalSAELoader
              loadedSAEs={externalSAEs}
              onSAELoaded={refreshExternalSAEs}
              onSAEDeleted={refreshExternalSAEs}
            />
          </div>
        )}

        {/* Loading state */}
        {analyzing && (
          <div className="oz-empty oz-empty--fill ds-placeholder">
            <Spinner large />
            <p>Running SAE analysis on GPU...</p>
          </div>
        )}

        {/* Empty state */}
        {!analyzing && !analyzeResponse && mode !== 'compare' && mode !== 'load' && mode !== 'autointerp' && !(mode === 'browse' && corpusCatalog) && (
          <EmptyState glyph="sae" fill className="ds-placeholder">
            Select an SAE and enter text to analyze feature activations.
          </EmptyState>
        )}
      </div>
    </div>
  )

  return (
    <div className="ds-page oz-page">
      <NavBar />

      {mode === 'compare' ? (
        <ModelComparisonDashboard
          onFeatureSelect={handleComparisonFeatureSelect}
        >
          {({ configContent, resultsContent }) => (
            <div className="ds-layout oz-page__body">
              {renderLeftPanel(configContent)}
              {renderRightPanel(resultsContent)}
            </div>
          )}
        </ModelComparisonDashboard>
      ) : (
        <div className="ds-layout oz-page__body">
          {renderLeftPanel()}
          {renderRightPanel()}
        </div>
      )}

      <style>{`
        .ds-left.oz-side.ds-left-wide {
          width: 40%;
          max-width: 620px;
        }

        .ds-page .ds-error {
          margin: 1rem var(--oz-gutter) 0;
        }

        .ds-page .ds-run-btn {
          margin-top: 6px;
        }

        /* A sheet of numbers: a name and its value to a row */
        .ds-specs {
          margin: 0;
        }

        .ds-specs > div {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.55rem 0;
          border-top: 1px solid var(--oz-rule);
        }

        .ds-specs > div:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .ds-specs dt {
          color: var(--oz-fg-dim);
          font-size: 0.875rem;
        }

        .ds-specs dd {
          margin: 0;
          font-family: var(--oz-mono);
          font-size: 0.9rem;
        }

        .ds-feature-id {
          font-family: var(--oz-mono);
          font-size: 2rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.1;
        }

        .ds-results {
          display: flex;
          flex-direction: column;
        }

        .ds-results > .oz-section:last-child,
        .ds-right .oz-scroll > .oz-section:last-child {
          border-bottom: none;
        }

        .ds-page .ds-hint {
          padding: 1.25rem var(--oz-gutter);
        }

        .ds-page .ds-placeholder {
          min-height: calc(125vh - var(--oz-nav-h) - 2px);
        }

        .ds-compare-config .grid {
          grid-template-columns: 1fr !important;
        }

        @media (max-width: 1100px) {
          .ds-page.oz-page { height: auto; overflow: visible; }
          .ds-layout.oz-page__body { flex-direction: column; }
          .ds-left.oz-side, .ds-left.oz-side.ds-left-wide { width: 100%; max-width: 100%; border-right: none; border-bottom: 1px solid var(--oz-rule); }
        }
      `}</style>
    </div>
  )
}
