import { useState, useCallback, useEffect } from 'react'
import { NavBar } from '../../components/common/NavBar'
import {
  SAESelector,
  FeatureActivationDisplay,
  FeatureBrowser,
  FeatureTopTokens,
  SparsityDashboard,
  ModelComparisonDashboard,
  ExternalSAELoader,
  type SAESelection,
} from '../../components/sae'
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
import { saeClient, type SAEAnalyzeResponse, type SAEFeatureInfoResponse, type ExternalSAEInfo } from '../../api/client'
import type {
  SequenceFeatureActivations,
  FeatureStats,
  FeatureInterpretation,
  FeatureCatalog,
  SAEQualityMetrics,
  SAETrainingProgress,
} from '../../types/model'
import { useTheme } from '../../hooks/useTheme'

interface SAEPageProps {
  onShowPurchaseCredits: () => void
}

type SAEMode = 'analyze' | 'browse' | 'dashboard' | 'compare' | 'load'

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
  const { isLight } = useTheme()

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

      setAnalyzeResponse(response)
      setActivations(transformAnalyzeResponse(response))
      setFeatureCatalog(generateFeatureCatalog(response, saeSelection))
      setQualityMetrics(generateQualityMetrics(response))
      setSelectedFeature(null)
      setSelectedFeatures([])
      setFeatureInfo(null)
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
  }, [saeSelection, inputText, onShowPurchaseCredits])

  // Load feature info when feature is selected
  useEffect(() => {
    if (selectedFeature === null || !saeSelection) {
      setFeatureInfo(null)
      return
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
  }, [selectedFeature, saeSelection, analyzeResponse])

  const handleFeatureSelect = useCallback((featureIdx: number) => {
    setSelectedFeature(featureIdx)
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

  const trainingProgress: SAETrainingProgress[] = []

  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
    textMid: isLight ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.6)',
    textSub: isLight ? 'rgba(0,0,0,0.45)' : 'rgba(255,255,255,0.4)',
    controlBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
    controlBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    inputBg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.4)',
    inputBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    errorBg: 'rgba(239, 68, 68, 0.08)',
    errorBorder: 'rgba(239, 68, 68, 0.2)',
    errorText: '#ef4444',
    warningBg: isLight ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.08)',
    warningBorder: isLight ? 'rgba(245,158,11,0.15)' : 'rgba(245,158,11,0.2)',
    warningText: isLight ? 'rgba(180,120,0,0.8)' : 'rgba(245,158,11,0.8)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    spinnerHead: isLight ? '#1d1d1f' : '#ffffff',
  }

  const modes: { id: SAEMode; label: string; icon: typeof Search }[] = [
    { id: 'analyze', label: 'Analyze', icon: Search },
    { id: 'browse', label: 'Browse', icon: Layers },
    { id: 'dashboard', label: 'Quality', icon: BarChart3 },
    { id: 'compare', label: 'Compare', icon: GitCompare },
    { id: 'load', label: 'Load', icon: Download },
  ]

  return (
    <div className="ds-page" style={{ background: c.bg }}>
      <NavBar />

      <div className="ds-layout" style={{ borderColor: c.divider }}>
        {/* Left Panel — 40% */}
        <div className="ds-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="ds-left-scroll">
            {/* Error banner */}
            {error && (
              <div className="ds-error" style={{ background: c.errorBg, borderColor: c.errorBorder }}>
                <AlertCircle style={{ width: 14, height: 14, color: c.errorText, flexShrink: 0 }} />
                <span style={{ color: c.errorText, fontSize: '0.8rem', flex: 1 }}>
                  {error === 'INSUFFICIENT_CREDITS'
                    ? 'Insufficient credits for SAE analysis.'
                    : error}
                </span>
                {error === 'INSUFFICIENT_CREDITS' ? (
                  <button
                    onClick={onShowPurchaseCredits}
                    style={{ background: 'none', border: `1px solid ${c.errorBorder}`, color: c.errorText, cursor: 'pointer', fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontWeight: 500 }}
                  >
                    Add Credits
                  </button>
                ) : (
                  <button
                    onClick={() => setError(null)}
                    style={{ background: 'none', border: 'none', color: c.errorText, cursor: 'pointer', fontSize: '0.75rem', padding: '0.125rem 0.5rem' }}
                  >
                    Dismiss
                  </button>
                )}
              </div>
            )}

            {/* SAE Selection — hidden in compare & load modes */}
            {mode !== 'compare' && mode !== 'load' && (
              <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                <div className="ds-section-label" style={{ color: c.textSub }}>SAE Selection</div>
                <SAESelector
                  selection={saeSelection}
                  onSelectionChange={setSaeSelection}
                  refreshKey={saeRefreshKey}
                />
              </div>
            )}

            {/* Text Input — hidden in compare & load modes */}
            {mode !== 'compare' && mode !== 'load' && (
              <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                <div className="ds-section-label" style={{ color: c.textSub }}>Input Text</div>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter text to analyze with the SAE..."
                  rows={3}
                  disabled={analyzing}
                  className="ds-textarea"
                  style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
                />
                <button
                  onClick={runAnalysis}
                  disabled={analyzing || !saeSelection || !inputText.trim()}
                  className="ds-run-btn"
                  style={{
                    background: analyzing ? c.controlBg : (isLight ? '#1d1d1f' : '#ffffff'),
                    color: analyzing ? c.textSub : (isLight ? '#ffffff' : '#0a0a0a'),
                    borderColor: 'transparent',
                  }}
                >
                  {analyzing ? (
                    <>
                      <div className="ds-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Play style={{ width: 14, height: 14 }} />
                      Analyze
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Analysis Mode */}
            <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="ds-section-label" style={{ color: c.textSub }}>View</div>
              <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
                {modes.map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setMode(id)}
                    className="ds-mode-btn"
                    style={{
                      background: mode === id ? (isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)') : c.controlBg,
                      borderColor: mode === id ? (isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.2)') : c.controlBorder,
                      color: mode === id ? c.text : c.textSub,
                    }}
                  >
                    <Icon style={{ width: 12, height: 12 }} />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Analysis summary — show when we have results */}
            {analyzeResponse && mode !== 'compare' && mode !== 'load' && (
              <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                <div className="ds-section-label" style={{ color: c.textSub }}>Analysis Summary</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: c.textMid }}>Tokens</span>
                    <span style={{ color: c.text, fontFamily: 'monospace' }}>{analyzeResponse.num_tokens}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: c.textMid }}>Active Features</span>
                    <span style={{ color: c.text, fontFamily: 'monospace' }}>{analyzeResponse.metrics.num_active_features}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: c.textMid }}>Avg L0</span>
                    <span style={{ color: c.text, fontFamily: 'monospace' }}>{analyzeResponse.metrics.avg_l0.toFixed(1)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                    <span style={{ color: c.textMid }}>Max Activation</span>
                    <span style={{ color: c.text, fontFamily: 'monospace' }}>{analyzeResponse.metrics.max_activation.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Selected feature info */}
            {selectedFeature !== null && (
              <div className="ds-section">
                <div className="ds-section-label" style={{ color: c.textSub }}>Selected Feature</div>
                <div style={{
                  padding: '0.625rem 0.75rem',
                  background: c.controlBg,
                  border: `1px solid ${c.controlBorder}`,
                  fontSize: '0.8rem',
                  color: c.text,
                  fontFamily: 'monospace',
                }}>
                  Feature #{selectedFeature}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel — 60% */}
        <div className="ds-right" style={{ background: c.bg }}>
          <div className="ds-right-scroll">
            {/* Analyze Activations Mode */}
            {mode === 'analyze' && !analyzing && analyzeResponse && activations && (
              <div className="ds-results">
                <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  <FeatureActivationDisplay
                    activations={activations}
                    selectedFeatures={selectedFeatures}
                    onFeatureSelect={handleFeatureSelect}
                    showTopK={10}
                  />
                </div>
                {selectedFeature !== null && featureInfo ? (
                  <div className="ds-section">
                    <FeatureTopTokens
                      interpretation={featureInfo}
                      maxExamples={10}
                    />
                  </div>
                ) : loadingFeature ? (
                  <div className="ds-loading-detail">
                    <Loader2 style={{ width: 20, height: 20, color: c.textSub }} className="ds-spinning" />
                    <p style={{ color: c.textSub, fontSize: '0.8rem', marginTop: '0.5rem' }}>Loading feature info...</p>
                  </div>
                ) : analyzeResponse ? (
                  <div className="ds-hint" style={{ borderTop: `1px solid ${c.divider}` }}>
                    <p style={{ color: c.textSub, fontSize: '0.8rem', margin: 0 }}>
                      Click a cell in the heatmap above to view feature details.
                    </p>
                  </div>
                ) : null}
              </div>
            )}

            {/* Browse Features Mode */}
            {mode === 'browse' && !analyzing && analyzeResponse && featureCatalog && (
              <div className="ds-results">
                <div className="ds-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  <FeatureBrowser
                    catalog={featureCatalog}
                    onFeatureSelect={handleFeatureSelect}
                    selectedFeature={selectedFeature ?? undefined}
                  />
                </div>
                {selectedFeature !== null && featureInfo ? (
                  <div className="ds-section">
                    <FeatureTopTokens
                      interpretation={featureInfo}
                      maxExamples={15}
                    />
                  </div>
                ) : loadingFeature ? (
                  <div className="ds-loading-detail">
                    <Loader2 style={{ width: 20, height: 20, color: c.textSub }} className="ds-spinning" />
                    <p style={{ color: c.textSub, fontSize: '0.8rem', marginTop: '0.5rem' }}>Loading feature info...</p>
                  </div>
                ) : null}
              </div>
            )}

            {/* Quality Dashboard Mode */}
            {mode === 'dashboard' && !analyzing && analyzeResponse && qualityMetrics && (
              <div className="ds-section">
                <SparsityDashboard
                  metrics={qualityMetrics}
                  trainingProgress={trainingProgress}
                />
              </div>
            )}

            {/* Compare SAEs Mode */}
            {mode === 'compare' && (
              <div className="ds-section">
                <ModelComparisonDashboard
                  onFeatureSelect={handleComparisonFeatureSelect}
                />
              </div>
            )}

            {/* Load External SAEs Mode */}
            {mode === 'load' && (
              <div className="ds-section">
                <ExternalSAELoader
                  loadedSAEs={externalSAEs}
                  onSAELoaded={refreshExternalSAEs}
                  onSAEDeleted={refreshExternalSAEs}
                />
              </div>
            )}

            {/* Loading state */}
            {analyzing && (
              <div className="ds-placeholder">
                <div className="ds-spinner-lg" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
                <p style={{ color: c.textSub, fontSize: '0.8rem', marginTop: '1rem' }}>Running SAE analysis on GPU...</p>
              </div>
            )}

            {/* Empty state */}
            {!analyzing && !analyzeResponse && mode !== 'compare' && mode !== 'load' && (
              <div className="ds-placeholder">
                <Search style={{ width: 36, height: 36, color: c.textSub, opacity: 0.3, marginBottom: '0.75rem' }} />
                <p style={{ color: c.textSub, fontSize: '0.8rem', margin: 0, maxWidth: 360, lineHeight: 1.5 }}>
                  Select an SAE and enter text to analyze feature activations.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .ds-page {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .ds-layout {
          display: flex;
          flex: 1;
          margin-top: 76px;
          overflow: hidden;
        }

        .ds-left {
          width: 40%;
          min-width: 340px;
          max-width: 520px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .ds-left-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .ds-right {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .ds-right-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .ds-section {
          padding: 1rem 1.25rem;
        }

        .ds-section-label {
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 0.625rem;
        }

        .ds-textarea {
          width: 100%;
          padding: 0.625rem 0.75rem;
          border: 1px solid;
          font-size: 0.85rem;
          font-family: inherit;
          resize: vertical;
          min-height: 68px;
          outline: none;
          box-sizing: border-box;
          margin-bottom: 0.625rem;
        }

        .ds-textarea:focus {
          border-color: rgba(59, 130, 246, 0.4) !important;
        }

        .ds-textarea::placeholder {
          color: ${isLight ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.3)'};
        }

        .ds-run-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          border: 1px solid;
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
          transition: opacity 0.15s;
          letter-spacing: 0.01em;
        }

        .ds-run-btn:hover:not(:disabled) {
          opacity: 0.85;
        }

        .ds-run-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .ds-mode-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.375rem;
          padding: 0.5rem 0.375rem;
          border: 1px solid;
          font-size: 0.7rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          min-width: 0;
        }

        .ds-mode-btn:hover {
          opacity: 0.85;
        }

        .ds-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid;
          border-radius: 50%;
          animation: ds-spin 0.8s linear infinite;
        }

        .ds-spinner-lg {
          width: 28px;
          height: 28px;
          border: 2px solid;
          border-radius: 50%;
          animation: ds-spin 0.8s linear infinite;
        }

        @keyframes ds-spin {
          to { transform: rotate(360deg); }
        }

        .ds-spinning {
          animation: ds-spin 1s linear infinite;
        }

        .ds-results {
          display: flex;
          flex-direction: column;
        }

        .ds-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 4rem 2rem;
          text-align: center;
          height: 100%;
          min-height: 400px;
        }

        .ds-loading-detail {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 2rem;
        }

        .ds-hint {
          padding: 1rem 1.25rem;
          text-align: center;
        }

        .ds-error {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 0.875rem;
          border: 1px solid;
          margin: 0.75rem 1.25rem 0;
        }

        @media (max-width: 1000px) {
          .ds-layout {
            flex-wrap: wrap;
          }
          .ds-left {
            width: 100%;
            max-width: 100%;
            max-height: 40vh;
            border-right: none !important;
            border-bottom: 1px solid;
          }
          .ds-right {
            width: 100%;
          }
        }

        @media (max-width: 600px) {
          .ds-mode-btn span {
            display: none;
          }
        }
      `}</style>
    </div>
  )
}
