import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { NavBar } from '../../components/common/NavBar'
import { HeadClassifier, AttentionPatternGrid, HeadComparer } from '../../components/analysis'
import { apiClient } from '../../api/client'
import { Dropdown } from '../../components/common/Dropdown'
import type {
  ClassifyHeadsResponse,
  CompareAttentionResponse,
  MinePatternResponse,
  HeadType,
} from '../../types/analysis'
import type { CapturedActivationSummary, PatchingModelInfo } from '../../types/patching'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../../types/analysis'
import {
  Search,
  GitCompare,
  Layers,
  AlertCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Zap,
  Copy,
  Check,
  X,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useTheme } from '../../hooks/useTheme'

interface AnalysisPageProps {
  onShowPurchaseCredits?: () => void
}

type AnalysisMode = 'classify' | 'compare' | 'mine'

export default function DefaultAnalysisPage({
  onShowPurchaseCredits,
}: AnalysisPageProps) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const { isLight } = useTheme()

  // Mode selection
  const [mode, setMode] = useState<AnalysisMode>('classify')

  // Model and activation selection
  const [models, setModels] = useState<PatchingModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [capturedActivations, setCapturedActivations] = useState<CapturedActivationSummary[]>([])
  const [selectedActivation1, setSelectedActivation1] = useState<string>('')
  const [selectedActivation2, setSelectedActivation2] = useState<string>('')

  // Prompt capture
  const [capturePrompt, setCapturePrompt] = useState('')
  const [capturing, setCapturing] = useState(false)

  // Results
  const [classificationResult, setClassificationResult] = useState<ClassifyHeadsResponse | null>(null)
  const [comparisonResult, setComparisonResult] = useState<CompareAttentionResponse | null>(null)
  const [miningResult, setMiningResult] = useState<MinePatternResponse | null>(null)

  // UI state
  const [selectedHead, setSelectedHead] = useState<{ layer: number; head: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showCapturedList, setShowCapturedList] = useState(true)
  const [expandedPatternHeads, setExpandedPatternHeads] = useState<number | null>(null)
  const [copiedPatternIdx, setCopiedPatternIdx] = useState<number | null>(null)

  useEffect(() => {
    loadModels()
    loadCapturedActivations()
  }, [])

  const loadModels = async () => {
    try {
      const modelList = await apiClient.getPatchingModels()
      setModels(modelList)
      if (modelList.length > 0) setSelectedModel(modelList[0].model_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    }
  }

  const loadCapturedActivations = useCallback(async () => {
    try {
      const activations = await apiClient.listCapturedActivations()
      setCapturedActivations(activations)
    } catch (err) {
      console.error('Failed to load captured activations:', err)
    }
  }, [])

  const filteredActivations = capturedActivations.filter(
    (a) => a.model_id === selectedModel
  )

  const getHeadType = (layer: number, head: number): HeadType | null => {
    if (!classificationResult || classificationResult.activation_id !== selectedActivation1) return null
    const classification = classificationResult.classifications.find(
      c => c.layer === layer && c.head === head
    )
    return classification?.primary_type || null
  }

  const copyHeadsList = async (heads: [number, number][], patternIdx: number) => {
    const text = heads.map(([l, h]) => `L${l}H${h}`).join(', ')
    await navigator.clipboard.writeText(text)
    setCopiedPatternIdx(patternIdx)
    setTimeout(() => setCopiedPatternIdx(null), 2000)
  }

  const handleCaptureActivations = async () => {
    if (!isAuthenticated) { navigate('/auth'); return }
    if (!capturePrompt.trim() || !selectedModel) {
      setError('Please enter a prompt and select a model')
      return
    }

    try {
      setCapturing(true)
      setError(null)
      const result = await apiClient.captureActivations({
        prompt: capturePrompt,
        model: selectedModel,
      })
      await loadCapturedActivations()
      setSelectedActivation1(result.activation_id)
      setCapturePrompt('')
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        onShowPurchaseCredits?.()
      } else {
        setError(err instanceof Error ? err.message : 'Failed to capture activations')
      }
    } finally {
      setCapturing(false)
    }
  }

  const handleClassify = async () => {
    if (!isAuthenticated) { navigate('/auth'); return }
    if (!selectedActivation1) { setError('Please select captured activations to analyze'); return }

    try {
      setLoading(true)
      setError(null)
      setClassificationResult(null)
      const result = await apiClient.classifyHeads({ activation_id: selectedActivation1 })
      setClassificationResult(result)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        onShowPurchaseCredits?.()
      } else {
        setError(err instanceof Error ? err.message : 'Failed to classify heads')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleCompare = async () => {
    if (!isAuthenticated) { navigate('/auth'); return }
    if (!selectedActivation1 || !selectedActivation2) {
      setError('Please select two activations to compare')
      return
    }

    try {
      setLoading(true)
      setError(null)
      setComparisonResult(null)
      const result = await apiClient.compareAttention({
        activation_id_1: selectedActivation1,
        activation_id_2: selectedActivation2,
      })
      setComparisonResult(result)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        onShowPurchaseCredits?.()
      } else {
        setError(err instanceof Error ? err.message : 'Failed to compare attention patterns')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleMinePatterns = async () => {
    if (!isAuthenticated) { navigate('/auth'); return }
    if (!selectedActivation1) { setError('Please select captured activations to analyze'); return }

    try {
      setLoading(true)
      setError(null)
      setMiningResult(null)
      const result = await apiClient.minePatterns({ activation_id: selectedActivation1 })
      setMiningResult(result)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        onShowPurchaseCredits?.()
      } else {
        setError(err instanceof Error ? err.message : 'Failed to mine patterns')
      }
    } finally {
      setLoading(false)
    }
  }

  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
    textMid: isLight ? '#1d1d1f' : '#ffffff',
    textSub: isLight ? '#1d1d1f' : '#ffffff',
    controlBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
    controlBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    controlHover: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.08)',
    inputBg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.4)',
    inputBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    errorBg: 'rgba(239, 68, 68, 0.08)',
    errorBorder: 'rgba(239, 68, 68, 0.2)',
    errorText: '#ef4444',
    accentBg: isLight ? 'rgba(59,130,246,0.08)' : 'rgba(59,130,246,0.12)',
    accentText: isLight ? '#2563eb' : 'rgba(96,165,250,0.9)',
    circuitBg: isLight ? 'rgba(168,85,247,0.06)' : 'rgba(168,85,247,0.1)',
    circuitBorder: isLight ? 'rgba(168,85,247,0.15)' : 'rgba(168,85,247,0.25)',
    circuitText: isLight ? '#7c3aed' : 'rgba(168,85,247,0.9)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    spinnerHead: isLight ? '#1d1d1f' : '#ffffff',
    warningBg: isLight ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.08)',
    warningBorder: isLight ? 'rgba(245,158,11,0.15)' : 'rgba(245,158,11,0.2)',
    warningText: isLight ? 'rgba(180,120,0,0.8)' : 'rgba(245,158,11,0.8)',
  }

  const modes: { id: AnalysisMode; label: string; icon: typeof Search }[] = [
    { id: 'classify', label: 'Classify Heads', icon: Search },
    { id: 'compare', label: 'Compare Prompts', icon: GitCompare },
    { id: 'mine', label: 'Mine Patterns', icon: Layers },
  ]

  return (
    <div className="da-page" style={{ background: c.bg }}>
      <NavBar />

      <div className="da-layout" style={{ borderColor: c.divider }}>
        {/* Left Panel — 30% */}
        <div className="da-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="da-left-scroll">
            {/* Error banner */}
            {error && (
              <div className="da-error" style={{ background: c.errorBg, borderColor: c.errorBorder }}>
                <AlertCircle style={{ width: 14, height: 14, color: c.errorText, flexShrink: 0 }} />
                <span style={{ color: c.errorText, fontSize: '0.8rem', flex: 1 }}>{error}</span>
                <button
                  onClick={() => setError(null)}
                  style={{ background: 'none', border: 'none', color: c.errorText, cursor: 'pointer', fontSize: '0.75rem', padding: '0.125rem 0.5rem' }}
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Model section */}
            <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="da-section-label" style={{ color: c.textSub }}>Model</div>
              <Dropdown
                value={selectedModel}
                onChange={(v) => {
                  setSelectedModel(v)
                  setSelectedActivation1('')
                  setSelectedActivation2('')
                }}
                options={models.map((model) => ({
                  value: model.model_id,
                  label: `${model.display_name} (${model.num_layers}L, ${model.num_heads}H)`,
                }))}
              />
            </div>

            {/* Capture Activations */}
            <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="da-section-label" style={{ color: c.textSub }}>Capture Activations</div>
              <textarea
                value={capturePrompt}
                onChange={(e) => setCapturePrompt(e.target.value)}
                placeholder="Enter a prompt to capture activations..."
                rows={3}
                disabled={capturing}
                className="da-textarea"
                style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
              />
              <button
                onClick={handleCaptureActivations}
                disabled={capturing || !capturePrompt.trim()}
                className="da-capture-btn"
                style={{
                  background: c.controlBg,
                  borderColor: c.controlBorder,
                  color: c.textMid,
                }}
              >
                {capturing ? (
                  <>
                    <RefreshCw style={{ width: 14, height: 14 }} className="da-spinning" />
                    Capturing...
                  </>
                ) : (
                  <>
                    <Zap style={{ width: 14, height: 14 }} />
                    Capture
                  </>
                )}
              </button>
            </div>

            {/* Captured Activations */}
            <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <button
                className="da-toggle-btn"
                onClick={() => setShowCapturedList(!showCapturedList)}
                style={{ color: c.textMid }}
              >
                <span style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Captured Activations ({filteredActivations.length})
                </span>
                {showCapturedList ? <ChevronUp style={{ width: 14, height: 14 }} /> : <ChevronDown style={{ width: 14, height: 14 }} />}
              </button>

              {showCapturedList && (
                <div style={{ marginTop: '0.75rem' }}>
                  {filteredActivations.length === 0 ? (
                    <div style={{
                      padding: '1.25rem',
                      textAlign: 'center',
                      color: c.textSub,
                      fontSize: '0.8rem',
                      background: c.controlBg,
                      border: `1px dashed ${c.controlBorder}`,
                    }}>
                      No activations captured for this model
                    </div>
                  ) : (
                    <div className="da-activations-list">
                      {filteredActivations.map((activation) => {
                        const isPrimary = selectedActivation1 === activation.id
                        const isSecondary = selectedActivation2 === activation.id
                        const isSelected = isPrimary || isSecondary
                        return (
                          <button
                            key={activation.id}
                            onClick={() => {
                              if (mode === 'compare') {
                                if (selectedActivation1 === activation.id) {
                                  setSelectedActivation1(selectedActivation2)
                                  setSelectedActivation2('')
                                } else if (selectedActivation2 === activation.id) {
                                  setSelectedActivation2('')
                                } else if (!selectedActivation1) {
                                  setSelectedActivation1(activation.id)
                                } else if (!selectedActivation2) {
                                  setSelectedActivation2(activation.id)
                                } else {
                                  setSelectedActivation2(activation.id)
                                }
                              } else {
                                setSelectedActivation1(activation.id)
                                setSelectedActivation2('')
                              }
                            }}
                            className="da-activation-item"
                            style={{
                              background: isSelected ? c.accentBg : c.controlBg,
                              borderColor: isSelected ? c.accentText : c.controlBorder,
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                              {isPrimary && mode === 'compare' && (
                                <span style={{ fontSize: '0.6rem', fontWeight: 600, color: c.accentText, textTransform: 'uppercase', letterSpacing: '0.04em' }}>1st</span>
                              )}
                              {isSecondary && (
                                <span style={{ fontSize: '0.6rem', fontWeight: 600, color: c.accentText, textTransform: 'uppercase', letterSpacing: '0.04em' }}>2nd</span>
                              )}
                              <span style={{ fontSize: '0.75rem', color: c.text, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {activation.prompt}
                              </span>
                            </div>
                            <div style={{ fontSize: '0.65rem', color: c.textSub, marginTop: '0.2rem' }}>
                              {activation.num_tokens} tokens · {activation.num_layers} layers
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  )}

                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.375rem',
                    marginTop: '0.625rem',
                    padding: '0.4rem 0.5rem',
                    background: c.warningBg,
                    border: `1px solid ${c.warningBorder}`,
                    fontSize: '0.65rem',
                    color: c.warningText,
                    lineHeight: 1.4,
                  }}>
                    <AlertCircle style={{ width: 11, height: 11, flexShrink: 0, marginTop: 1 }} />
                    <span>Activations are stored temporarily and may be cleared periodically.</span>
                  </div>
                </div>
              )}
            </div>

            {/* Analysis Mode */}
            <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="da-section-label" style={{ color: c.textSub }}>Analysis Mode</div>
              <div style={{ display: 'flex', gap: '0.375rem' }}>
                {modes.map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => {
                      setMode(id)
                      if (id !== 'compare') setSelectedActivation2('')
                    }}
                    className="da-mode-btn"
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

            {/* Run Analysis */}
            <div className="da-section">
              <button
                className="da-run-btn"
                onClick={() => {
                  if (mode === 'classify') handleClassify()
                  else if (mode === 'compare') handleCompare()
                  else handleMinePatterns()
                }}
                disabled={loading || !selectedActivation1 || (mode === 'compare' && !selectedActivation2)}
                style={{
                  background: loading ? c.controlBg : (isLight ? '#1d1d1f' : '#ffffff'),
                  color: loading ? c.textSub : (isLight ? '#ffffff' : '#0a0a0a'),
                  borderColor: 'transparent',
                }}
              >
                {loading ? (
                  <>
                    <div className="da-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search style={{ width: 14, height: 14 }} />
                    {mode === 'classify' && 'Classify Heads'}
                    {mode === 'compare' && 'Compare Patterns'}
                    {mode === 'mine' && 'Mine Patterns'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel — 70% */}
        <div className="da-right" style={{ background: c.bg }}>
          <div className="da-right-scroll">
            {/* Classification Results */}
            {mode === 'classify' && classificationResult && (
              <div className="da-results">
                <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  <div style={{ marginBottom: '0.75rem' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 500, color: c.textMid }}>
                      Prompt: <span style={{ fontFamily: 'monospace', color: c.text }}>{classificationResult.prompt}</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: c.textSub, marginTop: '0.25rem' }}>
                      {classificationResult.num_layers} layers · {classificationResult.num_heads} heads per layer
                    </div>
                  </div>

                  <AttentionPatternGrid
                    classifications={classificationResult.classifications}
                    numLayers={classificationResult.num_layers}
                    numHeads={classificationResult.num_heads}
                    onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                    selectedHead={selectedHead}
                  />
                </div>

                <div className="da-section">
                  <HeadClassifier
                    classifications={classificationResult.classifications}
                    numLayers={classificationResult.num_layers}
                    numHeads={classificationResult.num_heads}
                    summary={classificationResult.summary}
                    onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                    selectedHead={selectedHead}
                  />
                </div>
              </div>
            )}

            {/* Comparison Results */}
            {mode === 'compare' && comparisonResult && (
              <div className="da-section">
                <HeadComparer
                  comparison={comparisonResult}
                  onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                />
              </div>
            )}

            {/* Mining Results */}
            {mode === 'mine' && miningResult && (
              <div className="da-results">
                {/* Top Heads */}
                <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  <div className="da-section-label" style={{ color: c.textSub }}>Top Important Heads</div>
                  <div className="da-heads-grid">
                    {miningResult.top_heads.slice(0, 20).map(([layer, head, importance]) => {
                      const headType = getHeadType(layer, head)
                      const borderColor = headType ? HEAD_TYPE_COLORS[headType] : c.controlBorder
                      return (
                        <button
                          key={`${layer}-${head}`}
                          onClick={() => setSelectedHead({ layer, head })}
                          className="da-head-item"
                          style={{ borderColor, background: c.controlBg }}
                          title={headType ? HEAD_TYPE_NAMES[headType] : undefined}
                        >
                          <div style={{ fontSize: '0.6rem', color: c.textSub, fontFamily: 'monospace' }}>
                            L{layer}H{head}
                          </div>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: c.text, marginTop: '0.125rem' }}>
                            {(importance * 100).toFixed(0)}%
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Head Type Distribution */}
                <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  <div className="da-section-label" style={{ color: c.textSub }}>Head Type Distribution</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {Object.entries(miningResult.classification_summary).map(([type, count]) => (
                      <div
                        key={type}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.375rem',
                          padding: '0.375rem 0.625rem',
                          backgroundColor: `${HEAD_TYPE_COLORS[type as HeadType]}15`,
                          border: `1px solid ${HEAD_TYPE_COLORS[type as HeadType]}`,
                          fontSize: '0.75rem',
                          color: c.textMid,
                        }}
                      >
                        <div style={{
                          width: '0.5rem',
                          height: '0.5rem',
                          backgroundColor: HEAD_TYPE_COLORS[type as HeadType],
                        }} />
                        <span>{HEAD_TYPE_NAMES[type as HeadType]}: {count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cross-Layer Patterns */}
                {miningResult.cross_layer_patterns.length > 0 && (
                  <div className="da-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
                    <div className="da-section-label" style={{ color: c.textSub }}>Detected Patterns</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                      {miningResult.cross_layer_patterns.map((pattern, idx) => (
                        <div key={idx} style={{
                          padding: '0.75rem',
                          background: c.controlBg,
                          border: `1px solid ${c.controlBorder}`,
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                            <span style={{ fontSize: '0.8rem', fontWeight: 500, color: c.text, textTransform: 'capitalize' }}>
                              {pattern.pattern_type.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.65rem', color: c.textSub }}>
                              Confidence: {(pattern.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: c.textMid, marginBottom: '0.5rem' }}>
                            {pattern.description}
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                            {pattern.involved_heads.slice(0, 8).map(([layer, head]) => (
                              <span key={`${layer}-${head}`} style={{
                                fontSize: '0.65rem',
                                padding: '0.15rem 0.375rem',
                                background: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)',
                                color: c.textMid,
                              }}>
                                L{layer}H{head}
                              </span>
                            ))}
                            {pattern.involved_heads.length > 8 && (
                              <button
                                onClick={() => setExpandedPatternHeads(expandedPatternHeads === idx ? null : idx)}
                                style={{
                                  fontSize: '0.65rem',
                                  padding: '0.15rem 0.375rem',
                                  background: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)',
                                  border: `1px solid ${c.controlBorder}`,
                                  color: c.textSub,
                                  cursor: 'pointer',
                                }}
                              >
                                +{pattern.involved_heads.length - 8} more
                              </button>
                            )}
                          </div>
                          {expandedPatternHeads === idx && (
                            <div style={{
                              marginTop: '0.625rem',
                              background: isLight ? 'rgba(0,0,0,0.02)' : 'rgba(0,0,0,0.3)',
                              border: `1px solid ${c.controlBorder}`,
                              padding: '0.625rem',
                            }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span style={{ fontSize: '0.7rem', color: c.textSub }}>
                                  All Involved Heads ({pattern.involved_heads.length})
                                </span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                                  <button
                                    onClick={() => copyHeadsList(pattern.involved_heads, idx)}
                                    style={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '0.25rem',
                                      padding: '0.2rem 0.4rem',
                                      background: c.controlBg,
                                      border: `1px solid ${c.controlBorder}`,
                                      color: c.textMid,
                                      fontSize: '0.65rem',
                                      cursor: 'pointer',
                                    }}
                                  >
                                    {copiedPatternIdx === idx ? (
                                      <><Check style={{ width: 10, height: 10 }} /> Copied</>
                                    ) : (
                                      <><Copy style={{ width: 10, height: 10 }} /> Copy</>
                                    )}
                                  </button>
                                  <button
                                    onClick={() => setExpandedPatternHeads(null)}
                                    style={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      padding: '0.2rem',
                                      background: 'transparent',
                                      border: 'none',
                                      color: c.textSub,
                                      cursor: 'pointer',
                                    }}
                                  >
                                    <X style={{ width: 12, height: 12 }} />
                                  </button>
                                </div>
                              </div>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', maxHeight: 180, overflowY: 'auto' }}>
                                {pattern.involved_heads.map(([layer, head]) => (
                                  <span key={`${layer}-${head}`} style={{
                                    fontSize: '0.65rem',
                                    padding: '0.15rem 0.375rem',
                                    background: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)',
                                    color: c.textMid,
                                  }}>
                                    L{layer}H{head}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Circuit Candidates */}
                {miningResult.circuit_candidates.length > 0 && (
                  <div className="da-section">
                    <div className="da-section-label" style={{ color: c.circuitText }}>Potential Circuits</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                      {miningResult.circuit_candidates.map((circuit, idx) => (
                        <div key={idx} style={{
                          padding: '0.75rem',
                          background: c.circuitBg,
                          border: `1px solid ${c.circuitBorder}`,
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                            <span style={{ fontSize: '0.8rem', fontWeight: 500, color: c.circuitText, textTransform: 'capitalize' }}>
                              {circuit.type.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.65rem', color: c.textSub }}>
                              Confidence: {(circuit.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: c.textMid }}>
                            {circuit.description}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty State */}
            {!classificationResult && !comparisonResult && !miningResult && (
              <div className="da-placeholder">
                <Search style={{ width: 36, height: 36, color: c.textSub, opacity: 0.3, marginBottom: '0.75rem' }} />
                <p style={{ color: c.textSub, fontSize: '0.8rem', margin: 0, maxWidth: 360, lineHeight: 1.5 }}>
                  {mode === 'classify' && 'Capture activations and classify heads to identify attention head types.'}
                  {mode === 'compare' && 'Select two captured activations and compare to see how attention differs.'}
                  {mode === 'mine' && 'Capture activations and mine patterns to discover cross-layer patterns and circuits.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .da-page {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .da-layout {
          display: flex;
          flex: 1;
          margin-top: 76px;
          overflow: hidden;
        }

        .da-left {
          width: 30%;
          min-width: 300px;
          max-width: 420px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .da-left-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .da-right {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .da-right-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .da-section {
          padding: 1rem 1.25rem;
        }

        .da-section-label {
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 0.625rem;
        }

        .da-textarea {
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

        .da-textarea:focus {
          border-color: rgba(59, 130, 246, 0.4) !important;
        }

        .da-capture-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.5rem 0.75rem;
          border: 1px solid;
          font-size: 0.8rem;
          font-weight: 500;
          cursor: pointer;
          transition: opacity 0.15s;
        }

        .da-capture-btn:hover:not(:disabled) {
          opacity: 0.8;
        }

        .da-capture-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .da-toggle-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: transparent;
          border: none;
          cursor: pointer;
          padding: 0;
        }

        .da-activations-list {
          display: flex;
          flex-direction: column;
          gap: 0.375rem;
          max-height: 220px;
          overflow-y: auto;
        }

        .da-activation-item {
          width: 100%;
          text-align: left;
          padding: 0.5rem 0.625rem;
          border: 1px solid;
          cursor: pointer;
          transition: all 0.15s;
        }

        .da-activation-item:hover {
          opacity: 0.85;
        }

        .da-mode-btn {
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
        }

        .da-mode-btn:hover {
          opacity: 0.85;
        }

        .da-run-btn {
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

        .da-run-btn:hover:not(:disabled) {
          opacity: 0.85;
        }

        .da-run-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .da-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid;
          border-radius: 50%;
          animation: da-spin 0.8s linear infinite;
        }

        @keyframes da-spin {
          to { transform: rotate(360deg); }
        }

        .da-spinning {
          animation: da-spin 1s linear infinite;
        }

        .da-results {
          display: flex;
          flex-direction: column;
        }

        .da-heads-grid {
          display: grid;
          grid-template-columns: repeat(10, 1fr);
          gap: 0.375rem;
        }

        .da-head-item {
          border: 2px solid;
          padding: 0.375rem 0.25rem;
          cursor: pointer;
          transition: all 0.15s;
          text-align: center;
        }

        .da-head-item:hover {
          opacity: 0.8;
        }

        .da-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 4rem 2rem;
          text-align: center;
          height: 100%;
          min-height: 400px;
        }

        .da-error {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 0.875rem;
          border: 1px solid;
          margin: 0.75rem 1.25rem 0;
        }

        @media (max-width: 1000px) {
          .da-layout {
            flex-wrap: wrap;
          }
          .da-left {
            width: 100%;
            max-width: 100%;
            max-height: 40vh;
            border-right: none !important;
            border-bottom: 1px solid;
          }
          .da-right {
            width: 100%;
          }
          .da-heads-grid {
            grid-template-columns: repeat(5, 1fr);
          }
        }

        @media (max-width: 600px) {
          .da-heads-grid {
            grid-template-columns: repeat(4, 1fr);
          }
          .da-mode-btn span {
            display: none;
          }
        }
      `}</style>
    </div>
  )
}
