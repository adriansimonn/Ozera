import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { NavBar } from '../components/common/NavBar'
import { HeadClassifier, AttentionPatternGrid, HeadComparer } from '../components/analysis'
import { apiClient } from '../api/client'
import { EmptyState } from '../components/common/EmptyState'
import { Spinner } from '../components/common/Spinner'
import { VAR } from '../styles/appTheme'
import { Dropdown } from '../components/common/Dropdown'
import type {
  ClassifyHeadsResponse,
  CompareAttentionResponse,
  MinePatternResponse,
  HeadType,
} from '../types/analysis'
import type { CapturedActivationSummary, PatchingModelInfo } from '../types/patching'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../types/analysis'
import {
  
  
  
  AlertCircle,
  
  
  
  
  Copy,
  Check,
  X } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useAssistantPage } from '../hooks/useAssistantPage'
import { analysisBridge } from '../components/assistant/pages/analysis'

interface AnalysisPageProps {
  onShowPurchaseCredits?: () => void
}

type AnalysisMode = 'classify' | 'compare' | 'mine'

export default function DefaultAnalysisPage({
  onShowPurchaseCredits,
}: AnalysisPageProps) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

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
      if (modelList.length > 0) setSelectedModel((current) => current || modelList[0].model_id)
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

  // The assistant: what's on screen, set_inputs, and its classify, compare and mine runs' results
  useAssistantPage('analysis', analysisBridge({
    mode,
    setMode,
    model: selectedModel,
    setModel: (model) => {
      setSelectedModel(model)
      setSelectedActivation1('')
      setSelectedActivation2('')
    },
    setModelOnly: setSelectedModel,
    capturePrompt,
    setCapturePrompt,
    captured: capturedActivations,
    refreshCaptured: loadCapturedActivations,
    selected: [selectedActivation1, selectedActivation2],
    setSelected: (first, second) => {
      setSelectedActivation1(first)
      setSelectedActivation2(second)
    },
    busy: loading || capturing,
    classification: classificationResult,
    setClassification: setClassificationResult,
    comparison: comparisonResult,
    setComparison: setComparisonResult,
    mining: miningResult,
    setMining: setMiningResult,
  }))

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

  // The page's colors, by the app's custom properties (styles/app.css)
  const c = {
    text: VAR.fg,
    textMid: VAR.fg,
    textSub: VAR.fgDim,
    controlBg: 'transparent',
    controlBorder: VAR.rule,
  }

  const modes: { id: AnalysisMode; label: string }[] = [
    { id: 'classify', label: 'Classify Heads' },
    { id: 'compare', label: 'Compare Prompts' },
    { id: 'mine', label: 'Mine Patterns' },
  ]

  return (
    <div className="da-page oz-page">
      <NavBar />

      <div className="da-layout oz-page__body">
        {/* Left Panel — 30% */}
        <div className="da-left oz-side">
          <div className="oz-scroll oz-steps">
            {/* Error banner */}
            {error && (
              <div className="da-error oz-note oz-note--error" role="alert">
                <AlertCircle aria-hidden="true" />
                <span>{error}</span>
                <button className="oz-link" onClick={() => setError(null)}>
                  Dismiss
                </button>
              </div>
            )}

            {/* Model section */}
            <div className="oz-section">
              <h2 className="oz-title">Model</h2>
              <div data-assist="analysis.model">
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
            </div>

            {/* Capture Activations */}
            <div className="oz-section">
              <h2 className="oz-title">Capture activations</h2>
              <textarea
                data-assist="analysis.prompt"
                value={capturePrompt}
                onChange={(e) => setCapturePrompt(e.target.value)}
                placeholder="Enter a prompt to capture activations..."
                rows={3}
                disabled={capturing}
                className="oz-textarea"
              />
              <button
                onClick={handleCaptureActivations}
                data-assist="analysis.capture_button"
                disabled={capturing || !capturePrompt.trim()}
                className="oz-btn oz-btn--ghost oz-btn--block da-capture-btn"
              >
                {capturing ? (
                  <>
                    <Spinner />
                    Capturing...
                  </>
                ) : (
                  'Capture'
                )}
              </button>
            </div>

            {/* Captured Activations */}
            <div className="oz-section">
              <button
                className="da-toggle-btn"
                data-assist="analysis.activation"
                aria-expanded={showCapturedList}
                onClick={() => setShowCapturedList(!showCapturedList)}
              >
                <h2 className="oz-title">
                  Captured activations
                  <span className="oz-title__end">{filteredActivations.length}</span>
                </h2>
                <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
                  <path d="M1 1l4 4 4-4" />
                </svg>
              </button>

              {showCapturedList && (
                <div style={{ marginTop: '0.9rem' }}>
                  {filteredActivations.length === 0 ? (
                    <p className="oz-hint">No activations captured for this model</p>
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
                            className={`da-activation-item${isSelected ? ' is-active' : ''}`}
                            aria-pressed={isSelected}
                          >
                            <div className="da-activation-prompt">
                              {isPrimary && mode === 'compare' && <span className="oz-tag oz-tag--solid">1st</span>}
                              {isSecondary && <span className="oz-tag oz-tag--solid">2nd</span>}
                              <span>{activation.prompt}</span>
                            </div>
                            <div className="da-activation-meta">
                              {activation.num_tokens} tokens · {activation.num_layers} layers
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  )}

                  <p className="oz-hint" style={{ marginTop: '0.75rem' }}>
                    Activations are stored temporarily and may be cleared periodically.
                  </p>
                </div>
              )}
            </div>

            {/* Analysis Mode */}
            <div className="oz-section">
              <h2 className="oz-title">Analysis mode</h2>
              <div className="oz-chips da-modes" role="radiogroup" aria-label="Analysis mode" data-assist="analysis.mode">
                {modes.map(({ id, label }) => (
                  <button
                    key={id}
                    role="radio"
                    aria-checked={mode === id}
                    onClick={() => {
                      setMode(id)
                      if (id !== 'compare') setSelectedActivation2('')
                    }}
                    className="oz-chip"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Run Analysis */}
            <div className="oz-section da-run">
              <button
                className={`oz-btn oz-btn--block${loading ? '' : ' oz-btn--arrow'}`}
                data-assist="analysis.run_button"
                onClick={() => {
                  if (mode === 'classify') handleClassify()
                  else if (mode === 'compare') handleCompare()
                  else handleMinePatterns()
                }}
                disabled={loading || !selectedActivation1 || (mode === 'compare' && !selectedActivation2)}
              >
                {loading ? (
                  <>
                    <Spinner />
                    Analyzing...
                  </>
                ) : (
                  <>
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
        <div className="da-right oz-main">
          <div className="oz-scroll">
            {/* Classification Results */}
            {mode === 'classify' && classificationResult && (
              <div className="da-results">
                <div className="oz-section">
                  <div className="da-run-meta">
                    <span className="oz-mono">{classificationResult.prompt}</span>
                    <span className="oz-hint">
                      {classificationResult.num_layers} layers · {classificationResult.num_heads} heads per layer
                    </span>
                  </div>

                  <AttentionPatternGrid
                    classifications={classificationResult.classifications}
                    numLayers={classificationResult.num_layers}
                    numHeads={classificationResult.num_heads}
                    onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                    selectedHead={selectedHead}
                  />
                </div>

                <div className="oz-section da-last">
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
              <div className="oz-section da-last">
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
                <div className="oz-section">
                  <h3 className="da-heading">Top important heads</h3>
                  <div className="da-heads-grid">
                    {miningResult.top_heads.slice(0, 20).map(([layer, head, importance]) => {
                      const headType = getHeadType(layer, head)
                      const borderColor = headType ? HEAD_TYPE_COLORS[headType] : c.controlBorder
                      return (
                        <button
                          key={`${layer}-${head}`}
                          onClick={() => setSelectedHead({ layer, head })}
                          className="da-head-item"
                          style={{ borderTopColor: borderColor }}
                          title={headType ? HEAD_TYPE_NAMES[headType] : undefined}
                        >
                          <div style={{ fontSize: '0.78rem', color: c.textSub, fontFamily: 'var(--oz-mono)' }}>
                            L{layer}H{head}
                          </div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 500, color: c.text, marginTop: '0.125rem' }}>
                            {(importance * 100).toFixed(0)}%
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Head Type Distribution */}
                <div className="oz-section">
                  <h3 className="da-heading">Head type distribution</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {Object.entries(miningResult.classification_summary).map(([type, count]) => (
                      <div key={type} className="oz-tag da-type-tag">
                        <i style={{ backgroundColor: HEAD_TYPE_COLORS[type as HeadType] }} />
                        <span>{HEAD_TYPE_NAMES[type as HeadType]}</span>
                        <span className="da-type-count">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cross-Layer Patterns */}
                {miningResult.cross_layer_patterns.length > 0 && (
                  <div className="oz-section">
                    <h3 className="da-heading">Detected patterns</h3>
                    <div className="da-ruled">
                      {miningResult.cross_layer_patterns.map((pattern, idx) => (
                        <div key={idx} className="da-ruled-item">
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                            <span style={{ fontSize: '0.875rem', fontWeight: 500, color: c.text, textTransform: 'capitalize' }}>
                              {pattern.pattern_type.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.8rem', color: c.textSub }}>
                              Confidence: {(pattern.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ fontSize: '0.85rem', color: c.textMid, marginBottom: '0.5rem' }}>
                            {pattern.description}
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                            {pattern.involved_heads.slice(0, 8).map(([layer, head]) => (
                              <span key={`${layer}-${head}`} style={{
                                fontSize: '0.8rem',
                                padding: '0.15rem 0.375rem',
                                background: VAR.fillStrong,
                                color: c.textMid,
                              }}>
                                L{layer}H{head}
                              </span>
                            ))}
                            {pattern.involved_heads.length > 8 && (
                              <button
                                onClick={() => setExpandedPatternHeads(expandedPatternHeads === idx ? null : idx)}
                                style={{
                                  fontSize: '0.8rem',
                                  padding: '0.15rem 0.375rem',
                                  background: VAR.fillStrong,
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
                              background: 'transparent',
                              border: `1px solid ${c.controlBorder}`,
                              padding: '0.625rem',
                            }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span style={{ fontSize: '0.82rem', color: c.textSub }}>
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
                                      fontSize: '0.8rem',
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
                                    fontSize: '0.8rem',
                                    padding: '0.15rem 0.375rem',
                                    background: VAR.fillStrong,
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
                  <div className="oz-section da-last">
                    <h3 className="da-heading">Potential circuits</h3>
                    <div className="da-ruled">
                      {miningResult.circuit_candidates.map((circuit, idx) => (
                        <div key={idx} className="da-ruled-item">
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                            <span style={{ fontSize: '0.875rem', fontWeight: 500, color: c.text, textTransform: 'capitalize' }}>
                              {circuit.type.replace(/_/g, ' ')}
                            </span>
                            <span style={{ fontSize: '0.8rem', color: c.textSub }}>
                              Confidence: {(circuit.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ fontSize: '0.85rem', color: c.textMid }}>
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
              <EmptyState glyph="attention" fill className="da-placeholder">
                {mode === 'classify' && 'Capture activations and classify heads to identify attention head types.'}
                {mode === 'compare' && 'Select two captured activations and compare to see how attention differs.'}
                {mode === 'mine' && 'Capture activations and mine patterns to discover cross-layer patterns and circuits.'}
              </EmptyState>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .da-page .da-error {
          margin: 1rem var(--oz-gutter) 0;
        }

        .da-page .da-capture-btn {
          margin-top: 6px;
        }

        /* A section's title that also opens and closes it */
        .da-toggle-btn {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          width: 100%;
          padding: 0;
          text-align: left;
          cursor: pointer;
        }

        .da-toggle-btn .oz-title {
          flex: 1;
          margin: 0;
          transition: color 0.2s ease;
        }

        .da-toggle-btn:hover .oz-title {
          color: var(--oz-mark);
        }

        /* The captures: a ruled list, the chosen ones marked */
        .da-activations-list {
          display: flex;
          flex-direction: column;
          max-height: 260px;
          overflow-y: auto;
        }

        .da-activation-item {
          position: relative;
          width: 100%;
          padding: 0.65rem 0;
          border-top: 1px solid var(--oz-rule);
          color: var(--oz-fg-dim);
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }

        .da-activation-item:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .da-activation-item:hover,
        .da-activation-item.is-active {
          color: var(--oz-fg);
        }

        .da-activation-item.is-active::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 100%;
          height: 1px;
          background: var(--oz-mark);
        }

        .da-activation-prompt {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          font-weight: 500;
        }

        .da-activation-prompt > span:last-child {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .da-activation-meta {
          margin-top: 0.15rem;
          font-size: 0.8rem;
          font-variant-numeric: tabular-nums;
        }

        .da-page .da-modes .oz-chip {
          flex: 1 1 auto;
        }

        .da-page .da-run,
        .da-page .da-last {
          border-bottom: none;
        }

        .da-results {
          display: flex;
          flex-direction: column;
        }

        .da-run-meta {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          justify-content: space-between;
          gap: 0.35rem 1.5rem;
          margin-bottom: 1.25rem;
          font-size: 0.9rem;
        }

        /* A result's title */
        .da-heading {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        .da-heads-grid {
          display: grid;
          grid-template-columns: repeat(10, 1fr);
          gap: 6px;
        }

        /* A head: a cell under a bar of its type's color */
        .da-head-item {
          padding: 0.5rem 0.25rem 0.55rem;
          border: 1px solid var(--oz-rule);
          border-top: 2px solid var(--oz-rule-strong);
          text-align: center;
          cursor: pointer;
          transition: background 0.15s ease;
        }

        .da-head-item:hover {
          background: var(--oz-fill);
        }

        .da-page .da-type-tag {
          gap: 0.45rem;
          padding: 0.3rem 0.6rem;
          font-size: 0.85rem;
        }

        .da-type-tag i {
          width: 8px;
          height: 8px;
        }

        .da-type-count {
          color: var(--oz-fg-dim);
          font-variant-numeric: tabular-nums;
        }

        .da-ruled-item {
          padding: 0.9rem 0 1rem;
          border-top: 1px solid var(--oz-rule);
        }

        .da-ruled-item:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .da-page .da-placeholder {
          min-height: calc(125vh - var(--oz-nav-h) - 2px);
        }

        @media (max-width: 1100px) {
          .da-page.oz-page { height: auto; overflow: visible; }
          .da-layout.oz-page__body { flex-direction: column; }
          .da-left.oz-side { width: 100%; max-width: 100%; border-right: none; border-bottom: 1px solid var(--oz-rule); }
          .da-heads-grid { grid-template-columns: repeat(5, 1fr); }
        }
      `}</style>
    </div>
  )
}
