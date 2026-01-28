/**
 * Attention Pattern Analysis Page - Interactive interface for analyzing
 * attention head types, patterns, and comparing across prompts.
 */

import { useState, useEffect, useCallback } from 'react'
import { NavBar } from '../components/common/NavBar'
import { HeadClassifier, AttentionPatternGrid, HeadComparer } from '../components/analysis'
import { apiClient } from '../api/client'
import type {
  ClassifyHeadsResponse,
  CompareAttentionResponse,
  MinePatternResponse,
  HeadType,
} from '../types/analysis'
import type { CapturedActivationSummary, PatchingModelInfo } from '../types/patching'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../types/analysis'
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

interface AnalysisPageProps {
  onShowLogin: () => void
  onShowSignup: () => void
  onShowPurchaseCredits?: () => void
}

type AnalysisMode = 'classify' | 'compare' | 'mine'

export default function AnalysisPage({
  onShowLogin,
  onShowSignup,
  onShowPurchaseCredits,
}: AnalysisPageProps) {
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

  // Load models and activations on mount
  useEffect(() => {
    loadModels()
    loadCapturedActivations()
  }, [])

  const loadModels = async () => {
    try {
      const modelList = await apiClient.getPatchingModels()
      setModels(modelList)
      if (modelList.length > 0) {
        setSelectedModel(modelList[0].model_id)
      }
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

  // Filter activations by selected model
  const filteredActivations = capturedActivations.filter(
    (a) => a.model_id === selectedModel
  )

  // Get head type from classification result (if available for same activation)
  const getHeadType = (layer: number, head: number): HeadType | null => {
    if (!classificationResult || classificationResult.activation_id !== selectedActivation1) {
      return null
    }
    const classification = classificationResult.classifications.find(
      c => c.layer === layer && c.head === head
    )
    return classification?.primary_type || null
  }

  // Copy heads list to clipboard
  const copyHeadsList = async (heads: [number, number][], patternIdx: number) => {
    const text = heads.map(([l, h]) => `L${l}H${h}`).join(', ')
    await navigator.clipboard.writeText(text)
    setCopiedPatternIdx(patternIdx)
    setTimeout(() => setCopiedPatternIdx(null), 2000)
  }

  // Capture new activations
  const handleCaptureActivations = async () => {
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

      // Refresh the activations list
      await loadCapturedActivations()

      // Select the new activation
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

  // Run classification
  const handleClassify = async () => {
    if (!selectedActivation1) {
      setError('Please select captured activations to analyze')
      return
    }

    try {
      setLoading(true)
      setError(null)
      setClassificationResult(null)

      const result = await apiClient.classifyHeads({
        activation_id: selectedActivation1,
      })

      setClassificationResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify heads')
    } finally {
      setLoading(false)
    }
  }

  // Run comparison
  const handleCompare = async () => {
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
      setError(err instanceof Error ? err.message : 'Failed to compare attention patterns')
    } finally {
      setLoading(false)
    }
  }

  // Run pattern mining
  const handleMinePatterns = async () => {
    if (!selectedActivation1) {
      setError('Please select captured activations to analyze')
      return
    }

    try {
      setLoading(true)
      setError(null)
      setMiningResult(null)

      const result = await apiClient.minePatterns({
        activation_id: selectedActivation1,
      })

      setMiningResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to mine patterns')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="analysis-page">
      <NavBar onShowLogin={onShowLogin} onShowSignup={onShowSignup} />

      <div className="analysis-content">
        {/* Header */}
        <div className="analysis-header">
          <h1>Attention Pattern Analysis</h1>
          <p className="header-description">
            Classify attention heads, discover patterns, and compare across prompts
          </p>
        </div>

        {/* Mode Selection */}
        <div className="mode-tabs">
          {[
            { id: 'classify', label: 'Classify Heads', icon: Search },
            { id: 'compare', label: 'Compare Prompts', icon: GitCompare },
            { id: 'mine', label: 'Mine Patterns', icon: Layers },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id as AnalysisMode)}
              className={`mode-tab ${mode === id ? 'active' : ''}`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-banner">
            <AlertCircle className="error-icon" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="dismiss-btn">Dismiss</button>
          </div>
        )}

        <div className="analysis-grid">
          {/* Left Panel - Configuration */}
          <div className="left-column">
            {/* Model Selection */}
            <div className="section">
              <h2>Model</h2>
              <select
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value)
                  setSelectedActivation1('')
                  setSelectedActivation2('')
                }}
                className="model-select"
              >
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.display_name} ({model.num_layers}L, {model.num_heads}H)
                  </option>
                ))}
              </select>
            </div>

            {/* Capture New Activations */}
            <div className="section">
              <h2>Capture Activations</h2>
              <div className="capture-form">
                <textarea
                  value={capturePrompt}
                  onChange={(e) => setCapturePrompt(e.target.value)}
                  placeholder="Enter a prompt to capture activations..."
                  rows={3}
                  className="capture-textarea"
                />
                <button
                  onClick={handleCaptureActivations}
                  disabled={capturing || !capturePrompt.trim()}
                  className="capture-btn"
                >
                  {capturing ? (
                    <RefreshCw size={16} className="spinning" />
                  ) : (
                    <Zap size={16} />
                  )}
                  {capturing ? 'Capturing...' : 'Capture'}
                </button>
              </div>
            </div>

            {/* Captured Activations List */}
            <div className="section">
              <button
                onClick={() => setShowCapturedList(!showCapturedList)}
                className="section-toggle"
              >
                <h2>Captured Activations ({filteredActivations.length})</h2>
                {showCapturedList ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showCapturedList && (
                <div className="activations-list">
                  {filteredActivations.length === 0 ? (
                    <div className="empty-state">
                      No activations captured for this model
                    </div>
                  ) : (
                    filteredActivations.map((activation) => (
                      <button
                        key={activation.id}
                        onClick={() => {
                          if (mode === 'compare' && selectedActivation1 && selectedActivation1 !== activation.id) {
                            setSelectedActivation2(activation.id)
                          } else {
                            setSelectedActivation1(activation.id)
                            if (mode !== 'compare') {
                              setSelectedActivation2('')
                            }
                          }
                        }}
                        className={`activation-item ${
                          selectedActivation1 === activation.id
                            ? 'selected-primary'
                            : selectedActivation2 === activation.id
                              ? 'selected-secondary'
                              : ''
                        }`}
                      >
                        <div className="activation-prompt">
                          {activation.prompt}
                        </div>
                        <div className="activation-meta">
                          {activation.num_tokens} tokens • {activation.num_layers} layers
                        </div>
                      </button>
                    ))
                  )}
                </div>
              )}

              {mode === 'compare' && (
                <div className="selection-legend">
                  <span className="legend-primary">●</span> First prompt &nbsp;
                  <span className="legend-secondary">●</span> Second prompt
                </div>
              )}

              <div className="storage-warning">
                <AlertCircle size={12} />
                <span>Activations are stored temporarily and may be cleared periodically.</span>
              </div>
            </div>

            {/* Run Analysis Button */}
            <button
              onClick={() => {
                if (mode === 'classify') handleClassify()
                else if (mode === 'compare') handleCompare()
                else handleMinePatterns()
              }}
              disabled={loading || !selectedActivation1 || (mode === 'compare' && !selectedActivation2)}
              className="run-analysis-btn"
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="spinning" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={18} />
                  {mode === 'classify' && 'Classify Heads'}
                  {mode === 'compare' && 'Compare Patterns'}
                  {mode === 'mine' && 'Mine Patterns'}
                </>
              )}
            </button>
          </div>

          {/* Right Panel - Results */}
          <div className="right-column">
            {/* Classification Results */}
            {mode === 'classify' && classificationResult && (
              <div className="results-container">
                <div className="result-header">
                  <h3>
                    Prompt: <span className="prompt-text">{classificationResult.prompt}</span>
                  </h3>
                  <div className="result-meta">
                    {classificationResult.num_layers} layers • {classificationResult.num_heads} heads per layer
                  </div>
                </div>

                <AttentionPatternGrid
                  classifications={classificationResult.classifications}
                  numLayers={classificationResult.num_layers}
                  numHeads={classificationResult.num_heads}
                  onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                  selectedHead={selectedHead}
                />

                <HeadClassifier
                  classifications={classificationResult.classifications}
                  numLayers={classificationResult.num_layers}
                  numHeads={classificationResult.num_heads}
                  summary={classificationResult.summary}
                  onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
                  selectedHead={selectedHead}
                />
              </div>
            )}

            {/* Comparison Results */}
            {mode === 'compare' && comparisonResult && (
              <HeadComparer
                comparison={comparisonResult}
                onHeadSelect={(layer, head) => setSelectedHead({ layer, head })}
              />
            )}

            {/* Mining Results */}
            {mode === 'mine' && miningResult && (
              <div className="results-container">
                {/* Top Heads */}
                <div className="result-section">
                  <h3>Top Important Heads</h3>
                  <div className="top-heads-grid-compact">
                    {miningResult.top_heads.slice(0, 20).map(([layer, head, importance]) => {
                      const headType = getHeadType(layer, head)
                      const borderColor = headType ? HEAD_TYPE_COLORS[headType] : 'rgba(255, 255, 255, 0.15)'
                      return (
                        <button
                          key={`${layer}-${head}`}
                          onClick={() => setSelectedHead({ layer, head })}
                          className="top-head-item-compact"
                          style={{ borderColor }}
                          title={headType ? `${HEAD_TYPE_NAMES[headType]}` : undefined}
                        >
                          <div className="head-label-compact">
                            L{layer}H{head}
                          </div>
                          <div className="head-importance-compact">
                            {(importance * 100).toFixed(0)}%
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Classification Summary */}
                <div className="result-section">
                  <h3>Head Type Distribution</h3>
                  <div className="type-distribution">
                    {Object.entries(miningResult.classification_summary).map(([type, count]) => (
                      <div
                        key={type}
                        className="type-badge"
                        style={{
                          backgroundColor: `${HEAD_TYPE_COLORS[type as HeadType]}20`,
                          borderColor: HEAD_TYPE_COLORS[type as HeadType],
                        }}
                      >
                        <div
                          className="type-dot"
                          style={{ backgroundColor: HEAD_TYPE_COLORS[type as HeadType] }}
                        />
                        <span>
                          {HEAD_TYPE_NAMES[type as HeadType]}: {count}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cross-Layer Patterns */}
                {miningResult.cross_layer_patterns.length > 0 && (
                  <div className="result-section">
                    <h3>Detected Patterns</h3>
                    <div className="patterns-list">
                      {miningResult.cross_layer_patterns.map((pattern, idx) => (
                        <div key={idx} className="pattern-item">
                          <div className="pattern-header">
                            <span className="pattern-type">
                              {pattern.pattern_type.replace(/_/g, ' ')}
                            </span>
                            <span className="pattern-confidence">
                              Confidence: {(pattern.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="pattern-description">
                            {pattern.description}
                          </div>
                          <div className="pattern-heads">
                            {pattern.involved_heads.slice(0, 8).map(([layer, head]) => (
                              <span key={`${layer}-${head}`} className="head-tag">
                                L{layer}H{head}
                              </span>
                            ))}
                            {pattern.involved_heads.length > 8 && (
                              <button
                                className="head-tag more-btn"
                                onClick={() => setExpandedPatternHeads(expandedPatternHeads === idx ? null : idx)}
                              >
                                +{pattern.involved_heads.length - 8} more
                              </button>
                            )}
                          </div>
                          {/* Expanded heads popup */}
                          {expandedPatternHeads === idx && (
                            <div className="expanded-heads-popup">
                              <div className="expanded-heads-header">
                                <span>All Involved Heads ({pattern.involved_heads.length})</span>
                                <div className="expanded-heads-actions">
                                  <button
                                    className="copy-heads-btn"
                                    onClick={() => copyHeadsList(pattern.involved_heads, idx)}
                                  >
                                    {copiedPatternIdx === idx ? (
                                      <><Check size={12} /> Copied</>
                                    ) : (
                                      <><Copy size={12} /> Copy</>
                                    )}
                                  </button>
                                  <button
                                    className="close-popup-btn"
                                    onClick={() => setExpandedPatternHeads(null)}
                                  >
                                    <X size={14} />
                                  </button>
                                </div>
                              </div>
                              <div className="expanded-heads-list">
                                {pattern.involved_heads.map(([layer, head]) => (
                                  <span key={`${layer}-${head}`} className="head-tag">
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
                  <div className="result-section circuits">
                    <h3>Potential Circuits</h3>
                    <div className="circuits-list">
                      {miningResult.circuit_candidates.map((circuit, idx) => (
                        <div key={idx} className="circuit-item">
                          <div className="circuit-header">
                            <span className="circuit-type">
                              {circuit.type.replace(/_/g, ' ')}
                            </span>
                            <span className="circuit-confidence">
                              Confidence: {(circuit.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="circuit-description">
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
              <div className="results-placeholder">
                <div className="placeholder-content">
                  <Search className="placeholder-icon" />
                  <h3>No Analysis Results</h3>
                  <p>
                    {mode === 'classify' && 'Capture activations and click "Classify Heads" to identify attention head types.'}
                    {mode === 'compare' && 'Select two captured activations and click "Compare Patterns" to see how attention differs.'}
                    {mode === 'mine' && 'Capture activations and click "Mine Patterns" to discover cross-layer patterns and circuits.'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .analysis-page {
          min-height: 100vh;
          padding-top: 70px;
        }

        .analysis-content {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
        }

        .analysis-header {
          margin-bottom: 2rem;
        }

        .analysis-header h1 {
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

        .mode-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
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
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.4);
          color: rgba(255, 255, 255, 0.95);
        }

        .error-banner {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.9);
          margin-bottom: 1.5rem;
          font-size: 0.875rem;
        }

        .error-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
        }

        .dismiss-btn {
          margin-left: auto;
          padding: 0.25rem 0.75rem;
          background: transparent;
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.8);
          font-size: 0.8rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .dismiss-btn:hover {
          background: rgba(239, 68, 68, 0.1);
        }

        .analysis-grid {
          display: grid;
          grid-template-columns: 350px 1fr;
          gap: 1.5rem;
          align-items: start;
        }

        .left-column {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .section {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
        }

        .section h2 {
          margin: 0 0 0.75rem 0;
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.7);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .model-select {
          width: 100%;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.875rem;
          cursor: pointer;
        }

        .model-select:focus {
          outline: none;
          border-color: rgba(255, 255, 255, 0.4);
        }

        .capture-form {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .capture-textarea {
          width: 100%;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.875rem;
          font-family: inherit;
          resize: vertical;
          min-height: 80px;
        }

        .capture-textarea::placeholder {
          color: rgba(255, 255, 255, 0.35);
        }

        .capture-textarea:focus {
          outline: none;
          border-color: rgba(255, 255, 255, 0.4);
        }

        .capture-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .capture-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.35);
        }

        .capture-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .spinning {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .section-toggle {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0;
          background: transparent;
          border: none;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.7);
        }

        .section-toggle h2 {
          margin: 0;
        }

        .activations-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          max-height: 256px;
          overflow-y: auto;
          margin-top: 0.75rem;
        }

        .empty-state {
          padding: 1.5rem;
          text-align: center;
          color: rgba(255, 255, 255, 0.4);
          font-size: 0.8rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .activation-item {
          width: 100%;
          text-align: left;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          cursor: pointer;
          transition: all 0.2s;
        }

        .activation-item:hover {
          border-color: rgba(255, 255, 255, 0.25);
        }

        .activation-item.selected-primary {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.5);
        }

        .activation-item.selected-secondary {
          background: rgba(168, 85, 247, 0.15);
          border-color: rgba(168, 85, 247, 0.5);
        }

        .activation-prompt {
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.8);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .activation-meta {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
          margin-top: 0.25rem;
        }

        .selection-legend {
          margin-top: 0.75rem;
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .legend-primary {
          color: rgba(255, 255, 255, 0.9);
        }

        .legend-secondary {
          color: rgba(168, 85, 247, 0.9);
        }

        .storage-warning {
          display: flex;
          align-items: flex-start;
          gap: 0.5rem;
          margin-top: 0.75rem;
          padding: 0.5rem 0.625rem;
          background: rgba(245, 158, 11, 0.08);
          border: 1px solid rgba(245, 158, 11, 0.2);
          font-size: 0.7rem;
          color: rgba(245, 158, 11, 0.8);
          line-height: 1.4;
        }

        .storage-warning svg {
          flex-shrink: 0;
          margin-top: 1px;
        }

        .run-analysis-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          padding: 0.875rem 1.5rem;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.25);
          color: rgba(255, 255, 255, 0.95);
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.025em;
        }

        .run-analysis-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.4);
        }

        .run-analysis-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .right-column {
          min-height: 400px;
        }

        .results-container {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .result-header {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
        }

        .result-header h3 {
          margin: 0;
          font-size: 0.875rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.7);
        }

        .prompt-text {
          font-family: monospace;
          color: rgba(255, 255, 255, 0.9);
        }

        .result-meta {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.4);
          margin-top: 0.375rem;
        }

        .result-section {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
        }

        .result-section.circuits {
          border-color: rgba(168, 85, 247, 0.3);
        }

        .result-section h3 {
          margin: 0 0 1rem 0;
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.7);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .result-section.circuits h3 {
          color: rgba(168, 85, 247, 0.9);
        }

        .top-heads-grid-compact {
          display: grid;
          grid-template-columns: repeat(10, 1fr);
          gap: 0.375rem;
        }

        .top-head-item-compact {
          background: rgba(0, 0, 0, 0.3);
          border: 2px solid rgba(255, 255, 255, 0.15);
          padding: 0.375rem 0.25rem;
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }

        .top-head-item-compact:hover {
          background: rgba(255, 255, 255, 0.08);
        }

        .head-label-compact {
          font-size: 0.6rem;
          color: rgba(255, 255, 255, 0.6);
          font-family: monospace;
        }

        .head-importance-compact {
          font-size: 0.75rem;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.9);
          margin-top: 0.125rem;
        }

        .type-distribution {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem;
        }

        .type-badge {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 0.75rem;
          border: 1px solid;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.8);
        }

        .type-dot {
          width: 0.75rem;
          height: 0.75rem;
        }

        .patterns-list,
        .circuits-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .pattern-item {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 0.875rem;
        }

        .circuit-item {
          background: rgba(168, 85, 247, 0.1);
          border: 1px solid rgba(168, 85, 247, 0.25);
          padding: 0.875rem;
        }

        .pattern-header,
        .circuit-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }

        .pattern-type {
          font-size: 0.875rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.9);
          text-transform: capitalize;
        }

        .circuit-type {
          font-size: 0.875rem;
          font-weight: 500;
          color: rgba(168, 85, 247, 0.9);
          text-transform: capitalize;
        }

        .pattern-confidence,
        .circuit-confidence {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .pattern-description,
        .circuit-description {
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.6);
        }

        .pattern-heads {
          display: flex;
          flex-wrap: wrap;
          gap: 0.375rem;
          margin-top: 0.5rem;
        }

        .head-tag {
          font-size: 0.7rem;
          padding: 0.25rem 0.5rem;
          background: rgba(255, 255, 255, 0.08);
          color: rgba(255, 255, 255, 0.7);
        }

        .more-btn {
          color: rgba(255, 255, 255, 0.5);
          cursor: pointer;
          border: 1px solid rgba(255, 255, 255, 0.15);
          transition: all 0.2s;
        }

        .more-btn:hover {
          background: rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.8);
          border-color: rgba(255, 255, 255, 0.3);
        }

        .expanded-heads-popup {
          margin-top: 0.75rem;
          background: rgba(0, 0, 0, 0.6);
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 0.75rem;
        }

        .expanded-heads-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.625rem;
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.6);
        }

        .expanded-heads-actions {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .copy-heads-btn {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.25rem 0.5rem;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.7);
          font-size: 0.7rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .copy-heads-btn:hover {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.3);
        }

        .close-popup-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0.25rem;
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.4);
          cursor: pointer;
          transition: color 0.2s;
        }

        .close-popup-btn:hover {
          color: rgba(255, 255, 255, 0.8);
        }

        .expanded-heads-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.375rem;
          max-height: 200px;
          overflow-y: auto;
        }

        .results-placeholder {
          height: 400px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 0, 0, 0.2);
          border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .placeholder-content {
          text-align: center;
          color: rgba(255, 255, 255, 0.4);
          max-width: 400px;
        }

        .placeholder-icon {
          width: 48px;
          height: 48px;
          margin: 0 auto 1rem;
          opacity: 0.3;
        }

        .placeholder-content h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.5);
        }

        .placeholder-content p {
          margin: 0;
          font-size: 0.8rem;
        }

        @media (max-width: 1000px) {
          .analysis-grid {
            grid-template-columns: 1fr;
          }

          .top-heads-grid-compact {
            grid-template-columns: repeat(5, 1fr);
          }
        }

        @media (max-width: 600px) {
          .top-heads-grid-compact {
            grid-template-columns: repeat(4, 1fr);
          }
        }
      `}</style>
    </div>
  )
}
