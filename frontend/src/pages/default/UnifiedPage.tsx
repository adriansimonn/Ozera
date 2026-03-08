import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import TextGenerator from '../../components/model/TextGenerator'
import type { PlainGenerationResult } from '../../components/model/TextGenerator'
import ModelInfoBox from '../../components/model/ModelInfoBox'
import { AttentionHeatmap } from '../../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../../components/visualization/TransformationFlow'
import { GenerationFlow } from '../../components/visualization/GenerationFlow'
import { NavBar } from '../../components/common/NavBar'
import { apiClient } from '../../api/client'
import type { ActivationData, LayerActivations } from '../../types/model'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network, Copy, Check, BarChart3, FileText, Trash2 } from 'lucide-react'
import { Dropdown } from '../../components/common/Dropdown'
import { useTheme } from '../../hooks/useTheme'

type VisualizationType = 'network' | 'attention' | 'activations' | 'journey' | 'flow'
type RightPanelMode = 'visualizations' | 'outputs'

interface GeneratedOutput extends PlainGenerationResult {
  id: number
  timestamp: Date
  activationId?: string
}

interface UnifiedPageProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultUnifiedPage({ onShowPurchaseCredits }: UnifiedPageProps) {
  const [searchParams] = useSearchParams()
  const urlActivationId = searchParams.get('id')
  const { isLight } = useTheme()

  const [currentActivationId, setCurrentActivationId] = useState<string | null>(urlActivationId)
  const [selectedVisualization, setSelectedVisualization] = useState<VisualizationType>('network')
  const [infoBoxModel, setInfoBoxModel] = useState<string>('nano')
  const [rightPanelMode, setRightPanelMode] = useState<RightPanelMode>('visualizations')
  const [generatedOutputs, setGeneratedOutputs] = useState<GeneratedOutput[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [copiedId, setCopiedId] = useState<number | null>(null)
  const outputIdCounter = useRef(0)
  const [pendingActivationId, setPendingActivationId] = useState<string | null>(null)
  const [showClearConfirm, setShowClearConfirm] = useState(false)

  const [activationData, setActivationData] = useState<ActivationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const layerCacheRef = useRef<Map<number, LayerActivations>>(new Map())
  const currentActivationIdRef = useRef<string | null>(null)

  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)
  const [tokenInputValue, setTokenInputValue] = useState('0')
  const [showTextLabels, setShowTextLabels] = useState(false)

  useEffect(() => {
    if (urlActivationId) {
      setCurrentActivationId(urlActivationId)
      loadActivationSummary(urlActivationId)
    }
  }, [urlActivationId])

  async function loadActivationSummary(id: string) {
    try {
      setLoading(true)
      setError(null)
      if (currentActivationIdRef.current !== id) {
        layerCacheRef.current.clear()
        currentActivationIdRef.current = id
      }
      const data = await apiClient.getActivations(id)
      setActivationData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activations')
    } finally {
      setLoading(false)
    }
  }

  const loadLayerActivations = useCallback(async (layerIdx: number) => {
    if (!currentActivationIdRef.current) return null
    if (layerCacheRef.current.has(layerIdx)) {
      return layerCacheRef.current.get(layerIdx)!
    }
    if (activationData?.activations.layers?.[layerIdx]) {
      layerCacheRef.current.set(layerIdx, activationData.activations.layers[layerIdx])
      return activationData.activations.layers[layerIdx]
    }
    try {
      const result = await apiClient.getLayerActivations(currentActivationIdRef.current, layerIdx)
      layerCacheRef.current.set(layerIdx, result.activations as LayerActivations)
      return result.activations as LayerActivations
    } catch (err) {
      console.error(`Failed to load layer ${layerIdx}:`, err)
      return null
    }
  }, [activationData])

  useEffect(() => {
    if (activationData && (selectedVisualization === 'attention' || selectedVisualization === 'activations')) {
      loadLayerActivations(selectedLayer)
    }
  }, [selectedLayer, selectedVisualization, activationData, loadLayerActivations])

  const handleActivationGenerated = (newActivationId: string) => {
    setCurrentActivationId(newActivationId)
    setPendingActivationId(newActivationId)
    setRightPanelMode('visualizations')
    loadActivationSummary(newActivationId)
  }

  const handleStreamingText = (text: string, streaming: boolean) => {
    setStreamingText(text)
    setIsStreaming(streaming)
    if (streaming || text) {
      setRightPanelMode('outputs')
    }
  }

  const handlePlainTextGenerated = (result: PlainGenerationResult) => {
    const output: GeneratedOutput = {
      ...result,
      id: outputIdCounter.current++,
      timestamp: new Date(),
      activationId: pendingActivationId ?? undefined,
    }
    setPendingActivationId(null)
    setGeneratedOutputs(prev => [output, ...prev])
    setStreamingText('')
    setIsStreaming(false)
  }

  const handleVisualizeOutput = (activationId: string) => {
    setCurrentActivationId(activationId)
    loadActivationSummary(activationId)
    setRightPanelMode('visualizations')
  }

  const handleClearOutputs = () => {
    setGeneratedOutputs([])
    setStreamingText('')
    setIsStreaming(false)
    setShowClearConfirm(false)
  }

  const handleCopy = (id: number, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const numLayers = activationData?.activations.layers?.length || 0
  const numHeads = activationData?.activations.layers?.[0]?.attn_weights?.shape[1] || 0

  const visualizationOptions = [
    { value: 'network' as const, label: 'Neural Network Generation Flow', icon: Network },
    { value: 'attention' as const, label: 'Attention Heatmap', icon: Eye },
    { value: 'activations' as const, label: 'Layer Activations', icon: Sparkles },
    { value: 'journey' as const, label: 'Embedding Journey', icon: TrendingUp },
    { value: 'flow' as const, label: 'Transformation Flow', icon: Layers },
  ]

  // Color tokens for dark/light
  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    panelBorder: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
    textMid: isLight ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.6)',
    textSub: isLight ? 'rgba(0,0,0,0.45)' : 'rgba(255,255,255,0.4)',
    controlBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
    controlBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    controlHover: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.08)',
    metaBg: isLight ? 'rgba(0,0,0,0.02)' : 'rgba(255,255,255,0.02)',
    metaBorder: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)',
    chipBg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)',
    chipBorder: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    spinnerHead: isLight ? '#1d1d1f' : '#ffffff',
  }

  const renderVisualization = () => {
    if (generating) {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner">
            <div className="df-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
            <p style={{ color: c.textSub }}>Generating and capturing activations...</p>
          </div>
        </div>
      )
    }
    if (loading) {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner">
            <div className="df-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
            <p style={{ color: c.textSub }}>Loading activations...</p>
          </div>
        </div>
      )
    }
    if (!activationData) {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner">
            <Network style={{ width: 40, height: 40, color: c.textSub, opacity: 0.4, marginBottom: 12 }} />
            <p style={{ color: c.textSub }}>Generate text with visualizations to see activations here</p>
          </div>
        </div>
      )
    }
    if (error) {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner">
            <p style={{ color: c.textMid }}>{error}</p>
          </div>
        </div>
      )
    }

    return (
      <>
        {selectedVisualization === 'network' && (
          <GenerationFlow activationData={activationData} />
        )}
        {selectedVisualization === 'attention' && activationData.activations.layers?.[selectedLayer]?.attn_weights && (
          <AttentionHeatmap
            attentionWeights={activationData.activations.layers[selectedLayer].attn_weights!}
            layerIndex={selectedLayer}
            headIndex={selectedHead}
            tokens={activationData.metadata.decoded_tokens}
            showTextLabels={showTextLabels}
            activationId={currentActivationId ?? undefined}
          />
        )}
        {selectedVisualization === 'activations' && activationData.activations.layers?.[selectedLayer] && (
          <LayerActivationDisplay
            layerActivations={activationData.activations.layers[selectedLayer]}
            layerIndex={selectedLayer}
            activationId={currentActivationId ?? undefined}
          />
        )}
        {selectedVisualization === 'journey' && (
          <EmbeddingJourney activationData={activationData} selectedTokenIndex={selectedTokenIndex} />
        )}
        {selectedVisualization === 'flow' && (
          <TransformationFlow activationData={activationData} selectedTokenIndex={selectedTokenIndex} />
        )}
      </>
    )
  }

  return (
    <div className="df-page">
      <NavBar />

      <div className="df-layout" style={{ borderColor: c.divider }}>
        {/* Left Panel — Generator + Model Info */}
        <div className="df-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="df-left-scroll">
            <div className="df-generator-wrap">
              <TextGenerator
                defaultModel="nano"
                onActivationGenerated={handleActivationGenerated}
                onGeneratingChange={setGenerating}
                onModelChange={setInfoBoxModel}
                onPlainTextGenerated={handlePlainTextGenerated}
                onStreamingText={handleStreamingText}
                onShowPurchaseCredits={onShowPurchaseCredits}
              />
            </div>
            <div className="df-model-info-wrap">
              <ModelInfoBox selectedModel={infoBoxModel} onModelChange={setInfoBoxModel} />
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="df-right" style={{ background: c.bg }}>
          {rightPanelMode === 'visualizations' ? (
            <div className="df-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Unified toolbar */}
              <div className="df-toolbar" style={{ background: c.panelBg, borderBottom: `1px solid ${c.divider}` }}>
                <div className="df-toolbar-left">
                  <div className="df-switch-group" style={{ background: c.controlBg, borderColor: c.controlBorder }}>
                    <button
                      className="df-switch-btn active"
                      style={{ color: c.text, background: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)' }}
                    >
                      <BarChart3 style={{ width: 13, height: 13 }} />
                      Visualizations
                    </button>
                    <button
                      className="df-switch-btn"
                      onClick={() => setRightPanelMode('outputs')}
                      style={{ color: c.textSub }}
                    >
                      <FileText style={{ width: 13, height: 13 }} />
                      Outputs
                      {generatedOutputs.length > 0 && (
                        <span className="df-switch-badge" style={{ color: c.textSub }}>{generatedOutputs.length}</span>
                      )}
                    </button>
                  </div>
                </div>
                <div className="df-toolbar-right">
                  <Dropdown
                    id="df-vis-select"
                    value={selectedVisualization}
                    onChange={(v) => setSelectedVisualization(v as VisualizationType)}
                    options={visualizationOptions.map((opt) => ({ value: opt.value, label: opt.label }))}
                  />

                  {selectedVisualization === 'attention' && activationData && (
                    <div className="df-toolbar-group">
                      <span className="df-toolbar-label" style={{ color: c.textSub }}>Labels</span>
                      <Dropdown
                        id="df-labels-select"
                        value={showTextLabels ? 'text' : 'number'}
                        onChange={(v) => setShowTextLabels(v === 'text')}
                        options={[
                          { value: 'number', label: 'Token Number' },
                          { value: 'text', label: 'Token Text' },
                        ]}
                      />
                    </div>
                  )}

                  {activationData && selectedVisualization !== 'network' && (
                    <div className="df-controls">
                      {(selectedVisualization === 'attention' || selectedVisualization === 'activations') && (
                        <div className="df-control-item">
                          <Layers style={{ width: 14, height: 14, color: c.textSub }} />
                          <span style={{ color: c.textSub, fontSize: '0.75rem' }}>Layer</span>
                          <button
                            onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
                            disabled={selectedLayer === 0}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronLeft style={{ width: 12, height: 12 }} />
                          </button>
                          <span className="df-ctrl-val" style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.text }}>{selectedLayer}</span>
                          <button
                            onClick={() => setSelectedLayer(Math.min(numLayers - 1, selectedLayer + 1))}
                            disabled={selectedLayer === numLayers - 1}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronRight style={{ width: 12, height: 12 }} />
                          </button>
                        </div>
                      )}

                      {selectedVisualization === 'attention' && (
                        <div className="df-control-item">
                          <span style={{ color: c.textSub, fontSize: '0.75rem' }}>Head</span>
                          <button
                            onClick={() => setSelectedHead(Math.max(0, selectedHead - 1))}
                            disabled={selectedHead === 0}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronLeft style={{ width: 12, height: 12 }} />
                          </button>
                          <span className="df-ctrl-val" style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.text }}>{selectedHead}</span>
                          <button
                            onClick={() => setSelectedHead(Math.min(numHeads - 1, selectedHead + 1))}
                            disabled={selectedHead === numHeads - 1}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronRight style={{ width: 12, height: 12 }} />
                          </button>
                        </div>
                      )}

                      {(selectedVisualization === 'journey' || selectedVisualization === 'flow') && (
                        <div className="df-control-item">
                          <span style={{ color: c.textSub, fontSize: '0.75rem' }}>Token</span>
                          <button
                            onClick={() => {
                              const v = Math.max(0, selectedTokenIndex - 1)
                              setSelectedTokenIndex(v)
                              setTokenInputValue(String(v))
                            }}
                            disabled={selectedTokenIndex === 0}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronLeft style={{ width: 12, height: 12 }} />
                          </button>
                          <input
                            type="text"
                            className="df-ctrl-val df-ctrl-input"
                            value={tokenInputValue}
                            onChange={(e) => {
                              setTokenInputValue(e.target.value)
                              if (e.target.value === '') return
                              const num = parseInt(e.target.value, 10)
                              if (!isNaN(num) && num >= 0 && num <= activationData.tokens.length - 1) {
                                setSelectedTokenIndex(num)
                              }
                            }}
                            onKeyDown={(e) => {
                              if (e.key === 'ArrowUp') {
                                e.preventDefault()
                                const v = Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1)
                                setSelectedTokenIndex(v)
                                setTokenInputValue(String(v))
                              } else if (e.key === 'ArrowDown') {
                                e.preventDefault()
                                const v = Math.max(0, selectedTokenIndex - 1)
                                setSelectedTokenIndex(v)
                                setTokenInputValue(String(v))
                              }
                            }}
                            style={{
                              background: c.controlBg,
                              borderColor: c.controlBorder,
                              color: c.text,
                              width: `${Math.max(8, tokenInputValue.length + 2)}ch`,
                            }}
                          />
                          <button
                            onClick={() => {
                              const v = Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1)
                              setSelectedTokenIndex(v)
                              setTokenInputValue(String(v))
                            }}
                            disabled={selectedTokenIndex === activationData.tokens.length - 1}
                            className="df-ctrl-btn"
                            style={{ background: c.controlBg, borderColor: c.controlBorder, color: c.textMid }}
                          >
                            <ChevronRight style={{ width: 12, height: 12 }} />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Visualization canvas */}
              <div className="df-canvas">
                {renderVisualization()}
              </div>

              {/* Bottom bar: token reference + metadata */}
              {activationData && (
                <div className="df-bottom" style={{ borderTop: `1px solid ${c.divider}` }}>
                  {(selectedVisualization === 'attention' || selectedVisualization === 'journey' || selectedVisualization === 'flow') && activationData.metadata.decoded_tokens && (
                    <div className="df-token-ref" style={{ borderBottom: `1px solid ${c.divider}` }}>
                      <span className="df-token-ref-label" style={{ color: c.textSub }}>Token Reference</span>
                      <div className="df-token-ref-list">
                        {activationData.metadata.decoded_tokens.map((token, idx) => (
                          <span
                            key={idx}
                            className="df-chip"
                            title={`Token ${idx}: "${token}"`}
                            style={{ background: c.chipBg, borderColor: c.chipBorder }}
                          >
                            <span style={{ fontSize: 10, color: c.textSub, fontFamily: 'monospace' }}>{idx}</span>
                            <span style={{ fontSize: 11, color: c.textMid, fontFamily: 'monospace' }}>{token.replace(/ /g, '\u00B7') || '\u25AF'}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="df-meta" style={{ background: c.panelBg }}>
                    {[
                      { label: 'Model', value: activationData.model },
                      { label: 'Tokens', value: activationData.tokens.length },
                      { label: 'Temperature', value: activationData.metadata.temperature?.toFixed(2) || 'N/A' },
                      { label: 'Top-K', value: activationData.metadata.top_k?.toString() || 'N/A' },
                    ].map((m) => (
                      <div key={m.label} className="df-meta-item" style={{ borderColor: c.metaBorder }}>
                        <span style={{ fontSize: '0.65rem', color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{m.label}</span>
                        <span style={{ fontSize: '0.8rem', color: c.text, fontFamily: 'monospace', fontWeight: 500 }}>{m.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="df-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Unified toolbar */}
              <div className="df-toolbar" style={{ background: c.panelBg, borderBottom: `1px solid ${c.divider}` }}>
                <div className="df-toolbar-left">
                  <div className="df-switch-group" style={{ background: c.controlBg, borderColor: c.controlBorder }}>
                    <button
                      className="df-switch-btn"
                      onClick={() => setRightPanelMode('visualizations')}
                      style={{ color: c.textSub }}
                    >
                      <BarChart3 style={{ width: 13, height: 13 }} />
                      Visualizations
                    </button>
                    <button
                      className="df-switch-btn active"
                      style={{ color: c.text, background: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)' }}
                    >
                      <FileText style={{ width: 13, height: 13 }} />
                      Outputs
                      {generatedOutputs.length > 0 && (
                        <span className="df-switch-badge" style={{ color: c.textSub }}>{generatedOutputs.length}</span>
                      )}
                    </button>
                  </div>
                </div>
                <div className="df-toolbar-right">
                  {generatedOutputs.length > 0 && (
                    <button
                      className="df-clear-btn"
                      onClick={() => setShowClearConfirm(true)}
                      title="Clear all outputs"
                      style={{ color: c.textSub, background: c.controlBg, borderColor: c.controlBorder }}
                    >
                      <Trash2 style={{ width: 13, height: 13 }} />
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Outputs list */}
              <div className="df-canvas df-outputs-scroll">
                {generatedOutputs.length === 0 && !isStreaming ? (
                  <div className="df-placeholder">
                    <div className="df-placeholder-inner">
                      <p style={{ color: c.textSub }}>Generate text to see outputs here</p>
                    </div>
                  </div>
                ) : (
                  <div className="df-outputs-list">
                    {isStreaming && (
                      <div className="df-output-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                        <div className="df-output-text" style={{ color: c.text }}>
                          {streamingText}
                          <span className="df-cursor" style={{ color: c.textMid }}>|</span>
                        </div>
                      </div>
                    )}
                    {generatedOutputs.map((output) => (
                      <div key={output.id} className="df-output-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                        <div className="df-output-text-wrap">
                          <div className="df-output-text" style={{ color: c.text }}>
                            {output.text}
                          </div>
                          <div className="df-output-actions">
                            {output.activationId && (
                              <button
                                className="df-visualize-btn"
                                onClick={() => handleVisualizeOutput(output.activationId!)}
                                title="View visualizations"
                                style={{ color: c.textSub, background: c.controlBg, borderColor: c.controlBorder }}
                              >
                                <BarChart3 style={{ width: 13, height: 13 }} />
                                Visualize
                              </button>
                            )}
                            <button
                              className="df-copy-btn"
                              onClick={() => handleCopy(output.id, output.text)}
                              title="Copy to clipboard"
                              style={{ color: copiedId === output.id ? c.text : c.textSub, background: c.controlBg, borderColor: c.controlBorder }}
                            >
                              {copiedId === output.id ? <Check style={{ width: 14, height: 14 }} /> : <Copy style={{ width: 14, height: 14 }} />}
                            </button>
                          </div>
                        </div>
                        <div className="df-output-settings" style={{ borderTop: `1px solid ${c.divider}` }}>
                          {[
                            { label: 'Model', value: output.model },
                            { label: 'Tokens', value: output.maxTokens },
                            { label: 'Temp', value: output.temperature.toFixed(2) },
                            { label: 'Top-K', value: output.topK },
                            { label: 'Prompt', value: output.prompt.length > 60 ? output.prompt.slice(0, 60) + '...' : output.prompt },
                          ].map((s) => (
                            <div key={s.label} className="df-output-setting">
                              <span style={{ fontSize: '0.6rem', color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{s.label}</span>
                              <span style={{ fontSize: '0.75rem', color: c.textMid, fontFamily: 'monospace' }}>{s.value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Clear confirmation modal */}
          {showClearConfirm && (
            <div className="df-modal-overlay" onClick={() => setShowClearConfirm(false)}>
              <div className="df-modal" style={{ background: c.panelBg, borderColor: c.panelBorder }} onClick={(e) => e.stopPropagation()}>
                <p style={{ color: c.text, fontSize: '0.9rem', margin: '0 0 1rem' }}>Clear all generated outputs?</p>
                <div className="df-modal-actions">
                  <button
                    className="df-modal-btn"
                    onClick={() => setShowClearConfirm(false)}
                    style={{ color: c.textMid, background: c.controlBg, borderColor: c.controlBorder }}
                  >
                    Cancel
                  </button>
                  <button
                    className="df-modal-btn df-modal-btn-danger"
                    onClick={handleClearOutputs}
                  >
                    Clear
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .df-page {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .df-layout {
          display: flex;
          flex: 1;
          margin-top: 70px;
          overflow: hidden;
        }

        .df-left {
          width: 380px;
          min-width: 340px;
          max-width: 480px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .df-left-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .df-generator-wrap {
          padding: 0;
        }

        .df-generator-wrap > * {
          border: none !important;
          background: transparent !important;
        }

        .df-model-info-wrap {
          padding: 0;
        }

        .df-model-info-wrap > * {
          border: none !important;
          background: transparent !important;
        }

        .df-right {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .df-panel {
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .df-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.875rem 1.25rem;
          flex-shrink: 0;
          flex-wrap: wrap;
        }

        .df-toolbar-left {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          min-width: 220px;
        }

        .df-toolbar-right {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .df-toolbar-group {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .df-toolbar-label {
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          white-space: nowrap;
        }

        .df-toolbar-title {
          font-size: 0.875rem;
          font-weight: 600;
          letter-spacing: -0.01em;
          white-space: nowrap;
        }

        .df-controls {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .df-control-item {
          display: flex;
          align-items: center;
          gap: 0.375rem;
        }

        .df-ctrl-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0.25rem;
          border: 1px solid;
          cursor: pointer;
          transition: background 0.15s;
        }

        .df-ctrl-btn:hover:not(:disabled) {
          opacity: 0.8;
        }

        .df-ctrl-btn:disabled {
          opacity: 0.25;
          cursor: not-allowed;
        }

        .df-ctrl-val {
          padding: 0.25rem 0.5rem;
          border: 1px solid;
          font-family: monospace;
          font-size: 0.75rem;
          min-width: 32px;
          text-align: center;
        }

        .df-ctrl-input {
          outline: none;
          cursor: text;
        }

        .df-canvas {
          flex: 1;
          overflow: auto;
          min-height: 0;
        }

        .df-placeholder {
          height: 100%;
          min-height: 400px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2rem;
        }

        .df-placeholder-inner {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          max-width: 360px;
        }

        .df-placeholder-inner p {
          font-size: 0.8rem;
          margin: 0;
        }

        .df-spinner {
          width: 32px;
          height: 32px;
          border: 2px solid;
          border-radius: 50%;
          animation: df-spin 1s linear infinite;
          margin: 0 auto 0.75rem;
        }

        @keyframes df-spin {
          to { transform: rotate(360deg); }
        }

        .df-bottom {
          flex-shrink: 0;
        }

        .df-token-ref {
          padding: 0.625rem 1rem;
        }

        .df-token-ref-label {
          display: block;
          font-size: 0.6rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
          margin-bottom: 0.5rem;
        }

        .df-token-ref-list {
          display: flex;
          flex-wrap: wrap;
          gap: 3px;
        }

        .df-chip {
          display: inline-flex;
          align-items: baseline;
          gap: 3px;
          padding: 1px 5px;
          border: 1px solid;
          cursor: default;
        }

        .df-meta {
          display: flex;
          gap: 0;
          padding: 0;
        }

        .df-meta-item {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          padding: 0.625rem 1rem;
          border-right: 1px solid;
        }

        .df-meta-item:last-child {
          border-right: none;
        }

        .df-outputs-scroll {
          padding: 1.25rem;
          overflow-y: auto;
        }

        .df-outputs-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .df-output-card {
          border: 1px solid;
          overflow: hidden;
        }

        .df-output-text-wrap {
          position: relative;
        }

        .df-output-text {
          padding: 1.25rem 1.5rem;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.875rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .df-copy-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0.375rem;
          border: 1px solid;
          cursor: pointer;
          transition: all 0.15s;
        }

        .df-copy-btn:hover {
          filter: brightness(1.2);
        }

        .df-output-settings {
          display: flex;
          flex-wrap: wrap;
          gap: 0;
        }

        .df-output-setting {
          display: flex;
          flex-direction: column;
          gap: 0.2rem;
          padding: 0.5rem 0.875rem;
        }

        .df-cursor {
          animation: df-blink 0.8s infinite;
          margin-left: 2px;
          font-weight: normal;
        }

        @keyframes df-blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }

        .df-output-setting:last-child {
          flex: 1;
          min-width: 120px;
        }

        .df-switch-group {
          display: flex;
          align-items: center;
          gap: 0.125rem;
          padding: 0.1875rem;
          border: 1px solid;
        }

        .df-switch-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.4375rem 0.75rem;
          background: none;
          border: none;
          font-size: 0.775rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          letter-spacing: 0.01em;
          white-space: nowrap;
        }

        .df-switch-btn:hover:not(.active) {
          opacity: 0.7;
        }

        .df-switch-badge {
          font-size: 0.675rem;
          font-family: monospace;
          opacity: 0.7;
        }

        .df-output-actions {
          position: absolute;
          top: 0.75rem;
          right: 0.75rem;
          display: flex;
          gap: 0.375rem;
          opacity: 0;
          transition: opacity 0.15s;
        }

        .df-output-card:hover .df-output-actions {
          opacity: 1;
        }

        .df-visualize-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.35rem 0.625rem;
          border: 1px solid;
          cursor: pointer;
          font-size: 0.7rem;
          font-weight: 500;
          transition: all 0.15s;
        }

        .df-visualize-btn:hover {
          filter: brightness(1.2);
        }

        .df-clear-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.35rem 0.75rem;
          border: 1px solid;
          cursor: pointer;
          font-size: 0.75rem;
          font-weight: 500;
          transition: all 0.15s;
        }

        .df-clear-btn:hover {
          filter: brightness(1.2);
        }

        .df-modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 200;
        }

        .df-modal {
          padding: 1.5rem;
          border: 1px solid;
          min-width: 300px;
          box-shadow: 0 16px 48px rgba(0, 0, 0, 0.3);
        }

        .df-modal-actions {
          display: flex;
          gap: 0.5rem;
          justify-content: flex-end;
        }

        .df-modal-btn {
          padding: 0.5rem 1rem;
          border: 1px solid;
          font-size: 0.8rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
        }

        .df-modal-btn:hover {
          filter: brightness(1.1);
        }

        .df-modal-btn-danger {
          background: rgba(239, 68, 68, 0.15);
          border-color: rgba(239, 68, 68, 0.3);
          color: #ef4444;
        }

        .df-modal-btn-danger:hover {
          background: rgba(239, 68, 68, 0.25);
        }

        @media (max-width: 900px) {
          .df-layout {
            flex-direction: column;
          }
          .df-left {
            width: 100%;
            max-width: 100%;
            max-height: 40vh;
            border-right: none !important;
            border-bottom: 1px solid;
          }
        }
      `}</style>
    </div>
  )
}
