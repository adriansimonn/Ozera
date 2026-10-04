import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import TextGenerator from '../../components/model/TextGenerator'
import type { GenerationStreamState, PlainGenerationResult } from '../../components/model/TextGenerator'
import type { FinishReason } from '../../api/client'
import ModelInfoBox, { ModelDetails } from '../../components/model/ModelInfoBox'
import { AttentionHeatmap } from '../../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../../components/visualization/TransformationFlow'
import { GenerationFlow } from '../../components/visualization/GenerationFlow'
import { NavBar } from '../../components/common/NavBar'
import { useActivationData, type VisualizationType } from '../../hooks/useActivationData'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network, Copy, Check, BarChart3, FileText, Trash2, Info, Maximize2, MessageSquare, X } from 'lucide-react'
import { Dropdown } from '../../components/common/Dropdown'
import { ProbeMonitor } from '../../components/probes/ProbeMonitor'
import { useTheme } from '../../hooks/useTheme'

type RightPanelMode = 'visualizations' | 'outputs'

interface GeneratedOutput extends PlainGenerationResult {
  id: number
  timestamp: Date
  activationId?: string
}

// An output's prompt or model card, shown in a popup
interface OutputDetail {
  kind: 'prompt' | 'model'
  output: GeneratedOutput
}

interface UnifiedPageProps {
  onShowPurchaseCredits?: () => void
}

const FINISH_REASON_LABELS: Record<FinishReason, string> = {
  eos: 'EOS token',
  length: 'Token limit',
  stop: 'Stopped',
}

const STREAM_PHASE_MESSAGES: Record<GenerationStreamState['phase'], string> = {
  waiting: 'Generating and capturing activations...',
  streaming: 'Generating and capturing activations...',
  capturing: 'Capturing activations...',
  stopping: 'Stopping generation...',
}

const STREAM_PHASE_LABELS: Record<GenerationStreamState['phase'], string> = {
  waiting: 'Waiting for the model',
  streaming: 'Generating',
  capturing: 'Capturing activations',
  stopping: 'Stopping',
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
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const outputIdCounter = useRef(0)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [outputDetail, setOutputDetail] = useState<OutputDetail | null>(null)

  // The generation in progress, if any
  const [stream, setStream] = useState<GenerationStreamState | null>(null)
  const streamActiveRef = useRef(false)
  const streamBoxRef = useRef<HTMLDivElement>(null)

  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)
  const [tokenInputValue, setTokenInputValue] = useState('0')
  const [showTextLabels, setShowTextLabels] = useState(false)

  const { activationData, loading, layerLoading, error, summary, loadActivation } =
    useActivationData(selectedVisualization, selectedLayer)

  useEffect(() => {
    if (urlActivationId) {
      setCurrentActivationId(urlActivationId)
      loadActivation(urlActivationId)
    }
  }, [urlActivationId, loadActivation])

  const handleActivationGenerated = (newActivationId: string) => {
    setCurrentActivationId(newActivationId)
    setRightPanelMode('visualizations')
    loadActivation(newActivationId)
  }

  const handleStreamUpdate = (next: GenerationStreamState | null) => {
    // A generation starting: show the view its text streams into (both show it)
    if (next && !streamActiveRef.current) {
      setRightPanelMode(next.mode === 'visualize' ? 'visualizations' : 'outputs')
    }
    streamActiveRef.current = next !== null
    setStream(next)
  }

  const handlePlainTextGenerated = (result: PlainGenerationResult) => {
    const output: GeneratedOutput = {
      ...result,
      id: outputIdCounter.current++,
      timestamp: new Date(),
    }
    setGeneratedOutputs(prev => [output, ...prev])
    // Replace the live output with the finished one in the same render
    streamActiveRef.current = false
    setStream(null)
  }

  // Keep the newest streamed text in view, unless the user has scrolled up to read
  useEffect(() => {
    const el = streamBoxRef.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) {
      el.scrollTop = el.scrollHeight
    }
  }, [stream?.text])

  const handleVisualizeOutput = (activationId: string) => {
    setCurrentActivationId(activationId)
    loadActivation(activationId)
    setRightPanelMode('visualizations')
  }

  const handleClearOutputs = () => {
    setGeneratedOutputs([])
    setShowClearConfirm(false)
  }

  const handleCopy = (key: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  useEffect(() => {
    if (!outputDetail) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOutputDetail(null)
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [outputDetail])

  const numLayers = activationData?.activations.layers?.length || summary?.num_layers || 0
  const numHeads = activationData?.activations.layers?.[0]?.attn_weights?.shape[1]
    || summary?.layer_info?.[0]?.attn_weights?.shape[1]
    || 0

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
    textMid: isLight ? '#1d1d1f' : '#ffffff',
    textSub: isLight ? '#1d1d1f' : '#ffffff',
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

  const settingLabelStyle = { fontSize: '0.6rem', color: c.textSub, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 } as const
  const settingValueStyle = { fontSize: '0.75rem', color: c.textMid, fontFamily: 'monospace' } as const

  const renderLayerLoading = (message: string) => (
    <div className="df-placeholder">
      <div className="df-placeholder-inner">
        <div className="df-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
        <p style={{ color: c.textSub }}>{message}</p>
      </div>
    </div>
  )

  const renderVisualization = () => {
    if (stream?.mode === 'visualize') {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner df-generating">
            <div className="df-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
            <p style={{ color: c.textSub }}>{STREAM_PHASE_MESSAGES[stream.phase]}</p>
            {stream.text && (
              <div
                ref={streamBoxRef}
                className="df-stream-box"
                style={{ background: c.panelBg, borderColor: c.panelBorder, color: c.text }}
              >
                {stream.text}
                {stream.phase === 'streaming' && <span className="df-cursor" style={{ color: c.textMid }}>|</span>}
              </div>
            )}
            {stream.monitor && stream.monitorProbe && (
              <div className="df-stream-monitor" style={{ background: c.panelBg }}>
                <ProbeMonitor trace={stream.monitor} probe={stream.monitorProbe} live />
              </div>
            )}
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
        {selectedVisualization === 'attention' && (
          activationData.activations.layers?.[selectedLayer]?.attn_weights ? (
            <AttentionHeatmap
              attentionWeights={activationData.activations.layers[selectedLayer].attn_weights!}
              layerIndex={selectedLayer}
              headIndex={selectedHead}
              tokens={activationData.metadata.decoded_tokens}
              showTextLabels={showTextLabels}
              activationId={currentActivationId ?? undefined}
            />
          ) : layerLoading && renderLayerLoading(`Loading layer ${selectedLayer} attention data...`)
        )}
        {selectedVisualization === 'activations' && (
          activationData.activations.layers?.[selectedLayer]?.attn_output || activationData.activations.layers?.[selectedLayer]?.ff_output ? (
            <LayerActivationDisplay
              layerActivations={activationData.activations.layers[selectedLayer]}
              layerIndex={selectedLayer}
              activationId={currentActivationId ?? undefined}
            />
          ) : layerLoading && renderLayerLoading(`Loading layer ${selectedLayer} activations...`)
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
                onModelChange={setInfoBoxModel}
                onPlainTextGenerated={handlePlainTextGenerated}
                onStreamUpdate={handleStreamUpdate}
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
                    </button>
                  </div>
                </div>
                <div className="df-toolbar-right">
                  <div className="df-toolbar-group">
                    <span className="df-toolbar-label" style={{ color: c.textSub }}>Visualization:</span>
                    <Dropdown
                      id="df-vis-select"
                      value={selectedVisualization}
                      onChange={(v) => setSelectedVisualization(v as VisualizationType)}
                      options={visualizationOptions.map((opt) => ({ value: opt.value, label: opt.label }))}
                    />
                  </div>

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
                {generatedOutputs.length === 0 && !stream ? (
                  <div className="df-placeholder">
                    <div className="df-placeholder-inner">
                      <p style={{ color: c.textSub }}>Generate text to see outputs here</p>
                    </div>
                  </div>
                ) : (
                  <div className="df-outputs-list">
                    {stream && (
                      <div className="df-output-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                        <div className="df-output-text" style={{ color: c.text }}>
                          {stream.text || (
                            <span style={{ color: c.textSub, opacity: 0.5 }}>Waiting for the first token...</span>
                          )}
                          {stream.phase === 'streaming' && <span className="df-cursor" style={{ color: c.textMid }}>|</span>}
                        </div>
                        {stream.monitor && stream.monitorProbe && (
                          <div className="df-output-monitor">
                            <ProbeMonitor trace={stream.monitor} probe={stream.monitorProbe} live />
                          </div>
                        )}
                        <div className="df-output-settings" style={{ borderTop: `1px solid ${c.divider}` }}>
                          <div className="df-output-setting df-output-live">
                            <span className="df-live-dot" style={{ background: c.text }} />
                            <span style={settingLabelStyle}>
                              {STREAM_PHASE_LABELS[stream.phase]}
                              {stream.mode === 'visualize' && stream.phase !== 'capturing' && ' · with visualization'}
                            </span>
                          </div>
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
                              onClick={() => handleCopy(`text-${output.id}`, output.text)}
                              title="Copy to clipboard"
                              style={{ color: copiedKey === `text-${output.id}` ? c.text : c.textSub, background: c.controlBg, borderColor: c.controlBorder }}
                            >
                              {copiedKey === `text-${output.id}` ? <Check style={{ width: 14, height: 14 }} /> : <Copy style={{ width: 14, height: 14 }} />}
                            </button>
                          </div>
                        </div>
                        {output.monitor && output.monitorProbe && (
                          <div className="df-output-monitor">
                            <ProbeMonitor trace={output.monitor} probe={output.monitorProbe} />
                          </div>
                        )}
                        <div className="df-output-settings" style={{ borderTop: `1px solid ${c.divider}` }}>
                          <button
                            className="df-output-detail df-output-detail-model"
                            onClick={() => setOutputDetail({ kind: 'model', output })}
                            title="Show model card"
                            style={{ borderRight: `1px solid ${c.divider}` }}
                          >
                            <span className="df-output-detail-label" style={settingLabelStyle}>
                              Model
                              <Info className="df-output-detail-icon" style={{ width: 10, height: 10 }} />
                            </span>
                            <span className="df-output-detail-value" style={settingValueStyle}>{output.model}</span>
                          </button>
                          <div className="df-output-stats">
                            {[
                              { label: 'Tokens', value: output.generatedTokens ?? output.maxTokens },
                              ...(output.finishReason ? [{ label: 'Ended', value: FINISH_REASON_LABELS[output.finishReason] }] : []),
                              { label: 'Temp', value: output.temperature.toFixed(2) },
                              { label: 'Top-K', value: output.topK },
                            ].map((s) => (
                              <div key={s.label} className="df-output-setting">
                                <span style={settingLabelStyle}>{s.label}</span>
                                <span style={settingValueStyle}>{s.value}</span>
                              </div>
                            ))}
                          </div>
                          <button
                            className="df-output-detail df-output-detail-prompt"
                            onClick={() => setOutputDetail({ kind: 'prompt', output })}
                            title="Show full prompt"
                            style={{ borderLeft: `1px solid ${c.divider}` }}
                          >
                            <span className="df-output-detail-label" style={settingLabelStyle}>
                              Prompt
                              <Maximize2 className="df-output-detail-icon" style={{ width: 10, height: 10 }} />
                            </span>
                            <span className="df-output-detail-value" style={settingValueStyle}>{output.prompt}</span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Output prompt / model card popup */}
          {outputDetail && (
            <div className="df-modal-overlay" onClick={() => setOutputDetail(null)}>
              <div
                className={`df-modal df-detail-modal ${outputDetail.kind === 'model' ? 'df-detail-modal-model' : ''}`}
                role="dialog"
                aria-modal="true"
                aria-labelledby="df-detail-title"
                style={{ background: c.panelBg, borderColor: c.panelBorder }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="df-detail-header" style={{ borderBottom: `1px solid ${c.divider}` }}>
                  {outputDetail.kind === 'prompt'
                    ? <MessageSquare style={{ width: 18, height: 18, color: c.text, flexShrink: 0 }} />
                    : <Info style={{ width: 18, height: 18, color: c.text, flexShrink: 0 }} />}
                  <h3 id="df-detail-title" style={{ color: c.text }}>
                    {outputDetail.kind === 'prompt' ? 'Prompt' : outputDetail.output.modelName}
                  </h3>
                  <div className="df-detail-actions">
                    {outputDetail.kind === 'prompt' && (
                      <button
                        className="df-copy-btn"
                        onClick={() => handleCopy(`prompt-${outputDetail.output.id}`, outputDetail.output.prompt)}
                        title="Copy prompt"
                        style={{
                          color: copiedKey === `prompt-${outputDetail.output.id}` ? c.text : c.textSub,
                          background: c.controlBg,
                          borderColor: c.controlBorder,
                        }}
                      >
                        {copiedKey === `prompt-${outputDetail.output.id}`
                          ? <Check style={{ width: 14, height: 14 }} />
                          : <Copy style={{ width: 14, height: 14 }} />}
                      </button>
                    )}
                    <button
                      className="df-copy-btn"
                      onClick={() => setOutputDetail(null)}
                      title="Close"
                      aria-label="Close"
                      style={{ color: c.textSub, background: c.controlBg, borderColor: c.controlBorder }}
                    >
                      <X style={{ width: 14, height: 14 }} />
                    </button>
                  </div>
                </div>
                <div className="df-detail-body">
                  {outputDetail.kind === 'prompt' ? (
                    <div className="df-detail-prompt" style={{ color: c.text, background: c.metaBg, borderColor: c.metaBorder }}>
                      {outputDetail.output.prompt}
                    </div>
                  ) : (
                    <ModelDetails modelId={outputDetail.output.model} />
                  )}
                </div>
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
          margin-top: 76px;
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
          padding: 0 1.25rem;
          height: 60px;
          box-sizing: border-box;
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

        .df-generating {
          width: 100%;
          max-width: 720px;
        }

        .df-stream-box {
          width: 100%;
          max-height: 45vh;
          overflow-y: auto;
          margin-top: 1.25rem;
          padding: 1rem 1.25rem;
          border: 1px solid;
          text-align: left;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .df-output-live {
          flex-direction: row !important;
          align-items: center;
          gap: 0.5rem !important;
        }

        .df-live-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          animation: df-pulse 1.2s ease-in-out infinite;
        }

        @keyframes df-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.25; }
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

        .df-stream-monitor {
          width: 100%;
        }

        .df-output-monitor {
          padding: 0 1.5rem 1.25rem;
        }

        .df-output-monitor .probe-monitor {
          margin-top: 0;
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
          align-items: stretch;
        }

        .df-output-stats {
          display: flex;
          flex-wrap: wrap;
          align-content: center;
          min-width: 0;
        }

        /* Model and prompt: open their popup; span the full height of the settings bar */
        .df-output-detail {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.2rem;
          min-width: 0;
          padding: 0.5rem 0.875rem;
          background: none;
          border: none;
          font: inherit;
          text-align: left;
          cursor: pointer;
          transition: background 0.15s;
        }

        .df-output-detail:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        [data-bg="light"] .df-output-detail:hover {
          background: rgba(0, 0, 0, 0.04);
        }

        .df-output-detail:focus-visible {
          outline: 1px solid currentColor;
          outline-offset: -2px;
        }

        .df-output-detail-model {
          flex-shrink: 0;
          max-width: 240px;
        }

        .df-output-detail-prompt {
          flex: 1;
          min-width: 120px;
        }

        .df-output-detail-label {
          display: flex;
          align-items: center;
          gap: 0.3rem;
        }

        .df-output-detail-icon {
          opacity: 0.45;
          transition: opacity 0.15s;
        }

        .df-output-detail:hover .df-output-detail-icon {
          opacity: 1;
        }

        .df-output-detail-value {
          max-width: 100%;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
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

        .df-switch-group {
          display: flex;
          align-items: center;
          gap: 0.125rem;
          padding: 0.1875rem;
          border: 1px solid;
          line-height: 1;
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
          line-height: 1;
        }

        .df-switch-btn:hover:not(.active) {
          opacity: 0.7;
        }

        .df-switch-badge {
          font-size: 0.675rem;
          font-family: monospace;
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

        .df-detail-modal {
          display: flex;
          flex-direction: column;
          width: min(640px, calc(100vw - 2rem));
          max-height: 100vh;
          padding: 0;
        }

        /* Fits the spec cards three to a row */
        .df-detail-modal-model {
          width: min(560px, calc(100vw - 2rem));
        }

        .df-detail-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem 1rem 1rem 1.5rem;
          flex-shrink: 0;
        }

        .df-detail-header h3 {
          flex: 1;
          min-width: 0;
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
          letter-spacing: -0.01em;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .df-detail-actions {
          display: flex;
          gap: 0.375rem;
          flex-shrink: 0;
        }

        .df-detail-body {
          padding: 1.5rem;
          overflow-y: auto;
          min-height: 0;
        }

        .df-detail-prompt {
          padding: 1rem 1.25rem;
          border: 1px solid;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.85rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
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
