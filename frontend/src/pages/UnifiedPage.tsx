import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import TextGenerator from '../components/model/TextGenerator'
import type { GenerationStreamState, PlainGenerationResult } from '../components/model/TextGenerator'
import type { FinishReason } from '../api/client'
import ModelInfoBox, { ModelDetails } from '../components/model/ModelInfoBox'
import { AttentionHeatmap } from '../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../components/visualization/TransformationFlow'
import { GenerationFlow } from '../components/visualization/GenerationFlow'
import { NavBar } from '../components/common/NavBar'
import { useActivationData, type VisualizationType } from '../hooks/useActivationData'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network, Copy, Check,    Info, Maximize2,  X } from 'lucide-react'
import { Dropdown } from '../components/common/Dropdown'
import { EmptyState } from '../components/common/EmptyState'
import { Spinner } from '../components/common/Spinner'
import { VAR } from '../styles/appTheme'
import { ProbeMonitor } from '../components/probes/ProbeMonitor'
import { useAssistantPage } from '../hooks/useAssistantPage'
import { visualizationBridge } from '../components/assistant/pages/generate'

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
  const [tokenRefOpen, setTokenRefOpen] = useState(false)

  const { activationData, loading, layerLoading, error, summary, loadActivation } =
    useActivationData(selectedVisualization, selectedLayer)

  // The assistant: which stored generation is open, and how it's shown
  const navigate = useNavigate()
  useAssistantPage('generate', visualizationBridge({
    activationId: currentActivationId,
    openActivation: (id) => {
      setRightPanelMode('visualizations')
      navigate(`/?id=${encodeURIComponent(id)}`)
    },
    loaded: summary ? { model: summary.model, prompt: summary.prompt?.slice(0, 300), tokens: summary.num_tokens } : null,
    visualization: selectedVisualization,
    setVisualization: setSelectedVisualization,
    layer: selectedLayer,
    head: selectedHead,
    tokenIndex: selectedTokenIndex,
  }))

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
  // The page's colors, by the app's custom properties (styles/app.css)
  const c = {
    bg: VAR.bg,
    panelBg: VAR.bg,
    panelBorder: VAR.rule,
    divider: VAR.rule,
    text: VAR.fg,
    textMid: VAR.fg,
    textSub: VAR.fgDim,
  }

  const settingLabelStyle = { fontSize: '0.78rem', color: c.textSub, fontWeight: 500 } as const
  const settingValueStyle = { fontSize: '0.85rem', color: c.text, fontFamily: 'var(--oz-mono)' } as const

  const renderLayerLoading = (message: string) => (
    <div className="df-placeholder">
      <div className="df-placeholder-inner">
        <Spinner large />
        <p>{message}</p>
      </div>
    </div>
  )

  const renderVisualization = () => {
    if (stream?.mode === 'visualize') {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner df-generating">
            <Spinner large />
            <p>{STREAM_PHASE_MESSAGES[stream.phase]}</p>
            {stream.text && (
              <div
                ref={streamBoxRef}
                className="df-stream-box"
              >
                {stream.text}
                {stream.phase === 'streaming' && <span className="df-cursor" style={{ color: c.textMid }}>|</span>}
              </div>
            )}
            {stream.monitor && stream.monitorProbe && (
              <div className="df-stream-monitor">
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
            <Spinner large />
            <p>Loading activations...</p>
          </div>
        </div>
      )
    }
    if (!activationData) {
      return (
        <EmptyState glyph="generate" fill>
          Generate text with visualizations to see activations here
        </EmptyState>
      )
    }
    if (error) {
      return (
        <div className="df-placeholder">
          <div className="df-placeholder-inner">
            <p>{error}</p>
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
    <div className="df-page oz-page">
      <NavBar />

      <div className="df-layout oz-page__body">
        {/* Left Panel — Generator + Model Info */}
        <div className="df-left oz-side">
          <div className="df-left-scroll oz-scroll oz-steps">
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
        <div className="df-right oz-main">
          {rightPanelMode === 'visualizations' ? (
            <div className="df-panel" data-assist="generate.activation" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Unified toolbar */}
              <div className="df-toolbar">
                <div className="df-tabs" role="tablist">
                  <button className="oz-tab active" role="tab" aria-selected="true">
                    Visualizations
                  </button>
                  <button className="oz-tab" role="tab" aria-selected="false" onClick={() => setRightPanelMode('outputs')}>
                    Outputs
                  </button>
                </div>
                <div className="df-toolbar-right">
                  <div className="df-toolbar-group" data-assist="generate.visualization">
                    <span className="df-toolbar-label">Visualization</span>
                    <Dropdown
                      id="df-vis-select"
                      value={selectedVisualization}
                      onChange={(v) => setSelectedVisualization(v as VisualizationType)}
                      options={visualizationOptions.map((opt) => ({ value: opt.value, label: opt.label }))}
                    />
                  </div>

                  {selectedVisualization === 'attention' && activationData && (
                    <div className="df-toolbar-group">
                      <span className="df-toolbar-label">Labels</span>
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
                          <span className="df-ctrl-label">Layer</span>
                          <button
                            onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
                            disabled={selectedLayer === 0}
                            className="df-ctrl-btn"
                          >
                            <ChevronLeft />
                          </button>
                          <span className="df-ctrl-val">{selectedLayer}</span>
                          <button
                            onClick={() => setSelectedLayer(Math.min(numLayers - 1, selectedLayer + 1))}
                            disabled={selectedLayer === numLayers - 1}
                            className="df-ctrl-btn"
                          >
                            <ChevronRight />
                          </button>
                        </div>
                      )}

                      {selectedVisualization === 'attention' && (
                        <div className="df-control-item">
                          <span className="df-ctrl-label">Head</span>
                          <button
                            onClick={() => setSelectedHead(Math.max(0, selectedHead - 1))}
                            disabled={selectedHead === 0}
                            className="df-ctrl-btn"
                          >
                            <ChevronLeft />
                          </button>
                          <span className="df-ctrl-val">{selectedHead}</span>
                          <button
                            onClick={() => setSelectedHead(Math.min(numHeads - 1, selectedHead + 1))}
                            disabled={selectedHead === numHeads - 1}
                            className="df-ctrl-btn"
                          >
                            <ChevronRight />
                          </button>
                        </div>
                      )}

                      {(selectedVisualization === 'journey' || selectedVisualization === 'flow') && (
                        <div className="df-control-item">
                          <span className="df-ctrl-label">Token</span>
                          <button
                            onClick={() => {
                              const v = Math.max(0, selectedTokenIndex - 1)
                              setSelectedTokenIndex(v)
                              setTokenInputValue(String(v))
                            }}
                            disabled={selectedTokenIndex === 0}
                            className="df-ctrl-btn"
                          >
                            <ChevronLeft />
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
                            style={{ width: `${Math.max(8, tokenInputValue.length + 2)}ch` }}
                          />
                          <button
                            onClick={() => {
                              const v = Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1)
                              setSelectedTokenIndex(v)
                              setTokenInputValue(String(v))
                            }}
                            disabled={selectedTokenIndex === activationData.tokens.length - 1}
                            className="df-ctrl-btn"
                          >
                            <ChevronRight />
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
                <div className="df-bottom">
                  {(selectedVisualization === 'attention' || selectedVisualization === 'journey' || selectedVisualization === 'flow') && activationData.metadata.decoded_tokens && (
                    <div className="df-token-ref">
                      <button
                        type="button"
                        className="df-token-ref-toggle"
                        aria-expanded={tokenRefOpen}
                        aria-controls="df-token-ref-list"
                        onClick={() => setTokenRefOpen((open) => !open)}
                      >
                        <span>Token reference</span>
                        <span className="df-token-ref-count">{activationData.metadata.decoded_tokens.length}</span>
                        <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
                          <path d="M1 1l4 4 4-4" />
                        </svg>
                      </button>
                      {tokenRefOpen && (
                        <div className="df-token-ref-list" id="df-token-ref-list">
                          {activationData.metadata.decoded_tokens.map((token, idx) => (
                            <span
                              key={idx}
                              className="df-chip"
                              title={`Token ${idx}: "${token}"`}
                            >
                              <span className="df-chip-index">{idx}</span>
                              <span>{token.replace(/ /g, '\u00B7') || '\u25AF'}</span>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="df-meta">
                    {[
                      { label: 'Model', value: activationData.model },
                      { label: 'Tokens', value: activationData.tokens.length },
                      { label: 'Temperature', value: activationData.metadata.temperature?.toFixed(2) || 'N/A' },
                      { label: 'Top-K', value: activationData.metadata.top_k?.toString() || 'N/A' },
                    ].map((m) => (
                      <div key={m.label} className="df-meta-item">
                        <span className="df-meta-label">{m.label}</span>
                        <span className="df-meta-value">{m.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="df-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Unified toolbar */}
              <div className="df-toolbar">
                <div className="df-tabs" role="tablist">
                  <button className="oz-tab" role="tab" aria-selected="false" onClick={() => setRightPanelMode('visualizations')}>
                    Visualizations
                  </button>
                  <button className="oz-tab active" role="tab" aria-selected="true">
                    Outputs
                  </button>
                </div>
                <div className="df-toolbar-right">
                  {generatedOutputs.length > 0 && (
                    <button
                      className="oz-btn oz-btn--ghost oz-btn--small"
                      onClick={() => setShowClearConfirm(true)}
                      title="Clear all outputs"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Outputs list */}
              <div className="df-canvas df-outputs-scroll">
                {generatedOutputs.length === 0 && !stream ? (
                  <EmptyState glyph="generate" fill>
                    Generate text to see outputs here
                  </EmptyState>
                ) : (
                  <div className="df-outputs-list">
                    {stream && (
                      <div className="df-output-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                        <div className="df-output-text" style={{ color: c.text }}>
                          {stream.text || (
                            <span style={{ color: VAR.fgFaint }}>Waiting for the first token...</span>
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
                                className="oz-btn oz-btn--ghost oz-btn--small"
                                onClick={() => handleVisualizeOutput(output.activationId!)}
                                title="View visualizations"
                              >
                                Visualize
                              </button>
                            )}
                            <button
                              className="oz-icon-btn oz-icon-btn--outlined"
                              onClick={() => handleCopy(`text-${output.id}`, output.text)}
                              title="Copy to clipboard"
                              aria-label="Copy to clipboard"
                            >
                              {copiedKey === `text-${output.id}` ? <Check /> : <Copy />}
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
                              <Info className="df-output-detail-icon" />
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
                              <Maximize2 className="df-output-detail-icon" />
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
            <div className="oz-overlay" onClick={() => setOutputDetail(null)}>
              <div
                className={`oz-modal df-detail-modal ${outputDetail.kind === 'model' ? 'df-detail-modal-model' : ''}`}
                role="dialog"
                aria-modal="true"
                aria-labelledby="df-detail-title"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="df-detail-header">
                  <h3 id="df-detail-title">
                    {outputDetail.kind === 'prompt' ? 'Prompt' : outputDetail.output.modelName}
                  </h3>
                  <div className="df-detail-actions">
                    {outputDetail.kind === 'prompt' && (
                      <button
                        className="oz-icon-btn"
                        onClick={() => handleCopy(`prompt-${outputDetail.output.id}`, outputDetail.output.prompt)}
                        title="Copy prompt"
                        aria-label="Copy prompt"
                      >
                        {copiedKey === `prompt-${outputDetail.output.id}` ? <Check /> : <Copy />}
                      </button>
                    )}
                    <button
                      className="oz-icon-btn"
                      onClick={() => setOutputDetail(null)}
                      title="Close"
                      aria-label="Close"
                    >
                      <X />
                    </button>
                  </div>
                </div>
                <div className="df-detail-body">
                  {outputDetail.kind === 'prompt' ? (
                    <div className="df-detail-prompt">
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
            <div className="oz-overlay" onClick={() => setShowClearConfirm(false)}>
              <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
                <h3 className="oz-modal__title">Clear all generated outputs?</h3>
                <div className="oz-modal__actions">
                  <button className="oz-btn oz-btn--ghost" onClick={() => setShowClearConfirm(false)}>
                    Cancel
                  </button>
                  <button className="oz-btn oz-btn--danger" onClick={handleClearOutputs}>
                    Clear
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .df-generator-wrap > *,
        .df-model-info-wrap > * {
          border: none !important;
          background: transparent !important;
        }

        .df-panel {
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        /* The bar over the results: what's shown, and how */
        .df-toolbar {
          display: flex;
          align-items: stretch;
          justify-content: space-between;
          gap: 1.5rem;
          flex-shrink: 0;
          flex-wrap: wrap;
          min-height: 3.75rem;
          padding: 0 var(--oz-gutter);
          border-bottom: 1px solid var(--oz-rule);
        }

        .df-tabs {
          display: flex;
          align-items: stretch;
          gap: 1.75rem;
        }

        .df-tabs .oz-tab {
          padding: 0;
        }

        .df-toolbar-right {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          flex-wrap: wrap;
          padding: 0.6rem 0;
        }

        .df-toolbar-group {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .df-toolbar-group .ozera-dropdown {
          min-width: 15rem;
        }

        #df-labels-select {
          min-width: 10rem;
        }

        .df-toolbar-label,
        .df-ctrl-label {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
          white-space: nowrap;
        }

        .df-controls {
          display: flex;
          align-items: center;
          gap: 1.25rem;
        }

        .df-control-item {
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }

        .df-ctrl-label {
          margin-right: 0.2rem;
        }

        /* A stepper: a value between two arrows */
        .df-ctrl-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 2rem;
          height: 2rem;
          border: 1px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          cursor: pointer;
          transition: border-color 0.2s ease, color 0.2s ease;
        }

        .df-ctrl-btn svg {
          width: 15px;
          height: 15px;
          stroke-width: 2.25;
        }

        .df-ctrl-btn:hover:not(:disabled) {
          border-color: var(--oz-mark);
          color: var(--oz-mark);
        }

        .df-ctrl-btn:disabled {
          border-color: var(--oz-rule);
          color: var(--oz-fg-faint);
          cursor: not-allowed;
        }

        .df-ctrl-val {
          min-width: 2.4rem;
          height: 2rem;
          padding: 0 0.5rem;
          background: transparent;
          border: 1px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.85rem;
          line-height: calc(2rem - 2px);
          text-align: center;
        }

        .df-ctrl-input {
          outline: none;
          cursor: text;
        }

        .df-ctrl-input:focus {
          border-color: var(--oz-fg);
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
          gap: 1.25rem;
          max-width: 380px;
          color: var(--oz-fg-dim);
          text-align: center;
        }

        .df-placeholder-inner p {
          margin: 0;
          font-size: 0.9375rem;
        }

        .df-generating {
          width: 100%;
          max-width: 720px;
        }

        /* The text as it arrives */
        .df-stream-box {
          width: 100%;
          max-height: 45vh;
          overflow-y: auto;
          padding: 1.1rem 1.25rem;
          border: 1px solid var(--oz-rule);
          color: var(--oz-fg);
          text-align: left;
          font-family: var(--oz-mono);
          font-size: 0.875rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .df-stream-monitor {
          width: 100%;
        }

        .df-output-live {
          flex-direction: row !important;
          align-items: center;
          gap: 0.55rem !important;
        }

        /* A lit cell that pulses while a generation runs */
        .df-live-dot {
          width: 7px;
          height: 7px;
          background: var(--oz-accent) !important;
          animation: df-pulse 1.2s ease-in-out infinite;
        }

        @keyframes df-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.2; }
        }

        .df-bottom {
          flex-shrink: 0;
          border-top: 1px solid var(--oz-rule);
        }

        .df-token-ref {
          border-bottom: 1px solid var(--oz-rule);
        }

        /* Closed until asked for: a row that opens the list under it */
        .df-token-ref-toggle {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          width: 100%;
          padding: 0.7rem var(--oz-gutter);
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
          text-align: left;
          cursor: pointer;
          transition: color 0.15s ease;
        }

        .df-token-ref-toggle:hover,
        .df-token-ref-toggle[aria-expanded='true'] {
          color: var(--oz-fg);
        }

        .df-token-ref-count {
          color: var(--oz-fg-faint);
          font-family: var(--oz-mono);
          font-size: 0.72rem;
        }

        .df-token-ref-toggle .oz-caret {
          margin-left: auto;
        }

        .df-token-ref-list {
          display: flex;
          flex-wrap: wrap;
          gap: 3px;
          padding: 0 var(--oz-gutter) 0.95rem;
        }

        /* A token and its position, as a cell */
        .df-chip {
          display: inline-flex;
          align-items: baseline;
          gap: 5px;
          padding: 2px 6px;
          border: 1px solid var(--oz-rule);
          font-family: var(--oz-mono);
          font-size: 0.78rem;
          cursor: default;
          transition: border-color 0.15s ease;
        }

        .df-chip:hover {
          border-color: var(--oz-fg);
        }

        .df-chip-index {
          color: var(--oz-fg-faint);
          font-size: 0.7rem;
        }

        .df-meta {
          display: flex;
        }

        .df-meta-item {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 0.2rem;
          padding: 0.75rem var(--oz-gutter) 0.85rem;
          border-right: 1px solid var(--oz-rule);
        }

        .df-meta-item:last-child {
          border-right: none;
        }

        .df-meta-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
        }

        .df-meta-value {
          font-family: var(--oz-mono);
          font-size: 0.9rem;
        }

        /* Outputs: one under another, each ruled off with its settings */
        .df-outputs-scroll {
          overflow-y: auto;
        }

        .df-outputs-list {
          display: flex;
          flex-direction: column;
        }

        .df-output-card {
          border: none !important;
          border-bottom: 1px solid var(--oz-rule) !important;
        }

        .df-output-text-wrap {
          position: relative;
        }

        .df-output-monitor {
          padding: 0 var(--oz-gutter) 1.25rem;
        }

        .df-output-monitor .probe-monitor {
          margin-top: 0;
        }

        .df-output-text {
          padding: 1.5rem var(--oz-gutter);
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .df-output-settings {
          display: flex;
          align-items: stretch;
          padding: 0 calc(var(--oz-gutter) - 0.875rem);
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
          padding: 0.6rem 0.875rem 0.7rem;
          background: none;
          border: none;
          font: inherit;
          text-align: left;
          cursor: pointer;
          transition: background 0.15s ease;
        }

        .df-output-detail:hover {
          background: var(--oz-fill);
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
          gap: 0.35rem;
        }

        .df-output-detail-icon {
          width: 12px;
          height: 12px;
          color: var(--oz-fg-faint);
          transition: color 0.15s ease;
        }

        .df-output-detail:hover .df-output-detail-icon {
          color: var(--oz-mark);
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
          padding: 0.6rem 0.875rem 0.7rem;
        }

        .df-cursor {
          animation: df-blink 0.8s infinite;
          margin-left: 2px;
          color: var(--oz-mark) !important;
          font-weight: normal;
        }

        @keyframes df-blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }

        .df-output-actions {
          position: absolute;
          top: 1rem;
          right: var(--oz-gutter);
          display: flex;
          gap: 6px;
          opacity: 0;
          transition: opacity 0.15s ease;
        }

        .df-output-actions > * {
          background: var(--oz-bg) !important;
        }

        .df-output-card:hover .df-output-actions,
        .df-output-actions:focus-within {
          opacity: 1;
        }

        /* An output's prompt or model card */
        .df-detail-modal {
          display: flex;
          flex-direction: column;
          width: min(640px, calc(100vw - 2rem));
          max-width: none;
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
          border-bottom: 1px solid var(--oz-rule);
          flex-shrink: 0;
        }

        .df-detail-header h3 {
          flex: 1;
          min-width: 0;
          margin: 0;
          font-size: 1.25rem;
          font-weight: 500;
          letter-spacing: -0.025em;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .df-detail-actions {
          display: flex;
          gap: 0.25rem;
          flex-shrink: 0;
        }

        .df-detail-body {
          padding: 1.5rem;
          overflow-y: auto;
          min-height: 0;
        }

        .df-detail-prompt {
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        @media (max-width: 900px) {
          .df-page.oz-page { height: auto; overflow: visible; }
          .df-layout.oz-page__body { flex-direction: column; }
          .df-left.oz-side {
            width: 100%;
            max-width: 100%;
            border-right: none;
            border-bottom: 1px solid var(--oz-rule);
          }
        }
      `}</style>
    </div>
  )
}
