/**
 * Unified interface combining text generation and visualizations.
 * Supports single view and split screen modes.
 * Uses lazy loading to fetch activation data on-demand for better performance.
 */

import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import TextGenerator from '../components/model/TextGenerator'
import type { GenerationStreamState } from '../components/model/TextGenerator'
import ModelInfoBox from '../components/model/ModelInfoBox'
import { AttentionHeatmap } from '../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../components/visualization/TransformationFlow'
import { GenerationFlow } from '../components/visualization/GenerationFlow'
import { NavBar } from '../components/common/NavBar'
import { useActivationData, type VisualizationType } from '../hooks/useActivationData'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network } from 'lucide-react'
import { Dropdown } from '../components/common/Dropdown'

type ViewMode = 'single' | 'split'
type SingleViewType = 'generator' | 'visualizations'

interface UnifiedPageProps {
  onShowPurchaseCredits?: () => void
}

export function UnifiedPage({ onShowPurchaseCredits }: UnifiedPageProps) {
  const [searchParams] = useSearchParams()
  const urlActivationId = searchParams.get('id')

  // Current activation ID (from URL or newly generated)
  const [currentActivationId, setCurrentActivationId] = useState<string | null>(urlActivationId)

  // View mode state
  const [viewMode, setViewMode] = useState<ViewMode>('split')
  const [singleViewType, setSingleViewType] = useState<SingleViewType>('generator')
  const [selectedVisualization, setSelectedVisualization] = useState<VisualizationType>('network')

  // Model info box state (independent from TextGenerator)
  const [infoBoxModel, setInfoBoxModel] = useState<string>('nano')

  // The generation in progress, if any
  const [stream, setStream] = useState<GenerationStreamState | null>(null)
  const streamBoxRef = useRef<HTMLDivElement>(null)

  // Visualization controls
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)
  const [tokenInputValue, setTokenInputValue] = useState('0')
  const [showTextLabels, setShowTextLabels] = useState(false)

  // Activation data state - uses lazy loading
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
    loadActivation(newActivationId)
  }

  // Keep the newest streamed text in view, unless the user has scrolled up to read
  useEffect(() => {
    const el = streamBoxRef.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) {
      el.scrollTop = el.scrollHeight
    }
  }, [stream?.text])

  const numLayers = activationData?.activations.layers?.length || summary?.num_layers || 0
  // Try loaded layer data first, then fall back to summary layer_info for head count
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

  const renderVisualization = () => {
    if (stream?.mode === 'visualize') {
      return (
        <div className="visualization-placeholder">
          <div className="placeholder-content generating-content">
            <div className="spinner" />
            <p className="placeholder-text">
              {stream.phase === 'capturing' ? 'Capturing activations...' :
               stream.phase === 'stopping' ? 'Stopping generation...' :
               'Generating and capturing activations...'}
            </p>
            {stream.text && (
              <div ref={streamBoxRef} className="stream-box">
                {stream.text}
                {stream.phase === 'streaming' && <span className="stream-cursor">|</span>}
              </div>
            )}
          </div>
        </div>
      )
    }

    if (loading) {
      return (
        <div className="visualization-placeholder">
          <div className="placeholder-content">
            <div className="spinner" />
            <p className="placeholder-text">Loading activations...</p>
          </div>
        </div>
      )
    }

    if (!activationData) {
      return (
        <div className="visualization-placeholder">
          <div className="placeholder-content">
            <Network className="placeholder-icon" />
            <p className="placeholder-text">Generate text with visualizations to see activations here</p>
          </div>
        </div>
      )
    }

    if (error) {
      return (
        <div className="visualization-placeholder">
          <div className="placeholder-content">
            <p className="placeholder-error">{error}</p>
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
          ) : layerLoading ? (
            <div className="visualization-placeholder">
              <div className="placeholder-content">
                <div className="spinner" />
                <p className="placeholder-text">Loading layer {selectedLayer} attention data...</p>
              </div>
            </div>
          ) : null
        )}

        {selectedVisualization === 'activations' && (
          activationData.activations.layers?.[selectedLayer]?.attn_output || activationData.activations.layers?.[selectedLayer]?.ff_output ? (
            <LayerActivationDisplay
              layerActivations={activationData.activations.layers[selectedLayer]}
              layerIndex={selectedLayer}
              activationId={currentActivationId ?? undefined}
            />
          ) : layerLoading ? (
            <div className="visualization-placeholder">
              <div className="placeholder-content">
                <div className="spinner" />
                <p className="placeholder-text">Loading layer {selectedLayer} activations...</p>
              </div>
            </div>
          ) : null
        )}

        {selectedVisualization === 'journey' && (
          <EmbeddingJourney
            activationData={activationData}
            selectedTokenIndex={selectedTokenIndex}
          />
        )}

        {selectedVisualization === 'flow' && (
          <TransformationFlow
            activationData={activationData}
            selectedTokenIndex={selectedTokenIndex}
          />
        )}
      </>
    )
  }

  const renderVisualizationControls = () => {
    if (!activationData || selectedVisualization === 'network') return null

    return (
      <div className="visualization-controls">
        {(selectedVisualization === 'attention' || selectedVisualization === 'activations') && (
          <div className="control-item">
            <Layers className="control-icon" />
            <span className="control-label">Layer:</span>
            <button
              onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
              disabled={selectedLayer === 0}
              className="control-btn"
            >
              <ChevronLeft className="btn-icon" />
            </button>
            <span className="control-value">{selectedLayer}</span>
            <button
              onClick={() => setSelectedLayer(Math.min(numLayers - 1, selectedLayer + 1))}
              disabled={selectedLayer === numLayers - 1}
              className="control-btn"
            >
              <ChevronRight className="btn-icon" />
            </button>
          </div>
        )}

        {selectedVisualization === 'attention' && (
          <div className="control-item">
            <span className="control-label">Head:</span>
            <button
              onClick={() => setSelectedHead(Math.max(0, selectedHead - 1))}
              disabled={selectedHead === 0}
              className="control-btn"
            >
              <ChevronLeft className="btn-icon" />
            </button>
            <span className="control-value">{selectedHead}</span>
            <button
              onClick={() => setSelectedHead(Math.min(numHeads - 1, selectedHead + 1))}
              disabled={selectedHead === numHeads - 1}
              className="control-btn"
            >
              <ChevronRight className="btn-icon" />
            </button>
          </div>
        )}

        {(selectedVisualization === 'journey' || selectedVisualization === 'flow') && activationData && (
          <div className="control-item">
            <span className="control-label">Token:</span>
            <button
              onClick={() => {
                const newVal = Math.max(0, selectedTokenIndex - 1);
                setSelectedTokenIndex(newVal);
                setTokenInputValue(String(newVal));
              }}
              disabled={selectedTokenIndex === 0}
              className="control-btn"
            >
              <ChevronLeft className="btn-icon" />
            </button>
            <input
              type="text"
              className="control-value control-input"
              value={tokenInputValue}
              onChange={(e) => {
                const val = e.target.value;
                setTokenInputValue(val);
                if (val === '') return;
                const num = parseInt(val, 10);
                if (!isNaN(num) && num >= 0 && num <= activationData.tokens.length - 1) {
                  setSelectedTokenIndex(num);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  const newVal = Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1);
                  setSelectedTokenIndex(newVal);
                  setTokenInputValue(String(newVal));
                } else if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  const newVal = Math.max(0, selectedTokenIndex - 1);
                  setSelectedTokenIndex(newVal);
                  setTokenInputValue(String(newVal));
                }
              }}
              style={{ width: `${Math.max(8, tokenInputValue.length + 2)}ch`, textAlign: 'center' }}
            />
            <button
              onClick={() => {
                const newVal = Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1);
                setSelectedTokenIndex(newVal);
                setTokenInputValue(String(newVal));
              }}
              disabled={selectedTokenIndex === activationData.tokens.length - 1}
              className="control-btn"
            >
              <ChevronRight className="btn-icon" />
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="unified-page">
      <NavBar
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        showViewToggle={true}
      />

      <div className={`page-content ${viewMode === 'split' ? 'split-view' : 'single-view'}`}>
        {viewMode === 'single' && (
          <div className="single-view-selector">
            <button
              onClick={() => setSingleViewType('generator')}
              className={`view-selector-btn ${singleViewType === 'generator' ? 'active' : ''}`}
            >
              Text Generation
            </button>
            <button
              onClick={() => setSingleViewType('visualizations')}
              className={`view-selector-btn ${singleViewType === 'visualizations' ? 'active' : ''}`}
            >
              Visualizations
            </button>
          </div>
        )}
        {(viewMode === 'split' || singleViewType === 'generator') && (
          <div className="generator-section">
            <TextGenerator
              defaultModel="nano"
              onActivationGenerated={handleActivationGenerated}
              onStreamUpdate={setStream}
              onModelChange={setInfoBoxModel}
              onShowPurchaseCredits={onShowPurchaseCredits}
            />
            <div className="model-info-section">
              <ModelInfoBox
                selectedModel={infoBoxModel}
                onModelChange={setInfoBoxModel}
              />
            </div>
          </div>
        )}

        {(viewMode === 'split' || singleViewType === 'visualizations') && (
          <div className="visualization-section">
            <div className="visualization-header">
              <div className="visualization-dropdown">
                <label htmlFor="vis-select">Visualization:</label>
                <Dropdown
                  id="vis-select"
                  value={selectedVisualization}
                  onChange={(v) => setSelectedVisualization(v as VisualizationType)}
                  options={visualizationOptions.map((opt) => ({ value: opt.value, label: opt.label }))}
                />
              </div>

              {selectedVisualization === 'attention' && activationData && (
                <div className="labels-dropdown">
                  <label htmlFor="labels-select">Labels:</label>
                  <Dropdown
                    id="labels-select"
                    value={showTextLabels ? 'text' : 'number'}
                    onChange={(v) => setShowTextLabels(v === 'text')}
                    options={[
                      { value: 'number', label: 'Token Number' },
                      { value: 'text', label: 'Token Text' },
                    ]}
                  />
                </div>
              )}

              {renderVisualizationControls()}
            </div>

            <div className="visualization-content">
              {renderVisualization()}
            </div>

            {(selectedVisualization === 'attention' || selectedVisualization === 'journey' || selectedVisualization === 'flow') && activationData?.metadata.decoded_tokens && (
              <div className="token-reference-box">
                <div className="token-reference-header">Token Reference</div>
                <div className="token-reference-list">
                  {activationData.metadata.decoded_tokens.map((token, idx) => (
                    <span
                      key={idx}
                      className="token-chip"
                      title={`Token ${idx}: "${token}"`}
                    >
                      <span className="token-idx">{idx}</span>
                      <span className="token-text">{token.replace(/ /g, '·') || '▯'}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {activationData && (
              <div className="visualization-metadata">
                <div className="metadata-card">
                  <span className="metadata-label">Model</span>
                  <span className="metadata-value">{activationData.model}</span>
                </div>
                <div className="metadata-card">
                  <span className="metadata-label">Tokens</span>
                  <span className="metadata-value">{activationData.tokens.length}</span>
                </div>
                <div className="metadata-card">
                  <span className="metadata-label">Temperature</span>
                  <span className="metadata-value">{activationData.metadata.temperature?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metadata-card">
                  <span className="metadata-label">Top-K</span>
                  <span className="metadata-value">{activationData.metadata.top_k?.toString() || 'N/A'}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <style>{`
        .unified-page {
          min-height: 125vh;
          padding-top: 70px;
        }

        .page-content {
          max-width: calc(100% - 4rem);
          margin: 0 auto;
          padding: 2rem;
        }

        @media (min-width: 1600px) {
          .page-content {
            max-width: calc(100% - 6rem);
          }
        }

        @media (min-width: 2000px) {
          .page-content {
            max-width: 1900px;
          }
        }

        .single-view-selector {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
          justify-content: center;
        }

        .view-selector-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #ffffff;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-selector-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: #ffffff;
        }

        .view-selector-btn.active {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.4);
          color: #ffffff;
        }

        .page-content.split-view {
          display: grid;
          grid-template-columns: 1fr 2fr;
          gap: 1.5rem;
        }

        .page-content.single-view {
          display: block;
        }

        @media (max-width: 1200px) {
          .page-content.split-view {
            grid-template-columns: 1fr;
          }
        }

        .generator-section {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          height: fit-content;
        }

        .generator-section > :first-child {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .model-info-section {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .visualization-section {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .visualization-header {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          align-items: center;
        }

        .visualization-dropdown {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex: 1;
          min-width: 250px;
        }

        .visualization-dropdown label {
          font-weight: 600;
          font-size: 0.8rem;
          color: #ffffff;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .visualization-select {
          flex: 1;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .visualization-select:hover {
          border-color: rgba(255, 255, 255, 0.25);
        }

        .visualization-select:focus {
          outline: none;
          border-color: rgba(255, 255, 255, 0.4);
        }

        .visualization-controls {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          align-items: center;
        }

        .control-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .control-icon {
          width: 16px;
          height: 16px;
          color: #ffffff;
        }

        .control-label {
          font-size: 0.8rem;
          color: #ffffff;
          font-weight: 500;
        }

        .control-btn {
          padding: 0.375rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #ffffff;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .control-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .control-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .btn-icon {
          width: 14px;
          height: 14px;
        }

        .control-value {
          padding: 0.375rem 0.75rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #ffffff;
          font-family: monospace;
          font-size: 0.8rem;
          min-width: 40px;
          text-align: center;
        }

        .control-input {
          outline: none;
          cursor: text;
          border-radius: 4px;
        }

        .control-input:focus {
          border-color: rgba(255, 255, 255, 0.3);
          background: rgba(0, 0, 0, 0.5);
        }

        .labels-dropdown {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .labels-dropdown label {
          font-weight: 600;
          font-size: 0.8rem;
          color: #ffffff;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .labels-select {
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.875rem;
          min-width: 150px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .labels-select:hover {
          border-color: rgba(255, 255, 255, 0.25);
        }

        .labels-select:focus {
          outline: none;
          border-color: rgba(255, 255, 255, 0.4);
        }

        .token-reference-box {
          padding: 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .token-reference-header {
          font-size: 0.7rem;
          color: #ffffff;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }

        .token-reference-list {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }

        .token-chip {
          display: inline-flex;
          align-items: baseline;
          gap: 3px;
          padding: 2px 6px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          cursor: default;
          transition: background 0.15s;
        }

        .token-chip:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .token-idx {
          font-size: 10px;
          color: #ffffff;
          font-family: monospace;
        }

        .token-text {
          font-size: 12px;
          color: #ffffff;
          font-family: monospace;
        }

        .visualization-content {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          min-height: 500px;
          overflow: auto;
        }

        .visualization-placeholder {
          min-height: 500px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2rem;
        }

        .placeholder-content {
          text-align: center;
          color: #ffffff;
          max-width: 400px;
        }

        .placeholder-icon {
          width: 48px;
          height: 48px;
          margin: 0 auto 1rem;
          opacity: 0.3;
        }

        .placeholder-text {
          font-size: 0.875rem;
          margin: 0;
        }

        .placeholder-error {
          color: #ffffff;
          font-size: 0.875rem;
        }

        .generating-content {
          width: 100%;
          max-width: 720px;
        }

        .stream-box {
          max-height: 45vh;
          overflow-y: auto;
          margin-top: 1.5rem;
          padding: 1rem 1.25rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          text-align: left;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.85rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .stream-cursor {
          margin-left: 2px;
          animation: stream-blink 0.8s infinite;
        }

        @keyframes stream-blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 2px solid rgba(255, 255, 255, 0.1);
          border-top-color: rgba(255, 255, 255, 0.8);
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 1rem;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .visualization-metadata {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 0.75rem;
        }

        .metadata-card {
          padding: 0.875rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          display: flex;
          flex-direction: column;
          gap: 0.375rem;
        }

        .metadata-label {
          font-size: 0.7rem;
          color: #ffffff;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 600;
        }

        .metadata-value {
          font-size: 0.875rem;
          color: #ffffff;
          font-family: monospace;
          font-weight: 500;
        }

        /* Light mode */
        [data-bg="light"] .view-selector-btn { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: #1d1d1f; }
        [data-bg="light"] .view-selector-btn:hover { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); color: #1d1d1f; }
        [data-bg="light"] .view-selector-btn.active { background: rgba(0,0,0,0.08); border-color: rgba(0,0,0,0.25); color: #1d1d1f; }
        [data-bg="light"] .generator-section > :first-child { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .model-info-section { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .visualization-header { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .visualization-dropdown label { color: #1d1d1f; }
        [data-bg="light"] .visualization-select { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .visualization-select:hover { border-color: rgba(0,0,0,0.2); }
        [data-bg="light"] .visualization-select:focus { border-color: rgba(0,0,0,0.3); }
        [data-bg="light"] .control-icon { color: #1d1d1f; }
        [data-bg="light"] .control-label { color: #1d1d1f; }
        [data-bg="light"] .control-btn { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: #1d1d1f; }
        [data-bg="light"] .control-btn:hover:not(:disabled) { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); }
        [data-bg="light"] .control-value { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: #1d1d1f; }
        [data-bg="light"] .control-input:focus { border-color: rgba(0,0,0,0.2); background: rgba(0,0,0,0.06); }
        [data-bg="light"] .labels-dropdown label { color: #1d1d1f; }
        [data-bg="light"] .labels-select { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .labels-select:hover { border-color: rgba(0,0,0,0.2); }
        [data-bg="light"] .labels-select:focus { border-color: rgba(0,0,0,0.3); }
        [data-bg="light"] .token-reference-box { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .token-reference-header { color: #1d1d1f; }
        [data-bg="light"] .token-chip { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.08); }
        [data-bg="light"] .token-chip:hover { background: rgba(0,0,0,0.08); }
        [data-bg="light"] .token-idx { color: #1d1d1f; }
        [data-bg="light"] .token-text { color: #1d1d1f; }
        [data-bg="light"] .visualization-content { background: rgba(0,0,0,0.02); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .placeholder-content { color: #1d1d1f; }
        [data-bg="light"] .placeholder-error { color: #1d1d1f; }
        [data-bg="light"] .spinner { border-color: rgba(0,0,0,0.15); border-top-color: #1d1d1f; }
        [data-bg="light"] .stream-box { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); color: #1d1d1f; }
        [data-bg="light"] .metadata-card { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .metadata-label { color: #1d1d1f; }
        [data-bg="light"] .metadata-value { color: #1d1d1f; }
      `}</style>
    </div>
  )
}

export default UnifiedPage
