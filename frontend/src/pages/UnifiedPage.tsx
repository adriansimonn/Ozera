/**
 * Unified interface combining text generation and visualizations.
 * Supports single view and split screen modes.
 * Uses lazy loading to fetch activation data on-demand for better performance.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import TextGenerator from '../components/model/TextGenerator'
import ModelInfoBox from '../components/model/ModelInfoBox'
import { AttentionHeatmap } from '../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../components/visualization/TransformationFlow'
import { GenerationFlow } from '../components/visualization/GenerationFlow'
import { NavBar } from '../components/common/NavBar'
import { apiClient } from '../api/client'
import type { ActivationData, ActivationSummaryWithInfo, LayerActivations, TensorData, TopKLogits } from '../types/model'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network } from 'lucide-react'
import { Dropdown } from '../components/common/Dropdown'

type ViewMode = 'single' | 'split'
type SingleViewType = 'generator' | 'visualizations'
type VisualizationType = 'network' | 'attention' | 'activations' | 'journey' | 'flow'

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

  // Activation data state - uses lazy loading
  const [activationData, setActivationData] = useState<ActivationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [layerLoading, setLayerLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Cache for lazily loaded layers and tensors
  const layerCacheRef = useRef<Map<number, LayerActivations>>(new Map())
  const tensorCacheRef = useRef<Map<string, TensorData | TopKLogits>>(new Map())
  const currentActivationIdRef = useRef<string | null>(null)
  const summaryRef = useRef<ActivationSummaryWithInfo | null>(null)
  // Track which visualization's data has been loaded to avoid redundant fetches
  const loadedVizRef = useRef<Set<string>>(new Set())

  // Visualization controls
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)
  const [tokenInputValue, setTokenInputValue] = useState('0')
  const [showTextLabels, setShowTextLabels] = useState(false)

  // Build ActivationData from summary + cached layers/tensors
  const buildActivationData = useCallback((summary: ActivationSummaryWithInfo): ActivationData => {
    const layers: LayerActivations[] = []
    for (let i = 0; i < summary.num_layers; i++) {
      layers.push(layerCacheRef.current.get(i) || {})
    }

    return {
      id: summary.id,
      prompt: summary.prompt,
      model: summary.model,
      timestamp: summary.timestamp,
      tokens: summary.tokens,
      metadata: summary.metadata,
      activations: {
        token_embeddings: tensorCacheRef.current.get('token_embeddings') as TensorData | undefined,
        positional_embeddings: tensorCacheRef.current.get('positional_embeddings') as TensorData | undefined,
        combined_embeddings: tensorCacheRef.current.get('combined_embeddings') as TensorData | undefined,
        final_layer_norm: tensorCacheRef.current.get('final_layer_norm') as TensorData | undefined,
        logits: tensorCacheRef.current.get('logits') as TensorData | undefined,
        top_k_logits: tensorCacheRef.current.get('top_k_logits') as TopKLogits | undefined,
        layers,
      },
    }
  }, [])

  // Load a specific tensor by name and merge into activation data
  const loadTensor = useCallback(async (tensorName: string): Promise<void> => {
    const id = currentActivationIdRef.current
    if (!id || tensorCacheRef.current.has(tensorName)) return

    try {
      const result = await apiClient.getTensorActivation(id, tensorName)
      if (currentActivationIdRef.current !== id) return // stale
      tensorCacheRef.current.set(tensorName, result.data)
    } catch (err) {
      console.error(`Failed to load tensor ${tensorName}:`, err)
    }
  }, [])

  // Load a specific layer and merge into activation data
  const loadLayer = useCallback(async (layerIdx: number): Promise<void> => {
    const id = currentActivationIdRef.current
    if (!id || layerCacheRef.current.has(layerIdx)) return

    try {
      const result = await apiClient.getLayerActivations(id, layerIdx)
      if (currentActivationIdRef.current !== id) return // stale
      layerCacheRef.current.set(layerIdx, result.activations as LayerActivations)
    } catch (err) {
      console.error(`Failed to load layer ${layerIdx}:`, err)
    }
  }, [])

  // Load data needed for a specific visualization type
  const loadDataForVisualization = useCallback(async (vizType: VisualizationType, layer?: number) => {
    const summary = summaryRef.current
    if (!summary) return

    const id = currentActivationIdRef.current

    try {
      setLayerLoading(true)

      if (vizType === 'network') {
        if (!loadedVizRef.current.has('network')) {
          // GenerationFlow needs: top_k_logits, combined_embeddings (or token_embeddings), and all layers
          const promises: Promise<void>[] = []
          if (summary.tensor_info.top_k_logits) promises.push(loadTensor('top_k_logits'))
          if (summary.tensor_info.combined_embeddings) promises.push(loadTensor('combined_embeddings'))
          else if (summary.tensor_info.token_embeddings) promises.push(loadTensor('token_embeddings'))
          // Load all layers for node activation visualization
          for (let i = 0; i < summary.num_layers; i++) {
            promises.push(loadLayer(i))
          }
          await Promise.all(promises)
          if (currentActivationIdRef.current === id) loadedVizRef.current.add('network')
        }
      } else if (vizType === 'attention' || vizType === 'activations') {
        // Only need the selected layer
        const targetLayer = layer ?? 0
        await loadLayer(targetLayer)
      } else if (vizType === 'journey' || vizType === 'flow') {
        if (!loadedVizRef.current.has('journey_flow')) {
          // Need all embeddings, all layers, and final_layer_norm
          const promises: Promise<void>[] = []
          if (summary.tensor_info.token_embeddings) promises.push(loadTensor('token_embeddings'))
          if (summary.tensor_info.combined_embeddings) promises.push(loadTensor('combined_embeddings'))
          if (summary.tensor_info.final_layer_norm) promises.push(loadTensor('final_layer_norm'))
          for (let i = 0; i < summary.num_layers; i++) {
            promises.push(loadLayer(i))
          }
          await Promise.all(promises)
          if (currentActivationIdRef.current === id) loadedVizRef.current.add('journey_flow')
        }
      }

      // Rebuild activation data from caches
      if (currentActivationIdRef.current === id) {
        setActivationData(buildActivationData(summary))
      }
    } finally {
      setLayerLoading(false)
    }
  }, [buildActivationData, loadTensor, loadLayer])

  useEffect(() => {
    if (urlActivationId) {
      setCurrentActivationId(urlActivationId)
      loadActivationSummary(urlActivationId)
    }
  }, [urlActivationId])

  // Load summary first, then load data for current visualization
  async function loadActivationSummary(id: string) {
    try {
      setLoading(true)
      setError(null)

      // Clear caches if loading a new activation
      if (currentActivationIdRef.current !== id) {
        layerCacheRef.current.clear()
        tensorCacheRef.current.clear()
        loadedVizRef.current.clear()
        currentActivationIdRef.current = id
      }

      // Get lightweight summary first (metadata + shapes, no tensor values)
      const summary = await apiClient.getActivationSummary(id)
      summaryRef.current = summary

      // Build skeleton ActivationData immediately so UI can render
      setActivationData(buildActivationData(summary))
      setLoading(false)

      // Then load data needed for the current visualization
      await loadDataForVisualization(selectedVisualization, selectedLayer)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activations')
      setLoading(false)
    }
  }

  // Load data when visualization type changes
  useEffect(() => {
    if (summaryRef.current && currentActivationIdRef.current) {
      loadDataForVisualization(selectedVisualization, selectedLayer)
    }
  }, [selectedVisualization, loadDataForVisualization])

  // Load layer data when selected layer changes (for attention/activations views)
  useEffect(() => {
    if (summaryRef.current && (selectedVisualization === 'attention' || selectedVisualization === 'activations')) {
      loadDataForVisualization(selectedVisualization, selectedLayer)
    }
  }, [selectedLayer, selectedVisualization, loadDataForVisualization])

  const handleActivationGenerated = (newActivationId: string) => {
    setCurrentActivationId(newActivationId)
    loadActivationSummary(newActivationId)
  }

  const numLayers = activationData?.activations.layers?.length || summaryRef.current?.num_layers || 0
  // Try loaded layer data first, then fall back to summary layer_info for head count
  const numHeads = activationData?.activations.layers?.[0]?.attn_weights?.shape[1]
    || summaryRef.current?.layer_info?.[0]?.attn_weights?.shape[1]
    || 0

  const visualizationOptions = [
    { value: 'network' as const, label: 'Neural Network Generation Flow', icon: Network },
    { value: 'attention' as const, label: 'Attention Heatmap', icon: Eye },
    { value: 'activations' as const, label: 'Layer Activations', icon: Sparkles },
    { value: 'journey' as const, label: 'Embedding Journey', icon: TrendingUp },
    { value: 'flow' as const, label: 'Transformation Flow', icon: Layers },
  ]

  const renderVisualization = () => {
    if (generating) {
      return (
        <div className="visualization-placeholder">
          <div className="placeholder-content">
            <div className="spinner" />
            <p className="placeholder-text">Generating and capturing activations...</p>
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
              onGeneratingChange={setGenerating}
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
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-selector-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.7);
        }

        .view-selector-btn.active {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.4);
          color: rgba(255, 255, 255, 0.95);
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
          color: rgba(255, 255, 255, 0.7);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .visualization-select {
          flex: 1;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
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
          color: rgba(255, 255, 255, 0.5);
        }

        .control-label {
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.6);
          font-weight: 500;
        }

        .control-btn {
          padding: 0.375rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.7);
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
          color: rgba(255, 255, 255, 0.9);
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
          color: rgba(255, 255, 255, 0.7);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .labels-select {
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
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
          color: rgba(255, 255, 255, 0.5);
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
          color: rgba(255, 255, 255, 0.4);
          font-family: monospace;
        }

        .token-text {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.75);
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
          color: rgba(255, 255, 255, 0.4);
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
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.875rem;
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
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 600;
        }

        .metadata-value {
          font-size: 0.875rem;
          color: rgba(255, 255, 255, 0.9);
          font-family: monospace;
          font-weight: 500;
        }

        /* Light mode */
        [data-bg="light"] .view-selector-btn { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: rgba(0,0,0,0.5); }
        [data-bg="light"] .view-selector-btn:hover { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); color: rgba(0,0,0,0.7); }
        [data-bg="light"] .view-selector-btn.active { background: rgba(0,0,0,0.08); border-color: rgba(0,0,0,0.25); color: #1d1d1f; }
        [data-bg="light"] .generator-section > :first-child { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .model-info-section { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .visualization-header { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .visualization-dropdown label { color: rgba(0,0,0,0.55); }
        [data-bg="light"] .visualization-select { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .visualization-select:hover { border-color: rgba(0,0,0,0.2); }
        [data-bg="light"] .visualization-select:focus { border-color: rgba(0,0,0,0.3); }
        [data-bg="light"] .control-icon { color: rgba(0,0,0,0.45); }
        [data-bg="light"] .control-label { color: rgba(0,0,0,0.55); }
        [data-bg="light"] .control-btn { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: rgba(0,0,0,0.6); }
        [data-bg="light"] .control-btn:hover:not(:disabled) { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); }
        [data-bg="light"] .control-value { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.1); color: #1d1d1f; }
        [data-bg="light"] .control-input:focus { border-color: rgba(0,0,0,0.2); background: rgba(0,0,0,0.06); }
        [data-bg="light"] .labels-dropdown label { color: rgba(0,0,0,0.55); }
        [data-bg="light"] .labels-select { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .labels-select:hover { border-color: rgba(0,0,0,0.2); }
        [data-bg="light"] .labels-select:focus { border-color: rgba(0,0,0,0.3); }
        [data-bg="light"] .token-reference-box { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .token-reference-header { color: rgba(0,0,0,0.5); }
        [data-bg="light"] .token-chip { background: rgba(0,0,0,0.04); border-color: rgba(0,0,0,0.08); }
        [data-bg="light"] .token-chip:hover { background: rgba(0,0,0,0.08); }
        [data-bg="light"] .token-idx { color: rgba(0,0,0,0.4); }
        [data-bg="light"] .token-text { color: rgba(0,0,0,0.65); }
        [data-bg="light"] .visualization-content { background: rgba(0,0,0,0.02); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .placeholder-content { color: rgba(0,0,0,0.4); }
        [data-bg="light"] .placeholder-error { color: rgba(0,0,0,0.6); }
        [data-bg="light"] .spinner { border-color: rgba(0,0,0,0.15); border-top-color: #1d1d1f; }
        [data-bg="light"] .metadata-card { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .metadata-label { color: rgba(0,0,0,0.5); }
        [data-bg="light"] .metadata-value { color: #1d1d1f; }
      `}</style>
    </div>
  )
}

export default UnifiedPage
