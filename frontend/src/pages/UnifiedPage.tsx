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
import type { ActivationData, LayerActivations } from '../types/model'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network } from 'lucide-react'

type ViewMode = 'single' | 'split'
type SingleViewType = 'generator' | 'visualizations'
type VisualizationType = 'network' | 'attention' | 'activations' | 'journey' | 'flow'

interface UnifiedPageProps {
  onShowLogin: () => void
  onShowSignup: () => void
  onShowPurchaseCredits?: () => void
}

export function UnifiedPage({ onShowLogin, onShowSignup, onShowPurchaseCredits }: UnifiedPageProps) {
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

  // Cache for lazily loaded layers
  const layerCacheRef = useRef<Map<number, LayerActivations>>(new Map())
  const currentActivationIdRef = useRef<string | null>(null)

  // Visualization controls
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)

  useEffect(() => {
    if (urlActivationId) {
      setCurrentActivationId(urlActivationId)
      loadActivationSummary(urlActivationId)
    }
  }, [urlActivationId])

  // Load only summary initially (lazy loading)
  async function loadActivationSummary(id: string) {
    try {
      setLoading(true)
      setError(null)

      // Clear cache if loading a new activation
      if (currentActivationIdRef.current !== id) {
        layerCacheRef.current.clear()
        currentActivationIdRef.current = id
      }

      // Get full activation data - for now we still load everything
      // but the backend sends summary first and we can progressively load
      const data = await apiClient.getActivations(id)
      setActivationData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activations')
    } finally {
      setLoading(false)
    }
  }

  // Lazy load a specific layer's activations
  const loadLayerActivations = useCallback(async (layerIdx: number) => {
    if (!currentActivationIdRef.current) return null

    // Check cache first
    if (layerCacheRef.current.has(layerIdx)) {
      return layerCacheRef.current.get(layerIdx)!
    }

    // If we already have the data in activationData, use it
    if (activationData?.activations.layers?.[layerIdx]) {
      layerCacheRef.current.set(layerIdx, activationData.activations.layers[layerIdx])
      return activationData.activations.layers[layerIdx]
    }

    // Otherwise, fetch from API
    try {
      setLayerLoading(true)
      const result = await apiClient.getLayerActivations(currentActivationIdRef.current, layerIdx)
      layerCacheRef.current.set(layerIdx, result.activations as LayerActivations)
      return result.activations as LayerActivations
    } catch (err) {
      console.error(`Failed to load layer ${layerIdx}:`, err)
      return null
    } finally {
      setLayerLoading(false)
    }
  }, [activationData])

  // Preload layer when selected layer changes
  useEffect(() => {
    if (activationData && (selectedVisualization === 'attention' || selectedVisualization === 'activations')) {
      loadLayerActivations(selectedLayer)
    }
  }, [selectedLayer, selectedVisualization, activationData, loadLayerActivations])

  const handleActivationGenerated = (newActivationId: string) => {
    setCurrentActivationId(newActivationId)
    loadActivationSummary(newActivationId)
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

        {selectedVisualization === 'attention' && activationData.activations.layers?.[selectedLayer]?.attn_weights && (
          <AttentionHeatmap
            attentionWeights={activationData.activations.layers[selectedLayer].attn_weights!}
            layerIndex={selectedLayer}
            headIndex={selectedHead}
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
              onClick={() => setSelectedTokenIndex(Math.max(0, selectedTokenIndex - 1))}
              disabled={selectedTokenIndex === 0}
              className="control-btn"
            >
              <ChevronLeft className="btn-icon" />
            </button>
            <span className="control-value">{selectedTokenIndex}</span>
            <button
              onClick={() => setSelectedTokenIndex(Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1))}
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
        onShowLogin={onShowLogin}
        onShowSignup={onShowSignup}
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
                <select
                  id="vis-select"
                  value={selectedVisualization}
                  onChange={(e) => setSelectedVisualization(e.target.value as VisualizationType)}
                  className="visualization-select"
                >
                  {visualizationOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {renderVisualizationControls()}
            </div>

            <div className="visualization-content">
              {renderVisualization()}
            </div>

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
          min-height: 100vh;
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
      `}</style>
    </div>
  )
}

export default UnifiedPage
