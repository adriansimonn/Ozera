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

      <div className={`page-content ${viewMode === 'split' ? 'split-view' : 'single-view'}`}>
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
          padding-top: 100px;
        }

        .page-main {
          padding: 2rem;
        }

        .single-view-selector {
          max-width: 1800px;
          margin: 0 auto 2rem;
          display: flex;
          gap: 1rem;
          justify-content: center;
        }

        .view-selector-btn {
          padding: 1rem 2rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.7);
          font-size: 1rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.05em;
        }

        .view-selector-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.25);
        }

        .view-selector-btn.active {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.35);
          color: #ffffff;
          box-shadow: 0 0 20px rgba(255, 255, 255, 0.15);
        }

        .page-content {
          max-width: 1800px;
          margin: 0 auto;
        }

        .page-content.split-view {
          display: grid;
          grid-template-columns: 1fr 2fr;
          gap: 2rem;
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
          gap: 1.5rem;
          height: fit-content;
        }

        .generator-section > :first-child {
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        }

        .model-info-section {
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        }

        .visualization-section {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .visualization-header {
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
          padding: 1.5rem;
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
          align-items: center;
        }

        .visualization-dropdown {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex: 1;
          min-width: 250px;
        }

        .visualization-dropdown label {
          font-weight: 500;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .visualization-select {
          flex: 1;
          padding: 0.875rem 1rem;
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          font-size: 0.95rem;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .visualization-select:hover {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .visualization-select:focus {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            0 0 0 2px rgba(255, 255, 255, 0.05);
          outline: none;
        }

        .visualization-controls {
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
          align-items: center;
        }

        .control-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .control-icon {
          width: 18px;
          height: 18px;
          color: rgba(255, 255, 255, 0.5);
        }

        .control-label {
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.6);
          font-weight: 500;
        }

        .control-btn {
          padding: 0.5rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: rgba(255, 255, 255, 0.8);
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .control-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .control-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .btn-icon {
          width: 16px;
          height: 16px;
        }

        .control-value {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.9rem;
          min-width: 50px;
          text-align: center;
        }

        .visualization-content {
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
          min-height: 600px;
          overflow: auto;
        }

        .visualization-placeholder {
          min-height: 600px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 3rem;
        }

        .placeholder-content {
          text-align: center;
          color: rgba(255, 255, 255, 0.4);
        }

        .placeholder-icon {
          width: 64px;
          height: 64px;
          margin: 0 auto 1.5rem;
          opacity: 0.3;
        }

        .placeholder-text {
          font-size: 1rem;
          margin: 0;
        }

        .placeholder-error {
          color: rgba(255, 255, 255, 0.6);
          font-size: 1rem;
        }

        .spinner {
          width: 48px;
          height: 48px;
          border: 3px solid rgba(255, 255, 255, 0.1);
          border-top-color: #ffffff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 1.5rem;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .visualization-metadata {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 1rem;
        }

        .metadata-card {
          padding: 1.25rem;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .metadata-label {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 400;
        }

        .metadata-value {
          font-size: 1rem;
          color: #ffffff;
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
        }
      `}</style>
    </div>
  )
}

export default UnifiedPage
