/**
 * Visualization page for exploring model activations.
 * Displays attention patterns and layer activations with interactive controls.
 */

import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { ActivationData } from '../types/model'
import { AttentionHeatmap } from '../components/visualization/AttentionHeatmap'
import { LayerActivationDisplay } from '../components/visualization/LayerActivationDisplay'
import { EmbeddingJourney } from '../components/visualization/EmbeddingJourney'
import { TransformationFlow } from '../components/visualization/TransformationFlow'
import { GenerationFlow } from '../components/visualization/GenerationFlow'
import { ChevronLeft, ChevronRight, Layers, Eye, Sparkles, TrendingUp, Network } from 'lucide-react'

export function VisualizationPage() {
  const [searchParams] = useSearchParams()
  const activationId = searchParams.get('id')

  const [activationData, setActivationData] = useState<ActivationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0)
  const [view, setView] = useState<'attention' | 'activations' | 'journey' | 'flow' | 'network'>('attention')

  useEffect(() => {
    if (!activationId) {
      setError('No activation ID provided')
      setLoading(false)
      return
    }

    loadActivations()
  }, [activationId])

  async function loadActivations() {
    if (!activationId) return

    try {
      setLoading(true)
      setError(null)
      const data = await apiClient.getActivations(activationId)
      setActivationData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activations')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin h-12 w-12 border-2 border-white border-t-transparent mb-4"></div>
          <div className="text-white opacity-60">Loading activations...</div>
        </div>
      </div>
    )
  }

  if (error || !activationData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-white mb-2">Error</div>
          <div className="text-white opacity-60">{error || 'No activation data available'}</div>
        </div>
      </div>
    )
  }

  const numLayers = activationData.activations.layers?.length || 0
  const numHeads = activationData.activations.layers?.[0]?.attn_weights?.shape[1] || 0

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="border-b border-white/[0.08] glass-strong sticky top-0 z-10">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                Activation Visualization
              </h1>
              <div className="text-sm text-white/60 mt-1">
                Model: <span className="text-white font-mono">{activationData.model}</span>
                {' • '}
                Tokens: <span className="text-white font-mono">{activationData.tokens.length}</span>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setView('attention')}
                className={`px-4 py-2 flex items-center gap-2 transition-all ${
                  view === 'attention'
                    ? 'bg-white/10 text-white border border-white/20'
                    : 'bg-white/[0.03] text-white/60 border border-white/[0.08] hover:bg-white/[0.05]'
                }`}
              >
                <Eye className="w-4 h-4" />
                Attention
              </button>
              <button
                onClick={() => setView('activations')}
                className={`px-4 py-2 flex items-center gap-2 transition-all ${
                  view === 'activations'
                    ? 'bg-white/10 text-white border border-white/20'
                    : 'bg-white/[0.03] text-white/60 border border-white/[0.08] hover:bg-white/[0.05]'
                }`}
              >
                <Sparkles className="w-4 h-4" />
                Activations
              </button>
              <button
                onClick={() => setView('journey')}
                className={`px-4 py-2 flex items-center gap-2 transition-all ${
                  view === 'journey'
                    ? 'bg-white/10 text-white border border-white/20'
                    : 'bg-white/[0.03] text-white/60 border border-white/[0.08] hover:bg-white/[0.05]'
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                Journey
              </button>
              <button
                onClick={() => setView('flow')}
                className={`px-4 py-2 flex items-center gap-2 transition-all ${
                  view === 'flow'
                    ? 'bg-white/10 text-white border border-white/20'
                    : 'bg-white/[0.03] text-white/60 border border-white/[0.08] hover:bg-white/[0.05]'
                }`}
              >
                <Layers className="w-4 h-4" />
                Flow
              </button>
              <button
                onClick={() => setView('network')}
                className={`px-4 py-2 flex items-center gap-2 transition-all ${
                  view === 'network'
                    ? 'bg-white/10 text-white border border-white/20'
                    : 'bg-white/[0.03] text-white/60 border border-white/[0.08] hover:bg-white/[0.05]'
                }`}
              >
                <Network className="w-4 h-4" />
                Network
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[1800px] mx-auto px-6 py-8">
        {/* Prompt Display */}
        <div className="mb-6 p-4 glass border border-white/[0.08]">
          <div className="text-xs text-white/50 mb-2 uppercase tracking-wider">Prompt</div>
          <div className="text-white font-mono text-sm">{activationData.prompt}</div>
        </div>

        {/* Layer/Head/Token Controls */}
        <div className="mb-6 flex flex-wrap items-center gap-4 p-4 glass border border-white/[0.08]">
          {(view === 'attention' || view === 'activations') && (
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-white/70" />
              <span className="text-sm text-white/60">Layer:</span>
              <button
                onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
                disabled={selectedLayer === 0}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-white/[0.05] font-mono text-white border border-white/[0.08]">
                {selectedLayer}
              </span>
              <button
                onClick={() => setSelectedLayer(Math.min(numLayers - 1, selectedLayer + 1))}
                disabled={selectedLayer === numLayers - 1}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {view === 'attention' && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-white/60">Head:</span>
              <button
                onClick={() => setSelectedHead(Math.max(0, selectedHead - 1))}
                disabled={selectedHead === 0}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-white/[0.05] font-mono text-white border border-white/[0.08]">
                {selectedHead}
              </span>
              <button
                onClick={() => setSelectedHead(Math.min(numHeads - 1, selectedHead + 1))}
                disabled={selectedHead === numHeads - 1}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {(view === 'journey' || view === 'flow' || view === 'network') && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-white/60">Token:</span>
              <button
                onClick={() => setSelectedTokenIndex(Math.max(0, selectedTokenIndex - 1))}
                disabled={selectedTokenIndex === 0}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-white/[0.05] font-mono text-white border border-white/[0.08]">
                {selectedTokenIndex}
              </span>
              <button
                onClick={() => setSelectedTokenIndex(Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1))}
                disabled={selectedTokenIndex === activationData.tokens.length - 1}
                className="p-1 bg-white/[0.05] hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/80 border border-white/[0.08]"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Visualization Area */}
        {view === 'attention' && activationData.activations.layers?.[selectedLayer]?.attn_weights && (
          <div className="flex justify-center">
            <AttentionHeatmap
              attentionWeights={activationData.activations.layers[selectedLayer].attn_weights!}
              layerIndex={selectedLayer}
              headIndex={selectedHead}
            />
          </div>
        )}

        {view === 'activations' && activationData.activations.layers?.[selectedLayer] && (
          <LayerActivationDisplay
            layerActivations={activationData.activations.layers[selectedLayer]}
            layerIndex={selectedLayer}
          />
        )}

        {view === 'journey' && (
          <EmbeddingJourney
            activationData={activationData}
            selectedTokenIndex={selectedTokenIndex}
          />
        )}

        {view === 'flow' && (
          <TransformationFlow
            activationData={activationData}
            selectedTokenIndex={selectedTokenIndex}
          />
        )}

        {view === 'network' && (
          <GenerationFlow
            activationData={activationData}
          />
        )}

        {/* Metadata */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetadataCard
            label="Temperature"
            value={activationData.metadata.temperature?.toFixed(2) || 'N/A'}
          />
          <MetadataCard
            label="Top-K"
            value={activationData.metadata.top_k?.toString() || 'N/A'}
          />
          <MetadataCard
            label="Prompt Tokens"
            value={activationData.metadata.prompt_tokens?.toString() || 'N/A'}
          />
          <MetadataCard
            label="Generated Tokens"
            value={activationData.metadata.generated_tokens?.toString() || 'N/A'}
          />
        </div>
      </div>
    </div>
  )
}

interface MetadataCardProps {
  label: string
  value: string
}

function MetadataCard({ label, value }: MetadataCardProps) {
  return (
    <div className="p-4 glass border border-white/[0.08]">
      <div className="text-xs text-white/50 mb-1 uppercase tracking-wider">{label}</div>
      <div className="text-lg font-mono text-white">{value}</div>
    </div>
  )
}
