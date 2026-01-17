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
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-cyan-500 border-t-transparent mb-4"></div>
          <div className="text-slate-400">Loading activations...</div>
        </div>
      </div>
    )
  }

  if (error || !activationData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 mb-2">Error</div>
          <div className="text-slate-400">{error || 'No activation data available'}</div>
        </div>
      </div>
    )
  }

  const numLayers = activationData.activations.layers?.length || 0
  const numHeads = activationData.activations.layers?.[0]?.attn_weights?.shape[1] || 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800/50 bg-slate-900/30 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                Activation Visualization
              </h1>
              <div className="text-sm text-slate-400 mt-1">
                Model: <span className="text-cyan-400 font-mono">{activationData.model}</span>
                {' • '}
                Tokens: <span className="text-cyan-400 font-mono">{activationData.tokens.length}</span>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setView('attention')}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                  view === 'attention'
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:bg-slate-800'
                }`}
              >
                <Eye className="w-4 h-4" />
                Attention
              </button>
              <button
                onClick={() => setView('activations')}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                  view === 'activations'
                    ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:bg-slate-800'
                }`}
              >
                <Sparkles className="w-4 h-4" />
                Activations
              </button>
              <button
                onClick={() => setView('journey')}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                  view === 'journey'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:bg-slate-800'
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                Journey
              </button>
              <button
                onClick={() => setView('flow')}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                  view === 'flow'
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:bg-slate-800'
                }`}
              >
                <Layers className="w-4 h-4" />
                Flow
              </button>
              <button
                onClick={() => setView('network')}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                  view === 'network'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:bg-slate-800'
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
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Prompt Display */}
        <div className="mb-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-2">Prompt</div>
          <div className="text-slate-200 font-mono text-sm">{activationData.prompt}</div>
        </div>

        {/* Layer/Head/Token Controls */}
        <div className="mb-6 flex flex-wrap items-center gap-4 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
          {(view === 'attention' || view === 'activations') && (
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              <span className="text-sm text-slate-400">Layer:</span>
              <button
                onClick={() => setSelectedLayer(Math.max(0, selectedLayer - 1))}
                disabled={selectedLayer === 0}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-slate-800 rounded font-mono text-cyan-400">
                {selectedLayer}
              </span>
              <button
                onClick={() => setSelectedLayer(Math.min(numLayers - 1, selectedLayer + 1))}
                disabled={selectedLayer === numLayers - 1}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {view === 'attention' && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Head:</span>
              <button
                onClick={() => setSelectedHead(Math.max(0, selectedHead - 1))}
                disabled={selectedHead === 0}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-slate-800 rounded font-mono text-purple-400">
                {selectedHead}
              </span>
              <button
                onClick={() => setSelectedHead(Math.min(numHeads - 1, selectedHead + 1))}
                disabled={selectedHead === numHeads - 1}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {(view === 'journey' || view === 'flow' || view === 'network') && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Token:</span>
              <button
                onClick={() => setSelectedTokenIndex(Math.max(0, selectedTokenIndex - 1))}
                disabled={selectedTokenIndex === 0}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-4 py-1 bg-slate-800 rounded font-mono text-green-400">
                {selectedTokenIndex}
              </span>
              <button
                onClick={() => setSelectedTokenIndex(Math.min(activationData.tokens.length - 1, selectedTokenIndex + 1))}
                disabled={selectedTokenIndex === activationData.tokens.length - 1}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
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
    <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="text-lg font-mono text-slate-200">{value}</div>
    </div>
  )
}
