/**
 * Lazy loading of stored activations for the visualization panel.
 * Loads a lightweight summary first, then only the data the selected visualization needs.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { apiClient } from '../api/client'
import { FLOW_NODES_PER_LAYER } from '../components/visualization/GenerationFlow'
import type {
  ActivationData,
  ActivationSummaryWithInfo,
  FlowActivations,
  LayerActivations,
  TensorData,
  TopKLogits,
} from '../types/model'

export type VisualizationType = 'network' | 'attention' | 'activations' | 'journey' | 'flow'

export function useActivationData(selectedVisualization: VisualizationType, selectedLayer: number) {
  const [activationData, setActivationData] = useState<ActivationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [layerLoading, setLayerLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Cache for lazily loaded layers and tensors
  const layerCacheRef = useRef<Map<number, LayerActivations>>(new Map())
  const tensorCacheRef = useRef<Map<string, TensorData | TopKLogits>>(new Map())
  // Leading dimensions of every layer for the network view (kept apart from the full layers)
  const flowRef = useRef<FlowActivations | null>(null)
  const currentActivationIdRef = useRef<string | null>(null)
  const summaryRef = useRef<ActivationSummaryWithInfo | null>(null)
  // Track which visualization's data has been loaded to avoid redundant fetches
  const loadedVizRef = useRef<Set<string>>(new Set())
  // Current selection, read by loads that finish after it changes
  const selectedVisualizationRef = useRef(selectedVisualization)
  const selectedLayerRef = useRef(selectedLayer)
  // Requests in flight, so overlapping loads of the same data share one fetch
  const inflightRef = useRef<Map<string, Promise<void>>>(new Map())

  useEffect(() => {
    selectedVisualizationRef.current = selectedVisualization
    selectedLayerRef.current = selectedLayer
  }, [selectedVisualization, selectedLayer])

  const dedupe = useCallback((key: string, load: () => Promise<void>): Promise<void> => {
    const pending = inflightRef.current.get(key)
    if (pending) return pending
    const promise = load().finally(() => inflightRef.current.delete(key))
    inflightRef.current.set(key, promise)
    return promise
  }, [])

  // Build ActivationData from summary + cached layers/tensors
  const buildActivationData = useCallback((summary: ActivationSummaryWithInfo, vizType: VisualizationType): ActivationData => {
    const flow = vizType === 'network' ? flowRef.current : null
    const layers: LayerActivations[] = []
    for (let i = 0; i < summary.num_layers; i++) {
      layers.push((flow ? flow.layers[i] : layerCacheRef.current.get(i)) || {})
    }

    return {
      id: summary.id,
      prompt: summary.prompt,
      model: summary.model,
      timestamp: summary.timestamp,
      tokens: summary.tokens,
      metadata: summary.metadata,
      activations: {
        token_embeddings: (flow ? flow.token_embeddings : tensorCacheRef.current.get('token_embeddings')) as TensorData | undefined,
        positional_embeddings: tensorCacheRef.current.get('positional_embeddings') as TensorData | undefined,
        combined_embeddings: (flow ? flow.combined_embeddings : tensorCacheRef.current.get('combined_embeddings')) as TensorData | undefined,
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

    return dedupe(`${id}:tensor:${tensorName}`, async () => {
      try {
        const result = await apiClient.getTensorActivation(id, tensorName)
        if (currentActivationIdRef.current !== id) return // stale
        tensorCacheRef.current.set(tensorName, result.data)
      } catch (err) {
        console.error(`Failed to load tensor ${tensorName}:`, err)
      }
    })
  }, [dedupe])

  // Load a specific layer and merge into activation data
  const loadLayer = useCallback(async (layerIdx: number): Promise<void> => {
    const id = currentActivationIdRef.current
    if (!id || layerCacheRef.current.has(layerIdx)) return

    return dedupe(`${id}:layer:${layerIdx}`, async () => {
      try {
        const result = await apiClient.getLayerActivations(id, layerIdx)
        if (currentActivationIdRef.current !== id) return // stale
        layerCacheRef.current.set(layerIdx, result.activations as LayerActivations)
      } catch (err) {
        console.error(`Failed to load layer ${layerIdx}:`, err)
      }
    })
  }, [dedupe])

  // Load the leading dimensions of every layer and the embeddings for the network view
  const loadFlow = useCallback(async (): Promise<void> => {
    const id = currentActivationIdRef.current
    if (!id || flowRef.current) return

    return dedupe(`${id}:flow`, async () => {
      try {
        const result = await apiClient.getFlowActivations(id, FLOW_NODES_PER_LAYER)
        if (currentActivationIdRef.current !== id) return // stale
        flowRef.current = result
      } catch (err) {
        console.error('Failed to load flow activations:', err)
      }
    })
  }, [dedupe])

  // Load data needed for a specific visualization type
  const loadDataForVisualization = useCallback(async (vizType: VisualizationType, layer?: number) => {
    const summary = summaryRef.current
    if (!summary) return

    const id = currentActivationIdRef.current

    try {
      setLayerLoading(true)

      if (vizType === 'network') {
        if (!loadedVizRef.current.has('network')) {
          // GenerationFlow needs: top_k_logits, plus a few dimensions of the embeddings and every layer
          const promises: Promise<void>[] = [loadFlow()]
          if (summary.tensor_info.top_k_logits) promises.push(loadTensor('top_k_logits'))
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
        setActivationData(buildActivationData(summary, selectedVisualizationRef.current))
      }
    } finally {
      setLayerLoading(false)
    }
  }, [buildActivationData, loadTensor, loadLayer, loadFlow])

  // Load summary first, then load data for current visualization
  const loadActivation = useCallback(async (id: string) => {
    try {
      setLoading(true)
      setError(null)

      // Clear caches if loading a new activation
      if (currentActivationIdRef.current !== id) {
        layerCacheRef.current.clear()
        tensorCacheRef.current.clear()
        flowRef.current = null
        loadedVizRef.current.clear()
        currentActivationIdRef.current = id
      }

      // Get lightweight summary first (metadata + shapes, no tensor values)
      const summary = await apiClient.getActivationSummary(id)
      summaryRef.current = summary

      // Build skeleton ActivationData immediately so UI can render
      setActivationData(buildActivationData(summary, selectedVisualizationRef.current))
      setLoading(false)

      // Then load data needed for the current visualization
      await loadDataForVisualization(selectedVisualizationRef.current, selectedLayerRef.current)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activations')
      setLoading(false)
    }
  }, [buildActivationData, loadDataForVisualization])

  // Load data when the visualization type changes, or the layer changes
  // (layer controls only exist in the per-layer views)
  useEffect(() => {
    if (summaryRef.current && currentActivationIdRef.current) {
      loadDataForVisualization(selectedVisualization, selectedLayer)
    }
  }, [selectedVisualization, selectedLayer, loadDataForVisualization])

  return {
    activationData,
    loading,
    layerLoading,
    error,
    summary: summaryRef.current,
    loadActivation,
  }
}
