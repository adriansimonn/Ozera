/**
 * React hooks for model management.
 */

import { useState, useEffect, useCallback } from 'react'
import { apiClient, ModelInfo } from '../api/client'

export interface ModelsState {
  models: string[]
  modelNames: Record<string, string>  // Maps model_id to display name
  loading: boolean
  error: string | null
}

export interface ModelInfoState {
  info: ModelInfo | null
  loading: boolean
  error: string | null
}

/**
 * Hook to fetch and manage available models.
 * Combines base models from /models, custom trained models from /training/models,
 * and uploaded models from /training/models/uploaded.
 */
export function useModels() {
  const [state, setState] = useState<ModelsState>({
    models: [],
    modelNames: {},
    loading: true,
    error: null,
  })

  const fetchModels = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      // Fetch base models, custom trained models, and uploaded models in parallel
      // All have catch handlers so we always get arrays
      const [baseModels, customModels, uploadedModels] = await Promise.all([
        apiClient.listModels().catch((err) => {
          console.error('Failed to fetch base models:', err)
          return [] as string[]
        }),
        apiClient.listCustomModels().catch((err) => {
          // Don't log error for custom models - user might not be logged in
          return []
        }),
        apiClient.listUploadedModels().catch((err) => {
          // Don't log error for uploaded models - user might not be logged in
          return []
        }),
      ])

      // Build display name mapping
      const nameMap: Record<string, string> = {}

      // Base models use their formatted names
      baseModels.forEach(m => {
        nameMap[m] = m === 'nano' || m === 'mini' ? `ozera-${m}` : m
      })

      // Custom trained models use their name field
      customModels.forEach(m => {
        nameMap[m.model_id] = m.name
      })

      // Uploaded models use their name field
      uploadedModels.forEach(m => {
        nameMap[m.model_id] = m.name
      })

      // Combine all model sources, avoiding duplicates
      const customModelIds = customModels.map(m => m.model_id)
      const uploadedModelIds = uploadedModels.map(m => m.model_id)
      const allModels = [...new Set([...baseModels, ...customModelIds, ...uploadedModelIds])]

      // If no models at all, show an error
      if (allModels.length === 0) {
        setState({ models: [], modelNames: {}, loading: false, error: 'No models available' })
      } else {
        setState({ models: allModels, modelNames: nameMap, loading: false, error: null })
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setState({ models: [], loading: false, error: errorMessage })
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return {
    ...state,
    refresh: fetchModels,
  }
}

/**
 * Hook to fetch model information.
 */
export function useModelInfo(modelName: string | null) {
  const [state, setState] = useState<ModelInfoState>({
    info: null,
    loading: false,
    error: null,
  })

  const fetchModelInfo = useCallback(async (name: string) => {
    setState({ info: null, loading: true, error: null })

    try {
      const info = await apiClient.getModelInfo(name)
      setState({ info, loading: false, error: null })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setState({ info: null, loading: false, error: errorMessage })
    }
  }, [])

  useEffect(() => {
    if (modelName) {
      fetchModelInfo(modelName)
    } else {
      setState({ info: null, loading: false, error: null })
    }
  }, [modelName, fetchModelInfo])

  return {
    ...state,
    refresh: () => modelName && fetchModelInfo(modelName),
  }
}
