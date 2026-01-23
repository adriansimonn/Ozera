/**
 * React hooks for model management.
 */

import { useState, useEffect, useCallback } from 'react'
import { apiClient, ModelInfo } from '../api/client'
import type { OpenSourceModelInfo, ModelFamily } from '../types/model'

export interface ModelsState {
  models: string[]
  modelNames: Record<string, string>  // Maps model_id to display name
  modelFamilies: Record<string, ModelFamily>  // Maps model_id to family
  openSourceModels: OpenSourceModelInfo[]
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
 * uploaded models from /training/models/uploaded, and open-source models.
 */
export function useModels() {
  const [state, setState] = useState<ModelsState>({
    models: [],
    modelNames: {},
    modelFamilies: {},
    openSourceModels: [],
    loading: true,
    error: null,
  })

  const fetchModels = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      // Fetch all model sources in parallel
      const [baseModels, customModels, uploadedModels, openSourceModels] = await Promise.all([
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
        apiClient.listOpenSourceModels().catch((err) => {
          console.error('Failed to fetch open-source models:', err)
          return [] as OpenSourceModelInfo[]
        }),
      ])

      // Build display name and family mappings
      const nameMap: Record<string, string> = {}
      const familyMap: Record<string, ModelFamily> = {}

      // Base models use their formatted names and belong to 'ozera' family
      baseModels.forEach(m => {
        nameMap[m] = m === 'nano' || m === 'mini' ? `ozera-${m}` : m
        familyMap[m] = 'ozera'
      })

      // Custom trained models use their name field and belong to 'ozera' family
      customModels.forEach(m => {
        nameMap[m.model_id] = m.name
        familyMap[m.model_id] = 'ozera'
      })

      // Uploaded models use their name field and belong to 'ozera' family
      uploadedModels.forEach(m => {
        nameMap[m.model_id] = m.name
        familyMap[m.model_id] = 'ozera'
      })

      // Open-source models use their display_name and have their own family
      openSourceModels.forEach(m => {
        nameMap[m.id] = m.display_name
        familyMap[m.id] = m.family
      })

      // Combine all model sources, avoiding duplicates
      const customModelIds = customModels.map(m => m.model_id)
      const uploadedModelIds = uploadedModels.map(m => m.model_id)
      const openSourceModelIds = openSourceModels.map(m => m.id)
      const allModels = [...new Set([
        ...baseModels,
        ...customModelIds,
        ...uploadedModelIds,
        ...openSourceModelIds
      ])]

      // If no models at all, show an error
      if (allModels.length === 0) {
        setState({
          models: [],
          modelNames: {},
          modelFamilies: {},
          openSourceModels: [],
          loading: false,
          error: 'No models available'
        })
      } else {
        setState({
          models: allModels,
          modelNames: nameMap,
          modelFamilies: familyMap,
          openSourceModels,
          loading: false,
          error: null
        })
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setState({
        models: [],
        modelNames: {},
        modelFamilies: {},
        openSourceModels: [],
        loading: false,
        error: errorMessage
      })
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
