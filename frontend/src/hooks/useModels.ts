/**
 * React hooks for model management.
 */

import { useState, useEffect, useCallback } from 'react'
import { apiClient, ModelInfo } from '../api/client'

export interface ModelsState {
  models: string[]
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
 */
export function useModels() {
  const [state, setState] = useState<ModelsState>({
    models: [],
    loading: true,
    error: null,
  })

  const fetchModels = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      const models = await apiClient.listModels()
      setState({ models, loading: false, error: null })
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
