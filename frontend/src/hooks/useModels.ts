/**
 * React hooks for model management.
 */

import { useState, useEffect, useCallback } from 'react'
import { apiClient, ModelInfo } from '../api/client'
import type { OpenSourceModelInfo, ModelFamily } from '../types/model'
import {
  STATIC_BASE_MODELS,
  STATIC_BASE_MODEL_INFO,
  STATIC_OPEN_SOURCE_MODELS,
  STATIC_OPEN_SOURCE_MODEL_INFO,
} from '../data/defaultModels'

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

function buildStaticState(): ModelsState {
  const nameMap: Record<string, string> = {}
  const familyMap: Record<string, ModelFamily> = {}

  STATIC_BASE_MODELS.forEach(m => {
    nameMap[m] = `ozera-${m}`
    familyMap[m] = 'ozera'
  })

  STATIC_OPEN_SOURCE_MODELS.forEach(m => {
    nameMap[m.id] = m.display_name
    familyMap[m.id] = m.family
  })

  return {
    models: [...STATIC_BASE_MODELS, ...STATIC_OPEN_SOURCE_MODELS.map(m => m.id)],
    modelNames: nameMap,
    modelFamilies: familyMap,
    openSourceModels: STATIC_OPEN_SOURCE_MODELS,
    loading: false,
    error: null,
  }
}

/**
 * Hook to fetch and manage available models.
 * Base and open-source models are loaded from static data instantly.
 * Custom trained and uploaded models are fetched asynchronously.
 */
export function useModels() {
  const [state, setState] = useState<ModelsState>(buildStaticState)

  const fetchCustomModels = useCallback(async () => {
    try {
      const [customModels, uploadedModels] = await Promise.all([
        apiClient.listCustomModels().catch(() => []),
        apiClient.listUploadedModels().catch(() => []),
      ])

      if (customModels.length === 0 && uploadedModels.length === 0) return

      setState(prev => {
        const nameMap = { ...prev.modelNames }
        const familyMap = { ...prev.modelFamilies }

        customModels.forEach(m => {
          nameMap[m.model_id] = m.name
          familyMap[m.model_id] = 'ozera'
        })

        uploadedModels.forEach(m => {
          nameMap[m.model_id] = m.name
          familyMap[m.model_id] = 'ozera'
        })

        const customIds = customModels.map(m => m.model_id)
        const uploadedIds = uploadedModels.map(m => m.model_id)
        const allModels = [...new Set([...prev.models, ...customIds, ...uploadedIds])]

        return { ...prev, models: allModels, modelNames: nameMap, modelFamilies: familyMap }
      })
    } catch {
      // Custom models are optional — silently ignore errors
    }
  }, [])

  useEffect(() => {
    fetchCustomModels()
  }, [fetchCustomModels])

  return {
    ...state,
    refresh: fetchCustomModels,
  }
}

/**
 * Hook to fetch model information.
 * Returns static data instantly for base and open-source models.
 * Only calls the API for custom/uploaded models.
 */
export function useModelInfo(modelName: string | null) {
  const [state, setState] = useState<ModelInfoState>(() => {
    if (!modelName) return { info: null, loading: false, error: null }
    const staticInfo = STATIC_BASE_MODEL_INFO[modelName] ?? STATIC_OPEN_SOURCE_MODEL_INFO[modelName]
    if (staticInfo) return { info: staticInfo, loading: false, error: null }
    return { info: null, loading: false, error: null }
  })

  const fetchModelInfo = useCallback(async (name: string) => {
    // Use static data if available
    const staticInfo = STATIC_BASE_MODEL_INFO[name] ?? STATIC_OPEN_SOURCE_MODEL_INFO[name]
    if (staticInfo) {
      setState({ info: staticInfo, loading: false, error: null })
      return
    }

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
