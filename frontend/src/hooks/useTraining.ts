/**
 * Hooks for training functionality.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import {
  apiClient,
  TrainingEstimate,
  TrainingJobResponse,
  TrainingProgress,
  TrainingJobListItem,
  TrainingStreamEvent,
  CustomModelInfo,
  CustomModelCount,
  GpuPricingInfo,
  GpuType,
  UploadedModelInfo,
  ModelUploadResponse,
} from '../api/client'

// useDatasets hook removed - datasets are now session-only and uploaded directly via apiClient

/**
 * Hook for GPU pricing.
 */
export function useGpuPricing() {
  const [pricing, setPricing] = useState<GpuPricingInfo[]>([])
  const [defaultGpu, setDefaultGpu] = useState<GpuType>('a10g')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPricing = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.getGpuPricing()
      setPricing(data.pricing)
      setDefaultGpu(data.default_gpu)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch GPU pricing')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPricing()
  }, [fetchPricing])

  return {
    pricing,
    defaultGpu,
    loading,
    error,
    fetchPricing,
  }
}

/**
 * Hook for training cost estimation.
 */
export function useTrainingEstimate() {
  const [estimate, setEstimate] = useState<TrainingEstimate | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getEstimate = useCallback(async (
    datasetId: string,
    modelConfig: 'nano' | 'mini',
    epochs: number,
    batchSize: number,
    seqLen: number,
    gpuType: GpuType = 'a10g'
  ) => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.getTrainingEstimate(
        datasetId,
        modelConfig,
        epochs,
        batchSize,
        seqLen,
        gpuType
      )
      setEstimate(data)
      return data
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to get estimate'
      setError(message)
      setEstimate(null)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const clearEstimate = useCallback(() => {
    setEstimate(null)
    setError(null)
  }, [])

  return {
    estimate,
    loading,
    error,
    getEstimate,
    clearEstimate,
  }
}

/**
 * Hook for managing training jobs.
 */
export function useTrainingJobs() {
  const [jobs, setJobs] = useState<TrainingJobListItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchJobs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.listTrainingJobs()
      setJobs(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs')
    } finally {
      setLoading(false)
    }
  }, [])

  const startJob = useCallback(async (
    datasetId: string,
    baseModel: 'nano' | 'mini',
    modelName: string,
    epochs: number,
    batchSize: number,
    learningRate: number,
    seqLen: number,
    gpuType: GpuType = 'a10g',
    overwriteExisting: boolean = false
  ): Promise<TrainingJobResponse> => {
    setError(null)
    try {
      const response = await apiClient.startTrainingJob({
        dataset_id: datasetId,
        base_model: baseModel,
        model_name: modelName,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        seq_len: seqLen,
        gpu_type: gpuType,
        overwrite_existing: overwriteExisting,
      })
      // Refresh job list
      await fetchJobs()
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start training'
      setError(message)
      throw err
    }
  }, [fetchJobs])

  const cancelJob = useCallback(async (jobId: string) => {
    setError(null)
    try {
      await apiClient.cancelTrainingJob(jobId)
      await fetchJobs()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel job'
      setError(message)
      throw err
    }
  }, [fetchJobs])

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  return {
    jobs,
    loading,
    error,
    fetchJobs,
    startJob,
    cancelJob,
  }
}

/**
 * Hook for streaming training progress.
 */
export function useTrainingProgress(jobId: string | null) {
  const [progress, setProgress] = useState<TrainingProgress | null>(null)
  const [completed, setCompleted] = useState(false)
  const [completedModelName, setCompletedModelName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef(false)

  useEffect(() => {
    if (!jobId) {
      setProgress(null)
      setCompleted(false)
      setCompletedModelName(null)
      setError(null)
      return
    }

    abortRef.current = false

    const streamProgress = async () => {
      try {
        await apiClient.streamTrainingProgress(
          jobId,
          (event: TrainingStreamEvent) => {
            if (abortRef.current) return
            setProgress({
              job_id: event.job_id || jobId,
              status: event.status || 'running',
              current_epoch: event.current_epoch || 0,
              total_epochs: event.total_epochs || 0,
              current_step: 0,
              total_steps: 0,
              train_loss: event.train_loss ?? null,
              val_loss: event.val_loss ?? null,
              train_ppl: event.train_ppl ?? null,
              val_ppl: event.val_ppl ?? null,
              elapsed_seconds: event.elapsed_seconds || 0,
              estimated_remaining_seconds: event.estimated_remaining_seconds || 0,
              last_update: null,
              error_message: null,
            })
          },
          (modelName: string) => {
            if (abortRef.current) return
            setCompleted(true)
            setCompletedModelName(modelName)
          },
          (message: string) => {
            if (abortRef.current) return
            setError(message)
          }
        )
      } catch (err) {
        if (abortRef.current) return
        setError(err instanceof Error ? err.message : 'Failed to stream progress')
      }
    }

    streamProgress()

    return () => {
      abortRef.current = true
    }
  }, [jobId])

  const reset = useCallback(() => {
    setProgress(null)
    setCompleted(false)
    setCompletedModelName(null)
    setError(null)
  }, [])

  return {
    progress,
    completed,
    completedModelName,
    error,
    reset,
  }
}

/**
 * Hook for managing custom models.
 */
export function useCustomModels() {
  const [models, setModels] = useState<CustomModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.listCustomModels()
      setModels(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models')
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteModel = useCallback(async (modelId: string) => {
    setError(null)
    try {
      await apiClient.deleteCustomModel(modelId)
      setModels(prev => prev.filter(m => m.model_id !== modelId))
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete model'
      setError(message)
      throw err
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return {
    models,
    loading,
    error,
    fetchModels,
    deleteModel,
  }
}

/**
 * Hook for managing uploaded models.
 */
export function useUploadedModels() {
  const [models, setModels] = useState<UploadedModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.listUploadedModels()
      setModels(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch uploaded models')
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteModel = useCallback(async (modelId: string) => {
    setError(null)
    try {
      await apiClient.deleteUploadedModel(modelId)
      setModels(prev => prev.filter(m => m.model_id !== modelId))
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete model'
      setError(message)
      throw err
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return {
    models,
    loading,
    error,
    fetchModels,
    deleteModel,
  }
}

/**
 * Hook for model upload.
 */
export function useModelUpload() {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [uploadedModel, setUploadedModel] = useState<ModelUploadResponse | null>(null)

  const uploadModel = useCallback(async (
    file: File,
    modelName: string,
    overwriteExisting: boolean = false
  ): Promise<ModelUploadResponse> => {
    setUploading(true)
    setProgress(0)
    setError(null)
    setUploadedModel(null)

    try {
      // Simulate progress (actual upload doesn't give progress updates)
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const response = await apiClient.uploadModel(file, modelName, overwriteExisting)

      clearInterval(progressInterval)
      setProgress(100)
      setUploadedModel(response)

      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload model'
      setError(message)
      throw err
    } finally {
      setUploading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setUploading(false)
    setProgress(0)
    setError(null)
    setUploadedModel(null)
  }, [])

  return {
    uploading,
    progress,
    error,
    uploadedModel,
    uploadModel,
    reset,
  }
}

/**
 * Hook for getting custom model count (trained + uploaded).
 */
export function useCustomModelCount() {
  const [count, setCount] = useState<CustomModelCount | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchCount = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.getCustomModelCount()
      setCount(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model count')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCount()
  }, [fetchCount])

  return {
    count,
    loading,
    error,
    fetchCount,
  }
}
