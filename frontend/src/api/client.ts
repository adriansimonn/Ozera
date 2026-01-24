/**
 * API client for Ozera inference server.
 */

import type {
  ActivationData,
  ActivationSummary,
  GenerateWithActivationsResponse,
  OpenSourceModelInfo,
  ModelCacheStatus,
  ModelDownloadResponse,
  ModelFamilyInfo,
} from '../types/model'
import type {
  CaptureActivationsRequest,
  CaptureActivationsResponse,
  CapturedActivationSummary,
  CapturedActivationDetail,
  RunPatchingRequest,
  RunPatchingWithCapturedRequest,
  PatchingResult,
  PatchingModelInfo,
  ModelLayerInfo,
} from '../types/patching'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Get auth token from localStorage (used by Zustand persist).
 */
function getAuthToken(): string | null {
  const authStorage = localStorage.getItem('auth-storage')
  if (authStorage) {
    try {
      const { state } = JSON.parse(authStorage)
      return state?.token || null
    } catch {
      return null
    }
  }
  return null
}

/**
 * Get auth headers if token is available.
 */
function getAuthHeaders(): Record<string, string> {
  const token = getAuthToken()
  if (token) {
    return { Authorization: `Bearer ${token}` }
  }
  return {}
}

// Dataset types
export interface DatasetMetadata {
  dataset_id: string
  name: string
  size_bytes: number
  num_tokens: number
  created_at: string
}

export interface DatasetDetail extends DatasetMetadata {
  preview: string
}

export interface GenericDatasetInfo {
  id: string  // e.g., "generic:tinystories"
  name: string  // e.g., "Tinystories"
  description?: string
}

// Training types
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type GpuType = 't4' | 'a10g' | 'a100'

export interface GpuPricingInfo {
  gpu_type: GpuType
  display_name: string
  rate_per_hour: number
  description: string
  nano_tokens_per_sec: number
  mini_tokens_per_sec: number
}

export interface TrainingJobRequest {
  dataset_id: string
  base_model: 'nano' | 'mini'
  model_name: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  seq_len?: number
  gpu_type?: GpuType
  overwrite_existing?: boolean
}

export interface CustomModelCount {
  trained_count: number
  uploaded_count: number
  total_count: number
  max_allowed: number
}

export interface UploadedModelInfo {
  model_id: string
  name: string
  file_size_bytes: number
  num_parameters: number | null
  num_layers: number | null
  num_heads: number | null
  hidden_dim: number | null
  vocab_size: number | null
  max_seq_len: number | null
  uploaded_at: string
  model_type: 'uploaded'
}

export interface ModelUploadResponse {
  model_id: string
  name: string
  file_size_bytes: number
  num_parameters: number | null
  num_layers: number | null
  status: string
}

export interface TrainingJobResponse {
  job_id: string
  status: JobStatus
  model_name: string
  dataset_id: string
  config: {
    epochs: number
    batch_size: number
    learning_rate: number
    seq_len: number
    model_config: string
  }
  created_at: string
  estimated_minutes: number
  estimated_cost_usd: number
}

export interface TrainingProgress {
  job_id: string
  status: JobStatus
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  train_loss: number | null
  val_loss: number | null
  train_ppl: number | null
  val_ppl: number | null
  elapsed_seconds: number
  estimated_remaining_seconds: number
  last_update: string | null
  error_message: string | null
}

export interface TrainingEstimate {
  estimated_minutes: number
  estimated_cost_usd: number
  total_tokens: number
  tokens_per_epoch: number
  warning: string | null
}

export interface TrainingJobListItem {
  job_id: string
  status: JobStatus
  model_name: string
  dataset_name: string
  created_at: string
  current_epoch: number
  total_epochs: number
}

export interface CustomModelInfo {
  model_id: string
  name: string
  base_config: string
  dataset_id: string
  dataset_name: string
  trained_at: string
  val_loss: number
  parameters: number
}

export interface TrainingStreamEvent {
  type: 'progress' | 'completed' | 'error' | 'cancelled'
  job_id?: string
  status?: JobStatus
  current_epoch?: number
  total_epochs?: number
  train_loss?: number | null
  val_loss?: number | null
  train_ppl?: number | null
  val_ppl?: number | null
  elapsed_seconds?: number
  estimated_remaining_seconds?: number
  model_name?: string
  message?: string
}

export interface GenerateRequest {
  prompt: string
  model: string
  max_tokens?: number
  temperature?: number
  top_k?: number
  top_p?: number
}

export interface GenerateResponse {
  text: string
  prompt: string
  model: string
  prompt_tokens: number
  generated_tokens: number
  total_tokens: number
  temperature: number
  top_k: number | null
  top_p: number | null
}

export interface ModelInfo {
  name: string
  parameters: number
  layers: number
  heads: number
  hidden_dim: number
  vocab_size: number
}

export interface HealthResponse {
  status: string
  available_models: string[]
}

export interface StreamToken {
  type: 'start' | 'token' | 'done' | 'error'
  text?: string
  prompt?: string
  message?: string
}

class OzeraAPIClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Check API health and get available models.
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`)

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List available models.
   */
  async listModels(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/models`)

    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get information about a specific model.
   */
  async getModelInfo(modelName: string): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/models/${modelName}`)

    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Prepare a model for inference (pre-loads and caches it).
   * For custom models, this triggers download from Modal volume if needed.
   */
  async prepareModel(modelName: string): Promise<{ status: string; model: string; parameters: number; layers: number }> {
    const response = await fetch(`${this.baseUrl}/models/${modelName}/prepare`, {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to prepare model: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Generate text from a prompt.
   */
  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Generation failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Generate text with streaming (Server-Sent Events).
   *
   * @param request Generation request
   * @param onToken Callback for each token
   * @param onStart Callback when generation starts
   * @param onDone Callback when generation completes
   * @param onError Callback on error
   */
  async generateStream(
    request: GenerateRequest,
    onToken: (token: string) => void,
    onStart?: (prompt: string) => void,
    onDone?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/generate/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      // Check for insufficient credits (402 Payment Required)
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Streaming failed: ${response.statusText}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')

        // Keep the last incomplete line in buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)

            try {
              const event: StreamToken = JSON.parse(data)

              switch (event.type) {
                case 'start':
                  if (onStart && event.prompt) {
                    onStart(event.prompt)
                  }
                  break

                case 'token':
                  if (event.text) {
                    onToken(event.text)
                  }
                  break

                case 'done':
                  if (onDone) {
                    onDone()
                  }
                  break

                case 'error':
                  if (onError && event.message) {
                    onError(event.message)
                  }
                  break
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  /**
   * Generate text with activation capture for visualization.
   */
  async generateWithActivations(request: GenerateRequest): Promise<GenerateWithActivationsResponse> {
    const response = await fetch(`${this.baseUrl}/generate/with-activations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      // Check for insufficient credits (402 Payment Required)
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Generation with activations failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get activation data by ID.
   */
  async getActivations(activationId: string): Promise<ActivationData> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}`)

    if (!response.ok) {
      throw new Error(`Failed to get activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get activation summary (metadata only, no tensors).
   * Enhanced with layer info and tensor info for lazy loading.
   */
  async getActivationSummary(activationId: string): Promise<ActivationSummary> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}/summary`)

    if (!response.ok) {
      throw new Error(`Failed to get activation summary: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get activations for a specific layer (lazy loading).
   * Use this to load individual layers on-demand instead of loading all at once.
   */
  async getLayerActivations(activationId: string, layerIdx: number): Promise<{ layer_idx: number; activations: Record<string, any> }> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}/layer/${layerIdx}`)

    if (!response.ok) {
      throw new Error(`Failed to get layer activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get a specific top-level tensor (lazy loading).
   * Valid tensor names: token_embeddings, positional_embeddings, combined_embeddings, final_layer_norm, logits
   */
  async getTensorActivation(activationId: string, tensorName: string): Promise<{ tensor_name: string; data: Record<string, any> }> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}/tensor/${tensorName}`)

    if (!response.ok) {
      throw new Error(`Failed to get tensor activation: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List all stored activations.
   */
  async listActivations(): Promise<ActivationSummary[]> {
    const response = await fetch(`${this.baseUrl}/activations`)

    if (!response.ok) {
      throw new Error(`Failed to list activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete activation data.
   */
  async deleteActivations(activationId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete activations: ${response.statusText}`)
    }
  }

  // ============= Dataset Management =============

  /**
   * Upload a dataset file.
   */
  async uploadDataset(file: File): Promise<DatasetMetadata> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/datasets/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Upload failed: ${response.statusText}`)
    }

    return response.json()
  }

  // listDatasets, getDataset, and deleteDataset removed - datasets are now session-only

  /**
   * List generic datasets available for training.
   * These are pre-uploaded datasets available to all users.
   */
  async listGenericDatasets(): Promise<GenericDatasetInfo[]> {
    const response = await fetch(`${this.baseUrl}/datasets/generic/list`)

    if (!response.ok) {
      throw new Error(`Failed to list generic datasets: ${response.statusText}`)
    }

    const data = await response.json()
    return data.datasets
  }

  // ============= Training Jobs =============

  /**
   * Get GPU pricing information.
   */
  async getGpuPricing(): Promise<{ pricing: GpuPricingInfo[]; default_gpu: GpuType }> {
    const response = await fetch(`${this.baseUrl}/training/gpu-pricing`)

    if (!response.ok) {
      throw new Error(`Failed to get GPU pricing: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get training cost estimate.
   */
  async getTrainingEstimate(
    datasetId: string,
    baseModel: 'nano' | 'mini',
    epochs: number,
    batchSize: number,
    seqLen: number,
    gpuType: GpuType = 'a10g'
  ): Promise<TrainingEstimate> {
    const response = await fetch(`${this.baseUrl}/training/estimate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        base_model: baseModel,
        epochs,
        batch_size: batchSize,
        seq_len: seqLen,
        gpu_type: gpuType,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to get estimate: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Start a training job.
   */
  async startTrainingJob(request: TrainingJobRequest): Promise<TrainingJobResponse> {
    const response = await fetch(`${this.baseUrl}/training/jobs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to start training: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List training jobs for the current user.
   */
  async listTrainingJobs(): Promise<TrainingJobListItem[]> {
    const response = await fetch(`${this.baseUrl}/training/jobs`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to list jobs: ${response.statusText}`)
    }

    const data = await response.json()
    return data.jobs
  }

  /**
   * Get training job status.
   */
  async getTrainingJobStatus(jobId: string): Promise<TrainingProgress> {
    const response = await fetch(`${this.baseUrl}/training/jobs/${jobId}`)

    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Stream training job progress via SSE.
   */
  async streamTrainingProgress(
    jobId: string,
    onProgress: (event: TrainingStreamEvent) => void,
    onComplete?: (modelName: string) => void,
    onError?: (message: string) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/training/jobs/${jobId}/stream`)

    if (!response.ok) {
      throw new Error(`Failed to stream progress: ${response.statusText}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const event: TrainingStreamEvent = JSON.parse(data)

              if (event.type === 'progress') {
                onProgress(event)
              } else if (event.type === 'completed') {
                if (onComplete && event.model_name) {
                  onComplete(event.model_name)
                }
                return
              } else if (event.type === 'error') {
                if (onError && event.message) {
                  onError(event.message)
                }
                return
              } else if (event.type === 'cancelled') {
                if (onError) {
                  onError('Training was cancelled')
                }
                return
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  /**
   * Cancel a training job.
   */
  async cancelTrainingJob(jobId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/training/jobs/${jobId}/cancel`, {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to cancel job: ${response.statusText}`)
    }
  }

  // ============= Custom Models =============

  /**
   * Get count of custom models for current user.
   */
  async getCustomModelCount(): Promise<CustomModelCount> {
    const response = await fetch(`${this.baseUrl}/training/models/count`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to get model count: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List custom trained models.
   */
  async listCustomModels(): Promise<CustomModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/training/models`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to list custom models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete a custom model.
   */
  async deleteCustomModel(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/training/models/${modelId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete model: ${response.statusText}`)
    }
  }

  /**
   * Download a custom model as a zip file.
   */
  async downloadCustomModel(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/training/models/${modelId}/download`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to download model: ${response.statusText}`)
    }

    // Get the blob and trigger download
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${modelId}.zip`
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  // ============= Model Upload =============

  /**
   * Upload a .safetensors model file.
   */
  async uploadModel(
    file: File,
    modelName: string,
    overwriteExisting: boolean = false
  ): Promise<ModelUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_name', modelName)
    formData.append('overwrite_existing', String(overwriteExisting))

    const response = await fetch(`${this.baseUrl}/training/models/upload`, {
      method: 'POST',
      headers: {
        ...getAuthHeaders(),
      },
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to upload model: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List uploaded models for the current user.
   */
  async listUploadedModels(): Promise<UploadedModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/training/models/uploaded`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to list uploaded models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete an uploaded model.
   */
  async deleteUploadedModel(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/training/models/uploaded/${modelId}`, {
      method: 'DELETE',
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to delete uploaded model: ${response.statusText}`)
    }
  }

  // ============= Open-Source Models =============

  /**
   * List all available open-source models.
   */
  async listOpenSourceModels(): Promise<OpenSourceModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/open-source/models`)

    if (!response.ok) {
      throw new Error(`Failed to list open-source models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get info for a specific open-source model.
   */
  async getOpenSourceModelInfo(modelId: string): Promise<OpenSourceModelInfo> {
    const response = await fetch(`${this.baseUrl}/open-source/models/${modelId}`)

    if (!response.ok) {
      throw new Error(`Failed to get open-source model info: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List open-source models grouped by family.
   */
  async listOpenSourceFamilies(): Promise<ModelFamilyInfo[]> {
    const response = await fetch(`${this.baseUrl}/open-source/families`)

    if (!response.ok) {
      throw new Error(`Failed to list model families: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Check cache status for an open-source model.
   */
  async getOpenSourceCacheStatus(modelId: string): Promise<ModelCacheStatus> {
    const response = await fetch(`${this.baseUrl}/open-source/cache/${modelId}/status`)

    if (!response.ok) {
      throw new Error(`Failed to get cache status: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Trigger download of an open-source model to Modal cache.
   */
  async downloadOpenSourceModel(modelId: string, force: boolean = false): Promise<ModelDownloadResponse> {
    const response = await fetch(`${this.baseUrl}/open-source/cache/${modelId}/download?force=${force}`, {
      method: 'POST',
    })

    if (!response.ok) {
      throw new Error(`Failed to download model: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete cached open-source model.
   */
  async deleteOpenSourceCache(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/open-source/cache/${modelId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete cached model: ${response.statusText}`)
    }
  }

  /**
   * Warmup an open-source model (pre-load into GPU memory).
   */
  async warmupOpenSourceModel(modelId: string): Promise<{ status: string; model: string; display_name: string; parameters: number; gpu_tier: string }> {
    const response = await fetch(`${this.baseUrl}/open-source/models/${modelId}/warmup`, {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to warmup model: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Generate text using an open-source model.
   */
  async generateOpenSource(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${this.baseUrl}/open-source/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Generation failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Stream text generation using an open-source model.
   */
  async generateOpenSourceStream(
    request: GenerateRequest,
    onToken: (token: string) => void,
    onStart?: (prompt: string) => void,
    onDone?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/open-source/generate/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Streaming failed: ${response.statusText}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)

            try {
              const event: StreamToken = JSON.parse(data)

              switch (event.type) {
                case 'start':
                  if (onStart && event.prompt) {
                    onStart(event.prompt)
                  }
                  break

                case 'token':
                  if (event.text) {
                    onToken(event.text)
                  }
                  break

                case 'done':
                  if (onDone) {
                    onDone()
                  }
                  break

                case 'error':
                  if (onError && event.message) {
                    onError(event.message)
                  }
                  break
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  /**
   * Generate text with activations using an open-source model.
   */
  async generateOpenSourceWithActivations(request: GenerateRequest): Promise<GenerateWithActivationsResponse> {
    const response = await fetch(`${this.baseUrl}/open-source/generate/with-activations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Generation with activations failed: ${response.statusText}`)
    }

    return response.json()
  }

  // ============= Activation Patching =============

  /**
   * Capture source activations for patching.
   */
  async captureActivations(request: CaptureActivationsRequest): Promise<CaptureActivationsResponse> {
    const response = await fetch(`${this.baseUrl}/patching/capture`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to capture activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List all captured activations.
   */
  async listCapturedActivations(): Promise<CapturedActivationSummary[]> {
    const response = await fetch(`${this.baseUrl}/patching/activations`)

    if (!response.ok) {
      throw new Error(`Failed to list captured activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get detailed info for a captured activation.
   */
  async getCapturedActivation(activationId: string): Promise<CapturedActivationDetail> {
    const response = await fetch(`${this.baseUrl}/patching/activations/${activationId}`)

    if (!response.ok) {
      throw new Error(`Failed to get captured activation: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete a specific captured activation.
   */
  async deleteCapturedActivation(activationId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/patching/activations/${activationId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete captured activation: ${response.statusText}`)
    }
  }

  /**
   * Clear all captured activations.
   */
  async clearCapturedActivations(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/patching/activations`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to clear captured activations: ${response.statusText}`)
    }
  }

  /**
   * Run a full patching experiment (captures source, then runs patching).
   */
  async runPatchingExperiment(request: RunPatchingRequest): Promise<PatchingResult> {
    const response = await fetch(`${this.baseUrl}/patching/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to run patching experiment: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Run patching with pre-captured activations.
   */
  async runPatchingWithCaptured(request: RunPatchingWithCapturedRequest): Promise<PatchingResult> {
    const response = await fetch(`${this.baseUrl}/patching/run-with-captured`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to run patching with captured: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get list of available models for patching.
   */
  async getPatchingModels(): Promise<PatchingModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/patching/models`)

    if (!response.ok) {
      throw new Error(`Failed to get patching models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get layer info for a specific model (for patch configuration).
   */
  async getModelLayerInfo(modelId: string): Promise<ModelLayerInfo> {
    const response = await fetch(`${this.baseUrl}/patching/models/${modelId}/layers`)

    if (!response.ok) {
      throw new Error(`Failed to get model layer info: ${response.statusText}`)
    }

    return response.json()
  }
}

// Export singleton instance
export const apiClient = new OzeraAPIClient()

// Export class for custom instances
export default OzeraAPIClient
