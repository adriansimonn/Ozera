/**
 * API client for Ozera inference server.
 */

import { getSupabaseToken } from '../lib/supabase'
import { useAuthStore } from '../stores/authStore'

import type {
  ActivationData,
  ActivationSummary,
  ActivationSummaryWithInfo,
  GenerateWithActivationsResponse,
  OpenSourceModelInfo,
  ModelCacheStatus,
  ModelDownloadResponse,
  ModelFamilyInfo,
  TensorData,
  TopKLogits,
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
import type {
  ClassifyHeadsRequest,
  ClassifyHeadsResponse,
  CompareAttentionRequest,
  CompareAttentionResponse,
  MinePatternRequest,
  MinePatternResponse,
  HeadImportanceRequest,
  HeadImportanceResponse,
} from '../types/analysis'
import type {
  PresetInfo,
  AttentionHeatmapExportRequest,
  MultiHeadHeatmapExportRequest,
  ActivationHistogramExportRequest,
  PatchingComparisonExportRequest,
  BatchExportRequest,
} from '../types/export'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Get auth headers if token is available.
 */
function getAuthHeaders(): Record<string, string> {
  const token = getSupabaseToken()
  if (token) {
    return { Authorization: `Bearer ${token}` }
  }
  return {}
}

/**
 * Refresh the user's credit balance in the auth store (fire-and-forget).
 * Called after credit-consuming API operations so the navbar updates immediately.
 */
function notifyCreditsChanged() {
  useAuthStore.getState().refreshUser().catch(() => {})
}

/**
 * Decode a base64-encoded float32 buffer into a nested JS array matching the given shape.
 */
function decodeBase64Float32(b64: string, shape: number[]): any {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  const floats = new Float32Array(bytes.buffer)
  return reshapeFlat(floats, shape)
}

function reshapeFlat(flat: Float32Array, shape: number[]): any {
  if (shape.length === 0) return flat[0]
  if (shape.length === 1) return Array.from(flat)
  const outerSize = shape[0]
  const innerSize = flat.length / outerSize
  const innerShape = shape.slice(1)
  const result: any[] = []
  for (let i = 0; i < outerSize; i++) {
    result.push(reshapeFlat(flat.subarray(i * innerSize, (i + 1) * innerSize), innerShape))
  }
  return result
}

/**
 * Recursively decode any base64-encoded TensorData objects in an activation response.
 * Converts `{ values: "<base64>", encoding: "base64_float32", shape: [...] }` to nested arrays.
 */
function decodeTensorData(obj: any): any {
  if (obj == null || typeof obj !== 'object') return obj

  // Check if this is a base64-encoded TensorData
  if (obj.encoding === 'base64_float32' && typeof obj.values === 'string' && Array.isArray(obj.shape)) {
    return {
      ...obj,
      values: decodeBase64Float32(obj.values, obj.shape),
      encoding: undefined,
    }
  }

  // Recurse into arrays
  if (Array.isArray(obj)) {
    return obj.map(decodeTensorData)
  }

  // Recurse into objects
  const decoded: any = {}
  for (const key of Object.keys(obj)) {
    decoded[key] = decodeTensorData(obj[key])
  }
  return decoded
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

const DEFAULT_TIMEOUT_MS = 30_000

async function fetchWithTimeout(
  input: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(input, { ...init, signal: controller.signal })
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs}ms`)
    }
    throw err
  } finally {
    clearTimeout(id)
  }
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
    const response = await fetchWithTimeout(`${this.baseUrl}/health`)

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List available models.
   */
  async listModels(): Promise<string[]> {
    const response = await fetchWithTimeout(`${this.baseUrl}/models`)

    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get information about a specific model.
   */
  async getModelInfo(modelName: string): Promise<ModelInfo> {
    const response = await fetchWithTimeout(`${this.baseUrl}/models/${modelName}`)

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
    const response = await fetchWithTimeout(`${this.baseUrl}/models/${modelName}/prepare`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/generate`, {
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

    notifyCreditsChanged()
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
      notifyCreditsChanged()
    }
  }

  /**
   * Generate text with activation capture for visualization.
   */
  async generateWithActivations(request: GenerateRequest): Promise<GenerateWithActivationsResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/generate/with-activations`, {
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

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Get activation data by ID.
   */
  async getActivations(activationId: string): Promise<ActivationData> {
    const response = await fetchWithTimeout(`${this.baseUrl}/activations/${activationId}`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get activations: ${response.statusText}`)
    }

    const data = await response.json()
    return decodeTensorData(data)
  }

  /**
   * Get activation summary (metadata only, no tensors).
   * Enhanced with layer info and tensor info for lazy loading.
   */
  async getActivationSummary(activationId: string): Promise<ActivationSummaryWithInfo> {
    const response = await fetchWithTimeout(`${this.baseUrl}/activations/${activationId}/summary`, {
      headers: getAuthHeaders(),
    })

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
    const response = await fetchWithTimeout(`${this.baseUrl}/activations/${activationId}/layer/${layerIdx}`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get layer activations: ${response.statusText}`)
    }

    const data = await response.json()
    return decodeTensorData(data)
  }

  /**
   * Get a specific top-level tensor (lazy loading).
   * Valid tensor names: token_embeddings, positional_embeddings, combined_embeddings, final_layer_norm, logits
   */
  async getTensorActivation(activationId: string, tensorName: string): Promise<{ tensor_name: string; data: TensorData | TopKLogits }> {
    const response = await fetchWithTimeout(`${this.baseUrl}/activations/${activationId}/tensor/${tensorName}`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get tensor activation: ${response.statusText}`)
    }

    const data = await response.json()
    return decodeTensorData(data)
  }

  /**
   * List all stored activations.
   */
  async listActivations(): Promise<ActivationSummary[]> {
    const response = await fetchWithTimeout(`${this.baseUrl}/activations`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to list activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete activation data.
   */
  async deleteActivations(activationId: string): Promise<void> {
    const response = await fetchWithTimeout(`${this.baseUrl}/activations/${activationId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
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

    const response = await fetchWithTimeout(`${this.baseUrl}/datasets/upload`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Upload failed: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get the user's currently uploaded dataset, if any.
   */
  async getCurrentDataset(): Promise<DatasetMetadata | null> {
    const response = await fetchWithTimeout(`${this.baseUrl}/datasets/current`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      return null
    }

    const data = await response.json()
    return data?.dataset || null
  }

  /**
   * List generic datasets available for training.
   */
  async listGenericDatasets(): Promise<GenericDatasetInfo[]> {
    const response = await fetchWithTimeout(`${this.baseUrl}/datasets/generic/list`)

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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/gpu-pricing`)

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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/estimate`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/jobs`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/jobs`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/jobs/${jobId}`, {
      headers: getAuthHeaders(),
    })

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
    const response = await fetch(`${this.baseUrl}/training/jobs/${jobId}/stream`, {
      headers: getAuthHeaders(),
    })

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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/jobs/${jobId}/cancel`, {
      method: 'POST',
      headers: getAuthHeaders(),
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/count`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/${modelId}`, {
      method: 'DELETE',
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to delete model: ${response.statusText}`)
    }
  }

  /**
   * Download a custom model as a zip file.
   */
  async downloadCustomModel(modelId: string): Promise<void> {
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/${modelId}/download`, {
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

    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/upload`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/uploaded`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/training/models/uploaded/${modelId}`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/models`)

    if (!response.ok) {
      throw new Error(`Failed to list open-source models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get info for a specific open-source model.
   */
  async getOpenSourceModelInfo(modelId: string): Promise<OpenSourceModelInfo> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/models/${modelId}`)

    if (!response.ok) {
      throw new Error(`Failed to get open-source model info: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List open-source models grouped by family.
   */
  async listOpenSourceFamilies(): Promise<ModelFamilyInfo[]> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/families`)

    if (!response.ok) {
      throw new Error(`Failed to list model families: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Check cache status for an open-source model.
   */
  async getOpenSourceCacheStatus(modelId: string): Promise<ModelCacheStatus> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/cache/${modelId}/status`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get cache status: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Trigger download of an open-source model to Modal cache.
   */
  async downloadOpenSourceModel(modelId: string, force: boolean = false): Promise<ModelDownloadResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/cache/${modelId}/download?force=${force}`, {
      method: 'POST',
      headers: getAuthHeaders(),
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
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/cache/${modelId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to delete cached model: ${response.statusText}`)
    }
  }

  /**
   * Warmup an open-source model (pre-load into GPU memory).
   */
  async warmupOpenSourceModel(modelId: string): Promise<{ status: string; model: string; display_name: string; parameters: number; gpu_tier: string }> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/models/${modelId}/warmup`, {
      method: 'POST',
      headers: getAuthHeaders(),
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
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/generate`, {
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

    notifyCreditsChanged()
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
      notifyCreditsChanged()
    }
  }

  /**
   * Generate text with activations using an open-source model.
   */
  async generateOpenSourceWithActivations(request: GenerateRequest): Promise<GenerateWithActivationsResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/open-source/generate/with-activations`, {
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

    notifyCreditsChanged()
    return response.json()
  }

  // ============= Activation Patching =============

  /**
   * Capture source activations for patching.
   */
  async captureActivations(request: CaptureActivationsRequest): Promise<CaptureActivationsResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/capture`, {
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
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/activations`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to list captured activations: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get detailed info for a captured activation.
   */
  async getCapturedActivation(activationId: string): Promise<CapturedActivationDetail> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/activations/${activationId}`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get captured activation: ${response.statusText}`)
    }

    const data = await response.json()
    return decodeTensorData(data)
  }

  /**
   * Delete a specific captured activation.
   */
  async deleteCapturedActivation(activationId: string): Promise<void> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/activations/${activationId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to delete captured activation: ${response.statusText}`)
    }
  }

  /**
   * Clear all captured activations.
   */
  async clearCapturedActivations(): Promise<void> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/activations`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to clear captured activations: ${response.statusText}`)
    }
  }

  /**
   * Run a full patching experiment (captures source, then runs patching).
   */
  async runPatchingExperiment(request: RunPatchingRequest): Promise<PatchingResult> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/run`, {
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

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Run patching with pre-captured activations.
   */
  async runPatchingWithCaptured(request: RunPatchingWithCapturedRequest): Promise<PatchingResult> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/run-with-captured`, {
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

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Get list of available models for patching.
   */
  async getPatchingModels(): Promise<PatchingModelInfo[]> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/models`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to get patching models: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get layer info for a specific model (for patch configuration).
   */
  async getModelLayerInfo(modelId: string): Promise<ModelLayerInfo> {
    const response = await fetchWithTimeout(`${this.baseUrl}/patching/models/${modelId}/layers`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error(`Failed to get model layer info: ${response.statusText}`)
    }

    return response.json()
  }

  // ============= Attention Pattern Analysis =============

  /**
   * Classify attention heads for a captured activation.
   * Identifies head types: induction, previous token, positional, copying.
   * Requires authentication and charges credits.
   */
  async classifyHeads(request: ClassifyHeadsRequest): Promise<ClassifyHeadsResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/analysis/attention/classify-heads`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402 || error.detail === 'INSUFFICIENT_CREDITS') {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to classify heads: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Compare attention patterns between two captured activations.
   * Requires authentication and charges credits.
   */
  async compareAttention(request: CompareAttentionRequest): Promise<CompareAttentionResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/analysis/attention/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402 || error.detail === 'INSUFFICIENT_CREDITS') {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to compare attention: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Run comprehensive pattern mining on captured activations.
   * Requires authentication and charges credits.
   */
  async minePatterns(request: MinePatternRequest): Promise<MinePatternResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/analysis/attention/mine-patterns`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402 || error.detail === 'INSUFFICIENT_CREDITS') {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to mine patterns: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Get importance scores for attention heads.
   * Requires authentication and charges credits.
   */
  async getHeadImportance(request: HeadImportanceRequest): Promise<HeadImportanceResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/analysis/attention/head-importance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      if (response.status === 402 || error.detail === 'INSUFFICIENT_CREDITS') {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      throw new Error(error.detail || `Failed to get head importance: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  // ============= Figure Export =============

  /**
   * Get available publication presets.
   */
  async getExportPresets(): Promise<Record<string, PresetInfo>> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/presets`)

    if (!response.ok) {
      throw new Error(`Failed to get export presets: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Export attention heatmap as publication-ready figure.
   * Returns the file as a Blob for download.
   */
  async exportAttentionHeatmap(request: AttentionHeatmapExportRequest): Promise<Blob> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/attention-heatmap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to export attention heatmap: ${response.statusText}`)
    }

    return response.blob()
  }

  /**
   * Export multi-head attention heatmap as publication-ready figure.
   * Returns the file as a Blob for download.
   */
  async exportMultiHeadHeatmap(request: MultiHeadHeatmapExportRequest): Promise<Blob> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/attention-heatmap/multi-head`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to export multi-head heatmap: ${response.statusText}`)
    }

    return response.blob()
  }

  /**
   * Export activation histogram as publication-ready figure.
   * Returns the file as a Blob for download.
   */
  async exportActivationHistogram(request: ActivationHistogramExportRequest): Promise<Blob> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/activation-histogram`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to export activation histogram: ${response.statusText}`)
    }

    return response.blob()
  }

  /**
   * Export patching comparison figure.
   * Returns the file as a Blob for download.
   */
  async exportPatchingComparison(request: PatchingComparisonExportRequest): Promise<Blob> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/patching-comparison`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to export patching comparison: ${response.statusText}`)
    }

    return response.blob()
  }

  /**
   * Export multiple figures as a ZIP archive.
   * Returns the ZIP file as a Blob for download.
   */
  async exportBatch(request: BatchExportRequest): Promise<Blob> {
    const response = await fetchWithTimeout(`${this.baseUrl}/export/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to batch export: ${response.statusText}`)
    }

    return response.blob()
  }
}

// ============= SAE API Types =============

export interface ExternalSAEInfo {
  id: string
  path: string
  source?: string
  source_id?: string
  display_name?: string
  base_model?: string
  hookpoint?: string
  activation_type?: string
  d_input?: number
  d_hidden?: number
  created_at?: string
  num_parameters?: number
  extra?: Record<string, any>
}

export interface SAEListResponse {
  models: {
    [modelName: string]: {
      num_layers: number
      d_model: number
      sae_d_hidden: number
      saes: Array<{
        layer: number
        activation_type: 'residual' | 'mlp_output'
        path: string
        d_input?: number
        d_hidden?: number
        activation?: string
        created_at?: string
        num_parameters?: number
        training?: {
          final_loss?: number
          final_l0?: number
          dead_features?: number
        }
      }>
    }
  }
  external_saes?: ExternalSAEInfo[]
  total_saes: number
}

export interface ExternalSAELoadRequest {
  repo_id: string
  hookpoint?: string
  name?: string
}

export interface ExternalSAELoadResponse {
  status?: string
  sae_id?: string
  display_name?: string
  source?: string
  source_id?: string
  base_model?: string
  hookpoint?: string
  d_input?: number
  d_hidden?: number
  activation_type?: string
  extra?: Record<string, any>
  error?: string
}

export interface ExternalSAESourcesResponse {
  repo_id: string
  available: Array<{
    hookpoint: string
    repo_id: string
    d_in?: number
    num_latents?: number
    k?: number
    layer?: number
    width?: number
    l0?: number
    site?: string
    error?: string
    [key: string]: any
  }>
  count: number
  error?: string
}

export interface ExternalSAEFeatureInfoResponse {
  sae_id: string
  feature_id: number
  d_input: number
  d_hidden: number
  decoder_direction: number[]
  decoder_norm: number
  encoder_weights: number[]
  encoder_bias: number | null
  display_name: string
  activation_type: string
  error?: string
}

export interface ExternalSAEUploadRequest {
  name: string
  weights_base64: string
  config?: Record<string, any>
}

export interface SAEAnalyzeRequest {
  model: 'nano' | 'mini'
  layer: number
  activation_type: 'residual' | 'mlp_output'
  text: string
  top_k?: number
}

export interface ExternalSAEAnalyzeRequest {
  sae_id: string
  text: string
  top_k?: number
}

export interface SAEAnalyzeResponse {
  tokens: string[]
  token_ids: number[]
  num_tokens: number
  model: string
  layer: number
  activation_type: string
  features: {
    shape: [number, number]
    sparse_activations: Array<Record<string, number>>
    l0_per_token: number[]
  }
  top_features: Array<{
    token_idx: number
    token: string
    feature_id: number
    activation: number
  }>
  metrics: {
    avg_l0: number
    max_activation: number
    num_active_features: number
    total_features: number
  }
  error?: string
}

export interface SAEFeatureInfoResponse {
  feature_id: number
  model: string
  layer: number
  activation_type: string
  d_input: number
  d_hidden: number
  decoder_direction: number[]
  decoder_norm: number
  encoder_weights: number[]
  encoder_bias: number | null
  error?: string
}

export interface SAEAnalyzeBatchRequest {
  model: 'nano' | 'mini'
  layer: number
  activation_type: 'residual' | 'mlp_output'
  texts: string[]
  top_k_per_text?: number
}

export interface SAEAnalyzeBatchResponse {
  model: string
  layer: number
  activation_type: string
  num_texts: number
  results: Array<{
    text: string
    tokens: string[]
    num_tokens: number
    top_features: Array<{
      token_idx: number
      token: string
      feature_id: number
      activation: number
    }>
    metrics: {
      avg_l0: number
      num_active_features: number
    }
  }>
  error?: string
}

// ============= SAE API Client =============

class SAEAPIClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    // Use backend API instead of direct Modal access
    this.baseUrl = baseUrl
  }

  /**
   * List all available SAEs with metadata.
   * Public endpoint (no auth required).
   */
  async listSAEs(): Promise<SAEListResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/list`)

    if (!response.ok) {
      throw new Error(`Failed to list SAEs: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Analyze text through transformer + SAE.
   * Requires authentication and charges credits.
   */
  async analyzeText(request: SAEAnalyzeRequest): Promise<SAEAnalyzeResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      // Check for insufficient credits (402 Payment Required)
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to analyze text: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Analyze text using an external SAE.
   * Loads the base model from HuggingFace and runs the full pipeline.
   * Requires authentication and charges credits.
   */
  async analyzeExternalSAE(request: ExternalSAEAnalyzeRequest): Promise<SAEAnalyzeResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to analyze with external SAE: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Get information about a specific SAE feature.
   * Requires authentication and charges a small fee.
   */
  async getFeatureInfo(
    model: 'nano' | 'mini',
    layer: number,
    activationType: 'residual' | 'mlp_output',
    featureId: number
  ): Promise<SAEFeatureInfoResponse> {
    const params = new URLSearchParams({
      model,
      layer: layer.toString(),
      activation_type: activationType,
      feature_id: featureId.toString(),
    })

    const response = await fetchWithTimeout(`${this.baseUrl}/sae/feature?${params}`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to get feature info: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Analyze multiple texts through transformer + SAE in batch.
   * Requires authentication and charges credits.
   */
  async analyzeBatch(request: SAEAnalyzeBatchRequest): Promise<SAEAnalyzeBatchResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/analyze-batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to analyze batch: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Compare features between two SAEs.
   * Requires authentication and charges credits.
   */
  async compareSAEs(request: SAECompareRequest): Promise<SAECompareResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to compare SAEs: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Compute layer-by-layer CKA similarity matrix.
   * Requires authentication and charges credits.
   */
  async compareLayers(request: SAECompareLayersRequest): Promise<SAECompareLayersResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/compare-layers`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to compare layers: ${response.statusText}`)
    }

    notifyCreditsChanged()
    return response.json()
  }

  /**
   * Health check endpoint.
   */
  async health(): Promise<{ status: string; cuda_available: boolean; cuda_device: string | null }> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/health`)

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`)
    }

    return response.json()
  }

  // ============= External SAE Methods =============

  /**
   * Load an external SAE from HuggingFace or Gemma Scope.
   * Requires authentication.
   */
  async loadExternalSAE(request: ExternalSAELoadRequest): Promise<ExternalSAELoadResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/load`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to load external SAE: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List available hookpoints in an external SAE repository.
   */
  async listExternalSAESources(repoId: string): Promise<ExternalSAESourcesResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/list-sources`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify({ repo_id: repoId }),
    })

    if (!response.ok) {
      throw new Error(`Failed to list external SAE sources: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List loaded external SAEs for the current user.
   * Requires authentication.
   */
  async listLoadedExternalSAEs(): Promise<{ external_saes: ExternalSAEInfo[]; count: number }> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/list-loaded`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to list loaded external SAEs: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Check if the current user already has an uploaded SAE.
   * Requires authentication.
   */
  async hasUploadedSAE(): Promise<{ has_upload: boolean; upload_name: string | null; upload_sae_id: string | null }> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/has-upload`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to check upload status: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete a loaded external SAE.
   * Requires authentication.
   */
  async deleteExternalSAE(saeId: string): Promise<{ status?: string; error?: string }> {
    const params = new URLSearchParams({ sae_id: saeId })
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/delete?${params}`, {
      method: 'DELETE',
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to delete external SAE: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get feature info for an external SAE.
   * Requires authentication and charges a small fee.
   */
  async getExternalFeatureInfo(
    saeId: string,
    featureId: number
  ): Promise<ExternalSAEFeatureInfoResponse> {
    const params = new URLSearchParams({
      sae_id: saeId,
      feature_id: featureId.toString(),
    })

    const response = await fetchWithTimeout(`${this.baseUrl}/sae/external/feature?${params}`, {
      headers: {
        ...getAuthHeaders(),
      },
    })

    if (!response.ok) {
      if (response.status === 402) {
        throw new Error('INSUFFICIENT_CREDITS')
      }
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to get external feature info: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Upload a user SAE (base64-encoded safetensors).
   * Requires authentication.
   */
  async uploadSAE(request: ExternalSAEUploadRequest): Promise<ExternalSAELoadResponse> {
    const response = await fetchWithTimeout(`${this.baseUrl}/sae/upload`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Failed to upload SAE: ${response.statusText}`)
    }

    return response.json()
  }
}

// ============= SAE Comparison API Types =============

export interface SAECompareRequest {
  model_a: 'nano' | 'mini'
  layer_a: number
  activation_type_a: 'residual' | 'mlp_output'
  model_b: 'nano' | 'mini'
  layer_b: number
  activation_type_b: 'residual' | 'mlp_output'
  external_id_a?: string
  external_id_b?: string
  text: string
  top_k?: number
}

export interface SAECompareResponse {
  overall_similarity: number
  cka_score: number
  matched_features: number
  unmatched_a: number
  unmatched_b: number
  top_matches: Array<{
    feature_a: number
    feature_b: number
    similarity: number
    shared_tokens: string[]
    label_a: string
    label_b: string
  }>
  divergent_features_a: number[]
  divergent_features_b: number[]
  similarity_matrix_sample: number[][] | null
  feature_indices_a: number[] | null
  feature_indices_b: number[] | null
  sae_a: {
    model: string
    layer: number
    activation_type: string
    d_hidden: number
  }
  sae_b: {
    model: string
    layer: number
    activation_type: string
    d_hidden: number
  }
  tokens: string[]
  num_tokens: number
  error?: string
}

export interface SAECompareLayersRequest {
  model_a: 'nano' | 'mini'
  activation_type_a: 'residual' | 'mlp_output'
  model_b?: 'nano' | 'mini'
  activation_type_b?: 'residual' | 'mlp_output'
  external_id_a?: string
  external_id_b?: string
  text: string
}

export interface SAECompareLayersResponse {
  model_a: string
  activation_type_a: string
  layers_a: number[]
  num_layers_a: number
  model_b: string
  activation_type_b: string
  layers_b: number[]
  num_layers_b: number
  cka_matrix: number[][]
  num_tokens: number
  error?: string
}

// Export SAE client singleton
export const saeClient = new SAEAPIClient()

// Export singleton instance
export const apiClient = new OzeraAPIClient()

// Export class for custom instances
export default OzeraAPIClient
