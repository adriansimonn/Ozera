/**
 * API client for Ozera inference server.
 */

import type {
  ActivationData,
  ActivationSummary,
  GenerateWithActivationsResponse
} from '../types/model'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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

// Training types
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface TrainingJobRequest {
  dataset_id: string
  base_model: 'nano' | 'mini'
  model_name: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  seq_len?: number
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
  model: 'nano' | 'mini'
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
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
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
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
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
   */
  async getActivationSummary(activationId: string): Promise<ActivationSummary> {
    const response = await fetch(`${this.baseUrl}/activations/${activationId}/summary`)

    if (!response.ok) {
      throw new Error(`Failed to get activation summary: ${response.statusText}`)
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

  /**
   * List all datasets.
   */
  async listDatasets(): Promise<DatasetMetadata[]> {
    const response = await fetch(`${this.baseUrl}/datasets`)

    if (!response.ok) {
      throw new Error(`Failed to list datasets: ${response.statusText}`)
    }

    const data = await response.json()
    return data.datasets
  }

  /**
   * Get dataset details.
   */
  async getDataset(datasetId: string): Promise<DatasetDetail> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}`)

    if (!response.ok) {
      throw new Error(`Failed to get dataset: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Delete a dataset.
   */
  async deleteDataset(datasetId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete dataset: ${response.statusText}`)
    }
  }

  // ============= Training Jobs =============

  /**
   * Get training cost estimate.
   */
  async getTrainingEstimate(
    datasetId: string,
    baseModel: 'nano' | 'mini',
    epochs: number,
    batchSize: number,
    seqLen: number
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || `Failed to start training: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * List all training jobs.
   */
  async listTrainingJobs(): Promise<TrainingJobListItem[]> {
    const response = await fetch(`${this.baseUrl}/training/jobs`)

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
   * List custom trained models.
   */
  async listCustomModels(): Promise<CustomModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/training/models`)

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
}

// Export singleton instance
export const apiClient = new OzeraAPIClient()

// Export class for custom instances
export default OzeraAPIClient
