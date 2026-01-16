/**
 * API client for Ozera inference server.
 */

import type {
  ActivationData,
  ActivationSummary,
  GenerateWithActivationsResponse
} from '../types/model'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
}

// Export singleton instance
export const apiClient = new OzeraAPIClient()

// Export class for custom instances
export default OzeraAPIClient
