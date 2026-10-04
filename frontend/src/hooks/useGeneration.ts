/**
 * React hooks for text generation.
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { apiClient, GenerateRequest, GenerateResponse, FinishReason, GenerationDone } from '../api/client'
import type { MonitorTrace } from '../types/probes'
import { appendMonitorScores } from '../components/probes/probeUtils'

export interface GenerationState {
  loading: boolean
  error: string | null
  result: GenerateResponse | null
}

export interface StreamingState {
  loading: boolean  // Request sent, no tokens yet (includes the model warming up)
  streaming: boolean  // Tokens arriving
  capturing: boolean  // Generation done, activations being captured
  stopping: boolean  // Stop requested, waiting for the stream to end
  error: string | null
  text: string
  insufficientCredits: boolean
  // A monitoring probe's scores so far (requests with a probe_id)
  monitor: MonitorTrace | null
}

export interface StreamingResult {
  text: string
  generatedTokens: number | null
  finishReason: FinishReason | null
  activationId: string | null
  monitor: MonitorTrace | null
}

const IDLE_STATE: StreamingState = {
  loading: false,
  streaming: false,
  capturing: false,
  stopping: false,
  error: null,
  text: '',
  insufficientCredits: false,
  monitor: null,
}

function newGenerationId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('')
}

/**
 * Hook for non-streaming text generation.
 */
export function useGeneration() {
  const [state, setState] = useState<GenerationState>({
    loading: false,
    error: null,
    result: null,
  })

  const generate = useCallback(async (request: GenerateRequest) => {
    setState({ loading: true, error: null, result: null })

    try {
      const result = await apiClient.generate(request)
      setState({ loading: false, error: null, result })
      return result
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setState({ loading: false, error: errorMessage, result: null })
      throw error
    }
  }, [])

  const reset = useCallback(() => {
    setState({ loading: false, error: null, result: null })
  }, [])

  return {
    ...state,
    generate,
    reset,
  }
}

/**
 * Hook for streaming text generation, optionally with activation capture, that can be stopped.
 */
export function useStreamingGeneration() {
  const [state, setState] = useState<StreamingState>(IDLE_STATE)
  const generationIdRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Leaving the page closes the stream; the backend then stops the generation
  useEffect(() => () => abortRef.current?.abort(), [])

  /**
   * Stream a generation. Resolves with its result, or null if it failed (the error is in
   * state); rejects if the request is refused (e.g. INSUFFICIENT_CREDITS).
   */
  const generate = useCallback(async (
    request: GenerateRequest,
    options: { withActivations?: boolean } = {},
  ): Promise<StreamingResult | null> => {
    const generationId = newGenerationId()
    const controller = new AbortController()
    generationIdRef.current = generationId
    abortRef.current = controller
    setState({ ...IDLE_STATE, loading: true })

    let accumulated = ''
    let monitor: MonitorTrace | null = null
    let done: GenerationDone | null = null
    let failed = false
    const stream = options.withActivations ? apiClient.generateWithActivationsStream : apiClient.generateStream

    try {
      await stream.call(apiClient, { ...request, generation_id: generationId }, {
        onToken: (token) => {
          accumulated += token
          setState(prev => ({ ...prev, loading: false, streaming: true, text: prev.text + token }))
        },
        onCapturing: () => {
          setState(prev => ({ ...prev, loading: false, streaming: false, capturing: true }))
        },
        onProbe: (chunk) => {
          monitor = appendMonitorScores(monitor, chunk)
          const trace = monitor
          setState(prev => ({ ...prev, monitor: trace }))
        },
        onDone: (event) => {
          done = event
        },
        onError: (error) => {
          failed = true
          setState(prev => ({ ...prev, error }))
        },
      }, controller.signal)
    } catch (error) {
      if (controller.signal.aborted) {
        // Closed by stop() after its request failed, or on unmount: the backend stopped it
        failed = true
        setState(prev => ({ ...prev, error: prev.error ?? 'Generation stopped before it could report its result.' }))
      } else {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        const isInsufficientCredits = errorMessage === 'INSUFFICIENT_CREDITS'
        setState(prev => ({
          ...prev,
          loading: false,
          streaming: false,
          capturing: false,
          stopping: false,
          error: isInsufficientCredits ? 'Insufficient credits. Please add more credits to continue.' : errorMessage,
          insufficientCredits: isInsufficientCredits,
        }))
        throw error
      }
    } finally {
      if (generationIdRef.current === generationId) {
        generationIdRef.current = null
        abortRef.current = null
        setState(prev => ({ ...prev, loading: false, streaming: false, capturing: false, stopping: false }))
      }
    }

    if (failed || !done) return null
    const result: GenerationDone = done
    return {
      text: accumulated,
      generatedTokens: result.generated_tokens ?? null,
      finishReason: result.finish_reason ?? null,
      activationId: result.activation_id ?? null,
      monitor,
    }
  }, [])

  /**
   * Stop the running generation. Its stream ends with what was generated so far (charged
   * for those tokens). If the stop request fails, the stream is closed instead, which
   * also stops it.
   */
  const stop = useCallback(() => {
    const generationId = generationIdRef.current
    if (!generationId) return
    setState(prev => (prev.stopping ? prev : { ...prev, stopping: true }))
    const controller = abortRef.current
    apiClient.stopGeneration(generationId).catch(() => controller?.abort())
  }, [])

  const reset = useCallback(() => {
    setState(IDLE_STATE)
  }, [])

  const clearInsufficientCredits = useCallback(() => {
    setState(prev => ({ ...prev, insufficientCredits: false, error: null }))
  }, [])

  return {
    ...state,
    generate,
    stop,
    reset,
    clearInsufficientCredits,
  }
}
