/**
 * React hooks for text generation.
 */

import { useState, useCallback } from 'react'
import { apiClient, GenerateRequest, GenerateResponse } from '../api/client'

export interface GenerationState {
  loading: boolean
  error: string | null
  result: GenerateResponse | null
}

export interface StreamingState {
  loading: boolean
  streaming: boolean
  error: string | null
  text: string
  insufficientCredits: boolean
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
 * Hook for streaming text generation.
 */
export function useStreamingGeneration() {
  const [state, setState] = useState<StreamingState>({
    loading: false,
    streaming: false,
    error: null,
    text: '',
    insufficientCredits: false,
  })

  const generate = useCallback(async (request: GenerateRequest): Promise<string> => {
    setState({ loading: true, streaming: false, error: null, text: '', insufficientCredits: false })

    let accumulated = ''
    try {
      await apiClient.generateStream(
        request,
        // onToken
        (token) => {
          accumulated += token
          setState(prev => ({
            ...prev,
            loading: false,
            streaming: true,
            text: prev.text + token,
          }))
        },
        // onStart
        () => {
          setState(prev => ({ ...prev, loading: false, streaming: true }))
        },
        // onDone
        () => {
          setState(prev => ({ ...prev, streaming: false }))
        },
        // onError
        (error) => {
          setState(prev => ({
            ...prev,
            loading: false,
            streaming: false,
            error,
          }))
        }
      )
      return accumulated
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      const isInsufficientCredits = errorMessage === 'INSUFFICIENT_CREDITS'
      setState(prev => ({
        ...prev,
        loading: false,
        streaming: false,
        error: isInsufficientCredits ? 'Insufficient credits. Please add more credits to continue.' : errorMessage,
        insufficientCredits: isInsufficientCredits,
      }))
      throw error
    }
  }, [])

  const reset = useCallback(() => {
    setState({ loading: false, streaming: false, error: null, text: '', insufficientCredits: false })
  }, [])

  const clearInsufficientCredits = useCallback(() => {
    setState(prev => ({ ...prev, insufficientCredits: false, error: null }))
  }, [])

  return {
    ...state,
    generate,
    reset,
    clearInsufficientCredits,
  }
}
