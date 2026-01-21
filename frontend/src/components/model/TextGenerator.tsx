/**
 * Text generation component with model selection and streaming support.
 * Supports both local and Modal cloud inference with cold start handling.
 */

import React, { useState, useEffect, useRef } from 'react'
import { useStreamingGeneration } from '../../hooks/useGeneration'
import { useModels } from '../../hooks/useModels'
import { apiClient } from '../../api/client'

interface TextGeneratorProps {
  defaultModel?: string
  defaultPrompt?: string
  onGenerate?: (text: string) => void
  onActivationGenerated?: (activationId: string) => void
  onGeneratingChange?: (isGenerating: boolean) => void
  onModelChange?: (model: string) => void
  externalModel?: string
  onShowPurchaseCredits?: () => void
}

// Cold start threshold - show "warming up" message after this delay
const COLD_START_THRESHOLD_MS = 3000

export const TextGenerator: React.FC<TextGeneratorProps> = ({
  defaultModel = 'nano',
  defaultPrompt = '',
  onGenerate,
  onActivationGenerated,
  onGeneratingChange,
  onModelChange,
  externalModel,
  onShowPurchaseCredits,
}) => {
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [internalModel, setInternalModel] = useState<string>(defaultModel)

  // Use external model if provided, otherwise use internal state
  const model = externalModel ?? internalModel

  const handleModelChange = (newModel: string) => {
    setInternalModel(newModel)
    onModelChange?.(newModel)
  }
  const [maxTokens, setMaxTokens] = useState(200)
  const [temperature, setTemperature] = useState(0.7)
  const [topK, setTopK] = useState(40)
  const [capturingActivations, setCapturingActivations] = useState(false)
  const [isWarmingUp, setIsWarmingUp] = useState(false)
  const warmupTimerRef = useRef<NodeJS.Timeout | null>(null)

  const { models, loading: modelsLoading, error: modelsError } = useModels()
  const { text, loading, streaming, error, insufficientCredits, generate, reset, clearInsufficientCredits } = useStreamingGeneration()
  const [activationInsufficientCredits, setActivationInsufficientCredits] = useState(false)

  // Clear warmup timer on unmount
  useEffect(() => {
    return () => {
      if (warmupTimerRef.current) {
        clearTimeout(warmupTimerRef.current)
      }
    }
  }, [])

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      return
    }

    reset()
    setIsWarmingUp(false)

    // Start a timer to show "warming up" message if response takes too long
    warmupTimerRef.current = setTimeout(() => {
      setIsWarmingUp(true)
    }, COLD_START_THRESHOLD_MS)

    try {
      await generate({
        prompt: prompt.trim(),
        model,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
      })

      if (onGenerate) {
        onGenerate(text)
      }
    } catch (err) {
      console.error('Generation error:', err)
    } finally {
      // Clear warmup timer
      if (warmupTimerRef.current) {
        clearTimeout(warmupTimerRef.current)
        warmupTimerRef.current = null
      }
      setIsWarmingUp(false)
    }
  }

  const handleReset = () => {
    reset()
    setPrompt('')
  }

  const handleGenerateWithActivations = async () => {
    if (!prompt.trim()) {
      return
    }

    setIsWarmingUp(false)
    setActivationInsufficientCredits(false)

    // Start warmup timer
    warmupTimerRef.current = setTimeout(() => {
      setIsWarmingUp(true)
    }, COLD_START_THRESHOLD_MS)

    try {
      setCapturingActivations(true)
      onGeneratingChange?.(true)
      const result = await apiClient.generateWithActivations({
        prompt: prompt.trim(),
        model,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
      })

      // Call callback with activation ID if provided
      if (onActivationGenerated) {
        onActivationGenerated(result.activation_id)
      }
    } catch (err) {
      console.error('Activation generation error:', err)
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      if (errorMessage === 'INSUFFICIENT_CREDITS') {
        setActivationInsufficientCredits(true)
      } else {
        alert('Failed to generate activations: ' + errorMessage)
      }
    } finally {
      // Clear warmup timer
      if (warmupTimerRef.current) {
        clearTimeout(warmupTimerRef.current)
        warmupTimerRef.current = null
      }
      setIsWarmingUp(false)
      setCapturingActivations(false)
      onGeneratingChange?.(false)
    }
  }

  return (
    <div className="text-generator">
      <div className="generator-header">
        <h2>Text Generation</h2>
        {modelsLoading && <span className="status">Loading models...</span>}
        {modelsError && <span className="error">Error: {modelsError}</span>}
      </div>

      <div className="controls">
        <div className="control-group">
          <label htmlFor="model-select">Model:</label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={loading || streaming || modelsLoading || models.length === 0}
          >
            {models.length === 0 && (
              <option value="">No models available</option>
            )}
            {/* Base models */}
            {models.filter(m => m === 'nano' || m === 'mini').map((m) => (
              <option key={m} value={m}>
                ozera-{m}
              </option>
            ))}
            {/* Custom models - show with different formatting */}
            {models.filter(m => m !== 'nano' && m !== 'mini').length > 0 && (
              <option disabled>── Custom Models ──</option>
            )}
            {models.filter(m => m !== 'nano' && m !== 'mini').map((m) => (
              <option key={m} value={m}>
                {m} (custom)
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="max-tokens">Max Tokens: {maxTokens}</label>
          <input
            id="max-tokens"
            type="range"
            min="1"
            max="500"
            step="1"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            disabled={loading || streaming}
          />
        </div>
      </div>

      <div className="controls">
        <div className="control-group">
          <label htmlFor="temperature">Temperature: {temperature.toFixed(2)}</label>
          <input
            id="temperature"
            type="range"
            min="0"
            max="2.0"
            step="0.01"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            disabled={loading || streaming}
          />
        </div>

        <div className="control-group">
          <label htmlFor="top-k">Top-K: {topK}</label>
          <input
            id="top-k"
            type="range"
            min="1"
            max="100"
            step="1"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            disabled={loading || streaming}
          />
        </div>
      </div>

      <div className="prompt-area">
        <label htmlFor="prompt">Prompt:</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          rows={4}
          disabled={loading || streaming}
        />
      </div>

      <div className="actions">
        <button
          onClick={handleGenerate}
          disabled={loading || streaming || capturingActivations || !prompt.trim()}
          className="btn-primary"
        >
          {loading && isWarmingUp ? 'Warming up model...' : loading ? 'Loading...' : streaming ? 'Generating...' : 'Generate'}
        </button>
        <button
          onClick={handleGenerateWithActivations}
          disabled={loading || streaming || capturingActivations || !prompt.trim()}
          className="btn-visualize"
        >
          {capturingActivations && isWarmingUp ? 'Warming up...' : capturingActivations ? 'Capturing...' : 'Visualize'}
        </button>
        <button
          onClick={handleReset}
          disabled={loading || streaming || capturingActivations}
          className="btn-secondary"
        >
          Reset
        </button>
      </div>

      {isWarmingUp && (loading || capturingActivations) && (
        <div className="warmup-message">
          <span className="warmup-spinner"></span>
          <span>Model is warming up. This may take 10-30 seconds on first request...</span>
        </div>
      )}

      {(insufficientCredits || activationInsufficientCredits) && (
        <div className="insufficient-credits-message">
          <div className="insufficient-credits-content">
            <strong>Insufficient Credits</strong>
            <p>You don't have enough credits to generate text. Please add more credits to continue.</p>
            {onShowPurchaseCredits && (
              <button
                className="add-credits-button"
                onClick={() => {
                  clearInsufficientCredits()
                  setActivationInsufficientCredits(false)
                  onShowPurchaseCredits()
                }}
              >
                Add Credits
              </button>
            )}
          </div>
        </div>
      )}

      {error && !insufficientCredits && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {text && (
        <div className="output-area">
          <div className="output-header">
            <h3>Generated Text</h3>
            {streaming && <span className="streaming-indicator">●</span>}
          </div>
          <div className="output-content">
            {text}
            {streaming && <span className="cursor">|</span>}
          </div>
        </div>
      )}

      <style>{`
        .text-generator {
          max-width: 100%;
          padding: 2.5rem;
        }

        .generator-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2.5rem;
          padding-bottom: 1.5rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .generator-header h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 600;
          color: #ffffff;
          letter-spacing: -0.02em;
        }

        .status {
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.85rem;
          font-weight: 400;
        }

        .controls {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 2rem;
          margin-bottom: 2.5rem;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .control-group label {
          font-weight: 500;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .control-group select {
          padding: 0.875rem 1rem;
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          font-size: 0.95rem;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .control-group select:hover {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .control-group select:focus {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            0 0 0 2px rgba(255, 255, 255, 0.05);
        }

        .control-group input[type="range"] {
          width: 100%;
          height: 2px;
          background: rgba(255, 255, 255, 0.1);
          outline: none;
          -webkit-appearance: none;
        }

        .control-group input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 14px;
          height: 14px;
          background: #ffffff;
          border-radius: 50%;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.3),
            0 0 0 2px rgba(255, 255, 255, 0.1);
        }

        .control-group input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.4),
            0 0 0 3px rgba(255, 255, 255, 0.15),
            0 0 12px rgba(255, 255, 255, 0.3);
        }

        .prompt-area {
          margin-bottom: 2rem;
        }

        .prompt-area label {
          display: block;
          margin-bottom: 1rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.85rem;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .prompt-area textarea {
          width: 100%;
          padding: 1.25rem;
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.95rem;
          resize: vertical;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          line-height: 1.6;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .prompt-area textarea:hover {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .prompt-area textarea:focus {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            0 0 0 2px rgba(255, 255, 255, 0.05);
        }

        .prompt-area textarea::placeholder {
          color: rgba(255, 255, 255, 0.3);
        }

        .actions {
          display: flex;
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .btn-primary,
        .btn-secondary,
        .btn-visualize {
          padding: 1rem 2rem;
          border: 1px solid rgba(255, 255, 255, 0.15);
          font-size: 0.9rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          position: relative;
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .btn-primary {
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
        }

        .btn-primary:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.3);
          box-shadow:
            0 6px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          transform: translateY(-2px);
        }

        .btn-visualize {
          background: rgba(255, 255, 255, 0.06);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.12);
        }

        .btn-visualize:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 6px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
          transform: translateY(-2px);
        }

        .btn-secondary {
          background: rgba(255, 255, 255, 0.02);
          color: rgba(255, 255, 255, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.18);
          color: #ffffff;
          box-shadow:
            0 6px 16px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
          transform: translateY(-2px);
        }

        .btn-primary:disabled,
        .btn-secondary:disabled,
        .btn-visualize:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .warmup-message {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem 1.25rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.8);
          margin-bottom: 1.5rem;
          font-size: 0.9rem;
          animation: fadeIn 0.3s ease-in-out;
        }

        .warmup-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.2);
          border-top-color: rgba(255, 255, 255, 0.8);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .error-message {
          padding: 1rem 1.25rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.8);
          margin-bottom: 2rem;
          font-size: 0.9rem;
        }

        .insufficient-credits-message {
          padding: 1.5rem;
          background: rgba(255, 200, 100, 0.08);
          border: 1px solid rgba(255, 200, 100, 0.25);
          margin-bottom: 2rem;
          animation: fadeIn 0.3s ease-in-out;
        }

        .insufficient-credits-content {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .insufficient-credits-content strong {
          color: rgba(255, 220, 150, 0.95);
          font-size: 1rem;
        }

        .insufficient-credits-content p {
          color: rgba(255, 255, 255, 0.7);
          font-size: 0.9rem;
          margin: 0;
        }

        .add-credits-button {
          align-self: flex-start;
          padding: 0.75rem 1.5rem;
          background: rgba(255, 200, 100, 0.15);
          border: 1px solid rgba(255, 200, 100, 0.3);
          color: rgba(255, 220, 150, 0.95);
          font-size: 0.85rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          letter-spacing: 0.03em;
        }

        .add-credits-button:hover {
          background: rgba(255, 200, 100, 0.25);
          border-color: rgba(255, 200, 100, 0.5);
          transform: translateY(-1px);
        }

        .output-area {
          border: 1px solid rgba(255, 255, 255, 0.15);
          padding: 2rem;
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .output-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .output-header h3 {
          margin: 0;
          font-size: 0.85rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .streaming-indicator {
          color: rgba(255, 255, 255, 0.8);
          font-size: 1.2rem;
          animation: pulse 1s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }

        .output-content {
          white-space: pre-wrap;
          word-wrap: break-word;
          line-height: 1.8;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.95rem;
          color: #ffffff;
        }

        .cursor {
          color: rgba(255, 255, 255, 0.8);
          animation: blink 0.8s infinite;
          margin-left: 2px;
          font-weight: normal;
        }

        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}

export default TextGenerator
