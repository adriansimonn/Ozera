/**
 * Text generation component with model selection and streaming support.
 * Supports both local and Modal cloud inference with cold start handling.
 */

import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { useStreamingGeneration } from '../../hooks/useGeneration'
import { useModels } from '../../hooks/useModels'
import { apiClient } from '../../api/client'
import { useAuthStore } from '../../stores/authStore'
import { Dropdown, type DropdownGroup } from '../common/Dropdown'

export interface PlainGenerationResult {
  text: string
  prompt: string
  model: string
  maxTokens: number
  temperature: number
  topK: number
}

interface TextGeneratorProps {
  defaultModel?: string
  defaultPrompt?: string
  onGenerate?: (text: string) => void
  onActivationGenerated?: (activationId: string) => void
  onGeneratingChange?: (isGenerating: boolean) => void
  onModelChange?: (model: string) => void
  onPlainTextGenerated?: (result: PlainGenerationResult) => void
  onStreamingText?: (text: string, streaming: boolean) => void
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
  onPlainTextGenerated,
  onStreamingText,
  externalModel,
  onShowPurchaseCredits,
}) => {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [internalModel, setInternalModel] = useState<string>(defaultModel)

  // Use external model if provided, otherwise use internal state
  const model = externalModel ?? internalModel

  const handleModelChange = (newModel: string) => {
    setInternalModel(newModel)
    onModelChange?.(newModel)
  }
  const [maxTokens, setMaxTokens] = useState(100)
  const [temperature, setTemperature] = useState(0.7)
  const [topK, setTopK] = useState(40)
  const [capturingActivations, setCapturingActivations] = useState(false)
  const [isWarmingUp, setIsWarmingUp] = useState(false)
  const warmupTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Generation mode dropdown state
  type GenerationMode = 'visualize' | 'tokens'
  const [generationMode, setGenerationMode] = useState<GenerationMode>('visualize')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { models, modelNames, modelFamilies, loading: modelsLoading, error: modelsError } = useModels()
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

  // Forward streaming text to parent for token-by-token display
  useEffect(() => {
    if (onStreamingText && generationMode === 'tokens' && (loading || streaming || text)) {
      onStreamingText(text, loading || streaming)
    }
  }, [text, loading, streaming, generationMode, onStreamingText])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
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
      const finalText = await generate({
        prompt: prompt.trim(),
        model,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
      })

      if (onGenerate) {
        onGenerate(finalText)
      }

      if (onPlainTextGenerated && finalText) {
        onPlainTextGenerated({
          text: finalText,
          prompt: prompt.trim(),
          model,
          maxTokens,
          temperature,
          topK,
        })
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

      // Also store the generated text in outputs
      if (onPlainTextGenerated && result.text) {
        onPlainTextGenerated({
          text: result.text,
          prompt: prompt.trim(),
          model,
          maxTokens,
          temperature,
          topK,
        })
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

  const handleGenerateClick = () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    if (generationMode === 'visualize') {
      handleGenerateWithActivations()
    } else {
      handleGenerate()
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
          <Dropdown
            id="model-select"
            value={model}
            onChange={handleModelChange}
            disabled={loading || streaming || modelsLoading || models.length === 0}
            groups={(() => {
              const groups: DropdownGroup[] = []
              const base = models.filter(m => m === 'nano' || m === 'mini')
              if (base.length > 0) {
                groups.push({ label: 'Ozera Models', options: base.map(m => ({ value: m, label: `ozera-${m}` })) })
              }
              const os = models.filter(m => { const f = modelFamilies[m]; return f && f !== 'ozera' })
              if (os.length > 0) {
                groups.push({ label: 'Open Source', options: os.map(m => ({ value: m, label: m })) })
              }
              const custom = models.filter(m => { const f = modelFamilies[m]; return (f === 'ozera' || !f) && m !== 'nano' && m !== 'mini' })
              if (custom.length > 0) {
                groups.push({ label: 'Custom Models', options: custom.map(m => ({ value: m, label: modelNames[m] || m })) })
              }
              if (groups.length === 0) {
                groups.push({ label: 'Models', options: [{ value: '', label: 'No models available' }] })
              }
              return groups
            })()}
          />
        </div>

        <div className="control-group">
          <label htmlFor="max-tokens">
            Tokens: <span className="control-value-display">{maxTokens}</span>
          </label>
          <div className="control-row">
            <input
              id="max-tokens"
              type="range"
              min="1"
              max="300"
              step="1"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              disabled={loading || streaming}
            />
            <input
              type="number"
              className="control-number-input"
              min={1}
              max={300}
              value={maxTokens}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                if (!isNaN(v)) setMaxTokens(Math.max(1, Math.min(300, v)))
              }}
              disabled={loading || streaming}
            />
          </div>
        </div>
      </div>

      <div className="controls">
        <div className="control-group">
          <label htmlFor="temperature">
            Temperature: <span className="control-value-display">{temperature.toFixed(2)}</span>
          </label>
          <div className="control-row">
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
            <input
              type="number"
              className="control-number-input"
              min={0}
              max={2}
              step={0.01}
              value={temperature}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (!isNaN(v)) setTemperature(Math.max(0, Math.min(2, v)))
              }}
              disabled={loading || streaming}
            />
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="top-k">
            Top-K: <span className="control-value-display">{topK}</span>
          </label>
          <div className="control-row">
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
            <input
              type="number"
              className="control-number-input"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                if (!isNaN(v)) setTopK(Math.max(1, Math.min(100, v)))
              }}
              disabled={loading || streaming}
            />
          </div>
        </div>
      </div>

      <div className="prompt-area">
        <label htmlFor="prompt">Prompt:</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          rows={6}
          disabled={loading || streaming}
        />
      </div>

      <div className="actions">
        <div className="generate-dropdown" ref={dropdownRef}>
          <button
            onClick={handleGenerateClick}
            disabled={loading || streaming || capturingActivations || !prompt.trim()}
            className="btn-generate-main"
          >
            {loading && isWarmingUp ? 'Warming up model...' :
             loading ? 'Loading...' :
             streaming ? 'Generating...' :
             capturingActivations && isWarmingUp ? 'Warming up...' :
             capturingActivations ? 'Capturing...' :
             generationMode === 'visualize' ? 'Generate + Visualize' : 'Generate'}
          </button>
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            disabled={loading || streaming || capturingActivations}
            className="btn-generate-toggle"
          >
            <ChevronDown size={16} className={dropdownOpen ? 'chevron-up' : ''} />
          </button>
          {dropdownOpen && (
            <div className="generate-dropdown-menu">
              <button
                onClick={() => {
                  setGenerationMode('visualize')
                  setDropdownOpen(false)
                }}
                className={`dropdown-item ${generationMode === 'visualize' ? 'active' : ''}`}
              >
                Generate + Visualize
                <span className="dropdown-item-desc">Generate tokens and capture activations for visualization</span>
              </button>
              <button
                onClick={() => {
                  setGenerationMode('tokens')
                  setDropdownOpen(false)
                }}
                className={`dropdown-item ${generationMode === 'tokens' ? 'active' : ''}`}
              >
                Generate
                <span className="dropdown-item-desc">Generate tokens without visualization</span>
              </button>
            </div>
          )}
        </div>
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

      {text && !(onPlainTextGenerated && generationMode === 'tokens') && (
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

        .control-row {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .control-row input[type="range"] {
          flex: 1;
        }

        .control-number-input {
          display: none;
          width: 64px;
          padding: 0.4rem 0.5rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: #ffffff;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8rem;
          text-align: center;
          outline: none;
          transition: border-color 0.2s;
          -moz-appearance: textfield;
        }

        [data-interface="default"] .control-number-input {
          display: block;
        }

        .control-number-input::-webkit-inner-spin-button,
        .control-number-input::-webkit-outer-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }

        .control-number-input:focus {
          border-color: rgba(255, 255, 255, 0.3);
        }

        .control-number-input:disabled {
          opacity: 0.3;
        }

        .control-value-display {
          font-family: 'JetBrains Mono', monospace;
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

        .generate-dropdown {
          position: relative;
          display: flex;
        }

        .btn-generate-main {
          padding: 1rem 1.5rem;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-right: none;
          color: #ffffff;
          font-size: 0.9rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .btn-generate-main:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.15);
        }

        .btn-generate-main:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .btn-generate-toggle {
          padding: 1rem 0.75rem;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-left: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.7);
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .btn-generate-toggle:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.15);
          color: #ffffff;
        }

        .btn-generate-toggle:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .btn-generate-toggle .chevron-up {
          transform: rotate(180deg);
        }

        .btn-generate-toggle svg {
          transition: transform 0.2s;
        }

        .generate-dropdown-menu {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          margin-top: 4px;
          background: rgba(20, 20, 20, 0.98);
          border: 1px solid rgba(255, 255, 255, 0.15);
          z-index: 100;
          min-width: 240px;
        }

        .dropdown-item {
          width: 100%;
          padding: 0.875rem 1rem;
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.8);
          font-size: 0.85rem;
          font-weight: 500;
          text-align: left;
          cursor: pointer;
          transition: all 0.15s;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .dropdown-item:hover {
          background: rgba(255, 255, 255, 0.08);
        }

        .dropdown-item.active {
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
        }

        .dropdown-item-desc {
          font-size: 0.75rem;
          font-weight: 400;
          color: rgba(255, 255, 255, 0.4);
          text-transform: none;
          letter-spacing: normal;
        }

        .dropdown-item + .dropdown-item {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .btn-secondary {
          padding: 1rem 2rem;
          background: rgba(255, 255, 255, 0.02);
          color: rgba(255, 255, 255, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.1);
          font-size: 0.9rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .btn-secondary:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.18);
          color: #ffffff;
        }

        .btn-secondary:disabled {
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
          min-width: 16px;
          min-height: 16px;
          flex-shrink: 0;
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

        /* Light mode */
        [data-bg="light"] .generator-header {
          border-bottom-color: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .generator-header h2 {
          color: #1d1d1f;
        }

        [data-bg="light"] .status {
          color: rgba(0, 0, 0, 0.55);
        }

        [data-bg="light"] .control-group label {
          color: rgba(0, 0, 0, 0.55);
        }

        [data-bg="light"] .control-group input[type="range"] {
          background: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .control-group input[type="range"]::-webkit-slider-thumb {
          background: #1d1d1f;
          box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.15),
            0 0 0 2px rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .control-group input[type="range"]::-webkit-slider-thumb:hover {
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.2),
            0 0 0 3px rgba(0, 0, 0, 0.12),
            0 0 12px rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .prompt-area label {
          color: rgba(0, 0, 0, 0.55);
        }

        [data-bg="light"] .prompt-area textarea {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.12);
          color: #1d1d1f;
        }

        [data-bg="light"] .prompt-area textarea:hover {
          background: rgba(0, 0, 0, 0.05);
          border-color: rgba(0, 0, 0, 0.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .prompt-area textarea:focus {
          background: rgba(0, 0, 0, 0.05);
          border-color: rgba(0, 0, 0, 0.2);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(0, 0, 0, 0.1),
            0 0 0 2px rgba(0, 0, 0, 0.04);
        }

        [data-bg="light"] .prompt-area textarea::placeholder {
          color: rgba(0, 0, 0, 0.3);
        }

        [data-bg="light"] .btn-generate-main {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.12);
          color: #1d1d1f;
        }

        [data-bg="light"] .btn-generate-main:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .btn-generate-toggle {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.12);
          border-left-color: rgba(0, 0, 0, 0.1);
          color: rgba(0, 0, 0, 0.6);
        }

        [data-bg="light"] .btn-generate-toggle:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.1);
          color: #1d1d1f;
        }

        [data-bg="light"] .generate-dropdown-menu {
          background: rgba(255, 255, 255, 0.98);
          border-color: rgba(0, 0, 0, 0.12);
        }

        [data-bg="light"] .dropdown-item {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .dropdown-item:hover {
          background: rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .dropdown-item.active {
          background: rgba(0, 0, 0, 0.06);
          color: #1d1d1f;
        }

        [data-bg="light"] .dropdown-item-desc {
          color: rgba(0, 0, 0, 0.4);
        }

        [data-bg="light"] .dropdown-item + .dropdown-item {
          border-top-color: rgba(0, 0, 0, 0.06);
        }

        [data-bg="light"] .btn-secondary {
          background: rgba(0, 0, 0, 0.03);
          color: rgba(0, 0, 0, 0.6);
          border-color: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .btn-secondary:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.15);
          color: #1d1d1f;
        }

        [data-bg="light"] .warmup-message {
          background: rgba(0, 0, 0, 0.04);
          border-color: rgba(0, 0, 0, 0.12);
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .warmup-spinner {
          border-color: rgba(0, 0, 0, 0.15);
          border-top-color: #1d1d1f;
        }

        [data-bg="light"] .error-message {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.1);
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .insufficient-credits-content p {
          color: rgba(0, 0, 0, 0.6);
        }

        [data-bg="light"] .output-area {
          border-color: rgba(0, 0, 0, 0.12);
          background: rgba(0, 0, 0, 0.03);
          box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .output-header {
          border-bottom-color: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .output-header h3 {
          color: rgba(0, 0, 0, 0.55);
        }

        [data-bg="light"] .streaming-indicator {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .output-content {
          color: #1d1d1f;
        }

        [data-bg="light"] .cursor {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .control-number-input {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.12);
          color: #1d1d1f;
        }

        [data-bg="light"] .control-number-input:focus {
          border-color: rgba(0, 0, 0, 0.25);
        }

        /* Default interface: stack reset below generate */
        [data-interface="default"] .actions {
          flex-direction: column;
        }

        [data-interface="default"] .generate-dropdown {
          width: 100%;
        }

        [data-interface="default"] .btn-generate-main {
          flex: 1;
          padding: 0.625rem 1.5rem;
        }

        [data-interface="default"] .btn-secondary {
          width: 100%;
          padding: 0.625rem 2rem;
        }

        [data-interface="default"] .control-row {
          flex-direction: row-reverse;
        }

        [data-interface="default"] .control-value-display {
          display: none;
        }
      `}</style>
    </div>
  )
}

export default TextGenerator
