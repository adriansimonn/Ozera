/**
 * Text generation component with model selection and streaming support.
 */

import React, { useState } from 'react'
import { useStreamingGeneration } from '../../hooks/useGeneration'
import { useModels } from '../../hooks/useModels'

interface TextGeneratorProps {
  defaultModel?: 'nano' | 'mini'
  defaultPrompt?: string
  onGenerate?: (text: string) => void
}

export const TextGenerator: React.FC<TextGeneratorProps> = ({
  defaultModel = 'nano',
  defaultPrompt = '',
  onGenerate,
}) => {
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [model, setModel] = useState<'nano' | 'mini'>(defaultModel)
  const [maxTokens, setMaxTokens] = useState(200)
  const [temperature, setTemperature] = useState(0.7)
  const [topK, setTopK] = useState(40)

  const { models, loading: modelsLoading, error: modelsError } = useModels()
  const { text, loading, streaming, error, generate, reset } = useStreamingGeneration()

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      return
    }

    reset()

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
    }
  }

  const handleReset = () => {
    reset()
    setPrompt('')
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
            onChange={(e) => setModel(e.target.value as 'nano' | 'mini')}
            disabled={loading || streaming || modelsLoading}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                ozera-{m}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="max-tokens">Max Tokens: {maxTokens}</label>
          <input
            id="max-tokens"
            type="range"
            min="10"
            max="500"
            step="10"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            disabled={loading || streaming}
          />
        </div>

        <div className="control-group">
          <label htmlFor="temperature">Temperature: {temperature.toFixed(2)}</label>
          <input
            id="temperature"
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
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
          disabled={loading || streaming || !prompt.trim()}
          className="btn-primary"
        >
          {loading ? 'Loading...' : streaming ? 'Generating...' : 'Generate'}
        </button>
        <button
          onClick={handleReset}
          disabled={loading || streaming}
          className="btn-secondary"
        >
          Reset
        </button>
      </div>

      {error && (
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
          max-width: 800px;
          margin: 0 auto;
          padding: 2rem;
        }

        .generator-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
        }

        .generator-header h2 {
          margin: 0;
        }

        .status {
          color: #666;
          font-size: 0.9rem;
        }

        .controls {
          display: grid;
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .control-group label {
          font-weight: 500;
          font-size: 0.9rem;
        }

        .control-group select,
        .control-group input[type="range"] {
          width: 100%;
        }

        .prompt-area {
          margin-bottom: 1.5rem;
        }

        .prompt-area label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
        }

        .prompt-area textarea {
          width: 100%;
          padding: 0.75rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: inherit;
          font-size: 1rem;
          resize: vertical;
        }

        .actions {
          display: flex;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .btn-primary,
        .btn-secondary {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: opacity 0.2s;
        }

        .btn-primary {
          background: #007bff;
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          background: #0056b3;
        }

        .btn-secondary {
          background: #6c757d;
          color: white;
        }

        .btn-secondary:hover:not(:disabled) {
          background: #545b62;
        }

        .btn-primary:disabled,
        .btn-secondary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .error-message {
          padding: 1rem;
          background: #f8d7da;
          border: 1px solid #f5c6cb;
          border-radius: 4px;
          color: #721c24;
          margin-bottom: 1.5rem;
        }

        .output-area {
          border: 1px solid #ddd;
          border-radius: 4px;
          padding: 1.5rem;
          background: #f8f9fa;
        }

        .output-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .output-header h3 {
          margin: 0;
          font-size: 1.1rem;
        }

        .streaming-indicator {
          color: #28a745;
          font-size: 1.5rem;
          animation: pulse 1s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }

        .output-content {
          white-space: pre-wrap;
          word-wrap: break-word;
          line-height: 1.6;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .cursor {
          animation: blink 1s infinite;
          margin-left: 2px;
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
