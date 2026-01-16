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
            min="1"
            max="500"
            step="1"
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
          max-width: 100%;
          padding: 2rem;
        }

        .generator-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .generator-header h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 600;
          color: #fff;
          letter-spacing: 0.5px;
        }

        .status {
          color: #00f5ff;
          font-size: 0.85rem;
          font-weight: 500;
        }

        .controls {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .control-group label {
          font-weight: 500;
          font-size: 0.85rem;
          color: #aaa;
          letter-spacing: 0.3px;
        }

        .control-group select {
          padding: 0.75rem;
          background: rgba(30, 30, 30, 0.6);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          color: #fff;
          font-size: 0.95rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .control-group select:hover {
          border-color: rgba(0, 245, 255, 0.3);
        }

        .control-group select:focus {
          outline: none;
          border-color: #00f5ff;
          box-shadow: 0 0 0 2px rgba(0, 245, 255, 0.1);
        }

        .control-group input[type="range"] {
          width: 100%;
          height: 6px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          outline: none;
          -webkit-appearance: none;
        }

        .control-group input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          background: linear-gradient(135deg, #00f5ff 0%, #0088ff 100%);
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
          transition: all 0.2s;
        }

        .control-group input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 15px rgba(0, 245, 255, 0.8);
        }

        .prompt-area {
          margin-bottom: 1.5rem;
        }

        .prompt-area label {
          display: block;
          margin-bottom: 0.75rem;
          font-weight: 500;
          color: #aaa;
          font-size: 0.85rem;
          letter-spacing: 0.3px;
        }

        .prompt-area textarea {
          width: 100%;
          padding: 1rem;
          background: rgba(30, 30, 30, 0.6);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #fff;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.95rem;
          resize: vertical;
          transition: all 0.2s;
        }

        .prompt-area textarea:hover {
          border-color: rgba(0, 245, 255, 0.3);
        }

        .prompt-area textarea:focus {
          outline: none;
          border-color: #00f5ff;
          box-shadow: 0 0 0 2px rgba(0, 245, 255, 0.1);
        }

        .prompt-area textarea::placeholder {
          color: #555;
        }

        .actions {
          display: flex;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .btn-primary,
        .btn-secondary {
          padding: 1rem 2rem;
          border: none;
          border-radius: 10px;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          letter-spacing: 0.5px;
          position: relative;
          overflow: hidden;
        }

        .btn-primary {
          background: linear-gradient(135deg, #00f5ff 0%, #0088ff 100%);
          color: #000;
          box-shadow: 0 4px 15px rgba(0, 245, 255, 0.4);
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 25px rgba(0, 245, 255, 0.6);
        }

        .btn-primary:active:not(:disabled) {
          transform: translateY(0);
        }

        .btn-secondary {
          background: rgba(60, 60, 60, 0.6);
          color: #fff;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover:not(:disabled) {
          background: rgba(80, 80, 80, 0.8);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-primary:disabled,
        .btn-secondary:disabled {
          opacity: 0.4;
          cursor: not-allowed;
          transform: none;
        }

        .error-message {
          padding: 1rem 1.25rem;
          background: rgba(220, 38, 38, 0.1);
          border: 1px solid rgba(220, 38, 38, 0.3);
          border-radius: 10px;
          color: #ff6b6b;
          margin-bottom: 1.5rem;
          font-size: 0.9rem;
        }

        .output-area {
          border: 1px solid rgba(0, 245, 255, 0.2);
          border-radius: 12px;
          padding: 1.5rem;
          background: rgba(15, 15, 15, 0.6);
          box-shadow: 0 0 30px rgba(0, 245, 255, 0.1);
        }

        .output-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.25rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .output-header h3 {
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
          color: #fff;
          letter-spacing: 0.5px;
        }

        .streaming-indicator {
          color: #00f5ff;
          font-size: 1.5rem;
          animation: pulse 1s infinite;
          filter: drop-shadow(0 0 8px rgba(0, 245, 255, 0.6));
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }

        .output-content {
          white-space: pre-wrap;
          word-wrap: break-word;
          line-height: 1.8;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.95rem;
          color: #e0e0e0;
        }

        .cursor {
          color: #00f5ff;
          animation: blink 0.8s infinite;
          margin-left: 2px;
          font-weight: bold;
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
