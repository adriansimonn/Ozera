/**
 * Model selector with detailed model information.
 */

import React from 'react'
import { useModels, useModelInfo } from '../../hooks/useModels'

interface ModelSelectorProps {
  selectedModel: string | null
  onModelSelect: (model: string) => void
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelSelect,
}) => {
  const { models, loading: modelsLoading, error: modelsError } = useModels()
  const { info, loading: infoLoading } = useModelInfo(selectedModel)

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`
    return num.toString()
  }

  return (
    <div className="model-selector">
      <div className="selector-header">
        <h3>Available Models</h3>
        {modelsLoading && <span className="loading">Loading...</span>}
      </div>

      {modelsError && (
        <div className="error-box">
          Failed to load models: {modelsError}
        </div>
      )}

      <div className="models-grid">
        {models.length === 0 && !modelsLoading && !modelsError && (
          <div className="no-models">
            No models available. Make sure the backend is running.
          </div>
        )}
        {models.map((model) => (
          <button
            key={model}
            className={`model-card ${selectedModel === model ? 'selected' : ''}`}
            onClick={() => onModelSelect(model)}
            disabled={modelsLoading}
          >
            <div className="model-name">{model.startsWith('ozera-') ? model : `ozera-${model}`}</div>
            <div className="model-badge">{model.toUpperCase()}</div>
          </button>
        ))}
      </div>

      {selectedModel && (
        <div className="model-details">
          <h4>Model Details</h4>
          {infoLoading ? (
            <div className="loading-details">Loading model info...</div>
          ) : info ? (
            <div className="details-grid">
              <div className="detail-item">
                <span className="detail-label">Parameters:</span>
                <span className="detail-value">{formatNumber(info.parameters)}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Layers:</span>
                <span className="detail-value">{info.layers}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Attention Heads:</span>
                <span className="detail-value">{info.heads}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Hidden Dimension:</span>
                <span className="detail-value">{info.hidden_dim}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Vocab Size:</span>
                <span className="detail-value">{info.vocab_size.toLocaleString()}</span>
              </div>
            </div>
          ) : (
            <div className="no-info">No details available</div>
          )}
        </div>
      )}

      <style>{`
        .model-selector {
          padding: 2rem;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        }

        .selector-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding-bottom: 1.5rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .selector-header h3 {
          margin: 0;
          font-size: 0.85rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .loading {
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.85rem;
          font-weight: 400;
        }

        .error-box {
          padding: 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.8);
          margin-bottom: 1.5rem;
          font-size: 0.9rem;
        }

        .no-models {
          padding: 2rem 1rem;
          text-align: center;
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.9rem;
        }

        .models-grid {
          display: grid;
          gap: 1rem;
          margin-bottom: 2rem;
        }

        .model-card {
          padding: 1.5rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          text-align: center;
          position: relative;
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .model-card:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
          transform: translateY(-2px);
        }

        .model-card.selected {
          border-color: rgba(255, 255, 255, 0.35);
          background: rgba(255, 255, 255, 0.12);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 20px rgba(255, 255, 255, 0.1);
        }

        .model-card:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .model-name {
          font-weight: 500;
          margin-bottom: 0.75rem;
          color: #ffffff;
          font-size: 1rem;
          letter-spacing: -0.01em;
        }

        .model-badge {
          display: inline-block;
          padding: 0.4rem 0.9rem;
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
          font-size: 0.7rem;
          font-weight: 500;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .model-details {
          padding: 1.5rem;
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.12);
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        .model-details h4 {
          margin: 0 0 1.5rem 0;
          font-size: 0.85rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .loading-details,
        .no-info {
          color: rgba(255, 255, 255, 0.4);
          text-align: center;
          padding: 1rem;
          font-size: 0.9rem;
        }

        .details-grid {
          display: grid;
          gap: 0.75rem;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
          padding: 0.875rem;
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .detail-item:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: rgba(255, 255, 255, 0.15);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
          transform: translateX(4px);
        }

        .detail-label {
          font-weight: 400;
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.85rem;
        }

        .detail-value {
          font-weight: 500;
          color: #ffffff;
          font-size: 0.9rem;
          font-family: 'JetBrains Mono', monospace;
        }
      `}</style>
    </div>
  )
}

export default ModelSelector
