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
        {models.map((model) => (
          <button
            key={model}
            className={`model-card ${selectedModel === model ? 'selected' : ''}`}
            onClick={() => onModelSelect(model)}
            disabled={modelsLoading}
          >
            <div className="model-name">ozera-{model}</div>
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
          padding: 1.5rem;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .selector-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .selector-header h3 {
          margin: 0;
          font-size: 1.2rem;
        }

        .loading {
          color: #666;
          font-size: 0.9rem;
        }

        .error-box {
          padding: 1rem;
          background: #f8d7da;
          border: 1px solid #f5c6cb;
          border-radius: 4px;
          color: #721c24;
          margin-bottom: 1rem;
        }

        .models-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .model-card {
          padding: 1rem;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          background: white;
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }

        .model-card:hover:not(:disabled) {
          border-color: #007bff;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .model-card.selected {
          border-color: #007bff;
          background: #e7f3ff;
        }

        .model-card:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .model-name {
          font-weight: 600;
          margin-bottom: 0.5rem;
        }

        .model-badge {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          background: #007bff;
          color: white;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .model-details {
          padding: 1rem;
          background: #f8f9fa;
          border-radius: 8px;
        }

        .model-details h4 {
          margin: 0 0 1rem 0;
          font-size: 1rem;
        }

        .loading-details,
        .no-info {
          color: #666;
          text-align: center;
          padding: 1rem;
        }

        .details-grid {
          display: grid;
          gap: 0.75rem;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem;
          background: white;
          border-radius: 4px;
        }

        .detail-label {
          font-weight: 500;
          color: #666;
        }

        .detail-value {
          font-weight: 600;
          color: #333;
        }
      `}</style>
    </div>
  )
}

export default ModelSelector
