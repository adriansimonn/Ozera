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
          background: rgba(20, 20, 20, 0.6);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .selector-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .selector-header h3 {
          margin: 0;
          font-size: 1.2rem;
          font-weight: 600;
          color: #fff;
          letter-spacing: 0.5px;
        }

        .loading {
          color: #00f5ff;
          font-size: 0.85rem;
          font-weight: 500;
        }

        .error-box {
          padding: 1rem;
          background: rgba(220, 38, 38, 0.1);
          border: 1px solid rgba(220, 38, 38, 0.3);
          border-radius: 8px;
          color: #ff6b6b;
          margin-bottom: 1rem;
          font-size: 0.9rem;
        }

        .models-grid {
          display: grid;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .model-card {
          padding: 1.25rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          background: rgba(30, 30, 30, 0.4);
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          text-align: center;
          position: relative;
          overflow: hidden;
        }

        .model-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, rgba(0, 245, 255, 0) 0%, rgba(0, 136, 255, 0) 100%);
          opacity: 0;
          transition: opacity 0.3s;
        }

        .model-card:hover:not(:disabled)::before {
          opacity: 0.1;
        }

        .model-card:hover:not(:disabled) {
          border-color: rgba(0, 245, 255, 0.5);
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(0, 245, 255, 0.2);
        }

        .model-card.selected {
          border-color: #00f5ff;
          background: rgba(0, 136, 255, 0.1);
          box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
        }

        .model-card:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .model-name {
          font-weight: 600;
          margin-bottom: 0.75rem;
          color: #fff;
          font-size: 1rem;
          position: relative;
          z-index: 1;
        }

        .model-badge {
          display: inline-block;
          padding: 0.35rem 0.75rem;
          background: linear-gradient(135deg, #00f5ff 0%, #0088ff 100%);
          color: #000;
          border-radius: 6px;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 1px;
          position: relative;
          z-index: 1;
          box-shadow: 0 4px 12px rgba(0, 245, 255, 0.3);
        }

        .model-details {
          padding: 1.25rem;
          background: rgba(15, 15, 15, 0.6);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 12px;
        }

        .model-details h4 {
          margin: 0 0 1rem 0;
          font-size: 1rem;
          font-weight: 600;
          color: #fff;
          letter-spacing: 0.5px;
        }

        .loading-details,
        .no-info {
          color: #666;
          text-align: center;
          padding: 1rem;
          font-size: 0.9rem;
        }

        .details-grid {
          display: grid;
          gap: 0.5rem;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
          padding: 0.75rem;
          background: rgba(30, 30, 30, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          transition: all 0.2s;
        }

        .detail-item:hover {
          background: rgba(40, 40, 40, 0.6);
          border-color: rgba(255, 255, 255, 0.1);
        }

        .detail-label {
          font-weight: 500;
          color: #888;
          font-size: 0.85rem;
        }

        .detail-value {
          font-weight: 600;
          color: #00f5ff;
          font-size: 0.9rem;
          font-family: 'Monaco', 'Courier New', monospace;
        }
      `}</style>
    </div>
  )
}

export default ModelSelector
