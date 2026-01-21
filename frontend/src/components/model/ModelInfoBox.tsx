/**
 * Model information box component with model selector and detailed specs.
 * Supports both base models (nano/mini) and custom trained models.
 */

import React from 'react'
import { Info, Cpu, Layers, Grid3X3, Hash, Database } from 'lucide-react'
import { useModels, useModelInfo } from '../../hooks/useModels'

// Static model information for base models
const BASE_MODEL_INFO: Record<'nano' | 'mini', {
  name: string
  description: string
}> = {
  nano: {
    name: 'ozera-nano',
    description: 'Fast experimentation and educational demonstrations. Optimized for rapid iteration and full interpretability.',
  },
  mini: {
    name: 'ozera-mini',
    description: 'More coherent outputs while remaining computationally accessible. Better for observing emergent behaviors.',
  },
}

interface ModelInfoBoxProps {
  selectedModel: string
  onModelChange: (model: string) => void
}

export const ModelInfoBox: React.FC<ModelInfoBoxProps> = ({
  selectedModel,
  onModelChange,
}) => {
  const { models } = useModels()
  const { info: dynamicInfo, loading: infoLoading } = useModelInfo(selectedModel)

  // Determine if this is a base model or custom model
  const isBaseModel = selectedModel === 'nano' || selectedModel === 'mini'
  const description = isBaseModel ? BASE_MODEL_INFO[selectedModel].description : 'Your custom trained model.'

  // Separate models into base and custom
  const baseModels = models.filter(m => m === 'nano' || m === 'mini')
  const customModels = models.filter(m => m !== 'nano' && m !== 'mini')

  const formatNumber = (num: number): string => {
    if (num >= 1_000_000) {
      return `${(num / 1_000_000).toFixed(1)}M`
    } else if (num >= 1_000) {
      return `${(num / 1_000).toFixed(1)}K`
    }
    return num.toLocaleString()
  }

  return (
    <div className="model-info-box">
      <div className="model-info-header">
        <Info className="header-icon" />
        <h3>Model Information</h3>
      </div>

      <div className="model-selector">
        <label htmlFor="model-info-select">Select Model:</label>
        <select
          id="model-info-select"
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {/* Base models */}
          {baseModels.map((m) => (
            <option key={m} value={m}>
              ozera-{m}
            </option>
          ))}
          {/* Custom models */}
          {customModels.length > 0 && (
            <option disabled>── Custom Models ──</option>
          )}
          {customModels.map((m) => (
            <option key={m} value={m}>
              {m} (custom)
            </option>
          ))}
        </select>
      </div>

      <div className="model-info-content">
        {infoLoading ? (
          <p className="model-description loading">Loading model information...</p>
        ) : dynamicInfo ? (
          <>
            <p className="model-description">
              {description}
              {!isBaseModel && (
                <span className="custom-badge">Custom Model</span>
              )}
            </p>

            <div className="info-grid">
              <div className="info-card">
                <div className="info-icon-wrapper">
                  <Cpu className="info-icon" />
                </div>
                <div className="info-details">
                  <span className="info-label">Parameters</span>
                  <span className="info-value">{formatNumber(dynamicInfo.parameters)}</span>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon-wrapper">
                  <Layers className="info-icon" />
                </div>
                <div className="info-details">
                  <span className="info-label">Layers</span>
                  <span className="info-value">{dynamicInfo.layers}</span>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon-wrapper">
                  <Grid3X3 className="info-icon" />
                </div>
                <div className="info-details">
                  <span className="info-label">Attention Heads</span>
                  <span className="info-value">{dynamicInfo.heads}</span>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon-wrapper">
                  <Hash className="info-icon" />
                </div>
                <div className="info-details">
                  <span className="info-label">Hidden Dimension</span>
                  <span className="info-value">{dynamicInfo.hidden_dim}</span>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon-wrapper">
                  <Database className="info-icon" />
                </div>
                <div className="info-details">
                  <span className="info-label">Vocabulary Size</span>
                  <span className="info-value">{formatNumber(dynamicInfo.vocab_size)}</span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <p className="model-description">
            Unable to load model information for: {selectedModel}
          </p>
        )}
      </div>

      <style>{`
        .model-info-box {
          padding: 1.5rem;
        }

        .model-info-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .header-icon {
          width: 20px;
          height: 20px;
          color: rgba(255, 255, 255, 0.6);
        }

        .model-info-header h3 {
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
          color: #ffffff;
          letter-spacing: -0.01em;
        }

        .model-selector {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .model-selector label {
          font-weight: 500;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .model-selector select {
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

        .model-selector select:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .model-selector select:focus {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.25);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.12),
            0 0 0 2px rgba(255, 255, 255, 0.05);
          outline: none;
        }

        .model-selector select:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .model-info-content {
          min-height: 120px;
        }

        .model-description {
          margin: 0 0 1.25rem 0;
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.9rem;
          line-height: 1.5;
        }

        .model-description.loading {
          color: rgba(255, 255, 255, 0.4);
          font-style: italic;
        }

        .custom-badge {
          display: inline-block;
          margin-left: 0.75rem;
          padding: 0.25rem 0.6rem;
          background: rgba(147, 112, 219, 0.2);
          color: rgba(200, 180, 255, 0.9);
          font-size: 0.7rem;
          font-weight: 500;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          vertical-align: middle;
        }

        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 0.75rem;
        }

        .info-card {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          transition: all 0.2s ease;
        }

        .info-card:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.12);
        }

        .info-icon-wrapper {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 36px;
          height: 36px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          flex-shrink: 0;
        }

        .info-icon {
          width: 16px;
          height: 16px;
          color: rgba(255, 255, 255, 0.7);
        }

        .info-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          min-width: 0;
        }

        .info-label {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 500;
        }

        .info-value {
          font-size: 1rem;
          color: #ffffff;
          font-family: 'JetBrains Mono', monospace;
          font-weight: 600;
        }
      `}</style>
    </div>
  )
}

export default ModelInfoBox
