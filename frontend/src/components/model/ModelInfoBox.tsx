/**
 * Model information box component with model selector and detailed specs.
 * Model data is stored statically since it doesn't change.
 */

import React from 'react'
import { Info, Cpu, Layers, Grid3X3, Hash, Database } from 'lucide-react'

// Static model information - stored on frontend since it doesn't change
const MODEL_INFO: Record<'nano' | 'mini', {
  name: string
  parameters: number
  layers: number
  heads: number
  hidden_dim: number
  vocab_size: number
  description: string
}> = {
  nano: {
    name: 'ozera-nano',
    parameters: 4_336_128,
    layers: 4,
    heads: 4,
    hidden_dim: 256,
    vocab_size: 50257,
    description: 'Fast experimentation and educational demonstrations. Optimized for rapid iteration and full interpretability.',
  },
  mini: {
    name: 'ozera-mini',
    parameters: 51_459_584,
    layers: 8,
    heads: 8,
    hidden_dim: 512,
    vocab_size: 50257,
    description: 'More coherent outputs while remaining computationally accessible. Better for observing emergent behaviors.',
  },
}

const AVAILABLE_MODELS: ('nano' | 'mini')[] = ['nano', 'mini']

interface ModelInfoBoxProps {
  selectedModel: 'nano' | 'mini'
  onModelChange: (model: 'nano' | 'mini') => void
}

export const ModelInfoBox: React.FC<ModelInfoBoxProps> = ({
  selectedModel,
  onModelChange,
}) => {
  const info = MODEL_INFO[selectedModel]

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
          onChange={(e) => onModelChange(e.target.value as 'nano' | 'mini')}
        >
          {AVAILABLE_MODELS.map((m) => (
            <option key={m} value={m}>
              ozera-{m}
            </option>
          ))}
        </select>
      </div>

      <div className="model-info-content">
        <p className="model-description">{info.description}</p>

        <div className="info-grid">
          <div className="info-card">
            <div className="info-icon-wrapper">
              <Cpu className="info-icon" />
            </div>
            <div className="info-details">
              <span className="info-label">Parameters</span>
              <span className="info-value">{formatNumber(info.parameters)}</span>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon-wrapper">
              <Layers className="info-icon" />
            </div>
            <div className="info-details">
              <span className="info-label">Layers</span>
              <span className="info-value">{info.layers}</span>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon-wrapper">
              <Grid3X3 className="info-icon" />
            </div>
            <div className="info-details">
              <span className="info-label">Attention Heads</span>
              <span className="info-value">{info.heads}</span>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon-wrapper">
              <Hash className="info-icon" />
            </div>
            <div className="info-details">
              <span className="info-label">Hidden Dimension</span>
              <span className="info-value">{info.hidden_dim}</span>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon-wrapper">
              <Database className="info-icon" />
            </div>
            <div className="info-details">
              <span className="info-label">Vocabulary Size</span>
              <span className="info-value">{formatNumber(info.vocab_size)}</span>
            </div>
          </div>
        </div>
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
