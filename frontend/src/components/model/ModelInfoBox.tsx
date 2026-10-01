/**
 * Model information box component with model selector and detailed specs.
 * Supports base models (nano/mini), open source models, and custom trained models.
 */

import React from 'react'
import { Info, Cpu, Layers, Grid3X3, Hash, Database } from 'lucide-react'
import { useModels, useModelInfo } from '../../hooks/useModels'
import type { ModelFamily } from '../../types/model'
import { STATIC_OPEN_SOURCE_MODELS } from '../../data/defaultModels'
import { Dropdown, type DropdownGroup } from '../common/Dropdown'

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

// Developer names for open source model families
const FAMILY_DEVELOPERS: Record<Exclude<ModelFamily, 'ozera'>, string> = {
  smollm: 'Hugging Face',
  qwen: 'Alibaba',
  gemma: 'Google DeepMind',
}

const formatNumber = (num: number | null): string => {
  if (num === null) {
    return '—'
  } else if (num >= 1_000_000_000) {
    return `${(num / 1_000_000_000).toFixed(1)}B`
  } else if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`
  } else if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}K`
  }
  return num.toLocaleString()
}

interface ModelDetailsProps {
  modelId: string
}

/**
 * A model's description and specs: the body of the model information box, and of the
 * model cards shown for generated outputs.
 */
export const ModelDetails: React.FC<ModelDetailsProps> = ({ modelId }) => {
  const { info: dynamicInfo, loading: infoLoading } = useModelInfo(modelId)

  // Custom models are Ozera models too
  const family: ModelFamily = STATIC_OPEN_SOURCE_MODELS.find(m => m.id === modelId)?.family ?? 'ozera'
  const isBaseModel = modelId === 'nano' || modelId === 'mini'
  const isOpenSourceModel = family !== 'ozera'
  const isCustomModel = !isBaseModel && !isOpenSourceModel

  const getDescription = (): string => {
    if (isBaseModel) {
      return BASE_MODEL_INFO[modelId as 'nano' | 'mini'].description
    }
    if (isOpenSourceModel) {
      const developer = FAMILY_DEVELOPERS[family as Exclude<ModelFamily, 'ozera'>]
      return `Small model by ${developer}.`
    }
    return 'Your custom trained model.'
  }

  return (
    <div className="model-info-content">
      {infoLoading ? (
        <p className="model-description loading">Loading model information...</p>
      ) : dynamicInfo ? (
        <>
          <p className="model-description">
            {getDescription()}
            {isOpenSourceModel && (
              <span className="os-badge">Open Source</span>
            )}
            {isCustomModel && (
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
                <span className="info-value">{dynamicInfo.layers ?? '—'}</span>
              </div>
            </div>

            <div className="info-card">
              <div className="info-icon-wrapper">
                <Grid3X3 className="info-icon" />
              </div>
              <div className="info-details">
                <span className="info-label">Attention Heads</span>
                <span className="info-value">{dynamicInfo.heads ?? '—'}</span>
              </div>
            </div>

            <div className="info-card">
              <div className="info-icon-wrapper">
                <Hash className="info-icon" />
              </div>
              <div className="info-details">
                <span className="info-label">Hidden Dimension</span>
                <span className="info-value">{dynamicInfo.hidden_dim ?? '—'}</span>
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
          Unable to load model information for: {modelId}
        </p>
      )}

      <style>{`
        .model-info-content {
          min-height: 120px;
        }

        .model-description {
          margin: 0 0 1.25rem 0;
          color: #ffffff;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        .model-description.loading {
          color: #ffffff;
          font-style: italic;
        }

        .custom-badge,
        .os-badge {
          display: inline-block;
          margin-left: 0.75rem;
          padding: 0.25rem 0.6rem;
          font-size: 0.7rem;
          font-weight: 500;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          vertical-align: middle;
        }

        .custom-badge {
          background: rgba(147, 112, 219, 0.2);
          color: rgba(200, 180, 255, 0.9);
        }

        .os-badge {
          background: rgba(34, 197, 94, 0.15);
          color: rgba(134, 239, 172, 0.95);
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
          color: #ffffff;
        }

        .info-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          min-width: 0;
        }

        .info-label {
          font-size: 0.7rem;
          color: #ffffff;
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

        /* Light mode */
        [data-bg="light"] .model-description {
          color: #1d1d1f;
        }

        [data-bg="light"] .model-description.loading {
          color: #1d1d1f;
        }

        [data-bg="light"] .info-card {
          background: rgba(0, 0, 0, 0.03);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .info-card:hover {
          background: rgba(0, 0, 0, 0.04);
          border-color: rgba(0, 0, 0, 0.12);
        }

        [data-bg="light"] .info-icon-wrapper {
          background: rgba(0, 0, 0, 0.04);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .info-icon {
          color: #1d1d1f;
        }

        [data-bg="light"] .info-label {
          color: #1d1d1f;
        }

        [data-bg="light"] .info-value {
          color: #1d1d1f;
        }
      `}</style>
    </div>
  )
}

interface ModelInfoBoxProps {
  selectedModel: string
  onModelChange: (model: string) => void
}

export const ModelInfoBox: React.FC<ModelInfoBoxProps> = ({
  selectedModel,
  onModelChange,
}) => {
  const { models, modelFamilies, modelNames } = useModels()

  // Helper to get family for a model
  const getFamily = (modelId: string): ModelFamily => {
    return modelFamilies[modelId] || 'ozera'
  }

  // Separate models into categories
  const baseModels = models.filter(m => m === 'nano' || m === 'mini')
  const osModels = models.filter(m => {
    const family = getFamily(m)
    return family !== 'ozera'
  })
  const customModels = models.filter(m => {
    const family = getFamily(m)
    return family === 'ozera' && m !== 'nano' && m !== 'mini'
  })

  return (
    <div className="model-info-box">
      <div className="model-info-header">
        <Info className="header-icon" />
        <h3>Model Information</h3>
      </div>

      <div className="model-selector">
        <label htmlFor="model-info-select">Select Model:</label>
        <Dropdown
          id="model-info-select"
          value={selectedModel}
          onChange={onModelChange}
          groups={(() => {
            const groups: DropdownGroup[] = []
            if (baseModels.length > 0) {
              groups.push({ label: 'Ozera Models', options: baseModels.map(m => ({ value: m, label: `ozera-${m}` })) })
            }
            if (osModels.length > 0) {
              groups.push({ label: 'Open Source', options: osModels.map(m => ({ value: m, label: modelNames[m] || m })) })
            }
            if (customModels.length > 0) {
              groups.push({ label: 'Custom Models', options: customModels.map(m => ({ value: m, label: `${modelNames[m] || m} (custom)` })) })
            }
            return groups
          })()}
        />
      </div>

      <ModelDetails modelId={selectedModel} />

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
          color: #ffffff;
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
          color: #ffffff;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        /* Light mode */
        [data-bg="light"] .model-info-header {
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .header-icon {
          color: #1d1d1f;
        }

        [data-bg="light"] .model-info-header h3 {
          color: #1d1d1f;
        }

        [data-bg="light"] .model-selector label {
          color: #1d1d1f;
        }
      `}</style>
    </div>
  )
}

export default ModelInfoBox
