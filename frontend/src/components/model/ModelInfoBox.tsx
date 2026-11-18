/**
 * Model information box component with model selector and detailed specs.
 * Supports base models (nano/mini), open source models, and custom trained models.
 */

import React from 'react'
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
  const openModel = STATIC_OPEN_SOURCE_MODELS.find(m => m.id === modelId)
  const family: ModelFamily = openModel?.family ?? 'ozera'
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
            {openModel && (
              <a
                className="oz-tag model-weights-link"
                href={`https://huggingface.co/${openModel.hf_id}`}
                target="_blank"
                rel="noopener noreferrer"
                title={`${openModel.hf_id} on Hugging Face`}
              >
                Open weight
              </a>
            )}
            {isCustomModel && (
              <span className="oz-tag">Custom model</span>
            )}
          </p>

          <div className="info-grid">
            <div className="info-card">
              <div className="info-details">
                <span className="info-label">Parameters</span>
                <span className="info-value">{formatNumber(dynamicInfo.parameters)}</span>
              </div>
            </div>

            <div className="info-card">
              <div className="info-details">
                <span className="info-label">Layers</span>
                <span className="info-value">{dynamicInfo.layers ?? '—'}</span>
              </div>
            </div>

            <div className="info-card">
              <div className="info-details">
                <span className="info-label">Attention Heads</span>
                <span className="info-value">{dynamicInfo.heads ?? '—'}</span>
              </div>
            </div>

            <div className="info-card">
              <div className="info-details">
                <span className="info-label">Hidden Dimension</span>
                <span className="info-value">{dynamicInfo.hidden_dim ?? '—'}</span>
              </div>
            </div>

            <div className="info-card">
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
        .model-info-box .model-selector {
          margin-bottom: 1.1rem;
        }

        .model-info-content .model-description {
          margin: 0 0 1.1rem;
          color: var(--oz-fg-dim);
          font-size: 0.9rem;
          line-height: 1.55;
          text-wrap: pretty;
        }

        .model-info-content .model-description .oz-tag {
          margin-left: 0.5rem;
          vertical-align: 1px;
        }

        .model-info-content .model-weights-link {
          text-decoration: none;
          transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
        }

        .model-info-content .model-weights-link::after {
          content: '↗';
        }

        .model-info-content .model-weights-link:hover,
        .model-info-content .model-weights-link:focus-visible {
          background: var(--oz-accent);
          border-color: var(--oz-accent);
          color: var(--oz-on-accent);
        }

        /* The specs: a ruled sheet, a name and its number to a row */
        .model-info-content .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
          column-gap: 2rem;
        }

        .model-info-content .info-card {
          border-top: 1px solid var(--oz-rule);
        }

        .model-info-content .info-details {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.6rem 0;
        }

        .model-info-content .info-label {
          color: var(--oz-fg-dim);
          font-size: 0.875rem;
        }

        .model-info-content .info-value {
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          font-variant-numeric: tabular-nums;
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
    <section className="model-info-box oz-section">
      <h2 className="oz-title">Model information</h2>

      <div className="model-selector">
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
    </section>
  )
}

export default ModelInfoBox
