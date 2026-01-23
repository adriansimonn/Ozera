/**
 * Model selector with detailed model information and family grouping.
 */

import React from 'react'
import { useModels, useModelInfo } from '../../hooks/useModels'
import type { ModelFamily, OpenSourceModelInfo } from '../../types/model'

interface ModelSelectorProps {
  selectedModel: string | null
  onModelSelect: (model: string) => void
}

const FAMILY_LABELS: Record<ModelFamily, string> = {
  ozera: 'Ozera Models',
  gemma: 'Gemma',
  qwen: 'Qwen',
  smollm: 'SmolLM',
}

const FAMILY_ORDER: ModelFamily[] = ['ozera', 'smollm', 'qwen', 'gemma']

const FAMILY_COLORS: Record<ModelFamily, { bg: string; border: string; text: string; badge: string }> = {
  ozera: {
    bg: 'rgba(255, 255, 255, 0.03)',
    border: 'rgba(255, 255, 255, 0.1)',
    text: '#ffffff',
    badge: 'rgba(255, 255, 255, 0.1)',
  },
  smollm: {
    bg: 'rgba(59, 130, 246, 0.05)',
    border: 'rgba(59, 130, 246, 0.2)',
    text: 'rgba(147, 197, 253, 0.95)',
    badge: 'rgba(59, 130, 246, 0.15)',
  },
  qwen: {
    bg: 'rgba(168, 85, 247, 0.05)',
    border: 'rgba(168, 85, 247, 0.2)',
    text: 'rgba(216, 180, 254, 0.95)',
    badge: 'rgba(168, 85, 247, 0.15)',
  },
  gemma: {
    bg: 'rgba(34, 197, 94, 0.05)',
    border: 'rgba(34, 197, 94, 0.2)',
    text: 'rgba(134, 239, 172, 0.95)',
    badge: 'rgba(34, 197, 94, 0.15)',
  },
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelSelect,
}) => {
  const {
    models,
    modelNames,
    modelFamilies,
    openSourceModels,
    loading: modelsLoading,
    error: modelsError
  } = useModels()
  const { info, loading: infoLoading } = useModelInfo(selectedModel)

  // Helper to get display name for a model
  const getDisplayName = (modelId: string): string => {
    return modelNames[modelId] || modelId
  }

  // Helper to get family for a model
  const getFamily = (modelId: string): ModelFamily => {
    return modelFamilies[modelId] || 'ozera'
  }

  // Helper to get open source model info
  const getOpenSourceInfo = (modelId: string): OpenSourceModelInfo | undefined => {
    return openSourceModels.find(m => m.id === modelId)
  }

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`
    return num.toString()
  }

  // Group models by family
  const groupedModels: Record<ModelFamily, string[]> = {
    ozera: [],
    smollm: [],
    qwen: [],
    gemma: [],
  }

  models.forEach(modelId => {
    const family = getFamily(modelId)
    if (groupedModels[family]) {
      groupedModels[family].push(modelId)
    }
  })

  // Separate ozera models into base and custom
  const baseOzeraModels = groupedModels.ozera.filter(m => m === 'nano' || m === 'mini')
  const customOzeraModels = groupedModels.ozera.filter(m => m !== 'nano' && m !== 'mini')

  // Get selected model's family for details styling
  const selectedFamily = selectedModel ? getFamily(selectedModel) : 'ozera'
  const selectedOsInfo = selectedModel ? getOpenSourceInfo(selectedModel) : undefined

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

      <div className="models-container">
        {models.length === 0 && !modelsLoading && !modelsError && (
          <div className="no-models">
            No models available. Make sure the backend is running.
          </div>
        )}

        {/* Ozera Models Section */}
        {(baseOzeraModels.length > 0 || customOzeraModels.length > 0) && (
          <div className="family-section">
            <div className="family-header">
              <span className="family-name">{FAMILY_LABELS.ozera}</span>
              <span className="family-badge">Base</span>
            </div>
            <div className="family-models">
              {baseOzeraModels.map((model) => (
                <button
                  key={model}
                  className={`model-card ${selectedModel === model ? 'selected' : ''}`}
                  onClick={() => onModelSelect(model)}
                  disabled={modelsLoading}
                  data-family="ozera"
                >
                  <div className="model-name">ozera-{model}</div>
                  <div className="model-badge">{model.toUpperCase()}</div>
                </button>
              ))}
            </div>

            {customOzeraModels.length > 0 && (
              <>
                <div className="subsection-divider">
                  <span>Your Custom Models</span>
                </div>
                <div className="family-models">
                  {customOzeraModels.map((model) => (
                    <button
                      key={model}
                      className={`model-card custom-model ${selectedModel === model ? 'selected' : ''}`}
                      onClick={() => onModelSelect(model)}
                      disabled={modelsLoading}
                      data-family="ozera"
                    >
                      <div className="model-name">{getDisplayName(model)}</div>
                      <div className="model-badge custom">CUSTOM</div>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Open Source Model Families */}
        {FAMILY_ORDER.filter(f => f !== 'ozera').map(family => {
          const familyModels = groupedModels[family]
          if (familyModels.length === 0) return null

          const colors = FAMILY_COLORS[family]

          return (
            <div key={family} className="family-section" data-family={family}>
              <div className="family-header">
                <span className="family-name">{FAMILY_LABELS[family]}</span>
                <span className="family-badge os-badge" style={{ background: colors.badge, color: colors.text }}>
                  Open Source
                </span>
              </div>
              <div className="family-models">
                {familyModels.map((modelId) => {
                  const osInfo = getOpenSourceInfo(modelId)
                  return (
                    <button
                      key={modelId}
                      className={`model-card os-model ${selectedModel === modelId ? 'selected' : ''}`}
                      onClick={() => onModelSelect(modelId)}
                      disabled={modelsLoading}
                      data-family={family}
                      style={{
                        '--family-bg': colors.bg,
                        '--family-border': colors.border,
                        '--family-text': colors.text,
                        '--family-badge': colors.badge,
                      } as React.CSSProperties}
                    >
                      <div className="model-name">{getDisplayName(modelId)}</div>
                      <div className="model-meta">
                        {osInfo && (
                          <>
                            <span className="model-params">{formatNumber(osInfo.parameters)}</span>
                            <span className="model-gpu">{osInfo.gpu_tier.toUpperCase()}</span>
                          </>
                        )}
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {selectedModel && (
        <div
          className="model-details"
          data-family={selectedFamily}
          style={{
            '--family-border': FAMILY_COLORS[selectedFamily].border,
            '--family-bg': FAMILY_COLORS[selectedFamily].bg,
          } as React.CSSProperties}
        >
          <h4>Model Details</h4>
          {infoLoading ? (
            <div className="loading-details">Loading model info...</div>
          ) : selectedOsInfo ? (
            // Open source model details
            <div className="details-grid">
              <div className="detail-item">
                <span className="detail-label">Parameters:</span>
                <span className="detail-value">{formatNumber(selectedOsInfo.parameters)}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Layers:</span>
                <span className="detail-value">{selectedOsInfo.layers}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Attention Heads:</span>
                <span className="detail-value">{selectedOsInfo.heads}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">KV Heads:</span>
                <span className="detail-value">{selectedOsInfo.kv_heads}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Hidden Dimension:</span>
                <span className="detail-value">{selectedOsInfo.hidden_dim}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Vocab Size:</span>
                <span className="detail-value">{selectedOsInfo.vocab_size.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Max Sequence:</span>
                <span className="detail-value">{selectedOsInfo.max_seq_len.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">GPU Tier:</span>
                <span className="detail-value">{selectedOsInfo.gpu_tier.toUpperCase()}</span>
              </div>
            </div>
          ) : info ? (
            // Ozera model details
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

        .models-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
          margin-bottom: 2rem;
        }

        .family-section {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .family-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .family-name {
          font-size: 0.8rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.5);
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .family-badge {
          padding: 0.25rem 0.6rem;
          background: rgba(255, 255, 255, 0.08);
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.65rem;
          font-weight: 500;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          border-radius: 2px;
        }

        .family-badge.os-badge {
          background: var(--family-badge, rgba(255, 255, 255, 0.08));
        }

        .family-models {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 0.75rem;
        }

        .subsection-divider {
          padding: 0.75rem 0;
          text-align: center;
          color: rgba(255, 255, 255, 0.3);
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          position: relative;
        }

        .subsection-divider::before,
        .subsection-divider::after {
          content: '';
          position: absolute;
          top: 50%;
          width: 25%;
          height: 1px;
          background: rgba(255, 255, 255, 0.08);
        }

        .subsection-divider::before {
          left: 0;
        }

        .subsection-divider::after {
          right: 0;
        }

        .model-card {
          padding: 1.25rem 1rem;
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

        .model-card.os-model {
          background: var(--family-bg, rgba(255, 255, 255, 0.03));
          border-color: var(--family-border, rgba(255, 255, 255, 0.1));
        }

        .model-card.os-model:hover:not(:disabled) {
          background: var(--family-bg, rgba(255, 255, 255, 0.08));
          border-color: var(--family-border, rgba(255, 255, 255, 0.25));
          filter: brightness(1.3);
        }

        .model-card.os-model.selected {
          border-color: var(--family-border, rgba(255, 255, 255, 0.35));
          filter: brightness(1.5);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 20px var(--family-border, rgba(255, 255, 255, 0.1));
        }

        .model-name {
          font-weight: 500;
          margin-bottom: 0.5rem;
          color: #ffffff;
          font-size: 0.9rem;
          letter-spacing: -0.01em;
        }

        .model-badge {
          display: inline-block;
          padding: 0.3rem 0.7rem;
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
          font-size: 0.65rem;
          font-weight: 500;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .model-badge.custom {
          background: rgba(147, 112, 219, 0.2);
          color: rgba(200, 180, 255, 0.9);
        }

        .model-meta {
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          margin-top: 0.5rem;
        }

        .model-params,
        .model-gpu {
          font-size: 0.7rem;
          padding: 0.2rem 0.5rem;
          background: rgba(255, 255, 255, 0.08);
          color: rgba(255, 255, 255, 0.7);
          font-family: 'JetBrains Mono', monospace;
        }

        .model-card.custom-model {
          border-color: rgba(147, 112, 219, 0.2);
        }

        .model-card.custom-model:hover:not(:disabled) {
          border-color: rgba(147, 112, 219, 0.4);
          box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(200, 180, 255, 0.1);
        }

        .model-card.custom-model.selected {
          border-color: rgba(147, 112, 219, 0.5);
          background: rgba(147, 112, 219, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(200, 180, 255, 0.15),
            0 0 20px rgba(147, 112, 219, 0.2);
        }

        .model-details {
          padding: 1.5rem;
          background: var(--family-bg, rgba(255, 255, 255, 0.03));
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid var(--family-border, rgba(255, 255, 255, 0.12));
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
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
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
