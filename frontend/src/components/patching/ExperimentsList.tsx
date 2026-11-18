/**
 * List of saved patching experiments.
 * Uses localStorage for persistence (backend API integration ready for future).
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Trash2,
  
  
  
  
  
  
  Copy,
  Check } from 'lucide-react'
import type {
  PatchingExperiment,
  PatchSpec,
  PatchingResult,
} from '../../types/patching'

interface ExperimentsListProps {
  currentExperiment?: {
    sourcePrompt?: string
    targetPrompt: string
    modelId: string
    patches: PatchSpec[]
    result?: PatchingResult
  }
  onLoadExperiment: (experiment: PatchingExperiment) => void
  disabled?: boolean
}

const STORAGE_KEY = 'ozera_patching_experiments'

function generateId(): string {
  return `exp_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - 3) + '...'
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

export function ExperimentsList({
  currentExperiment,
  onLoadExperiment,
  disabled = false,
}: ExperimentsListProps) {
  const [experiments, setExperiments] = useState<PatchingExperiment[]>([])
  const [isExpanded, setIsExpanded] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [showSaveForm, setShowSaveForm] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as PatchingExperiment[]
        parsed.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        setExperiments(parsed)
      }
    } catch (error) {
      console.error('Failed to load experiments:', error)
    }
  }, [])

  const saveToStorage = useCallback((exps: PatchingExperiment[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(exps))
    } catch (error) {
      console.error('Failed to save experiments:', error)
    }
  }, [])

  const handleSave = useCallback(() => {
    if (!currentExperiment || !saveName.trim()) return

    const newExperiment: PatchingExperiment = {
      id: generateId(),
      name: saveName.trim(),
      source_prompt: currentExperiment.sourcePrompt,
      target_prompt: currentExperiment.targetPrompt,
      model_id: currentExperiment.modelId,
      model_type: 'ozera',
      patches: currentExperiment.patches,
      baseline_output: currentExperiment.result?.baseline_output,
      patched_output: currentExperiment.result?.patched_output,
      baseline_decoded: currentExperiment.result?.baseline_decoded,
      patched_decoded: currentExperiment.result?.patched_decoded,
      effect_summary: currentExperiment.result?.effect_summary,
      created_at: new Date().toISOString(),
    }

    const updated = [newExperiment, ...experiments]
    setExperiments(updated)
    saveToStorage(updated)
    setSaveName('')
    setShowSaveForm(false)
    setIsExpanded(true)
  }, [currentExperiment, saveName, experiments, saveToStorage])

  const handleDelete = useCallback((id: string) => {
    const updated = experiments.filter(e => e.id !== id)
    setExperiments(updated)
    saveToStorage(updated)
  }, [experiments, saveToStorage])

  const handleLoad = useCallback((experiment: PatchingExperiment) => {
    onLoadExperiment(experiment)
    setIsExpanded(false)
  }, [onLoadExperiment])

  const handleDuplicate = useCallback((experiment: PatchingExperiment) => {
    const duplicated: PatchingExperiment = {
      ...experiment,
      id: generateId(),
      name: `${experiment.name} (copy)`,
      created_at: new Date().toISOString(),
    }
    const updated = [duplicated, ...experiments]
    setExperiments(updated)
    saveToStorage(updated)
    setCopiedId(experiment.id)
    setTimeout(() => setCopiedId(null), 2000)
  }, [experiments, saveToStorage])

  const handleExport = useCallback(() => {
    const dataStr = JSON.stringify(experiments, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `patching-experiments-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [experiments])

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target?.result as string) as PatchingExperiment[]
        const existingIds = new Set(experiments.map(exp => exp.id))
        const newExperiments = imported.filter(exp => !existingIds.has(exp.id))
        const updated = [...newExperiments, ...experiments]
        updated.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        setExperiments(updated)
        saveToStorage(updated)
      } catch (error) {
        console.error('Failed to import experiments:', error)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [experiments, saveToStorage])

  const requiresSourcePrompt = currentExperiment?.patches.some(p => p.intervention_type === 'patch') ?? false

  const canSave = currentExperiment &&
    currentExperiment.patches.length > 0 &&
    currentExperiment.targetPrompt.trim() &&
    (!requiresSourcePrompt || (currentExperiment.sourcePrompt?.trim() ?? false))

  return (
    <div className="exp-list">
      <button
        className="exp-toggle"
        aria-expanded={isExpanded}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2>
          Saved experiments <span className="exp-count">{experiments.length}</span>
        </h2>
        <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
          <path d="M1 1l4 4 4-4" />
        </svg>
      </button>

      {isExpanded && (
        <div className="exp-content">
          {canSave && (
            <div className="exp-save-section">
              {showSaveForm ? (
                <div className="exp-save-form">
                  <input
                    type="text"
                    className="oz-input"
                    value={saveName}
                    onChange={e => setSaveName(e.target.value)}
                    placeholder="Experiment name..."
                    onKeyDown={e => e.key === 'Enter' && handleSave()}
                    autoFocus
                  />
                  <button
                    className="oz-btn oz-btn--small"
                    onClick={handleSave}
                    disabled={!saveName.trim()}
                  >
                    Save
                  </button>
                  <button
                    className="oz-btn oz-btn--ghost oz-btn--small"
                    onClick={() => setShowSaveForm(false)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="oz-btn oz-btn--ghost oz-btn--small"
                  onClick={() => setShowSaveForm(true)}
                  disabled={disabled}
                >
                  Save current experiment
                </button>
              )}
            </div>
          )}

          <div className="exp-actions-row">
            <label className="oz-link">
              <span>Import</span>
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                style={{ display: 'none' }}
              />
            </label>
            <button
              className="oz-link"
              onClick={handleExport}
              disabled={experiments.length === 0}
            >
              <span>Export</span>
            </button>
          </div>

          {experiments.length === 0 ? (
            <p className="oz-hint exp-empty">No saved experiments yet</p>
          ) : (
            <div className="exp-grid">
              {experiments.map(experiment => (
                <div key={experiment.id} className="exp-card">
                  <div className="exp-card-header">
                    <span className="exp-name">{experiment.name}</span>
                    <div className="exp-card-actions">
                      <button
                        className="oz-icon-btn"
                        onClick={() => handleDuplicate(experiment)}
                        title="Duplicate"
                        aria-label="Duplicate"
                      >
                        {copiedId === experiment.id ? (
                          <Check className="exp-action-icon exp-success" />
                        ) : (
                          <Copy className="exp-action-icon" />
                        )}
                      </button>
                      <button
                        className="oz-icon-btn"
                        onClick={() => handleDelete(experiment.id)}
                        title="Delete"
                        aria-label="Delete"
                      >
                        <Trash2 className="exp-action-icon" />
                      </button>
                    </div>
                  </div>

                  <div className="exp-card-body">
                    {experiment.source_prompt ? (
                      <>
                        <div className="exp-prompt-preview">
                          <span className="exp-prompt-label">Source</span>
                          <span className="exp-prompt-text">{truncate(experiment.source_prompt, 40)}</span>
                        </div>
                        <div className="exp-prompt-arrow" aria-hidden="true">→</div>
                      </>
                    ) : (
                      <div className="exp-ablation-badge">
                        <span className="oz-tag">Ablation</span>
                      </div>
                    )}
                    <div className="exp-prompt-preview">
                      <span className="exp-prompt-label">Target</span>
                      <span className="exp-prompt-text">{truncate(experiment.target_prompt, 40)}</span>
                    </div>
                  </div>

                  <div className="exp-card-meta">
                    <span className="exp-meta-item">
                      <span className="exp-meta-label">Model</span>
                      <span className="exp-meta-value">{experiment.model_id}</span>
                    </span>
                    <span className="exp-meta-item">
                      <span className="exp-meta-label">Patches</span>
                      <span className="exp-meta-value">{experiment.patches.length}</span>
                    </span>
                    {experiment.effect_summary && (
                      <span className="exp-meta-item exp-has-result">
                        <span className="exp-meta-label">Changes</span>
                        <span className="exp-meta-value">{experiment.effect_summary.token_changes}</span>
                      </span>
                    )}
                  </div>

                  <div className="exp-card-footer">
                    <span className="exp-timestamp">
                      {formatRelativeTime(experiment.created_at)}
                    </span>
                    <button
                      className="oz-btn oz-btn--ghost oz-btn--small"
                      onClick={() => handleLoad(experiment)}
                      disabled={disabled}
                    >
                      Load
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <style>{`
        /* The section's title, which also opens and closes it */
        .exp-list .exp-toggle {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          width: 100%;
          padding: 0;
          text-align: left;
          cursor: pointer;
        }

        .exp-list .exp-toggle h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
          transition: color 0.2s ease;
        }

        .exp-list .exp-toggle:hover h2 {
          color: var(--oz-mark);
        }

        .exp-list .exp-count {
          margin-left: 0.35rem;
          color: var(--oz-fg-faint);
          font-variant-numeric: tabular-nums;
        }

        .exp-list .exp-content {
          margin-top: 1.1rem;
        }

        .exp-list .exp-save-section {
          margin-bottom: 1rem;
        }

        .exp-list .exp-save-form {
          display: flex;
          gap: 6px;
        }

        .exp-list .exp-save-form .oz-input {
          flex: 1;
          min-width: 0;
        }

        .exp-list .exp-actions-row {
          display: flex;
          gap: 1.25rem;
          margin-bottom: 1rem;
          font-size: 0.85rem;
        }

        .exp-list .exp-empty {
          margin: 0;
        }

        /* The saved experiments: a ruled list */
        .exp-list .exp-card {
          padding: 0.9rem 0 1rem;
          border-top: 1px solid var(--oz-rule);
        }

        .exp-list .exp-card:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .exp-list .exp-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.4rem;
        }

        .exp-list .exp-name {
          overflow: hidden;
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .exp-list .exp-card-actions {
          display: flex;
          gap: 0.1rem;
          flex: none;
        }

        .exp-list .exp-success {
          color: var(--oz-ok);
        }

        .exp-list .exp-card-body {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: 0.35rem 0.75rem;
          margin-bottom: 0.6rem;
        }

        .exp-list .exp-prompt-preview {
          display: flex;
          align-items: baseline;
          gap: 0.5rem;
          min-width: 0;
        }

        .exp-list .exp-prompt-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
        }

        .exp-list .exp-prompt-text {
          overflow: hidden;
          font-family: var(--oz-mono);
          font-size: 0.82rem;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .exp-list .exp-prompt-arrow {
          color: var(--oz-fg-faint);
        }

        .exp-list .exp-card-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem 1.25rem;
          margin-bottom: 0.75rem;
        }

        .exp-list .exp-meta-item {
          display: inline-flex;
          align-items: baseline;
          gap: 0.4rem;
          font-size: 0.85rem;
        }

        .exp-list .exp-meta-label {
          color: var(--oz-fg-dim);
        }

        .exp-list .exp-meta-value {
          font-variant-numeric: tabular-nums;
        }

        .exp-list .exp-card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
        }

        .exp-list .exp-timestamp {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
        }
      `}</style>
    </div>
  )
}

export default ExperimentsList
