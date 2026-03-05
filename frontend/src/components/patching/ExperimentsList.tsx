/**
 * List of saved patching experiments.
 * Uses localStorage for persistence (backend API integration ready for future).
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Trash2,
  Clock,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Download,
  Upload,
  Copy,
  Check,
  Sparkles
} from 'lucide-react'
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
    <>
      <button
        className="section-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2>Saved Experiments ({experiments.length})</h2>
        {isExpanded ? <ChevronUp /> : <ChevronDown />}
      </button>

      {isExpanded && (
        <div className="exp-content">
          {canSave && (
            <div className="exp-save-section">
              {showSaveForm ? (
                <div className="exp-save-form">
                  <input
                    type="text"
                    value={saveName}
                    onChange={e => setSaveName(e.target.value)}
                    placeholder="Experiment name..."
                    onKeyDown={e => e.key === 'Enter' && handleSave()}
                    autoFocus
                  />
                  <button
                    className="exp-save-btn"
                    onClick={handleSave}
                    disabled={!saveName.trim()}
                  >
                    Save
                  </button>
                  <button
                    className="exp-cancel-btn"
                    onClick={() => setShowSaveForm(false)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="exp-show-save-btn"
                  onClick={() => setShowSaveForm(true)}
                  disabled={disabled}
                >
                  <Sparkles className="exp-btn-icon" />
                  Save Current Experiment
                </button>
              )}
            </div>
          )}

          <div className="exp-actions-row">
            <label className="exp-action-btn">
              <Upload className="exp-action-icon" />
              <span>Import</span>
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                style={{ display: 'none' }}
              />
            </label>
            <button
              className="exp-action-btn"
              onClick={handleExport}
              disabled={experiments.length === 0}
            >
              <Download className="exp-action-icon" />
              <span>Export</span>
            </button>
          </div>

          {experiments.length === 0 ? (
            <div className="exp-empty">
              <p>No saved experiments yet</p>
              <p className="exp-empty-hint">Run an experiment and save it for later</p>
            </div>
          ) : (
            <div className="exp-grid">
              {experiments.map(experiment => (
                <div key={experiment.id} className="exp-card">
                  <div className="exp-card-header">
                    <span className="exp-name">{experiment.name}</span>
                    <div className="exp-card-actions">
                      <button
                        className="exp-card-action-btn"
                        onClick={() => handleDuplicate(experiment)}
                        title="Duplicate"
                      >
                        {copiedId === experiment.id ? (
                          <Check className="exp-action-icon exp-success" />
                        ) : (
                          <Copy className="exp-action-icon" />
                        )}
                      </button>
                      <button
                        className="exp-card-action-btn exp-delete"
                        onClick={() => handleDelete(experiment.id)}
                        title="Delete"
                      >
                        <Trash2 className="exp-action-icon" />
                      </button>
                    </div>
                  </div>

                  <div className="exp-card-body">
                    {experiment.source_prompt ? (
                      <>
                        <div className="exp-prompt-preview">
                          <span className="exp-prompt-label">Source:</span>
                          <span className="exp-prompt-text">{truncate(experiment.source_prompt, 40)}</span>
                        </div>
                        <div className="exp-prompt-arrow">
                          <ArrowRight className="exp-arrow-icon" />
                        </div>
                      </>
                    ) : (
                      <div className="exp-ablation-badge">
                        <span className="exp-ablation-label">Ablation</span>
                      </div>
                    )}
                    <div className="exp-prompt-preview">
                      <span className="exp-prompt-label">Target:</span>
                      <span className="exp-prompt-text">{truncate(experiment.target_prompt, 40)}</span>
                    </div>
                  </div>

                  <div className="exp-card-meta">
                    <span className="exp-meta-item">
                      <span className="exp-meta-label">Model:</span>
                      <span className="exp-meta-value">{experiment.model_id}</span>
                    </span>
                    <span className="exp-meta-item">
                      <span className="exp-meta-label">Patches:</span>
                      <span className="exp-meta-value">{experiment.patches.length}</span>
                    </span>
                    {experiment.effect_summary && (
                      <span className="exp-meta-item exp-has-result">
                        <span className="exp-meta-label">Changes:</span>
                        <span className="exp-meta-value">{experiment.effect_summary.token_changes}</span>
                      </span>
                    )}
                  </div>

                  <div className="exp-card-footer">
                    <span className="exp-timestamp">
                      <Clock className="exp-time-icon" />
                      {formatRelativeTime(experiment.created_at)}
                    </span>
                    <button
                      className="exp-load-btn"
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
        .exp-content {
          margin-top: 0.75rem;
        }

        .exp-save-section {
          margin-bottom: 1rem;
        }

        .exp-save-form {
          display: flex;
          gap: 0.5rem;
        }

        .exp-save-form input {
          flex: 1;
          padding: 0.5rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.85rem;
        }

        .exp-save-form input:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .exp-save-btn, .exp-cancel-btn {
          padding: 0.5rem 0.875rem;
          font-size: 0.8rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          border: 1px solid;
        }

        .exp-save-btn {
          background: rgba(59, 130, 246, 0.2);
          border-color: rgba(59, 130, 246, 0.3);
          color: rgba(59, 130, 246, 1);
        }

        .exp-save-btn:hover:not(:disabled) {
          background: rgba(59, 130, 246, 0.3);
        }

        .exp-save-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .exp-cancel-btn {
          background: transparent;
          border-color: rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.6);
        }

        .exp-cancel-btn:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .exp-show-save-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(34, 197, 94, 0.15);
          border: 1px solid rgba(34, 197, 94, 0.25);
          color: rgba(34, 197, 94, 0.9);
          font-size: 0.85rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .exp-show-save-btn:hover:not(:disabled) {
          background: rgba(34, 197, 94, 0.25);
          border-color: rgba(34, 197, 94, 0.4);
        }

        .exp-show-save-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .exp-btn-icon {
          width: 14px;
          height: 14px;
        }

        .exp-actions-row {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .exp-action-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.375rem;
          padding: 0.5rem 0.75rem;
          background: transparent;
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .exp-action-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.8);
        }

        .exp-action-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .exp-action-icon {
          width: 14px;
          height: 14px;
        }

        .exp-empty {
          padding: 2rem 1rem;
          text-align: center;
        }

        .exp-empty p {
          margin: 0;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .exp-empty-hint {
          margin-top: 0.25rem !important;
          font-size: 0.75rem !important;
          color: rgba(255, 255, 255, 0.35) !important;
        }

        .exp-grid {
          display: flex;
          flex-direction: column;
          gap: 0.625rem;
        }

        .exp-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 0.75rem;
          transition: border-color 0.2s;
        }

        .exp-card:hover {
          border-color: rgba(255, 255, 255, 0.2);
        }

        .exp-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.625rem;
        }

        .exp-name {
          font-size: 0.9rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
        }

        .exp-card-actions {
          display: flex;
          gap: 0.25rem;
        }

        .exp-card-action-btn {
          padding: 0.25rem;
          background: transparent;
          border: none;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.3);
          transition: color 0.2s;
        }

        .exp-card-action-btn:hover {
          color: rgba(255, 255, 255, 0.7);
        }

        .exp-card-action-btn.exp-delete:hover {
          color: rgba(239, 68, 68, 0.8);
        }

        .exp-card-action-btn .exp-action-icon {
          width: 14px;
          height: 14px;
        }

        .exp-success {
          color: rgba(34, 197, 94, 0.8);
        }

        .exp-card-body {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.625rem;
        }

        .exp-prompt-preview {
          flex: 1;
          min-width: 0;
        }

        .exp-ablation-badge {
          flex-shrink: 0;
          margin-right: 0.5rem;
        }

        .exp-ablation-label {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          background: rgba(168, 85, 247, 0.2);
          border: 1px solid rgba(168, 85, 247, 0.3);
          color: rgba(168, 85, 247, 0.9);
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .exp-prompt-label {
          display: block;
          font-size: 0.65rem;
          color: rgba(255, 255, 255, 0.4);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.125rem;
        }

        .exp-prompt-text {
          display: block;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.7);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .exp-prompt-arrow {
          flex-shrink: 0;
        }

        .exp-arrow-icon {
          width: 14px;
          height: 14px;
          color: rgba(255, 255, 255, 0.3);
        }

        .exp-card-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem;
          margin-bottom: 0.625rem;
        }

        .exp-meta-item {
          font-size: 0.7rem;
        }

        .exp-meta-label {
          color: rgba(255, 255, 255, 0.4);
          margin-right: 0.25rem;
        }

        .exp-meta-value {
          color: rgba(255, 255, 255, 0.7);
          font-weight: 500;
        }

        .exp-has-result .exp-meta-value {
          color: rgba(34, 197, 94, 0.8);
        }

        .exp-card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 0.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .exp-timestamp {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .exp-time-icon {
          width: 12px;
          height: 12px;
        }

        .exp-load-btn {
          padding: 0.375rem 0.875rem;
          background: rgba(59, 130, 246, 0.15);
          border: 1px solid rgba(59, 130, 246, 0.25);
          color: rgba(59, 130, 246, 0.9);
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .exp-load-btn:hover:not(:disabled) {
          background: rgba(59, 130, 246, 0.25);
          border-color: rgba(59, 130, 246, 0.4);
        }

        .exp-load-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Light mode */
        [data-bg="light"] .exp-save-form input { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .exp-cancel-btn { border-color: rgba(0,0,0,0.12); color: rgba(0,0,0,0.5); }
        [data-bg="light"] .exp-cancel-btn:hover { background: rgba(0,0,0,0.05); }
        [data-bg="light"] .exp-action-btn { border-color: rgba(0,0,0,0.12); color: rgba(0,0,0,0.5); }
        [data-bg="light"] .exp-action-btn:hover:not(:disabled) { background: rgba(0,0,0,0.05); border-color: rgba(0,0,0,0.2); color: rgba(0,0,0,0.7); }
        [data-bg="light"] .exp-empty p { color: rgba(0,0,0,0.5); }
        [data-bg="light"] .exp-empty-hint { color: rgba(0,0,0,0.35) !important; }
        [data-bg="light"] .exp-card { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .exp-card:hover { border-color: rgba(0,0,0,0.15); }
        [data-bg="light"] .exp-name { color: #1d1d1f; }
        [data-bg="light"] .exp-card-action-btn { color: rgba(0,0,0,0.3); }
        [data-bg="light"] .exp-card-action-btn:hover { color: rgba(0,0,0,0.7); }
        [data-bg="light"] .exp-prompt-label { color: rgba(0,0,0,0.4); }
        [data-bg="light"] .exp-prompt-text { color: rgba(0,0,0,0.6); }
        [data-bg="light"] .exp-arrow-icon { color: rgba(0,0,0,0.3); }
        [data-bg="light"] .exp-meta-label { color: rgba(0,0,0,0.4); }
        [data-bg="light"] .exp-meta-value { color: rgba(0,0,0,0.6); }
        [data-bg="light"] .exp-card-footer { border-top-color: rgba(0,0,0,0.06); }
        [data-bg="light"] .exp-timestamp { color: rgba(0,0,0,0.4); }
      `}</style>
    </>
  )
}

export default ExperimentsList
