/**
 * List of saved patching experiments.
 * Uses localStorage for persistence (backend API integration ready for future).
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Beaker,
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

// Generate unique ID
function generateId(): string {
  return `exp_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`
}

// Truncate text for preview
function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - 3) + '...'
}

// Format relative time
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

  // Load experiments from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as PatchingExperiment[]
        // Sort by created_at descending
        parsed.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        setExperiments(parsed)
      }
    } catch (error) {
      console.error('Failed to load experiments:', error)
    }
  }, [])

  // Save experiments to localStorage
  const saveToStorage = useCallback((exps: PatchingExperiment[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(exps))
    } catch (error) {
      console.error('Failed to save experiments:', error)
    }
  }, [])

  // Save current experiment
  const handleSave = useCallback(() => {
    if (!currentExperiment || !saveName.trim()) return

    const newExperiment: PatchingExperiment = {
      id: generateId(),
      name: saveName.trim(),
      source_prompt: currentExperiment.sourcePrompt,
      target_prompt: currentExperiment.targetPrompt,
      model_id: currentExperiment.modelId,
      model_type: 'ozera', // Default, could be enhanced
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

  // Delete experiment
  const handleDelete = useCallback((id: string) => {
    const updated = experiments.filter(e => e.id !== id)
    setExperiments(updated)
    saveToStorage(updated)
  }, [experiments, saveToStorage])

  // Load experiment
  const handleLoad = useCallback((experiment: PatchingExperiment) => {
    onLoadExperiment(experiment)
    setIsExpanded(false)
  }, [onLoadExperiment])

  // Duplicate experiment
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

  // Export experiments to JSON
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

  // Import experiments from JSON
  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target?.result as string) as PatchingExperiment[]
        // Merge with existing, avoiding duplicates by ID
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

  // Check if source prompt is required (any patch uses 'patch' intervention type)
  const requiresSourcePrompt = currentExperiment?.patches.some(p => p.intervention_type === 'patch') ?? false

  const canSave = currentExperiment &&
    currentExperiment.patches.length > 0 &&
    currentExperiment.targetPrompt.trim() &&
    (!requiresSourcePrompt || (currentExperiment.sourcePrompt?.trim() ?? false))

  return (
    <div className="experiments-list">
      <button
        className="section-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="toggle-left">
          <Beaker className="toggle-icon" />
          <h3>Saved Experiments ({experiments.length})</h3>
        </div>
        {isExpanded ? <ChevronUp /> : <ChevronDown />}
      </button>

      {isExpanded && (
        <div className="experiments-content">
          {/* Save current experiment */}
          {canSave && (
            <div className="save-section">
              {showSaveForm ? (
                <div className="save-form">
                  <input
                    type="text"
                    value={saveName}
                    onChange={e => setSaveName(e.target.value)}
                    placeholder="Experiment name..."
                    onKeyDown={e => e.key === 'Enter' && handleSave()}
                    autoFocus
                  />
                  <button
                    className="save-btn"
                    onClick={handleSave}
                    disabled={!saveName.trim()}
                  >
                    Save
                  </button>
                  <button
                    className="cancel-btn"
                    onClick={() => setShowSaveForm(false)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="show-save-btn"
                  onClick={() => setShowSaveForm(true)}
                  disabled={disabled}
                >
                  <Sparkles className="btn-icon" />
                  Save Current Experiment
                </button>
              )}
            </div>
          )}

          {/* Import/Export buttons */}
          <div className="actions-row">
            <label className="action-btn import-btn">
              <Upload className="action-icon" />
              <span>Import</span>
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                style={{ display: 'none' }}
              />
            </label>
            <button
              className="action-btn export-btn"
              onClick={handleExport}
              disabled={experiments.length === 0}
            >
              <Download className="action-icon" />
              <span>Export</span>
            </button>
          </div>

          {/* Experiments list */}
          {experiments.length === 0 ? (
            <div className="empty-state">
              <Beaker className="empty-icon" />
              <p>No saved experiments yet</p>
              <p className="empty-hint">Run an experiment and save it for later</p>
            </div>
          ) : (
            <div className="experiments-grid">
              {experiments.map(experiment => (
                <div key={experiment.id} className="experiment-card">
                  <div className="card-header">
                    <span className="experiment-name">{experiment.name}</span>
                    <div className="card-actions">
                      <button
                        className="card-action-btn"
                        onClick={() => handleDuplicate(experiment)}
                        title="Duplicate"
                      >
                        {copiedId === experiment.id ? (
                          <Check className="action-icon success" />
                        ) : (
                          <Copy className="action-icon" />
                        )}
                      </button>
                      <button
                        className="card-action-btn delete"
                        onClick={() => handleDelete(experiment.id)}
                        title="Delete"
                      >
                        <Trash2 className="action-icon" />
                      </button>
                    </div>
                  </div>

                  <div className="card-body">
                    {experiment.source_prompt ? (
                      <>
                        <div className="prompt-preview">
                          <span className="prompt-label">Source:</span>
                          <span className="prompt-text">{truncate(experiment.source_prompt, 40)}</span>
                        </div>
                        <div className="prompt-arrow">
                          <ArrowRight className="arrow-icon" />
                        </div>
                      </>
                    ) : (
                      <div className="ablation-badge">
                        <span className="ablation-label">Ablation</span>
                      </div>
                    )}
                    <div className="prompt-preview">
                      <span className="prompt-label">Target:</span>
                      <span className="prompt-text">{truncate(experiment.target_prompt, 40)}</span>
                    </div>
                  </div>

                  <div className="card-meta">
                    <span className="meta-item">
                      <span className="meta-label">Model:</span>
                      <span className="meta-value">{experiment.model_id}</span>
                    </span>
                    <span className="meta-item">
                      <span className="meta-label">Patches:</span>
                      <span className="meta-value">{experiment.patches.length}</span>
                    </span>
                    {experiment.effect_summary && (
                      <span className="meta-item has-result">
                        <span className="meta-label">Changes:</span>
                        <span className="meta-value">{experiment.effect_summary.token_changes}</span>
                      </span>
                    )}
                  </div>

                  <div className="card-footer">
                    <span className="timestamp">
                      <Clock className="time-icon" />
                      {formatRelativeTime(experiment.created_at)}
                    </span>
                    <button
                      className="load-btn"
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
        .experiments-list {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .section-toggle {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.875rem 1rem;
          background: transparent;
          border: none;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.8);
          transition: background 0.2s;
        }

        .section-toggle:hover {
          background: rgba(255, 255, 255, 0.03);
        }

        .toggle-left {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .toggle-icon {
          width: 16px;
          height: 16px;
          color: rgba(255, 255, 255, 0.5);
        }

        .section-toggle h3 {
          margin: 0;
          font-size: 0.85rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.8);
        }

        .experiments-content {
          padding: 0 1rem 1rem;
        }

        .save-section {
          margin-bottom: 1rem;
        }

        .save-form {
          display: flex;
          gap: 0.5rem;
        }

        .save-form input {
          flex: 1;
          padding: 0.5rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.85rem;
        }

        .save-form input:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .save-btn, .cancel-btn {
          padding: 0.5rem 0.875rem;
          font-size: 0.8rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          border: 1px solid;
        }

        .save-btn {
          background: rgba(59, 130, 246, 0.2);
          border-color: rgba(59, 130, 246, 0.3);
          color: rgba(59, 130, 246, 1);
        }

        .save-btn:hover:not(:disabled) {
          background: rgba(59, 130, 246, 0.3);
        }

        .save-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .cancel-btn {
          background: transparent;
          border-color: rgba(255, 255, 255, 0.15);
          color: rgba(255, 255, 255, 0.6);
        }

        .cancel-btn:hover {
          background: rgba(255, 255, 255, 0.05);
        }

        .show-save-btn {
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

        .show-save-btn:hover:not(:disabled) {
          background: rgba(34, 197, 94, 0.25);
          border-color: rgba(34, 197, 94, 0.4);
        }

        .show-save-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-icon {
          width: 14px;
          height: 14px;
        }

        .actions-row {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .action-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.375rem;
          padding: 0.5rem 0.75rem;
          background: transparent;
          border: 1px solid rgba(255, 255, 255, 0.12);
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .action-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.8);
        }

        .action-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .action-icon {
          width: 14px;
          height: 14px;
        }

        .empty-state {
          padding: 2rem 1rem;
          text-align: center;
        }

        .empty-icon {
          width: 32px;
          height: 32px;
          color: rgba(255, 255, 255, 0.2);
          margin-bottom: 0.75rem;
        }

        .empty-state p {
          margin: 0;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .empty-hint {
          margin-top: 0.25rem !important;
          font-size: 0.75rem !important;
          color: rgba(255, 255, 255, 0.35) !important;
        }

        .experiments-grid {
          display: flex;
          flex-direction: column;
          gap: 0.625rem;
        }

        .experiment-card {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.08);
          padding: 0.75rem;
          transition: border-color 0.2s;
        }

        .experiment-card:hover {
          border-color: rgba(255, 255, 255, 0.15);
        }

        .card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.625rem;
        }

        .experiment-name {
          font-size: 0.9rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
        }

        .card-actions {
          display: flex;
          gap: 0.25rem;
        }

        .card-action-btn {
          padding: 0.25rem;
          background: transparent;
          border: none;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.3);
          transition: color 0.2s;
        }

        .card-action-btn:hover {
          color: rgba(255, 255, 255, 0.7);
        }

        .card-action-btn.delete:hover {
          color: rgba(239, 68, 68, 0.8);
        }

        .card-action-btn .action-icon {
          width: 14px;
          height: 14px;
        }

        .card-action-btn .action-icon.success {
          color: rgba(34, 197, 94, 0.8);
        }

        .card-body {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.625rem;
        }

        .prompt-preview {
          flex: 1;
          min-width: 0;
        }

        .ablation-badge {
          flex-shrink: 0;
          margin-right: 0.5rem;
        }

        .ablation-label {
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

        .prompt-label {
          display: block;
          font-size: 0.65rem;
          color: rgba(255, 255, 255, 0.4);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.125rem;
        }

        .prompt-text {
          display: block;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.7);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .prompt-arrow {
          flex-shrink: 0;
        }

        .arrow-icon {
          width: 14px;
          height: 14px;
          color: rgba(255, 255, 255, 0.3);
        }

        .card-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem;
          margin-bottom: 0.625rem;
        }

        .meta-item {
          font-size: 0.7rem;
        }

        .meta-label {
          color: rgba(255, 255, 255, 0.4);
          margin-right: 0.25rem;
        }

        .meta-value {
          color: rgba(255, 255, 255, 0.7);
          font-weight: 500;
        }

        .meta-item.has-result .meta-value {
          color: rgba(34, 197, 94, 0.8);
        }

        .card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 0.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .timestamp {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .time-icon {
          width: 12px;
          height: 12px;
        }

        .load-btn {
          padding: 0.375rem 0.875rem;
          background: rgba(59, 130, 246, 0.15);
          border: 1px solid rgba(59, 130, 246, 0.25);
          color: rgba(59, 130, 246, 0.9);
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .load-btn:hover:not(:disabled) {
          background: rgba(59, 130, 246, 0.25);
          border-color: rgba(59, 130, 246, 0.4);
        }

        .load-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  )
}

export default ExperimentsList
