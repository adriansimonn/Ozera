import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, Download } from 'lucide-react'
import { TrainingPanel } from '../components/training/TrainingPanel'
import { TrainingProgress } from '../components/training/TrainingProgress'
import { ModelUploadPanel } from '../components/training/ModelUploadPanel'
import { ConfirmModal } from '../components/common/ConfirmModal'
import {
  useTrainingJobs,
  useTrainingProgress,
  useCustomModels,
  useUploadedModels,
} from '../hooks/useTraining'
import { NavBar } from '../components/common/NavBar'
import { apiClient, GpuType } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import { VAR } from '../styles/appTheme'
import { Spinner } from '../components/common/Spinner'

type LeftMode = 'training' | 'upload'

export default function DefaultTrainingPage() {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const [mode, setMode] = useState<LeftMode>('training')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null)
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<{ modelId: string; modelName: string; type: 'trained' | 'uploaded' } | null>(null)
  const autoDownloadTriggeredRef = useRef(false)

  const { jobs, fetchJobs, startJob, cancelJob } = useTrainingJobs()
  const { progress, completed, completedModelName, error, reset: resetProgress } = useTrainingProgress(activeJobId)
  const { models, fetchModels, deleteModel } = useCustomModels()
  const { models: uploadedModels, fetchModels: fetchUploadedModels, deleteModel: deleteUploadedModel } = useUploadedModels()

  useEffect(() => {
    if (completed && completedModelName && autoDownloadEnabled && !autoDownloadTriggeredRef.current) {
      autoDownloadTriggeredRef.current = true
      handleDownload()
    }
  }, [completed, completedModelName, autoDownloadEnabled])

  const handleDownload = useCallback(async () => {
    if (!completedModelName) return
    setDownloading(true)
    try {
      await apiClient.downloadCustomModel(completedModelName)
    } catch (err) {
      console.error('Failed to download model:', err)
    } finally {
      setDownloading(false)
    }
  }, [completedModelName])

  const handleDownloadModel = useCallback(async (modelId: string) => {
    setDownloadingModelId(modelId)
    try {
      await apiClient.downloadCustomModel(modelId)
    } catch (err) {
      console.error('Failed to download model:', err)
    } finally {
      setDownloadingModelId(null)
    }
  }, [])

  const handleStartTraining = useCallback(async (
    datasetId: string,
    modelConfig: 'nano' | 'mini',
    modelName: string,
    epochs: number,
    batchSize: number,
    learningRate: number,
    seqLen: number,
    gpuType: GpuType,
    autoDownload: boolean,
    overwriteExisting: boolean
  ) => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    autoDownloadTriggeredRef.current = false
    setAutoDownloadEnabled(autoDownload)
    const response = await startJob(datasetId, modelConfig, modelName, epochs, batchSize, learningRate, seqLen, gpuType, overwriteExisting)
    setActiveJobId(response.job_id)
  }, [isAuthenticated, navigate, startJob])

  const handleCancel = useCallback(async () => {
    if (activeJobId) await cancelJob(activeJobId)
  }, [activeJobId, cancelJob])

  const handleDismiss = useCallback(() => {
    setActiveJobId(null)
    resetProgress()
    fetchJobs()
    fetchModels()
    fetchUploadedModels()
  }, [resetProgress, fetchJobs, fetchModels, fetchUploadedModels])

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteConfirm) return
    const { modelId, type } = deleteConfirm
    setDeletingModelId(modelId)
    try {
      if (type === 'trained') await deleteModel(modelId)
      else await deleteUploadedModel(modelId)
    } catch (err) {
      console.error('Failed to delete model:', err)
    } finally {
      setDeletingModelId(null)
      setDeleteConfirm(null)
    }
  }, [deleteConfirm, deleteModel, deleteUploadedModel])

  const handleUploadComplete = useCallback(() => {
    fetchModels()
    fetchUploadedModels()
  }, [fetchModels, fetchUploadedModels])

  const isTrainingActive = progress && progress.status === 'running'

  const allCustomModels = [
    ...models.map(m => ({ ...m, type: 'trained' as const })),
    ...uploadedModels.map(m => ({ ...m, type: 'uploaded' as const })),
  ]

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return VAR.ok
      case 'running': return VAR.accent
      case 'failed': return VAR.danger
      case 'cancelled': return VAR.warn
      default: return VAR.fgFaint
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatParams = (params: number | null | undefined) => {
    if (!params) return null
    if (params < 1000000) return `${(params / 1000).toFixed(0)}K params`
    if (params < 1000000000) return `${(params / 1000000).toFixed(1)}M params`
    return `${(params / 1000000000).toFixed(2)}B params`
  }

  return (
    <div className="dt-page oz-page">
      <NavBar />

      <div className="dt-layout oz-page__body">
        {/* Left Panel — Training / Upload (60%) */}
        <div className="dt-left">
          <div className="dt-tabs" role="tablist">
            <button role="tab" aria-selected={mode === 'training'} className={`oz-tab ${mode === 'training' ? 'active' : ''}`} onClick={() => setMode('training')}>
              Train Model
            </button>
            <button role="tab" aria-selected={mode === 'upload'} className={`oz-tab ${mode === 'upload' ? 'active' : ''}`} onClick={() => setMode('upload')}>
              Upload Model
            </button>
          </div>

          <div className="oz-scroll">
            <div className="dt-panel-wrap">
              {mode === 'training' ? (
                <TrainingPanel
                  onStartTraining={handleStartTraining}
                  disabled={isTrainingActive || false}
                  onShowLogin={() => navigate('/auth')}
                />
              ) : (
                <ModelUploadPanel
                  onUploadComplete={handleUploadComplete}
                  disabled={isTrainingActive || false}
                  onShowLogin={() => navigate('/auth')}
                />
              )}
            </div>
          </div>
        </div>

        {/* Right Panel — Jobs & Models (40%) */}
        <div className="dt-right oz-main">
          <div className="oz-scroll">
            {/* Active Training Progress */}
            {(mode === 'training' || progress) && (
              <div className="oz-section">
                <TrainingProgress
                  progress={progress}
                  completed={completed}
                  completedModelName={completedModelName}
                  error={error}
                  onCancel={handleCancel}
                  onDismiss={handleDismiss}
                  onDownload={handleDownload}
                  downloading={downloading}
                />
              </div>
            )}

            {/* Custom Models */}
            <div className="oz-section">
              <h2 className="dt-heading">Custom models</h2>

              {allCustomModels.length === 0 ? (
                <p className="oz-hint">No custom models yet</p>
              ) : (
                <div className="dt-list">
                  {allCustomModels.map((model) => (
                    <div key={model.model_id} className="dt-row">
                      <div className="dt-row-info">
                        <div className="dt-row-name">
                          <span>{model.name}</span>
                          <span className="oz-tag">{model.type === 'trained' ? 'Trained' : 'Uploaded'}</span>
                        </div>
                        <div className="dt-row-meta">
                          {model.type === 'trained' ? (
                            <>{model.base_config} · Val loss: {model.val_loss.toFixed(4)}</>
                          ) : (
                            <>{formatBytes(model.file_size_bytes)}{formatParams(model.num_parameters) && ` · ${formatParams(model.num_parameters)}`}</>
                          )}
                        </div>
                      </div>
                      <div className="dt-row-actions">
                        <button
                          onClick={() => navigate('/generate')}
                          title="Use model for generation"
                          className="oz-btn oz-btn--ghost oz-btn--small"
                        >
                          Use
                        </button>
                        {model.type === 'trained' && (
                          <button
                            onClick={() => handleDownloadModel(model.model_id)}
                            disabled={downloadingModelId === model.model_id}
                            title="Download model"
                            aria-label="Download model"
                            className="oz-icon-btn"
                          >
                            {downloadingModelId === model.model_id ? <Spinner /> : <Download />}
                          </button>
                        )}
                        <button
                          onClick={() => setDeleteConfirm({ modelId: model.model_id, modelName: model.name, type: model.type })}
                          disabled={deletingModelId === model.model_id}
                          title="Delete model"
                          aria-label="Delete model"
                          className="oz-icon-btn"
                        >
                          {deletingModelId === model.model_id ? <Spinner /> : <Trash2 />}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Recent Jobs */}
            {mode === 'training' && (
              <div className="oz-section">
                <h2 className="dt-heading">Recent jobs</h2>

                {jobs.length === 0 ? (
                  <p className="oz-hint">No training jobs yet</p>
                ) : (
                  <div className="dt-list">
                    {jobs.slice(0, 5).map((job) => (
                      <div key={job.job_id} className="dt-row">
                        <div className="dt-job-dot" style={{ background: getStatusColor(job.status) }} title={job.status} />
                        <div className="dt-row-info">
                          <div className="dt-row-name">{job.model_name}</div>
                          <div className="dt-row-meta">
                            {job.dataset_name} · {job.current_epoch}/{job.total_epochs} epochs
                          </div>
                        </div>
                        <div className="dt-row-meta">{formatDate(job.created_at)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <ConfirmModal
        isOpen={!!deleteConfirm}
        title="Delete Model"
        message={`Are you sure you want to delete "${deleteConfirm?.modelName}"? This action cannot be undone.`}
        confirmLabel="Delete"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteConfirm(null)}
        isLoading={!!deletingModelId}
      />

      <style>{`
        /* Two columns: what to train or upload, and what's running and done */
        .dt-left {
          display: flex;
          flex-direction: column;
          flex: none;
          width: 58%;
          min-width: 400px;
          overflow: hidden;
          border-right: 1px solid var(--oz-rule);
        }

        .dt-tabs {
          display: flex;
          align-items: stretch;
          gap: 1.75rem;
          flex: none;
          min-height: 3.75rem;
          padding: 0 var(--oz-gutter);
          border-bottom: 1px solid var(--oz-rule);
        }

        .dt-tabs .oz-tab {
          padding: 0;
        }

        .dt-panel-wrap > * {
          border: none !important;
        }

        .dt-right .oz-section:last-child {
          border-bottom: none;
        }

        .dt-heading {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        /* Models and jobs: ruled lists */
        .dt-row {
          display: flex;
          align-items: center;
          gap: 0.9rem;
          padding: 0.8rem 0;
          border-top: 1px solid var(--oz-rule);
        }

        .dt-row:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .dt-row-info {
          flex: 1;
          min-width: 0;
        }

        .dt-row-name {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
        }

        .dt-row-meta {
          margin-top: 0.1rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-variant-numeric: tabular-nums;
        }

        .dt-row-actions {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          flex-shrink: 0;
        }

        /* A job's state, as a cell */
        .dt-job-dot {
          flex-shrink: 0;
          width: 8px;
          height: 8px;
        }

        @media (max-width: 1000px) {
          .dt-page.oz-page { height: auto; overflow: visible; }
          .dt-layout.oz-page__body { flex-direction: column; }
          .dt-left { width: 100%; min-width: 0; border-right: none; border-bottom: 1px solid var(--oz-rule); }
        }
      `}</style>
    </div>
  )
}
