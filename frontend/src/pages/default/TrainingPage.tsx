import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, Clock, Download, Loader2, Play, Upload, Cpu, UploadCloud } from 'lucide-react'
import { TrainingPanel } from '../../components/training/TrainingPanel'
import { TrainingProgress } from '../../components/training/TrainingProgress'
import { ModelUploadPanel } from '../../components/training/ModelUploadPanel'
import { ConfirmModal } from '../../components/common/ConfirmModal'
import {
  useTrainingJobs,
  useTrainingProgress,
  useCustomModels,
  useUploadedModels,
} from '../../hooks/useTraining'
import { NavBar } from '../../components/common/NavBar'
import { apiClient, GpuType } from '../../api/client'
import { useAuthStore } from '../../stores/authStore'
import { useTheme } from '../../hooks/useTheme'

type LeftMode = 'training' | 'upload'

export default function DefaultTrainingPage() {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const { isLight } = useTheme()
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
      case 'completed': return '#22c55e'
      case 'running': return '#3b82f6'
      case 'failed': return '#ef4444'
      case 'cancelled': return '#f59e0b'
      default: return c.textSub
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

  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    panelBorder: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
    textMid: isLight ? '#1d1d1f' : '#ffffff',
    textSub: isLight ? '#1d1d1f' : '#ffffff',
    controlBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
    controlBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    controlHover: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.08)',
    cardBg: isLight ? 'rgba(0,0,0,0.02)' : 'rgba(255,255,255,0.02)',
    cardBorder: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    spinnerHead: isLight ? '#1d1d1f' : '#ffffff',
  }

  return (
    <div className="dt-page">
      <NavBar />

      <div className="dt-layout" style={{ borderColor: c.divider }}>
        {/* Left Panel — Training / Upload (60%) */}
        <div className="dt-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="dt-left-header" style={{ borderBottom: `1px solid ${c.divider}` }}>
            <div className="dt-mode-switch" style={{ background: c.controlBg, borderColor: c.controlBorder }}>
              <button
                className={`dt-mode-btn ${mode === 'training' ? 'active' : ''}`}
                onClick={() => setMode('training')}
                style={{
                  color: mode === 'training' ? c.text : c.textSub,
                  background: mode === 'training' ? (isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)') : 'transparent',
                }}
              >
                <Cpu style={{ width: 13, height: 13 }} />
                Train Model
              </button>
              <button
                className={`dt-mode-btn ${mode === 'upload' ? 'active' : ''}`}
                onClick={() => setMode('upload')}
                style={{
                  color: mode === 'upload' ? c.text : c.textSub,
                  background: mode === 'upload' ? (isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)') : 'transparent',
                }}
              >
                <UploadCloud style={{ width: 13, height: 13 }} />
                Upload Model
              </button>
            </div>
          </div>

          <div className="dt-left-scroll">
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
        <div className="dt-right" style={{ background: c.bg }}>
          <div className="dt-right-scroll">
            {/* Active Training Progress */}
            {(mode === 'training' || progress) && (
              <div className="dt-section">
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
            <div className="dt-section">
              <div className="dt-section-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                <div className="dt-section-header">
                  <h3 style={{ color: c.text }}>Custom Models</h3>
                </div>

                {allCustomModels.length === 0 ? (
                  <div className="dt-empty" style={{ color: c.textSub }}>
                    No custom models yet
                  </div>
                ) : (
                  <div className="dt-model-list">
                    {allCustomModels.map((model) => (
                      <div key={model.model_id} className="dt-model-row" style={{ background: c.cardBg, borderColor: c.cardBorder }}>
                        <div className="dt-model-icon" style={{
                          background: model.type === 'trained' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                        }}>
                          {model.type === 'trained' ? (
                            <Play size={11} color="#22c55e" />
                          ) : (
                            <Upload size={11} color="#3b82f6" />
                          )}
                        </div>
                        <div className="dt-model-info">
                          <div className="dt-model-name" style={{ color: c.text }}>
                            {model.name}
                            <span className="dt-model-badge" style={{
                              background: model.type === 'trained' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                              color: model.type === 'trained' ? '#22c55e' : '#3b82f6',
                            }}>
                              {model.type}
                            </span>
                          </div>
                          <div className="dt-model-meta" style={{ color: c.textSub }}>
                            {model.type === 'trained' ? (
                              <>{model.base_config} · Val loss: {model.val_loss.toFixed(4)}</>
                            ) : (
                              <>{formatBytes(model.file_size_bytes)}{formatParams(model.num_parameters) && ` · ${formatParams(model.num_parameters)}`}</>
                            )}
                          </div>
                        </div>
                        <div className="dt-model-actions">
                          <button
                            onClick={() => navigate('/')}
                            title="Use model for generation"
                            className="dt-use-btn"
                            style={{ color: '#22c55e', background: 'rgba(34, 197, 94, 0.1)', borderColor: 'rgba(34, 197, 94, 0.25)' }}
                          >
                            <Play size={11} />
                            Use
                          </button>
                          {model.type === 'trained' && (
                            <button
                              onClick={() => handleDownloadModel(model.model_id)}
                              disabled={downloadingModelId === model.model_id}
                              title="Download model"
                              className="dt-icon-btn"
                              style={{ color: c.textSub }}
                            >
                              {downloadingModelId === model.model_id ? (
                                <Loader2 size={13} className="dt-spin" />
                              ) : (
                                <Download size={13} />
                              )}
                            </button>
                          )}
                          <button
                            onClick={() => setDeleteConfirm({ modelId: model.model_id, modelName: model.name, type: model.type })}
                            disabled={deletingModelId === model.model_id}
                            title="Delete model"
                            className="dt-icon-btn"
                            style={{ color: c.textSub }}
                          >
                            {deletingModelId === model.model_id ? (
                              <Loader2 size={13} className="dt-spin" />
                            ) : (
                              <Trash2 size={13} />
                            )}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Recent Jobs */}
            {mode === 'training' && (
              <div className="dt-section">
                <div className="dt-section-card" style={{ background: c.panelBg, borderColor: c.panelBorder }}>
                  <div className="dt-section-header">
                    <h3 style={{ color: c.text }}>Recent Jobs</h3>
                  </div>

                  {jobs.length === 0 ? (
                    <div className="dt-empty" style={{ color: c.textSub }}>
                      No training jobs yet
                    </div>
                  ) : (
                    <div className="dt-job-list">
                      {jobs.slice(0, 5).map((job) => (
                        <div key={job.job_id} className="dt-job-row" style={{ background: c.cardBg, borderColor: c.cardBorder }}>
                          <div className="dt-job-dot" style={{ background: getStatusColor(job.status) }} />
                          <div className="dt-job-info">
                            <div className="dt-job-name" style={{ color: c.text }}>{job.model_name}</div>
                            <div className="dt-job-meta" style={{ color: c.textSub }}>
                              {job.dataset_name} · {job.current_epoch}/{job.total_epochs} epochs
                            </div>
                          </div>
                          <div className="dt-job-time" style={{ color: c.textSub }}>
                            <Clock size={10} />
                            {formatDate(job.created_at)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
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
        .dt-page {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dt-layout {
          display: flex;
          flex: 1;
          margin-top: 70px;
          overflow: hidden;
        }

        /* Left panel — 60% */
        .dt-left {
          width: 60%;
          min-width: 400px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dt-left-header {
          padding: 0.875rem 1.25rem;
          flex-shrink: 0;
        }

        .dt-mode-switch {
          display: flex;
          align-items: center;
          gap: 0.125rem;
          padding: 0.1875rem;
          border: 1px solid;
          width: fit-content;
        }

        .dt-mode-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.4375rem 0.75rem;
          background: none;
          border: none;
          font-size: 0.775rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          letter-spacing: 0.01em;
          white-space: nowrap;
        }

        .dt-mode-btn:hover:not(.active) {
          opacity: 0.7;
        }

        .dt-left-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .dt-panel-wrap {
          padding: 0;
        }

        .dt-panel-wrap > * {
          border: none !important;
          background: transparent !important;
        }

        /* Right panel — 40% */
        .dt-right {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dt-right-scroll {
          flex: 1;
          overflow-y: auto;
          padding: 1.25rem;
        }

        .dt-section {
          margin-bottom: 1rem;
        }

        .dt-section-card {
          border: 1px solid;
          padding: 1.25rem;
        }

        .dt-section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 1rem;
        }

        .dt-section-header h3 {
          font-size: 0.8rem;
          font-weight: 600;
          margin: 0;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .dt-count {
          font-size: 0.7rem;
          font-family: monospace;
        }

        .dt-empty {
          font-size: 0.8rem;
          text-align: center;
          padding: 1.5rem 0;
        }

        /* Model list */
        .dt-model-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .dt-model-row {
          display: flex;
          align-items: center;
          gap: 0.625rem;
          padding: 0.75rem;
          border: 1px solid;
        }

        .dt-model-icon {
          width: 26px;
          height: 26px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .dt-model-info {
          flex: 1;
          min-width: 0;
        }

        .dt-model-name {
          font-size: 0.8rem;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .dt-model-badge {
          font-size: 0.55rem;
          padding: 0.1rem 0.375rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .dt-model-meta {
          font-size: 0.7rem;
          margin-top: 0.125rem;
        }

        .dt-model-actions {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          flex-shrink: 0;
        }

        .dt-use-btn {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.3rem 0.5rem;
          border: 1px solid;
          font-size: 0.7rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
        }

        .dt-use-btn:hover {
          filter: brightness(1.2);
        }

        .dt-icon-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0.35rem;
          background: transparent;
          border: none;
          cursor: pointer;
          transition: opacity 0.15s;
        }

        .dt-icon-btn:hover {
          opacity: 0.7;
        }

        .dt-icon-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        /* Job list */
        .dt-job-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .dt-job-row {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem;
          border: 1px solid;
        }

        .dt-job-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .dt-job-info {
          flex: 1;
          min-width: 0;
        }

        .dt-job-name {
          font-size: 0.8rem;
          font-weight: 500;
        }

        .dt-job-meta {
          font-size: 0.7rem;
          margin-top: 0.125rem;
        }

        .dt-job-time {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          font-size: 0.7rem;
          flex-shrink: 0;
        }

        .dt-spin {
          animation: dt-spin 1s linear infinite;
        }

        @keyframes dt-spin {
          to { transform: rotate(360deg); }
        }

        @media (max-width: 900px) {
          .dt-layout {
            flex-direction: column;
          }
          .dt-left {
            width: 100%;
            min-width: 100%;
            max-height: 50vh;
            border-right: none !important;
            border-bottom: 1px solid;
          }
        }
      `}</style>
    </div>
  )
}
