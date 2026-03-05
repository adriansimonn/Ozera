/**
 * Custom Models page - supports both training and model upload.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, Clock, Download, Loader2, Play, Upload } from 'lucide-react'
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
import { NavBar, CustomModelsMode } from '../components/common/NavBar'
import { apiClient, GpuType } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import { useThemeColors } from '../hooks/useTheme'

export const TrainingPage: React.FC = () => {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const [mode, setMode] = useState<CustomModelsMode>('training')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null)
  const [deletingUploadedModelId, setDeletingUploadedModelId] = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<{ modelId: string; modelName: string; type: 'trained' | 'uploaded' } | null>(null)
  const autoDownloadTriggeredRef = useRef(false)

  const { jobs, fetchJobs, startJob, cancelJob } = useTrainingJobs()
  const { progress, completed, completedModelName, error, reset: resetProgress } = useTrainingProgress(activeJobId)
  const { models, fetchModels, deleteModel } = useCustomModels()
  const { models: uploadedModels, fetchModels: fetchUploadedModels, deleteModel: deleteUploadedModel } = useUploadedModels()
  const tc = useThemeColors()

  // Handle auto-download when training completes
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
    // Reset auto-download trigger for new job
    autoDownloadTriggeredRef.current = false
    setAutoDownloadEnabled(autoDownload)

    const response = await startJob(
      datasetId,
      modelConfig,
      modelName,
      epochs,
      batchSize,
      learningRate,
      seqLen,
      gpuType,
      overwriteExisting
    )
    setActiveJobId(response.job_id)
  }, [isAuthenticated, navigate, startJob])

  const handleCancel = useCallback(async () => {
    if (activeJobId) {
      await cancelJob(activeJobId)
    }
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
    setDeletingUploadedModelId(modelId)
    try {
      if (type === 'trained') {
        await deleteModel(modelId)
      } else {
        await deleteUploadedModel(modelId)
      }
    } catch (err) {
      console.error('Failed to delete model:', err)
    } finally {
      setDeletingUploadedModelId(null)
      setDeleteConfirm(null)
    }
  }, [deleteConfirm, deleteModel, deleteUploadedModel])

  const handleUploadComplete = useCallback(() => {
    fetchModels()
    fetchUploadedModels()
  }, [fetchModels, fetchUploadedModels])

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#22c55e'
      case 'running': return '#3b82f6'
      case 'failed': return '#ef4444'
      case 'cancelled': return '#f59e0b'
      default: return tc.textSub
    }
  }

  // Check if there's an active running job
  const isTrainingActive = progress && progress.status === 'running'

  // Combine trained and uploaded models for the list
  const allCustomModels = [
    ...models.map(m => ({ ...m, type: 'trained' as const })),
    ...uploadedModels.map(m => ({ ...m, type: 'uploaded' as const })),
  ]

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
    <div style={{ minHeight: '125vh', paddingTop: '100px' }}>
      <NavBar
        showCustomModelsToggle={true}
        customModelsMode={mode}
        onCustomModelsModeChange={setMode}
      />

      <div
        style={{
          padding: '2rem',
          maxWidth: '1200px',
          margin: '0 auto',
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: '24px' }}>
          <h1 style={{ color: tc.text, fontSize: '28px', fontWeight: 700, margin: 0 }}>
            {mode === 'training' ? 'Custom Model Training' : 'Upload Model'}
          </h1>
          <p style={{ color: tc.textSub, fontSize: '14px', marginTop: '8px' }}>
            {mode === 'training'
              ? 'Train and save your own Ozera models on custom datasets'
              : 'Upload a .safetensors model file to use for generation'
            }
          </p>
        </div>

        {/* Main Content */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Left Column - Training Config or Upload Panel */}
          <div>
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

          {/* Right Column - Progress & History */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Active Training Progress (only show in training mode or when there's active progress) */}
            {(mode === 'training' || progress) && (
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
            )}

            {/* Custom Models List (shows both trained and uploaded) */}
            <div
              style={{
                background: tc.surface,
                backdropFilter: 'blur(20px)',
                borderRadius: '0',
                border: `1px solid ${tc.border}`,
                padding: '20px',
              }}
            >
              <h3 style={{ color: tc.text, fontSize: '14px', fontWeight: 600, margin: '0 0 16px 0' }}>
                Custom Models
              </h3>

              {allCustomModels.length === 0 ? (
                <div style={{ color: tc.textMuted, fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
                  No custom models yet
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {allCustomModels.map((model) => (
                    <div
                      key={model.model_id}
                      style={{
                        background: tc.deepBg,
                        borderRadius: '0',
                        padding: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div
                          style={{
                            width: '28px',
                            height: '28px',
                            borderRadius: '0',
                            background: model.type === 'trained' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          {model.type === 'trained' ? (
                            <Play size={12} color="#22c55e" />
                          ) : (
                            <Upload size={12} color="#3b82f6" />
                          )}
                        </div>
                        <div>
                          <div style={{ color: tc.text, fontSize: '13px', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {model.name}
                            <span
                              style={{
                                fontSize: '9px',
                                padding: '2px 6px',
                                borderRadius: '0',
                                background: model.type === 'trained' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                                color: model.type === 'trained' ? '#22c55e' : '#3b82f6',
                                textTransform: 'uppercase',
                                fontWeight: 600,
                              }}
                            >
                              {model.type}
                            </span>
                          </div>
                          <div style={{ color: tc.textMuted, fontSize: '11px', marginTop: '2px' }}>
                            {model.type === 'trained' ? (
                              <>{model.base_config} · Val loss: {model.val_loss.toFixed(4)}</>
                            ) : (
                              <>{formatBytes(model.file_size_bytes)}{formatParams(model.num_parameters) && ` · ${formatParams(model.num_parameters)}`}</>
                            )}
                          </div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <button
                          onClick={() => navigate('/')}
                          title="Use model for generation"
                          style={{
                            background: 'rgba(34, 197, 94, 0.15)',
                            border: '1px solid rgba(34, 197, 94, 0.3)',
                            borderRadius: '0',
                            padding: '4px 8px',
                            cursor: 'pointer',
                            color: '#22c55e',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '11px',
                            fontWeight: 500,
                          }}
                        >
                          <Play size={12} />
                          Use
                        </button>
                        {model.type === 'trained' && (
                          <button
                            onClick={() => handleDownloadModel(model.model_id)}
                            disabled={downloadingModelId === model.model_id}
                            title="Download model"
                            style={{
                              background: 'transparent',
                              border: 'none',
                              padding: '6px',
                              cursor: downloadingModelId === model.model_id ? 'not-allowed' : 'pointer',
                              color: tc.textMuted,
                            }}
                          >
                            {downloadingModelId === model.model_id ? (
                              <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                            ) : (
                              <Download size={14} />
                            )}
                          </button>
                        )}
                        <button
                          onClick={() => setDeleteConfirm({ modelId: model.model_id, modelName: model.name, type: model.type })}
                          disabled={deletingUploadedModelId === model.model_id}
                          title="Delete model"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            padding: '6px',
                            cursor: deletingUploadedModelId === model.model_id ? 'not-allowed' : 'pointer',
                            color: tc.textMuted,
                          }}
                        >
                          {deletingUploadedModelId === model.model_id ? (
                            <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                          ) : (
                            <Trash2 size={14} />
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Job History (only show in training mode) */}
            {mode === 'training' && (
              <div
                style={{
                  background: tc.surface,
                  backdropFilter: 'blur(20px)',
                  borderRadius: '0',
                  border: `1px solid ${tc.border}`,
                  padding: '20px',
                }}
              >
                <h3 style={{ color: tc.text, fontSize: '14px', fontWeight: 600, margin: '0 0 16px 0' }}>
                  Recent Jobs
                </h3>

                {jobs.length === 0 ? (
                  <div style={{ color: tc.textMuted, fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
                    No training jobs yet
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {jobs.slice(0, 5).map((job) => (
                      <div
                        key={job.job_id}
                        style={{
                          background: tc.deepBg,
                          borderRadius: '0',
                          padding: '12px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px',
                        }}
                      >
                        <div
                          style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: getStatusColor(job.status),
                          }}
                        />
                        <div style={{ flex: 1 }}>
                          <div style={{ color: tc.text, fontSize: '13px', fontWeight: 500 }}>
                            {job.model_name}
                          </div>
                          <div style={{ color: tc.textMuted, fontSize: '11px', marginTop: '2px' }}>
                            {job.dataset_name} · {job.current_epoch}/{job.total_epochs} epochs
                          </div>
                        </div>
                        <div style={{ color: tc.textMuted, fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Clock size={10} />
                          {formatDate(job.created_at)}
                        </div>
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
        isLoading={!!deletingUploadedModelId}
      />

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

export default TrainingPage
