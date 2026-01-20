/**
 * Training page for custom model training.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Trash2, Clock, Download, Loader2 } from 'lucide-react'
import { TrainingPanel } from '../components/training/TrainingPanel'
import { TrainingProgress } from '../components/training/TrainingProgress'
import {
  useTrainingJobs,
  useTrainingProgress,
  useCustomModels,
} from '../hooks/useTraining'
import { NavBar } from '../components/common/NavBar'
import { apiClient, GpuType } from '../api/client'

interface TrainingPageProps {
  onShowLogin: () => void
  onShowSignup: () => void
}

export const TrainingPage: React.FC<TrainingPageProps> = ({ onShowLogin, onShowSignup }) => {
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null)
  const autoDownloadTriggeredRef = useRef(false)

  const { jobs, fetchJobs, startJob, cancelJob } = useTrainingJobs()
  const { progress, completed, completedModelName, error, reset: resetProgress } = useTrainingProgress(activeJobId)
  const { models, fetchModels, deleteModel } = useCustomModels()

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
  }, [startJob])

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
  }, [resetProgress, fetchJobs, fetchModels])

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
      default: return 'rgba(255,255,255,0.5)'
    }
  }

  // Check if there's an active running job
  const isTrainingActive = progress && progress.status === 'running'

  return (
    <div style={{ minHeight: '100vh', paddingTop: '100px' }}>
      <NavBar onShowLogin={onShowLogin} onShowSignup={onShowSignup} />

      <div
        style={{
          padding: '2rem',
          maxWidth: '1200px',
          margin: '0 auto',
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: '24px' }}>
          <h1 style={{ color: '#fff', fontSize: '28px', fontWeight: 700, margin: 0 }}>
            Custom Model Training
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px', marginTop: '8px' }}>
            Train and save your own Ozera models on custom datasets
          </p>
        </div>

        {/* Main Content */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Left Column - Training Config */}
          <div>
            <TrainingPanel
              onStartTraining={handleStartTraining}
              disabled={isTrainingActive || false}
            />
          </div>

          {/* Right Column - Progress & History */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Active Training Progress */}
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

            {/* Custom Models List */}
            <div
              style={{
                background: 'rgba(255,255,255,0.03)',
                backdropFilter: 'blur(20px)',
                borderRadius: '0',
                border: '1px solid rgba(255,255,255,0.1)',
                padding: '20px',
              }}
            >
              <h3 style={{ color: '#fff', fontSize: '14px', fontWeight: 600, margin: '0 0 16px 0' }}>
                Custom Models
              </h3>

              {models.length === 0 ? (
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
                  No custom models yet
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {models.slice(0, 1).map((model) => (
                    <div
                      key={model.model_id}
                      style={{
                        background: 'rgba(0,0,0,0.2)',
                        borderRadius: '0',
                        padding: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ color: '#fff', fontSize: '13px', fontWeight: 500 }}>
                          {model.name}
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginTop: '2px' }}>
                          {model.base_config} · Val loss: {model.val_loss.toFixed(4)}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <button
                          onClick={() => handleDownloadModel(model.model_id)}
                          disabled={downloadingModelId === model.model_id}
                          title="Download model"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            padding: '6px',
                            cursor: downloadingModelId === model.model_id ? 'not-allowed' : 'pointer',
                            color: 'rgba(255,255,255,0.4)',
                          }}
                        >
                          {downloadingModelId === model.model_id ? (
                            <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                          ) : (
                            <Download size={14} />
                          )}
                        </button>
                        <button
                          onClick={() => deleteModel(model.model_id)}
                          title="Delete model"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            padding: '6px',
                            cursor: 'pointer',
                            color: 'rgba(255,255,255,0.4)',
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Job History */}
            <div
              style={{
                background: 'rgba(255,255,255,0.03)',
                backdropFilter: 'blur(20px)',
                borderRadius: '0',
                border: '1px solid rgba(255,255,255,0.1)',
                padding: '20px',
              }}
            >
              <h3 style={{ color: '#fff', fontSize: '14px', fontWeight: 600, margin: '0 0 16px 0' }}>
                Recent Jobs
              </h3>

              {jobs.length === 0 ? (
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
                  No training jobs yet
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {jobs.slice(0, 5).map((job) => (
                    <div
                      key={job.job_id}
                      style={{
                        background: 'rgba(0,0,0,0.2)',
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
                        <div style={{ color: '#fff', fontSize: '13px', fontWeight: 500 }}>
                          {job.model_name}
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginTop: '2px' }}>
                          {job.dataset_name} · {job.current_epoch}/{job.total_epochs} epochs
                        </div>
                      </div>
                      <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={10} />
                        {formatDate(job.created_at)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

export default TrainingPage
