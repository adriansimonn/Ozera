/**
 * Training progress display with real-time updates.
 */

import React, { useMemo, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, CheckCircle, AlertCircle, Clock, Activity, Download, Play } from 'lucide-react'
import type { TrainingProgress as TrainingProgressType } from '../../api/client'

interface TrainingProgressProps {
  progress: TrainingProgressType | null
  completed: boolean
  completedModelName: string | null
  error: string | null
  onCancel?: () => void
  onDismiss?: () => void
  onDownload?: () => void
  downloading?: boolean
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  progress,
  completed,
  completedModelName,
  error,
  onCancel,
  onDismiss,
  onDownload,
  downloading = false,
}) => {
  const navigate = useNavigate()

  // Live elapsed time counter
  const [liveElapsedSeconds, setLiveElapsedSeconds] = useState(0)

  // Confirmation modal state
  const [showCancelConfirm, setShowCancelConfirm] = useState(false)

  // Update live elapsed time every second when training is active
  useEffect(() => {
    if (!progress || completed || error) {
      return
    }

    // Initialize with current elapsed time
    setLiveElapsedSeconds(progress.elapsed_seconds)

    // Update every second
    const interval = setInterval(() => {
      setLiveElapsedSeconds(prev => prev + 1)
    }, 1000)

    return () => clearInterval(interval)
  }, [progress?.job_id, completed, error]) // Reset when job changes

  // Sync with backend updates
  useEffect(() => {
    if (progress) {
      setLiveElapsedSeconds(progress.elapsed_seconds)
    }
  }, [progress?.elapsed_seconds])

  // Calculate progress percentage
  const progressPercent = useMemo(() => {
    if (!progress) return 0
    if (progress.total_epochs === 0) return 0
    return Math.round((progress.current_epoch / progress.total_epochs) * 100)
  }, [progress])

  // Format time
  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${mins}m`
  }

  // Loss history for mini chart (simplified - just show current values)
  const lossDisplay = useMemo(() => {
    if (!progress?.train_loss) return null
    return {
      train: progress.train_loss.toFixed(4),
      val: progress.val_loss?.toFixed(4) || '-',
      trainPpl: progress.train_ppl?.toFixed(2) || '-',
      valPpl: progress.val_ppl?.toFixed(2) || '-',
    }
  }, [progress])

  if (completed) {
    return (
      <div
        style={{
          background: 'rgba(34, 197, 94, 0.1)',
          border: '1px solid rgba(34, 197, 94, 0.3)',
          borderRadius: '0',
          padding: '20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <CheckCircle size={24} color="#22c55e" />
            <div>
              <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 600 }}>
                Training Complete
              </div>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
                Model ready: {completedModelName}
              </div>
            </div>
          </div>
          {onDismiss && (
            <button
              onClick={onDismiss}
              style={{
                background: 'transparent',
                border: 'none',
                padding: '4px',
                cursor: 'pointer',
                color: 'rgba(255,255,255,0.5)',
              }}
            >
              <X size={18} />
            </button>
          )}
        </div>
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '12px', marginBottom: '12px' }}>
          Your custom model has been trained and is ready for text generation and visualization.
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => navigate('/')}
            style={{
              flex: 1,
              padding: '10px 16px',
              background: 'rgba(34, 197, 94, 0.2)',
              border: '1px solid rgba(34, 197, 94, 0.4)',
              borderRadius: '0',
              color: '#22c55e',
              fontSize: '13px',
              fontWeight: 500,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
          >
            <Play size={14} />
            Use Model
          </button>
          {onDownload && (
            <button
              onClick={onDownload}
              disabled={downloading}
              style={{
                flex: 1,
                padding: '10px 16px',
                background: downloading ? 'rgba(59, 130, 246, 0.1)' : 'rgba(59, 130, 246, 0.2)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                borderRadius: '0',
                color: '#3b82f6',
                fontSize: '13px',
                fontWeight: 500,
                cursor: downloading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              {downloading ? (
                <>
                  <div
                    style={{
                      width: '14px',
                      height: '14px',
                      border: '2px solid rgba(59, 130, 246, 0.3)',
                      borderTopColor: '#3b82f6',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                    }}
                  />
                  Downloading...
                </>
              ) : (
                <>
                  <Download size={14} />
                  Download
                </>
              )}
            </button>
          )}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '0',
          padding: '20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <AlertCircle size={24} color="#ef4444" />
            <div>
              <div style={{ color: '#ef4444', fontSize: '16px', fontWeight: 600 }}>
                Training Failed
              </div>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
                {error}
              </div>
            </div>
          </div>
          {onDismiss && (
            <button
              onClick={onDismiss}
              style={{
                background: 'transparent',
                border: 'none',
                padding: '4px',
                cursor: 'pointer',
                color: 'rgba(255,255,255,0.5)',
              }}
            >
              <X size={18} />
            </button>
          )}
        </div>
      </div>
    )
  }

  if (!progress) {
    return (
      <div
        style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '0',
          padding: '40px 20px',
          textAlign: 'center',
        }}
      >
        <Activity size={32} color="rgba(255,255,255,0.3)" style={{ marginBottom: '12px' }} />
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '13px' }}>
          No active training job
        </div>
      </div>
    )
  }

  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        borderRadius: '0',
        border: '1px solid rgba(255,255,255,0.1)',
        padding: '20px',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: '#22c55e',
              animation: 'pulse 2s ease-in-out infinite',
            }}
          />
          <span style={{ color: '#fff', fontSize: '14px', fontWeight: 600 }}>
            Training in Progress
          </span>
        </div>
        {onCancel && (
          <button
            onClick={() => setShowCancelConfirm(true)}
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '0',
              padding: '6px 12px',
              color: '#ef4444',
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <X size={12} />
            Cancel
          </button>
        )}
      </div>

      {/* Progress Bar */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
          <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
            Epoch {progress.current_epoch} / {progress.total_epochs}
          </span>
          <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
            {progressPercent}%
          </span>
        </div>
        <div
          style={{
            height: '8px',
            background: 'rgba(255,255,255,0.1)',
            borderRadius: '0',
            overflow: 'hidden',
          }}
        >
          <div
            className="animated-gradient-bar"
            style={{
              height: '100%',
              width: `${progressPercent}%`,
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6)',
              backgroundSize: '200% 100%',
              borderRadius: '0',
              transition: 'width 0.5s ease',
              animation: 'gradientShift 3s ease infinite',
            }}
          />
        </div>
      </div>

      {/* Stats Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
        {/* Time */}
        <div
          style={{
            background: 'rgba(0,0,0,0.2)',
            borderRadius: '0',
            padding: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <Clock size={12} color="rgba(255,255,255,0.5)" />
            <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '11px' }}>Time</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div>
              <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Elapsed</div>
              <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
                {formatTime(liveElapsedSeconds)}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Remaining</div>
              <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
                {formatTime(Math.max(0, progress.estimated_remaining_seconds - (liveElapsedSeconds - progress.elapsed_seconds)))}
              </div>
            </div>
          </div>
        </div>

        {/* Loss */}
        {lossDisplay && (
          <div
            style={{
              background: 'rgba(0,0,0,0.2)',
              borderRadius: '0',
              padding: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <Activity size={12} color="rgba(255,255,255,0.5)" />
              <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '11px' }}>Loss</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Train</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
                  {lossDisplay.train}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Val</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
                  {lossDisplay.val}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Perplexity */}
      {lossDisplay && (
        <div
          style={{
            background: 'rgba(0,0,0,0.2)',
            borderRadius: '0',
            padding: '12px',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <div style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Train PPL</div>
            <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
              {lossDisplay.trainPpl}
            </div>
          </div>
          <div
            style={{
              width: '1px',
              height: '30px',
              background: 'rgba(255,255,255,0.1)',
            }}
          />
          <div style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Val PPL</div>
            <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
              {lossDisplay.valPpl}
            </div>
          </div>
        </div>
      )}

      {/* Cancel Confirmation Modal */}
      {showCancelConfirm && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => setShowCancelConfirm(false)}
        >
          <div
            style={{
              background: 'rgba(20, 20, 20, 0.95)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '0',
              padding: '24px',
              maxWidth: '400px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <AlertCircle size={24} color="#ef4444" />
                <h3 style={{ color: '#fff', fontSize: '18px', fontWeight: 600, margin: 0 }}>
                  Cancel Training?
                </h3>
              </div>
              <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '14px', lineHeight: '1.5', margin: 0 }}>
                Are you sure you want to cancel this training job? All progress will be lost and cannot be recovered.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowCancelConfirm(false)}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '0',
                  padding: '8px 16px',
                  color: 'rgba(255,255,255,0.8)',
                  fontSize: '13px',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Keep Training
              </button>
              <button
                onClick={() => {
                  setShowCancelConfirm(false)
                  onCancel?.()
                }}
                style={{
                  background: '#ef4444',
                  border: '1px solid #dc2626',
                  borderRadius: '0',
                  padding: '8px 16px',
                  color: '#fff',
                  fontSize: '13px',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Cancel Training
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </div>
  )
}

export default TrainingProgress
