/**
 * Training progress display with real-time updates.
 */

import React, { useMemo, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, CheckCircle, AlertCircle, Clock, Activity, Download, Play } from 'lucide-react'
import type { TrainingProgress as TrainingProgressType } from '../../api/client'
import { useThemeColors } from '../../hooks/useTheme'
import { EmptyState } from '../common/EmptyState'

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
  const tc = useThemeColors()

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
          background: 'transparent',
          border: `1px solid ${tc.ok}`,
          padding: '20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <CheckCircle size={24} color={tc.ok} />
            <div>
              <div style={{ color: tc.ok, fontSize: '17px', fontWeight: 500 }}>
                Training Complete
              </div>
              <div style={{ color: tc.textSub, fontSize: '14px' }}>
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
                color: tc.textSub,
              }}
            >
              <X size={18} />
            </button>
          )}
        </div>
        <div style={{ color: tc.textSub, fontSize: '14px', marginBottom: '12px' }}>
          Your custom model has been trained and is ready for text generation and visualization.
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => navigate('/generate')}
            style={{
              flex: 1,
              padding: '10px 16px',
              background: 'transparent',
              border: `1px solid ${tc.ok}`,
              color: tc.ok,
              fontSize: '15px',
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
                background: downloading ? 'transparent' : tc.surfaceActive,
                border: `1px solid ${tc.borderHover}`,
                color: tc.mark,
                fontSize: '15px',
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
                      border: `1px solid ${tc.borderHover}`,
                      borderTopColor: tc.mark,
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
          background: 'transparent',
          border: `1px solid ${tc.danger}`,
          padding: '20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <AlertCircle size={24} color={tc.danger} />
            <div>
              <div style={{ color: tc.danger, fontSize: '17px', fontWeight: 500 }}>
                Training Failed
              </div>
              <div style={{ color: tc.textSub, fontSize: '14px' }}>
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
                color: tc.textSub,
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
      <EmptyState glyph="training">No active training job</EmptyState>
    )
  }

  return (
    <div
      style={{
        background: tc.surface,
        border: `1px solid ${tc.border}`,
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
              background: tc.ok,
              animation: 'pulse 2s ease-in-out infinite',
            }}
          />
          <span style={{ color: tc.text, fontSize: '15.5px', fontWeight: 500 }}>
            Training in Progress
          </span>
        </div>
        {onCancel && (
          <button
            onClick={() => setShowCancelConfirm(true)}
            style={{
              background: 'transparent',
              border: `1px solid ${tc.danger}`,
              padding: '6px 12px',
              color: tc.danger,
              fontSize: '14px',
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
          <span style={{ color: tc.textSub, fontSize: '14px' }}>
            Epoch {progress.current_epoch} / {progress.total_epochs}
          </span>
          <span style={{ color: tc.textSub, fontSize: '14px' }}>
            {progressPercent}%
          </span>
        </div>
        <div
          style={{
            height: '8px',
            background: tc.border,
            overflow: 'hidden',
          }}
        >
          <div
            className="animated-gradient-bar"
            style={{
              height: '100%',
              width: `${progressPercent}%`,
              background: tc.mark,
              backgroundSize: '200% 100%',
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
            background: tc.deepBg,
            padding: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <Clock size={12} color={tc.textSub} />
            <span style={{ color: tc.textSub, fontSize: '13.5px' }}>Time</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div>
              <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Elapsed</div>
              <div style={{ color: tc.text, fontSize: '15.5px', fontWeight: 500 }}>
                {formatTime(liveElapsedSeconds)}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Remaining</div>
              <div style={{ color: tc.text, fontSize: '15.5px', fontWeight: 500 }}>
                {formatTime(Math.max(0, progress.estimated_remaining_seconds - (liveElapsedSeconds - progress.elapsed_seconds)))}
              </div>
            </div>
          </div>
        </div>

        {/* Loss */}
        {lossDisplay && (
          <div
            style={{
              background: tc.deepBg,
              padding: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <Activity size={12} color={tc.textSub} />
              <span style={{ color: tc.textSub, fontSize: '13.5px' }}>Loss</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Train</div>
                <div style={{ color: tc.text, fontSize: '15.5px', fontWeight: 500 }}>
                  {lossDisplay.train}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Val</div>
                <div style={{ color: tc.text, fontSize: '15.5px', fontWeight: 500 }}>
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
            background: tc.deepBg,
            padding: '12px',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <div style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Train PPL</div>
            <div style={{ color: tc.text, fontSize: '17px', fontWeight: 500 }}>
              {lossDisplay.trainPpl}
            </div>
          </div>
          <div
            style={{
              width: '1px',
              height: '30px',
              background: tc.border,
            }}
          />
          <div style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ color: tc.textMuted, fontSize: '12.5px' }}>Val PPL</div>
            <div style={{ color: tc.text, fontSize: '17px', fontWeight: 500 }}>
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
            background: 'color-mix(in srgb, var(--oz-bg) 84%, transparent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => setShowCancelConfirm(false)}
        >
          <div
            style={{
              background: tc.bg,
              border: `1px solid ${tc.border}`,
              padding: '24px',
              maxWidth: '400px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <AlertCircle size={24} color={tc.danger} />
                <h3 style={{ color: tc.text, fontSize: '22px', fontWeight: 500, margin: 0 }}>
                  Cancel Training?
                </h3>
              </div>
              <p style={{ color: tc.textSub, fontSize: '15.5px', lineHeight: '1.5', margin: 0 }}>
                Are you sure you want to cancel this training job? All progress will be lost and cannot be recovered.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowCancelConfirm(false)}
                style={{
                  background: tc.surface,
                  border: `1px solid ${tc.border}`,
                  padding: '8px 16px',
                  color: tc.textSub,
                  fontSize: '15px',
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
                  background: tc.danger,
                  border: `1px solid ${tc.danger}`,
                  padding: '8px 16px',
                  color: tc.text,
                  fontSize: '15px',
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
