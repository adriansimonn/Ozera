/**
 * Training progress display with real-time updates.
 */

import React, { useMemo } from 'react'
import { X, CheckCircle, AlertCircle, Clock, Activity } from 'lucide-react'
import type { TrainingProgress as TrainingProgressType } from '../../api/client'

interface TrainingProgressProps {
  progress: TrainingProgressType | null
  completed: boolean
  completedModelName: string | null
  error: string | null
  onCancel?: () => void
  onDismiss?: () => void
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  progress,
  completed,
  completedModelName,
  error,
  onCancel,
  onDismiss,
}) => {
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
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '12px' }}>
          Your custom model is now available in the model selector.
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
            onClick={onCancel}
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
            style={{
              height: '100%',
              width: `${progressPercent}%`,
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
              borderRadius: '0',
              transition: 'width 0.5s ease',
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
                {formatTime(progress.elapsed_seconds)}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Remaining</div>
              <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>
                {formatTime(progress.estimated_remaining_seconds)}
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
            justifyContent: 'space-around',
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Train PPL</div>
            <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
              {lossDisplay.trainPpl}
            </div>
          </div>
          <div
            style={{
              width: '1px',
              background: 'rgba(255,255,255,0.1)',
            }}
          />
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px' }}>Val PPL</div>
            <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
              {lossDisplay.valPpl}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}

export default TrainingProgress
