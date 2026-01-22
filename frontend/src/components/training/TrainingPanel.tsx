/**
 * Training configuration panel.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Play, AlertCircle, Database, Cpu, Settings, Zap, Download } from 'lucide-react'
import { DatasetUpload } from './DatasetUpload'
import { useTrainingEstimate, useGpuPricing } from '../../hooks/useTraining'
import type { GpuType, DatasetMetadata, GenericDatasetInfo } from '../../api/client'
import { apiClient } from '../../api/client'

// Dataset sources - "uploaded" is user's uploaded dataset, others are generic datasets from Modal volume
type DatasetSource = 'uploaded' | string

interface TrainingPanelProps {
  onStartTraining: (
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
  ) => Promise<void>
  disabled?: boolean
}

export const TrainingPanel: React.FC<TrainingPanelProps> = ({
  onStartTraining,
  disabled = false,
}) => {
  // Dataset state - tracks the selected source and the current session's uploaded dataset
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('uploaded')
  const [uploadedDataset, setUploadedDataset] = useState<DatasetMetadata | null>(null)
  const [genericDatasets, setGenericDatasets] = useState<GenericDatasetInfo[]>([])

  // Fetch generic datasets on mount
  useEffect(() => {
    apiClient.listGenericDatasets()
      .then(setGenericDatasets)
      .catch(() => {
        // Silently fail - generic datasets are optional
      })
  }, [])

  // GPU pricing
  const { pricing: gpuPricing, defaultGpu } = useGpuPricing()
  const [gpuType, setGpuType] = useState<GpuType>('a10g')

  // Training config
  const [modelConfig, setModelConfig] = useState<'nano' | 'mini'>('nano')
  const [modelName, setModelName] = useState('')
  const [epochs, setEpochs] = useState(20)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.0003)
  const [seqLen, setSeqLen] = useState(256)

  // Estimate
  const { estimate, getEstimate, clearEstimate } = useTrainingEstimate()

  // Set default GPU when pricing loads
  useEffect(() => {
    if (defaultGpu) {
      setGpuType(defaultGpu)
    }
  }, [defaultGpu])

  // Starting state
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Auto-download option
  const [autoDownload, setAutoDownload] = useState(false)

  // Overwrite confirmation modal
  const [showOverwriteModal, setShowOverwriteModal] = useState(false)

  // Get the current dataset ID based on source
  const currentDatasetId = datasetSource === 'uploaded'
    ? uploadedDataset?.dataset_id
    : datasetSource

  // Fetch estimate when config changes
  useEffect(() => {
    if (currentDatasetId) {
      getEstimate(currentDatasetId, modelConfig, epochs, batchSize, seqLen, gpuType).catch(() => {})
    } else {
      clearEstimate()
    }
  }, [currentDatasetId, modelConfig, epochs, batchSize, seqLen, gpuType, getEstimate, clearEstimate])

  const handleUpload = useCallback(async (file: File) => {
    const metadata = await apiClient.uploadDataset(file)
    setUploadedDataset(metadata)
    setDatasetSource('uploaded')
    return metadata
  }, [])

  const handleStartTraining = async () => {
    if (!currentDatasetId || !modelName.trim()) {
      setError('Please upload a dataset and enter a model name')
      return
    }

    setError(null)

    // Check if user has existing models
    try {
      const { count } = await apiClient.getCustomModelCount()
      if (count > 0) {
        setShowOverwriteModal(true)
        return
      }
    } catch {
      // If we can't check, proceed anyway (backend will handle it)
    }

    await startTrainingWithOverwrite(false)
  }

  const startTrainingWithOverwrite = async (overwrite: boolean) => {
    if (!currentDatasetId) return

    setStarting(true)
    setShowOverwriteModal(false)

    try {
      await onStartTraining(
        currentDatasetId,
        modelConfig,
        modelName.trim(),
        epochs,
        batchSize,
        learningRate,
        seqLen,
        gpuType,
        autoDownload,
        overwrite
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setStarting(false)
    }
  }

  const formatTokens = (tokens: number) => {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  // Reserved model names that cannot be used
  const RESERVED_MODEL_NAMES = ['ozera-nano', 'ozera-mini']
  const isReservedName = RESERVED_MODEL_NAMES.includes(modelName.trim().toLowerCase())

  const hasDataset = datasetSource === 'uploaded' ? !!uploadedDataset : !!datasetSource
  const isValid = hasDataset && modelName.trim().length > 0 && !isReservedName

  // Build dataset options for dropdown
  const datasetOptions = [
    { id: 'uploaded', name: 'Uploaded Dataset', tokens: uploadedDataset?.num_tokens },
    ...genericDatasets.map(d => ({ id: d.id, name: d.name, tokens: undefined })),
  ]

  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        borderRadius: '0',
        border: '1px solid rgba(255,255,255,0.1)',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Cpu size={20} color="rgba(255,255,255,0.7)" />
        <h3 style={{ color: '#fff', fontSize: '16px', fontWeight: 600, margin: 0 }}>
          Train Custom Model
        </h3>
      </div>

      {/* Dataset Section */}
      <div>
        <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
          <Database size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
          Dataset
        </label>

        {/* Dataset source selector */}
        <select
          value={datasetSource}
          onChange={(e) => setDatasetSource(e.target.value)}
          disabled={disabled || starting}
          style={{
            width: '100%',
            padding: '10px 12px',
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: '0',
            color: '#fff',
            fontSize: '13px',
            cursor: 'pointer',
            marginBottom: '12px',
          }}
        >
          {datasetOptions.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.name}
              {opt.tokens ? ` (${formatTokens(opt.tokens)} tokens)` : ''}
            </option>
          ))}
        </select>

        {/* Show upload component when "Uploaded Dataset" is selected */}
        {datasetSource === 'uploaded' && (
          <>
            {uploadedDataset ? (
              <div
                style={{
                  background: 'rgba(34, 197, 94, 0.1)',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  padding: '10px 12px',
                  marginBottom: '12px',
                  fontSize: '12px',
                }}
              >
                <div style={{ color: '#22c55e', fontWeight: 500 }}>
                  {uploadedDataset.name}
                </div>
                <div style={{ color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                  {formatTokens(uploadedDataset.num_tokens)} tokens • {(uploadedDataset.size_bytes / 1024).toFixed(1)} KB
                </div>
              </div>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '12px', marginBottom: '12px' }}>
                No dataset uploaded yet
              </div>
            )}
            <DatasetUpload onUpload={handleUpload} disabled={disabled || starting} />
          </>
        )}
      </div>

      {/* Model Configuration */}
      <div>
        <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
          <Settings size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
          Model Configuration
        </label>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {/* Architecture */}
          <div>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Architecture
            </label>
            <select
              value={modelConfig}
              onChange={(e) => setModelConfig(e.target.value as 'nano' | 'mini')}
              disabled={disabled || starting}
              style={{
                width: '100%',
                padding: '8px 10px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '12px',
              }}
            >
              <option value="nano">Nano (~4M params)</option>
              <option value="mini">Mini (~51M params)</option>
            </select>
          </div>

          {/* Model Name */}
          <div>
            <label style={{ display: 'block', color: isReservedName ? '#ef4444' : 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Model Name
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
              placeholder="my-custom-model"
              disabled={disabled || starting}
              style={{
                width: '100%',
                padding: '8px 10px',
                background: 'rgba(0,0,0,0.3)',
                border: isReservedName ? '1px solid rgba(239, 68, 68, 0.5)' : '1px solid rgba(255,255,255,0.15)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '12px',
                boxSizing: 'border-box',
              }}
            />
            {isReservedName && (
              <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px' }}>
                This name is reserved for default Ozera models
              </div>
            )}
          </div>
        </div>
      </div>

      {/* GPU Selection */}
      <div>
        <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
          <Zap size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
          GPU Selection
        </label>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {gpuPricing.map((gpu) => {
            const isSelected = gpuType === gpu.gpu_type
            const tokensPerSec = modelConfig === 'nano' ? gpu.nano_tokens_per_sec : gpu.mini_tokens_per_sec
            return (
              <button
                key={gpu.gpu_type}
                type="button"
                onClick={() => setGpuType(gpu.gpu_type)}
                disabled={disabled || starting}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(0,0,0,0.2)',
                  border: isSelected ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '0',
                  cursor: disabled || starting ? 'not-allowed' : 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: '#fff', fontSize: '13px', fontWeight: 500 }}>
                      {gpu.display_name}
                      {gpu.gpu_type === 'a10g' && (
                        <span style={{
                          marginLeft: '8px',
                          fontSize: '10px',
                          color: '#22c55e',
                          background: 'rgba(34, 197, 94, 0.15)',
                          padding: '2px 6px',
                          borderRadius: '2px',
                        }}>
                          Recommended
                        </span>
                      )}
                    </div>
                    <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginTop: '2px' }}>
                      {gpu.description}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: '#fff', fontSize: '13px', fontWeight: 600 }}>
                      ${gpu.rate_per_hour.toFixed(2)}/hr
                    </div>
                    <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px', marginTop: '2px' }}>
                      ~{(tokensPerSec / 1000).toFixed(0)}K tok/s
                    </div>
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Hyperparameters */}
      <div>
        <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
          Hyperparameters
        </label>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {/* Epochs */}
          <div>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Epochs: {epochs}
            </label>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              disabled={disabled || starting}
              style={{ width: '100%', accentColor: '#fff' }}
            />
          </div>

          {/* Batch Size */}
          <div>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Batch Size
            </label>
            <select
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              disabled={disabled || starting}
              style={{
                width: '100%',
                padding: '8px 10px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '12px',
              }}
            >
              <option value={8}>8</option>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Learning Rate: {learningRate.toExponential(0)}
            </label>
            <input
              type="range"
              min={-5}
              max={-2}
              step={0.5}
              value={Math.log10(learningRate)}
              onChange={(e) => setLearningRate(Math.pow(10, parseFloat(e.target.value)))}
              disabled={disabled || starting}
              style={{ width: '100%', accentColor: '#fff' }}
            />
          </div>

          {/* Sequence Length */}
          <div>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginBottom: '4px' }}>
              Sequence Length
            </label>
            <select
              value={seqLen}
              onChange={(e) => setSeqLen(parseInt(e.target.value))}
              disabled={disabled || starting}
              style={{
                width: '100%',
                padding: '8px 10px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '12px',
              }}
            >
              <option value={128}>128</option>
              <option value={256}>256</option>
              <option value={512}>512</option>
            </select>
          </div>
        </div>
      </div>

      {/* Cost Estimate */}
      {estimate && (
        <div
          style={{
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '0',
            padding: '12px 16px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '11px', marginBottom: '2px' }}>
                Estimated Time
              </div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
                {estimate.estimated_minutes < 60
                  ? `${estimate.estimated_minutes.toFixed(0)} min`
                  : `${(estimate.estimated_minutes / 60).toFixed(1)} hr`}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '11px', marginBottom: '2px' }}>
                Estimated Cost
              </div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>
                ${estimate.estimated_cost_usd.toFixed(2)}
              </div>
            </div>
          </div>
          {estimate.warning && (
            <div style={{ color: '#fbbf24', fontSize: '11px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <AlertCircle size={12} />
              {estimate.warning}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '0',
            padding: '10px 12px',
            color: '#ef4444',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      {/* Auto-download option */}
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '12px',
          background: 'rgba(0,0,0,0.2)',
          border: '1px solid rgba(255,255,255,0.1)',
          cursor: disabled || starting ? 'not-allowed' : 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={autoDownload}
          onChange={(e) => setAutoDownload(e.target.checked)}
          disabled={disabled || starting}
          style={{
            width: '16px',
            height: '16px',
            accentColor: '#3b82f6',
            cursor: disabled || starting ? 'not-allowed' : 'pointer',
          }}
        />
        <div>
          <div style={{ color: '#fff', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Download size={14} />
            Download model when complete
          </div>
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', marginTop: '2px' }}>
            Automatically download model weights as a .zip file
          </div>
        </div>
      </label>

      {/* Start Button */}
      <button
        onClick={handleStartTraining}
        disabled={disabled || starting || !isValid}
        style={{
          width: '100%',
          padding: '12px 16px',
          background: isValid && !disabled && !starting
            ? 'linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05))'
            : 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: '0',
          color: isValid && !disabled && !starting ? '#fff' : 'rgba(255,255,255,0.4)',
          fontSize: '14px',
          fontWeight: 600,
          cursor: isValid && !disabled && !starting ? 'pointer' : 'not-allowed',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'all 0.2s ease',
        }}
      >
        {starting ? (
          <>
            <div
              style={{
                width: '16px',
                height: '16px',
                border: '2px solid rgba(255,255,255,0.2)',
                borderTopColor: '#fff',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }}
            />
            Starting...
          </>
        ) : (
          <>
            <Play size={16} />
            Start Training
          </>
        )}
      </button>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Overwrite Confirmation Modal */}
      {showOverwriteModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowOverwriteModal(false)}
        >
          <div
            style={{
              background: '#1a1a1a',
              border: '1px solid rgba(255,255,255,0.15)',
              padding: '24px',
              maxWidth: '400px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ color: '#fff', fontSize: '16px', fontWeight: 600, margin: '0 0 12px 0' }}>
              Replace Existing Model?
            </h3>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '13px', margin: '0 0 20px 0', lineHeight: 1.5 }}>
              You already have a custom model. Ozera currently limits users to 1 custom model.
              Training a new model will permanently delete your existing model.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowOverwriteModal(false)}
                style={{
                  padding: '10px 16px',
                  background: 'transparent',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: 'rgba(255,255,255,0.7)',
                  fontSize: '13px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => startTrainingWithOverwrite(true)}
                style={{
                  padding: '10px 16px',
                  background: 'rgba(239, 68, 68, 0.2)',
                  border: '1px solid rgba(239, 68, 68, 0.5)',
                  color: '#ef4444',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Replace & Train
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingPanel
