/**
 * Training configuration panel.
 */

import React, { useState, useEffect, useCallback } from 'react'
import {  AlertCircle } from 'lucide-react'
import { DatasetUpload } from './DatasetUpload'
import { useTrainingEstimate, useGpuPricing } from '../../hooks/useTraining'
import type { GpuType, DatasetMetadata, GenericDatasetInfo } from '../../api/client'
import { apiClient } from '../../api/client'
import { useAuthStore } from '../../stores/authStore'
import { useThemeColors } from '../../hooks/useTheme'
import { Dropdown } from '../common/Dropdown'
import { Spinner } from '../common/Spinner'
import { isReservedModelName, sanitizeModelName } from '../../data/defaultModels'

// Dataset sources - "uploaded" is user's uploaded dataset, "upload-new" shows upload UI, others are generic datasets
type DatasetSource = 'uploaded' | 'upload-new' | string

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
  onShowLogin?: () => void
}

export const TrainingPanel: React.FC<TrainingPanelProps> = ({
  onStartTraining,
  disabled = false,
  onShowLogin,
}) => {
  const { isAuthenticated } = useAuthStore()
  const tc = useThemeColors()
  // Dataset state - tracks the selected source and the current session's uploaded dataset
  const [datasetSource, setDatasetSource] = useState<DatasetSource>('uploaded')
  const [uploadedDataset, setUploadedDataset] = useState<DatasetMetadata | null>(null)
  const [genericDatasets, setGenericDatasets] = useState<GenericDatasetInfo[]>([])
  const [showDatasetOverwriteWarning, setShowDatasetOverwriteWarning] = useState(false)
  const [pendingUploadFile, setPendingUploadFile] = useState<File | null>(null)

  // Fetch user's current dataset and generic datasets on mount
  useEffect(() => {
    if (isAuthenticated) {
      apiClient.getCurrentDataset()
        .then((dataset) => {
          if (dataset) {
            setUploadedDataset(dataset)
            setDatasetSource('uploaded')
          }
        })
        .catch(() => {})
    }

    apiClient.listGenericDatasets()
      .then(setGenericDatasets)
      .catch(() => {})
  }, [isAuthenticated])

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
    : datasetSource === 'upload-new'
      ? undefined
      : datasetSource

  // Fetch estimate when config changes
  useEffect(() => {
    if (currentDatasetId) {
      getEstimate(currentDatasetId, modelConfig, epochs, batchSize, seqLen, gpuType).catch(() => {})
    } else {
      clearEstimate()
    }
  }, [currentDatasetId, modelConfig, epochs, batchSize, seqLen, gpuType, getEstimate, clearEstimate])

  // Ref to hold the resolve/reject for the pending overwrite confirmation
  const overwriteResolveRef = React.useRef<{ resolve: (v: any) => void; reject: (e: any) => void } | null>(null)

  const doUpload = useCallback(async (file: File) => {
    const metadata = await apiClient.uploadDataset(file)
    setUploadedDataset(metadata)
    setDatasetSource('uploaded')
    return metadata
  }, [])

  const handleUpload = useCallback(async (file: File) => {
    if (!isAuthenticated) {
      onShowLogin?.()
      return undefined as any
    }
    // If user already has a dataset, show overwrite warning and wait for confirmation
    if (uploadedDataset) {
      setPendingUploadFile(file)
      setShowDatasetOverwriteWarning(true)
      return new Promise<DatasetMetadata>((resolve, reject) => {
        overwriteResolveRef.current = { resolve, reject }
      })
    }
    return doUpload(file)
  }, [isAuthenticated, onShowLogin, uploadedDataset, doUpload])

  const handleConfirmOverwrite = useCallback(async () => {
    setShowDatasetOverwriteWarning(false)
    if (pendingUploadFile) {
      const file = pendingUploadFile
      setPendingUploadFile(null)
      try {
        const metadata = await doUpload(file)
        overwriteResolveRef.current?.resolve(metadata)
      } catch (err) {
        overwriteResolveRef.current?.reject(err)
      }
      overwriteResolveRef.current = null
    }
  }, [pendingUploadFile, doUpload])

  const handleCancelOverwrite = useCallback(() => {
    setShowDatasetOverwriteWarning(false)
    setPendingUploadFile(null)
    overwriteResolveRef.current?.reject(new Error('Upload cancelled'))
    overwriteResolveRef.current = null
  }, [])

  const handleStartTraining = async () => {
    if (!currentDatasetId || !modelName.trim()) {
      setError('Please upload a dataset and enter a model name')
      return
    }

    setError(null)

    // Check if user has existing models
    try {
      const { total_count } = await apiClient.getCustomModelCount()
      if (total_count > 0) {
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

  const isReservedName = isReservedModelName(modelName)

  const hasDataset = datasetSource === 'uploaded' ? !!uploadedDataset : datasetSource !== 'upload-new'
  const isValid = hasDataset && modelName.trim().length > 0 && !isReservedName

  // Build dataset options for dropdown
  const datasetOptions = uploadedDataset
    ? [
        { id: 'uploaded', name: uploadedDataset.name.replace(/\.txt$/, ''), tokens: uploadedDataset.num_tokens },
        { id: 'upload-new', name: 'Upload Dataset', tokens: undefined },
        ...genericDatasets.map(d => ({ id: d.id, name: d.name, tokens: undefined, description: d.description })),
      ]
    : [
        { id: 'uploaded', name: 'Upload Dataset', tokens: undefined },
        ...genericDatasets.map(d => ({ id: d.id, name: d.name, tokens: undefined, description: d.description })),
      ]

  return (
    <div className="training-panel oz-steps">
      {/* Dataset Section */}
      <div className="oz-section">
        <h2 className="oz-title">Dataset</h2>

        {/* Dataset source selector */}
        <div style={{ marginBottom: '12px' }}>
          <Dropdown
            value={datasetSource}
            onChange={(v) => setDatasetSource(v)}
            disabled={disabled || starting}
            options={datasetOptions.map((opt) => ({
              value: opt.id,
              label: `${opt.name}${opt.tokens ? ` (${formatTokens(opt.tokens)} tokens)` : ''}`,
            }))}
          />
        </div>

        {/* Show upload component when upload-related source is selected */}
        {(datasetSource === 'uploaded' || datasetSource === 'upload-new') && (
          <>
            {datasetSource === 'uploaded' && uploadedDataset ? (
              <div
                style={{
                  background: 'transparent',
                  border: `1px solid ${tc.ok}`,
                  padding: '10px 12px',
                  marginBottom: '12px',
                  fontSize: '14px',
                }}
              >
                <div style={{ color: tc.text, fontWeight: 500 }}>
                  {uploadedDataset.name}
                </div>
                <div style={{ color: tc.textSub, marginTop: '4px' }}>
                  {formatTokens(uploadedDataset.num_tokens)} tokens • {(uploadedDataset.size_bytes / 1024).toFixed(1)} KB
                </div>
              </div>
            ) : datasetSource === 'uploaded' && !uploadedDataset ? (
              <div style={{ color: tc.textMuted, fontSize: '14px', marginBottom: '12px' }}>
                No dataset uploaded yet
              </div>
            ) : null}
            <DatasetUpload onUpload={handleUpload} disabled={disabled || starting} />
          </>
        )}
      </div>

      {/* Model Configuration */}
      <div className="oz-section">
        <h2 className="oz-title">Model configuration</h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {/* Architecture */}
          <div>
            <label style={{ display: 'block', color: tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
              Architecture
            </label>
            <Dropdown
              value={modelConfig}
              onChange={(v) => setModelConfig(v as 'nano' | 'mini')}
              disabled={disabled || starting}
              options={[
                { value: 'nano', label: 'Nano (~4M params)' },
                { value: 'mini', label: 'Mini (~51M params)' },
              ]}
            />
          </div>

          {/* Model Name */}
          <div>
            <label style={{ display: 'block', color: isReservedName ? tc.danger : tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
              Model name
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(sanitizeModelName(e.target.value))}
              placeholder="my-custom-model"
              disabled={disabled || starting}
              className="oz-input"
              style={isReservedName ? { borderColor: tc.danger } : undefined}
            />
            {isReservedName && (
              <div style={{ color: tc.danger, fontSize: '13.5px', marginTop: '4px' }}>
                This name is reserved for a built-in model
              </div>
            )}
          </div>
        </div>
      </div>

      {/* GPU Selection */}
      <div className="oz-section">
        <h2 className="oz-title">GPU selection</h2>

        <div className="tp-gpus">
          {gpuPricing.map((gpu) => {
            const isSelected = gpuType === gpu.gpu_type
            const tokensPerSec = modelConfig === 'nano' ? gpu.nano_tokens_per_sec : gpu.mini_tokens_per_sec
            return (
              <button
                key={gpu.gpu_type}
                type="button"
                onClick={() => setGpuType(gpu.gpu_type)}
                disabled={disabled || starting}
                className={`tp-gpu${isSelected ? ' is-active' : ''}`}
                aria-pressed={isSelected}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: tc.text, fontSize: '15px', fontWeight: 500 }}>
                      {gpu.display_name}
                      {gpu.gpu_type === 'a10g' && (
                        <span className="oz-tag" style={{ marginLeft: '8px' }}>Recommended</span>
                      )}
                    </div>
                    <div style={{ color: tc.textMuted, fontSize: '13.5px', marginTop: '2px' }}>
                      {gpu.description}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: tc.text, fontSize: '15px', fontWeight: 500 }}>
                      ${gpu.rate_per_hour.toFixed(2)}/hr
                    </div>
                    <div style={{ color: tc.textMuted, fontSize: '12.5px', marginTop: '2px' }}>
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
      <div className="oz-section">
        <h2 className="oz-title">Hyperparameters</h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {/* Epochs */}
          <div>
            <label style={{ display: 'block', color: tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
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
              style={{ width: '100%' }}
            />
          </div>

          {/* Batch Size */}
          <div>
            <label style={{ display: 'block', color: tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
              Batch size
            </label>
            <Dropdown
              value={String(batchSize)}
              onChange={(v) => setBatchSize(parseInt(v))}
              disabled={disabled || starting}
              options={[
                { value: '8', label: '8' },
                { value: '16', label: '16' },
                { value: '32', label: '32' },
                { value: '64', label: '64' },
              ]}
            />
          </div>

          {/* Learning Rate */}
          <div>
            <label style={{ display: 'block', color: tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
              Learning rate: {learningRate.toExponential(0)}
            </label>
            <input
              type="range"
              min={-5}
              max={-2}
              step={0.5}
              value={Math.log10(learningRate)}
              onChange={(e) => setLearningRate(Math.pow(10, parseFloat(e.target.value)))}
              disabled={disabled || starting}
              style={{ width: '100%' }}
            />
          </div>

          {/* Sequence Length */}
          <div>
            <label style={{ display: 'block', color: tc.textMuted, fontSize: '13.5px', marginBottom: '4px' }}>
              Sequence length
            </label>
            <Dropdown
              value={String(seqLen)}
              onChange={(v) => setSeqLen(parseInt(v))}
              disabled={disabled || starting}
              options={[
                { value: '128', label: '128' },
                { value: '256', label: '256' },
                { value: '512', label: '512' },
              ]}
            />
          </div>
        </div>
      </div>

      <div className="oz-section tp-run">
        {/* Cost Estimate */}
        {estimate && (
          <div className="tp-estimate">
            <div>
              <div className="tp-estimate-label">Estimated time</div>
              <div className="tp-estimate-value">
                {estimate.estimated_minutes < 60
                  ? `${estimate.estimated_minutes.toFixed(0)} min`
                  : `${(estimate.estimated_minutes / 60).toFixed(1)} hr`}
              </div>
            </div>
            <div>
              <div className="tp-estimate-label">Estimated cost</div>
              <div className="tp-estimate-value">${estimate.estimated_cost_usd.toFixed(2)}</div>
            </div>
          </div>
        )}
        {estimate?.warning && (
          <div className="oz-note oz-note--warn">
            <AlertCircle aria-hidden="true" />
            <span>{estimate.warning}</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="oz-note oz-note--error" role="alert">
            <AlertCircle aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}

        {/* Auto-download option */}
        <label className={`tp-option${disabled || starting ? ' is-off' : ''}`}>
          <input
            type="checkbox"
            checked={autoDownload}
            onChange={(e) => setAutoDownload(e.target.checked)}
            disabled={disabled || starting}
          />
          <span>
            <span className="tp-option-name">Download model when complete</span>
            <span className="oz-hint">Automatically download model weights as a .zip file</span>
          </span>
        </label>

        {/* Start Button */}
        <button
          onClick={handleStartTraining}
          disabled={disabled || starting || !isValid}
          className={`oz-btn oz-btn--block${starting ? '' : ' oz-btn--arrow'}`}
        >
          {starting ? (
            <>
              <Spinner />
              Starting...
            </>
          ) : (
            'Start Training'
          )}
        </button>
      </div>

      <style>{`
        .training-panel .oz-section:last-of-type { border-bottom: none; }
        .training-panel .tp-gpus { display: flex; flex-direction: column; }
        /* The GPUs: a ruled list, the chosen one marked */
        .training-panel .tp-gpu {
          position: relative;
          width: 100%;
          padding: 0.8rem 0;
          border-top: 1px solid var(--oz-rule);
          color: var(--oz-fg-dim);
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }
        .training-panel .tp-gpu:last-child { border-bottom: 1px solid var(--oz-rule); }
        .training-panel .tp-gpu:hover:not(:disabled), .training-panel .tp-gpu.is-active { color: var(--oz-fg); }
        .training-panel .tp-gpu:disabled { cursor: not-allowed; }
        .training-panel .tp-gpu.is-active::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 100%;
          height: 1px;
          background: var(--oz-mark);
        }
        .training-panel .tp-gpu div { color: inherit !important; }
        .training-panel .tp-run { display: flex; flex-direction: column; gap: 1rem; }
        .training-panel .tp-estimate { display: flex; gap: 3rem; }
        .training-panel .tp-estimate-label { color: var(--oz-fg-dim); font-size: 0.85rem; font-weight: 500; }
        .training-panel .tp-estimate-value {
          margin-top: 0.15rem;
          font-size: 2rem;
          font-weight: 500;
          letter-spacing: -0.035em;
          line-height: 1.1;
          font-variant-numeric: tabular-nums;
        }
        .training-panel .tp-option { display: flex; align-items: flex-start; gap: 0.65rem; cursor: pointer; }
        .training-panel .tp-option.is-off { cursor: not-allowed; }
        .training-panel .tp-option input { margin-top: 0.2rem; }
        .training-panel .tp-option > span { display: flex; flex-direction: column; gap: 0.1rem; }
        .training-panel .tp-option-name { font-size: 0.9375rem; font-weight: 500; }
      `}</style>

      {/* Dataset Overwrite Warning Modal */}
      {showDatasetOverwriteWarning && (
        <div className="oz-overlay" onClick={handleCancelOverwrite}>
          <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <h3 className="oz-modal__title">Replace existing dataset?</h3>
            <p className="oz-modal__text">
              You already have a dataset uploaded: <strong>{uploadedDataset?.name}</strong>. Uploading a new dataset will
              permanently replace it.
            </p>
            <div className="oz-modal__actions">
              <button className="oz-btn oz-btn--ghost" onClick={handleCancelOverwrite}>
                Cancel
              </button>
              <button className="oz-btn oz-btn--danger" onClick={handleConfirmOverwrite}>
                Replace Dataset
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Overwrite Confirmation Modal */}
      {showOverwriteModal && (
        <div className="oz-overlay" onClick={() => setShowOverwriteModal(false)}>
          <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <h3 className="oz-modal__title">Replace existing model?</h3>
            <p className="oz-modal__text">
              You already have a custom model. Ozera currently limits users to 1 custom model.
              Your existing model will be permanently deleted once the new model finishes
              training. If training fails or is cancelled, your existing model is kept.
            </p>
            <div className="oz-modal__actions">
              <button className="oz-btn oz-btn--ghost" onClick={() => setShowOverwriteModal(false)}>
                Cancel
              </button>
              <button className="oz-btn oz-btn--danger" onClick={() => startTrainingWithOverwrite(true)}>
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
