/**
 * Model upload panel for uploading .safetensors model files.
 */

import React, { useState, useCallback, useRef } from 'react'
import {  X,  Check, AlertCircle,  Info } from 'lucide-react'
import { useModelUpload, useCustomModelCount } from '../../hooks/useTraining'
import { useAuthStore } from '../../stores/authStore'
import { MODEL_NAME_MAX_LENGTH, isReservedModelName, sanitizeModelName } from '../../data/defaultModels'
import { Spinner } from '../common/Spinner'

interface ModelUploadPanelProps {
  onUploadComplete?: () => void
  disabled?: boolean
  onShowLogin?: () => void
}

export const ModelUploadPanel: React.FC<ModelUploadPanelProps> = ({
  onUploadComplete,
  disabled = false,
  onShowLogin,
}) => {
  const { isAuthenticated } = useAuthStore()
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [modelName, setModelName] = useState('')
  const [overwriteExisting, setOverwriteExisting] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { uploading, progress, error: uploadError, uploadedModel, uploadModel, reset } = useModelUpload()
  const { count, fetchCount } = useCustomModelCount()

  const hasExistingModel = !!(count && count.total_count > 0)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled && !uploading) {
      setIsDragging(true)
    }
  }, [disabled, uploading])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const validateFile = useCallback((file: File): string | null => {
    if (!file.name.endsWith('.safetensors')) {
      return 'Only .safetensors files are accepted'
    }

    if (file.size > 500 * 1024 * 1024) {
      return 'File too large. Maximum size is 500MB'
    }

    if (file.size < 1024) {
      return 'File too small. Model files should be at least 1KB'
    }

    return null
  }, [])

  const handleFile = useCallback((file: File) => {
    const error = validateFile(file)
    if (error) {
      setLocalError(error)
      return
    }

    setLocalError(null)
    setSelectedFile(file)
    // Auto-generate model name from file name
    if (!modelName) {
      const baseName = file.name.replace('.safetensors', '')
      setModelName(sanitizeModelName(baseName))
    }
  }, [validateFile, modelName])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (disabled || uploading) return

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [disabled, uploading, handleFile])

  const handleClick = useCallback(() => {
    if (!disabled && !uploading && !uploadedModel) {
      fileInputRef.current?.click()
    }
  }, [disabled, uploading, uploadedModel])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
    e.target.value = ''
  }, [handleFile])

  const handleUpload = useCallback(async () => {
    if (!isAuthenticated) {
      onShowLogin?.()
      return
    }
    if (!selectedFile || !modelName.trim()) {
      setLocalError('Please select a file and enter a model name')
      return
    }

    if (isReservedModelName(modelName)) {
      setLocalError('This name is reserved for a built-in model')
      return
    }

    if (hasExistingModel && !overwriteExisting) {
      setLocalError('You already have a custom model. Enable "Replace existing" to overwrite it.')
      return
    }

    setLocalError(null)

    try {
      await uploadModel(selectedFile, modelName.trim(), overwriteExisting)
      fetchCount()
      if (onUploadComplete) {
        onUploadComplete()
      }
    } catch {
      // Error is handled by the hook
    }
  }, [isAuthenticated, onShowLogin, selectedFile, modelName, hasExistingModel, overwriteExisting, uploadModel, fetchCount, onUploadComplete])

  const handleReset = useCallback(() => {
    setSelectedFile(null)
    setModelName('')
    setOverwriteExisting(false)
    setLocalError(null)
    reset()
  }, [reset])

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatParams = (params: number | null) => {
    if (!params) return 'Unknown'
    if (params < 1000000) return `${(params / 1000).toFixed(0)}K`
    if (params < 1000000000) return `${(params / 1000000).toFixed(1)}M`
    return `${(params / 1000000000).toFixed(2)}B`
  }

  const error = localError || uploadError

  return (
    <div className="model-upload oz-section">
      <h2 className="oz-title">Upload model</h2>
      <p className="oz-hint mu-intro">
        A .safetensors file of an Ozera-architecture model, up to 500MB. It becomes available on every page. One custom
        model at a time, trained or uploaded.
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept=".safetensors"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {uploadedModel ? (
        // Show upload success
        <div className="oz-note oz-note--ok">
          <Check aria-hidden="true" />
          <div>
            <div className="mu-file-name">Model uploaded: {uploadedModel.name}</div>
            <div className="oz-hint">
              {formatBytes(uploadedModel.file_size_bytes)} · {formatParams(uploadedModel.num_parameters)} parameters
            </div>
            <button onClick={handleReset} className="oz-btn oz-btn--ghost oz-btn--small mu-again">
              Upload Another
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* File drop zone */}
          {!selectedFile ? (
            <div
              onClick={handleClick}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`mu-drop${isDragging ? ' is-over' : ''}${disabled || uploading ? ' is-off' : ''}`}
            >
              {isDragging ? 'Drop model file here' : 'Drag & drop a .safetensors file or click to browse'}
            </div>
          ) : (
            // Selected file info
            <div className="mu-file">
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="mu-file-name">{selectedFile.name}</div>
                <div className="oz-hint">{formatBytes(selectedFile.size)}</div>
              </div>
              <button
                onClick={() => setSelectedFile(null)}
                disabled={uploading}
                className="oz-icon-btn"
                aria-label="Remove file"
                title="Remove file"
              >
                <X />
              </button>
            </div>
          )}

          {/* Model name input */}
          {selectedFile && (
            <div className="mu-field">
              <label className="oz-label">Model name</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(sanitizeModelName(e.target.value))}
                disabled={uploading}
                maxLength={MODEL_NAME_MAX_LENGTH}
                placeholder="e.g., my-custom-model"
                className="oz-input"
              />
            </div>
          )}

          {/* Overwrite checkbox */}
          {selectedFile && hasExistingModel && (
            <div className="mu-field">
              <label className="mu-check">
                <input
                  type="checkbox"
                  checked={overwriteExisting}
                  onChange={(e) => setOverwriteExisting(e.target.checked)}
                  disabled={uploading}
                />
                <span>Replace existing custom model</span>
              </label>
              <div className="oz-note oz-note--warn mu-note">
                <AlertCircle aria-hidden="true" />
                <span>You already have a custom model. Uploading will permanently delete your existing model.</span>
              </div>
            </div>
          )}

          {/* Upload button */}
          {selectedFile && (
            <button
              onClick={handleUpload}
              disabled={uploading || !modelName.trim() || (hasExistingModel && !overwriteExisting)}
              className={`oz-btn oz-btn--block mu-submit${uploading ? '' : ' oz-btn--arrow'}`}
            >
              {uploading ? (
                <>
                  <Spinner />
                  Uploading... {progress}%
                </>
              ) : (
                'Upload Model'
              )}
            </button>
          )}
        </>
      )}

      {/* Error display */}
      {error && (
        <div className="oz-note oz-note--error mu-note" role="alert">
          <AlertCircle aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {/* Architecture warning */}
      <div className="oz-note oz-note--warn mu-note">
        <Info aria-hidden="true" />
        <span>
          Models from other architectures (SmolLM, Gemma, Qwen, Llama, etc.) will produce incorrect outputs. For
          open-source models, use the models already in the model menus.
        </span>
      </div>

      <style>{`
        .model-upload.oz-section { border-bottom: none; }
        .model-upload .mu-intro { max-width: 46rem; margin: 0 0 1.25rem; }
        .model-upload .mu-drop {
          padding: 2.25rem 1.5rem;
          border: 1px dashed var(--oz-rule-strong);
          color: var(--oz-fg-dim);
          font-size: 0.9375rem;
          text-align: center;
          cursor: pointer;
          transition: border-color 0.2s ease, color 0.2s ease;
        }
        .model-upload .mu-drop:hover, .model-upload .mu-drop.is-over { border-color: var(--oz-mark); color: var(--oz-mark); }
        .model-upload .mu-drop.is-off { opacity: 0.5; cursor: not-allowed; }
        .model-upload .mu-file {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.7rem 0;
          border-top: 1px solid var(--oz-rule);
          border-bottom: 1px solid var(--oz-rule);
        }
        .model-upload .mu-file-name {
          overflow: hidden;
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .model-upload .mu-field { margin-top: 1.25rem; }
        .model-upload .mu-check { display: flex; align-items: center; gap: 0.6rem; font-size: 0.9375rem; cursor: pointer; }
        .model-upload .mu-note { margin-top: 1rem; }
        .model-upload .mu-submit { margin-top: 1.25rem; }
        .model-upload .mu-again { margin-top: 0.75rem; }
      `}</style>
    </div>
  )
}

export default ModelUploadPanel
