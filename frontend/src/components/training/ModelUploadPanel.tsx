/**
 * Model upload panel for uploading .safetensors model files.
 */

import React, { useState, useCallback, useRef } from 'react'
import { Upload, X, Box, Check, AlertCircle, Loader2, Info } from 'lucide-react'
import { useModelUpload, useCustomModelCount } from '../../hooks/useTraining'

interface ModelUploadPanelProps {
  onUploadComplete?: () => void
  disabled?: boolean
}

export const ModelUploadPanel: React.FC<ModelUploadPanelProps> = ({
  onUploadComplete,
  disabled = false,
}) => {
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
      setModelName(baseName.slice(0, 64))
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
    if (!selectedFile || !modelName.trim()) {
      setLocalError('Please select a file and enter a model name')
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
  }, [selectedFile, modelName, hasExistingModel, overwriteExisting, uploadModel, fetchCount, onUploadComplete])

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
        Upload Model
      </h3>

      <input
        ref={fileInputRef}
        type="file"
        accept=".safetensors"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {uploadedModel ? (
        // Show upload success
        <div
          style={{
            background: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '0',
            padding: '16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '0',
                background: 'rgba(34, 197, 94, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Check size={20} color="#22c55e" />
            </div>
            <div>
              <div style={{ color: '#22c55e', fontSize: '14px', fontWeight: 600 }}>
                Model Uploaded Successfully
              </div>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginTop: '2px' }}>
                {uploadedModel.name}
              </div>
            </div>
          </div>
          <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '12px', marginBottom: '12px' }}>
            {formatBytes(uploadedModel.file_size_bytes)} · {formatParams(uploadedModel.num_parameters)} parameters
          </div>
          <button
            onClick={handleReset}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '0',
              padding: '8px 16px',
              color: '#fff',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            Upload Another
          </button>
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
              style={{
                border: `2px dashed ${isDragging ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.2)'}`,
                borderRadius: '8px',
                padding: '32px',
                textAlign: 'center',
                cursor: disabled || uploading ? 'not-allowed' : 'pointer',
                background: isDragging ? 'rgba(255,255,255,0.05)' : 'transparent',
                transition: 'all 0.2s ease',
                opacity: disabled ? 0.5 : 1,
              }}
            >
              <div
                style={{
                  width: '48px',
                  height: '48px',
                  margin: '0 auto 12px',
                  borderRadius: '0',
                  background: 'rgba(255,255,255,0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {isDragging ? <Box size={24} color="#fff" /> : <Upload size={24} color="rgba(255,255,255,0.6)" />}
              </div>
              <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '14px', marginBottom: '4px' }}>
                {isDragging ? 'Drop model file here' : 'Drag & drop a .safetensors file'}
              </div>
              <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '14px', marginBottom: '8px' }}>
                or click to browse
              </div>
              <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '12px' }}>
                Maximum file size: 500MB
              </div>
            </div>
          ) : (
            // Selected file info
            <div
              style={{
                background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: '0',
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '16px',
              }}
            >
              <div
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '0',
                  background: 'rgba(59, 130, 246, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Box size={18} color="#3b82f6" />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    color: '#fff',
                    fontSize: '13px',
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {selectedFile.name}
                </div>
                <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '11px', marginTop: '2px' }}>
                  {formatBytes(selectedFile.size)}
                </div>
              </div>
              <button
                onClick={() => setSelectedFile(null)}
                disabled={uploading}
                style={{
                  background: 'transparent',
                  border: 'none',
                  padding: '4px',
                  cursor: uploading ? 'not-allowed' : 'pointer',
                  color: 'rgba(255,255,255,0.5)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <X size={16} />
              </button>
            </div>
          )}

          {/* Model name input */}
          {selectedFile && (
            <div style={{ marginBottom: '16px' }}>
              <label
                style={{
                  display: 'block',
                  color: 'rgba(255,255,255,0.7)',
                  fontSize: '12px',
                  marginBottom: '6px',
                }}
              >
                Model Name
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={uploading}
                maxLength={64}
                placeholder="e.g., my-custom-model"
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '0',
                  padding: '10px 12px',
                  color: '#fff',
                  fontSize: '13px',
                  outline: 'none',
                }}
              />
            </div>
          )}

          {/* Overwrite checkbox */}
          {selectedFile && hasExistingModel && (
            <div style={{ marginBottom: '16px' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={overwriteExisting}
                  onChange={(e) => setOverwriteExisting(e.target.checked)}
                  disabled={uploading}
                  style={{ accentColor: '#3b82f6' }}
                />
                <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: '12px' }}>
                  Replace existing custom model
                </span>
              </label>
              <div
                style={{
                  marginTop: '6px',
                  padding: '8px 12px',
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.3)',
                  borderRadius: '0',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                }}
              >
                <AlertCircle size={14} color="#f59e0b" style={{ flexShrink: 0, marginTop: '1px' }} />
                <span style={{ color: '#f59e0b', fontSize: '11px' }}>
                  You already have a custom model. Uploading will permanently delete your existing model.
                </span>
              </div>
            </div>
          )}

          {/* Upload button */}
          {selectedFile && (
            <button
              onClick={handleUpload}
              disabled={uploading || !modelName.trim() || (hasExistingModel && !overwriteExisting)}
              style={{
                width: '100%',
                background: uploading || !modelName.trim() || (hasExistingModel && !overwriteExisting)
                  ? 'rgba(59, 130, 246, 0.3)'
                  : 'rgba(59, 130, 246, 0.8)',
                border: '1px solid rgba(59, 130, 246, 0.5)',
                borderRadius: '0',
                padding: '12px',
                color: '#fff',
                fontSize: '14px',
                fontWeight: 500,
                cursor: uploading || !modelName.trim() || (hasExistingModel && !overwriteExisting)
                  ? 'not-allowed'
                  : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              {uploading ? (
                <>
                  <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                  Uploading... {progress}%
                </>
              ) : (
                <>
                  <Upload size={16} />
                  Upload Model
                </>
              )}
            </button>
          )}
        </>
      )}

      {/* Error display */}
      {error && (
        <div
          style={{
            marginTop: '12px',
            padding: '10px 12px',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '0',
            color: '#ef4444',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '8px',
          }}
        >
          <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '1px' }} />
          {error}
        </div>
      )}

      {/* Architecture warning */}
      <div
        style={{
          marginTop: '16px',
          padding: '10px 12px',
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '4px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '10px',
        }}
      >
        <Info size={14} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '11px' }}>
          <div style={{ color: '#f59e0b', fontWeight: 500, marginBottom: '4px' }}>
            Architecture Compatibility
          </div>
          <div style={{ color: 'rgba(251, 191, 36, 0.9)' }}>
            Only models trained with the Ozera architecture are supported. Models from other architectures (SmolLM, Gemma, Qwen, Llama, etc.) will produce incorrect outputs. For open-source models, use the pre-registered models from the dropdown menu instead.
          </div>
        </div>
      </div>

      {/* Info section */}
      <div
        style={{
          marginTop: '12px',
          padding: '12px',
          background: 'rgba(0,0,0,0.2)',
          borderRadius: '0',
          fontSize: '11px',
          color: 'rgba(255,255,255,0.5)',
        }}
      >
        <div style={{ fontWeight: 500, marginBottom: '6px', color: 'rgba(255,255,255,0.7)' }}>
          Supported format
        </div>
        <div>
          Upload a .safetensors model file. The model will be available for text generation
          and visualization on the Generate page.
        </div>
        <div style={{ marginTop: '8px', color: 'rgba(255,255,255,0.4)' }}>
          Note: You can only have one custom model at a time (trained or uploaded).
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

export default ModelUploadPanel
