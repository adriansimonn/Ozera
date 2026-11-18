/**
 * Dataset upload component with drag-and-drop support.
 */

import React, { useState, useCallback, useRef } from 'react'
import {  X,  Check } from 'lucide-react'
import type { DatasetMetadata } from '../../api/client'
import { useThemeColors } from '../../hooks/useTheme'
import { Spinner } from '../common/Spinner'

interface DatasetUploadProps {
  onUpload: (file: File) => Promise<DatasetMetadata>
  disabled?: boolean
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onUpload,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<DatasetMetadata | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const tc = useThemeColors()

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

  const handleFile = useCallback(async (file: File) => {
    // Validate file type
    if (!file.name.endsWith('.txt')) {
      setError('Only .txt files are accepted')
      return
    }

    // Validate file size (client-side check)
    if (file.size > 50 * 1024 * 1024) {
      setError('File too large. Maximum size is 50MB')
      return
    }

    if (file.size < 10 * 1024) {
      setError('File too small. Minimum size is 10KB')
      return
    }

    setError(null)
    setUploading(true)
    setUploadedFile(null)

    try {
      const metadata = await onUpload(file)
      if (metadata) {
        setUploadedFile(metadata)
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed'
      // Don't show error for user-cancelled uploads
      if (msg !== 'Upload cancelled') {
        setError(msg)
      }
    } finally {
      setUploading(false)
    }
  }, [onUpload])

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
    if (!disabled && !uploading) {
      fileInputRef.current?.click()
    }
  }, [disabled, uploading])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
    // Reset input so the same file can be selected again
    e.target.value = ''
  }, [handleFile])

  const handleClear = useCallback(() => {
    setUploadedFile(null)
    setError(null)
  }, [])

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatTokens = (tokens: number) => {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  return (
    <div style={{ width: '100%' }}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {uploadedFile ? (
        // Show uploaded file info
        <div
          style={{
            background: 'transparent',
            border: `1px solid ${tc.ok}`,
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <div
            style={{
              width: '32px',
              height: '32px',
              background: 'transparent',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Check size={16} color={tc.ok} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                color: tc.text,
                fontSize: '15px',
                fontWeight: 500,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {uploadedFile.name}
            </div>
            <div style={{ color: tc.textSub, fontSize: '13.5px', marginTop: '2px' }}>
              {formatBytes(uploadedFile.size_bytes)} · {formatTokens(uploadedFile.num_tokens)} tokens
            </div>
          </div>
          <button
            onClick={handleClear}
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px',
              cursor: 'pointer',
              color: tc.textSub,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        // Show upload area
        <div
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${isDragging ? tc.textSub : tc.borderHover}`,
            padding: '24px',
            textAlign: 'center',
            cursor: disabled || uploading ? 'not-allowed' : 'pointer',
            background: isDragging ? tc.surfaceHover : 'transparent',
            transition: 'all 0.2s ease',
            opacity: disabled ? 0.5 : 1,
          }}
        >
          {uploading ? (
            <>
              <div style={{ color: tc.textMid, fontSize: '15px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                <Spinner />
                Uploading...
              </div>
            </>
          ) : (
            <>
              <div style={{ color: tc.textMid, fontSize: '15px', marginBottom: '4px' }}>
                {isDragging ? 'Drop file here' : 'Drag & drop a .txt file or click to browse'}
              </div>
              <div style={{ color: tc.textMuted, fontSize: '13.5px' }}>
                10KB - 50MB, UTF-8 encoded text
              </div>
            </>
          )}
        </div>
      )}

      {error && (
        <div
          style={{
            marginTop: '8px',
            padding: '8px 12px',
            background: 'transparent',
            border: `1px solid ${tc.danger}`,
            color: tc.danger,
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

export default DatasetUpload
