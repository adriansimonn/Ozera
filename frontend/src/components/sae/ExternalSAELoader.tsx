/**
 * External SAE Loader component.
 * Provides UI for loading SAEs from HuggingFace, Gemma Scope, and user uploads.
 */

import { useState, useCallback, useEffect } from 'react'
import {
  
  
  
  
  AlertCircle,
  AlertTriangle,
  Check,
  Trash2 } from 'lucide-react'
import {
  saeClient,
  type ExternalSAEInfo,
  type ExternalSAESourcesResponse,
} from '../../api/client'
import { EmptyState } from '../common/EmptyState'
import { Spinner } from '../common/Spinner'

interface ExternalSAELoaderProps {
  loadedSAEs: ExternalSAEInfo[]
  onSAELoaded: () => void
  onSAEDeleted: () => void
}

type LoaderTab = 'huggingface' | 'upload' | 'loaded'

// The SAE service's upload limit
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024

// Common HuggingFace SAE repositories for quick access
const SUGGESTED_REPOS = [
  {
    repo_id: 'EleutherAI/sae-SmolLM2-135M-64x',
    label: 'EleutherAI SmolLM2-135M',
    description: 'TopK SAEs for SmolLM2-135M MLP layers (64x expansion)',
    activation: 'TopK',
  },
  {
    repo_id: 'google/gemma-scope-2-1b-pt',
    label: 'Gemma Scope 2 1B',
    description: 'JumpReLU SAEs for Gemma 3 1B (residual, MLP, attention)',
    activation: 'JumpReLU',
  },
  {
    repo_id: 'google/gemma-scope-2-1b-it',
    label: 'Gemma Scope 2 1B Instruct',
    description: 'JumpReLU SAEs for Gemma 3 1B Instruct (residual, MLP, attention)',
    activation: 'JumpReLU',
  },
]

export function ExternalSAELoader({
  loadedSAEs,
  onSAELoaded,
  onSAEDeleted,
}: ExternalSAELoaderProps) {
  const [tab, setTab] = useState<LoaderTab>('huggingface')

  // HuggingFace loading state
  const [repoId, setRepoId] = useState('')
  const [browsing, setBrowsing] = useState(false)
  const [browseResult, setBrowseResult] = useState<ExternalSAESourcesResponse | null>(null)
  const [browseError, setBrowseError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingHookpoint, setLoadingHookpoint] = useState<string | null>(null)
  const [loadResult, setLoadResult] = useState<{ sae_id: string; display_name: string } | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadName, setUploadName] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [existingUpload, setExistingUpload] = useState<{ name: string; sae_id: string } | null>(null)
  const [showReplaceWarning, setShowReplaceWarning] = useState(false)

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null)

  // Check if user already has an uploaded SAE
  useEffect(() => {
    saeClient.hasUploadedSAE().then(result => {
      if (result.has_upload && result.upload_name && result.upload_sae_id) {
        setExistingUpload({ name: result.upload_name, sae_id: result.upload_sae_id })
      } else {
        setExistingUpload(null)
      }
    }).catch(() => {})
  }, [loadedSAEs]) // Re-check when SAE list changes

  const handleBrowseRepo = useCallback(async () => {
    if (!repoId.trim()) return

    setBrowsing(true)
    setBrowseError(null)
    setBrowseResult(null)
    setLoadResult(null)

    try {
      const result = await saeClient.listExternalSAESources(repoId.trim())
      if (result.error) {
        setBrowseError(result.error)
      } else {
        setBrowseResult(result)
      }
    } catch (err) {
      setBrowseError(err instanceof Error ? err.message : 'Failed to browse repository')
    } finally {
      setBrowsing(false)
    }
  }, [repoId])

  const handleLoadHookpoint = useCallback(async (hookpoint: string, customName?: string) => {
    if (!repoId.trim()) return

    setLoading(true)
    setLoadingHookpoint(hookpoint)
    setLoadError(null)
    setLoadResult(null)

    try {
      const result = await saeClient.loadExternalSAE({
        repo_id: repoId.trim(),
        hookpoint: hookpoint || undefined,
        name: customName,
      })

      if (result.error) {
        setLoadError(result.error)
      } else {
        setLoadResult({
          sae_id: result.sae_id || '',
          display_name: result.display_name || '',
        })
        onSAELoaded()
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to load SAE')
    } finally {
      setLoading(false)
      setLoadingHookpoint(null)
    }
  }, [repoId, onSAELoaded])

  const handleQuickLoad = useCallback(async (quickRepoId: string) => {
    setRepoId(quickRepoId)
    setBrowsing(true)
    setBrowseError(null)
    setBrowseResult(null)

    try {
      const result = await saeClient.listExternalSAESources(quickRepoId)
      if (result.error) {
        setBrowseError(result.error)
      } else {
        setBrowseResult(result)
      }
    } catch (err) {
      setBrowseError(err instanceof Error ? err.message : 'Failed to browse repository')
    } finally {
      setBrowsing(false)
    }
  }, [])

  const doUpload = useCallback(async () => {
    if (!uploadFile || !uploadName.trim()) return

    setUploading(true)
    setUploadError(null)
    setShowReplaceWarning(false)

    try {
      if (uploadFile.size > MAX_UPLOAD_BYTES) {
        throw new Error('File too large. Maximum 500MB.')
      }

      // Read file as base64 (the browser encodes a data URL natively: "data:<type>;base64,<data>")
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'))
        reader.readAsDataURL(uploadFile)
      })
      const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1)

      const result = await saeClient.uploadSAE({
        name: uploadName.trim(),
        weights_base64: base64,
      })

      if (result.error) {
        setUploadError(result.error)
      } else {
        setUploadFile(null)
        setUploadName('')
        setExistingUpload(null)
        onSAELoaded()
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [uploadFile, uploadName, onSAELoaded])

  const handleUpload = useCallback(() => {
    if (!uploadFile || !uploadName.trim()) return

    if (existingUpload) {
      setShowReplaceWarning(true)
    } else {
      doUpload()
    }
  }, [uploadFile, uploadName, existingUpload, doUpload])

  const handleDelete = useCallback(async (saeId: string) => {
    setDeletingId(saeId)
    try {
      const result = await saeClient.deleteExternalSAE(saeId)
      if (!result.error) {
        onSAEDeleted()
      }
    } catch {
      // Silently handle
    } finally {
      setDeletingId(null)
    }
  }, [onSAEDeleted])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleBrowseRepo()
    }
  }

  return (
    <div className="external-sae-loader">
      {/* Tab Navigation */}
      <div className="oz-tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'huggingface'} onClick={() => setTab('huggingface')} className={`oz-tab ${tab === 'huggingface' ? 'active' : ''}`}>
          Repository
        </button>
        <button role="tab" aria-selected={tab === 'upload'} onClick={() => setTab('upload')} className={`oz-tab ${tab === 'upload' ? 'active' : ''}`}>
          Upload Safetensors
        </button>
        <button role="tab" aria-selected={tab === 'loaded'} onClick={() => setTab('loaded')} className={`oz-tab ${tab === 'loaded' ? 'active' : ''}`}>
          Loaded ({loadedSAEs.length})
        </button>
      </div>

      {/* HuggingFace Tab */}
      {tab === 'huggingface' && (
        <div className="tab-content">
          {/* Suggested Repos */}
          <div className="oz-label">Quick load</div>
          <div className="oz-chips">
            {SUGGESTED_REPOS.map(repo => (
              <button
                key={repo.repo_id}
                onClick={() => handleQuickLoad(repo.repo_id)}
                className="oz-chip suggested-repo-btn"
                title={repo.description}
              >
                <span>{repo.label}</span>
                <span className="suggested-repo-kind">{repo.activation}</span>
              </button>
            ))}
          </div>

          {/* Repo Input */}
          <div className="loader-field">
            <div className="oz-label">Or enter repository ID</div>
            <div className="loader-row">
              <input
                type="text"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="org/repo-name (e.g. EleutherAI/sae-SmolLM2-135M-64x)"
                className="oz-input"
                disabled={browsing || loading}
              />
              <button
                onClick={handleBrowseRepo}
                disabled={browsing || loading || !repoId.trim()}
                className={`oz-btn${browsing ? '' : ' oz-btn--arrow'}`}
              >
                {browsing && <Spinner />}
                Browse
              </button>
            </div>
          </div>

          {/* Browse Error */}
          {browseError && (
            <div className="oz-note oz-note--error loader-note">
              <AlertCircle aria-hidden="true" />
              <span>{browseError}</span>
            </div>
          )}

          {/* Browse Results */}
          {browseResult && browseResult.available.length > 0 && (
            <div className="loader-field">
              {/* Check if first item has an error */}
              {browseResult.available[0].error ? (
                <div className="oz-note oz-note--error">
                  <AlertCircle aria-hidden="true" />
                  <div>
                    <div className="font-medium">Failed to browse repository:</div>
                    <div>{browseResult.available[0].error}</div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="oz-label">Available SAEs ({browseResult.count})</div>
                  <div className="hookpoint-list">
                    {browseResult.available.map((item, idx) => (
                      <div key={idx} className="hookpoint-item">
                        <div className="hookpoint-info">
                          <span className="hookpoint-name">{item.hookpoint || 'root'}</span>
                          <div className="hookpoint-meta">
                            {item.d_in && <span>d_in: {item.d_in}</span>}
                            {item.num_latents && <span>features: {item.num_latents.toLocaleString()}</span>}
                            {item.k && <span>k: {item.k}</span>}
                            {item.layer !== undefined && <span>layer: {item.layer}</span>}
                            {item.width && <span>width: {item.width.toLocaleString()}</span>}
                            {item.l0 && <span>L0: {item.l0}</span>}
                            {item.site && <span>{item.site}</span>}
                          </div>
                        </div>
                        <button
                          onClick={() => handleLoadHookpoint(item.hookpoint)}
                          disabled={loading}
                          className="oz-btn oz-btn--ghost oz-btn--small"
                        >
                          {loading && loadingHookpoint === item.hookpoint && <Spinner />}
                          Load
                        </button>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {browseResult && browseResult.available.length === 0 && (
            <p className="oz-hint loader-note">No SAE hookpoints found in this repository.</p>
          )}

          {/* Load Result */}
          {loadResult && (
            <div className="oz-note oz-note--ok loader-note">
              <Check aria-hidden="true" />
              <span>Loaded: {loadResult.display_name}</span>
            </div>
          )}

          {loadError && (
            <div className="oz-note oz-note--error loader-note">
              <AlertCircle aria-hidden="true" />
              <span>{loadError}</span>
            </div>
          )}
        </div>
      )}

      {/* Upload Tab */}
      {tab === 'upload' && (
        <div className="tab-content">
          <p className="oz-hint loader-intro">
            A .safetensors file of SAE weights, in Ozera, EleutherAI/sparsify or Gemma Scope format. One uploaded SAE per account.
          </p>

          {existingUpload && (
            <div className="oz-note oz-note--warn loader-note">
              <AlertTriangle aria-hidden="true" />
              <span>
                You already have an uploaded SAE: <span className="font-medium">{existingUpload.name}</span>.
                Uploading a new one will replace it.
              </span>
            </div>
          )}

          <div className="loader-field">
            <label className="oz-label">Display name</label>
            <input
              type="text"
              value={uploadName}
              onChange={(e) => setUploadName(e.target.value)}
              placeholder="my-custom-sae"
              className="oz-input"
              disabled={uploading}
            />
          </div>

          <div className="loader-field">
            <label className="oz-label">Safetensors file</label>
            <label className="upload-zone">
              <input
                type="file"
                accept=".safetensors"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="hidden"
                disabled={uploading}
              />
              {uploadFile ? (
                <span>
                  {uploadFile.name}
                  <span className="upload-size"> ({(uploadFile.size / 1024 / 1024).toFixed(1)} MB)</span>
                </span>
              ) : (
                <span>Click to select .safetensors file (max 500MB)</span>
              )}
            </label>
          </div>

          {/* Replace confirmation warning */}
          {showReplaceWarning && existingUpload && (
            <div className="oz-note oz-note--warn loader-note">
              <AlertTriangle aria-hidden="true" />
              <div>
                <div>
                  This will replace your existing upload <span className="font-medium">"{existingUpload.name}"</span>. Continue?
                </div>
                <div className="loader-confirm">
                  <button onClick={doUpload} className="oz-btn oz-btn--small">
                    Replace & Upload
                  </button>
                  <button onClick={() => setShowReplaceWarning(false)} className="oz-btn oz-btn--ghost oz-btn--small">
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {!showReplaceWarning && (
            <button
              onClick={handleUpload}
              disabled={uploading || !uploadFile || !uploadName.trim()}
              className={`oz-btn oz-btn--block loader-submit${uploading ? '' : ' oz-btn--arrow'}`}
            >
              {uploading ? (
                <>
                  <Spinner />
                  Uploading...
                </>
              ) : existingUpload ? (
                'Replace & Upload SAE'
              ) : (
                'Upload SAE'
              )}
            </button>
          )}

          {uploadError && (
            <div className="oz-note oz-note--error loader-note">
              <AlertCircle aria-hidden="true" />
              <span>{uploadError}</span>
            </div>
          )}
        </div>
      )}

      {/* Loaded SAEs Tab */}
      {tab === 'loaded' && (
        <div className="tab-content">
          {loadedSAEs.length === 0 ? (
            <EmptyState glyph="sae">No external SAEs loaded yet. Load one from a repository.</EmptyState>
          ) : (
            <div className="loaded-sae-list">
              {loadedSAEs.map(sae => (
                <div key={sae.id} className="loaded-sae-item">
                  <div className="loaded-sae-info">
                    <div className="loaded-sae-name">{sae.display_name || sae.id}</div>
                    <div className="loaded-sae-meta">
                      {sae.source && <span className="capitalize">{sae.source}</span>}
                      {sae.base_model && <span>{sae.base_model}</span>}
                      {sae.hookpoint && <span className="font-mono">{sae.hookpoint}</span>}
                      {sae.d_input && <span>d_in: {sae.d_input}</span>}
                      {sae.d_hidden && <span>features: {sae.d_hidden.toLocaleString()}</span>}
                      {sae.activation_type && <span>{sae.activation_type}</span>}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(sae.id)}
                    disabled={deletingId === sae.id}
                    className="oz-icon-btn"
                    title="Delete this SAE"
                    aria-label="Delete this SAE"
                  >
                    {deletingId === sae.id ? <Spinner /> : <Trash2 />}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <style>{`
        .external-sae-loader {
          max-width: 60rem;
        }
        .external-sae-loader .tab-content {
          padding-top: 1.5rem;
        }
        .external-sae-loader .suggested-repo-btn {
          flex-direction: column;
          align-items: flex-start;
          gap: 0.1rem;
          padding: 0.55rem 0.8rem;
          text-align: left;
        }
        .external-sae-loader .suggested-repo-kind {
          opacity: 0.7;
          font-size: 0.78rem;
          font-weight: 400;
        }
        .external-sae-loader .loader-field {
          margin-top: 1.4rem;
        }
        .external-sae-loader .loader-row {
          display: flex;
          gap: 6px;
        }
        .external-sae-loader .loader-row .oz-input {
          flex: 1;
          min-width: 0;
        }
        .external-sae-loader .loader-note {
          margin-top: 1rem;
        }
        .external-sae-loader .loader-intro {
          max-width: 46rem;
          margin: 0;
        }
        .external-sae-loader .loader-confirm {
          display: flex;
          gap: 6px;
          margin-top: 0.75rem;
        }
        .external-sae-loader .loader-submit {
          margin-top: 1.4rem;
        }
        /* What a repository holds, and what's loaded: ruled lists */
        .external-sae-loader .hookpoint-list {
          max-height: 340px;
          overflow-y: auto;
          border-bottom: 1px solid var(--oz-rule);
        }
        .external-sae-loader .hookpoint-item,
        .external-sae-loader .loaded-sae-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.7rem 0;
          border-top: 1px solid var(--oz-rule);
        }
        .external-sae-loader .loaded-sae-item:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }
        .external-sae-loader .hookpoint-info,
        .external-sae-loader .loaded-sae-info {
          display: flex;
          flex-direction: column;
          gap: 0.15rem;
          min-width: 0;
        }
        .external-sae-loader .hookpoint-name {
          font-family: var(--oz-mono);
          font-size: 0.9rem;
        }
        .external-sae-loader .loaded-sae-name {
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
        }
        .external-sae-loader .hookpoint-meta,
        .external-sae-loader .loaded-sae-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 0.15rem 0.9rem;
          color: var(--oz-fg-dim);
          font-size: 0.82rem;
          font-variant-numeric: tabular-nums;
        }
        .external-sae-loader .upload-zone {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.75rem;
          border: 1px dashed var(--oz-rule-strong);
          color: var(--oz-fg-dim);
          font-size: 0.9rem;
          cursor: pointer;
          transition: border-color 0.2s ease, color 0.2s ease;
        }
        .external-sae-loader .upload-zone:hover {
          border-color: var(--oz-mark);
          color: var(--oz-mark);
        }
        .external-sae-loader .upload-size {
          opacity: 0.7;
        }
      `}</style>
    </div>
  )
}
