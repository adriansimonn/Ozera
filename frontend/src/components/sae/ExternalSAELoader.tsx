/**
 * External SAE Loader component.
 * Provides UI for loading SAEs from HuggingFace, Gemma Scope, and user uploads.
 */

import { useState, useCallback } from 'react'
import {
  Download,
  Upload,
  Search,
  Loader2,
  AlertCircle,
  Check,
  Trash2,
  Database,
  ChevronRight,
} from 'lucide-react'
import {
  saeClient,
  type ExternalSAEInfo,
  type ExternalSAESourcesResponse,
} from '../../api/client'

interface ExternalSAELoaderProps {
  loadedSAEs: ExternalSAEInfo[]
  onSAELoaded: () => void
  onSAEDeleted: () => void
}

type LoaderTab = 'huggingface' | 'upload' | 'loaded'

// Common HuggingFace SAE repositories for quick access
const SUGGESTED_REPOS = [
  {
    repo_id: 'EleutherAI/sae-SmolLM2-135M-64x',
    label: 'EleutherAI SmolLM2-135M',
    description: 'TopK SAEs for SmolLM2-135M MLP layers (64x expansion)',
    activation: 'TopK',
  },
  {
    repo_id: 'google/gemma-scope-2b-pt',
    label: 'Gemma Scope 2B',
    description: 'JumpReLU SAEs for Gemma-2 2B (all hookpoints)',
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

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null)

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

  const handleUpload = useCallback(async () => {
    if (!uploadFile || !uploadName.trim()) return

    setUploading(true)
    setUploadError(null)

    try {
      // Read file as base64
      const arrayBuffer = await uploadFile.arrayBuffer()
      const bytes = new Uint8Array(arrayBuffer)
      let binary = ''
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i])
      }
      const base64 = btoa(binary)

      const result = await saeClient.uploadSAE({
        name: uploadName.trim(),
        weights_base64: base64,
      })

      if (result.error) {
        setUploadError(result.error)
      } else {
        setUploadFile(null)
        setUploadName('')
        onSAELoaded()
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [uploadFile, uploadName, onSAELoaded])

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
      <div className="loader-tabs">
        <button
          onClick={() => setTab('huggingface')}
          className={`loader-tab ${tab === 'huggingface' ? 'active' : ''}`}
        >
          <Download size={14} />
          HuggingFace / Gemma Scope
        </button>
        <button
          onClick={() => setTab('upload')}
          className={`loader-tab ${tab === 'upload' ? 'active' : ''}`}
        >
          <Upload size={14} />
          Upload Safetensors
        </button>
        <button
          onClick={() => setTab('loaded')}
          className={`loader-tab ${tab === 'loaded' ? 'active' : ''}`}
        >
          <Database size={14} />
          Loaded ({loadedSAEs.length})
        </button>
      </div>

      {/* HuggingFace Tab */}
      {tab === 'huggingface' && (
        <div className="tab-content">
          {/* Suggested Repos */}
          <div className="suggested-repos">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Quick Load</div>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_REPOS.map(repo => (
                <button
                  key={repo.repo_id}
                  onClick={() => handleQuickLoad(repo.repo_id)}
                  className="suggested-repo-btn"
                  title={repo.description}
                >
                  <span className="text-white text-xs">{repo.label}</span>
                  <span className="text-gray-500 text-[10px]">{repo.activation}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Repo Input */}
          <div className="repo-input-section mt-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">
              Or enter repository ID
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="org/repo-name (e.g. EleutherAI/sae-SmolLM2-135M-64x)"
                className="flex-1 bg-black/50 border border-gray-700 text-white px-3 py-2 text-sm"
                style={{ outline: 'none' }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
                disabled={browsing || loading}
              />
              <button
                onClick={handleBrowseRepo}
                disabled={browsing || loading || !repoId.trim()}
                style={{
                  background: (browsing || loading || !repoId.trim()) ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.1)',
                  border: `1px solid ${(browsing || loading || !repoId.trim()) ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.25)'}`,
                  color: (browsing || loading || !repoId.trim()) ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.95)',
                  cursor: (browsing || loading || !repoId.trim()) ? 'not-allowed' : 'pointer',
                }}
                className="px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2"
                onMouseEnter={(e) => {
                  if (!browsing && !loading && repoId.trim()) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.15)'
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.4)'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!browsing && !loading && repoId.trim()) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)'
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'
                  }
                }}
              >
                {browsing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Browse
              </button>
            </div>
          </div>

          {/* Browse Error */}
          {browseError && (
            <div className="mt-2 flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {browseError}
            </div>
          )}

          {/* Browse Results */}
          {browseResult && browseResult.available.length > 0 && (
            <div className="browse-results mt-3">
              {/* Check if first item has an error */}
              {browseResult.available[0].error ? (
                <div className="mt-2 flex items-start gap-2 text-red-400 text-sm bg-red-400/10 border border-red-400/20 px-3 py-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium mb-1">Failed to browse repository:</div>
                    <div className="text-xs">{browseResult.available[0].error}</div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">
                    Available SAEs ({browseResult.count})
                  </div>
                  <div className="hookpoint-list">
                    {browseResult.available.map((item, idx) => (
                      <div key={idx} className="hookpoint-item">
                        <div className="hookpoint-info">
                          <span className="text-white text-sm font-mono">
                            {item.hookpoint || 'root'}
                          </span>
                          <div className="hookpoint-meta text-gray-500 text-xs">
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
                          className="load-btn"
                        >
                          {loading && loadingHookpoint === item.hookpoint ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            <ChevronRight className="w-3 h-3" />
                          )}
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
            <div className="mt-3 text-gray-500 text-sm">
              No SAE hookpoints found in this repository.
            </div>
          )}

          {/* Load Result */}
          {loadResult && (
            <div className="mt-3 flex items-center gap-2 text-green-400 text-sm bg-green-400/10 border border-green-400/20 px-3 py-2">
              <Check className="w-4 h-4" />
              Loaded: {loadResult.display_name}
            </div>
          )}

          {loadError && (
            <div className="mt-3 flex items-center gap-2 text-red-400 text-sm bg-red-400/10 border border-red-400/20 px-3 py-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {loadError}
            </div>
          )}
        </div>
      )}

      {/* Upload Tab */}
      {tab === 'upload' && (
        <div className="tab-content">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">
            Upload SAE Weights
          </div>
          <p className="text-gray-500 text-xs mb-3">
            Upload a .safetensors file containing SAE weights. Supports Ozera native format,
            EleutherAI/sparsify format, and Gemma Scope format.
          </p>

          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Display Name</label>
              <input
                type="text"
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                placeholder="my-custom-sae"
                className="w-full bg-black/50 border border-gray-700 text-white px-3 py-2 text-sm"
                style={{ outline: 'none' }}
                onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'}
                onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
                disabled={uploading}
              />
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">Safetensors File</label>
              <label className="upload-zone">
                <input
                  type="file"
                  accept=".safetensors"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="hidden"
                  disabled={uploading}
                />
                {uploadFile ? (
                  <div className="text-sm">
                    <span className="text-white">{uploadFile.name}</span>
                    <span className="text-gray-500 ml-2">
                      ({(uploadFile.size / 1024 / 1024).toFixed(1)} MB)
                    </span>
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm">
                    Click to select .safetensors file (max 500MB)
                  </div>
                )}
              </label>
            </div>

            <button
              onClick={handleUpload}
              disabled={uploading || !uploadFile || !uploadName.trim()}
              style={{
                background: (uploading || !uploadFile || !uploadName.trim()) ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.1)',
                border: `1px solid ${(uploading || !uploadFile || !uploadName.trim()) ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.25)'}`,
                color: (uploading || !uploadFile || !uploadName.trim()) ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.95)',
                cursor: (uploading || !uploadFile || !uploadName.trim()) ? 'not-allowed' : 'pointer',
              }}
              className="w-full px-4 py-2 text-sm font-medium transition-colors flex items-center justify-center gap-2"
              onMouseEnter={(e) => {
                if (!uploading && uploadFile && uploadName.trim()) {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.15)'
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.4)'
                }
              }}
              onMouseLeave={(e) => {
                if (!uploading && uploadFile && uploadName.trim()) {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.1)'
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'
                }
              }}
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload SAE
                </>
              )}
            </button>

            {uploadError && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {uploadError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loaded SAEs Tab */}
      {tab === 'loaded' && (
        <div className="tab-content">
          {loadedSAEs.length === 0 ? (
            <div className="text-center py-8">
              <Database className="w-8 h-8 text-gray-600 mx-auto mb-2" />
              <p className="text-gray-500 text-sm">No external SAEs loaded yet</p>
              <p className="text-gray-600 text-xs mt-1">
                Use the HuggingFace tab to load SAEs from popular repositories
              </p>
            </div>
          ) : (
            <div className="loaded-sae-list">
              {loadedSAEs.map(sae => (
                <div key={sae.id} className="loaded-sae-item">
                  <div className="loaded-sae-info">
                    <div className="text-white text-sm font-medium">
                      {sae.display_name || sae.id}
                    </div>
                    <div className="loaded-sae-meta text-gray-500 text-xs">
                      {sae.source && <span className="capitalize">{sae.source}</span>}
                      {sae.base_model && <span>{sae.base_model}</span>}
                      {sae.hookpoint && <span className="font-mono">{sae.hookpoint}</span>}
                    </div>
                    <div className="loaded-sae-dims text-gray-500 text-xs">
                      {sae.d_input && <span>d_in: {sae.d_input}</span>}
                      {sae.d_hidden && <span>features: {sae.d_hidden.toLocaleString()}</span>}
                      {sae.activation_type && <span>{sae.activation_type}</span>}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(sae.id)}
                    disabled={deletingId === sae.id}
                    className="delete-btn"
                    title="Delete this SAE"
                  >
                    {deletingId === sae.id ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Trash2 className="w-3 h-3" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <style>{`
        .external-sae-loader {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .loader-tabs {
          display: flex;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .loader-tab {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          background: transparent;
          border: none;
          color: rgba(255, 255, 255, 0.4);
          font-size: 0.8rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .loader-tab:hover {
          color: rgba(255, 255, 255, 0.6);
          background: rgba(255, 255, 255, 0.03);
        }

        .loader-tab.active {
          color: rgba(255, 255, 255, 0.9);
          background: rgba(255, 255, 255, 0.1);
          border-bottom: 2px solid rgba(255, 255, 255, 0.4);
        }

        .tab-content {
          padding: 1rem;
        }

        .suggested-repo-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 0.125rem;
          padding: 0.5rem 0.75rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          cursor: pointer;
          transition: all 0.2s;
        }

        .suggested-repo-btn:hover {
          border-color: rgba(255, 255, 255, 0.25);
          background: rgba(255, 255, 255, 0.08);
        }

        .hookpoint-list {
          display: flex;
          flex-direction: column;
          gap: 1px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          max-height: 300px;
          overflow-y: auto;
        }

        .hookpoint-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
        }

        .hookpoint-item:hover {
          background: rgba(0, 0, 0, 0.6);
        }

        .hookpoint-info {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .hookpoint-meta {
          display: flex;
          gap: 0.75rem;
        }

        .load-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.375rem 0.75rem;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.95);
          font-size: 0.75rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .load-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.3);
        }

        .load-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .upload-zone {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.5rem;
          border: 1px dashed rgba(255, 255, 255, 0.2);
          cursor: pointer;
          transition: all 0.2s;
        }

        .upload-zone:hover {
          border-color: rgba(255, 255, 255, 0.25);
          background: rgba(255, 255, 255, 0.05);
        }

        .loaded-sae-list {
          display: flex;
          flex-direction: column;
          gap: 1px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .loaded-sae-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.75rem;
          background: rgba(0, 0, 0, 0.4);
        }

        .loaded-sae-info {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .loaded-sae-meta,
        .loaded-sae-dims {
          display: flex;
          gap: 0.75rem;
        }

        .delete-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          background: transparent;
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.4);
          cursor: pointer;
          transition: all 0.2s;
        }

        .delete-btn:hover:not(:disabled) {
          border-color: rgba(239, 68, 68, 0.5);
          color: rgb(239, 68, 68);
          background: rgba(239, 68, 68, 0.1);
        }

        .delete-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  )
}
