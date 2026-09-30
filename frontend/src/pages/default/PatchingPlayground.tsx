import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { NavBar } from '../../components/common/NavBar'
import { PatchConfigPanel } from '../../components/patching/PatchConfigPanel'
import { PatchSelector } from '../../components/patching/PatchSelector'
import { PromptComparer } from '../../components/patching/PromptComparer'
import { ExperimentsList } from '../../components/patching/ExperimentsList'
import { apiClient } from '../../api/client'
import type {
  PatchSpec,
  PatchingResult,
  CapturedActivationSummary,
  PatchingModelInfo,
  ModelLayerInfo,
  PatchingExperiment,
} from '../../types/patching'
import { Play, Zap, Trash2, AlertCircle, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { Dropdown, type DropdownGroup } from '../../components/common/Dropdown'
import { useAuthStore } from '../../stores/authStore'
import { useTheme } from '../../hooks/useTheme'

interface PatchingPlaygroundProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultPatchingPlayground({ onShowPurchaseCredits }: PatchingPlaygroundProps) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const { isLight } = useTheme()

  // Model selection
  const [models, setModels] = useState<PatchingModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [modelInfo, setModelInfo] = useState<ModelLayerInfo | null>(null)
  const [loadingModels, setLoadingModels] = useState(true)

  // Prompts
  const [sourcePrompt, setSourcePrompt] = useState('')
  const [targetPrompt, setTargetPrompt] = useState('')

  // Patches
  const [patches, setPatches] = useState<PatchSpec[]>([])

  // Generation settings
  const [maxTokens, setMaxTokens] = useState(20)
  const [temperature, setTemperature] = useState(0)

  // Captured activations
  const [capturedActivations, setCapturedActivations] = useState<CapturedActivationSummary[]>([])
  const [showCaptured, setShowCaptured] = useState(false)

  // Results
  const [result, setResult] = useState<PatchingResult | null>(null)

  // Loading/error states
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadModels() }, [])

  useEffect(() => {
    if (selectedModel) loadModelInfo(selectedModel)
  }, [selectedModel])

  const loadModels = async () => {
    try {
      setLoadingModels(true)
      const modelList = await apiClient.getPatchingModels()
      setModels(modelList)
      if (modelList.length > 0) setSelectedModel(modelList[0].model_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    } finally {
      setLoadingModels(false)
    }
  }

  const loadModelInfo = async (modelId: string) => {
    try {
      const info = await apiClient.getModelLayerInfo(modelId)
      setModelInfo(info)
    } catch (err) {
      console.error('Failed to load model info:', err)
      setModelInfo(null)
    }
  }

  const loadCapturedActivations = useCallback(async () => {
    try {
      const activations = await apiClient.listCapturedActivations()
      setCapturedActivations(activations)
    } catch (err) {
      console.error('Failed to load captured activations:', err)
    }
  }, [])

  useEffect(() => { loadCapturedActivations() }, [loadCapturedActivations])

  const hasPatches = patches.some(p => p.intervention_type === 'patch')
  const hasAblations = patches.some(p => p.intervention_type !== 'patch')
  const isMixed = hasPatches && hasAblations
  const requiresSourcePrompt = patches.length === 0 || hasPatches

  const handleRunExperiment = async () => {
    if (!isAuthenticated) { navigate('/auth'); return }
    if (!selectedModel || patches.length === 0) {
      setError('Please select a model and add at least one patch')
      return
    }
    if (!targetPrompt.trim()) { setError('Please enter a target prompt'); return }
    if (requiresSourcePrompt && !sourcePrompt.trim()) {
      setError('Please enter a source prompt (required for patching interventions)')
      return
    }

    try {
      setRunning(true)
      setError(null)
      setResult(null)

      const patchResult = await apiClient.runPatchingExperiment({
        source_prompt: requiresSourcePrompt ? sourcePrompt : undefined,
        target_prompt: targetPrompt,
        model: selectedModel,
        patches,
        max_tokens: maxTokens,
        temperature,
      })

      setResult(patchResult)
      loadCapturedActivations()
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') {
        onShowPurchaseCredits?.()
      } else {
        setError(err instanceof Error ? err.message : 'Failed to run experiment')
      }
    } finally {
      setRunning(false)
    }
  }

  const handleAddPatch = (patch: PatchSpec) => { setPatches(prev => [...prev, patch]) }
  const handleRemovePatch = (index: number) => { setPatches(prev => prev.filter((_, i) => i !== index)) }
  const handleClearPatches = () => { setPatches([]) }

  const handleDeleteCaptured = async (activationId: string) => {
    try {
      await apiClient.deleteCapturedActivation(activationId)
      loadCapturedActivations()
    } catch (err) {
      console.error('Failed to delete activation:', err)
    }
  }

  const handleClearAllCaptured = async () => {
    try {
      await apiClient.clearCapturedActivations()
      setCapturedActivations([])
    } catch (err) {
      console.error('Failed to clear activations:', err)
    }
  }

  const handleLoadExperiment = useCallback((experiment: PatchingExperiment) => {
    setSourcePrompt(experiment.source_prompt ?? '')
    setTargetPrompt(experiment.target_prompt)
    setPatches(experiment.patches)

    const model = models.find(m => m.model_id === experiment.model_id)
    if (model) setSelectedModel(experiment.model_id)

    if (experiment.baseline_output && experiment.patched_output && experiment.effect_summary) {
      setResult({
        baseline_output: experiment.baseline_output,
        patched_output: experiment.patched_output,
        baseline_tokens: [],
        patched_tokens: [],
        baseline_decoded: experiment.baseline_decoded ?? [experiment.baseline_output],
        patched_decoded: experiment.patched_decoded ?? [experiment.patched_output],
        source_activation_id: '',
        patches_applied: experiment.patches,
        effect_summary: experiment.effect_summary,
      })
    } else {
      setResult(null)
    }
  }, [models])

  const numLayers = modelInfo?.num_layers ?? 6
  const numHeads = modelInfo?.num_heads ?? 8

  // Color tokens
  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
    textMid: isLight ? '#1d1d1f' : '#ffffff',
    textSub: isLight ? '#1d1d1f' : '#ffffff',
    controlBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.04)',
    controlBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    controlHover: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.08)',
    inputBg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.4)',
    inputBorder: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)',
    errorBg: 'rgba(239, 68, 68, 0.08)',
    errorBorder: 'rgba(239, 68, 68, 0.2)',
    errorText: '#ef4444',
    accentBg: isLight ? 'rgba(59,130,246,0.08)' : 'rgba(59,130,246,0.12)',
    accentText: isLight ? '#2563eb' : 'rgba(96,165,250,0.9)',
    ablationBg: isLight ? 'rgba(168,85,247,0.06)' : 'rgba(168,85,247,0.1)',
    ablationBorder: isLight ? 'rgba(168,85,247,0.15)' : 'rgba(168,85,247,0.25)',
    ablationText: isLight ? '#7c3aed' : 'rgba(168,85,247,0.9)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    spinnerHead: isLight ? '#1d1d1f' : '#ffffff',
  }

  return (
    <div className="dp-page" style={{ background: c.bg }}>
      <NavBar />

      <div className="dp-layout" style={{ borderColor: c.divider }}>
        {/* Left Panel — 20%: Model, Prompts, Settings */}
        <div className="dp-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="dp-left-scroll">
            {/* Error banner */}
            {error && (
              <div className="dp-error" style={{ background: c.errorBg, borderColor: c.errorBorder }}>
                <AlertCircle style={{ width: 14, height: 14, color: c.errorText, flexShrink: 0 }} />
                <span style={{ color: c.errorText, fontSize: '0.8rem', flex: 1 }}>{error}</span>
                <button
                  onClick={() => setError(null)}
                  style={{ background: 'none', border: 'none', color: c.errorText, cursor: 'pointer', fontSize: '0.75rem', padding: '0.125rem 0.5rem' }}
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Model section */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="dp-section-label" style={{ color: c.textSub }}>Model</div>
              <Dropdown
                value={selectedModel}
                onChange={v => setSelectedModel(v)}
                disabled={loadingModels || running}
                groups={(() => {
                  if (loadingModels) return [{ label: 'Models', options: [{ value: '', label: 'Loading models...' }] }]
                  const groups: DropdownGroup[] = []
                  const ozera = models.filter(m => m.model_type === 'ozera' && (m.model_id === 'nano' || m.model_id === 'mini'))
                  if (ozera.length > 0) {
                    groups.push({ label: 'Ozera Models', options: ozera.map(m => ({ value: m.model_id, label: `${m.display_name} · ${m.num_layers ?? '?'}L / ${m.num_heads ?? '?'}H` })) })
                  }
                  const os = models.filter(m => m.model_type === 'open_source')
                  if (os.length > 0) {
                    groups.push({ label: 'Open Source', options: os.map(m => ({ value: m.model_id, label: `${m.display_name} · ${m.num_layers ?? '?'}L / ${m.num_heads ?? '?'}H` })) })
                  }
                  const custom = models.filter(m => m.model_type === 'custom')
                  if (custom.length > 0) {
                    groups.push({ label: 'Custom Models', options: custom.map(m => ({ value: m.model_id, label: `${m.display_name} · ${m.num_layers ?? '?'}L / ${m.num_heads ?? '?'}H` })) })
                  }
                  return groups
                })()}
              />
              {modelInfo && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginTop: '0.625rem' }}>
                  <span style={{
                    padding: '0.2rem 0.5rem',
                    background: c.controlBg,
                    border: `1px solid ${c.controlBorder}`,
                    color: c.textSub,
                    fontSize: '0.65rem',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}>
                    {modelInfo.model_type.replace(/_/g, ' ')}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: c.textSub }}>
                    {modelInfo.num_layers} layers, {modelInfo.num_heads} heads
                  </span>
                </div>
              )}
            </div>

            {/* Prompts section */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="dp-section-label" style={{ color: c.textSub }}>Prompts</div>

              {requiresSourcePrompt && (
                <div style={{ marginBottom: '0.875rem' }}>
                  <div style={{ marginBottom: '0.375rem' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 500, color: c.textMid }}>Source Prompt</div>
                    <div style={{ fontSize: '0.7rem', color: c.textSub }}>Activations captured from this prompt</div>
                  </div>
                  <textarea
                    value={sourcePrompt}
                    onChange={e => setSourcePrompt(e.target.value)}
                    placeholder="Enter source prompt..."
                    disabled={running}
                    rows={3}
                    className="dp-textarea"
                    style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
                  />
                </div>
              )}

              {!requiresSourcePrompt && patches.length > 0 && (
                <div style={{
                  padding: '0.625rem 0.75rem',
                  background: c.ablationBg,
                  border: `1px solid ${c.ablationBorder}`,
                  color: c.ablationText,
                  fontSize: '0.8rem',
                  marginBottom: '0.875rem',
                }}>
                  Ablation mode: No source prompt needed
                </div>
              )}

              <div>
                <div style={{ marginBottom: '0.375rem' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 500, color: c.textMid }}>
                    {requiresSourcePrompt ? 'Target Prompt' : 'Prompt'}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: c.textSub }}>
                    {isMixed
                      ? 'Generation runs on this prompt with interventions'
                      : requiresSourcePrompt
                        ? 'Generation runs on this prompt with patches'
                        : 'Generation runs on this prompt with ablations'}
                  </div>
                </div>
                <textarea
                  value={targetPrompt}
                  onChange={e => setTargetPrompt(e.target.value)}
                  placeholder={requiresSourcePrompt ? 'Enter target prompt...' : 'Enter prompt...'}
                  disabled={running}
                  rows={3}
                  className="dp-textarea"
                  style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
                />
              </div>
            </div>

            {/* Generation Settings */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="dp-section-label" style={{ color: c.textSub }}>Generation Settings</div>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.7rem', color: c.textSub, marginBottom: '0.3rem', fontWeight: 500 }}>Max Tokens</div>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={e => setMaxTokens(parseInt(e.target.value) || 20)}
                    min={1} max={100}
                    disabled={running}
                    className="dp-input"
                    style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.7rem', color: c.textSub, marginBottom: '0.3rem', fontWeight: 500 }}>Temperature</div>
                  <input
                    type="number"
                    value={temperature}
                    onChange={e => setTemperature(parseFloat(e.target.value) || 0)}
                    min={0} max={2} step={0.1}
                    disabled={running}
                    className="dp-input"
                    style={{ background: c.inputBg, borderColor: c.inputBorder, color: c.text }}
                  />
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', marginTop: '0.5rem' }}>
                <Info style={{ width: 12, height: 12, color: c.textSub }} />
                <span style={{ fontSize: '0.7rem', color: c.textSub }}>Temperature 0 recommended for deterministic comparison</span>
              </div>
            </div>

            {/* Run button */}
            <div className="dp-section">
              <button
                className="dp-run-btn"
                onClick={handleRunExperiment}
                disabled={running || !selectedModel || patches.length === 0}
                style={{
                  background: running ? c.controlBg : (isLight ? '#1d1d1f' : '#ffffff'),
                  color: running ? c.textSub : (isLight ? '#ffffff' : '#0a0a0a'),
                  borderColor: 'transparent',
                }}
              >
                {running ? (
                  <>
                    <div className="dp-spinner" style={{ borderColor: c.spinnerTrack, borderTopColor: c.spinnerHead }} />
                    Running...
                  </>
                ) : (
                  <>
                    <Play style={{ width: 14, height: 14 }} />
                    Run Experiment
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Center Panel — 40%: Layer Selector, Config, Captured Activations */}
        <div className="dp-center" style={{ background: c.bg, borderRight: `1px solid ${c.divider}` }}>
          <div className="dp-center-scroll">
            {/* Visual Layer Selector */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <PatchSelector
                modelInfo={modelInfo}
                patches={patches}
                onAddPatch={handleAddPatch}
                disabled={running}
              />
            </div>

            {/* Intervention Configuration */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <PatchConfigPanel
                modelId={selectedModel}
                numLayers={numLayers}
                numHeads={numHeads}
                mlpNeurons={modelInfo?.mlp_neurons ?? null}
                patches={patches}
                onAddPatch={handleAddPatch}
                onRemovePatch={handleRemovePatch}
                onClearPatches={handleClearPatches}
                disabled={running}
              />
            </div>

            {/* Captured Activations */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <button
                className="dp-toggle-btn"
                onClick={() => setShowCaptured(!showCaptured)}
                style={{ color: c.textMid }}
              >
                <span style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Captured Activations ({capturedActivations.length})
                </span>
                {showCaptured ? <ChevronUp style={{ width: 14, height: 14 }} /> : <ChevronDown style={{ width: 14, height: 14 }} />}
              </button>

              {showCaptured && (
                <div style={{ marginTop: '0.75rem' }}>
                  {capturedActivations.length === 0 ? (
                    <div style={{
                      padding: '1.25rem',
                      textAlign: 'center',
                      color: c.textSub,
                      fontSize: '0.8rem',
                      background: c.controlBg,
                      border: `1px dashed ${c.controlBorder}`,
                    }}>
                      No captured activations yet
                    </div>
                  ) : (
                    <>
                      <button
                        onClick={handleClearAllCaptured}
                        style={{
                          width: '100%',
                          padding: '0.4rem',
                          background: 'transparent',
                          border: `1px solid ${c.errorBorder}`,
                          color: c.errorText,
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                          marginBottom: '0.625rem',
                        }}
                      >
                        Clear All
                      </button>
                      {capturedActivations.map(activation => (
                        <div
                          key={activation.id}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '0.625rem 0.75rem',
                            background: c.controlBg,
                            border: `1px solid ${c.controlBorder}`,
                            marginBottom: '0.375rem',
                          }}
                        >
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', flex: 1, minWidth: 0 }}>
                            <span style={{ fontSize: '0.75rem', fontWeight: 500, color: c.accentText }}>{activation.model_id}</span>
                            <span style={{ fontSize: '0.75rem', color: c.textSub, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              "{activation.prompt.slice(0, 50)}..."
                            </span>
                            <span style={{ fontSize: '0.65rem', color: c.textSub }}>
                              {activation.num_tokens} tokens, {activation.num_layers} layers
                            </span>
                          </div>
                          <button
                            onClick={() => handleDeleteCaptured(activation.id)}
                            style={{ background: 'none', border: 'none', color: c.textSub, cursor: 'pointer', padding: '0.25rem' }}
                          >
                            <Trash2 style={{ width: 14, height: 14 }} />
                          </button>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel — 40%: Results + Saved Experiments */}
        <div className="dp-right" style={{ background: c.panelBg }}>
          <div className="dp-right-scroll">
            {/* Results */}
            <div className="dp-section" style={{ borderBottom: `1px solid ${c.divider}` }}>
              <div className="dp-section-label" style={{ color: c.textSub }}>Results</div>
              {result ? (
                <PromptComparer
                  baselineOutput={result.baseline_output}
                  patchedOutput={result.patched_output}
                  baselineDecoded={result.baseline_decoded}
                  patchedDecoded={result.patched_decoded}
                  effectSummary={result.effect_summary}
                  interventionMode={isMixed ? 'mixed' : hasAblations ? 'ablation' : 'patch'}
                />
              ) : (
                <div className="dp-placeholder">
                  <Zap style={{ width: 36, height: 36, color: c.textSub, opacity: 0.3, marginBottom: '0.75rem' }} />
                  <p style={{ color: c.textSub, fontSize: '0.8rem', margin: 0 }}>
                    Configure patches and run an experiment to see results
                  </p>
                </div>
              )}
            </div>

            {/* Saved Experiments */}
            <div className="dp-section">
              <ExperimentsList
                currentExperiment={{
                  sourcePrompt,
                  targetPrompt,
                  modelId: selectedModel,
                  patches,
                  result: result || undefined,
                }}
                onLoadExperiment={handleLoadExperiment}
                disabled={running}
              />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .dp-page {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dp-layout {
          display: flex;
          flex: 1;
          margin-top: 76px;
          overflow: hidden;
        }

        /* Left: 20% */
        .dp-left {
          width: 20%;
          min-width: 280px;
          max-width: 380px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dp-left-scroll {
          flex: 1;
          overflow-y: auto;
        }

        /* Center: 40% */
        .dp-center {
          width: 40%;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dp-center-scroll {
          flex: 1;
          overflow-y: auto;
        }

        /* Right: 40% */
        .dp-right {
          width: 40%;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .dp-right-scroll {
          flex: 1;
          overflow-y: auto;
        }

        .dp-section {
          padding: 1rem 1.25rem;
        }

        .dp-section-label {
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 0.625rem;
        }

        .dp-textarea {
          width: 100%;
          padding: 0.625rem 0.75rem;
          border: 1px solid;
          font-size: 0.85rem;
          font-family: inherit;
          resize: vertical;
          min-height: 68px;
          outline: none;
          box-sizing: border-box;
        }

        .dp-textarea:focus {
          border-color: rgba(59, 130, 246, 0.4) !important;
        }

        .dp-input {
          width: 100%;
          padding: 0.5rem 0.625rem;
          border: 1px solid;
          font-size: 0.85rem;
          outline: none;
          box-sizing: border-box;
        }

        .dp-input:focus {
          border-color: rgba(59, 130, 246, 0.4) !important;
        }

        .dp-run-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          border: 1px solid;
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
          transition: opacity 0.15s;
          letter-spacing: 0.01em;
        }

        .dp-run-btn:hover:not(:disabled) {
          opacity: 0.85;
        }

        .dp-run-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .dp-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid;
          border-radius: 50%;
          animation: dp-spin 0.8s linear infinite;
        }

        @keyframes dp-spin {
          to { transform: rotate(360deg); }
        }

        .dp-toggle-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: transparent;
          border: none;
          cursor: pointer;
          padding: 0;
        }

        .dp-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem 2rem;
          text-align: center;
        }

        .dp-error {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 0.875rem;
          border: 1px solid;
          margin: 0.75rem 1.25rem 0;
        }

        @media (max-width: 1200px) {
          .dp-layout {
            flex-wrap: wrap;
          }
          .dp-left {
            width: 100%;
            max-width: 100%;
            max-height: 35vh;
            border-right: none !important;
            border-bottom: 1px solid;
          }
          .dp-center {
            width: 50%;
            border-right: 1px solid;
          }
          .dp-right {
            width: 50%;
          }
        }

        @media (max-width: 768px) {
          .dp-center,
          .dp-right {
            width: 100%;
            border-right: none !important;
          }
        }

        /* ===== Default interface overrides for PatchSelector ===== */
        [data-interface="default"] .patch-selector {
          background: none;
          border: none;
          margin-bottom: 0;
        }

        [data-interface="default"] .selector-header {
          padding: 0 0 0.5rem 0;
          border-bottom: none;
        }

        [data-interface="default"] .selector-header h3 {
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        [data-interface="default"] .header-icon {
          width: 13px;
          height: 13px;
        }

        [data-interface="default"] .patches-legend {
          padding: 0.5rem 0 0;
          border-top: none;
        }

        [data-interface="default"] .legend-label {
          font-size: 0.7rem;
        }

        [data-interface="default"] .legend-badge {
          font-size: 0.65rem;
          padding: 0.1rem 0.375rem;
        }

        /* ===== Default interface overrides for PatchConfigPanel ===== */
        [data-interface="default"] .patch-config-panel {
          background: none;
          border: none;
          padding: 0;
        }

        [data-interface="default"] .patch-config-panel .panel-header {
          padding-bottom: 0.5rem;
          border-bottom: none;
          margin-bottom: 0.75rem;
        }

        [data-interface="default"] .patch-config-panel .panel-header h3 {
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        [data-interface="default"] .patch-config-panel .config-form {
          gap: 0.75rem;
        }

        [data-interface="default"] .patch-config-panel .form-group label {
          font-size: 0.65rem;
        }

        [data-interface="default"] .patch-config-panel .form-group select,
        [data-interface="default"] .patch-config-panel .form-group input[type="text"] {
          padding: 0.5rem 0.625rem;
          font-size: 0.8rem;
        }

        [data-interface="default"] .patch-config-panel .intervention-hint {
          font-size: 0.65rem;
        }

        [data-interface="default"] .patch-config-panel .add-patch-btn {
          padding: 0.5rem 0.75rem;
          font-size: 0.8rem;
        }

        [data-interface="default"] .patch-config-panel .patches-list {
          margin-top: 0.75rem;
          padding-top: 0.75rem;
        }

        [data-interface="default"] .patch-config-panel .list-header {
          font-size: 0.7rem;
          margin-bottom: 0.5rem;
        }

        [data-interface="default"] .patch-config-panel .patch-item {
          padding: 0.5rem 0.625rem;
          margin-bottom: 0.375rem;
        }

        [data-interface="default"] .patch-config-panel .patch-info {
          gap: 0.5rem;
        }

        [data-interface="default"] .patch-config-panel .patch-intervention {
          font-size: 0.6rem;
          padding: 0.1rem 0.3rem;
        }

        [data-interface="default"] .patch-config-panel .patch-layer {
          font-size: 0.75rem;
        }

        [data-interface="default"] .patch-config-panel .patch-type {
          font-size: 0.7rem;
          padding: 0.1rem 0.375rem;
        }

        [data-interface="default"] .patch-config-panel .patch-detail {
          font-size: 0.7rem;
        }

        [data-interface="default"] .patch-config-panel .patch-blend {
          font-size: 0.7rem;
        }

        [data-interface="default"] .patch-config-panel .blend-labels {
          font-size: 0.65rem;
        }

        /* ===== Default interface overrides for ExperimentsList ===== */
        [data-interface="default"] .section-toggle {
          padding: 0;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: transparent;
          border: none;
          cursor: pointer;
          color: #ffffff;
        }

        [data-interface="default"][data-bg="light"] .section-toggle {
          color: #1d1d1f;
        }

        [data-interface="default"] .section-toggle h2 {
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin: 0;
        }

        [data-interface="default"] .section-toggle svg {
          width: 14px;
          height: 14px;
        }

        [data-interface="default"] .exp-content {
          margin-top: 0.625rem;
        }

        [data-interface="default"] .exp-save-section {
          margin-bottom: 0.625rem;
        }

        [data-interface="default"] .exp-save-form {
          gap: 0.375rem;
        }

        [data-interface="default"] .exp-save-form input {
          padding: 0.375rem 0.625rem;
          font-size: 0.75rem;
        }

        [data-interface="default"] .exp-save-btn,
        [data-interface="default"] .exp-cancel-btn {
          padding: 0.375rem 0.625rem;
          font-size: 0.7rem;
        }

        [data-interface="default"] .exp-show-save-btn {
          padding: 0.4rem 0.75rem;
          font-size: 0.75rem;
          gap: 0.375rem;
        }

        [data-interface="default"] .exp-btn-icon {
          width: 12px;
          height: 12px;
        }

        [data-interface="default"] .exp-actions-row {
          margin-bottom: 0.625rem;
          gap: 0.375rem;
        }

        [data-interface="default"] .exp-action-btn {
          padding: 0.35rem 0.625rem;
          font-size: 0.7rem;
          gap: 0.25rem;
        }

        [data-interface="default"] .exp-action-icon {
          width: 12px;
          height: 12px;
        }

        [data-interface="default"] .exp-empty {
          padding: 1.25rem 0.75rem;
        }

        [data-interface="default"] .exp-empty p {
          font-size: 0.75rem;
        }

        [data-interface="default"] .exp-empty-hint {
          font-size: 0.65rem !important;
        }

        [data-interface="default"] .exp-grid {
          gap: 0.375rem;
        }

        [data-interface="default"] .exp-card {
          padding: 0.5rem 0.625rem;
        }

        [data-interface="default"] .exp-card-header {
          margin-bottom: 0.375rem;
        }

        [data-interface="default"] .exp-name {
          font-size: 0.75rem;
        }

        [data-interface="default"] .exp-card-action-btn .exp-action-icon {
          width: 12px;
          height: 12px;
        }

        [data-interface="default"] .exp-card-body {
          margin-bottom: 0.375rem;
          gap: 0.375rem;
        }

        [data-interface="default"] .exp-prompt-label {
          font-size: 0.6rem;
          margin-bottom: 0.05rem;
        }

        [data-interface="default"] .exp-prompt-text {
          font-size: 0.7rem;
        }

        [data-interface="default"] .exp-arrow-icon {
          width: 12px;
          height: 12px;
        }

        [data-interface="default"] .exp-ablation-label {
          font-size: 0.6rem;
          padding: 0.15rem 0.375rem;
        }

        [data-interface="default"] .exp-card-meta {
          margin-bottom: 0.375rem;
          gap: 0.5rem;
        }

        [data-interface="default"] .exp-meta-item {
          font-size: 0.65rem;
        }

        [data-interface="default"] .exp-card-footer {
          padding-top: 0.375rem;
        }

        [data-interface="default"] .exp-timestamp {
          font-size: 0.65rem;
          gap: 0.25rem;
        }

        [data-interface="default"] .exp-time-icon {
          width: 10px;
          height: 10px;
        }

        [data-interface="default"] .exp-load-btn {
          padding: 0.25rem 0.625rem;
          font-size: 0.65rem;
        }

        /* ===== Default interface dark-mode input overrides ===== */
        [data-interface="default"] .patch-config-panel .form-group select,
        [data-interface="default"] .patch-config-panel .form-group input[type="text"] {
          background: rgba(0, 0, 0, 0.4);
          border-color: rgba(255, 255, 255, 0.08);
        }

        [data-interface="default"] .patch-config-panel .add-patch-btn {
          background: rgba(255, 255, 255, 0.04);
          border-color: rgba(255, 255, 255, 0.08);
        }

        [data-interface="default"] .patch-config-panel .add-patch-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.08);
        }

        [data-interface="default"] .patch-config-panel .patch-item {
          background: rgba(255, 255, 255, 0.02);
          border-color: rgba(255, 255, 255, 0.06);
        }

        [data-interface="default"] .patch-config-panel .patches-list {
          border-top-color: rgba(255, 255, 255, 0.06);
        }

        [data-interface="default"] .patch-config-panel .clear-btn {
          font-size: 0.65rem;
        }

        [data-interface="default"] .exp-card {
          background: rgba(255, 255, 255, 0.02);
          border-color: rgba(255, 255, 255, 0.06);
        }

        [data-interface="default"] .exp-card:hover {
          border-color: rgba(255, 255, 255, 0.12);
        }

        [data-interface="default"] .exp-action-btn {
          border-color: rgba(255, 255, 255, 0.06);
        }

        [data-interface="default"] .exp-save-form input {
          background: rgba(0, 0, 0, 0.4);
          border-color: rgba(255, 255, 255, 0.08);
        }

        /* ===== Default interface light-mode overrides ===== */
        [data-interface="default"][data-bg="light"] .patch-config-panel .form-group select,
        [data-interface="default"][data-bg="light"] .patch-config-panel .form-group input[type="text"] {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.1);
        }

        [data-interface="default"][data-bg="light"] .patch-config-panel .add-patch-btn {
          background: rgba(0, 0, 0, 0.04);
          border-color: rgba(0, 0, 0, 0.1);
          color: #1d1d1f;
        }

        [data-interface="default"][data-bg="light"] .patch-config-panel .add-patch-btn:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.07);
        }

        [data-interface="default"][data-bg="light"] .patch-config-panel .patch-item {
          background: rgba(0, 0, 0, 0.02);
          border-color: rgba(0, 0, 0, 0.06);
        }

        [data-interface="default"][data-bg="light"] .patch-config-panel .patches-list {
          border-top-color: rgba(0, 0, 0, 0.06);
        }

        [data-interface="default"][data-bg="light"] .selector-header h3,
        [data-interface="default"][data-bg="light"] .patch-config-panel .panel-header h3 {
          color: #1d1d1f;
        }

        [data-interface="default"][data-bg="light"] .patch-config-panel .form-group label {
          color: #1d1d1f;
        }

        [data-interface="default"][data-bg="light"] .exp-card {
          background: rgba(0, 0, 0, 0.02);
          border-color: rgba(0, 0, 0, 0.06);
        }

        [data-interface="default"][data-bg="light"] .exp-card:hover {
          border-color: rgba(0, 0, 0, 0.12);
        }

        [data-interface="default"][data-bg="light"] .exp-action-btn {
          border-color: rgba(0, 0, 0, 0.08);
        }

        [data-interface="default"][data-bg="light"] .exp-save-form input {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.1);
        }
      `}</style>
    </div>
  )
}
