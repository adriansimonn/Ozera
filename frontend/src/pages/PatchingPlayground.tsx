import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { NavBar } from '../components/common/NavBar'
import { PatchConfigPanel } from '../components/patching/PatchConfigPanel'
import { PatchSelector } from '../components/patching/PatchSelector'
import { PromptComparer } from '../components/patching/PromptComparer'
import { ExperimentsList } from '../components/patching/ExperimentsList'
import { apiClient } from '../api/client'
import type {
  PatchSpec,
  PatchingResult,
  CapturedActivationSummary,
  PatchingModelInfo,
  ModelLayerInfo,
  PatchingExperiment,
} from '../types/patching'
import { Trash2, AlertCircle } from 'lucide-react'
import { Dropdown, type DropdownGroup } from '../components/common/Dropdown'
import { EmptyState } from '../components/common/EmptyState'
import { Spinner } from '../components/common/Spinner'
import { useAuthStore } from '../stores/authStore'
import { useAssistantPage } from '../hooks/useAssistantPage'
import { patchingBridge } from '../components/assistant/pages/patching'

const MODEL_TYPE_LABELS: Record<string, string> = { ozera: 'Ozera', open_source: 'Open source', custom: 'Custom' }

interface PatchingPlaygroundProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultPatchingPlayground({ onShowPurchaseCredits }: PatchingPlaygroundProps) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

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

  // The assistant: what's on screen, set_inputs, and a run_patching run's result
  useAssistantPage('patching', patchingBridge({
    model: selectedModel,
    setModel: setSelectedModel,
    layers: modelInfo?.num_layers ?? null,
    heads: modelInfo?.num_heads ?? null,
    sourcePrompt,
    setSourcePrompt,
    targetPrompt,
    setTargetPrompt,
    patches,
    setPatches,
    maxTokens,
    setMaxTokens,
    temperature,
    setTemperature,
    running,
    result,
    setResult,
    captured: capturedActivations.length,
  }))

  useEffect(() => { loadModels() }, [])

  useEffect(() => {
    if (selectedModel) loadModelInfo(selectedModel)
  }, [selectedModel])

  const loadModels = async () => {
    try {
      setLoadingModels(true)
      const modelList = await apiClient.getPatchingModels()
      setModels(modelList)
      if (modelList.length > 0) setSelectedModel((current) => current || modelList[0].model_id)
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
  return (
    <div className="dp-page oz-page">
      <NavBar />

      <div className="dp-layout oz-page__body">
        {/* Left Panel — 20%: Model, Prompts, Settings */}
        <div className="dp-left">
          <div className="oz-scroll">
            {/* Error banner */}
            {error && (
              <div className="dp-error oz-note oz-note--error" role="alert">
                <AlertCircle aria-hidden="true" />
                <span>{error}</span>
                <button className="oz-link" onClick={() => setError(null)}>
                  Dismiss
                </button>
              </div>
            )}

            {/* Model section */}
            <div className="oz-section">
              <h2 className="oz-title">Model</h2>
              <div data-assist="patching.model">
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
              </div>
              {modelInfo && (
                <div className="dp-model-meta">
                  <span className="oz-tag">{MODEL_TYPE_LABELS[modelInfo.model_type] ?? modelInfo.model_type.replace(/_/g, ' ')}</span>
                  <span className="oz-hint">
                    {modelInfo.num_layers} layers, {modelInfo.num_heads} heads
                  </span>
                </div>
              )}
            </div>

            {/* Prompts section */}
            <div className="oz-section">
              <h2 className="oz-title">Prompts</h2>

              {requiresSourcePrompt && (
                <div className="dp-field">
                  <div className="dp-field-head">
                    <div className="dp-field-name">Source prompt</div>
                    <div className="oz-hint">Activations captured from this prompt</div>
                  </div>
                  <textarea
                    data-assist="patching.source_prompt"
                    value={sourcePrompt}
                    onChange={e => setSourcePrompt(e.target.value)}
                    placeholder="Enter source prompt..."
                    disabled={running}
                    rows={3}
                    className="oz-textarea"
                  />
                </div>
              )}

              {!requiresSourcePrompt && patches.length > 0 && (
                <div className="dp-field oz-note">Ablation mode: no source prompt needed</div>
              )}

              <div className="dp-field">
                <div className="dp-field-head">
                  <div className="dp-field-name">{requiresSourcePrompt ? 'Target prompt' : 'Prompt'}</div>
                  <div className="oz-hint">
                    {isMixed
                      ? 'Generation runs on this prompt with interventions'
                      : requiresSourcePrompt
                        ? 'Generation runs on this prompt with patches'
                        : 'Generation runs on this prompt with ablations'}
                  </div>
                </div>
                <textarea
                  data-assist="patching.target_prompt"
                  value={targetPrompt}
                  onChange={e => setTargetPrompt(e.target.value)}
                  placeholder={requiresSourcePrompt ? 'Enter target prompt...' : 'Enter prompt...'}
                  disabled={running}
                  rows={3}
                  className="oz-textarea"
                />
              </div>
            </div>

            {/* Generation Settings */}
            <div className="oz-section">
              <h2 className="oz-title">Generation settings</h2>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <div style={{ flex: 1 }}>
                  <div className="oz-label">Max tokens</div>
                  <input
                    type="number"
                    data-assist="patching.max_tokens"
                    value={maxTokens}
                    onChange={e => setMaxTokens(parseInt(e.target.value) || 20)}
                    min={1} max={100}
                    disabled={running}
                    className="oz-input oz-input--number"
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <div className="oz-label">Temperature</div>
                  <input
                    type="number"
                    data-assist="patching.temperature"
                    value={temperature}
                    onChange={e => setTemperature(parseFloat(e.target.value) || 0)}
                    min={0} max={2} step={0.1}
                    disabled={running}
                    className="oz-input oz-input--number"
                  />
                </div>
              </div>
              <p className="oz-hint" style={{ marginTop: '0.6rem' }}>Temperature 0 recommended for deterministic comparison</p>
            </div>

            {/* Run button */}
            <div className="oz-section dp-run">
              <button
                className={`oz-btn oz-btn--block${running ? '' : ' oz-btn--arrow'}`}
                data-assist="patching.run_button"
                onClick={handleRunExperiment}
                disabled={running || !selectedModel || patches.length === 0}
              >
                {running ? (
                  <>
                    <Spinner />
                    Running...
                  </>
                ) : (
                  'Run Experiment'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Center Panel — 40%: Layer Selector, Config, Captured Activations */}
        <div className="dp-center">
          <div className="oz-scroll">
            {/* Visual Layer Selector */}
            <div className="oz-section">
              <PatchSelector
                modelInfo={modelInfo}
                patches={patches}
                onAddPatch={handleAddPatch}
                disabled={running}
              />
            </div>

            {/* Intervention Configuration */}
            <div className="oz-section">
              <div data-assist="patching.patches">
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
            </div>

            {/* Captured Activations */}
            <div className="oz-section">
              <button
                className="dp-toggle-btn"
                aria-expanded={showCaptured}
                onClick={() => setShowCaptured(!showCaptured)}
              >
                <h2 className="oz-title">
                  Captured activations
                  <span className="oz-title__end">{capturedActivations.length}</span>
                </h2>
                <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
                  <path d="M1 1l4 4 4-4" />
                </svg>
              </button>

              {showCaptured && (
                <div className="dp-captured">
                  {capturedActivations.length === 0 ? (
                    <p className="oz-hint">No captured activations yet</p>
                  ) : (
                    <>
                      {capturedActivations.map(activation => (
                        <div key={activation.id} className="dp-captured-item">
                          <div className="dp-captured-main">
                            <span className="dp-captured-prompt">"{activation.prompt.slice(0, 50)}..."</span>
                            <span className="oz-hint">
                              {activation.model_id} · {activation.num_tokens} tokens, {activation.num_layers} layers
                            </span>
                          </div>
                          <button
                            className="oz-icon-btn"
                            onClick={() => handleDeleteCaptured(activation.id)}
                            aria-label="Delete captured activations"
                            title="Delete"
                          >
                            <Trash2 />
                          </button>
                        </div>
                      ))}
                      <button className="oz-link dp-captured-clear" onClick={handleClearAllCaptured}>
                        Clear all
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel — 40%: Results + Saved Experiments */}
        <div className="dp-right">
          <div className="oz-scroll">
            {/* Results */}
            <div className="oz-section dp-results">
              <h2 className="dp-heading">Results</h2>
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
                <EmptyState glyph="patching">Configure patches and run an experiment to see results</EmptyState>
              )}
            </div>

            {/* Saved Experiments */}
            <div className="oz-section dp-saved">
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
        /* Three columns: what to run, where to intervene, what came back */
        .dp-left {
          display: flex;
          flex-direction: column;
          flex: none;
          width: 22%;
          min-width: 300px;
          max-width: 400px;
          overflow: hidden;
          border-right: 1px solid var(--oz-rule);
        }

        .dp-center {
          display: flex;
          flex-direction: column;
          flex: 1 1 0;
          min-width: 0;
          overflow: hidden;
          border-right: 1px solid var(--oz-rule);
        }

        .dp-right {
          display: flex;
          flex-direction: column;
          flex: 1 1 0;
          min-width: 0;
          overflow: hidden;
        }

        .dp-page .dp-error {
          margin: 1rem var(--oz-gutter) 0;
        }

        .dp-model-meta {
          display: flex;
          align-items: center;
          gap: 0.65rem;
          margin-top: 0.75rem;
        }

        .dp-field + .dp-field {
          margin-top: 1.1rem;
        }

        .dp-field-head {
          margin-bottom: 0.5rem;
        }

        .dp-field-name {
          font-size: 0.9rem;
          font-weight: 500;
        }

        .dp-page .dp-run {
          border-bottom: none;
        }

        /* A section's title that also opens and closes it */
        .dp-toggle-btn {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          width: 100%;
          padding: 0;
          text-align: left;
          cursor: pointer;
        }

        .dp-toggle-btn .oz-title {
          flex: 1;
          margin: 0;
          transition: color 0.2s ease;
        }

        .dp-toggle-btn:hover .oz-title {
          color: var(--oz-mark);
        }

        .dp-captured {
          margin-top: 0.9rem;
        }

        .dp-captured-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          padding: 0.65rem 0;
          border-top: 1px solid var(--oz-rule);
        }

        .dp-captured-main {
          display: flex;
          flex-direction: column;
          gap: 0.15rem;
          flex: 1;
          min-width: 0;
        }

        .dp-captured-prompt {
          overflow: hidden;
          font-size: 0.9rem;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .dp-page .dp-captured-clear {
          margin-top: 0.75rem;
          font-size: 0.85rem;
        }

        /* A heading in the results column */
        .dp-heading {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        @media (max-width: 1200px) {
          .dp-page.oz-page { height: auto; overflow: visible; }
          .dp-layout.oz-page__body { flex-wrap: wrap; }
          .dp-left { width: 100%; max-width: 100%; border-right: none; border-bottom: 1px solid var(--oz-rule); }
          .dp-center, .dp-right { flex: 1 1 50%; }
        }

        @media (max-width: 768px) {
          .dp-center, .dp-right { flex: 1 1 100%; border-right: none; }
        }
      `}</style>
    </div>
  )
}
