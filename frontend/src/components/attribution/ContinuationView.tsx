/**
 * The Continuation view: generate from a target prompt with and without interventions (patches
 * copied from a source prompt, or ablations) and compare the two texts. One prompt and no
 * metric: what the Activation Patching page was. It takes the whole page below the nav bar
 * while it's shown, with the page's view tabs and model at the top of its first column.
 */

import { useState, useEffect, useCallback, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { PatchConfigPanel } from '../patching/PatchConfigPanel'
import { PatchSelector } from '../patching/PatchSelector'
import { PromptComparer } from '../patching/PromptComparer'
import { ExperimentsList } from '../patching/ExperimentsList'
import { apiClient } from '../../api/client'
import type { PatchSpec, PatchingResult, ModelLayerInfo, PatchingExperiment } from '../../types/patching'
import { AlertCircle } from 'lucide-react'
import { EmptyState } from '../common/EmptyState'
import { Spinner } from '../common/Spinner'
import { ModelPicker, type ModelChoice } from '../lab/ModelPicker'
import { useAuthStore } from '../../stores/authStore'
import { useAssistantPage } from '../../hooks/useAssistantPage'
import { continuationPart } from '../assistant/pages/attribution'

interface ContinuationViewProps {
  /** The page's models and the one chosen (shared with its other views) */
  lab: ModelChoice
  /** The page's view tabs, shown at the top of the first column */
  tabs: ReactNode
  hidden: boolean
  /** Switch the page to this view (a run's result opens in it) */
  show: () => void
  onShowPurchaseCredits?: () => void
}

export function ContinuationView({ lab, tabs, hidden, show, onShowPurchaseCredits }: ContinuationViewProps) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  const selectedModel = lab.modelId
  const setSelectedModel = lab.setModelId
  const [modelInfo, setModelInfo] = useState<ModelLayerInfo | null>(null)

  // Prompts
  const [sourcePrompt, setSourcePrompt] = useState('')
  const [targetPrompt, setTargetPrompt] = useState('')

  // Patches
  const [patches, setPatches] = useState<PatchSpec[]>([])

  // Generation settings
  const [maxTokens, setMaxTokens] = useState(20)
  const [temperature, setTemperature] = useState(0)

  // Results
  const [result, setResult] = useState<PatchingResult | null>(null)

  // Loading/error states
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // The assistant: what's on screen, set_inputs, and a run_patching run's result
  useAssistantPage('attribution', continuationPart({
    model: selectedModel,
    setModel: setSelectedModel,
    show,
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
  }))

  useEffect(() => {
    if (!selectedModel) return
    let cancelled = false
    apiClient
      .getModelLayerInfo(selectedModel)
      .then((info) => !cancelled && setModelInfo(info))
      .catch((err) => {
        console.error('Failed to load model info:', err)
        if (!cancelled) setModelInfo(null)
      })
    return () => {
      cancelled = true
    }
  }, [selectedModel])

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

  const handleLoadExperiment = useCallback((experiment: PatchingExperiment) => {
    setSourcePrompt(experiment.source_prompt ?? '')
    setTargetPrompt(experiment.target_prompt)
    setPatches(experiment.patches)

    if (lab.models.some(m => m.model_id === experiment.model_id)) setSelectedModel(experiment.model_id)

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
  }, [lab.models, setSelectedModel])

  const numLayers = modelInfo?.num_layers ?? 6
  const numHeads = modelInfo?.num_heads ?? 8

  return (
    <div className="dp-page dp-layout oz-page__body" style={hidden ? { display: 'none' } : undefined}>
        {/* Left Panel: View, Model, Prompts, Settings (as wide as the other views' side column) */}
        <div className="oz-side dp-left">
          <div className="oz-scroll">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              {tabs}
            </section>

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
              <ModelPicker lab={lab} disabled={running} assist="attribution.model" />
              {modelInfo && (
                <div className="dp-model-meta">
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
                    data-assist="attribution.source_prompt"
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
                  data-assist="attribution.target_prompt"
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
                    data-assist="attribution.max_tokens"
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
                    data-assist="attribution.temperature"
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
                data-assist="attribution.patch_button"
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

        {/* Center Panel: Layer Selector, Config */}
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
              <div data-assist="attribution.patches">
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
          </div>
        </div>

        {/* Right Panel: Results + Saved Experiments */}
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

      <style>{`
        /* Three columns: what to run (the page's side column), where to intervene, what came back */
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

        /* A heading in the results column */
        .dp-heading {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        /* Narrow: the columns stack, each as tall as its content (the lab pages' breakpoint) */
        @media (max-width: 1100px) {
          .dp-page .dp-center, .dp-page .dp-right { flex: none; overflow: visible; border-right: none; }
          .dp-page .dp-center { border-bottom: 1px solid var(--oz-rule); }
        }
      `}</style>
    </div>
  )
}
