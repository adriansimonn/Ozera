/**
 * Activation Patching Playground - Interactive interface for swapping activations
 * between different prompts to understand model behavior.
 */

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
import { Play, Zap, Trash2, AlertCircle, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { Dropdown, type DropdownGroup } from '../components/common/Dropdown'
import { useAuthStore } from '../stores/authStore'

interface PatchingPlaygroundProps {
  onShowPurchaseCredits?: () => void
}

export function PatchingPlayground({ onShowPurchaseCredits }: PatchingPlaygroundProps) {
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

  // Load available models on mount
  useEffect(() => {
    loadModels()
  }, [])

  // Load model layer info when model changes
  useEffect(() => {
    if (selectedModel) {
      loadModelInfo(selectedModel)
    }
  }, [selectedModel])

  const loadModels = async () => {
    try {
      setLoadingModels(true)
      const modelList = await apiClient.getPatchingModels()
      setModels(modelList)
      if (modelList.length > 0) {
        setSelectedModel(modelList[0].model_id)
      }
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

  useEffect(() => {
    loadCapturedActivations()
  }, [loadCapturedActivations])

  // Check if any patches require source activations (patch intervention type)
  // Show source prompt by default (when no patches added), hide only when all patches are ablation
  const hasPatches = patches.some(p => p.intervention_type === 'patch')
  const hasAblations = patches.some(p => p.intervention_type !== 'patch')
  const isMixed = hasPatches && hasAblations
  const requiresSourcePrompt = patches.length === 0 || hasPatches

  const handleRunExperiment = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    if (!selectedModel || patches.length === 0) {
      setError('Please select a model and add at least one patch')
      return
    }

    if (!targetPrompt.trim()) {
      setError('Please enter a target prompt')
      return
    }

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
      // Refresh captured activations list
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

  const handleAddPatch = (patch: PatchSpec) => {
    setPatches(prev => [...prev, patch])
  }

  const handleRemovePatch = (index: number) => {
    setPatches(prev => prev.filter((_, i) => i !== index))
  }

  const handleClearPatches = () => {
    setPatches([])
  }

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

    // Switch to the experiment's model if available
    const model = models.find(m => m.model_id === experiment.model_id)
    if (model) {
      setSelectedModel(experiment.model_id)
    }

    // If experiment has results, show them
    if (experiment.baseline_output && experiment.patched_output && experiment.effect_summary) {
      setResult({
        baseline_output: experiment.baseline_output,
        patched_output: experiment.patched_output,
        baseline_tokens: [],
        patched_tokens: [],
        // Use stored decoded tokens if available, otherwise fall back to output string as single token
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

  return (
    <div className="patching-playground">
      <NavBar />

      <div className="playground-content">
        <div className="playground-header">
          <div className="header-title">
            <h1>Activation Patching Playground</h1>
          </div>
          <p className="header-description">
            Perform causal interventions to understand how the model processes information.
            Swap activations between prompts (patching) or ablate activations (zero, mean, noise) to measure component importance.
          </p>
        </div>

        {error && (
          <div className="error-banner">
            <AlertCircle className="error-icon" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="dismiss-btn">Dismiss</button>
          </div>
        )}

        <div className="playground-grid">
          {/* Left Column: Prompts and Settings */}
          <div className="left-column">
            <div className="section model-section">
              <h2>Model</h2>
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
                <div className="model-info-row">
                  <span className="info-badge">{modelInfo.model_type.replace(/_/g, ' ')}</span>
                  <span className="info-text">{modelInfo.num_layers} layers, {modelInfo.num_heads} heads</span>
                </div>
              )}
            </div>

            <div className="section prompts-section">
              <h2>Prompts</h2>

              {requiresSourcePrompt && (
                <div className="prompt-group">
                  <label>
                    <span className="label-main">Source Prompt</span>
                    <span className="label-hint">Activations will be captured from this prompt</span>
                  </label>
                  <textarea
                    value={sourcePrompt}
                    onChange={e => setSourcePrompt(e.target.value)}
                    placeholder="Enter source prompt..."
                    disabled={running}
                    rows={3}
                  />
                </div>
              )}

              {!requiresSourcePrompt && patches.length > 0 && (
                <div className="ablation-notice">
                  <span>Ablation mode: No source prompt needed. Activations will be zeroed, averaged, or replaced with noise.</span>
                </div>
              )}

              <div className="prompt-group">
                <label>
                  <span className="label-main">{requiresSourcePrompt ? 'Target Prompt' : 'Prompt'}</span>
                  <span className="label-hint">
                    {isMixed
                      ? 'Generation will run on this prompt with interventions applied'
                      : requiresSourcePrompt
                        ? 'Generation will run on this prompt with patches applied'
                        : 'Generation will run on this prompt with ablations applied'}
                  </span>
                </label>
                <textarea
                  value={targetPrompt}
                  onChange={e => setTargetPrompt(e.target.value)}
                  placeholder={requiresSourcePrompt ? 'Enter target prompt...' : 'Enter prompt...'}
                  disabled={running}
                  rows={3}
                />
              </div>
            </div>

            <div className="section settings-section">
              <h2>Generation Settings</h2>
              <div className="settings-row">
                <div className="setting-group">
                  <label>Max Tokens</label>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={e => setMaxTokens(parseInt(e.target.value) || 20)}
                    min={1}
                    max={100}
                    disabled={running}
                  />
                </div>
                <div className="setting-group">
                  <label>Temperature</label>
                  <input
                    type="number"
                    value={temperature}
                    onChange={e => setTemperature(parseFloat(e.target.value) || 0)}
                    min={0}
                    max={2}
                    step={0.1}
                    disabled={running}
                  />
                </div>
              </div>
              <div className="settings-hint">
                <Info className="hint-icon" />
                <span>Temperature 0 recommended for deterministic comparison</span>
              </div>
            </div>

            <button
              className="run-experiment-btn"
              onClick={handleRunExperiment}
              disabled={running || !selectedModel || patches.length === 0}
            >
              {running ? (
                <>
                  <div className="spinner" />
                  Running Experiment...
                </>
              ) : (
                <>
                  <Play className="btn-icon" />
                  Run Experiment
                </>
              )}
            </button>
          </div>

          {/* Center Column: Intervention Configuration */}
          <div className="center-column">
            {/* Visual Layer Selector */}
            <div className="section section-transparent">
              <PatchSelector
                modelInfo={modelInfo}
                patches={patches}
                onAddPatch={handleAddPatch}
                disabled={running}
              />
            </div>

            {/* Detailed Intervention Configuration */}
            <div className="section section-transparent">
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
            <div className="section captured-section">
              <button
                className="section-toggle"
                onClick={() => setShowCaptured(!showCaptured)}
              >
                <h2>Captured Activations ({capturedActivations.length})</h2>
                {showCaptured ? <ChevronUp /> : <ChevronDown />}
              </button>

              {showCaptured && (
                <div className="captured-list">
                  {capturedActivations.length === 0 ? (
                    <div className="empty-state">No captured activations yet</div>
                  ) : (
                    <>
                      <button
                        className="clear-all-btn"
                        onClick={handleClearAllCaptured}
                      >
                        Clear All
                      </button>
                      {capturedActivations.map(activation => (
                        <div key={activation.id} className="captured-item">
                          <div className="captured-info">
                            <span className="captured-model">{activation.model_id}</span>
                            <span className="captured-prompt">"{activation.prompt.slice(0, 50)}..."</span>
                            <span className="captured-meta">{activation.num_tokens} tokens, {activation.num_layers} layers</span>
                          </div>
                          <button
                            className="delete-btn"
                            onClick={() => handleDeleteCaptured(activation.id)}
                          >
                            <Trash2 className="delete-icon" />
                          </button>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Saved Experiments */}
            <div className="section">
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

          {/* Right Column: Results */}
          <div className="right-column">
            <div className="section results-section">
              <h2>Results</h2>
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
                <div className="results-placeholder">
                  <div className="placeholder-content">
                    <Zap className="placeholder-icon" />
                    <p>Configure patches and run an experiment to see results</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .patching-playground {
          min-height: 125vh;
          padding-top: 70px;
        }

        .playground-content {
          max-width: 1800px;
          margin: 0 auto;
          padding: 2rem;
        }

        .playground-header {
          margin-bottom: 2rem;
        }

        .header-title {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.5rem;
        }

        .title-icon {
          width: 28px;
          height: 28px;
          color: rgba(59, 130, 246, 0.8);
        }

        .header-title h1 {
          margin: 0;
          font-size: 1.75rem;
          font-weight: 600;
          color: #ffffff;
          letter-spacing: -0.02em;
        }

        .header-description {
          margin: 0;
          font-size: 0.9rem;
          color: #ffffff;
          max-width: 700px;
          line-height: 1.5;
        }

        .error-banner {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.9);
          margin-bottom: 1.5rem;
        }

        .error-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
        }

        .dismiss-btn {
          margin-left: auto;
          padding: 0.25rem 0.75rem;
          background: transparent;
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.8);
          font-size: 0.8rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .dismiss-btn:hover {
          background: rgba(239, 68, 68, 0.1);
        }

        .playground-grid {
          display: grid;
          grid-template-columns: 350px 480px 1fr;
          gap: 1.5rem;
          align-items: start;
        }

        .section {
          margin-bottom: 1.5rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1.25rem;
        }

        .section-transparent {
          background: none;
          border: none;
          padding: 0;
        }

        .section h2 {
          margin: 0 0 1rem 0;
          font-size: 0.9rem;
          font-weight: 600;
          color: #ffffff;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .model-select {
          width: 100%;
          padding: 0.75rem 1rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.9rem;
          cursor: pointer;
        }

        .model-select:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .model-info-row {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-top: 0.75rem;
        }

        .info-badge {
          padding: 0.25rem 0.5rem;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.7rem;
          font-weight: 500;
          text-transform: uppercase;
        }

        .info-text {
          font-size: 0.8rem;
          color: #ffffff;
        }

        .prompt-group {
          margin-bottom: 1rem;
        }

        .prompt-group label {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          margin-bottom: 0.5rem;
        }

        .label-main {
          font-size: 0.85rem;
          font-weight: 500;
          color: #ffffff;
        }

        .label-hint {
          font-size: 0.75rem;
          color: #ffffff;
        }

        .prompt-group textarea {
          width: 100%;
          padding: 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.9rem;
          font-family: inherit;
          resize: vertical;
          min-height: 80px;
        }

        .prompt-group textarea:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .ablation-notice {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          background: rgba(168, 85, 247, 0.1);
          border: 1px solid rgba(168, 85, 247, 0.25);
          color: rgba(168, 85, 247, 0.9);
          margin-bottom: 1rem;
          font-size: 0.85rem;
        }

        .notice-icon {
          font-size: 1rem;
        }

        .settings-row {
          display: flex;
          gap: 1rem;
        }

        .setting-group {
          flex: 1;
        }

        .setting-group label {
          display: block;
          font-size: 0.8rem;
          color: #ffffff;
          margin-bottom: 0.375rem;
        }

        .setting-group input {
          width: 100%;
          padding: 0.625rem 0.75rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.15);
          color: #ffffff;
          font-size: 0.9rem;
        }

        .setting-group input:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.5);
        }

        .settings-hint {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 0.75rem;
          font-size: 0.75rem;
          color: #ffffff;
        }

        .hint-icon {
          width: 14px;
          height: 14px;
        }

        .run-experiment-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          padding: 1rem 1.5rem;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.25);
          color: #ffffff;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.025em;
        }

        .run-experiment-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.4);
        }

        .run-experiment-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-icon {
          width: 18px;
          height: 18px;
        }

        .spinner {
          width: 18px;
          height: 18px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top-color: #ffffff;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .section-toggle {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.75rem 0;
          background: transparent;
          border: none;
          cursor: pointer;
          color: #ffffff;
        }

        .section-toggle h2 {
          margin: 0;
        }

        .captured-list {
          margin-top: 0.75rem;
        }

        .empty-state {
          padding: 1.5rem;
          text-align: center;
          color: #ffffff;
          font-size: 0.85rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .clear-all-btn {
          width: 100%;
          padding: 0.5rem;
          background: transparent;
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.8);
          font-size: 0.8rem;
          cursor: pointer;
          margin-bottom: 0.75rem;
          transition: all 0.2s;
        }

        .clear-all-btn:hover {
          background: rgba(239, 68, 68, 0.1);
        }

        .captured-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          margin-bottom: 0.5rem;
        }

        .captured-info {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          flex: 1;
          min-width: 0;
        }

        .captured-model {
          font-size: 0.8rem;
          font-weight: 500;
          color: rgba(59, 130, 246, 0.9);
        }

        .captured-prompt {
          font-size: 0.8rem;
          color: #ffffff;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .captured-meta {
          font-size: 0.7rem;
          color: #ffffff;
        }

        .delete-btn {
          padding: 0.375rem;
          background: transparent;
          border: none;
          color: #ffffff;
          cursor: pointer;
          transition: color 0.2s;
        }

        .delete-btn:hover {
          color: rgba(239, 68, 68, 0.8);
        }

        .delete-icon {
          width: 16px;
          height: 16px;
        }

        .results-section {
          height: 100%;
        }

        .results-placeholder {
          height: 400px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.03);
          border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .placeholder-content {
          text-align: center;
          color: #ffffff;
        }

        .placeholder-icon {
          width: 48px;
          height: 48px;
          margin-bottom: 1rem;
          opacity: 0.3;
        }

        .placeholder-content p {
          margin: 0;
          font-size: 0.9rem;
        }

        @media (max-width: 1400px) {
          .playground-grid {
            grid-template-columns: 1fr 1fr;
          }

          .right-column {
            grid-column: span 2;
          }
        }

        @media (max-width: 900px) {
          .playground-grid {
            grid-template-columns: 1fr;
          }

          .right-column {
            grid-column: span 1;
          }
        }

        /* Light mode */
        [data-bg="light"] .header-title h1 { color: #1d1d1f; }
        [data-bg="light"] .header-description { color: #1d1d1f; }
        [data-bg="light"] .section { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .section h2 { color: #1d1d1f; }
        [data-bg="light"] .model-select { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .info-badge { background: rgba(0,0,0,0.05); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .info-text { color: #1d1d1f; }
        [data-bg="light"] .label-main { color: #1d1d1f; }
        [data-bg="light"] .label-hint { color: #1d1d1f; }
        [data-bg="light"] .prompt-group textarea { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .setting-group label { color: #1d1d1f; }
        [data-bg="light"] .setting-group input { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.12); color: #1d1d1f; }
        [data-bg="light"] .settings-hint { color: #1d1d1f; }
        [data-bg="light"] .run-experiment-btn { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); color: #1d1d1f; }
        [data-bg="light"] .run-experiment-btn:hover:not(:disabled) { background: rgba(0,0,0,0.1); border-color: rgba(0,0,0,0.25); }
        [data-bg="light"] .spinner { border-color: rgba(0,0,0,0.15); border-top-color: #1d1d1f; }
        [data-bg="light"] .section-toggle { color: #1d1d1f; }
        [data-bg="light"] .empty-state { color: #1d1d1f; background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.15); }
        [data-bg="light"] .captured-item { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.08); }
        [data-bg="light"] .captured-prompt { color: #1d1d1f; }
        [data-bg="light"] .captured-meta { color: #1d1d1f; }
        [data-bg="light"] .delete-btn { color: #1d1d1f; }
        [data-bg="light"] .results-placeholder { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.15); }
        [data-bg="light"] .placeholder-content { color: #1d1d1f; }
      `}</style>
    </div>
  )
}

export default PatchingPlayground
