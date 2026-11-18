/**
 * Text generation component with model selection and streaming support.
 * Supports both local and Modal cloud inference with cold start handling.
 */

import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Square } from 'lucide-react'
import { useStreamingGeneration } from '../../hooks/useGeneration'
import { useModels } from '../../hooks/useModels'
import { apiClient, type FinishReason } from '../../api/client'
import { useAuthStore } from '../../stores/authStore'
import type { MonitorProbeInfo, MonitorTrace, SavedProbe } from '../../types/probes'
import { Dropdown, type DropdownGroup } from '../common/Dropdown'
import { Spinner } from '../common/Spinner'
import { ProbeMonitor } from '../probes/ProbeMonitor'
import { monitorProbeInfo } from '../probes/probeUtils'
import { useAssistantPage } from '../../hooks/useAssistantPage'
import { generatorBridge } from '../assistant/pages/generate'

export type GenerationMode = 'visualize' | 'tokens'

export interface PlainGenerationResult {
  text: string
  prompt: string
  model: string
  modelName: string
  maxTokens: number
  temperature: number
  topK: number
  untilEos: boolean
  generatedTokens: number | null
  finishReason: FinishReason | null
  activationId?: string
  // A probe's scores of every token, if one monitored the generation
  monitor?: MonitorTrace | null
  monitorProbe?: MonitorProbeInfo | null
}

// A generation in progress, for showing its text as it streams in
export interface GenerationStreamState {
  mode: GenerationMode
  phase: 'waiting' | 'streaming' | 'capturing' | 'stopping'
  text: string
  monitor?: MonitorTrace | null
  monitorProbe?: MonitorProbeInfo | null
}

interface TextGeneratorProps {
  defaultModel?: string
  defaultPrompt?: string
  onGenerate?: (text: string) => void
  onActivationGenerated?: (activationId: string) => void
  onModelChange?: (model: string) => void
  onPlainTextGenerated?: (result: PlainGenerationResult) => void
  // Called as a generation progresses, and with null once it has ended
  onStreamUpdate?: (stream: GenerationStreamState | null) => void
  externalModel?: string
  onShowPurchaseCredits?: () => void
}

// Cold start threshold - show "warming up" message after this delay
const COLD_START_THRESHOLD_MS = 3000

// Token slider range, and the cap when generating until the end-of-sequence token
const MAX_TOKENS_SLIDER = 300
const EOS_MAX_TOKENS = 1000

export const TextGenerator: React.FC<TextGeneratorProps> = ({
  defaultModel = 'nano',
  defaultPrompt = '',
  onGenerate,
  onActivationGenerated,
  onModelChange,
  onPlainTextGenerated,
  onStreamUpdate,
  externalModel,
  onShowPurchaseCredits,
}) => {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [internalModel, setInternalModel] = useState<string>(defaultModel)

  // Use external model if provided, otherwise use internal state
  const model = externalModel ?? internalModel

  const handleModelChange = (newModel: string) => {
    setInternalModel(newModel)
    onModelChange?.(newModel)
  }
  const [maxTokens, setMaxTokens] = useState(100)
  const [untilEos, setUntilEos] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [topK, setTopK] = useState(40)
  const [isWarmingUp, setIsWarmingUp] = useState(false)
  const warmupTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Generation mode dropdown state
  const [generationMode, setGenerationMode] = useState<GenerationMode>('visualize')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { models, modelNames, modelFamilies, openSourceModels, loading: modelsLoading, error: modelsError } = useModels()
  const {
    text, loading, streaming, capturing, stopping, error, insufficientCredits, monitor,
    generate, stop, reset, clearInsufficientCredits,
  } = useStreamingGeneration()
  const busy = loading || streaming || capturing || stopping

  // Saved probes that can monitor this model's generations, the one chosen, and the one
  // monitoring the latest generation
  const [monitorProbes, setMonitorProbes] = useState<SavedProbe[]>([])
  const [monitorId, setMonitorId] = useState('')
  const [activeMonitor, setActiveMonitor] = useState<MonitorProbeInfo | null>(null)
  useEffect(() => {
    if (!isAuthenticated || !model) {
      setMonitorProbes([])
      return
    }
    let cancelled = false
    apiClient
      .listProbes(model)
      .then((list) => {
        if (cancelled) return
        const usable = list.filter((p) => p.model_available)
        setMonitorProbes(usable)
        setMonitorId((current) => (usable.some((p) => String(p.id) === current) ? current : ''))
      })
      .catch(() => !cancelled && setMonitorProbes([]))
    return () => {
      cancelled = true
    }
  }, [isAuthenticated, model])
  const monitorProbe = monitorProbes.find((p) => String(p.id) === monitorId) ?? null

  // Instruct models generate until their end-of-turn token by default, as chat models do. When
  // the assistant sets a model and "until EOS" together, its choice applies after the default.
  const isInstruct = openSourceModels.find(m => m.id === model)?.is_instruct ?? false
  const assistModelChange = useRef(false)
  const assistUntilEos = useRef<boolean | null>(null)
  useEffect(() => {
    const chosen = assistUntilEos.current
    assistModelChange.current = false
    assistUntilEos.current = null
    setUntilEos(chosen ?? isInstruct)
  }, [model, isInstruct])

  useAssistantPage('generate', generatorBridge({
    prompt,
    setPrompt,
    model,
    setModel: (m) => {
      if (m === model) return
      assistModelChange.current = true
      handleModelChange(m)
    },
    maxTokens,
    setMaxTokens,
    untilEos,
    setUntilEos: (on) => {
      if (assistModelChange.current) assistUntilEos.current = on
      else setUntilEos(on)
    },
    temperature,
    setTemperature,
    topK,
    setTopK,
    mode: generationMode,
    setMode: setGenerationMode,
    busy,
  }))

  // Clear warmup timer on unmount
  useEffect(() => {
    return () => {
      if (warmupTimerRef.current) {
        clearTimeout(warmupTimerRef.current)
      }
    }
  }, [])

  // Report the running generation's progress to the parent (the latest callback, without
  // re-reporting whenever the parent re-renders)
  const onStreamUpdateRef = useRef(onStreamUpdate)
  onStreamUpdateRef.current = onStreamUpdate
  const phase: GenerationStreamState['phase'] =
    stopping ? 'stopping' : capturing ? 'capturing' : streaming ? 'streaming' : 'waiting'
  useEffect(() => {
    onStreamUpdateRef.current?.(
      busy ? { mode: generationMode, phase, text, monitor, monitorProbe: activeMonitor } : null,
    )
  }, [busy, phase, text, generationMode, monitor, activeMonitor])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      return
    }

    const settings = {
      prompt: prompt.trim(),
      model,
      modelName: modelNames[model] || model,
      maxTokens: untilEos ? EOS_MAX_TOKENS : maxTokens,
      temperature,
      topK,
      untilEos,
    }

    reset()
    setIsWarmingUp(false)
    const monitorInfo = monitorProbe ? monitorProbeInfo(monitorProbe) : null
    setActiveMonitor(monitorInfo)

    // Start a timer to show "warming up" message if response takes too long
    warmupTimerRef.current = setTimeout(() => {
      setIsWarmingUp(true)
    }, COLD_START_THRESHOLD_MS)

    try {
      const result = await generate({
        prompt: settings.prompt,
        model,
        max_tokens: settings.maxTokens,
        temperature,
        top_k: topK,
        stop_at_eos: untilEos,
        probe_id: monitorProbe?.id,
      }, { withActivations: generationMode === 'visualize' })

      if (!result) {
        return
      }

      onGenerate?.(result.text)

      if (result.activationId) {
        onActivationGenerated?.(result.activationId)
      }

      // Stopped before generating anything: nothing to keep
      if (result.text || result.generatedTokens) {
        onPlainTextGenerated?.({
          ...settings,
          text: result.text,
          generatedTokens: result.generatedTokens,
          finishReason: result.finishReason,
          activationId: result.activationId ?? undefined,
          monitor: result.monitor,
          monitorProbe: monitorInfo,
        })
      }
    } catch (err) {
      console.error('Generation error:', err)
    } finally {
      // Clear warmup timer
      if (warmupTimerRef.current) {
        clearTimeout(warmupTimerRef.current)
        warmupTimerRef.current = null
      }
      setIsWarmingUp(false)
    }
  }

  const handleReset = () => {
    reset()
    setPrompt('')
  }

  const handleGenerateClick = () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    handleGenerate()
  }

  return (
    <div className="text-generator">
      <section className="oz-section">
        <h2 className="oz-title">
          Model
          {modelsLoading && <span className="oz-title__end">Loading models...</span>}
          {modelsError && <span className="oz-title__end tg-error">Error: {modelsError}</span>}
        </h2>
        <div className="control-group" data-assist="generate.model">
          <Dropdown
            id="model-select"
            value={model}
            onChange={handleModelChange}
            disabled={busy || modelsLoading || models.length === 0}
            groups={(() => {
              const groups: DropdownGroup[] = []
              const base = models.filter(m => m === 'nano' || m === 'mini')
              if (base.length > 0) {
                groups.push({ label: 'Ozera Models', options: base.map(m => ({ value: m, label: `ozera-${m}` })) })
              }
              const os = models.filter(m => { const f = modelFamilies[m]; return f && f !== 'ozera' })
              if (os.length > 0) {
                groups.push({ label: 'Open Source', options: os.map(m => ({ value: m, label: m })) })
              }
              const custom = models.filter(m => { const f = modelFamilies[m]; return (f === 'ozera' || !f) && m !== 'nano' && m !== 'mini' })
              if (custom.length > 0) {
                groups.push({ label: 'Custom Models', options: custom.map(m => ({ value: m, label: modelNames[m] || m })) })
              }
              if (groups.length === 0) {
                groups.push({ label: 'Models', options: [{ value: '', label: 'No models available' }] })
              }
              return groups
            })()}
          />
        </div>
      </section>

      <section className="oz-section">
        <h2 className="oz-title">Sampling</h2>
        <div className="control-group" data-assist="generate.max_tokens">
          <label htmlFor="max-tokens">Tokens</label>
          <div className="control-row">
            <input
              id="max-tokens"
              type="range"
              min="1"
              max={MAX_TOKENS_SLIDER}
              step="1"
              value={untilEos ? MAX_TOKENS_SLIDER : maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              disabled={busy || untilEos}
            />
            <input
              type="number"
              className="control-number-input"
              min={1}
              max={MAX_TOKENS_SLIDER}
              value={untilEos ? EOS_MAX_TOKENS : maxTokens}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                if (!isNaN(v)) setMaxTokens(Math.max(1, Math.min(MAX_TOKENS_SLIDER, v)))
              }}
              disabled={busy || untilEos}
            />
          </div>
          <label className={`eos-toggle ${busy ? 'disabled' : ''}`} data-assist="generate.until_eos">
            <input
              type="checkbox"
              checked={untilEos}
              onChange={(e) => setUntilEos(e.target.checked)}
              disabled={busy}
            />
            <span className="eos-toggle-title">Generate until EOS</span>
          </label>
        </div>

        <div className="control-group" data-assist="generate.temperature">
          <label htmlFor="temperature">Temperature</label>
          <div className="control-row">
            <input
              id="temperature"
              type="range"
              min="0"
              max="2.0"
              step="0.01"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              disabled={busy}
            />
            <input
              type="number"
              className="control-number-input"
              min={0}
              max={2}
              step={0.01}
              value={temperature}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (!isNaN(v)) setTemperature(Math.max(0, Math.min(2, v)))
              }}
              disabled={busy}
            />
          </div>
        </div>

        <div className="control-group" data-assist="generate.top_k">
          <label htmlFor="top-k">Top-K</label>
          <div className="control-row">
            <input
              id="top-k"
              type="range"
              min="1"
              max="100"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={busy}
            />
            <input
              type="number"
              className="control-number-input"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                if (!isNaN(v)) setTopK(Math.max(1, Math.min(100, v)))
              }}
              disabled={busy}
            />
          </div>
        </div>
      </section>

      {isAuthenticated && (
        <section className="oz-section">
          <h2 className="oz-title">Probe monitor</h2>
          <div className="control-group">
            <Dropdown
              id="probe-monitor"
              value={monitorId}
              onChange={setMonitorId}
              disabled={busy || monitorProbes.length === 0}
              options={[
                { value: '', label: monitorProbes.length === 0 ? 'No saved probes for this model' : 'Off' },
                ...monitorProbes.map((p) => ({ value: String(p.id), label: `${p.name} · layer ${p.layer}` })),
              ]}
            />
            <span className="probe-monitor-hint">
              {monitorProbes.length === 0 ? (
                <>
                  Score every token as it&apos;s generated with a probe.{' '}
                  <button type="button" className="probe-monitor-link" onClick={() => navigate('/probes')}>
                    Save one in the Probe Lab
                  </button>
                </>
              ) : (
                'Scores every token as the model reads it, at no extra cost.'
              )}
            </span>
          </div>
        </section>
      )}

      <section className="oz-section">
      <h2 className="oz-title">Prompt</h2>
      <div className="prompt-area" data-assist="generate.prompt">
        <textarea
          id="prompt"
          aria-label="Prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          rows={6}
          disabled={busy}
        />
      </div>

      <div className="actions">
        {busy ? (
          <button
            onClick={stop}
            disabled={stopping || capturing}
            className="oz-btn oz-btn--block btn-stop"
            title={capturing ? 'Generation is done; capturing activations' : 'Stop generating'}
          >
            {stopping ? 'Stopping...' : capturing ? 'Capturing...' : (
              <>
                <Square fill="currentColor" />
                Stop
              </>
            )}
          </button>
        ) : (
          <div className="generate-dropdown" ref={dropdownRef} data-assist="generate.generate_button">
            <button
              onClick={handleGenerateClick}
              disabled={!prompt.trim()}
              className="oz-btn oz-btn--arrow btn-generate-main"
            >
              {generationMode === 'visualize' ? 'Generate + Visualize' : 'Generate'}
            </button>
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="oz-btn oz-btn--ghost btn-generate-toggle"
              data-assist="generate.mode"
              title="Generation mode"
              aria-label="Generation mode"
              aria-expanded={dropdownOpen}
            >
              <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
                <path d="M1 1l4 4 4-4" />
              </svg>
            </button>
            {dropdownOpen && (
              <div className="generate-dropdown-menu oz-menu">
                <button
                  onClick={() => {
                    setGenerationMode('visualize')
                    setDropdownOpen(false)
                  }}
                  className={`oz-menu__item dropdown-item ${generationMode === 'visualize' ? 'active' : ''}`}
                >
                  Generate + Visualize
                  <span className="dropdown-item-desc">Generate tokens and capture activations for visualization</span>
                </button>
                <button
                  onClick={() => {
                    setGenerationMode('tokens')
                    setDropdownOpen(false)
                  }}
                  className={`oz-menu__item dropdown-item ${generationMode === 'tokens' ? 'active' : ''}`}
                >
                  Generate
                  <span className="dropdown-item-desc">Generate tokens without visualization</span>
                </button>
              </div>
            )}
          </div>
        )}
        <button
          onClick={handleReset}
          disabled={busy}
          className="oz-btn oz-btn--ghost oz-btn--block"
        >
          Reset
        </button>
      </div>

      {isWarmingUp && loading && (
        <div className="warmup-message oz-note">
          <Spinner />
          <span>Model is warming up. This may take 10-30 seconds on first request...</span>
        </div>
      )}

      {insufficientCredits && (
        <div className="insufficient-credits-message oz-note oz-note--warn">
          <div className="insufficient-credits-content">
            <strong>Insufficient Credits</strong>
            <p>You don't have enough credits to generate text. Please add more credits to continue.</p>
            {onShowPurchaseCredits && (
              <button
                className="oz-link"
                onClick={() => {
                  clearInsufficientCredits()
                  onShowPurchaseCredits()
                }}
              >
                Add Credits
              </button>
            )}
          </div>
        </div>
      )}

      {error && !insufficientCredits && (
        <div className="error-message oz-note oz-note--error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {text && !onPlainTextGenerated && (
        <div className="output-area">
          <div className="output-header">
            <h3>Generated Text</h3>
            {(streaming || capturing) && <span className="streaming-indicator">●</span>}
          </div>
          <div className="output-content">
            {text}
            {streaming && <span className="cursor">|</span>}
          </div>
        </div>
      )}

      {monitor && activeMonitor && !onPlainTextGenerated && (
        <ProbeMonitor trace={monitor} probe={activeMonitor} live={busy} />
      )}
      </section>

      <style>{`
        .text-generator {
          max-width: 100%;
        }

        .text-generator .tg-error {
          color: var(--oz-danger);
        }

        .text-generator .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .text-generator .control-group + .control-group {
          margin-top: 1.35rem;
        }

        .text-generator .control-group > label:not(.eos-toggle) {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }

        /* A slider with its value beside it */
        .text-generator .control-row {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .text-generator .control-row input[type="range"] {
          flex: 1;
          min-width: 0;
        }

        .text-generator .control-number-input {
          width: 4.75rem;
          padding: 0.4rem 0.5rem;
          background: transparent;
          border: 1px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.85rem;
          text-align: center;
          outline: none;
          transition: border-color 0.2s ease;
        }

        .text-generator .control-number-input:hover:not(:disabled):not(:focus) {
          border-color: var(--oz-fg-faint);
        }

        .text-generator .control-number-input:focus {
          border-color: var(--oz-fg);
        }

        .text-generator .control-number-input:disabled {
          border-color: var(--oz-rule);
          color: var(--oz-fg-faint);
          cursor: not-allowed;
        }

        .text-generator .eos-toggle {
          display: flex;
          align-items: center;
          gap: 0.55rem;
          margin-top: 0.3rem;
          font-size: 0.875rem;
          cursor: pointer;
        }

        .text-generator .eos-toggle.disabled {
          color: var(--oz-fg-faint);
          cursor: not-allowed;
        }

        .text-generator .probe-monitor-hint {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          line-height: 1.45;
        }

        .text-generator .probe-monitor-link {
          padding: 0 0 1px;
          background: none;
          border: none;
          border-bottom: 1px solid var(--oz-fg-faint);
          color: var(--oz-fg);
          font: inherit;
          font-weight: 500;
          cursor: pointer;
          transition: color 0.2s ease, border-color 0.2s ease;
        }

        .text-generator .probe-monitor-link:hover {
          border-color: var(--oz-mark);
          color: var(--oz-mark);
        }

        .text-generator .prompt-area textarea {
          display: block;
          width: 100%;
          min-height: 9.5rem;
          padding: 0.8rem 0.9rem;
          background: transparent;
          border: 1px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          line-height: 1.6;
          resize: vertical;
          outline: none;
          transition: border-color 0.2s ease;
        }

        .text-generator .prompt-area textarea:hover:not(:disabled):not(:focus) {
          border-color: var(--oz-fg-faint);
        }

        .text-generator .prompt-area textarea:focus {
          border-color: var(--oz-fg);
        }

        .text-generator .prompt-area textarea::placeholder {
          color: var(--oz-fg-faint);
        }

        .text-generator .prompt-area textarea:disabled {
          border-color: var(--oz-rule);
          color: var(--oz-fg-dim);
          cursor: not-allowed;
        }

        .text-generator .actions {
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-top: 1rem;
        }

        /* The run button and, beside it, the mode it runs in */
        .text-generator .generate-dropdown {
          position: relative;
          display: flex;
          gap: 6px;
        }

        .text-generator .btn-generate-main {
          flex: 1;
          min-width: 0;
        }

        .text-generator .btn-generate-toggle {
          flex: none;
          width: 2.9rem;
          padding: 0;
        }

        .text-generator .btn-stop svg {
          width: 0.7em;
          height: 0.7em;
        }

        .text-generator .generate-dropdown-menu {
          position: absolute;
          top: calc(100% + 6px);
          left: 0;
          right: 0;
          z-index: 100;
        }

        .text-generator .dropdown-item {
          flex-direction: column;
          align-items: flex-start;
          gap: 0.15rem;
          padding: 0.7rem 0.8rem;
        }

        /* The mode in use, marked */
        .text-generator .dropdown-item.active {
          box-shadow: inset 2px 0 0 var(--oz-mark);
        }

        .text-generator .dropdown-item-desc {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 400;
          line-height: 1.4;
          white-space: normal;
          transition: color 0.15s ease;
        }

        .text-generator .dropdown-item:hover .dropdown-item-desc {
          color: inherit;
        }

        .text-generator .oz-note {
          margin-top: 1rem;
        }

        .text-generator .warmup-message {
          align-items: center;
        }

        .text-generator .insufficient-credits-content {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 0.35rem;
        }

        .text-generator .insufficient-credits-content strong,
        .text-generator .error-message strong {
          font-weight: 500;
        }

        .text-generator .insufficient-credits-content p {
          margin: 0;
          color: var(--oz-fg-dim);
        }

        .text-generator .error-message {
          display: block;
          overflow-wrap: anywhere;
        }

        /* The text itself, where the page doesn't show outputs of its own */
        .text-generator .output-area {
          margin-top: 1.5rem;
          border-top: 1px solid var(--oz-rule);
        }

        .text-generator .output-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem 0 0.6rem;
        }

        .text-generator .output-header h3 {
          margin: 0;
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
        }

        .text-generator .streaming-indicator {
          color: var(--oz-mark);
          animation: tg-pulse 1.2s ease-in-out infinite;
        }

        .text-generator .output-content {
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          line-height: 1.7;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .text-generator .cursor {
          margin-left: 2px;
          color: var(--oz-mark);
          animation: tg-blink 0.8s infinite;
        }

        @keyframes tg-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.25; }
        }

        @keyframes tg-blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}

export default TextGenerator
