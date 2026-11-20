/**
 * The Circuit Tracing page's left column: the model and its transcoders, the prompt and target,
 * the estimate and the Trace button with the job's progress, and the user's graphs.
 */

import type { CircuitLab } from '../../hooks/useCircuitLab'
import type { CircuitJob } from '../../types/circuits'
import { Dropdown } from '../common/Dropdown'
import { Spinner } from '../common/Spinner'
import { useLabColors } from '../lab/labTheme'
import { percent, showToken, traceEstimateText } from './circuitUtils'

const MAX_PROMPT_CHARS = 600
const MAX_TARGET_CHARS = 40

export function CircuitModelPicker({ lab, disabled }: { lab: CircuitLab; disabled?: boolean }) {
  const sets = lab.model?.transcoders ?? []
  return (
    <div className="ct-model">
      <div data-assist="circuits.model">
        <Dropdown
          value={lab.modelId}
          onChange={lab.setModelId}
          disabled={disabled || lab.loadingModels}
          options={
            lab.loadingModels
              ? [{ value: '', label: 'Loading models…' }]
              : lab.models.map((m) => ({ value: m.model_id, label: `${m.display_name} · ${m.num_layers}L · d=${m.hidden_dim}` }))
          }
        />
      </div>
      <div data-assist="circuits.transcoders">
        <div className="pl-control-label" title="A transcoder per layer stands in for its MLP as sparse features; these are the sets published for this model">
          Transcoders
        </div>
        <Dropdown
          value={lab.setId}
          onChange={lab.setSetId}
          disabled={disabled || sets.length < 2}
          options={sets.map((s) => ({ value: s.id, label: s.name }))}
        />
        {lab.transcoders && (
          <div className="pl-hint ct-source" title={lab.transcoders.reference}>
            {lab.transcoders.width.toLocaleString()} features per layer · {lab.transcoders.source}
          </div>
        )}
      </div>
    </div>
  )
}

export function PromptPanel({ lab, disabled }: { lab: CircuitLab; disabled?: boolean }) {
  return (
    <div className="ct-prompt">
      <div data-assist="circuits.prompt">
        <textarea
          className="pl-textarea"
          value={lab.prompt}
          maxLength={MAX_PROMPT_CHARS}
          onChange={(e) => lab.setPrompt(e.target.value)}
          placeholder="The capital of France is"
          disabled={disabled}
          aria-label="Prompt"
        />
      </div>
      <div data-assist="circuits.target">
        <div className="pl-control-label" title="The next word to explain (its first token). Leave empty for the model’s top prediction">
          Target
        </div>
        <input
          className="pl-input"
          type="text"
          value={lab.target}
          maxLength={MAX_TARGET_CHARS}
          onChange={(e) => lab.setTarget(e.target.value)}
          placeholder="Top prediction"
          disabled={disabled}
        />
      </div>
    </div>
  )
}

function Progress({ job }: { job: CircuitJob }) {
  const colors = useLabColors()
  const fraction = job.progress?.fraction ?? 0
  return (
    <div className="ct-progress" role="status">
      <div className="pl-hint">
        {job.status === 'queued' ? 'Starting' : job.progress?.stage ?? 'Starting a GPU'} · about {Math.max(1, Math.ceil(job.estimated_seconds / 60))} min
      </div>
      <div className="ct-progress__bar">
        <div style={{ width: `${Math.round(fraction * 100)}%`, background: colors.text }} />
      </div>
    </div>
  )
}

export function TraceSettings({ lab }: { lab: CircuitLab }) {
  const colors = useLabColors()
  const estimate = lab.estimate
  const busy = lab.submitting || lab.building != null
  const ready = lab.modelId && lab.prompt.trim().length > 0
  return (
    <div className="ct-settings">
      <div className="ct-estimate">
        <span>Estimate</span>
        <span style={estimate && !estimate.within_limit ? { color: colors.warning } : undefined}>
          {!lab.isAuthenticated ? 'Sign in to run' : estimate ? traceEstimateText(estimate) : '–'}
        </span>
      </div>
      <button
        type="button"
        className={`oz-btn oz-btn--block${busy ? '' : ' oz-btn--arrow'}`}
        data-assist="circuits.trace_button"
        onClick={lab.trace}
        disabled={busy || !ready || (estimate != null && !estimate.within_limit)}
      >
        {busy ? (
          <>
            <Spinner />
            Tracing…
          </>
        ) : (
          'Trace'
        )}
      </button>
      {lab.building && <Progress job={lab.building} />}
    </div>
  )
}

function when(iso: string): string {
  // The backend's timestamps are UTC without a zone
  const date = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`)
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

export function GraphList({ lab }: { lab: CircuitLab }) {
  if (!lab.isAuthenticated) return <div className="pl-hint">Sign in to keep graphs</div>
  if (lab.jobs.length === 0) return <div className="pl-hint">None yet</div>
  return (
    <div className="ct-jobs" data-assist="circuits.graph">
      {lab.jobs.map((job) => {
        const open = lab.graph?.job_id === job.job_id
        const done = job.status === 'completed'
        return (
          <div key={job.job_id} className={`ct-job${open ? ' is-active' : ''}`}>
            <button type="button" className="ct-job__pick" onClick={() => done && lab.openJob(job.job_id)} disabled={!done || lab.loadingGraph != null} aria-pressed={open} title={job.prompt}>
              <span className="ct-job__prompt">{job.prompt}</span>
              <span className="ct-job__meta">
                {done && job.summary
                  ? `${showToken(job.summary.target.token)} · ${percent(job.summary.replacement)} · ${job.model}`
                  : job.status === 'failed'
                    ? `Failed · ${job.model}`
                    : `Building · ${job.model}`}
                {' · '}
                {when(job.created_at)}
              </span>
            </button>
          </div>
        )
      })}
    </div>
  )
}
