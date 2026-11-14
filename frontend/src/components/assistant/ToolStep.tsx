/**
 * A tool call in the reply's sequence, as a row the way agentic coding tools show them: its
 * status, what it did in plain words (e.g. "Read feature 1234 · Gemma Scope 270M L9"), and a run's
 * charge. A run the chat's allowance started without asking expands to its quote; while a run's
 * result is still in memory, Open shows it on its page again.
 */

import { useState } from 'react'
import { AlertCircle, Check, ChevronRight, Eye, Hourglass, Loader2, MousePointerClick, Play, SkipForward } from 'lucide-react'
import { useAssistantStore } from '../../stores/assistantStore'
import type { ToolChipInfo } from '../../types/assistant'
import { describeTool } from './toolWords'

function charge(value: number): string {
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`
}

function estimate(value: number): string {
  if (value === 0) return 'free'
  return value < 0.1 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`
}

export function ToolStep({ tool }: { tool: ToolChipInfo }) {
  const result = useAssistantStore((s) => s.results[tool.id])
  const [open, setOpen] = useState(false)
  const running = tool.status === 'running' || tool.status === 'pricing'
  const failed = tool.error || tool.status === 'error'
  const quote = tool.auto
  const Icon = running
    ? Loader2
    : failed
      ? AlertCircle
      : tool.status === 'approval'
        ? Hourglass
        : tool.status === 'skipped'
          ? SkipForward
          : tool.kind === 'ui'
            ? MousePointerClick
            : tool.kind === 'read'
              ? Eye
              : tool.kind === 'run'
                ? Play
                : Check
  const end = failed
    ? 'failed'
    : tool.status === 'skipped'
      ? 'skipped'
      : tool.status === 'approval'
        ? 'waiting'
        : tool.kind === 'run' && tool.status === 'done' && tool.charged != null
          ? charge(tool.charged)
          : null
  const [says, detail] = describeTool(tool)
  const head = (
    <>
      <Icon className={running ? 'ast-spin' : undefined} aria-label={running ? 'Running' : undefined} aria-hidden={running ? undefined : true} />
      <span className="ast-step-text">
        <span className="ast-step-says">{says}</span>
        {detail && <span className="ast-step-arg"> {detail}</span>}
      </span>
      {end && <span className="ast-step-end">{end}</span>}
      {quote && <ChevronRight className={`ast-step-chevron${open ? ' open' : ''}`} aria-hidden="true" />}
    </>
  )
  const sentence = detail ? `${says} ${detail}` : says
  const title = failed ? `${sentence}: failed` : tool.status === 'approval' ? `${sentence}: waiting for Run or Skip` : sentence
  return (
    <div className={`ast-step${failed ? ' ast-step--error' : ''}${tool.status === 'skipped' ? ' ast-step--skipped' : ''}`}>
      <div className="ast-step-row">
        {quote ? (
          <button type="button" className="ast-step-main" aria-expanded={open} title={title} onClick={() => setOpen((o) => !o)}>
            {head}
          </button>
        ) : (
          <span className="ast-step-main" title={title}>
            {head}
          </span>
        )}
        {result?.page && !failed && (
          <button type="button" className="ast-step-open" title="Open this result on its page" onClick={() => useAssistantStore.getState().applyResult(result)}>
            Open
          </button>
        )}
      </div>
      {quote && open && (
        <div className="ast-step-details">
          <div className="ast-step-title">{quote.title}</div>
          <dl className="ast-approval-details">
            {quote.details.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          <div className="ast-approval-note">Within allowance · est. {estimate(quote.estimate)}</div>
        </div>
      )}
    </div>
  )
}
