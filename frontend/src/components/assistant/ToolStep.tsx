/**
 * A tool call in the reply's sequence, as a row the way agentic coding tools show them: a cell
 * for its status (lit when done, pulsing while it runs, hollow while it waits), what it did in
 * plain words (e.g. "Read feature 1234 · Gemma Scope 270M L9"), and a run's charge. A run started
 * in auto mode expands to its quote; while a run's result is still in memory, Open shows it on
 * its page again.
 */

import { useState } from 'react'
import { useAssistantStore } from '../../stores/assistantStore'
import type { ToolChipInfo } from '../../types/assistant'
import { Caret } from '../common/Caret'
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
  const state = running ? 'running' : failed ? 'failed' : tool.status === 'approval' ? 'waiting' : tool.status === 'skipped' ? 'skipped' : 'done'
  const end = tool.blocked === 'budget'
    ? 'over budget'
    : failed
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
      <i className="ast-cell" role={running ? 'img' : undefined} aria-label={running ? 'Running' : undefined} aria-hidden={running ? undefined : true} />
      <span className="ast-step-text">
        <span className="ast-step-says">{says}</span>
        {detail && <span className="ast-step-arg"> {detail}</span>}
      </span>
      {end && <span className="ast-step-end">{end}</span>}
      {quote && <Caret />}
    </>
  )
  const sentence = detail ? `${says} ${detail}` : says
  const title = failed ? `${sentence}: failed` : tool.status === 'approval' ? `${sentence}: waiting for approval` : sentence
  return (
    <div className={`ast-step ast-step--${state}`}>
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
          <button type="button" className="oz-link ast-step-open" title="Open this result on its page" onClick={() => useAssistantStore.getState().applyResult(result)}>
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
          <div className="ast-approval-note">Auto · est. {estimate(quote.estimate)}</div>
        </div>
      )}
    </div>
  )
}
