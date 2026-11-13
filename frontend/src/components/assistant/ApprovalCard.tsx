/**
 * The runs a reply asked for, waiting for the user (ASSISTANT_PLAN.md, Phase 2): what each runs,
 * its estimate and hold, and Run / Skip. Several runs from one reply share the card; each can be
 * left out. The backend prices them again and checks the balance when Run is pressed.
 *
 * AutoRunCard is the card of a run the chat's allowance started without asking (Phase 3): what
 * ran, its estimate, and what it was charged.
 */

import { useEffect, useMemo, useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { useCreditBalance } from '../../hooks/useCredits'
import { useAssistantStore } from '../../stores/assistantStore'
import type { PendingRun, ToolChipInfo } from '../../types/assistant'

function money(value: number): string {
  if (value === 0) return 'free'
  return value < 0.1 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`
}

function needs(run: PendingRun): number {
  return Math.max(run.estimate, run.hold ?? 0)
}

function charge(value: number): string {
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`
}

/** A run that started without asking, within the chat's allowance. */
export function AutoRunCard({ tool }: { tool: ToolChipInfo }) {
  // Its full result while it's still in memory, to open on its page again
  const result = useAssistantStore((s) => s.results[tool.id])
  const quote = tool.auto
  if (!quote) return null
  const running = tool.status === 'running'
  const failed = tool.error || tool.status === 'error'
  return (
    <div className="ast-approval ast-approval--auto" role="group" aria-label="Run within the allowance">
      <div className="ast-approval-run">
        <div className="ast-approval-head">
          <span className="ast-approval-title">{quote.title}</span>
          <span className={`ast-approval-price${failed ? ' ast-warn' : ''}`}>
            {running ? <Loader2 className="ast-spin" aria-label="Running" /> : failed ? 'failed' : tool.charged != null ? charge(tool.charged) : money(quote.estimate)}
          </span>
        </div>
        <dl className="ast-approval-details">
          {quote.details.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <div className="ast-approval-auto-foot">
          <span className="ast-approval-note">Within allowance · est. {money(quote.estimate)}</span>
          {result?.page && !failed && (
            <button type="button" className="ast-link" onClick={() => useAssistantStore.getState().applyResult(result)}>
              Open
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export function ApprovalCard({
  runs,
  busy,
  notice,
  onDecide,
  onAddCredits,
}: {
  runs: PendingRun[]
  busy: boolean
  notice: string | null
  onDecide: (decisions: Record<string, 'run' | 'skip'>) => void
  onAddCredits: () => void
}) {
  const { balance, fetchBalance } = useCreditBalance()
  const [chosen, setChosen] = useState<Set<string>>(() => new Set(runs.map((r) => r.id)))
  const key = runs.map((r) => `${r.id}:${r.estimate}:${r.hold}`).join('|')

  // A new card (or new prices) starts with every run chosen, and reads the balance again
  useEffect(() => {
    setChosen(new Set(runs.map((r) => r.id)))
    fetchBalance()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const picked = useMemo(() => runs.filter((r) => chosen.has(r.id)), [runs, chosen])
  const total = picked.reduce((sum, r) => sum + r.estimate, 0)
  const required = picked.reduce((sum, r) => sum + needs(r), 0)
  const available = balance?.available_balance
  const short = available != null && required > 0 && available < required

  const decide = (run: boolean) => {
    const decisions: Record<string, 'run' | 'skip'> = {}
    runs.forEach((r) => {
      decisions[r.id] = run && chosen.has(r.id) ? 'run' : 'skip'
    })
    onDecide(decisions)
  }

  const toggle = (id: string) =>
    setChosen((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <div className="ast-approval" role="group" aria-label="Runs waiting for approval">
      {notice && <div className="ast-approval-notice">{notice}</div>}
      {runs.map((run) => (
        <div key={run.id} className={`ast-approval-run${chosen.has(run.id) ? '' : ' ast-approval-run--off'}`}>
          <div className="ast-approval-head">
            {runs.length > 1 ? (
              <label className="ast-approval-title">
                <input type="checkbox" checked={chosen.has(run.id)} onChange={() => toggle(run.id)} disabled={busy} />
                <span>{run.title}</span>
              </label>
            ) : (
              <span className="ast-approval-title">{run.title}</span>
            )}
            <span className="ast-approval-price">{money(run.estimate)}</span>
          </div>
          <dl className="ast-approval-details">
            {run.details.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          <div className="ast-approval-note">
            {run.hold != null && run.hold > run.estimate ? `Needs ${money(run.hold)} of your balance to start. ` : ''}
            {run.note}
          </div>
        </div>
      ))}
      <div className="ast-approval-footer">
        <span className={short ? 'ast-warn' : undefined}>
          {picked.length ? `${money(total)}` : 'Nothing chosen'}
          {available != null ? ` · balance $${available.toFixed(2)}` : ''}
        </span>
        {short && (
          <button type="button" className="ast-link" onClick={onAddCredits}>
            Add credits
          </button>
        )}
        <div className="ast-approval-actions">
          <button type="button" className="ast-approval-skip" onClick={() => decide(false)} disabled={busy}>
            Skip
          </button>
          <button type="button" className="ast-approval-go" onClick={() => decide(true)} disabled={busy || !picked.length || short}>
            <Play aria-hidden="true" />
            {runs.length > 1 ? `Run ${picked.length}` : 'Run'}
          </button>
        </div>
      </div>
    </div>
  )
}
