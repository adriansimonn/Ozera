/**
 * The tool calls a reply made, waiting for the user: runs with what each runs, its estimate and
 * hold, and in manual mode reads and page actions too (free), with Allow (Run when they're all
 * runs) / Skip. A reply's calls share the card; each can be left out. The backend prices the runs
 * again and checks the budget and balance when they're allowed. (In auto mode calls run at once
 * and show only as their ToolSteps.)
 */

import { useEffect, useMemo, useState } from 'react'
import { Check, Play } from 'lucide-react'
import { useCreditBalance } from '../../hooks/useCredits'
import type { PendingRun } from '../../types/assistant'
import { describeTool } from './toolWords'

function money(value: number): string {
  if (value === 0) return 'free'
  return value < 0.1 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`
}

function needs(run: PendingRun): number {
  return Math.max(run.estimate, run.hold ?? 0)
}

/** A run (priced), rather than a read or page action; an older backend's entries are all runs */
function priced(run: PendingRun): boolean {
  return run.kind !== 'read' && run.kind !== 'ui'
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

  const anyRun = runs.some(priced)

  // A new card (or new prices) starts with every call chosen, and reads the balance again
  useEffect(() => {
    setChosen(new Set(runs.map((r) => r.id)))
    if (anyRun) fetchBalance()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const picked = useMemo(() => runs.filter((r) => chosen.has(r.id)), [runs, chosen])
  const total = picked.reduce((sum, r) => sum + r.estimate, 0)
  const required = picked.reduce((sum, r) => sum + needs(r), 0)
  const available = anyRun ? balance?.available_balance : undefined
  const short = available != null && required > 0 && available < required
  const verb = runs.every(priced) ? 'Run' : 'Allow'

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
    <div className="ast-approval" role="group" aria-label="Tool calls waiting for approval">
      {notice && <div className="ast-approval-notice">{notice}</div>}
      {runs.map((run) => {
        const off = chosen.has(run.id) ? '' : ' ast-approval-run--off'
        if (!priced(run)) {
          const [says, detail] = describeTool({ id: run.id, name: run.name, kind: run.kind, chip: run.chip, status: 'approval' })
          const label = (
            <span className="ast-approval-call">
              <span className="ast-step-says">{says}</span>
              {detail && <span className="ast-step-arg"> {detail}</span>}
            </span>
          )
          return (
            <div key={run.id} className={`ast-approval-run ast-approval-run--call${off}`}>
              <div className="ast-approval-head">
                {runs.length > 1 ? (
                  <label className="ast-approval-title">
                    <input type="checkbox" checked={chosen.has(run.id)} onChange={() => toggle(run.id)} disabled={busy} />
                    {label}
                  </label>
                ) : (
                  <span className="ast-approval-title">{label}</span>
                )}
                <span className="ast-approval-price ast-approval-free">free</span>
              </div>
            </div>
          )
        }
        return (
        <div key={run.id} className={`ast-approval-run${off}`}>
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
        )
      })}
      <div className="ast-approval-footer">
        <span className={short ? 'ast-warn' : undefined}>
          {picked.length ? (anyRun ? money(total) : `${picked.length} chosen`) : 'Nothing chosen'}
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
            {verb === 'Run' ? <Play aria-hidden="true" /> : <Check aria-hidden="true" />}
            {runs.length > 1 ? `${verb} ${picked.length}` : verb}
          </button>
        </div>
      </div>
    </div>
  )
}
