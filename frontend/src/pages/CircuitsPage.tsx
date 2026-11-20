import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { CircuitViewTabs, CircuitViews } from '../components/circuits/CircuitViews'
import { CircuitModelPicker, GraphList, PromptPanel, TraceSettings } from '../components/circuits/TracePanel'
import { LabKitStyles } from '../components/lab/chartKit'
import { LabPageStyles } from '../components/lab/LabPageStyles'
import { useCircuitLab } from '../hooks/useCircuitLab'

interface CircuitsPageProps {
  onShowPurchaseCredits?: () => void
}

/**
 * Circuit Tracing: the attribution graph of a prompt's next word, built from the model's
 * transcoders, and interventions on its nodes that test it.
 */
export default function CircuitsPage({ onShowPurchaseCredits }: CircuitsPageProps) {
  const lab = useCircuitLab(onShowPurchaseCredits)
  const busy = lab.submitting || lab.building != null
  return (
    <div className="circuit-lab lab-kit lab-page oz-page">
      <NavBar />

      <div className="oz-page__body">
        <div className="oz-side">
          <div className="oz-scroll">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              <CircuitViewTabs lab={lab} />
            </section>
            {lab.error && (
              <div className="pl-error oz-note oz-note--error" role="alert">
                <AlertCircle aria-hidden="true" />
                <span>{lab.error}</span>
                <button type="button" className="oz-link" onClick={() => lab.setError(null)}>
                  Dismiss
                </button>
              </div>
            )}
            <section className="oz-section">
              <h2 className="oz-title">Model</h2>
              <CircuitModelPicker lab={lab} disabled={busy} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Prompt</h2>
              <PromptPanel lab={lab} disabled={busy} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Trace</h2>
              <TraceSettings lab={lab} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Graphs</h2>
              <GraphList lab={lab} />
            </section>
          </div>
        </div>

        <div className="oz-main">
          <div className="oz-scroll">
            <CircuitViews lab={lab} />
          </div>
        </div>
      </div>

      <LabKitStyles />
      <LabPageStyles />
      <style>{STYLES}</style>
    </div>
  )
}

// Scoped under the page's root class
const STYLES = `
  .circuit-lab .pl-view-tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }

  .circuit-lab .ct-model, .circuit-lab .ct-prompt { display: flex; flex-direction: column; gap: 1rem; }
  .circuit-lab .ct-source { margin-top: 0.45rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .circuit-lab .ct-prompt .pl-textarea { min-height: 96px; }
  .circuit-lab .ct-estimate {
    display: flex; justify-content: space-between; gap: 0.75rem; margin: 0 0 0.9rem; color: var(--oz-fg-dim); font-size: 0.85rem;
  }
  .circuit-lab .ct-estimate span:last-child { color: var(--oz-fg); text-align: right; font-variant-numeric: tabular-nums; }
  .circuit-lab .ct-progress { margin-top: 0.9rem; }
  .circuit-lab .ct-progress__bar { height: 3px; margin-top: 0.5rem; background: var(--oz-rule); }
  .circuit-lab .ct-progress__bar > div { height: 100%; transition: width 0.4s; }

  /* The user's graphs: a ruled list, the open one marked */
  .circuit-lab .ct-job { position: relative; border-top: 1px solid var(--oz-rule); }
  .circuit-lab .ct-job:last-child { border-bottom: 1px solid var(--oz-rule); }
  .circuit-lab .ct-job.is-active::after { content: ''; position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; background: var(--oz-mark); }
  .circuit-lab .ct-job__pick {
    display: flex; flex-direction: column; gap: 0.15rem; width: 100%; padding: 0.6rem 0; background: none; border: none;
    color: var(--oz-fg-dim); text-align: left; cursor: pointer; transition: color 0.2s ease;
  }
  .circuit-lab .ct-job__pick:hover:not(:disabled), .circuit-lab .ct-job.is-active .ct-job__pick { color: var(--oz-fg); }
  .circuit-lab .ct-job__pick:disabled { cursor: default; }
  .circuit-lab .ct-job__prompt { font-size: 0.9rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .circuit-lab .ct-job__meta { font-size: 0.78rem; color: var(--oz-fg-dim); font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* The graph scrolls sideways when the prompt has more tokens than fit */
  .circuit-lab .ct-graph { overflow-x: auto; overflow-y: hidden; }
  .circuit-lab .ct-graph__rows { position: sticky; left: 0; z-index: 2; background: var(--oz-bg); pointer-events: none; }
  .circuit-lab .ct-graph-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.9rem; }

  .circuit-lab .ct-mono, .circuit-lab .ct-token { font-family: var(--oz-mono); font-size: 0.8rem; white-space: pre; }
  .circuit-lab .ct-token { padding: 0.1rem 0.3rem; border: 1px solid var(--oz-rule); }
  .circuit-lab .ct-token--dim { color: var(--oz-fg-dim); }
  .circuit-lab .ct-token.is-here { outline: 1.5px solid var(--oz-fg); outline-offset: -1px; }
  .circuit-lab .ct-chips, .circuit-lab .ct-fires { display: flex; flex-wrap: wrap; gap: 4px; }

  .circuit-lab .ct-detail { display: flex; flex-direction: column; gap: 1rem; }
  .circuit-lab .ct-detail__head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; font-size: 0.95rem; }
  .circuit-lab .ct-detail__stats { display: flex; flex-wrap: wrap; gap: 0.3rem 1.25rem; color: var(--oz-fg-dim); font-size: 0.85rem; }
  .circuit-lab .ct-detail__stats b { color: var(--oz-fg); font-weight: 500; font-variant-numeric: tabular-nums; }
  .circuit-lab .ct-detail .pl-control-label { margin-bottom: 0.4rem; }
  .circuit-lab .ct-detail__edges { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem 1.5rem; }
  .circuit-lab .ct-edges ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
  .circuit-lab .ct-edges li { display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  .circuit-lab .ct-edges .pl-link { font-family: var(--oz-mono); font-size: 0.8rem; white-space: pre; overflow: hidden; text-overflow: ellipsis; }

  .circuit-lab .ct-table-scroll { max-height: 460px; overflow-y: auto; }
  .circuit-lab .ct-table tbody tr.is-active { background: var(--oz-fill-strong); }
  .circuit-lab .ct-table td.ct-mono { text-align: left; }
  .circuit-lab .ct-table th:nth-child(2), .circuit-lab .ct-table th:nth-child(3) { text-align: left; }
  .circuit-lab .ct-promotes { max-width: 11rem; overflow: hidden; text-overflow: ellipsis; }
  .circuit-lab .ct-predictions tbody tr, .circuit-lab .ct-compare .ct-table tbody tr { cursor: default; }
  .circuit-lab .ct-predictions th:nth-child(3), .circuit-lab .ct-predictions td:nth-child(3) { text-align: left; padding-left: 1.25rem; }

  .circuit-lab .ct-intervene-form { display: flex; flex-wrap: wrap; gap: 0.9rem 2rem; align-items: flex-start; }
  .circuit-lab .ct-pinned { flex: 1 1 18rem; display: flex; flex-direction: column; align-items: flex-start; gap: 0.5rem; }
  .circuit-lab .ct-pinned .pl-control-label { margin-bottom: 0; }
  .circuit-lab .ct-pin { display: inline-flex; align-items: stretch; border: 1px solid var(--oz-rule-strong); }
  .circuit-lab .ct-pin button { background: none; border: none; color: var(--oz-fg); cursor: pointer; transition: color 0.2s ease; }
  .circuit-lab .ct-pin button:hover:not(:disabled) { color: var(--oz-mark); }
  .circuit-lab .ct-pin__name { padding: 0.2rem 0.2rem 0.2rem 0.45rem; font-family: var(--oz-mono); font-size: 0.8rem; }
  .circuit-lab .ct-pin__remove { display: inline-flex; align-items: center; padding: 0 0.3rem; }
  .circuit-lab .ct-pin__remove svg { width: 13px; height: 13px; stroke-width: 2.25; }
  .circuit-lab .ct-quick { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; }
  .circuit-lab .ct-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem 2rem; margin-top: 1.5rem; align-items: start; }

  @media (max-width: 900px) {
    .circuit-lab .ct-compare, .circuit-lab .ct-detail__edges { grid-template-columns: 1fr; }
  }
`
