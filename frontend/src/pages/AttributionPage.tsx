import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { AttributionViewTabs, AttributionViews } from '../components/attribution/AttributionViews'
import { ContinuationView } from '../components/attribution/ContinuationView'
import { SweepSettings, TaskPanel } from '../components/attribution/TaskPanel'
import { LabKitStyles } from '../components/lab/chartKit'
import { LabPageStyles } from '../components/lab/LabPageStyles'
import { ModelPicker } from '../components/lab/ModelPicker'
import { useAttributionLab } from '../hooks/useAttributionLab'

interface AttributionPageProps {
  onShowPurchaseCredits?: () => void
}

/**
 * Attribution: which components a behaviour runs through, over prompt pairs. A sweep
 * (attribution patching), exact patching of its cells, its heads, ablations, and generating
 * with patches applied.
 */
export default function AttributionPage({ onShowPurchaseCredits }: AttributionPageProps) {
  const lab = useAttributionLab(onShowPurchaseCredits)
  const continuation = lab.view === 'continuation'
  const busy = lab.running || lab.verifying || lab.ablating
  return (
    <div className="attribution-lab lab-kit lab-page oz-page">
      <NavBar />

      <div className="oz-page__body" style={continuation ? { display: 'none' } : undefined}>
        <div className="oz-side">
          <div className="oz-scroll">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              <AttributionViewTabs lab={lab} />
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
              <ModelPicker lab={lab} disabled={busy} assist="attribution.model" />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Pairs</h2>
              <TaskPanel lab={lab} disabled={lab.running} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Sweep</h2>
              <SweepSettings lab={lab} />
            </section>
          </div>
        </div>

        <div className="oz-main">
          <div className="oz-scroll">
            <AttributionViews lab={lab} />
          </div>
        </div>
      </div>

      <ContinuationView
        lab={lab}
        tabs={<AttributionViewTabs lab={lab} />}
        hidden={!continuation}
        show={() => lab.setView('continuation')}
        onShowPurchaseCredits={onShowPurchaseCredits}
      />

      <LabKitStyles />
      <LabPageStyles />
      <style>{STYLES}</style>
    </div>
  )
}

// Scoped under the page's root class
const STYLES = `
  .attribution-lab .pl-view-tabs { grid-template-columns: repeat(3, minmax(0, 1fr)); }

  /* The built-in tasks: a ruled list, the chosen one open and marked */
  .attribution-lab .at-tasks { margin-top: 1rem; }
  .attribution-lab .at-task-item { position: relative; border-top: 1px solid var(--oz-rule); }
  .attribution-lab .at-task-item:last-child { border-bottom: 1px solid var(--oz-rule); }
  .attribution-lab .at-task-item.is-active::after { content: ''; position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; background: var(--oz-mark); }
  .attribution-lab .at-task-pick {
    display: block; width: 100%; padding: 0.7rem 0; background: none; border: none; color: var(--oz-fg-dim);
    font-size: 0.9375rem; font-weight: 500; letter-spacing: -0.01em; text-align: left; cursor: pointer; transition: color 0.2s ease;
  }
  .attribution-lab .at-task-pick:hover:not(:disabled),
  .attribution-lab .at-task-item.is-active .at-task-pick { color: var(--oz-fg); }
  .attribution-lab .at-task-pick:disabled { cursor: not-allowed; }
  .attribution-lab .at-task-detail { padding: 0 0 1rem; display: flex; flex-direction: column; align-items: flex-start; gap: 0.6rem; }

  /* A pair: its clean prompt, then its corrupted one, each with its answer */
  .attribution-lab .at-pairs { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.6rem; width: 100%; }
  .attribution-lab .at-pairs li { display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.85rem; line-height: 1.4; }
  .attribution-lab .at-pair__prompt b { font-weight: 600; text-decoration: underline; text-decoration-color: var(--oz-fg-faint); text-underline-offset: 3px; white-space: pre; }
  .attribution-lab .at-pair__prompt--corrupted { color: var(--oz-fg-dim); }

  .attribution-lab .at-setting-row { display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: flex-end; margin-top: 1.1rem; }
  .attribution-lab .at-parsed { border: 1px solid var(--oz-rule-strong); padding: 0.6rem 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }
  .attribution-lab .at-parsed-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
  .attribution-lab .at-parsed-name { font-size: 0.9rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .attribution-lab .at-estimate {
    display: flex; justify-content: space-between; gap: 0.75rem; margin: 1.25rem 0 0.9rem; padding-top: 0.8rem;
    border-top: 1px solid var(--oz-rule); color: var(--oz-fg-dim); font-size: 0.85rem;
  }
  .attribution-lab .at-estimate span:last-child { color: var(--oz-fg); text-align: right; font-variant-numeric: tabular-nums; }
  .attribution-lab .at-dropped { margin: -0.4rem 0 0.9rem; }

  /* Heatmaps scroll sideways when a prompt has more tokens than fit */
  .attribution-lab .at-heatmap__scroll { overflow-x: auto; overflow-y: hidden; }
  .attribution-lab .at-ramp { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; color: var(--oz-fg-dim); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
  .attribution-lab .at-ramp__bar { width: 120px; height: 8px; }
  .attribution-lab .at-ramp__unit { margin-left: 0.4rem; }
  .attribution-lab .at-open-head { margin-top: 0.75rem; }
  .attribution-lab .at-scope { min-width: 15rem; }
  .attribution-lab .at-sweep-grid { align-items: start; }

  /* A pair's two prompts, token by token */
  .attribution-lab .at-pairtext { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.85rem; line-height: 1.6; }
  .attribution-lab .at-pairtext__name { display: inline-block; width: 5.5rem; color: var(--oz-fg-dim); }
  .attribution-lab .at-pairtext__value { margin-left: 0.75rem; color: var(--oz-fg-dim); font-variant-numeric: tabular-nums; }
  .attribution-lab .at-token { white-space: pre; font-family: var(--oz-mono); font-size: 0.8rem; }
  .attribution-lab .at-token.is-changed { background: var(--oz-fill-strong); outline: 1px solid var(--oz-rule-strong); }
  .attribution-lab .at-token--answer { margin-left: 0.35rem; font-weight: 600; text-decoration: underline; text-underline-offset: 3px; }

  /* Two results side by side */
  .attribution-lab .at-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem 2rem; margin-top: 1.5rem; align-items: start; }
  .attribution-lab .at-table-scroll { max-height: 420px; overflow-y: auto; }
  .attribution-lab .at-table tbody tr { cursor: default; }
  .attribution-lab .at-table-scroll[data-assist] tbody tr { cursor: pointer; }
  .attribution-lab .at-table tbody tr.is-active { background: var(--oz-fill-strong); }
  .attribution-lab .at-verified { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0 0.75rem; }
  .attribution-lab .at-verified .pl-control-label { width: 100%; }
  .attribution-lab .at-attention { max-width: 560px; }

  .attribution-lab .at-ablate-form { display: flex; flex-wrap: wrap; gap: 0.9rem 1.5rem; align-items: flex-start; }
  .attribution-lab .at-ablate-components { flex: 1 1 16rem; display: flex; flex-direction: column; align-items: flex-start; gap: 0.45rem; }
  .attribution-lab .at-ablate-components .pl-control-label { margin-bottom: 0; }
  .attribution-lab .at-ablate-components .pl-input[aria-invalid='true'] { border-color: var(--oz-danger); }

  @media (max-width: 900px) {
    .attribution-lab .at-compare { grid-template-columns: 1fr; }
  }
`
