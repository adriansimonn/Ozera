import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { DiffViewTabs, DiffViews } from '../components/diff/DiffViews'
import { AlsoLinks, CompareSettings, PairPicker, TextsPanel } from '../components/diff/SetupPanel'
import { LabKitStyles } from '../components/lab/chartKit'
import { LabPageStyles } from '../components/lab/LabPageStyles'
import { useDiffLab } from '../hooks/useDiffLab'

interface DiffPageProps {
  onShowPurchaseCredits?: () => void
}

/**
 * Model Diff: where two models of the same shape differ on the same tokens. A base model and
 * its instruct sibling, or an Ozera model and the user's custom model: their layers compared,
 * the directions their difference lies along, and one patched into the other.
 */
export default function DiffPage({ onShowPurchaseCredits }: DiffPageProps) {
  const lab = useDiffLab(onShowPurchaseCredits)
  const busy = lab.running || lab.finding || lab.patching
  return (
    <div className="diff-lab lab-kit lab-page oz-page">
      <NavBar />

      <div className="oz-page__body">
        <div className="oz-side">
          <div className="oz-scroll">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              <DiffViewTabs lab={lab} />
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
              <h2 className="oz-title">Pair</h2>
              <PairPicker lab={lab} disabled={busy} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Texts</h2>
              <TextsPanel lab={lab} disabled={busy} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Compare</h2>
              <CompareSettings lab={lab} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Also</h2>
              <AlsoLinks />
            </section>
          </div>
        </div>

        <div className="oz-main">
          <div className="oz-scroll">
            <DiffViews lab={lab} />
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
  .diff-lab .pl-view-tabs { grid-template-columns: repeat(3, minmax(0, 1fr)); }

  .diff-lab .df-pair { display: flex; flex-direction: column; gap: 0.15rem; margin-top: 0.6rem; }
  .diff-lab .df-pair span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* The built-in samples: a ruled list, the chosen one open and marked */
  .diff-lab .df-samples { margin-top: 1rem; }
  .diff-lab .df-sample { position: relative; border-top: 1px solid var(--oz-rule); }
  .diff-lab .df-sample:last-child { border-bottom: 1px solid var(--oz-rule); }
  .diff-lab .df-sample.is-active::after { content: ''; position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; background: var(--oz-mark); }
  .diff-lab .df-sample__pick {
    display: block; width: 100%; padding: 0.7rem 0; background: none; border: none; color: var(--oz-fg-dim);
    font-size: 0.9375rem; font-weight: 500; letter-spacing: -0.01em; text-align: left; cursor: pointer; transition: color 0.2s ease;
  }
  .diff-lab .df-sample__pick:hover:not(:disabled),
  .diff-lab .df-sample.is-active .df-sample__pick { color: var(--oz-fg); }
  .diff-lab .df-sample__pick:disabled { cursor: not-allowed; }
  .diff-lab .df-sample__detail { padding: 0 0 1rem; display: flex; flex-direction: column; align-items: flex-start; gap: 0.6rem; }
  .diff-lab .df-examples { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.6rem; font-size: 0.85rem; line-height: 1.4; }
  .diff-lab .df-setting-row { display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: flex-end; margin-top: 1.1rem; }
  .diff-lab .df-own { margin-top: 0.9rem; display: flex; flex-direction: column; gap: 0.4rem; }
  .diff-lab .df-own .pl-textarea { min-height: 150px; }
  .diff-lab .df-template { margin-top: 1.1rem; }
  .diff-lab .df-estimate { display: flex; justify-content: space-between; gap: 0.75rem; margin: 0 0 0.9rem; color: var(--oz-fg-dim); font-size: 0.85rem; }
  .diff-lab .df-estimate span:last-child { color: var(--oz-fg); text-align: right; font-variant-numeric: tabular-nums; }
  .diff-lab .df-also { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.7rem; font-size: 0.9rem; }
  .diff-lab .df-also li { display: flex; flex-direction: column; gap: 0.1rem; }

  .diff-lab .df-title-tools { display: flex; align-items: center; gap: 0.5rem; }
  .diff-lab .df-table-scroll { max-height: 360px; overflow-y: auto; }
  .diff-lab .df-table-scroll tbody tr.is-active { background: var(--oz-fill-strong); }
  .diff-lab .df-open { margin-top: 0.9rem; }
  .diff-lab .df-token-grid { align-items: start; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); }

  /* The texts, token by token */
  .diff-lab .df-readout { color: var(--oz-fg-dim); font-size: 0.85rem; margin-bottom: 0.6rem; min-height: 2.5rem; font-variant-numeric: tabular-nums; }
  .diff-lab .df-tokens { font-family: var(--oz-mono); font-size: 0.8rem; line-height: 1.9; margin-bottom: 0.6rem; }
  .diff-lab .df-text { margin: 0 0 0.7rem; white-space: pre-wrap; word-break: break-word; }
  .diff-lab .df-token.is-left-out { outline: 1px dashed var(--oz-rule-strong); outline-offset: -1px; }
  .diff-lab .df-scale { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.9rem; color: var(--oz-fg-dim); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
  .diff-lab .df-scale__bar { width: 140px; height: 8px; }

  .diff-lab .df-divergent tbody tr { cursor: default; }
  .diff-lab .df-divergent th:nth-child(n + 3), .diff-lab .df-divergent td:nth-child(n + 3) { text-align: left; }
  .diff-lab .df-mono { font-family: var(--oz-mono); font-size: 0.8rem; white-space: pre; }
  .diff-lab .df-context { font-family: var(--oz-mono); font-size: 0.8rem; white-space: pre; overflow: hidden; text-overflow: ellipsis; max-width: 16rem; }
  .diff-lab .df-context__before { color: var(--oz-fg-dim); }
  .diff-lab .df-context b { font-weight: 600; background: var(--oz-fill-strong); outline: 1px solid var(--oz-rule-strong); }

  /* Two results side by side */
  .diff-lab .df-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem 2rem; margin-top: 1.5rem; align-items: start; }
  .diff-lab .df-component { margin-top: 1.75rem; padding-top: 1.25rem; border-top: 1px solid var(--oz-rule); }
  .diff-lab .df-component .df-compare { margin-top: 0.5rem; }
  .diff-lab .df-contexts { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
  .diff-lab .df-contexts li { display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; }
  .diff-lab .df-contexts .df-context { max-width: none; flex: 1 1 auto; min-width: 0; direction: rtl; text-align: left; }
  .diff-lab .df-contexts .df-context > * { direction: ltr; unicode-bidi: embed; }
  .diff-lab .df-context__value { flex: none; color: var(--oz-fg-dim); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
  .diff-lab .df-layer { min-width: 12rem; }

  .diff-lab .df-patch-form { display: flex; flex-wrap: wrap; gap: 0.9rem 1.5rem; align-items: flex-start; }
  .diff-lab .df-prompt { flex: 1 1 100%; }
  .diff-lab .df-patch-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 1.5rem 2rem; margin-top: 1.5rem; align-items: start; }

  /* The patch grid is the Attribution page's heatmap: it scrolls sideways when a prompt has more tokens than fit */
  .diff-lab .at-heatmap__scroll { overflow-x: auto; overflow-y: hidden; }
  .diff-lab .at-ramp { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; color: var(--oz-fg-dim); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
  .diff-lab .at-ramp__bar { width: 120px; height: 8px; }
  .diff-lab .at-ramp__unit { margin-left: 0.4rem; }

  @media (max-width: 1000px) {
    .diff-lab .df-compare, .diff-lab .df-patch-grid, .diff-lab .df-token-grid { grid-template-columns: 1fr; }
  }
`
