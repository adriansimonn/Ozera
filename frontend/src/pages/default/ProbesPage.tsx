import { AlertCircle } from 'lucide-react'
import { NavBar } from '../../components/common/NavBar'
import { ProbeLabStyles } from '../../components/probes/chartKit'
import { DatasetPanel } from '../../components/probes/DatasetPanel'
import { ProbeViewTabs, ProbeViews } from '../../components/probes/ProbeViews'
import { ModelPicker, RunSettings } from '../../components/probes/RunSettings'
import { useProbeLab } from '../../hooks/useProbeLab'
import { useTheme } from '../../hooks/useTheme'

interface ProbesPageProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultProbesPage({ onShowPurchaseCredits }: ProbesPageProps) {
  const lab = useProbeLab(onShowPurchaseCredits)
  const { isLight } = useTheme()

  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
  }

  return (
    <div className="probe-lab probe-lab--default" style={{ background: c.bg, color: c.text }}>
      <NavBar />

      <div className="pl-layout" style={{ ['--pl-divider' as string]: c.divider }}>
        <div className="pl-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="pl-scroll">
            <section className="pl-panel">
              <h2>View</h2>
              <ProbeViewTabs lab={lab} />
            </section>
            {lab.error && (
              <div className="pl-error" role="alert">
                <AlertCircle className="pl-icon" aria-hidden="true" />
                <span>{lab.error}</span>
                <button type="button" onClick={() => lab.setError(null)}>
                  Dismiss
                </button>
              </div>
            )}
            <section className="pl-panel">
              <h2>Model</h2>
              <ModelPicker lab={lab} disabled={lab.training} />
            </section>
            <section className="pl-panel">
              <h2>Dataset</h2>
              <DatasetPanel lab={lab} disabled={lab.training} />
            </section>
            <section className="pl-panel">
              <h2>Training</h2>
              <RunSettings lab={lab} />
            </section>
          </div>
        </div>

        <div className="pl-right">
          <div className="pl-scroll">
            <ProbeViews lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
        </div>
      </div>

      <ProbeLabStyles />
      <style>{`
        .probe-lab--default {
          height: 125vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .probe-lab--default .pl-layout { display: flex; flex: 1; margin-top: 76px; overflow: hidden; }
        .probe-lab--default .pl-left { width: 26%; min-width: 320px; max-width: 420px; display: flex; flex-direction: column; overflow: hidden; }
        .probe-lab--default .pl-right { flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
        .probe-lab--default .pl-scroll { flex: 1; overflow-y: auto; }

        .probe-lab--default .pl-panel { padding: 1rem 1.25rem; border-bottom: 1px solid var(--pl-divider); }
        .probe-lab--default .pl-panel h2,
        .probe-lab--default .pl-card-title h3 {
          margin: 0 0 0.625rem;
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .probe-lab--default .pl-card-title h3 { margin: 0; }

        .probe-lab--default .pl-view-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.375rem; }
        .probe-lab--default .pl-view-tab {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.375rem;
          padding: 0.5rem 0.375rem;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.04);
          color: inherit;
          font-size: 0.7rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          min-width: 0;
        }
        .probe-lab--default .pl-view-tab .pl-icon { width: 12px; height: 12px; }
        .probe-lab--default .pl-view-tab:hover { opacity: 0.85; }
        .probe-lab--default .pl-view-tab.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.2); }
        [data-bg="light"] .probe-lab--default .pl-view-tab { background: rgba(0, 0, 0, 0.04); border-color: rgba(0, 0, 0, 0.1); }
        [data-bg="light"] .probe-lab--default .pl-view-tab.active { background: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.2); }

        .probe-lab--default .pl-results-grid,
        .probe-lab--default .pl-tools {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .probe-lab--default .pl-results-grid > .pl-panel:nth-child(odd),
        .probe-lab--default .pl-tools > .pl-panel:nth-child(odd) { border-right: 1px solid var(--pl-divider); }

        .probe-lab--default .pl-error {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin: 0.75rem 1.25rem 0;
          padding: 0.625rem 0.875rem;
          background: rgba(239, 68, 68, 0.08);
          border: 1px solid rgba(239, 68, 68, 0.2);
          color: #ef4444;
          font-size: 0.8rem;
        }
        .probe-lab--default .pl-error button { margin-left: auto; background: none; border: none; color: inherit; cursor: pointer; font-size: 0.75rem; }

        .probe-lab--default .pl-train-button {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          border: 1px solid transparent;
          background: #ffffff;
          color: #0a0a0a;
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
        }
        .probe-lab--default .pl-train-button:hover:not(:disabled) { opacity: 0.85; }
        .probe-lab--default .pl-train-button:disabled { opacity: 0.4; cursor: not-allowed; }
        .probe-lab--default .pl-train-button .pl-spinner { border-color: rgba(0, 0, 0, 0.15); border-top-color: #0a0a0a; }
        [data-bg="light"] .probe-lab--default .pl-train-button { background: #1d1d1f; color: #ffffff; }
        [data-bg="light"] .probe-lab--default .pl-train-button .pl-spinner { border-color: rgba(255, 255, 255, 0.3); border-top-color: #ffffff; }

        @media (max-width: 1100px) {
          .probe-lab--default { height: auto; overflow: visible; }
          .probe-lab--default .pl-layout { flex-direction: column; overflow: visible; }
          .probe-lab--default .pl-left { width: 100%; max-width: none; border-right: none !important; }
          .probe-lab--default .pl-scroll { overflow: visible; }
        }
        @media (max-width: 800px) {
          .probe-lab--default .pl-results-grid,
          .probe-lab--default .pl-tools { grid-template-columns: 1fr; }
          .probe-lab--default .pl-results-grid > .pl-panel:nth-child(odd),
          .probe-lab--default .pl-tools > .pl-panel:nth-child(odd) { border-right: none; }
        }
      `}</style>
    </div>
  )
}
