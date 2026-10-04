import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { ProbeLabStyles } from '../components/probes/chartKit'
import { DatasetPanel } from '../components/probes/DatasetPanel'
import { ProbeViewTabs, ProbeViews } from '../components/probes/ProbeViews'
import { ModelPicker, RunSettings } from '../components/probes/RunSettings'
import { useProbeLab } from '../hooks/useProbeLab'

interface ProbesPageProps {
  onShowPurchaseCredits?: () => void
}

export default function ProbesPage({ onShowPurchaseCredits }: ProbesPageProps) {
  const lab = useProbeLab(onShowPurchaseCredits)

  return (
    <div className="probe-lab probe-lab--glass">
      <NavBar />

      <div className="pl-content">
        <div className="pl-header">
          <h1>Probe Lab</h1>
          <p>Train linear probes, check them against baselines, and test their directions</p>
        </div>

        <ProbeViewTabs lab={lab} />

        {lab.error && (
          <div className="pl-error" role="alert">
            <AlertCircle className="pl-icon" aria-hidden="true" />
            <span>{lab.error}</span>
            <button type="button" onClick={() => lab.setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        <div className="pl-layout">
          <div className="pl-setup">
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

          <div className="pl-main">
            <ProbeViews lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
        </div>
      </div>

      <ProbeLabStyles />
      <style>{`
        .probe-lab--glass {
          min-height: 125vh;
          padding-top: 70px;
          color: #ffffff;
        }
        [data-bg="light"] .probe-lab--glass { color: #1d1d1f; }

        .probe-lab--glass .pl-content { max-width: 1800px; margin: 0 auto; padding: 2rem; }
        .probe-lab--glass .pl-header { margin-bottom: 1.5rem; }
        .probe-lab--glass .pl-header h1 { margin: 0; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; }
        .probe-lab--glass .pl-header p { margin: 0.5rem 0 0; font-size: 0.9rem; }

        .probe-lab--glass .pl-view-tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
        .probe-lab--glass .pl-view-tab {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #ffffff;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        .probe-lab--glass .pl-view-tab .pl-icon { width: 16px; height: 16px; }
        .probe-lab--glass .pl-view-tab:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(255, 255, 255, 0.2); }
        .probe-lab--glass .pl-view-tab.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.4); }
        [data-bg="light"] .probe-lab--glass .pl-view-tab { background: rgba(0, 0, 0, 0.04); border-color: rgba(0, 0, 0, 0.1); color: #1d1d1f; }
        [data-bg="light"] .probe-lab--glass .pl-view-tab:hover { background: rgba(0, 0, 0, 0.06); border-color: rgba(0, 0, 0, 0.15); }
        [data-bg="light"] .probe-lab--glass .pl-view-tab.active { background: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.25); }

        .probe-lab--glass .pl-error {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          margin-bottom: 1.5rem;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.95);
          font-size: 0.85rem;
        }
        .probe-lab--glass .pl-error button {
          margin-left: auto;
          padding: 0.25rem 0.75rem;
          background: transparent;
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: inherit;
          cursor: pointer;
        }

        .probe-lab--glass .pl-layout { display: grid; grid-template-columns: 370px minmax(0, 1fr); gap: 1.5rem; align-items: start; }

        .probe-lab--glass .pl-panel {
          margin-bottom: 1.5rem;
          padding: 1.25rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        [data-bg="light"] .probe-lab--glass .pl-panel { background: rgba(0, 0, 0, 0.03); border-color: rgba(0, 0, 0, 0.1); }
        [data-bg="light"][data-interface="glass"] .probe-lab--glass .pl-panel { background: rgba(255, 255, 255, 0.55); }
        .probe-lab--glass .pl-panel h2 {
          margin: 0 0 1rem;
          font-size: 0.9rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .probe-lab--glass .pl-results-grid,
        .probe-lab--glass .pl-tools {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 1.5rem;
        }
        .probe-lab--glass .pl-results-grid .pl-panel,
        .probe-lab--glass .pl-tools .pl-panel { margin-bottom: 0; }
        .probe-lab--glass .pl-results-grid { margin-bottom: 1.5rem; }

        .probe-lab--glass .pl-train-button {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.6rem;
          padding: 0.9rem 1.25rem;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.25);
          color: #ffffff;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
        }
        .probe-lab--glass .pl-train-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.15); border-color: rgba(255, 255, 255, 0.4); }
        .probe-lab--glass .pl-train-button:disabled { opacity: 0.5; cursor: not-allowed; }
        .probe-lab--glass .pl-train-button .pl-spinner { border-color: rgba(255, 255, 255, 0.3); border-top-color: #ffffff; }
        [data-bg="light"] .probe-lab--glass .pl-train-button { background: rgba(0, 0, 0, 0.06); border-color: rgba(0, 0, 0, 0.15); color: #1d1d1f; }
        [data-bg="light"] .probe-lab--glass .pl-train-button:hover:not(:disabled) { background: rgba(0, 0, 0, 0.1); }
        [data-bg="light"] .probe-lab--glass .pl-train-button .pl-spinner { border-color: rgba(0, 0, 0, 0.15); border-top-color: #1d1d1f; }

        @media (max-width: 1200px) {
          .probe-lab--glass .pl-layout { grid-template-columns: 1fr; }
        }
        @media (max-width: 900px) {
          .probe-lab--glass .pl-content { padding: 1rem; }
          .probe-lab--glass .pl-view-tabs { flex-wrap: wrap; }
          .probe-lab--glass .pl-view-tab { flex: 1; min-width: 120px; justify-content: center; }
          .probe-lab--glass .pl-results-grid,
          .probe-lab--glass .pl-tools { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  )
}
