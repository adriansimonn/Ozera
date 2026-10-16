import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { LabKitStyles } from '../components/lab/chartKit'
import { LabPageStyles } from '../components/lab/LabPageStyles'
import { DatasetPanel } from '../components/probes/DatasetPanel'
import { ProbeViewTabs, ProbeViews } from '../components/probes/ProbeViews'
import { ModelPicker } from '../components/lab/ModelPicker'
import { RunSettings } from '../components/probes/RunSettings'
import { useProbeLab } from '../hooks/useProbeLab'

interface ProbesPageProps {
  onShowPurchaseCredits?: () => void
}

export default function ProbesPage({ onShowPurchaseCredits }: ProbesPageProps) {
  const lab = useProbeLab(onShowPurchaseCredits)

  return (
    <div className="probe-lab lab-kit lab-page--glass">
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

      <LabKitStyles />
      <LabPageStyles variant="glass" />
    </div>
  )
}
