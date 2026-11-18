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

export default function DefaultProbesPage({ onShowPurchaseCredits }: ProbesPageProps) {
  const lab = useProbeLab(onShowPurchaseCredits)
  return (
    <div className="probe-lab lab-kit lab-page oz-page">
      <NavBar />

      <div className="oz-page__body">
        <div className="oz-side">
          <div className="oz-scroll oz-steps">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              <ProbeViewTabs lab={lab} />
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
              <ModelPicker lab={lab} disabled={lab.training} assist="probes.model" />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Dataset</h2>
              <DatasetPanel lab={lab} disabled={lab.training} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Training</h2>
              <RunSettings lab={lab} />
            </section>
          </div>
        </div>

        <div className="oz-main">
          <div className="oz-scroll">
            <ProbeViews lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
        </div>
      </div>

      <LabKitStyles />
      <LabPageStyles />
    </div>
  )
}
