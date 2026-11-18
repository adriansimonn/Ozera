import { AlertCircle } from 'lucide-react'
import { NavBar } from '../components/common/NavBar'
import { ConceptPanel } from '../components/geometry/ConceptPanel'
import { GeometryViewTabs, GeometryViews } from '../components/geometry/GeometryViews'
import { GeometryRunSettings } from '../components/geometry/RunSettings'
import { LabKitStyles } from '../components/lab/chartKit'
import { LabPageStyles } from '../components/lab/LabPageStyles'
import { ModelPicker } from '../components/lab/ModelPicker'
import { useGeometryLab } from '../hooks/useGeometryLab'

interface GeometryPageProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultGeometryPage({ onShowPurchaseCredits }: GeometryPageProps) {
  const lab = useGeometryLab(onShowPurchaseCredits)
  return (
    <div className="geometry-lab lab-kit lab-page oz-page">
      <NavBar />

      <div className="oz-page__body">
        <div className="oz-side">
          <div className="oz-scroll oz-steps">
            <section className="oz-section">
              <h2 className="oz-title">View</h2>
              <GeometryViewTabs lab={lab} />
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
              <ModelPicker lab={lab} disabled={lab.running} assist="geometry.model" />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Concept</h2>
              <ConceptPanel lab={lab} disabled={lab.running} />
            </section>
            <section className="oz-section">
              <h2 className="oz-title">Run</h2>
              <GeometryRunSettings lab={lab} />
            </section>
          </div>
        </div>

        <div className="oz-main">
          <div className="oz-scroll">
            <GeometryViews lab={lab} />
          </div>
        </div>
      </div>

      <LabKitStyles />
      <LabPageStyles />
    </div>
  )
}
