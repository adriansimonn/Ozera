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

export default function GeometryPage({ onShowPurchaseCredits }: GeometryPageProps) {
  const lab = useGeometryLab(onShowPurchaseCredits)

  return (
    <div className="geometry-lab lab-kit lab-page--glass">
      <NavBar />

      <div className="pl-content">
        <div className="pl-header">
          <h1>Geometry Lab</h1>
          <p>Find the shape of a concept and test it against baselines</p>
        </div>

        <GeometryViewTabs lab={lab} />

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
              <ModelPicker lab={lab} disabled={lab.running} />
            </section>
            <section className="pl-panel">
              <h2>Concept</h2>
              <ConceptPanel lab={lab} disabled={lab.running} />
            </section>
            <section className="pl-panel">
              <h2>Run</h2>
              <GeometryRunSettings lab={lab} />
            </section>
          </div>

          <div className="pl-main">
            <GeometryViews lab={lab} />
          </div>
        </div>
      </div>

      <LabKitStyles />
      <LabPageStyles variant="glass" />
    </div>
  )
}
