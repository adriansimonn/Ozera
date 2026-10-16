import { AlertCircle } from 'lucide-react'
import { NavBar } from '../../components/common/NavBar'
import { ConceptPanel } from '../../components/geometry/ConceptPanel'
import { GeometryViewTabs, GeometryViews } from '../../components/geometry/GeometryViews'
import { GeometryRunSettings } from '../../components/geometry/RunSettings'
import { LabKitStyles } from '../../components/lab/chartKit'
import { LabPageStyles } from '../../components/lab/LabPageStyles'
import { ModelPicker } from '../../components/lab/ModelPicker'
import { useGeometryLab } from '../../hooks/useGeometryLab'
import { useTheme } from '../../hooks/useTheme'

interface GeometryPageProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultGeometryPage({ onShowPurchaseCredits }: GeometryPageProps) {
  const lab = useGeometryLab(onShowPurchaseCredits)
  const { isLight } = useTheme()

  const c = {
    bg: isLight ? '#f5f5f7' : '#0a0a0a',
    panelBg: isLight ? '#ffffff' : '#111111',
    divider: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)',
    text: isLight ? '#1d1d1f' : '#ffffff',
  }

  return (
    <div className="geometry-lab lab-kit lab-page--default" style={{ background: c.bg, color: c.text }}>
      <NavBar />

      <div className="pl-layout" style={{ ['--pl-divider' as string]: c.divider }}>
        <div className="pl-left" style={{ background: c.panelBg, borderRight: `1px solid ${c.divider}` }}>
          <div className="pl-scroll">
            <section className="pl-panel">
              <h2>View</h2>
              <GeometryViewTabs lab={lab} />
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
        </div>

        <div className="pl-right">
          <div className="pl-scroll">
            <GeometryViews lab={lab} />
          </div>
        </div>
      </div>

      <LabKitStyles />
      <LabPageStyles variant="default" />
    </div>
  )
}
