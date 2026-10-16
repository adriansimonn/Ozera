import { AlertCircle } from 'lucide-react'
import { NavBar } from '../../components/common/NavBar'
import { LabKitStyles } from '../../components/lab/chartKit'
import { LabPageStyles } from '../../components/lab/LabPageStyles'
import { DatasetPanel } from '../../components/probes/DatasetPanel'
import { ProbeViewTabs, ProbeViews } from '../../components/probes/ProbeViews'
import { ModelPicker } from '../../components/lab/ModelPicker'
import { RunSettings } from '../../components/probes/RunSettings'
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
    <div className="probe-lab lab-kit lab-page--default" style={{ background: c.bg, color: c.text }}>
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

      <LabKitStyles />
      <LabPageStyles variant="default" />
    </div>
  )
}
