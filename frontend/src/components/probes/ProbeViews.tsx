/**
 * The Probe Lab's views, one per tool, and the tabs that switch between them. Every view stays
 * mounted while hidden, so results that cost GPU time survive switching views.
 */

import type { ReactNode } from 'react'
import { Activity, Eraser, Grid3x3, Layers, Navigation, ScanText, type LucideIcon } from 'lucide-react'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { ProbeCausal } from './CausalPanel'
import { ProbeGeneralization } from './GeneralizationPanel'
import { ProbeResults, ProbeTools } from './ProbeResults'

export type ProbeViewId = 'sweep' | 'score' | 'generalize' | 'steer' | 'ablate' | 'sae'

const PROBE_VIEWS: { id: ProbeViewId; label: string; icon: LucideIcon }[] = [
  { id: 'sweep', label: 'Sweep', icon: Activity },
  { id: 'score', label: 'Score', icon: ScanText },
  { id: 'generalize', label: 'Generalize', icon: Grid3x3 },
  { id: 'steer', label: 'Steer', icon: Navigation },
  { id: 'ablate', label: 'Ablate', icon: Eraser },
  { id: 'sae', label: 'SAE', icon: Layers },
]

export function ProbeViewTabs({ lab }: { lab: ProbeLab }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Probe Lab views">
      {PROBE_VIEWS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          role="tab"
          aria-selected={lab.view === id}
          className={`pl-view-tab${lab.view === id ? ' active' : ''}`}
          onClick={() => lab.setView(id)}
        >
          <Icon className="pl-icon" aria-hidden="true" />
          <span>{label}</span>
        </button>
      ))}
    </div>
  )
}

export function ProbeViews({ lab, onShowPurchaseCredits }: { lab: ProbeLab; onShowPurchaseCredits?: () => void }) {
  const views: Record<ProbeViewId, ReactNode> = {
    sweep: <ProbeResults lab={lab} />,
    score: <ProbeTools lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />,
    generalize: <ProbeGeneralization lab={lab} onShowPurchaseCredits={onShowPurchaseCredits} />,
    steer: <ProbeCausal lab={lab} tool="steer" onShowPurchaseCredits={onShowPurchaseCredits} />,
    ablate: <ProbeCausal lab={lab} tool="ablate" onShowPurchaseCredits={onShowPurchaseCredits} />,
    sae: <ProbeCausal lab={lab} tool="sae" onShowPurchaseCredits={onShowPurchaseCredits} />,
  }
  return (
    <>
      {PROBE_VIEWS.map(({ id }) => (
        <div key={id} role="tabpanel" hidden={lab.view !== id}>
          {views[id]}
        </div>
      ))}
    </>
  )
}
