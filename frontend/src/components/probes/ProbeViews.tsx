/**
 * The Probe Lab's views, one per tool, and the tabs that switch between them. Every view stays
 * mounted while hidden, so results that cost GPU time survive switching views.
 */

import type { ReactNode } from 'react'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { ProbeCausal } from './CausalPanel'
import { ProbeGeneralization } from './GeneralizationPanel'
import { ProbeResults, ProbeTools } from './ProbeResults'

export type ProbeViewId = 'sweep' | 'score' | 'generalize' | 'steer' | 'ablate' | 'sae'

const PROBE_VIEWS: { id: ProbeViewId; label: string }[] = [
  { id: 'sweep', label: 'Sweep' },
  { id: 'score', label: 'Score' },
  { id: 'generalize', label: 'Generalize' },
  { id: 'steer', label: 'Steer' },
  { id: 'ablate', label: 'Ablate' },
  { id: 'sae', label: 'SAE' },
]

export function ProbeViewTabs({ lab }: { lab: ProbeLab }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Probe Lab views" data-assist="probes.view">
      {PROBE_VIEWS.map(({ id, label }) => (
        <button
          key={id}
          type="button"
          role="tab"
          aria-selected={lab.view === id}
          className={`oz-chip${lab.view === id ? ' active' : ''}`}
          onClick={() => lab.setView(id)}
        >
          {label}
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
