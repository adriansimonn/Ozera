/**
 * The Attribution page's views and the tabs that switch between them. The sweep's views stay
 * mounted while hidden, so results that cost GPU time survive switching views. (The
 * continuation view takes the whole page: see ContinuationView.)
 */

import type { ReactNode } from 'react'
import type { AttributionLab, AttributionViewId } from '../../hooks/useAttributionLab'
import { AblateView } from './AblateView'
import { HeadsView } from './HeadsView'
import { SweepView } from './SweepView'
import { VerifyView } from './VerifyView'

const VIEWS: { id: AttributionViewId; label: string; title: string }[] = [
  { id: 'sweep', label: 'Sweep', title: 'Attribution patching: an estimate of every component’s effect' },
  { id: 'verify', label: 'Verify', title: 'Exact activation patching of the sweep’s cells' },
  { id: 'heads', label: 'Heads', title: 'Each head’s effects and attention pattern' },
  { id: 'ablate', label: 'Ablate', title: 'Remove components, against random ones' },
  { id: 'continuation', label: 'Continuation', title: 'Generate with patches applied and compare the texts' },
]

export function AttributionViewTabs({ lab }: { lab: Pick<AttributionLab, 'view' | 'setView'> }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Attribution views" data-assist="attribution.view">
      {VIEWS.map(({ id, label, title }) => (
        <button
          key={id}
          type="button"
          role="tab"
          aria-selected={lab.view === id}
          title={title}
          className={`oz-chip${lab.view === id ? ' active' : ''}`}
          onClick={() => lab.setView(id)}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function AttributionViews({ lab }: { lab: AttributionLab }) {
  const views: Partial<Record<AttributionViewId, ReactNode>> = {
    sweep: <SweepView lab={lab} />,
    verify: <VerifyView lab={lab} />,
    heads: <HeadsView lab={lab} />,
    ablate: <AblateView lab={lab} />,
  }
  return (
    <>
      {(Object.keys(views) as AttributionViewId[]).map((id) => (
        <div key={id} role="tabpanel" hidden={lab.view !== id}>
          {views[id]}
        </div>
      ))}
    </>
  )
}
