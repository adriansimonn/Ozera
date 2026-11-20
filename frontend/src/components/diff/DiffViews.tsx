/**
 * The Model Diff page's views and the tabs that switch between them. All three stay mounted
 * while hidden, so results that cost GPU time survive switching views.
 */

import type { DiffLab, DiffViewId } from '../../hooks/useDiffLab'
import { DirectionsView } from './DirectionsView'
import { LayersView } from './LayersView'
import { PatchView } from './PatchView'

const VIEWS: { id: DiffViewId; label: string; title: string }[] = [
  { id: 'layers', label: 'Layers', title: 'Where the two models part, layer by layer and token by token' },
  { id: 'directions', label: 'Directions', title: 'The directions their difference lies along at one layer, tested by adding them to the base model' },
  { id: 'patch', label: 'Patch', title: 'One model’s activations patched into the other, on one prompt' },
]

export function DiffViewTabs({ lab }: { lab: Pick<DiffLab, 'view' | 'setView'> }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Model Diff views" data-assist="diff.view">
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

export function DiffViews({ lab }: { lab: DiffLab }) {
  return (
    <>
      <div role="tabpanel" hidden={lab.view !== 'layers'}>
        <LayersView lab={lab} />
      </div>
      <div role="tabpanel" hidden={lab.view !== 'directions'}>
        <DirectionsView lab={lab} />
      </div>
      <div role="tabpanel" hidden={lab.view !== 'patch'}>
        <PatchView lab={lab} />
      </div>
    </>
  )
}
