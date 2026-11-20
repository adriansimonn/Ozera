/**
 * The Circuit Tracing page's views and the tabs that switch between them. Both stay mounted
 * while hidden, so an intervention's result survives a look back at the graph.
 */

import type { CircuitLab, CircuitViewId } from '../../hooks/useCircuitLab'
import { GraphView } from './GraphView'
import { InterveneView } from './InterveneView'

const VIEWS: { id: CircuitViewId; label: string; title: string }[] = [
  { id: 'graph', label: 'Graph', title: 'The attribution graph of the prompt’s next word' },
  { id: 'intervene', label: 'Intervene', title: 'Change the graph’s nodes in the model, against controls' },
]

export function CircuitViewTabs({ lab }: { lab: Pick<CircuitLab, 'view' | 'setView'> }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Circuit Tracing views" data-assist="circuits.view">
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

export function CircuitViews({ lab }: { lab: CircuitLab }) {
  return (
    <>
      <div role="tabpanel" hidden={lab.view !== 'graph'}>
        <GraphView lab={lab} />
      </div>
      <div role="tabpanel" hidden={lab.view !== 'intervene'}>
        <InterveneView lab={lab} />
      </div>
    </>
  )
}
