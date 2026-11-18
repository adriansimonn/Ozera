/**
 * The Geometry Lab's views, one per tool, and the tabs that switch between them. Every view
 * stays mounted while hidden, so results survive switching views.
 */

import type { ReactNode } from 'react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import { AtlasView } from './AtlasView'
import { CompareView } from './CompareView'
import { LayerChart } from './LayerChart'
import { SaeView } from './SaeView'
import { ShapeView } from './ShapeView'
import { SimilarityMatrix } from './SimilarityMatrix'
import { SteerView } from './SteerView'

export type GeometryViewId = 'shape' | 'layers' | 'similarity' | 'compare' | 'steer' | 'sae' | 'atlas'

const GEOMETRY_VIEWS: { id: GeometryViewId; label: string }[] = [
  { id: 'shape', label: 'Shape' },
  { id: 'layers', label: 'Layers' },
  { id: 'similarity', label: 'Similarity' },
  { id: 'compare', label: 'Compare' },
  { id: 'steer', label: 'Steer' },
  { id: 'sae', label: 'SAE' },
  { id: 'atlas', label: 'Atlas' },
]

export function GeometryViewTabs({ lab }: { lab: GeometryLab }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Geometry Lab views" data-assist="geometry.view">
      {GEOMETRY_VIEWS.map(({ id, label }) => (
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

export function GeometryViews({ lab }: { lab: GeometryLab }) {
  const views: Record<GeometryViewId, ReactNode> = {
    shape: <ShapeView lab={lab} />,
    layers: <LayerChart lab={lab} />,
    similarity: <SimilarityMatrix lab={lab} />,
    compare: <CompareView lab={lab} />,
    steer: <SteerView lab={lab} />,
    sae: <SaeView lab={lab} />,
    atlas: <AtlasView lab={lab} />,
  }
  return (
    <>
      {GEOMETRY_VIEWS.map(({ id }) => (
        <div key={id} role="tabpanel" hidden={lab.view !== id}>
          {views[id]}
        </div>
      ))}
      <style>{`
        .geometry-lab .gl-slider { min-width: 220px; flex: 1; max-width: 420px; }
        .geometry-lab .gl-slider input { width: 100%; margin-top: 0.35rem; }
        .geometry-lab .gl-ramp-legend { display: flex; align-items: center; gap: 0.5rem; color: var(--oz-fg-dim); font-size: 0.8rem; margin-top: 0.25rem; }
        .geometry-lab .gl-ramp { display: inline-block; width: 140px; height: 8px; border: 1px solid var(--oz-rule-strong); }
      `}</style>
    </>
  )
}
