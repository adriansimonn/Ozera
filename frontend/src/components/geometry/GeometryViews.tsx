/**
 * The Geometry Lab's views, one per tool, and the tabs that switch between them. Every view
 * stays mounted while hidden, so results survive switching views.
 */

import type { ReactNode } from 'react'
import { GitCompareArrows, Grid3x3, LineChart, Map, Orbit, Puzzle, Spline, type LucideIcon } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import { AtlasView } from './AtlasView'
import { CompareView } from './CompareView'
import { LayerChart } from './LayerChart'
import { SaeView } from './SaeView'
import { ShapeView } from './ShapeView'
import { SimilarityMatrix } from './SimilarityMatrix'
import { SteerView } from './SteerView'

export type GeometryViewId = 'shape' | 'layers' | 'similarity' | 'compare' | 'steer' | 'sae' | 'atlas'

const GEOMETRY_VIEWS: { id: GeometryViewId; label: string; icon: LucideIcon }[] = [
  { id: 'shape', label: 'Shape', icon: Orbit },
  { id: 'layers', label: 'Layers', icon: LineChart },
  { id: 'similarity', label: 'Similarity', icon: Grid3x3 },
  { id: 'compare', label: 'Compare', icon: GitCompareArrows },
  { id: 'steer', label: 'Steer', icon: Spline },
  { id: 'sae', label: 'SAE', icon: Puzzle },
  { id: 'atlas', label: 'Atlas', icon: Map },
]

export function GeometryViewTabs({ lab }: { lab: GeometryLab }) {
  return (
    <div className="pl-view-tabs" role="tablist" aria-label="Geometry Lab views">
      {GEOMETRY_VIEWS.map(({ id, label, icon: Icon }) => (
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
        .geometry-lab .gl-ramp-legend { display: flex; align-items: center; gap: 0.5rem; font-size: 0.72rem; margin-top: 0.25rem; }
        .geometry-lab .gl-ramp { display: inline-block; width: 140px; height: 8px; border: 1px solid; border-radius: 4px; }
      `}</style>
    </>
  )
}
