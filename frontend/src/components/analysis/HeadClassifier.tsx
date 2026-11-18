/**
 * HeadClassifier component for displaying attention head classification results.
 */

import { useState } from 'react'
import type { HeadClassification, HeadType } from '../../types/analysis'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../../types/analysis'

interface HeadClassifierProps {
  classifications: HeadClassification[]
  numLayers: number
  numHeads: number
  summary: Record<HeadType, number>
  onHeadSelect?: (layer: number, head: number) => void
  selectedHead?: { layer: number; head: number } | null
}

export function HeadClassifier({
  classifications,
  numLayers,
  numHeads,
  summary,
  onHeadSelect,
  selectedHead,
}: HeadClassifierProps) {
  const [hoveredHead, setHoveredHead] = useState<HeadClassification | null>(null)
  const [filterType, setFilterType] = useState<HeadType | 'all'>('all')

  // Filter classifications by type
  const filteredClassifications = filterType === 'all'
    ? classifications
    : classifications.filter(c => c.primary_type === filterType)

  // Group by layer
  const groupedByLayer: Record<number, HeadClassification[]> = {}
  for (const c of filteredClassifications) {
    if (!groupedByLayer[c.layer]) {
      groupedByLayer[c.layer] = []
    }
    groupedByLayer[c.layer].push(c)
  }

  return (
    <div className="head-classifier">
      {/* Summary Stats */}
      <div className="summary-section">
        <h3>Classification summary</h3>
        <div className="summary-buttons oz-chips">
          {Object.entries(summary).map(([type, count]) => (
            <button
              key={type}
              onClick={() => setFilterType(filterType === type ? 'all' : type as HeadType)}
              className={`oz-chip summary-btn ${filterType === type ? 'active' : ''}`}
              aria-pressed={filterType === type}
            >
              <i style={{ backgroundColor: HEAD_TYPE_COLORS[type as HeadType] }} />
              {HEAD_TYPE_NAMES[type as HeadType]}
              <span className="summary-count">{count}</span>
            </button>
          ))}
          {filterType !== 'all' && (
            <button
              onClick={() => setFilterType('all')}
              className="oz-link show-all-btn"
            >
              Show all
            </button>
          )}
        </div>
      </div>

      {/* Classifications by Layer */}
      <div className="layers-container">
        {Array.from({ length: numLayers }).map((_, layerIdx) => {
          const layerHeads = groupedByLayer[layerIdx] || []
          if (filterType !== 'all' && layerHeads.length === 0) return null

          return (
            <div key={layerIdx} className="layer-section">
              <h4>Layer {layerIdx}</h4>
              <div className="layer-heads">
                {Array.from({ length: numHeads }).map((_, headIdx) => {
                  const classification = classifications.find(
                    c => c.layer === layerIdx && c.head === headIdx
                  )
                  if (!classification) return null

                  const isFiltered = filterType !== 'all' && classification.primary_type !== filterType
                  const isSelected = selectedHead?.layer === layerIdx && selectedHead?.head === headIdx
                  const isHovered = hoveredHead?.layer === layerIdx && hoveredHead?.head === headIdx

                  return (
                    <button
                      key={headIdx}
                      onClick={() => onHeadSelect?.(layerIdx, headIdx)}
                      onMouseEnter={() => setHoveredHead(classification)}
                      onMouseLeave={() => setHoveredHead(null)}
                      className={`head-btn ${isFiltered ? 'filtered' : ''} ${isSelected ? 'selected' : ''} ${isHovered && !isSelected ? 'hovered' : ''}`}
                      style={{ borderTopColor: HEAD_TYPE_COLORS[classification.primary_type] }}
                    >
                      <div className="head-label">H{headIdx}</div>
                      <div className="head-confidence">
                        {(classification.confidence * 100).toFixed(0)}%
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {/* Hover Detail */}
      {hoveredHead && (
        <div className="hover-detail">
          <div className="hover-header">
            <span
              className="hover-dot"
              style={{ backgroundColor: HEAD_TYPE_COLORS[hoveredHead.primary_type] }}
            />
            <span className="hover-title">
              Layer {hoveredHead.layer}, Head {hoveredHead.head}
            </span>
          </div>
          <div className="hover-type">
            {HEAD_TYPE_NAMES[hoveredHead.primary_type]} · {(hoveredHead.confidence * 100).toFixed(1)}% confidence
          </div>
          <div className="hover-summary">
            {hoveredHead.pattern_summary}
          </div>
          <div className="hover-scores">
            {Object.entries(hoveredHead.scores).map(([type, score]) => (
              <div key={type} className="score-row">
                <span className="score-label">{HEAD_TYPE_NAMES[type as HeadType] ?? type}</span>
                <div className="score-bar-container">
                  <div
                    className="score-bar"
                    style={{
                      width: `${score * 100}%`,
                      backgroundColor: HEAD_TYPE_COLORS[type as HeadType] || 'var(--oz-fg-dim)',
                    }}
                  />
                </div>
                <span className="score-value">{(score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .head-classifier {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .head-classifier .summary-section h3 {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        .head-classifier .summary-buttons {
          align-items: center;
        }

        /* A head type, keyed by a cell of its color; pressing one filters the heads to it */
        .head-classifier .summary-btn i {
          flex: none;
          width: 8px;
          height: 8px;
        }

        .head-classifier .summary-count {
          opacity: 0.7;
          font-variant-numeric: tabular-nums;
        }

        .head-classifier .show-all-btn {
          margin-left: 0.5rem;
          font-size: 0.85rem;
        }

        /* A row of head cells to a layer */
        .head-classifier .layer-section {
          display: grid;
          grid-template-columns: 5.5rem minmax(0, 1fr);
          align-items: center;
          padding: 0.6rem 0;
          border-top: 1px solid var(--oz-rule);
        }

        .head-classifier .layer-section:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .head-classifier .layer-section h4 {
          margin: 0;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
          letter-spacing: 0;
        }

        .head-classifier .layer-heads {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }

        .head-classifier .head-btn {
          position: relative;
          min-width: 3.4rem;
          padding: 0.35rem 0.5rem 0.4rem;
          border: 1px solid var(--oz-rule);
          border-top: 2px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.78rem;
          text-align: center;
          cursor: pointer;
          transition: background 0.15s ease, opacity 0.15s ease;
        }

        .head-classifier .head-btn:hover,
        .head-classifier .head-btn.hovered {
          background: var(--oz-fill-strong);
        }

        .head-classifier .head-btn.filtered {
          opacity: 0.25;
        }

        .head-classifier .head-btn.selected {
          outline: 1.5px solid var(--oz-fg);
          outline-offset: 1px;
        }

        .head-classifier .head-confidence {
          color: var(--oz-fg-dim);
          font-size: 0.72rem;
        }

        /* The head under the pointer, in the corner of the window */
        .head-classifier .hover-detail {
          position: fixed;
          bottom: 1.25rem;
          right: 1.25rem;
          z-index: 50;
          width: 21rem;
          padding: 1rem 1.1rem 1.1rem;
          background: var(--oz-bg);
          border: 1px solid var(--oz-rule-strong);
          pointer-events: none;
        }

        .head-classifier .hover-header {
          display: flex;
          align-items: center;
          gap: 0.55rem;
        }

        .head-classifier .hover-dot {
          width: 9px;
          height: 9px;
        }

        .head-classifier .hover-title {
          font-size: 1.0625rem;
          font-weight: 500;
          letter-spacing: -0.02em;
        }

        .head-classifier .hover-type {
          margin-top: 0.2rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
        }

        .head-classifier .hover-summary {
          margin-top: 0.6rem;
          font-size: 0.85rem;
          line-height: 1.45;
        }

        .head-classifier .hover-scores {
          margin-top: 0.8rem;
          display: flex;
          flex-direction: column;
          gap: 0.3rem;
        }

        .head-classifier .score-row {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          font-size: 0.8rem;
        }

        .head-classifier .score-label {
          width: 7rem;
          color: var(--oz-fg-dim);
        }

        .head-classifier .score-bar-container {
          flex: 1;
          height: 4px;
          background: var(--oz-rule);
        }

        .head-classifier .score-bar {
          height: 100%;
        }

        .head-classifier .score-value {
          width: 2.6rem;
          text-align: right;
          font-variant-numeric: tabular-nums;
        }
      `}</style>
    </div>
  )
}
