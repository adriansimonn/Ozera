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
        <h3>Classification Summary</h3>
        <div className="summary-buttons">
          {Object.entries(summary).map(([type, count]) => (
            <button
              key={type}
              onClick={() => setFilterType(filterType === type ? 'all' : type as HeadType)}
              className={`summary-btn ${filterType === type ? 'active' : ''}`}
              style={{
                backgroundColor: `${HEAD_TYPE_COLORS[type as HeadType]}20`,
                color: HEAD_TYPE_COLORS[type as HeadType],
                borderColor: HEAD_TYPE_COLORS[type as HeadType],
              }}
            >
              {HEAD_TYPE_NAMES[type as HeadType]}: {count}
            </button>
          ))}
          {filterType !== 'all' && (
            <button
              onClick={() => setFilterType('all')}
              className="show-all-btn"
            >
              Show All
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
                      style={{
                        backgroundColor: `${HEAD_TYPE_COLORS[classification.primary_type]}30`,
                        borderColor: HEAD_TYPE_COLORS[classification.primary_type],
                        color: HEAD_TYPE_COLORS[classification.primary_type],
                      }}
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
            Type: <span style={{ color: HEAD_TYPE_COLORS[hoveredHead.primary_type] }}>
              {HEAD_TYPE_NAMES[hoveredHead.primary_type]}
            </span>
          </div>
          <div className="hover-confidence">
            Confidence: <span>{(hoveredHead.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="hover-summary">
            {hoveredHead.pattern_summary}
          </div>
          <div className="hover-scores">
            {Object.entries(hoveredHead.scores).map(([type, score]) => (
              <div key={type} className="score-row">
                <span className="score-label">{type}:</span>
                <div className="score-bar-container">
                  <div
                    className="score-bar"
                    style={{
                      width: `${score * 100}%`,
                      backgroundColor: HEAD_TYPE_COLORS[type as HeadType] || '#6B7280',
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

        .summary-section {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
        }

        .summary-section h3 {
          margin: 0 0 0.75rem 0;
          font-size: 0.8rem;
          font-weight: 600;
          color: #ffffff;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .summary-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem;
        }

        .summary-btn {
          padding: 0.375rem 0.75rem;
          font-size: 0.75rem;
          font-weight: 500;
          transition: all 0.2s;
          border: 1px solid;
          cursor: pointer;
        }

        .summary-btn:hover {
          opacity: 0.8;
        }

        .summary-btn.active {
          box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.9), 0 0 0 4px currentColor;
        }

        .show-all-btn {
          padding: 0.375rem 0.75rem;
          font-size: 0.75rem;
          font-weight: 500;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #ffffff;
          cursor: pointer;
          transition: all 0.2s;
        }

        .show-all-btn:hover {
          background: rgba(255, 255, 255, 0.15);
        }

        .layers-container {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .layer-section {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem;
        }

        .layer-section h4 {
          margin: 0 0 0.75rem 0;
          font-size: 0.8rem;
          font-weight: 600;
          color: #ffffff;
        }

        .layer-heads {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }

        .head-btn {
          position: relative;
          padding: 0.5rem 0.75rem;
          font-size: 0.75rem;
          font-family: monospace;
          transition: all 0.2s;
          border: 1px solid;
          cursor: pointer;
        }

        .head-btn:hover {
          transform: scale(1.05);
        }

        .head-btn.filtered {
          opacity: 0.3;
        }

        .head-btn.selected {
          box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8);
        }

        .head-btn.hovered {
          box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.5);
        }

        .head-label {
          font-weight: 600;
        }

        .head-confidence {
          font-size: 0.625rem;
          opacity: 0.8;
        }

        .hover-detail {
          position: fixed;
          bottom: 1rem;
          right: 1rem;
          z-index: 50;
          background: rgba(0, 0, 0, 0.95);
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 1rem;
          max-width: 20rem;
          backdrop-filter: blur(8px);
        }

        .hover-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }

        .hover-dot {
          width: 0.75rem;
          height: 0.75rem;
        }

        .hover-title {
          font-weight: 600;
          color: #ffffff;
        }

        .hover-type {
          font-size: 0.875rem;
          color: #ffffff;
          margin-bottom: 0.25rem;
        }

        .hover-confidence {
          font-size: 0.875rem;
          color: #ffffff;
          margin-bottom: 0.5rem;
        }

        .hover-confidence span {
          color: #ffffff;
        }

        .hover-summary {
          font-size: 0.75rem;
          color: #ffffff;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          padding-top: 0.5rem;
          margin-top: 0.5rem;
        }

        .hover-scores {
          margin-top: 0.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .score-row {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.75rem;
        }

        .score-label {
          color: #ffffff;
          width: 6rem;
        }

        .score-bar-container {
          flex: 1;
          height: 0.375rem;
          background: rgba(255, 255, 255, 0.1);
          overflow: hidden;
        }

        .score-bar {
          height: 100%;
          transition: width 0.2s;
        }

        .score-value {
          color: #ffffff;
          width: 2.5rem;
          text-align: right;
        }

        /* Light mode (the .hover-detail panel stays dark in both themes) */
        [data-bg="light"] .head-classifier .summary-section,
        [data-bg="light"] .head-classifier .layer-section { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .head-classifier .summary-section h3,
        [data-bg="light"] .head-classifier .layer-section h4 { color: #1d1d1f; }
        [data-bg="light"] .head-classifier .show-all-btn { background: rgba(0,0,0,0.05); border-color: rgba(0,0,0,0.15); color: #1d1d1f; }
        [data-bg="light"] .head-classifier .show-all-btn:hover { background: rgba(0,0,0,0.08); }
        [data-bg="light"] .head-classifier .head-btn.selected { box-shadow: 0 0 0 2px rgba(0,0,0,0.8); }
      `}</style>
    </div>
  )
}
