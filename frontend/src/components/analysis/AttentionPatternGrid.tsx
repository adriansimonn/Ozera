/**
 * AttentionPatternGrid component for displaying a color-coded grid of all attention heads.
 * Shows head types, importance, and allows selection for detailed view.
 */

import { useState, useMemo, useRef, useEffect } from 'react'
import type { HeadClassification, HeadImportance, HeadType } from '../../types/analysis'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../../types/analysis'

interface AttentionPatternGridProps {
  classifications: HeadClassification[]
  importance?: HeadImportance[]
  numLayers: number
  numHeads: number
  onHeadSelect?: (layer: number, head: number) => void
  selectedHead?: { layer: number; head: number } | null
  showImportance?: boolean
}

type ColorMode = 'type' | 'importance' | 'confidence'

export function AttentionPatternGrid({
  classifications,
  importance,
  numLayers,
  numHeads,
  onHeadSelect,
  selectedHead,
  showImportance = false,
}: AttentionPatternGridProps) {
  const [colorMode, setColorMode] = useState<ColorMode>('type')
  const containerRef = useRef<HTMLDivElement>(null)
  const [cellSize, setCellSize] = useState(48) // Default 3rem = 48px

  // Calculate cell size based on container width and number of heads
  useEffect(() => {
    const calculateCellSize = () => {
      if (!containerRef.current) return

      const containerWidth = containerRef.current.offsetWidth
      const labelWidth = 40 // Reduced for compact display
      const availableWidth = containerWidth - labelWidth - 10
      const cellGap = 1 // 0.5px margin on each side

      // Calculate max cell size that fits all heads
      const maxCellSize = Math.floor((availableWidth - (numHeads * cellGap)) / numHeads)

      // Clamp between min (6px for very wide models) and max (48px)
      const newCellSize = Math.max(6, Math.min(48, maxCellSize))
      setCellSize(newCellSize)
    }

    calculateCellSize()

    const resizeObserver = new ResizeObserver(calculateCellSize)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => resizeObserver.disconnect()
  }, [numHeads])

  // Create a map for quick lookup
  const classificationMap = useMemo(() => {
    const map = new Map<string, HeadClassification>()
    for (const c of classifications) {
      map.set(`${c.layer}_${c.head}`, c)
    }
    return map
  }, [classifications])

  const importanceMap = useMemo(() => {
    if (!importance) return new Map<string, HeadImportance>()
    const map = new Map<string, HeadImportance>()
    for (const h of importance) {
      map.set(`${h.layer}_${h.head}`, h)
    }
    return map
  }, [importance])

  // Get color for a cell based on mode
  const getCellColor = (layer: number, head: number): string => {
    const key = `${layer}_${head}`
    const classification = classificationMap.get(key)
    const imp = importanceMap.get(key)

    if (colorMode === 'type' && classification) {
      return HEAD_TYPE_COLORS[classification.primary_type]
    } else if (colorMode === 'importance' && imp) {
      // Gradient from dark to bright cyan based on importance
      const intensity = Math.round(imp.overall_importance * 255)
      return `rgb(${intensity * 0.2}, ${intensity}, ${intensity * 0.9})`
    } else if (colorMode === 'confidence' && classification) {
      // Gradient from dark to bright based on confidence
      const intensity = Math.round(classification.confidence * 255)
      return `rgb(${intensity * 0.4}, ${intensity * 0.8}, ${intensity})`
    }
    return '#1e293b'
  }

  // Get opacity based on importance or confidence
  const getCellOpacity = (layer: number, head: number): number => {
    const key = `${layer}_${head}`
    const imp = importanceMap.get(key)
    const classification = classificationMap.get(key)

    if (showImportance && imp) {
      return 0.3 + imp.overall_importance * 0.7
    }
    if (classification) {
      return 0.5 + classification.confidence * 0.5
    }
    return 0.3
  }

  // Dynamic font size based on cell size
  const fontSize = cellSize < 16 ? '0.4rem' : cellSize < 30 ? '0.5rem' : '0.75rem'
  const showLabels = cellSize >= 16
  const showHeaderLabels = cellSize >= 12

  // Check if importance data is available
  const hasImportanceData = importance && importance.length > 0

  return (
    <div className="attention-pattern-grid" ref={containerRef}>
      {/* Legend */}
      <div className="legend">
        <span className="legend-label">Legend:</span>
        {colorMode === 'type' ? (
          Object.entries(HEAD_TYPE_COLORS).map(([type, color]) => (
            <div key={type} className="legend-item">
              <div
                className="legend-dot"
                style={{ backgroundColor: color }}
              />
              <span>{HEAD_TYPE_NAMES[type as HeadType]}</span>
            </div>
          ))
        ) : (
          <div className="gradient-legend">
            <span className="gradient-label">0%</span>
            <div
              className="gradient-bar"
              style={{
                background: colorMode === 'confidence'
                  ? 'linear-gradient(to right, rgb(25, 51, 64), rgb(102, 204, 255))'
                  : 'linear-gradient(to right, rgb(0, 0, 0), rgb(51, 255, 230))'
              }}
            />
            <span className="gradient-label">100%</span>
          </div>
        )}
      </div>

      {/* Grid */}
      <div className="grid-container">
        <div className="grid-inner">
          {/* Header row with head indices */}
          {showHeaderLabels && (
            <div className="grid-row">
              <div className="grid-corner" style={{ fontSize, width: cellSize < 16 ? '2rem' : '2.5rem' }}>
                {showLabels ? 'L/H' : ''}
              </div>
              {Array.from({ length: numHeads }).map((_, headIdx) => (
                <div
                  key={headIdx}
                  className="grid-header-cell"
                  style={{ width: cellSize, fontSize }}
                >
                  {showLabels ? headIdx : ''}
                </div>
              ))}
            </div>
          )}

          {/* Grid rows */}
          {Array.from({ length: numLayers }).map((_, layerIdx) => (
            <div key={layerIdx} className="grid-row">
              {/* Row label */}
              <div
                className="grid-row-label"
                style={{ height: cellSize, fontSize, width: cellSize < 16 ? '2rem' : '2.5rem' }}
              >
                {showLabels ? layerIdx : ''}
              </div>

              {/* Head cells */}
              {Array.from({ length: numHeads }).map((_, headIdx) => {
                const key = `${layerIdx}_${headIdx}`
                const classification = classificationMap.get(key)
                const imp = importanceMap.get(key)
                const isSelected = selectedHead?.layer === layerIdx && selectedHead?.head === headIdx

                return (
                  <button
                    key={headIdx}
                    onClick={() => onHeadSelect?.(layerIdx, headIdx)}
                    className={`grid-cell ${isSelected ? 'selected' : ''}`}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: getCellColor(layerIdx, headIdx),
                      opacity: getCellOpacity(layerIdx, headIdx),
                    }}
                    title={classification
                      ? `L${layerIdx} H${headIdx}: ${HEAD_TYPE_NAMES[classification.primary_type]} (${(classification.confidence * 100).toFixed(0)}%)`
                      : `L${layerIdx} H${headIdx}: No classification`
                    }
                  >
                    {/* Show importance indicator if enabled */}
                    {showImportance && imp && (
                      <div
                        className="importance-bar"
                        style={{ width: `${imp.overall_importance * 80}%` }}
                      />
                    )}

                    {/* Tooltip on hover */}
                    <div className="cell-tooltip">
                      <div className="tooltip-content">
                        <div className="tooltip-position">L{layerIdx} H{headIdx}</div>
                        {classification && colorMode === 'type' && (
                          <>
                            <div className="tooltip-type" style={{ color: HEAD_TYPE_COLORS[classification.primary_type] }}>
                              {HEAD_TYPE_NAMES[classification.primary_type]}
                            </div>
                            <div className="tooltip-detail">
                              Confidence: {(classification.confidence * 100).toFixed(0)}%
                            </div>
                          </>
                        )}
                        {classification && colorMode === 'confidence' && (
                          <div className="tooltip-metric">
                            Confidence: {(classification.confidence * 100).toFixed(1)}%
                          </div>
                        )}
                        {imp && colorMode === 'importance' && (
                          <div className="tooltip-metric">
                            Importance: {(imp.overall_importance * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Color mode toggle */}
      <div className="view-mode">
        <span className="view-mode-label">View mode:</span>
        <div className="view-mode-buttons">
          {(['type', 'confidence'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setColorMode(mode)}
              className={`view-mode-btn ${colorMode === mode ? 'active' : ''}`}
            >
              {mode}
            </button>
          ))}
          {hasImportanceData && (
            <button
              onClick={() => setColorMode('importance')}
              className={`view-mode-btn ${colorMode === 'importance' ? 'active' : ''}`}
            >
              importance
            </button>
          )}
        </div>
      </div>

      <style>{`
        .attention-pattern-grid {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          width: 100%;
        }

        .legend {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 1rem;
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .legend-label {
          font-size: 0.75rem;
          color: #ffffff;
          font-weight: 500;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          font-size: 0.75rem;
          color: #ffffff;
        }

        .legend-dot {
          width: 0.75rem;
          height: 0.75rem;
        }

        .gradient-legend {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex: 1;
          max-width: 300px;
        }

        .gradient-bar {
          flex: 1;
          height: 0.75rem;
          min-width: 120px;
        }

        .gradient-label {
          font-size: 0.7rem;
          color: #ffffff;
        }

        .grid-container {
          overflow-x: auto;
          overflow-y: visible;
          max-width: 100%;
          padding-bottom: 0.5rem;
        }

        .grid-container::-webkit-scrollbar {
          height: 6px;
        }

        .grid-container::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
        }

        .grid-container::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
        }

        .grid-container::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .grid-inner {
          display: inline-block;
          min-width: min-content;
        }

        .grid-row {
          display: flex;
          align-items: center;
        }

        .grid-corner {
          height: 1.25rem;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          flex-shrink: 0;
        }

        .grid-header-cell {
          height: 1.25rem;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-family: monospace;
          flex-shrink: 0;
        }

        .grid-row-label {
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-family: monospace;
          flex-shrink: 0;
        }

        .grid-cell {
          margin: 0.5px;
          border-radius: 2px;
          transition: transform 0.15s, box-shadow 0.15s;
          position: relative;
          border: none;
          cursor: pointer;
          flex-shrink: 0;
        }

        .grid-cell:hover {
          transform: scale(1.2);
          z-index: 10;
        }

        .grid-cell.selected {
          box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8);
        }

        .importance-bar {
          position: absolute;
          bottom: 2px;
          left: 50%;
          transform: translateX(-50%);
          height: 3px;
          background: rgba(255, 255, 255, 0.8);
        }

        .cell-tooltip {
          position: absolute;
          opacity: 0;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          margin-bottom: 0.5rem;
          z-index: 20;
          pointer-events: none;
          transition: opacity 0.2s;
        }

        .grid-cell:hover .cell-tooltip {
          opacity: 1;
        }

        .tooltip-content {
          background: rgba(0, 0, 0, 0.95);
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 0.375rem 0.5rem;
          font-size: 0.7rem;
          white-space: nowrap;
        }

        .tooltip-position {
          color: #ffffff;
          margin-bottom: 0.25rem;
          font-family: monospace;
        }

        .tooltip-type {
          font-weight: 600;
        }

        .tooltip-detail {
          color: #ffffff;
        }

        .tooltip-metric {
          color: #ffffff;
          font-weight: 500;
        }

        .view-mode {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.75rem;
          color: #ffffff;
        }

        .view-mode-label {
          color: #ffffff;
        }

        .view-mode-buttons {
          display: flex;
          gap: 2px;
          background: rgba(0, 0, 0, 0.3);
          padding: 2px;
        }

        .view-mode-btn {
          padding: 0.375rem 0.625rem;
          font-size: 0.75rem;
          text-transform: capitalize;
          transition: all 0.2s;
          background: transparent;
          border: none;
          color: #ffffff;
          cursor: pointer;
        }

        .view-mode-btn:hover {
          color: #ffffff;
        }

        .view-mode-btn.active {
          background: rgba(255, 255, 255, 0.15);
          color: #ffffff;
        }

        /* Light mode (cell tooltips stay dark in both themes) */
        [data-bg="light"] .attention-pattern-grid .legend { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .attention-pattern-grid .legend-label,
        [data-bg="light"] .attention-pattern-grid .legend-item,
        [data-bg="light"] .attention-pattern-grid .gradient-label,
        [data-bg="light"] .attention-pattern-grid .grid-corner,
        [data-bg="light"] .attention-pattern-grid .grid-header-cell,
        [data-bg="light"] .attention-pattern-grid .grid-row-label,
        [data-bg="light"] .attention-pattern-grid .view-mode,
        [data-bg="light"] .attention-pattern-grid .view-mode-label,
        [data-bg="light"] .attention-pattern-grid .view-mode-btn { color: #1d1d1f; }
        [data-bg="light"] .attention-pattern-grid .view-mode-buttons { background: rgba(0,0,0,0.05); }
        [data-bg="light"] .attention-pattern-grid .view-mode-btn.active { background: rgba(0,0,0,0.12); }
        [data-bg="light"] .attention-pattern-grid .grid-cell.selected { box-shadow: 0 0 0 2px rgba(0,0,0,0.8); }
      `}</style>
    </div>
  )
}
