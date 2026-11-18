/**
 * AttentionPatternGrid component for displaying a color-coded grid of all attention heads.
 * Shows head types, importance, and allows selection for detailed view.
 */

import { useState, useMemo, useRef, useEffect } from 'react'
import type { HeadClassification, HeadImportance, HeadType } from '../../types/analysis'
import { HEAD_TYPE_COLORS, HEAD_TYPE_NAMES } from '../../types/analysis'
import { useTheme } from '../../hooks/useTheme'
import { sequentialScale } from '../../styles/appTheme'

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
  // Importance and confidence are drawn in the app's sequential scale
  const { isLight } = useTheme()
  const scale = useMemo(() => sequentialScale(isLight), [isLight])
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
      return scale(imp.overall_importance)
    } else if (colorMode === 'confidence' && classification) {
      return scale(classification.confidence)
    }
    return 'var(--oz-fill-strong)'
  }

  // Get opacity based on importance or confidence
  const getCellOpacity = (layer: number, head: number): number => {
    const key = `${layer}_${head}`
    const imp = importanceMap.get(key)
    const classification = classificationMap.get(key)

    // The sequential scale already says how much; only a type's color is dimmed by confidence
    if (colorMode !== 'type') return 1
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
        <span className="legend-label">{colorMode === 'type' ? 'Head types' : colorMode === 'confidence' ? 'Confidence' : 'Importance'}</span>
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
              style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map(scale).join(', ')})` }}
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
                            <div className="tooltip-type">
                              <i style={{ background: HEAD_TYPE_COLORS[classification.primary_type] }} />
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
        <span className="view-mode-label">Color by</span>
        <div className="oz-chips" role="radiogroup" aria-label="Color by">
          {(['type', 'confidence'] as const).map((mode) => (
            <button
              key={mode}
              role="radio"
              aria-checked={colorMode === mode}
              onClick={() => setColorMode(mode)}
              className="oz-chip"
            >
              {mode === 'type' ? 'Type' : 'Confidence'}
            </button>
          ))}
          {hasImportanceData && (
            <button
              role="radio"
              aria-checked={colorMode === 'importance'}
              onClick={() => setColorMode('importance')}
              className="oz-chip"
            >
              Importance
            </button>
          )}
        </div>
      </div>

      <style>{`
        .attention-pattern-grid {
          display: flex;
          flex-direction: column;
          gap: 1.1rem;
          width: 100%;
        }

        .attention-pattern-grid .legend {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.4rem 1.1rem;
          font-size: 0.85rem;
        }

        .attention-pattern-grid .legend-label,
        .attention-pattern-grid .view-mode-label {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }

        .attention-pattern-grid .legend-item {
          display: flex;
          align-items: center;
          gap: 0.45rem;
        }

        .attention-pattern-grid .legend-dot {
          width: 9px;
          height: 9px;
        }

        .attention-pattern-grid .gradient-legend {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          flex: 1;
          max-width: 300px;
        }

        .attention-pattern-grid .gradient-bar {
          flex: 1;
          height: 8px;
          min-width: 120px;
          border: 1px solid var(--oz-rule-strong);
        }

        .attention-pattern-grid .gradient-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-variant-numeric: tabular-nums;
        }

        .attention-pattern-grid .grid-container {
          overflow-x: auto;
          overflow-y: visible;
          max-width: 100%;
          padding-bottom: 0.5rem;
        }

        .attention-pattern-grid .grid-inner {
          display: inline-block;
          min-width: min-content;
        }

        .attention-pattern-grid .grid-row {
          display: flex;
          align-items: center;
        }

        .attention-pattern-grid .grid-corner,
        .attention-pattern-grid .grid-header-cell,
        .attention-pattern-grid .grid-row-label {
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          color: var(--oz-fg-dim);
          font-family: var(--oz-mono);
        }

        .attention-pattern-grid .grid-corner,
        .attention-pattern-grid .grid-header-cell {
          height: 1.5rem;
        }

        /* A head: one cell of the layers-by-heads grid */
        .attention-pattern-grid .grid-cell {
          margin: 1px;
          position: relative;
          border: none;
          cursor: pointer;
          flex-shrink: 0;
        }

        .attention-pattern-grid .grid-cell:hover {
          z-index: 10;
          outline: 1.5px solid var(--oz-fg);
          outline-offset: 1px;
          opacity: 1 !important;
        }

        .attention-pattern-grid .grid-cell.selected {
          z-index: 5;
          outline: 2px solid var(--oz-fg);
          outline-offset: 1px;
          opacity: 1 !important;
        }

        .attention-pattern-grid .importance-bar {
          position: absolute;
          bottom: 2px;
          left: 50%;
          transform: translateX(-50%);
          height: 2px;
          background: var(--oz-fg);
        }

        .attention-pattern-grid .cell-tooltip {
          position: absolute;
          display: none;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          margin-bottom: 0.5rem;
          z-index: 20;
          pointer-events: none;
        }

        .attention-pattern-grid .grid-cell:hover .cell-tooltip {
          display: block;
        }

        .attention-pattern-grid .tooltip-content {
          padding: 0.45rem 0.6rem;
          background: var(--oz-bg);
          border: 1px solid var(--oz-rule-strong);
          color: var(--oz-fg);
          font-size: 0.8rem;
          text-align: left;
          white-space: nowrap;
        }

        .attention-pattern-grid .tooltip-position {
          margin-bottom: 0.2rem;
          font-family: var(--oz-mono);
          font-weight: 500;
        }

        .attention-pattern-grid .tooltip-type {
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }

        .attention-pattern-grid .tooltip-type i {
          width: 8px;
          height: 8px;
        }

        .attention-pattern-grid .tooltip-detail,
        .attention-pattern-grid .tooltip-metric {
          color: var(--oz-fg-dim);
        }

        .attention-pattern-grid .view-mode {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
      `}</style>
    </div>
  )
}
