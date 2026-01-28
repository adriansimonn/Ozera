/**
 * AttentionPatternGrid component for displaying a color-coded grid of all attention heads.
 * Shows head types, importance, and allows selection for detailed view.
 */

import { useMemo } from 'react'
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
  colorMode?: 'type' | 'importance' | 'confidence'
}

export function AttentionPatternGrid({
  classifications,
  importance,
  numLayers,
  numHeads,
  onHeadSelect,
  selectedHead,
  showImportance = false,
  colorMode = 'type',
}: AttentionPatternGridProps) {
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

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 p-3 bg-slate-900/50 border border-slate-700/50 rounded-lg">
        <span className="text-xs text-slate-400 font-medium">Legend:</span>
        {Object.entries(HEAD_TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: color }}
            />
            <span className="text-xs text-slate-400">{HEAD_TYPE_NAMES[type as HeadType]}</span>
          </div>
        ))}
      </div>

      {/* Grid */}
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header row with head indices */}
          <div className="flex items-center">
            <div className="w-16 h-8 flex items-center justify-center text-xs text-slate-500">
              Layer/Head
            </div>
            {Array.from({ length: numHeads }).map((_, headIdx) => (
              <div
                key={headIdx}
                className="w-12 h-8 flex items-center justify-center text-xs text-slate-400 font-mono"
              >
                H{headIdx}
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {Array.from({ length: numLayers }).map((_, layerIdx) => (
            <div key={layerIdx} className="flex items-center">
              {/* Row label */}
              <div className="w-16 h-12 flex items-center justify-center text-xs text-slate-400 font-mono">
                L{layerIdx}
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
                    className={`
                      w-12 h-12 m-0.5 rounded-md transition-all relative group
                      ${isSelected ? 'ring-2 ring-cyan-400 ring-offset-2 ring-offset-slate-900' : ''}
                      hover:scale-110 hover:z-10
                    `}
                    style={{
                      backgroundColor: getCellColor(layerIdx, headIdx),
                      opacity: getCellOpacity(layerIdx, headIdx),
                    }}
                    title={classification
                      ? `${HEAD_TYPE_NAMES[classification.primary_type]} (${(classification.confidence * 100).toFixed(0)}%)`
                      : 'No classification'
                    }
                  >
                    {/* Show importance indicator if enabled */}
                    {showImportance && imp && (
                      <div
                        className="absolute bottom-0.5 left-1/2 -translate-x-1/2 h-1 rounded-full bg-white/80"
                        style={{ width: `${imp.overall_importance * 80}%` }}
                      />
                    )}

                    {/* Tooltip on hover */}
                    <div className="absolute opacity-0 group-hover:opacity-100 bottom-full left-1/2 -translate-x-1/2 mb-2 z-20 pointer-events-none transition-opacity">
                      <div className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs whitespace-nowrap shadow-lg">
                        {classification && (
                          <>
                            <div className="font-semibold" style={{ color: HEAD_TYPE_COLORS[classification.primary_type] }}>
                              {HEAD_TYPE_NAMES[classification.primary_type]}
                            </div>
                            <div className="text-slate-400">
                              Conf: {(classification.confidence * 100).toFixed(0)}%
                            </div>
                          </>
                        )}
                        {imp && (
                          <div className="text-slate-400">
                            Imp: {(imp.overall_importance * 100).toFixed(0)}%
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
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span>View mode:</span>
        <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
          {(['type', 'confidence', 'importance'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => {
                // This would need to be controlled by parent - just showing UI
              }}
              className={`
                px-2 py-1 rounded text-xs capitalize transition-all
                ${colorMode === mode ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'}
              `}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
