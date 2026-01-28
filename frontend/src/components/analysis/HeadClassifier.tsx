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
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Classification Summary</h3>
        <div className="flex flex-wrap gap-3">
          {Object.entries(summary).map(([type, count]) => (
            <button
              key={type}
              onClick={() => setFilterType(filterType === type ? 'all' : type as HeadType)}
              className={`
                px-3 py-1.5 rounded-full text-xs font-medium transition-all
                ${filterType === type
                  ? 'ring-2 ring-offset-2 ring-offset-slate-900'
                  : 'hover:opacity-80'
                }
              `}
              style={{
                backgroundColor: `${HEAD_TYPE_COLORS[type as HeadType]}20`,
                color: HEAD_TYPE_COLORS[type as HeadType],
                borderColor: HEAD_TYPE_COLORS[type as HeadType],
                borderWidth: '1px',
              }}
            >
              {HEAD_TYPE_NAMES[type as HeadType]}: {count}
            </button>
          ))}
          {filterType !== 'all' && (
            <button
              onClick={() => setFilterType('all')}
              className="px-3 py-1.5 rounded-full text-xs font-medium bg-slate-700 text-slate-300 hover:bg-slate-600 transition-all"
            >
              Show All
            </button>
          )}
        </div>
      </div>

      {/* Classifications by Layer */}
      <div className="space-y-4">
        {Array.from({ length: numLayers }).map((_, layerIdx) => {
          const layerHeads = groupedByLayer[layerIdx] || []
          if (filterType !== 'all' && layerHeads.length === 0) return null

          return (
            <div key={layerIdx} className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-slate-400 mb-3">Layer {layerIdx}</h4>
              <div className="flex flex-wrap gap-2">
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
                      className={`
                        relative px-3 py-2 rounded-lg text-xs font-mono transition-all
                        ${isFiltered ? 'opacity-30' : ''}
                        ${isSelected ? 'ring-2 ring-cyan-400' : ''}
                        ${isHovered && !isSelected ? 'ring-1 ring-white/50' : ''}
                        hover:scale-105
                      `}
                      style={{
                        backgroundColor: `${HEAD_TYPE_COLORS[classification.primary_type]}30`,
                        borderColor: HEAD_TYPE_COLORS[classification.primary_type],
                        borderWidth: '1px',
                        color: HEAD_TYPE_COLORS[classification.primary_type],
                      }}
                    >
                      <div className="font-semibold">H{headIdx}</div>
                      <div className="text-[10px] opacity-80">
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
        <div className="fixed bottom-4 right-4 z-50 bg-slate-800/95 border border-slate-600 rounded-lg p-4 shadow-xl backdrop-blur-sm max-w-sm">
          <div className="flex items-center gap-2 mb-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: HEAD_TYPE_COLORS[hoveredHead.primary_type] }}
            />
            <span className="font-semibold text-slate-200">
              Layer {hoveredHead.layer}, Head {hoveredHead.head}
            </span>
          </div>
          <div className="text-sm text-slate-400 mb-2">
            Type: <span style={{ color: HEAD_TYPE_COLORS[hoveredHead.primary_type] }}>
              {HEAD_TYPE_NAMES[hoveredHead.primary_type]}
            </span>
          </div>
          <div className="text-sm text-slate-400 mb-2">
            Confidence: <span className="text-cyan-400">{(hoveredHead.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="text-xs text-slate-500 border-t border-slate-700 pt-2 mt-2">
            {hoveredHead.pattern_summary}
          </div>
          <div className="mt-2 space-y-1">
            {Object.entries(hoveredHead.scores).map(([type, score]) => (
              <div key={type} className="flex items-center gap-2 text-xs">
                <span className="text-slate-500 w-24">{type}:</span>
                <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${score * 100}%`,
                      backgroundColor: HEAD_TYPE_COLORS[type as HeadType] || '#6B7280',
                    }}
                  />
                </div>
                <span className="text-slate-400 w-10 text-right">{(score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
