/**
 * Feature comparison component for comparing features across SAEs.
 * Shows aligned features, similarity metrics, and divergences.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { GitCompare, ArrowRight, Layers, Zap } from 'lucide-react'
import { useThemeColors } from '../../hooks/useTheme'

interface FeatureMatch {
  feature_a: number
  feature_b: number
  similarity: number
  shared_tokens: string[]
  label_a: string
  label_b: string
}

interface ComparisonMetrics {
  overall_similarity: number
  matched_features: number
  unmatched_a: number
  unmatched_b: number
  top_matches: FeatureMatch[]
  divergent_features_a: number[]
  divergent_features_b: number[]
}

interface SAEInfo {
  id: string
  name: string
  num_features: number
  layer: number
  model: string
}

interface FeatureComparisonProps {
  saeA: SAEInfo
  saeB: SAEInfo
  comparison: ComparisonMetrics
  onFeatureSelect?: (saeId: string, featureIdx: number) => void
  className?: string
}

export function FeatureComparison({
  saeA,
  saeB,
  comparison,
  onFeatureSelect,
  className = '',
}: FeatureComparisonProps) {
  const matchChartRef = useRef<SVGSVGElement>(null)
  const [selectedMatch, setSelectedMatch] = useState<FeatureMatch | null>(null)
  const tc = useThemeColors()

  // Render feature matching visualization
  useEffect(() => {
    if (!matchChartRef.current) return

    d3.select(matchChartRef.current).selectAll('*').remove()

    const svg = d3.select(matchChartRef.current)
    const width = 600
    const height = 300
    const margin = { top: 40, right: 100, bottom: 40, left: 100 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Truncate long SAE names to fit within margins
    const truncate = (s: string, max: number) =>
      s.length > max ? s.slice(0, max - 1) + '…' : s
    const nameA = truncate(saeA.name, 25)
    const nameB = truncate(saeB.name, 25)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', tc.text)
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Feature Alignment')

    // Get top matches for visualization
    const topMatches = comparison.top_matches.slice(0, 15)

    // Y scales for both sides
    const yScaleA = d3.scalePoint()
      .domain(topMatches.map(m => String(m.feature_a)))
      .range([0, innerHeight])
      .padding(0.5)

    const yScaleB = d3.scalePoint()
      .domain(topMatches.map(m => String(m.feature_b)))
      .range([0, innerHeight])
      .padding(0.5)

    // Color scale for similarity
    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateRgb(tc.isLight ? '#ccc' : '#333', tc.isLight ? '#111' : '#fff'))

    // Draw SAE A labels on left
    svg.append('text')
      .attr('x', margin.left - 10)
      .attr('y', margin.top - 15)
      .attr('text-anchor', 'end')
      .attr('fill', tc.textMid)
      .attr('font-size', '12px')
      .text(nameA)

    // Draw SAE B labels on right
    svg.append('text')
      .attr('x', width - margin.right + 10)
      .attr('y', margin.top - 15)
      .attr('text-anchor', 'start')
      .attr('fill', tc.textMid)
      .attr('font-size', '12px')
      .text(nameB)

    // Draw connection lines
    topMatches.forEach((match) => {
      const y1 = yScaleA(String(match.feature_a)) || 0
      const y2 = yScaleB(String(match.feature_b)) || 0
      const isSelected = selectedMatch?.feature_a === match.feature_a

      // Connection line
      g.append('path')
        .attr('d', `M 0,${y1} C ${innerWidth / 2},${y1} ${innerWidth / 2},${y2} ${innerWidth},${y2}`)
        .attr('fill', 'none')
        .attr('stroke', colorScale(match.similarity))
        .attr('stroke-width', isSelected ? 3 : 1.5)
        .attr('opacity', isSelected ? 1 : 0.6)
        .style('cursor', 'pointer')
        .on('click', () => setSelectedMatch(match))
        .on('mouseenter', function() {
          d3.select(this).attr('stroke-width', 3).attr('opacity', 1)
        })
        .on('mouseleave', function() {
          if (!isSelected) {
            d3.select(this).attr('stroke-width', 1.5).attr('opacity', 0.6)
          }
        })

      // Left feature node
      g.append('circle')
        .attr('cx', 0)
        .attr('cy', y1)
        .attr('r', 6)
        .attr('fill', tc.textMid)
        .attr('stroke', isSelected ? tc.text : 'none')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', () => onFeatureSelect?.(saeA.id, match.feature_a))

      // Right feature node
      g.append('circle')
        .attr('cx', innerWidth)
        .attr('cy', y2)
        .attr('r', 6)
        .attr('fill', tc.textMid)
        .attr('stroke', isSelected ? tc.text : 'none')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', () => onFeatureSelect?.(saeB.id, match.feature_b))

      // Left feature label
      g.append('text')
        .attr('x', -10)
        .attr('y', y1)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('fill', tc.text)
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text(`F${match.feature_a}`)

      // Right feature label
      g.append('text')
        .attr('x', innerWidth + 10)
        .attr('y', y2)
        .attr('text-anchor', 'start')
        .attr('dominant-baseline', 'middle')
        .attr('fill', tc.text)
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text(`F${match.feature_b}`)

      // Similarity label
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', (y1 + y2) / 2 - 5)
        .attr('text-anchor', 'middle')
        .attr('fill', tc.text)
        .attr('font-size', '9px')
        .text(`${(match.similarity * 100).toFixed(0)}%`)
    })

    // Legend
    const legendWidth = 100
    const legendX = width - margin.right / 2 - legendWidth / 2
    const legendY = height - 20

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'similarity-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')

    gradient.selectAll('stop')
      .data(d3.range(0, 1.01, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(d))

    svg.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', 8)
      .style('fill', 'url(#similarity-gradient)')

    svg.append('text')
      .attr('x', legendX)
      .attr('y', legendY - 5)
      .attr('fill', tc.text)
      .attr('font-size', '9px')
      .text('Similarity')

    svg.append('text')
      .attr('x', legendX)
      .attr('y', legendY + 18)
      .attr('fill', tc.text)
      .attr('font-size', '8px')
      .text('0%')

    svg.append('text')
      .attr('x', legendX + legendWidth)
      .attr('y', legendY + 18)
      .attr('text-anchor', 'end')
      .attr('fill', tc.text)
      .attr('font-size', '8px')
      .text('100%')

  }, [comparison.top_matches, selectedMatch, saeA, saeB, onFeatureSelect, tc])

  return (
    <div className={`bg-black/40 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-2">
          <GitCompare className="w-5 h-5" style={{ color: tc.textMid }} />
          <h3 className="text-lg font-semibold text-white tracking-tight">
            Feature Comparison
          </h3>
        </div>
        <div className="flex items-center gap-3 text-sm text-white">
          <span style={{ color: tc.textMid }}>{saeA.name}</span>
          <ArrowRight className="w-4 h-4" />
          <span style={{ color: tc.textMid }}>{saeB.name}</span>
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-px bg-gray-800">
        <div className="bg-black/40 p-4 text-center">
          <div className="text-xs text-white uppercase tracking-wide mb-1">Overall Similarity</div>
          <div className="text-2xl font-mono text-white">
            {(comparison.overall_similarity * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-black/40 p-4 text-center">
          <div className="text-xs text-white uppercase tracking-wide mb-1">Matched Features</div>
          <div className="text-2xl font-mono text-green-400">
            {comparison.matched_features}
          </div>
        </div>
        <div className="bg-black/40 p-4 text-center">
          <div className="text-xs text-white uppercase tracking-wide mb-1">
            Unique to {saeA.name}
          </div>
          <div className="text-2xl font-mono" style={{ color: tc.textMid }}>
            {comparison.unmatched_a}
          </div>
        </div>
        <div className="bg-black/40 p-4 text-center">
          <div className="text-xs text-white uppercase tracking-wide mb-1">
            Unique to {saeB.name}
          </div>
          <div className="text-2xl font-mono" style={{ color: tc.textMid }}>
            {comparison.unmatched_b}
          </div>
        </div>
      </div>

      {/* Matching visualization */}
      {comparison.top_matches.length > 0 ? (
        <div className="p-4 border-t border-gray-800">
          <svg ref={matchChartRef} width="100%" height={300} />
        </div>
      ) : (
        <div className="p-4 border-t border-gray-800 flex flex-col items-center justify-center" style={{ minHeight: 120 }}>
          <GitCompare className="w-8 h-8 text-gray-700 mb-2" />
          <p className="text-white text-sm">
            No feature matches found above the similarity threshold.
          </p>
          <p className="text-white text-xs mt-1">
            These SAEs may capture very different feature representations.
          </p>
        </div>
      )}

      {/* Selected match details */}
      {selectedMatch && (
        <div className="p-4 border-t border-gray-800" style={{ background: tc.surface }}>
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4" style={{ color: tc.textMid }} />
            <span className="text-sm text-white font-semibold">Match Details</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-white uppercase tracking-wide mb-1">
                {saeA.name} - Feature {selectedMatch.feature_a}
              </div>
              <div className="text-sm" style={{ color: tc.textStrong }}>
                {selectedMatch.label_a || 'No label'}
              </div>
            </div>
            <div>
              <div className="text-xs text-white uppercase tracking-wide mb-1">
                {saeB.name} - Feature {selectedMatch.feature_b}
              </div>
              <div className="text-sm" style={{ color: tc.textStrong }}>
                {selectedMatch.label_b || 'No label'}
              </div>
            </div>
          </div>
          {selectedMatch.shared_tokens.length > 0 && (
            <div className="mt-3">
              <div className="text-xs text-white uppercase tracking-wide mb-2">
                Shared Activating Tokens
              </div>
              <div className="flex flex-wrap gap-1">
                {selectedMatch.shared_tokens.slice(0, 20).map((token, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 text-xs font-mono bg-gray-800 text-white rounded"
                  >
                    {token}
                  </span>
                ))}
                {selectedMatch.shared_tokens.length > 20 && (
                  <span className="px-2 py-0.5 text-xs text-white">
                    +{selectedMatch.shared_tokens.length - 20} more
                  </span>
                )}
              </div>
            </div>
          )}
          <div className="mt-3 flex items-center gap-4">
            <button
              onClick={() => onFeatureSelect?.(saeA.id, selectedMatch.feature_a)}
              style={{ color: tc.textMid }}
              className="text-xs transition-colors"
              onMouseEnter={(e) => e.currentTarget.style.color = tc.textStrong}
              onMouseLeave={(e) => e.currentTarget.style.color = tc.textMid}
            >
              View in {saeA.name}
            </button>
            <button
              onClick={() => onFeatureSelect?.(saeB.id, selectedMatch.feature_b)}
              style={{ color: tc.textMid }}
              className="text-xs transition-colors"
              onMouseEnter={(e) => e.currentTarget.style.color = tc.textStrong}
              onMouseLeave={(e) => e.currentTarget.style.color = tc.textMid}
            >
              View in {saeB.name}
            </button>
          </div>
        </div>
      )}

      {/* Model info footer */}
      <div className="grid grid-cols-2 gap-px bg-gray-800 border-t border-gray-800">
        <div className="bg-black/40 p-3">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-4 h-4" style={{ color: tc.textMid }} />
            <span className="text-xs text-white uppercase tracking-wide">{saeA.name}</span>
          </div>
          <div className="space-y-1 text-xs text-white">
            <div>Model: <span className="text-white">{saeA.model}</span></div>
            <div>Layer: <span className="text-white">{saeA.layer}</span></div>
            <div>Features: <span className="text-white">{saeA.num_features.toLocaleString()}</span></div>
          </div>
        </div>
        <div className="bg-black/40 p-3">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-4 h-4" style={{ color: tc.textMid }} />
            <span className="text-xs text-white uppercase tracking-wide">{saeB.name}</span>
          </div>
          <div className="space-y-1 text-xs text-white">
            <div>Model: <span className="text-white">{saeB.model}</span></div>
            <div>Layer: <span className="text-white">{saeB.layer}</span></div>
            <div>Features: <span className="text-white">{saeB.num_features.toLocaleString()}</span></div>
          </div>
        </div>
      </div>
    </div>
  )
}
