/**
 * Feature activation display component for SAE analysis.
 * Shows which features activate for a given input with strength heatmap.
 */

import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { useThemeColors } from '../../hooks/useTheme'
import { sequentialScale } from '../../styles/appTheme'
import { MONO } from '../home/homeKit'
import { pointerIn } from '../lab/labTheme'
import type { FeatureLabels } from '../../hooks/useFeatureLabels'
interface FeatureActivation {
  feature_idx: number
  activation_value: number
  rank: number
  percentile: number
}

interface TokenFeatureActivations {
  position: number
  token: string
  token_id: number
  top_features: FeatureActivation[]
  total_active_features: number
  l0_sparsity: number
  l1_norm: number
}

interface SequenceFeatureActivations {
  tokens: string[]
  token_ids: number[]
  per_token_activations: TokenFeatureActivations[]
  feature_activation_matrix: number[][]
  active_features_per_position: number[]
  most_active_features: number[]
}

interface FeatureActivationDisplayProps {
  activations: SequenceFeatureActivations
  selectedFeatures?: number[]
  onFeatureSelect?: (featureIdx: number) => void
  showTopK?: number
  className?: string
  /** Autointerp labels of the features shown */
  labels?: FeatureLabels
}

interface SelectedCell {
  featureIdx: number
  position: number
}

interface HoveredCell {
  position: number
  featureIdx: number
  value: number
  token: string
  x: number
  y: number
}

export function FeatureActivationDisplay({
  activations,
  selectedFeatures: _selectedFeatures,
  onFeatureSelect,
  showTopK = 50,
  className = '',
  labels,
}: FeatureActivationDisplayProps) {
  const tc = useThemeColors()
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null)
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 })

  // Always show top-k most active features (selectedFeatures is only used for highlighting)
  // Memoize to prevent creating new array on every render
  const featuresToShow = useMemo(
    () => activations.most_active_features.slice(0, showTopK),
    [activations.most_active_features, showTopK]
  )

  // Update dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth
        const containerHeight = Math.min(500, Math.max(300, featuresToShow.length * 8 + 100))
        setDimensions({ width: containerWidth, height: containerHeight })
      }
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => resizeObserver.disconnect()
  }, [featuresToShow.length])

  useEffect(() => {
    if (!svgRef.current || !activations.feature_activation_matrix) return

    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const margin = { top: 60, right: 30, bottom: 60, left: 80 }
    const innerWidth = dimensions.width - margin.left - margin.right
    const innerHeight = dimensions.height - margin.top - margin.bottom

    const seqLen = activations.tokens.length
    const numFeatures = featuresToShow.length

    // Extract data for selected features
    const matrix = featuresToShow.map(featIdx =>
      activations.feature_activation_matrix.map(pos => pos[featIdx] || 0)
    )

    // Create scales
    const xScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, innerWidth])
      .padding(0.02)

    const yScale = d3.scaleBand()
      .domain(d3.range(numFeatures).map(String))
      .range([0, innerHeight])
      .padding(0.02)

    // Find max value for color scale
    const maxValue = d3.max(matrix.flat()) || 1

    // The app's sequential scale: the theme's ramp on ink, grays on white
    const scale = sequentialScale(tc.isLight)
    const colorScale = (value: number) => scale(value / maxValue)

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add title
    svg.append('text')
      .attr('x', margin.left)
      .attr('y', 25)
      .attr('font-size', 17)
      .attr('font-weight', 500)
      .attr('letter-spacing', '-0.02em')
      .attr('fill', tc.text)
      .text('Feature activations')

    // Create heatmap cells
    matrix.forEach((featureRow, featDisplayIdx) => {
      const featIdx = featuresToShow[featDisplayIdx]

      featureRow.forEach((value, posIdx) => {
        const isSelected = selectedCell?.featureIdx === featIdx && selectedCell?.position === posIdx
        const rect = g.append('rect')
          .attr('x', xScale(String(posIdx)) || 0)
          .attr('y', yScale(String(featDisplayIdx)) || 0)
          .attr('width', xScale.bandwidth())
          .attr('height', yScale.bandwidth())
          .attr('fill', colorScale(value > 0 ? value : 0))
          .attr('stroke', isSelected ? tc.mark : tc.border)
          .attr('stroke-width', isSelected ? 2 : 0.5)
          .style('cursor', 'pointer')
          .on('mouseenter', function(event) {
            d3.select(this)
              .attr('stroke', tc.text)
              .attr('stroke-width', 1.5)

            // The pointer in the container's own pixels (the page is zoomed, see pointerIn)
            if (containerRef.current) {
              setHoveredCell({
                position: posIdx,
                featureIdx: featIdx,
                value: value,
                token: activations.tokens[posIdx] || '',
                ...pointerIn(event, containerRef.current),
              })
            }
          })
          .on('mousemove', function(event) {
            if (containerRef.current) {
              const at = pointerIn(event, containerRef.current)
              setHoveredCell(prev => prev ? { ...prev, ...at } : null)
            }
          })
          .on('mouseleave', function() {
            const isSelected = selectedCell?.featureIdx === featIdx && selectedCell?.position === posIdx
            d3.select(this)
              .attr('stroke', isSelected ? tc.mark : tc.border)
              .attr('stroke-width', isSelected ? 2 : 0.5)
            setHoveredCell(null)
          })

        // Attach click handler
        rect.on('click', function(_event) {
          setSelectedCell({ featureIdx: featIdx, position: posIdx })
          if (onFeatureSelect) {
            onFeatureSelect(featIdx)
          }
        })
      })
    })

    // Add token labels on x-axis
    const xAxis = d3.axisBottom(xScale).tickFormat(i => {
      const token = activations.tokens[parseInt(i)]
      return token ? token.substring(0, 6) : i
    })

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 12)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.text)
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')

    // Add feature labels on y-axis
    const yAxis = d3.axisLeft(yScale).tickFormat(i => `F${featuresToShow[parseInt(i)]}`)

    g.append('g')
      .call(yAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 12)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.text)

    // Axis labels
    svg.append('text')
      .attr('x', dimensions.width / 2)
      .attr('y', dimensions.height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 13.5)
      .attr('fill', tc.textMuted)
      .text('Token position')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -dimensions.height / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 13.5)
      .attr('fill', tc.textMuted)
      .text('Feature index')

    // Add color legend
    const legendWidth = 150
    const legendHeight = 8
    const legendX = dimensions.width - legendWidth - 30
    const legendY = 20

    const legendScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([0, legendWidth])

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'feature-legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')

    gradient.selectAll('stop')
      .data(d3.range(0, 1.01, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => scale(d))

    svg.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#feature-legend-gradient)')
      .attr('stroke', tc.borderStrong)
      .attr('stroke-width', 1)

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(4)
      .tickFormat(d => d3.format('.2f')(d as number))

    svg.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 11)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.textMuted)

  }, [activations, featuresToShow, selectedCell, dimensions, tc.text, tc.textMuted, tc.border, tc.borderStrong, tc.mark, tc.isLight])

  // Tooltip positioning
  const getTooltipStyle = () => {
    if (!hoveredCell) return {}

    const tooltipWidth = 290
    const tooltipHeight = labels?.has(hoveredCell.featureIdx) ? 140 : 100
    const offset = 12

    let x = hoveredCell.x + offset
    let y = hoveredCell.y + offset

    if (x + tooltipWidth > dimensions.width) {
      x = hoveredCell.x - tooltipWidth - offset
    }
    if (y + tooltipHeight > dimensions.height) {
      y = hoveredCell.y - tooltipHeight - offset
    }

    return { left: `${x}px`, top: `${y}px` }
  }

  return (
    <div className={`feature-activation-display relative ${className}`} ref={containerRef}>
      {/* Summary stats */}
      <div className="fad-stats">
        <div className="fad-stat">
          <span className="fad-stat-label">Tokens</span>
          <div className="fad-stat-value">{activations.tokens.length}</div>
        </div>
        <div className="fad-stat">
          <span className="fad-stat-label">Features shown</span>
          <div className="fad-stat-value">{featuresToShow.length}</div>
        </div>
        <div className="fad-stat">
          <span className="fad-stat-label">Avg L0</span>
          <div className="fad-stat-value">
            {(activations.active_features_per_position.reduce((a, b) => a + b, 0) /
              activations.active_features_per_position.length).toFixed(1)}
          </div>
        </div>
      </div>

      <svg
        ref={svgRef}
        width="100%"
        height={dimensions.height}
      />

      {hoveredCell && (
        <div
          className="oz-tip absolute z-20"
          style={getTooltipStyle()}
        >
          <div className="font-medium">
            <span className="font-mono">{hoveredCell.value.toFixed(4)}</span> · feature{' '}
            <span className="font-mono">F{hoveredCell.featureIdx}</span>
          </div>
          {labels?.get(hoveredCell.featureIdx) && (
            <div style={{ maxWidth: 260, whiteSpace: 'normal' }}>{labels.get(hoveredCell.featureIdx)?.explanation}</div>
          )}
          <div className="text-dim">
            Token <span className="font-mono">"{hoveredCell.token}"</span> at position{' '}
            <span className="font-mono">{hoveredCell.position}</span>
          </div>
        </div>
      )}

      <style>{`
        /* Headline numbers, each under a rule */
        .feature-activation-display .fad-stats {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem 1.5rem;
          margin-bottom: 1.25rem;
        }
        .feature-activation-display .fad-stat {
          min-width: 9rem;
          padding-top: 0.6rem;
          border-top: 1px solid var(--oz-rule-strong);
        }
        .feature-activation-display .fad-stat-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
        }
        .feature-activation-display .fad-stat-value {
          margin-top: 0.2rem;
          font-family: var(--oz-mono);
          font-size: 1.35rem;
          letter-spacing: -0.02em;
          line-height: 1.2;
        }
      `}</style>
    </div>
  )
}
