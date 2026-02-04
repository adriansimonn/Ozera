/**
 * Feature activation display component for SAE analysis.
 * Shows which features activate for a given input with strength heatmap.
 */

import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'

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
  selectedFeatures,
  onFeatureSelect,
  showTopK = 50,
  className = '',
}: FeatureActivationDisplayProps) {
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

    // Color scale - purple gradient for features
    const colorScale = d3.scaleSequential()
      .domain([0, maxValue])
      .interpolator(d3.interpolatePurples)

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add title
    svg.append('text')
      .attr('x', dimensions.width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-semibold')
      .attr('fill', '#e5e7eb')
      .text('Feature Activations')

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
          .attr('fill', value > 0 ? colorScale(value) : '#1a1a2e')
          .attr('stroke', isSelected ? '#a855f7' : '#1f2937')
          .attr('stroke-width', isSelected ? 2 : 0.5)
          .style('cursor', 'pointer')
          .on('mouseenter', function(event) {
            d3.select(this)
              .attr('stroke', '#ffffff')
              .attr('stroke-width', 2)

            const containerRect = containerRef.current?.getBoundingClientRect()
            if (containerRect) {
              setHoveredCell({
                position: posIdx,
                featureIdx: featIdx,
                value: value,
                token: activations.tokens[posIdx] || '',
                x: event.clientX - containerRect.left,
                y: event.clientY - containerRect.top,
              })
            }
          })
          .on('mousemove', function(event) {
            const containerRect = containerRef.current?.getBoundingClientRect()
            if (containerRect) {
              setHoveredCell(prev => prev ? {
                ...prev,
                x: event.clientX - containerRect.left,
                y: event.clientY - containerRect.top,
              } : null)
            }
          })
          .on('mouseleave', function() {
            const isSelected = selectedCell?.featureIdx === featIdx && selectedCell?.position === posIdx
            d3.select(this)
              .attr('stroke', isSelected ? '#a855f7' : '#1f2937')
              .attr('stroke-width', isSelected ? 2 : 0.5)
            setHoveredCell(null)
          })

        // Attach click handler
        rect.on('click', function(event) {
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
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')

    // Add feature labels on y-axis
    const yAxis = d3.axisLeft(yScale).tickFormat(i => `F${featuresToShow[parseInt(i)]}`)

    g.append('g')
      .call(yAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#9ca3af')

    // Axis labels
    svg.append('text')
      .attr('x', dimensions.width / 2)
      .attr('y', dimensions.height - 5)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm')
      .attr('fill', '#d1d5db')
      .text('Token Position')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -dimensions.height / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm')
      .attr('fill', '#d1d5db')
      .text('Feature Index')

    // Add color legend
    const legendWidth = 150
    const legendHeight = 12
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
      .attr('stop-color', d => colorScale(d * maxValue))

    svg.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#feature-legend-gradient)')
      .attr('stroke', '#374151')
      .attr('stroke-width', 1)

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(4)
      .tickFormat(d => d3.format('.2f')(d as number))

    svg.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#9ca3af')

  }, [activations, featuresToShow, selectedCell, dimensions])

  // Tooltip positioning
  const getTooltipStyle = () => {
    if (!hoveredCell) return {}

    const tooltipWidth = 220
    const tooltipHeight = 100
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
    <div className={`relative ${className}`} ref={containerRef}>
      {/* Summary stats */}
      <div className="flex gap-4 mb-4 text-sm">
        <div className="bg-black/30 border border-gray-800 px-3 py-2">
          <span className="text-gray-500 uppercase text-xs tracking-wide">Tokens</span>
          <div className="text-white font-mono">{activations.tokens.length}</div>
        </div>
        <div className="bg-black/30 border border-gray-800 px-3 py-2">
          <span className="text-gray-500 uppercase text-xs tracking-wide">Features Shown</span>
          <div className="text-white font-mono">{featuresToShow.length}</div>
        </div>
        <div className="bg-black/30 border border-gray-800 px-3 py-2">
          <span className="text-gray-500 uppercase text-xs tracking-wide">Avg L0</span>
          <div className="text-white font-mono">
            {(activations.active_features_per_position.reduce((a, b) => a + b, 0) /
              activations.active_features_per_position.length).toFixed(1)}
          </div>
        </div>
      </div>

      <svg
        ref={svgRef}
        width="100%"
        height={dimensions.height}
        className="bg-black/40 border border-gray-800"
      />

      {hoveredCell && (
        <div
          className="absolute z-20 bg-black/90 backdrop-blur-sm border border-gray-700 px-4 py-3 text-sm pointer-events-none"
          style={getTooltipStyle()}
        >
          <div className="text-white font-semibold mb-2 text-xs uppercase tracking-wide">
            Feature Activation
          </div>
          <div className="space-y-1 text-xs">
            <div className="text-gray-400">
              Token: <span className="text-white font-mono">"{hoveredCell.token}"</span>
            </div>
            <div className="text-gray-400">
              Feature: <span className="text-purple-300 font-mono">F{hoveredCell.featureIdx}</span>
            </div>
            <div className="text-gray-400">
              Value: <span className="text-white font-mono">{hoveredCell.value.toFixed(4)}</span>
            </div>
            <div className="text-gray-400">
              Position: <span className="text-white font-mono">{hoveredCell.position}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
