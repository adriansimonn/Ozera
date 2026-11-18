/**
 * Attention heatmap visualization component.
 * Displays attention weights as an interactive heatmap, in the app's sequential scale.
 */

import { memo, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { TensorData } from '../../types/model'
import { ExportButton } from '../export'
import { useThemeColors } from '../../hooks/useTheme'
import { sequentialScale } from '../../styles/appTheme'
import { pointerIn } from '../lab/labTheme'
import { MONO } from '../home/homeKit'

interface AttentionHeatmapProps {
  attentionWeights: TensorData
  layerIndex: number
  headIndex: number
  tokens?: string[]
  showTextLabels?: boolean
  className?: string
  activationId?: string
}

interface HoveredCell {
  from: number
  to: number
  value: number
  x: number
  y: number
}

export const AttentionHeatmap = memo(function AttentionHeatmap({
  attentionWeights,
  layerIndex,
  headIndex,
  tokens,
  showTextLabels = false,
  className = '',
  activationId,
}: AttentionHeatmapProps) {
  const tc = useThemeColors()
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 600 })

  // Update dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = Math.min(containerRef.current.clientWidth, 800)
        setDimensions({ width: containerWidth, height: containerWidth })
      }
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!svgRef.current || !attentionWeights.values) return

    // Clear previous rendering
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const width = dimensions.width
    const height = dimensions.height
    const margin = { top: 60, right: 20, bottom: 60, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Extract attention matrix for specific head
    // Shape: [batch, heads, seq_len, seq_len]
    const values = attentionWeights.values as number[][][][]
    const matrix = values[0][headIndex] // Get specific head

    const seqLen = matrix.length

    // Create scales
    const xScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, innerWidth])
      .padding(0.05)

    const yScale = d3.scaleBand()
      .domain(d3.range(seqLen).map(String))
      .range([0, innerHeight])
      .padding(0.05)

    // The app's sequential scale: the theme's ramp on ink, grays on white
    const maxWeight = d3.max(matrix.flat()) || 1
    const scale = sequentialScale(tc.isLight)
    const colorScale = (value: number) => scale(value / maxWeight)

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('x', margin.left)
      .attr('font-size', 17)
      .attr('font-weight', 500)
      .attr('letter-spacing', '-0.02em')
      .attr('fill', tc.text)
      .text(`Layer ${layerIndex} · Head ${headIndex}`)

    // Create heatmap cells
    g.selectAll('rect')
      .data(matrix.flatMap((row, i) => row.map((value, j) => ({ i, j, value }))))
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.j)) || 0)
      .attr('y', d => yScale(String(d.i)) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', tc.border)
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        d3.select(this)
          .attr('stroke', tc.text)
          .attr('stroke-width', 1.5)

        // The pointer in the container's own pixels (the page is zoomed, see pointerIn)
        if (containerRef.current) {
          setHoveredCell({ from: d.i, to: d.j, value: d.value, ...pointerIn(event, containerRef.current) })
        }
      })
      .on('mousemove', function(event, d) {
        if (containerRef.current) {
          setHoveredCell({ from: d.i, to: d.j, value: d.value, ...pointerIn(event, containerRef.current) })
        }
      })
      .on('mouseleave', function() {
        d3.select(this)
          .attr('stroke', tc.border)
          .attr('stroke-width', 0.5)
        setHoveredCell(null)
      })

    // Add axes
    const xAxis = d3.axisBottom(xScale).tickFormat(i => {
      if (showTextLabels && tokens && tokens[parseInt(i)]) {
        return tokens[parseInt(i)].substring(0, 8)
      }
      return i
    })

    const yAxis = d3.axisLeft(yScale).tickFormat(i => {
      if (showTextLabels && tokens && tokens[parseInt(i)]) {
        return tokens[parseInt(i)].substring(0, 8)
      }
      return i
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

    g.append('g')
      .call(yAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 12)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.text)

    // Add axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 13.5)
      .attr('fill', tc.textMuted)
      .text('To token')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', 13.5)
      .attr('fill', tc.textMuted)
      .text('From token')

    // Add color legend
    const legendWidth = 200
    const legendHeight = 8
    const legendX = width - legendWidth - 20
    const legendY = 15

    const legendScale = d3.scaleLinear()
      .domain([0, d3.max(matrix.flat()) || 1])
      .range([0, legendWidth])

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => d3.format('.3f')(d as number))

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', `legend-gradient-${layerIndex}-${headIndex}`)
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
      .style('fill', `url(#legend-gradient-${layerIndex}-${headIndex})`)
      .attr('stroke', tc.borderStrong)
      .attr('stroke-width', 1)

    svg.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 11)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.textMuted)

  }, [attentionWeights, layerIndex, headIndex, tokens, showTextLabels, dimensions, tc.text, tc.textMuted, tc.border, tc.borderStrong, tc.isLight])

  // Calculate tooltip position with boundary checks
  const getTooltipStyle = () => {
    if (!hoveredCell) return {}

    const tooltipWidth = 200
    const tooltipHeight = 80
    const offset = 12

    let x = hoveredCell.x + offset
    let y = hoveredCell.y + offset

    // Keep tooltip within container bounds
    if (x + tooltipWidth > dimensions.width) {
      x = hoveredCell.x - tooltipWidth - offset
    }
    if (y + tooltipHeight > dimensions.height) {
      y = hoveredCell.y - tooltipHeight - offset
    }

    return {
      left: `${x}px`,
      top: `${y}px`
    }
  }

  return (
    <div className={`relative ${className}`} ref={containerRef}>
      {/* Export button */}
      {activationId && (
        <div className="absolute top-2 right-2 z-10">
          <ExportButton
            exportType="attention-heatmap"
            activationId={activationId}
            layer={layerIndex}
            head={headIndex}
            tokenLabels={showTextLabels ? 'text' : 'number'}
          />
        </div>
      )}
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
            <span className="font-mono">{hoveredCell.value.toFixed(4)}</span> attention
          </div>
          <div className="text-dim">
            From token {hoveredCell.from}{tokens?.[hoveredCell.from] ? ` ("${tokens[hoveredCell.from]}")` : ''} → to token {hoveredCell.to}{tokens?.[hoveredCell.to] ? ` ("${tokens[hoveredCell.to]}")` : ''}
          </div>
        </div>
      )}

    </div>
  )
})
