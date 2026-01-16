/**
 * Attention heatmap visualization component.
 * Displays attention weights as an interactive heatmap with modern dark styling.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { TensorData } from '../../types/model'

interface AttentionHeatmapProps {
  attentionWeights: TensorData
  layerIndex: number
  headIndex: number
  tokens?: string[]
  className?: string
}

export function AttentionHeatmap({
  attentionWeights,
  layerIndex,
  headIndex,
  tokens,
  className = ''
}: AttentionHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredCell, setHoveredCell] = useState<{ from: number; to: number; value: number } | null>(null)

  useEffect(() => {
    if (!svgRef.current || !attentionWeights.values) return

    // Clear previous rendering
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const width = 600
    const height = 600
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

    // Color scale - cyan to purple gradient for dark theme
    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(matrix.flat()) || 1])
      .interpolator(d3.interpolateInferno)

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-semibold')
      .attr('fill', '#e2e8f0')
      .text(`Layer ${layerIndex} - Head ${headIndex}`)

    // Create heatmap cells
    const cells = g.selectAll('rect')
      .data(matrix.flatMap((row, i) => row.map((value, j) => ({ i, j, value }))))
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.j)) || 0)
      .attr('y', d => yScale(String(d.i)) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 0.5)
      .attr('rx', 1)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        d3.select(this)
          .attr('stroke', '#60a5fa')
          .attr('stroke-width', 2)
        setHoveredCell({ from: d.i, to: d.j, value: d.value })
      })
      .on('mouseleave', function() {
        d3.select(this)
          .attr('stroke', '#1e293b')
          .attr('stroke-width', 0.5)
        setHoveredCell(null)
      })

    // Add axes
    const xAxis = d3.axisBottom(xScale).tickFormat(i => {
      if (tokens && tokens[parseInt(i)]) {
        return tokens[parseInt(i)].substring(0, 8)
      }
      return i
    })

    const yAxis = d3.axisLeft(yScale).tickFormat(i => {
      if (tokens && tokens[parseInt(i)]) {
        return tokens[parseInt(i)].substring(0, 8)
      }
      return i
    })

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')

    g.append('g')
      .call(yAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#94a3b8')

    // Add axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm')
      .attr('fill', '#cbd5e1')
      .text('To Token')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm')
      .attr('fill', '#cbd5e1')
      .text('From Token')

    // Add color legend
    const legendWidth = 200
    const legendHeight = 15
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
      .attr('stop-color', d => colorScale(d * (d3.max(matrix.flat()) || 1)))

    svg.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#legend-gradient-${layerIndex}-${headIndex})`)
      .attr('stroke', '#475569')
      .attr('stroke-width', 1)

    svg.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#94a3b8')

  }, [attentionWeights, layerIndex, headIndex, tokens])

  return (
    <div className={`relative ${className}`}>
      <svg
        ref={svgRef}
        width={600}
        height={600}
        className="bg-slate-900/50 rounded-lg border border-slate-700/50"
      />
      {hoveredCell && (
        <div className="absolute top-4 left-4 bg-slate-800/95 border border-cyan-500/30 rounded-lg px-4 py-2 text-sm backdrop-blur-sm">
          <div className="text-cyan-400 font-semibold mb-1">Attention Weight</div>
          <div className="text-slate-300">
            From token {hoveredCell.from} → To token {hoveredCell.to}
          </div>
          <div className="text-slate-300">
            Weight: <span className="text-cyan-400 font-mono">{hoveredCell.value.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  )
}
