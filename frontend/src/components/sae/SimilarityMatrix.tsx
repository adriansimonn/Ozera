/**
 * Layer-by-layer CKA similarity matrix visualization.
 * Renders an interactive D3 heatmap showing representation similarity
 * across all layer pairs between two models (or the same model).
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Grid3x3, Info } from 'lucide-react'

interface SimilarityMatrixProps {
  ckaMatrix: number[][]
  layersA: number[]
  layersB: number[]
  modelA: string
  modelB: string
  activationTypeA: string
  activationTypeB: string
  className?: string
}

export function SimilarityMatrix({
  ckaMatrix,
  layersA,
  layersB,
  modelA,
  modelB,
  activationTypeA,
  activationTypeB,
  className = '',
}: SimilarityMatrixProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredCell, setHoveredCell] = useState<{
    layerA: number
    layerB: number
    value: number
  } | null>(null)

  useEffect(() => {
    if (!svgRef.current || !ckaMatrix.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const nRows = ckaMatrix.length
    const nCols = ckaMatrix[0]?.length || 0
    if (nRows === 0 || nCols === 0) return

    const cellSize = Math.min(40, 500 / Math.max(nRows, nCols))
    const margin = { top: 60, right: 30, bottom: 30, left: 70 }
    const width = margin.left + nCols * cellSize + margin.right
    const height = margin.top + nRows * cellSize + margin.bottom

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Color scale: dark to gray
    const colorScale = d3
      .scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateGreys)

    // Draw cells
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < nCols; j++) {
        const value = ckaMatrix[i][j]

        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', colorScale(value))
          .attr('stroke', 'rgba(255,255,255,0.05)')
          .attr('stroke-width', 0.5)
          .style('cursor', 'pointer')
          .on('mouseenter', function () {
            d3.select(this).attr('stroke', 'rgba(255,255,255,0.7)').attr('stroke-width', 2)
            setHoveredCell({ layerA: layersA[i], layerB: layersB[j], value })
          })
          .on('mouseleave', function () {
            d3.select(this)
              .attr('stroke', 'rgba(255,255,255,0.05)')
              .attr('stroke-width', 0.5)
            setHoveredCell(null)
          })

        // Show value text for larger cells
        if (cellSize >= 30) {
          g.append('text')
            .attr('x', j * cellSize + cellSize / 2)
            .attr('y', i * cellSize + cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', value > 0.5 ? '#fff' : '#9ca3af')
            .attr('font-size', '9px')
            .attr('font-family', 'monospace')
            .attr('pointer-events', 'none')
            .text(value.toFixed(2))
        }
      }
    }

    // Y-axis labels (Model A layers)
    layersA.forEach((layer, i) => {
      g.append('text')
        .attr('x', -8)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text(`L${layer}`)
    })

    // X-axis labels (Model B layers)
    layersB.forEach((layer, j) => {
      g.append('text')
        .attr('x', j * cellSize + cellSize / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('fill', '#9ca3af')
        .attr('font-size', '10px')
        .attr('font-family', 'monospace')
        .text(`L${layer}`)
    })

    // Y-axis title
    svg
      .append('text')
      .attr('x', 14)
      .attr('y', margin.top + (nRows * cellSize) / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'rgba(255,255,255,0.7)')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .attr('transform', `rotate(-90, 14, ${margin.top + (nRows * cellSize) / 2})`)
      .text(`${modelA} (${activationTypeA})`)

    // X-axis title
    svg
      .append('text')
      .attr('x', margin.left + (nCols * cellSize) / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.7)')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .text(`${modelB} (${activationTypeB})`)

    // Color legend
    const legendWidth = 120
    const legendHeight = 10
    const legendX = margin.left + nCols * cellSize - legendWidth
    const legendY = margin.top + nRows * cellSize + 16

    const defs = svg.append('defs')
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'cka-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')

    gradient
      .selectAll('stop')
      .data(d3.range(0, 1.01, 0.1))
      .enter()
      .append('stop')
      .attr('offset', (d) => `${d * 100}%`)
      .attr('stop-color', (d) => colorScale(d))

    svg
      .append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#cka-gradient)')

    svg
      .append('text')
      .attr('x', legendX)
      .attr('y', legendY + legendHeight + 12)
      .attr('fill', '#6b7280')
      .attr('font-size', '8px')
      .text('0.0')

    svg
      .append('text')
      .attr('x', legendX + legendWidth)
      .attr('y', legendY + legendHeight + 12)
      .attr('text-anchor', 'end')
      .attr('fill', '#6b7280')
      .attr('font-size', '8px')
      .text('1.0')

    svg
      .append('text')
      .attr('x', legendX + legendWidth / 2)
      .attr('y', legendY - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6b7280')
      .attr('font-size', '8px')
      .text('CKA Similarity')
  }, [ckaMatrix, layersA, layersB, modelA, modelB, activationTypeA, activationTypeB])

  const isSameModel = modelA === modelB && activationTypeA === activationTypeB

  return (
    <div className={`bg-black/40 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-1">
          <Grid3x3 className="w-5 h-5" style={{ color: 'rgba(255,255,255,0.6)' }} />
          <h3 className="text-lg font-semibold text-gray-200 tracking-tight">
            Layer Similarity Matrix
          </h3>
        </div>
        <p className="text-xs text-gray-500">
          {isSameModel
            ? `CKA similarity across layers of ${modelA}`
            : `CKA similarity: ${modelA} vs ${modelB}`}
        </p>
      </div>

      {/* Matrix */}
      <div className="p-4 overflow-x-auto">
        <svg ref={svgRef} />
      </div>

      {/* Hover info */}
      {hoveredCell && (
        <div className="px-4 pb-3 flex items-center gap-2 text-sm">
          <Info className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-gray-400">
            <span style={{ color: 'rgba(255,255,255,0.8)' }}>{modelA} L{hoveredCell.layerA}</span>
            {' vs '}
            <span style={{ color: 'rgba(255,255,255,0.8)' }}>{modelB} L{hoveredCell.layerB}</span>
            {': '}
            <span className="text-white font-mono">{hoveredCell.value.toFixed(4)}</span>
          </span>
        </div>
      )}

      {/* Interpretation guide */}
      <div className="px-4 pb-4">
        <div className="text-xs text-gray-600 space-y-1">
          <p>
            {isSameModel
              ? 'Diagonal values show self-similarity (always 1.0). Off-diagonal shows how similar representations are across layers.'
              : 'Higher values indicate more similar feature representations between the two models at those layers.'}
          </p>
          <p>
            Strong diagonal pattern suggests layer-aligned representations.
            Off-diagonal matches indicate shared concepts at different depths.
          </p>
        </div>
      </div>
    </div>
  )
}
