/**
 * Layer activation display component.
 * Shows activation statistics and distributions for a specific layer.
 */

import { memo, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { LayerActivations } from '../../types/model'
import { ExportButton } from '../export'

interface LayerActivationDisplayProps {
  layerActivations: LayerActivations
  layerIndex: number
  className?: string
  activationId?: string
}

export const LayerActivationDisplay = memo(function LayerActivationDisplay({
  layerActivations,
  layerIndex,
  className = '',
  activationId,
}: LayerActivationDisplayProps) {
  const attnChartRef = useRef<SVGSVGElement>(null)
  const ffChartRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 400, height: 200 })

  // Update dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth
        const chartWidth = Math.max(400, containerWidth / 2 - 30)
        setDimensions({ width: chartWidth, height: 200 })
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
    if (layerActivations.attn_output && attnChartRef.current) {
      renderDistribution(
        attnChartRef.current,
        layerActivations.attn_output.values as number[][][],
        'Attention Output',
        '#9ca3af', // gray-400
        dimensions.width,
        dimensions.height
      )
    }

    if (layerActivations.ff_output && ffChartRef.current) {
      renderDistribution(
        ffChartRef.current,
        layerActivations.ff_output.values as number[][][],
        'Feed-Forward Output',
        '#d8b4fe', // purple-300
        dimensions.width,
        dimensions.height
      )
    }
  }, [layerActivations, dimensions])

  function renderDistribution(
    svgElement: SVGSVGElement,
    values: number[][][],
    title: string,
    color: string,
    width: number,
    height: number
  ) {
    d3.select(svgElement).selectAll('*').remove()

    const svg = d3.select(svgElement)
    const margin = { top: 40, right: 20, bottom: 40, left: 50 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Flatten all values
    const flatValues = values.flat(2)

    // Create histogram
    const histogram = d3.bin()
      .domain([d3.min(flatValues) || -1, d3.max(flatValues) || 1])
      .thresholds(40)

    const bins = histogram(flatValues)

    // Scales
    const xScale = d3.scaleLinear()
      .domain([bins[0].x0 || 0, bins[bins.length - 1].x1 || 1])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length) || 0])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-sm font-semibold')
      .attr('fill', '#e5e7eb')
      .text(title)

    // Draw bars
    g.selectAll('rect')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.x0 || 0))
      .attr('y', d => yScale(d.length))
      .attr('width', d => Math.max(0, xScale(d.x1 || 0) - xScale(d.x0 || 0) - 1))
      .attr('height', d => innerHeight - yScale(d.length))
      .attr('fill', color)
      .attr('opacity', 0.7)

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d => d3.format('.2f')(d as number))
    const yAxis = d3.axisLeft(yScale).ticks(5)

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#9ca3af')

    g.append('g')
      .call(yAxis)
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('fill', '#9ca3af')

    // Axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs')
      .attr('fill', '#d1d5db')
      .text('Activation Value')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs')
      .attr('fill', '#d1d5db')
      .text('Frequency')
  }

  return (
    <div className={`bg-black/40 border border-gray-800 p-6 ${className}`} ref={containerRef}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-200 tracking-tight">Layer {layerIndex} Activations</h3>
        {activationId && (
          <ExportButton
            exportType="activation-histogram"
            activationId={activationId}
            layer={layerIndex}
          />
        )}
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {layerActivations.attn_output && (
          <StatCard
            label="Attn Mean"
            value={layerActivations.attn_output.mean}
            color="text-gray-300"
          />
        )}
        {layerActivations.attn_output && (
          <StatCard
            label="Attn Std"
            value={layerActivations.attn_output.std}
            color="text-gray-300"
          />
        )}
        {layerActivations.ff_output && (
          <StatCard
            label="FF Mean"
            value={layerActivations.ff_output.mean}
            color="text-purple-300"
          />
        )}
        {layerActivations.ff_output && (
          <StatCard
            label="FF Std"
            value={layerActivations.ff_output.std}
            color="text-purple-300"
          />
        )}
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {layerActivations.attn_output && (
          <svg
            ref={attnChartRef}
            width="100%"
            height={dimensions.height}
            className="bg-black/30 border border-gray-800"
          />
        )}
        {layerActivations.ff_output && (
          <svg
            ref={ffChartRef}
            width="100%"
            height={dimensions.height}
            className="bg-black/30 border border-gray-800"
          />
        )}
      </div>

      {/* Attention Weights Info */}
      {layerActivations.attn_weights && (
        <div className="mt-4 p-3 bg-black/30 border border-gray-800">
          <div className="text-sm text-gray-300">
            <span className="text-gray-500 uppercase tracking-wide text-xs">Attention Weights Shape:</span>{' '}
            <span className="font-mono text-white">
              {layerActivations.attn_weights.shape.join(' × ')}
            </span>
          </div>
        </div>
      )}
    </div>
  )
})

interface StatCardProps {
  label: string
  value: number
  color: string
}

function StatCard({ label, value, color }: StatCardProps) {
  return (
    <div className="bg-black/30 border border-gray-800 p-3">
      <div className="text-xs text-gray-500 mb-1 uppercase tracking-wide">{label}</div>
      <div className={`text-lg font-mono font-semibold ${color}`}>
        {value.toFixed(4)}
      </div>
    </div>
  )
}
