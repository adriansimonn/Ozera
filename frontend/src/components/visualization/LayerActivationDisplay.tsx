/**
 * Layer activation display component.
 * Shows activation statistics and distributions for a specific layer.
 */

import { memo, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { LayerActivations } from '../../types/model'
import { ExportButton } from '../export'
import { useThemeColors } from '../../hooks/useTheme'
import { MONO } from '../home/homeKit'

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
  const tc = useThemeColors()
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
        'Attention output',
        dimensions.width,
        dimensions.height
      )
    }

    if (layerActivations.ff_output && ffChartRef.current) {
      renderDistribution(
        ffChartRef.current,
        layerActivations.ff_output.values as number[][][],
        'Feed-forward output',
        dimensions.width,
        dimensions.height
      )
    }
  }, [layerActivations, dimensions, tc.text, tc.textMuted, tc.borderStrong])

  function renderDistribution(
    svgElement: SVGSVGElement,
    values: number[][][],
    title: string,
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
      .attr('x', margin.left)
      .attr('y', 20)
      .attr('font-size', 15)
      .attr('font-weight', 500)
      .attr('letter-spacing', '-0.015em')
      .attr('fill', tc.text)
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
      .attr('fill', tc.text)

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d => d3.format('.2f')(d as number))
    const yAxis = d3.axisLeft(yScale).ticks(5)

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 11.5)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.textMuted)

    g.append('g')
      .call(yAxis)
      .attr('color', tc.borderStrong)
      .attr('font-size', 11.5)
      .attr('font-family', MONO)
      .selectAll('text')
      .attr('fill', tc.textMuted)

    // Axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12.5)
      .attr('fill', tc.textMuted)
      .text('Activation value')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12.5)
      .attr('fill', tc.textMuted)
      .text('Frequency')
  }

  return (
    <div className={`layer-activations ${className}`} ref={containerRef}>
      <div className="la-head">
        <h3>Layer {layerIndex} activations</h3>
        {activationId && (
          <ExportButton
            exportType="activation-histogram"
            activationId={activationId}
            layer={layerIndex}
          />
        )}
      </div>

      {/* Headline numbers, each under a rule */}
      <div className="la-stats">
        {layerActivations.attn_output && <StatCard label="Attention mean" value={layerActivations.attn_output.mean} />}
        {layerActivations.attn_output && <StatCard label="Attention std" value={layerActivations.attn_output.std} />}
        {layerActivations.ff_output && <StatCard label="Feed-forward mean" value={layerActivations.ff_output.mean} />}
        {layerActivations.ff_output && <StatCard label="Feed-forward std" value={layerActivations.ff_output.std} />}
        {layerActivations.attn_weights && (
          <div className="la-stat">
            <div className="la-stat-label">Attention weights shape</div>
            <div className="la-stat-value la-stat-value--small">{layerActivations.attn_weights.shape.join(' × ')}</div>
          </div>
        )}
      </div>

      {/* Distribution charts */}
      <div className="la-charts">
        {layerActivations.attn_output && <svg ref={attnChartRef} width="100%" height={dimensions.height} />}
        {layerActivations.ff_output && <svg ref={ffChartRef} width="100%" height={dimensions.height} />}
      </div>

      <style>{`
        .layer-activations { padding: 1.5rem var(--oz-gutter) 2rem; }
        .layer-activations .la-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem; }
        .layer-activations .la-head h3 { margin: 0; font-size: 1.5rem; font-weight: 500; letter-spacing: -0.03em; line-height: 1.15; }
        .layer-activations .la-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 1.25rem 1.5rem; margin-bottom: 2.25rem; }
        .layer-activations .la-stat { padding-top: 0.65rem; border-top: 1px solid var(--oz-rule-strong); }
        .layer-activations .la-stat-label { color: var(--oz-fg-dim); font-size: 0.8rem; font-weight: 500; }
        .layer-activations .la-stat-value { margin-top: 0.2rem; font-family: var(--oz-mono); font-size: 1.35rem; letter-spacing: -0.02em; line-height: 1.2; }
        .layer-activations .la-stat-value--small { font-size: 1rem; line-height: 1.62; }
        .layer-activations .la-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 2rem; }
      `}</style>
    </div>
  )
})

interface StatCardProps {
  label: string
  value: number
}

function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="la-stat">
      <div className="la-stat-label">{label}</div>
      <div className="la-stat-value">{value.toFixed(4)}</div>
    </div>
  )
}
