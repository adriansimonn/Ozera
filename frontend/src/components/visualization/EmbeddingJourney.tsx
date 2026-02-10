/**
 * Embedding Journey Visualization
 * Shows how token embeddings transform as they pass through each layer of the model.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { ActivationData } from '../../types/model'
import { ChevronDown, ChevronsUpDown } from 'lucide-react'

interface EmbeddingJourneyProps {
  activationData: ActivationData
  selectedTokenIndex: number
  className?: string
}

export function EmbeddingJourney({
  activationData,
  selectedTokenIndex,
  className = ''
}: EmbeddingJourneyProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(new Set([0]))

  // Extract embeddings at each stage for the selected token
  const getTokenJourney = () => {
    const journey: Array<{
      stage: string
      values: number[]
      mean: number
      std: number
      norm: number
    }> = []

    // Token embeddings
    if (activationData.activations.token_embeddings) {
      const tokenEmb = activationData.activations.token_embeddings.values as number[][][]
      const values = tokenEmb[0][selectedTokenIndex]
      journey.push({
        stage: 'Token Embedding',
        values,
        mean: activationData.activations.token_embeddings.mean,
        std: activationData.activations.token_embeddings.std,
        norm: Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
      })
    }

    // Combined embeddings (token + positional)
    if (activationData.activations.combined_embeddings) {
      const combinedEmb = activationData.activations.combined_embeddings.values as number[][][]
      const values = combinedEmb[0][selectedTokenIndex]
      journey.push({
        stage: 'Combined (Token + Position)',
        values,
        mean: activationData.activations.combined_embeddings.mean,
        std: activationData.activations.combined_embeddings.std,
        norm: Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
      })
    }

    // Layer outputs
    activationData.activations.layers?.forEach((layer, idx) => {
      if (layer.post_attn) {
        const postAttn = layer.post_attn.values as number[][][]
        const values = postAttn[0][selectedTokenIndex]
        journey.push({
          stage: `Layer ${idx} - After Attention`,
          values,
          mean: layer.post_attn.mean,
          std: layer.post_attn.std,
          norm: Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
        })
      }

      if (layer.post_ff) {
        const postFF = layer.post_ff.values as number[][][]
        const values = postFF[0][selectedTokenIndex]
        journey.push({
          stage: `Layer ${idx} - After Feed-Forward`,
          values,
          mean: layer.post_ff.mean,
          std: layer.post_ff.std,
          norm: Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
        })
      }
    })

    // Final layer norm
    if (activationData.activations.final_layer_norm) {
      const finalNorm = activationData.activations.final_layer_norm.values as number[][][]
      const values = finalNorm[0][selectedTokenIndex]
      journey.push({
        stage: 'Final Layer Norm',
        values,
        mean: activationData.activations.final_layer_norm.mean,
        std: activationData.activations.final_layer_norm.std,
        norm: Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
      })
    }

    return journey
  }

  const journey = getTokenJourney()

  const toggleLayer = (index: number) => {
    setExpandedLayers(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const allExpanded = journey.length > 0 && expandedLayers.size === journey.length
  const toggleAll = () => {
    if (allExpanded) {
      setExpandedLayers(new Set())
    } else {
      setExpandedLayers(new Set(journey.map((_, i) => i)))
    }
  }

  return (
    <div className={`embedding-journey ${className}`}>
      <div className="journey-header">
        <div className="journey-header-top">
          <div>
            <h3 className="journey-title">
              Embedding Journey - Token {selectedTokenIndex}
            </h3>
            <p className="journey-subtitle">
              Watch how the token's vector representation transforms through each layer
            </p>
          </div>
          <button className="expand-collapse-btn" onClick={toggleAll}>
            <ChevronsUpDown className="expand-collapse-icon" />
            {allExpanded ? 'Collapse All' : 'Expand All'}
          </button>
        </div>
      </div>

      <div className="journey-flow">
        {journey.map((step, idx) => (
          <div key={idx} className="journey-stage">
            <div
              className="stage-header"
              onClick={() => toggleLayer(idx)}
            >
              <div className="stage-info">
                <div className="stage-title">{step.stage}</div>
                <div className="stage-stats">
                  <span className="stat">
                    Norm: <span className="stat-value">{step.norm.toFixed(3)}</span>
                  </span>
                  <span className="stat">
                    Mean: <span className="stat-value">{step.mean.toFixed(3)}</span>
                  </span>
                  <span className="stat">
                    Std: <span className="stat-value">{step.std.toFixed(3)}</span>
                  </span>
                </div>
              </div>
              <ChevronDown
                className={`expand-icon ${expandedLayers.has(idx) ? 'expanded' : ''}`}
              />
            </div>

            {expandedLayers.has(idx) && (
              <div className="stage-visualization">
                <EmbeddingVectorDisplay
                  values={step.values}
                  stage={step.stage}
                />
              </div>
            )}

            {idx < journey.length - 1 && (
              <div className="stage-arrow">
                <div className="arrow-line" />
                <div className="arrow-head">▼</div>
              </div>
            )}
          </div>
        ))}
      </div>

      <style>{`
        .embedding-journey {
          padding: 1.5rem;
        }

        .journey-header {
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .journey-header-top {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1rem;
        }

        .journey-title {
          font-size: 1rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
          margin: 0;
        }

        .journey-subtitle {
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.4);
          margin: 0.375rem 0 0 0;
        }

        .expand-collapse-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.5rem 0.75rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .expand-collapse-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.8);
        }

        .expand-collapse-icon {
          width: 14px;
          height: 14px;
        }

        .journey-flow {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .journey-stage {
          position: relative;
        }

        .stage-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.875rem 1rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.08);
          cursor: pointer;
          transition: all 0.2s;
        }

        .stage-header:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .stage-info {
          flex: 1;
        }

        .stage-title {
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
          font-size: 0.875rem;
          margin-bottom: 0.375rem;
        }

        .stage-stats {
          display: flex;
          gap: 1.5rem;
        }

        .stat {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .stat-value {
          font-family: monospace;
          color: rgba(255, 255, 255, 0.8);
          font-weight: 600;
        }

        .expand-icon {
          width: 18px;
          height: 18px;
          color: rgba(255, 255, 255, 0.3);
          transition: transform 0.2s;
        }

        .expand-icon.expanded {
          transform: rotate(180deg);
        }

        .stage-visualization {
          margin-top: 0.5rem;
          padding: 1rem;
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.06);
          animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .stage-arrow {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 0.375rem 0;
        }

        .arrow-line {
          width: 1px;
          height: 16px;
          background: rgba(255, 255, 255, 0.15);
        }

        .arrow-head {
          color: rgba(255, 255, 255, 0.25);
          font-size: 0.75rem;
          line-height: 1;
        }
      `}</style>
    </div>
  )
}

interface EmbeddingVectorDisplayProps {
  values: number[]
  stage: string
}

function EmbeddingVectorDisplay({ values, stage }: EmbeddingVectorDisplayProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 100 })

  // Update dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth
        setDimensions({ width: containerWidth, height: 100 })
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
    if (!svgRef.current) return

    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const width = dimensions.width
    const height = dimensions.height
    const margin = { top: 10, right: 10, bottom: 30, left: 10 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Limit to first 128 dimensions for visualization
    const displayValues = values.slice(0, 128)

    const xScale = d3.scaleBand()
      .domain(displayValues.map((_, i) => i.toString()))
      .range([0, innerWidth])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([d3.min(displayValues) || -1, d3.max(displayValues) || 1])
      .range([innerHeight, 0])

    const colorScale = d3.scaleSequential()
      .domain([d3.min(displayValues) || -1, d3.max(displayValues) || 1])
      .interpolator(t => d3.interpolateRgb('#6366f1', '#22d3ee')(t))

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Draw bars
    g.selectAll('rect')
      .data(displayValues)
      .enter()
      .append('rect')
      .attr('x', (_, i) => xScale(i.toString()) || 0)
      .attr('y', d => d >= 0 ? yScale(d) : yScale(0))
      .attr('width', xScale.bandwidth())
      .attr('height', d => Math.abs(yScale(d) - yScale(0)))
      .attr('fill', d => colorScale(d))
      .attr('opacity', 0.8)

    // Add zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', 'rgba(255, 255, 255, 0.15)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2')

    // Add dimension count label
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs')
      .attr('fill', 'rgba(255, 255, 255, 0.4)')
      .text(`Showing ${displayValues.length} / ${values.length} dimensions`)

  }, [values, stage, dimensions])

  return (
    <div className="vector-display" ref={containerRef}>
      <svg
        ref={svgRef}
        width="100%"
        height={100}
        style={{ background: 'rgba(0, 0, 0, 0.3)' }}
      />
      <style>{`
        .vector-display {
          display: flex;
          justify-content: center;
        }
      `}</style>
    </div>
  )
}
