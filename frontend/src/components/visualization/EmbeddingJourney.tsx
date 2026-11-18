/**
 * Embedding Journey Visualization
 * Shows how token embeddings transform as they pass through each layer of the model.
 */

import { memo, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { ActivationData } from '../../types/model'
import { useThemeColors } from '../../hooks/useTheme'

interface EmbeddingJourneyProps {
  activationData: ActivationData
  selectedTokenIndex: number
  className?: string
}

export const EmbeddingJourney = memo(function EmbeddingJourney({
  activationData,
  selectedTokenIndex,
  className = ''
}: EmbeddingJourneyProps) {
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
            <h3 className="journey-title">Embedding journey · Token {selectedTokenIndex}</h3>
          </div>
          <button className="oz-btn oz-btn--ghost oz-btn--small" onClick={toggleAll}>
            {allExpanded ? 'Collapse all' : 'Expand all'}
          </button>
        </div>
      </div>

      <div className="journey-flow">
        {journey.map((step, idx) => (
          <div key={idx} className={`journey-stage${expandedLayers.has(idx) ? ' is-open' : ''}`}>
            <button
              type="button"
              className="stage-header"
              aria-expanded={expandedLayers.has(idx)}
              onClick={() => toggleLayer(idx)}
            >
              <span className="stage-num">{String(idx + 1).padStart(2, '0')}</span>
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
              <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
                <path d="M1 1l4 4 4-4" />
              </svg>
            </button>

            {expandedLayers.has(idx) && (
              <div className="stage-visualization">
                <EmbeddingVectorDisplay
                  values={step.values}
                  stage={step.stage}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      <style>{`
        .embedding-journey {
          padding: 1.5rem var(--oz-gutter) 2rem;
        }

        .embedding-journey .journey-header-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .embedding-journey .journey-title {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        /* The stages: a numbered, ruled list; an open one shows its vector */
        .embedding-journey .journey-stage {
          position: relative;
          border-top: 1px solid var(--oz-rule);
        }

        .embedding-journey .journey-stage:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }

        .embedding-journey .journey-stage.is-open::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 100%;
          height: 1px;
          background: var(--oz-mark);
        }

        .embedding-journey .stage-header {
          display: grid;
          grid-template-columns: 2.75rem minmax(0, 1fr) auto;
          align-items: baseline;
          width: 100%;
          padding: 0.95rem 0;
          color: var(--oz-fg-dim);
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }

        .embedding-journey .stage-header:hover,
        .embedding-journey .journey-stage.is-open .stage-header {
          color: var(--oz-fg);
        }

        .embedding-journey .stage-num {
          color: var(--oz-fg-faint);
          font-size: 0.8rem;
          font-weight: 500;
          font-variant-numeric: tabular-nums;
          letter-spacing: 0.02em;
          transition: color 0.2s ease;
        }

        .embedding-journey .journey-stage.is-open .stage-num {
          color: var(--oz-mark);
        }

        .embedding-journey .stage-info {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          justify-content: space-between;
          gap: 0.25rem 1.5rem;
          padding-right: 1.5rem;
        }

        .embedding-journey .stage-title {
          font-size: 1.125rem;
          font-weight: 500;
          letter-spacing: -0.02em;
        }

        .embedding-journey .stage-stats {
          display: flex;
          gap: 1.5rem;
        }

        .embedding-journey .stat {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
        }

        .embedding-journey .stat-value {
          margin-left: 0.25rem;
          color: var(--oz-fg);
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }

        .embedding-journey .stage-visualization {
          padding: 0.25rem 0 1.25rem 2.75rem;
          animation: slideDown 0.3s cubic-bezier(0.16, 0.8, 0.2, 1);
        }

        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-6px); }
        }
      `}</style>
    </div>
  )
})

interface EmbeddingVectorDisplayProps {
  values: number[]
  stage: string
}

function EmbeddingVectorDisplay({ values, stage }: EmbeddingVectorDisplayProps) {
  const tc = useThemeColors()
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

    const displayValues = values

    const xScale = d3.scaleBand()
      .domain(displayValues.map((_, i) => i.toString()))
      .range([0, innerWidth])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([d3.min(displayValues) || -1, d3.max(displayValues) || 1])
      .range([innerHeight, 0])

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
      .attr('fill', d => (d >= 0 ? tc.text : tc.textMuted))

    // Add zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', tc.borderStrong)
      .attr('stroke-width', 1)

    // Add dimension count label
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', tc.textMuted)
      .text(`${displayValues.length} of ${values.length} dimensions`)

  }, [values, stage, dimensions, tc.text, tc.textMuted, tc.borderStrong])

  return (
    <div className="vector-display" ref={containerRef}>
      <svg
        ref={svgRef}
        width="100%"
        height={100}
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
