/**
 * Transformation Flow Visualization
 * Shows a visual flow diagram of how embeddings transform through the model.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { ActivationData } from '../../types/model'

interface TransformationFlowProps {
  activationData: ActivationData
  selectedTokenIndex: number
  className?: string
}

export function TransformationFlow({
  activationData,
  selectedTokenIndex,
  className = ''
}: TransformationFlowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1000, height: 600 })

  // Compute required height based on number of stages
  const computeRequiredHeight = () => {
    const layerSpacing = 80
    let stageCount = 0
    if (activationData.activations.token_embeddings) stageCount++
    if (activationData.activations.combined_embeddings) stageCount++
    activationData.activations.layers?.forEach((layer) => {
      if (layer.post_attn) stageCount++
      if (layer.post_ff) stageCount++
    })
    // Last stage y = 50 + layerSpacing * (stageCount - 1), plus padding for bottom
    return 50 + layerSpacing * Math.max(0, stageCount - 1) + 50
  }

  // Update canvas dimensions based on container size and content
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth
        const requiredHeight = computeRequiredHeight()
        setCanvasDimensions({ width: containerWidth, height: Math.max(400, requiredHeight) })
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
  }, [activationData, selectedTokenIndex])

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const width = canvasDimensions.width
    const height = canvasDimensions.height
    canvas.width = width
    canvas.height = height

    // Clear canvas
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, width, height)

    // Extract embeddings at each stage
    const stages: Array<{
      name: string
      embedding: number[]
      y: number
    }> = []

    const layerSpacing = 80
    let stageIdx = 0

    // Token embedding
    if (activationData.activations.token_embeddings) {
      const tokenEmb = activationData.activations.token_embeddings.values as number[][][]
      stages.push({
        name: 'Token Emb',
        embedding: tokenEmb[0][selectedTokenIndex],
        y: 50 + layerSpacing * stageIdx
      })
      stageIdx++
    }

    // Combined embedding
    if (activationData.activations.combined_embeddings) {
      const combinedEmb = activationData.activations.combined_embeddings.values as number[][][]
      stages.push({
        name: 'Combined',
        embedding: combinedEmb[0][selectedTokenIndex],
        y: 50 + layerSpacing * stageIdx
      })
      stageIdx++
    }

    // Layer transformations
    activationData.activations.layers?.forEach((layer, idx) => {
      if (layer.post_attn) {
        const postAttn = layer.post_attn.values as number[][][]
        stages.push({
          name: `L${idx} Attn`,
          embedding: postAttn[0][selectedTokenIndex],
          y: 50 + layerSpacing * stageIdx
        })
        stageIdx++
      }

      if (layer.post_ff) {
        const postFF = layer.post_ff.values as number[][][]
        stages.push({
          name: `L${idx} FF`,
          embedding: postFF[0][selectedTokenIndex],
          y: 50 + layerSpacing * stageIdx
        })
        stageIdx++
      }
    })

    // Draw connections
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'
    ctx.lineWidth = 2
    for (let i = 0; i < stages.length - 1; i++) {
      ctx.beginPath()
      ctx.moveTo(width / 2, stages[i].y + 20)
      ctx.lineTo(width / 2, stages[i + 1].y - 20)
      ctx.stroke()

      // Draw arrow
      const arrowY = (stages[i].y + stages[i + 1].y) / 2
      ctx.beginPath()
      ctx.moveTo(width / 2, arrowY)
      ctx.lineTo(width / 2 - 5, arrowY - 8)
      ctx.lineTo(width / 2 + 5, arrowY - 8)
      ctx.closePath()
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.fill()
    }

    // Draw each stage
    stages.forEach((stage, idx) => {
      const x = width / 2
      const y = stage.y

      // Draw embedding visualization (compact heatmap)
      const embWidth = 400
      const embHeight = 30
      const embX = x - embWidth / 2
      const cellWidth = embWidth / Math.min(stage.embedding.length, 200)

      // Use first 200 dimensions
      const displayEmb = stage.embedding.slice(0, 200)
      const min = Math.min(...displayEmb)
      const max = Math.max(...displayEmb)

      displayEmb.forEach((value, i) => {
        const normalized = (value - min) / (max - min || 1)
        const color = d3.interpolateViridis(normalized)
        ctx.fillStyle = color
        ctx.fillRect(embX + i * cellWidth, y - embHeight / 2, cellWidth, embHeight)
      })

      // Draw border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'
      ctx.lineWidth = 1
      ctx.strokeRect(embX, y - embHeight / 2, embWidth, embHeight)

      // Draw label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.font = 'bold 14px Monaco, monospace'
      ctx.textAlign = 'left'
      ctx.fillText(stage.name, embX - 100, y + 5)

      // Draw statistics
      const norm = Math.sqrt(stage.embedding.reduce((sum, v) => sum + v * v, 0))
      const mean = stage.embedding.reduce((sum, v) => sum + v, 0) / stage.embedding.length

      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'
      ctx.font = '11px Monaco, monospace'
      ctx.textAlign = 'right'
      ctx.fillText(`norm: ${norm.toFixed(2)}`, embX + embWidth + 80, y - 5)
      ctx.fillText(`mean: ${mean.toFixed(3)}`, embX + embWidth + 80, y + 10)
    })

    // Draw title
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.font = 'bold 16px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(`Token ${selectedTokenIndex} Transformation Flow`, width / 2, 25)

  }, [activationData, selectedTokenIndex, canvasDimensions])

  return (
    <div className={`transformation-flow ${className}`}>
      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} className="flow-canvas" />
      </div>
      <div className="legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde724)' }} />
          <span className="legend-label">Activation Value (Low → High)</span>
        </div>
      </div>

      <style>{`
        .transformation-flow {
          padding: 1.5rem;
        }

        .canvas-container {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.06);
          width: 100%;
        }

        .flow-canvas {
          display: block;
          image-rendering: pixelated;
          width: 100%;
          height: auto;
        }

        .legend {
          margin-top: 1rem;
          display: flex;
          justify-content: center;
          gap: 2rem;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .legend-color {
          width: 100px;
          height: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .legend-label {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.4);
        }

        /* Light mode */
        [data-bg="light"] .canvas-container {
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .legend-color {
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .legend-label {
          color: rgba(0, 0, 0, 0.4);
        }
      `}</style>
    </div>
  )
}
