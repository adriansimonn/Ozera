/**
 * Transformation Flow Visualization
 * Shows a visual flow diagram of how embeddings transform through the model.
 */

import { memo, useEffect, useRef, useState } from 'react'
import type { ActivationData } from '../../types/model'
import { useAppPalette } from '../../hooks/useTheme'
import { sequentialScale } from '../../styles/appTheme'
import { MONO } from '../home/homeKit'

interface TransformationFlowProps {
  activationData: ActivationData
  selectedTokenIndex: number
  className?: string
}

export const TransformationFlow = memo(function TransformationFlow({
  activationData,
  selectedTokenIndex,
  className = ''
}: TransformationFlowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const palette = useAppPalette()
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
    ctx.fillStyle = palette.bg
    ctx.fillRect(0, 0, width, height)
    const scale = sequentialScale(palette.isLight)

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
    ctx.strokeStyle = palette.ruleStrong
    ctx.lineWidth = 1
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
      ctx.fillStyle = palette.fgFaint
      ctx.fill()
    }

    // Draw each stage
    stages.forEach((stage, _idx) => {
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
        ctx.fillStyle = scale(normalized)
        ctx.fillRect(embX + i * cellWidth, y - embHeight / 2, cellWidth, embHeight)
      })

      // Draw border
      ctx.strokeStyle = palette.ruleStrong
      ctx.lineWidth = 1
      ctx.strokeRect(embX + 0.5, y - embHeight / 2 + 0.5, embWidth - 1, embHeight - 1)

      // Draw label
      ctx.fillStyle = palette.fg
      ctx.font = `500 14px ${MONO}`
      ctx.textAlign = 'left'
      ctx.fillText(stage.name, embX - 100, y + 5)

      // Draw statistics
      const norm = Math.sqrt(stage.embedding.reduce((sum, v) => sum + v * v, 0))
      const mean = stage.embedding.reduce((sum, v) => sum + v, 0) / stage.embedding.length

      ctx.fillStyle = palette.fgDim
      ctx.font = `12px ${MONO}`
      ctx.textAlign = 'right'
      ctx.fillText(`norm: ${norm.toFixed(2)}`, embX + embWidth + 80, y - 5)
      ctx.fillText(`mean: ${mean.toFixed(3)}`, embX + embWidth + 80, y + 10)
    })

  }, [activationData, selectedTokenIndex, canvasDimensions, palette])

  const legendScale = sequentialScale(palette.isLight)
  const legendStops = [0, 0.25, 0.5, 0.75, 1].map(legendScale)

  return (
    <div className={`transformation-flow ${className}`}>
      <h3 className="tf-title">Transformation flow · Token {selectedTokenIndex}</h3>
      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} className="flow-canvas" />
      </div>
      <div className="legend">
        <div className="legend-item">
          <span className="legend-label">Low</span>
          <div className="legend-color" style={{ background: `linear-gradient(to right, ${legendStops.join(', ')})` }} />
          <span className="legend-label">High activation</span>
        </div>
      </div>

      <style>{`
        .transformation-flow {
          padding: 1.5rem var(--oz-gutter) 2rem;
        }

        .transformation-flow .tf-title {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }

        .transformation-flow .canvas-container {
          width: 100%;
        }

        .transformation-flow .flow-canvas {
          display: block;
          image-rendering: pixelated;
          width: 100%;
          height: auto;
        }

        .transformation-flow .legend {
          margin-top: 1rem;
          display: flex;
          justify-content: center;
        }

        .transformation-flow .legend-item {
          display: flex;
          align-items: center;
          gap: 0.6rem;
        }

        .transformation-flow .legend-color {
          width: 140px;
          height: 8px;
          border: 1px solid var(--oz-rule-strong);
        }

        .transformation-flow .legend-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
        }
      `}</style>
    </div>
  )
})
