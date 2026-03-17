/**
 * Network Flow Visualization
 * Animated neural network diagram showing data flow through the transformer.
 */

import { useEffect, useRef, useState } from 'react'
import type { ActivationData } from '../../types/model'

interface NetworkFlowProps {
  activationData: ActivationData
  selectedTokenIndex?: number
  className?: string
}

interface Node {
  x: number
  y: number
  layer: number
  index: number
  value: number
  label?: string
}

interface Connection {
  from: Node
  to: Node
  strength: number
}

interface Particle {
  x: number
  y: number
  targetX: number
  targetY: number
  progress: number
  connection: Connection
}

export function NetworkFlow({
  activationData,
  selectedTokenIndex = 0,
  className = ''
}: NetworkFlowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()
  const particlesRef = useRef<Particle[]>([])
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState(1)
  const [showAllTokens, setShowAllTokens] = useState(false)
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1200, height: 800 })

  // Update canvas dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth
        const aspectRatio = 800 / 1200
        const newHeight = Math.max(600, containerWidth * aspectRatio)
        setCanvasDimensions({ width: containerWidth, height: newHeight })
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
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const width = canvasDimensions.width
    const height = canvasDimensions.height
    canvas.width = width
    canvas.height = height

    // Calculate network structure
    const numLayers = (activationData.activations.layers?.length || 0) + 3 // +3 for input, embedding, output
    const tokensToShow = showAllTokens ? activationData.tokens.length : Math.min(8, activationData.tokens.length)

    // Create nodes for each layer
    const layers: Node[][] = []
    const layerSpacing = (width - 200) / (numLayers - 1)

    // Input layer (tokens)
    const inputLayer: Node[] = []
    for (let i = 0; i < tokensToShow; i++) {
      const y = (height / (tokensToShow + 1)) * (i + 1)
      inputLayer.push({
        x: 100,
        y,
        layer: 0,
        index: i,
        value: 1,
        label: `Token ${i}`
      })
    }
    layers.push(inputLayer)

    // Embedding layer
    const embeddingLayer: Node[] = []
    const embNodesPerToken = 4 // Visual representation of embedding dimensions
    for (let i = 0; i < tokensToShow; i++) {
      for (let j = 0; j < embNodesPerToken; j++) {
        const baseY = (height / (tokensToShow + 1)) * (i + 1)
        const offset = (j - embNodesPerToken / 2) * 15
        embeddingLayer.push({
          x: 100 + layerSpacing,
          y: baseY + offset,
          layer: 1,
          index: i * embNodesPerToken + j,
          value: activationData.activations.token_embeddings
            ? Math.abs((activationData.activations.token_embeddings.values as number[][][])[0][i][j % 128] || 0)
            : Math.random()
        })
      }
    }
    layers.push(embeddingLayer)

    // Transformer layers
    const nodesPerLayer = Math.min(16, tokensToShow * 2)
    activationData.activations.layers?.forEach((layer, layerIdx) => {
      const transformerLayer: Node[] = []

      // Get activation values for this layer
      const postFF = layer.post_ff?.values as number[][][] | undefined

      for (let i = 0; i < nodesPerLayer; i++) {
        const y = (height / (nodesPerLayer + 1)) * (i + 1)

        // Use actual activation values if available
        let value = 0.5
        if (postFF && postFF[0] && postFF[0][selectedTokenIndex]) {
          const dims = postFF[0][selectedTokenIndex]
          value = Math.abs(dims[i % dims.length] || 0)
        }

        transformerLayer.push({
          x: 100 + layerSpacing * (layerIdx + 2),
          y,
          layer: layerIdx + 2,
          index: i,
          value: Math.min(1, value) // Normalize to 0-1
        })
      }
      layers.push(transformerLayer)
    })

    // Output layer
    const outputLayer: Node[] = []
    const outputNodes = Math.min(8, tokensToShow)
    for (let i = 0; i < outputNodes; i++) {
      const y = (height / (outputNodes + 1)) * (i + 1)
      outputLayer.push({
        x: 100 + layerSpacing * (numLayers - 1),
        y,
        layer: numLayers - 1,
        index: i,
        value: activationData.activations.top_k_logits
          && selectedTokenIndex < activationData.activations.top_k_logits.seq_len
          ? Math.abs(activationData.activations.top_k_logits.values[selectedTokenIndex][i % activationData.activations.top_k_logits.k] || 0)
          : Math.random(),
        label: 'Output'
      })
    }
    layers.push(outputLayer)

    // Create connections between layers
    const connections: Connection[] = []
    for (let l = 0; l < layers.length - 1; l++) {
      const currentLayer = layers[l]
      const nextLayer = layers[l + 1]

      // Connect each node to multiple nodes in next layer
      for (const fromNode of currentLayer) {
        // Connect to nearby nodes in next layer
        const connectionsPerNode = Math.min(5, nextLayer.length)
        const indices = Array.from({ length: nextLayer.length }, (_, i) => i)
          .sort(() => Math.random() - 0.5)
          .slice(0, connectionsPerNode)

        for (const toIdx of indices) {
          const toNode = nextLayer[toIdx]
          connections.push({
            from: fromNode,
            to: toNode,
            strength: (fromNode.value + toNode.value) / 2
          })
        }
      }
    }

    // Animation function
    const animate = () => {
      ctx.fillStyle = '#0a0a0a'
      ctx.fillRect(0, 0, width, height)

      // Draw connections
      ctx.lineWidth = 1
      for (const conn of connections) {
        const alpha = conn.strength * 0.15
        ctx.strokeStyle = `rgba(34, 211, 238, ${alpha})`
        ctx.beginPath()
        ctx.moveTo(conn.from.x, conn.from.y)
        ctx.lineTo(conn.to.x, conn.to.y)
        ctx.stroke()
      }

      // Update and draw particles
      if (isAnimating) {
        // Add new particles
        if (Math.random() < 0.1 * animationSpeed) {
          const randomConn = connections[Math.floor(Math.random() * connections.length)]
          particlesRef.current.push({
            x: randomConn.from.x,
            y: randomConn.from.y,
            targetX: randomConn.to.x,
            targetY: randomConn.to.y,
            progress: 0,
            connection: randomConn
          })
        }

        // Update particles
        particlesRef.current = particlesRef.current.filter(particle => {
          particle.progress += 0.02 * animationSpeed

          if (particle.progress >= 1) {
            return false // Remove completed particles
          }

          // Interpolate position
          particle.x = particle.connection.from.x + (particle.targetX - particle.connection.from.x) * particle.progress
          particle.y = particle.connection.from.y + (particle.targetY - particle.connection.from.y) * particle.progress

          // Draw particle
          const alpha = Math.sin(particle.progress * Math.PI) // Fade in and out
          const gradient = ctx.createRadialGradient(particle.x, particle.y, 0, particle.x, particle.y, 8)
          gradient.addColorStop(0, `rgba(34, 211, 238, ${alpha})`)
          gradient.addColorStop(1, `rgba(34, 211, 238, 0)`)

          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(particle.x, particle.y, 8, 0, Math.PI * 2)
          ctx.fill()

          return true
        })
      }

      // Draw nodes
      for (const layer of layers) {
        for (const node of layer) {
          // Node glow effect
          const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 12)
          const intensity = node.value
          gradient.addColorStop(0, `rgba(34, 211, 238, ${intensity * 0.8})`)
          gradient.addColorStop(0.5, `rgba(34, 211, 238, ${intensity * 0.4})`)
          gradient.addColorStop(1, 'rgba(34, 211, 238, 0)')

          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(node.x, node.y, 12, 0, Math.PI * 2)
          ctx.fill()

          // Node core
          ctx.fillStyle = `rgba(34, 211, 238, ${0.6 + intensity * 0.4})`
          ctx.beginPath()
          ctx.arc(node.x, node.y, 6, 0, Math.PI * 2)
          ctx.fill()

          // Highlight selected token path
          if (node.index === selectedTokenIndex && node.layer < 2) {
            ctx.strokeStyle = '#fbbf24'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(node.x, node.y, 10, 0, Math.PI * 2)
            ctx.stroke()
          }
        }
      }

      // Draw layer labels
      ctx.font = 'bold 14px Monaco, monospace'
      ctx.textAlign = 'center'
      ctx.fillStyle = '#94a3b8'

      const labels = ['Input', 'Embeddings', ...Array(layers.length - 3).fill(0).map((_, i) => `Layer ${i}`), 'Output']
      labels.forEach((label, i) => {
        const x = 100 + layerSpacing * i
        ctx.fillText(label, x, 30)
      })

      // Draw title
      ctx.font = 'bold 18px sans-serif'
      ctx.fillStyle = '#22d3ee'
      ctx.textAlign = 'center'
      ctx.fillText('Neural Network Flow', width / 2, height - 20)

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [activationData, selectedTokenIndex, isAnimating, animationSpeed, showAllTokens, canvasDimensions])

  return (
    <div className={`network-flow ${className}`}>
      <div className="controls-bar">
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className={`btn-control ${isAnimating ? 'active' : ''}`}
        >
          {isAnimating ? '⏸ Pause Animation' : '▶ Start Animation'}
        </button>

        <div className="speed-control">
          <label>Speed: {animationSpeed.toFixed(1)}x</label>
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.1"
            value={animationSpeed}
            onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
          />
        </div>

        <button
          onClick={() => setShowAllTokens(!showAllTokens)}
          className="btn-control"
        >
          {showAllTokens ? 'Show Fewer Tokens' : 'Show All Tokens'}
        </button>
      </div>

      <div className="canvas-container" ref={containerRef}>
        <canvas ref={canvasRef} className="flow-canvas" />
      </div>

      <div className="info-panel">
        <div className="info-item">
          <span className="info-label">Viewing Token:</span>
          <span className="info-value">{selectedTokenIndex}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Total Layers:</span>
          <span className="info-value">{activationData.activations.layers?.length || 0}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Sequence Length:</span>
          <span className="info-value">{activationData.tokens.length}</span>
        </div>
      </div>

      <style>{`
        .network-flow {
          background: linear-gradient(to bottom, rgba(10, 10, 10, 0.95), rgba(20, 20, 20, 0.95));
          border: 1px solid rgba(100, 116, 139, 0.3);
          border-radius: 12px;
          padding: 1.5rem;
        }

        .controls-bar {
          display: flex;
          gap: 1rem;
          align-items: center;
          margin-bottom: 1rem;
          padding: 1rem;
          background: rgba(30, 41, 59, 0.6);
          border-radius: 8px;
          flex-wrap: wrap;
        }

        .btn-control {
          padding: 0.5rem 1rem;
          background: rgba(34, 211, 238, 0.1);
          border: 1px solid rgba(34, 211, 238, 0.3);
          border-radius: 6px;
          color: #22d3ee;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-control:hover {
          background: rgba(34, 211, 238, 0.2);
          box-shadow: 0 0 15px rgba(34, 211, 238, 0.2);
        }

        .btn-control.active {
          background: rgba(34, 211, 238, 0.3);
          border-color: #22d3ee;
        }

        .speed-control {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .speed-control label {
          font-size: 0.75rem;
          color: #94a3b8;
          font-weight: 500;
        }

        .speed-control input[type="range"] {
          width: 150px;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          outline: none;
          -webkit-appearance: none;
        }

        .speed-control input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          background: #22d3ee;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(34, 211, 238, 0.5);
        }

        .canvas-container {
          border-radius: 8px;
          background: #0a0a0a;
          margin-bottom: 1rem;
          width: 100%;
        }

        .flow-canvas {
          display: block;
          image-rendering: pixelated;
          width: 100%;
          height: auto;
        }

        .info-panel {
          display: flex;
          gap: 2rem;
          justify-content: center;
          padding: 1rem;
          background: rgba(30, 41, 59, 0.4);
          border-radius: 8px;
        }

        .info-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.25rem;
        }

        .info-label {
          font-size: 0.7rem;
          color: #64748b;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .info-value {
          font-size: 1.1rem;
          color: #22d3ee;
          font-weight: 600;
          font-family: 'Monaco', monospace;
        }
      `}</style>
    </div>
  )
}
