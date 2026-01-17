/**
 * Generation Flow Visualization
 * Complete animated view of autoregressive generation process.
 * Shows all tokens flowing through the model in a single continuous animation.
 */

import { useEffect, useRef, useState } from 'react'
import type { ActivationData } from '../../types/model'

interface GenerationFlowProps {
  activationData: ActivationData
  className?: string
}

export function GenerationFlow({
  activationData,
  className = ''
}: GenerationFlowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const outputBoxRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()
  const [isPlaying, setIsPlaying] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState(1)
  const [showLabels, setShowLabels] = useState(true)
  const progressRef = useRef(0)
  const [currentOutputTokens, setCurrentOutputTokens] = useState<string[]>([])
  const [hoveredToken, setHoveredToken] = useState<{idx: number, x: number, y: number} | null>(null)
  const canvasContainerRef = useRef<HTMLDivElement>(null)
  const [lastGeneratedToken, setLastGeneratedToken] = useState<{text: string, index: number} | null>(null)

  const width = 1200
  const height = 550
  const numLayers = activationData.activations.layers?.length || 4
  const promptTokens = activationData.metadata.prompt_tokens || 0
  const generatedTokens = activationData.metadata.generated_tokens || 0
  const totalTokens = activationData.tokens.length
  const decodedTokens = activationData.metadata.decoded_tokens || []

  // Extract actual activation magnitudes for node visualization
  const getNodeActivations = (layerIdx: number, tokenIdx: number, nodeIdx: number): number => {
    const layers = activationData.activations.layers
    if (!layers || layerIdx >= layers.length) return 0.2

    const layer = layers[layerIdx]

    // Use post_ff activations if available, otherwise post_attn
    const activationTensor = layer.post_ff || layer.post_attn
    if (!activationTensor) return 0.2

    const values = activationTensor.values as number[][][]
    if (!values || !values[0] || !values[0][tokenIdx]) return 0.2

    const tokenActivations = values[0][tokenIdx]
    const dimIdx = nodeIdx % tokenActivations.length
    const rawValue = Math.abs(tokenActivations[dimIdx])

    // Normalize based on layer statistics
    const normalized = Math.min(1, rawValue / (activationTensor.std * 3))
    return 0.2 + normalized * 0.8
  }

  // Get embedding activation for input/embed layers
  const getEmbeddingActivation = (tokenIdx: number, nodeIdx: number): number => {
    const embeddings = activationData.activations.combined_embeddings || activationData.activations.token_embeddings
    if (!embeddings) return 0.2

    const values = embeddings.values as number[][][]
    if (!values || !values[0] || !values[0][tokenIdx]) return 0.2

    const tokenEmb = values[0][tokenIdx]
    const dimIdx = nodeIdx % tokenEmb.length
    const rawValue = Math.abs(tokenEmb[dimIdx])

    const normalized = Math.min(1, rawValue / (embeddings.std * 3))
    return 0.2 + normalized * 0.8
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height

    // Network layout - horizontal flow with multiple nodes per layer
    const layerSpacing = (width - 200) / (numLayers + 2)
    const tokenHeight = 40
    const maxVisibleTokens = Math.floor((height - 200) / tokenHeight)
    const nodesPerLayer = 10 // Number of nodes to display per layer
    const nodeRadius = 6
    const networkTop = 100
    const networkHeight = 380
    const nodeSpacing = networkHeight / (nodesPerLayer + 1)

    // Create layers with nodes
    const layers = [
      { x: 100, label: 'Input', nodes: [] as Array<{x: number, y: number, active: number}> },
      { x: 100 + layerSpacing, label: 'Embed', nodes: [] as Array<{x: number, y: number, active: number}> },
      ...Array(numLayers).fill(0).map((_, i) => ({
        x: 100 + layerSpacing * (i + 2),
        label: `L${i}`,
        nodes: [] as Array<{x: number, y: number, active: number}>
      })),
      { x: 100 + layerSpacing * (numLayers + 2), label: 'Output', nodes: [] as Array<{x: number, y: number, active: number}> }
    ]

    // Initialize nodes for each layer
    layers.forEach(layer => {
      for (let i = 0; i < nodesPerLayer; i++) {
        layer.nodes.push({
          x: layer.x,
          y: networkTop + nodeSpacing * (i + 1),
          active: 0
        })
      }
    })

    const animate = () => {
      if (!ctx) return

      // Clear canvas
      ctx.fillStyle = '#0a0a0a'
      ctx.fillRect(0, 0, width, height)

      // Update progress
      if (isPlaying) {
        progressRef.current += 0.002 * animationSpeed
        if (progressRef.current > 1) {
          progressRef.current = 0
        }
      }

      const progress = progressRef.current

      // Calculate generation progress (shared by all drawing functions)
      const totalSteps = generatedTokens + 1
      const currentStep = Math.floor(progress * totalSteps)
      const stepProgress = (progress * totalSteps) - currentStep
      const tokensInCurrentStep = Math.min(promptTokens + currentStep, totalTokens)
      const currentTokenIdx = tokensInCurrentStep > 0 ? tokensInCurrentStep - 1 : 0
      const flowingTokenIdx = currentTokenIdx
      const isPromptToken = flowingTokenIdx < promptTokens

      // Update current output tokens for display
      if (tokensInCurrentStep > promptTokens) {
        const outputTokenIndices = Array.from(
          { length: tokensInCurrentStep - promptTokens },
          (_, i) => promptTokens + i
        )
        const newOutputTokens = outputTokenIndices.map(idx => {
          return decodedTokens[idx] || `[${activationData.tokens[idx]}]`
        })
        if (JSON.stringify(newOutputTokens) !== JSON.stringify(currentOutputTokens)) {
          setCurrentOutputTokens(newOutputTokens)
        }
      } else {
        if (currentOutputTokens.length > 0) {
          setCurrentOutputTokens([])
        }
      }

      // Draw network structure
      drawNetwork(ctx, layers, progress, tokensInCurrentStep, stepProgress, flowingTokenIdx, isPromptToken)

      // Draw generation visualization
      drawGenerationProcess(ctx, layers, progress, tokenHeight, maxVisibleTokens, stepProgress, flowingTokenIdx, isPromptToken, tokensInCurrentStep)

      // Draw info
      drawInfo(ctx, progress)

      animationRef.current = requestAnimationFrame(animate)
    }

    const drawNetwork = (
      ctx: CanvasRenderingContext2D,
      layers: Array<{ x: number; label: string; nodes: Array<{x: number, y: number, active: number}> }>,
      progress: number,
      tokensInCurrentStep: number,
      stepProgress: number,
      flowingTokenIdx: number,
      isPrompt: boolean
    ) => {

      // Calculate layer progress for connections
      const layerProgress = stepProgress * layers.length
      const activeLayerIdx = Math.floor(layerProgress)
      const layerStepProgress = layerProgress - activeLayerIdx

      // Draw all connections between layers based on actual activation patterns
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i]
        const nextLayer = layers[i + 1]

        // Calculate activation progress for this connection layer
        const layerActivationProgress = (stepProgress * layers.length) - i
        const isProcessing = layerActivationProgress > 0 && layerActivationProgress <= 1

        // Connect each node to all nodes in next layer
        for (let fromIdx = 0; fromIdx < currentLayer.nodes.length; fromIdx++) {
          const fromNode = currentLayer.nodes[fromIdx]
          for (let toIdx = 0; toIdx < nextLayer.nodes.length; toIdx++) {
            const toNode = nextLayer.nodes[toIdx]

            // Base opacity for inactive connections
            let opacity = 0.02
            let lineWidth = 0.4
            let color = '34, 211, 238'

            if (tokensInCurrentStep > 0 && flowingTokenIdx < totalTokens) {
              color = flowingTokenIdx >= promptTokens ? '167, 139, 250' : '34, 211, 238'

              // Use actual node activations to determine connection strength
              const fromIntensity = currentLayer.nodes[fromIdx]?.active || 0.2
              const toIntensity = nextLayer.nodes[toIdx]?.active || 0.2

              // Connection strength is influenced by both nodes' activations
              const connectionStrength = (fromIntensity + toIntensity) / 2

              // Apply activation boost when this layer is processing
              if (isProcessing) {
                // Modulate based on activation wave progress
                const waveFactor = Math.sin(layerActivationProgress * Math.PI)
                const boostedStrength = connectionStrength * (0.5 + waveFactor * 0.5)

                opacity = 0.03 + boostedStrength * 0.3
                lineWidth = 0.5 + boostedStrength * 1.5
              } else {
                // Subtle base activation
                opacity = 0.02 + connectionStrength * 0.08
                lineWidth = 0.4 + connectionStrength * 0.6
              }

              ctx.strokeStyle = `rgba(${color}, ${opacity})`
            } else {
              // Random variation for visual interest when inactive
              const connectionId = fromNode.y * 1000 + toNode.y
              opacity = 0.02 + (Math.sin(connectionId) * 0.5 + 0.5) * 0.04
              ctx.strokeStyle = `rgba(${color}, ${opacity})`
            }

            ctx.lineWidth = lineWidth
            ctx.beginPath()
            ctx.moveTo(fromNode.x, fromNode.y)
            ctx.lineTo(toNode.x, toNode.y)
            ctx.stroke()
          }
        }
      }

      // Draw nodes
      layers.forEach((layer, layerIdx) => {
        // Calculate activation progress for this layer
        const layerActivationProgress = (stepProgress * layers.length) - layerIdx
        const isProcessing = layerActivationProgress > 0 && layerActivationProgress <= 1
        const hasProcessed = layerActivationProgress > 1

        layer.nodes.forEach((node, nodeIdx) => {
          // Get actual activation intensity from data
          let baseIntensity = 0.15

          if (tokensInCurrentStep > 0 && flowingTokenIdx < totalTokens) {
            if (layerIdx === 0 || layerIdx === 1) {
              // Input or Embedding layer
              baseIntensity = getEmbeddingActivation(flowingTokenIdx, nodeIdx)
            } else if (layerIdx < layers.length - 1) {
              // Transformer layer
              const transformerLayerIdx = layerIdx - 2
              baseIntensity = getNodeActivations(transformerLayerIdx, flowingTokenIdx, nodeIdx)
            } else {
              // Output layer - use logits if available
              if (activationData.activations.logits) {
                const logits = activationData.activations.logits.values as number[][][]
                if (logits && logits[0] && logits[0][flowingTokenIdx]) {
                  const logitValues = logits[0][flowingTokenIdx]
                  const dimIdx = nodeIdx % logitValues.length
                  const rawValue = Math.abs(logitValues[dimIdx])
                  baseIntensity = 0.2 + Math.min(1, rawValue / 10) * 0.8
                }
              }
            }
          }

          // Apply dramatic activation wave when layer is processing
          let intensity = baseIntensity
          if (isProcessing && stepProgress > 0) {
            // Staggered activation across nodes
            const nodePhase = (nodeIdx / nodesPerLayer) * Math.PI
            const activationWave = Math.sin(layerActivationProgress * Math.PI + nodePhase)
            const waveBoost = Math.max(0, activationWave) * 0.6

            intensity = Math.min(1, baseIntensity + waveBoost)
          } else if (hasProcessed) {
            // Gradual fade after processing
            const fadeAmount = Math.max(0, 1 - (layerActivationProgress - 1) * 0.5)
            intensity = baseIntensity * (0.7 + fadeAmount * 0.3)
          }

          node.active = intensity

          // Determine node color based on layer type
          let nodeColor = '34, 211, 238' // cyan for prompt
          if (flowingTokenIdx >= promptTokens) {
            nodeColor = '167, 139, 250' // purple for generated
          }

          // Enhanced glow during activation
          const glowSize = nodeRadius * (2 + intensity * 2)
          const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowSize)
          gradient.addColorStop(0, `rgba(${nodeColor}, ${intensity * 0.5})`)
          gradient.addColorStop(0.5, `rgba(${nodeColor}, ${intensity * 0.2})`)
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(node.x, node.y, glowSize, 0, Math.PI * 2)
          ctx.fill()

          // Node core with intensity-based sizing
          const coreSize = nodeRadius * (0.8 + intensity * 0.4)
          ctx.fillStyle = `rgba(${nodeColor}, ${0.3 + intensity * 0.7})`
          ctx.beginPath()
          ctx.arc(node.x, node.y, coreSize, 0, Math.PI * 2)
          ctx.fill()

          // Bright center for high activation
          if (intensity > 0.6) {
            ctx.fillStyle = `rgba(255, 255, 255, ${(intensity - 0.6) * 1.5})`
            ctx.beginPath()
            ctx.arc(node.x, node.y, coreSize * 0.4, 0, Math.PI * 2)
            ctx.fill()
          }

          // Node border
          ctx.strokeStyle = `rgba(${nodeColor}, ${0.5 + intensity * 0.5})`
          ctx.lineWidth = intensity > 0.6 ? 1.5 : 1
          ctx.beginPath()
          ctx.arc(node.x, node.y, coreSize, 0, Math.PI * 2)
          ctx.stroke()
        })

        // Layer label with highlight when active
        if (showLabels) {
          ctx.font = 'bold 13px Monaco'

          if (isProcessing) {
            ctx.fillStyle = flowingTokenIdx >= promptTokens ? '#a78bfa' : '#22d3ee'
            ctx.shadowColor = ctx.fillStyle
            ctx.shadowBlur = 10
          } else {
            ctx.fillStyle = '#94a3b8'
            ctx.shadowBlur = 0
          }

          ctx.textAlign = 'center'
          ctx.fillText(layer.label, layer.x, 85)
          ctx.shadowBlur = 0
        }
      })
    }

    const drawGenerationProcess = (
      ctx: CanvasRenderingContext2D,
      layers: Array<{ x: number; label: string; nodes: Array<{x: number, y: number, active: number}> }>,
      progress: number,
      tokenHeight: number,
      maxVisibleTokens: number,
      stepProgress: number,
      flowingTokenIdx: number,
      isPrompt: boolean,
      tokensInCurrentStep: number
    ) => {
      // Show generated token appearing next to output layer
      if (tokensInCurrentStep > 0) {
        const color = isPrompt ? '#22d3ee' : '#a78bfa'

        // Calculate layer progress
        const layerProgress = stepProgress * layers.length
        const currentLayerIdx = Math.floor(layerProgress)

        // Show current processing token at the input layer
        const inputLayer = layers[0]
        const inputY = networkTop + networkHeight / 2

        // Draw token being processed indicator at input
        if (showLabels && stepProgress > 0) {
          ctx.font = 'bold 11px Monaco'
          ctx.fillStyle = color
          ctx.textAlign = 'right'
          const labelText = decodedTokens[flowingTokenIdx] || `T${flowingTokenIdx}`
          const displayLabel = labelText.length > 12 ? labelText.substring(0, 11) + '…' : labelText

          // Pulsing effect
          const pulseAlpha = 0.6 + Math.sin(stepProgress * Math.PI * 8) * 0.4
          ctx.globalAlpha = pulseAlpha
          ctx.fillText(`Processing: ${displayLabel}`, inputLayer.x - 15, inputY)
          ctx.globalAlpha = 1
        }

        const outputLayer = layers[layers.length - 1]
        const outputY = networkTop + networkHeight / 2

        // Update last generated token when processing completes
        if (stepProgress > 0.95 || currentLayerIdx >= layers.length - 1) {
          const tokenText = decodedTokens[flowingTokenIdx] || `[${activationData.tokens[flowingTokenIdx]}]`
          setLastGeneratedToken({ text: tokenText, index: flowingTokenIdx })
        }

        // Always show the last generated token (persists until next one)
        if (lastGeneratedToken !== null) {
          const isCurrentToken = lastGeneratedToken.index === flowingTokenIdx
          const tokenColor = lastGeneratedToken.index >= promptTokens ? '#a78bfa' : '#22d3ee'

          // Scale animation only for newly appearing token
          let scale = 1
          if (isCurrentToken && (stepProgress > 0.95 || currentLayerIdx >= layers.length - 1)) {
            const appearProgress = Math.min(1, (stepProgress - 0.95) / 0.05)
            scale = easeInOutCubic(appearProgress)
          }

          if (scale > 0) {
            ctx.save()
            ctx.translate(outputLayer.x + 20, outputY)
            ctx.scale(scale, scale)

            // Token box
            const boxWidth = 80
            const boxHeight = 30

            // Glow effect (stronger for newly generated)
            ctx.shadowColor = tokenColor
            ctx.shadowBlur = isCurrentToken ? 20 * scale : 10

            ctx.fillStyle = `${tokenColor}33`
            ctx.strokeStyle = tokenColor
            ctx.lineWidth = 2
            ctx.fillRect(-boxWidth/2, -boxHeight/2, boxWidth, boxHeight)
            ctx.strokeRect(-boxWidth/2, -boxHeight/2, boxWidth, boxHeight)

            ctx.shadowBlur = 0

            // Token text
            ctx.font = 'bold 12px Monaco'
            ctx.fillStyle = '#ffffff'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            const displayLabel = lastGeneratedToken.text.length > 10
              ? lastGeneratedToken.text.substring(0, 9) + '…'
              : lastGeneratedToken.text
            ctx.fillText(displayLabel, 0, 0)

            ctx.restore()
          }
        }

        // Draw activation wave indicator showing progress through layers
        if (showLabels && currentLayerIdx < layers.length && stepProgress > 0) {
          const currentLayer = layers[currentLayerIdx]
          const indicatorY = networkTop - 25

          // Arrow indicator showing current layer
          ctx.fillStyle = color
          ctx.globalAlpha = 0.7
          ctx.beginPath()
          ctx.moveTo(currentLayer.x, indicatorY)
          ctx.lineTo(currentLayer.x - 6, indicatorY - 8)
          ctx.lineTo(currentLayer.x + 6, indicatorY - 8)
          ctx.closePath()
          ctx.fill()
          ctx.globalAlpha = 1
        }
      }
    }

    const drawInfo = (ctx: CanvasRenderingContext2D, progress: number) => {
      // Title
      ctx.font = 'bold 24px sans-serif'
      ctx.fillStyle = '#22d3ee'
      ctx.textAlign = 'center'
      ctx.fillText('Neural Network Generation Flow', width / 2, 40)

      // Subtitle
      ctx.font = '14px Monaco'
      ctx.fillStyle = '#64748b'
      ctx.fillText('Watch tokens flow through the transformer architecture', width / 2, 65)

      // Legend at top right
      ctx.font = '12px Monaco'
      ctx.textAlign = 'right'

      ctx.fillStyle = '#22d3ee'
      ctx.fillRect(width - 200, 25, 16, 16)
      ctx.fillStyle = '#94a3b8'
      ctx.fillText('Prompt Tokens', width - 210, 37)

      ctx.fillStyle = '#a78bfa'
      ctx.fillRect(width - 200, 50, 16, 16)
      ctx.fillStyle = '#94a3b8'
      ctx.fillText('Generated Tokens', width - 210, 62)

      // Progress bar at bottom
      const barWidth = 500
      const barHeight = 6
      const barX = (width - barWidth) / 2
      const barY = height - 30

      ctx.fillStyle = 'rgba(100, 116, 139, 0.3)'
      ctx.fillRect(barX, barY, barWidth, barHeight)

      const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0)
      gradient.addColorStop(0, '#22d3ee')
      gradient.addColorStop(1, '#a78bfa')
      ctx.fillStyle = gradient
      ctx.fillRect(barX, barY, barWidth * progress, barHeight)

      // Progress percentage
      ctx.font = '11px Monaco'
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.fillText(`${Math.floor(progress * 100)}%`, width / 2, barY + barHeight + 15)
    }

    const easeInOutCubic = (t: number): number => {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [activationData, isPlaying, animationSpeed, showLabels, numLayers, promptTokens, generatedTokens, totalTokens, decodedTokens, width, height, lastGeneratedToken])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    progressRef.current = 0
    setLastGeneratedToken(null)
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    // Check if hovering over any network node
    const progress = progressRef.current
    const totalSteps = generatedTokens + 1
    const currentStep = Math.floor(progress * totalSteps)
    const tokensInCurrentStep = Math.min(promptTokens + currentStep, totalTokens)

    if (tokensInCurrentStep === 0) {
      setHoveredToken(null)
      return
    }

    const currentTokenIdx = tokensInCurrentStep - 1

    // Network layout parameters (must match the animation)
    const layerSpacing = (width - 200) / (numLayers + 2)
    const networkTop = 100
    const networkHeight = 380
    const nodesPerLayer = 10
    const nodeSpacing = networkHeight / (nodesPerLayer + 1)
    const nodeRadius = 6

    // Create layers
    const layers = [
      { x: 100, label: 'Input' },
      { x: 100 + layerSpacing, label: 'Embed' },
      ...Array(numLayers).fill(0).map((_, i) => ({
        x: 100 + layerSpacing * (i + 2),
        label: `L${i}`
      })),
      { x: 100 + layerSpacing * (numLayers + 2), label: 'Output' }
    ]

    // Check each layer's nodes
    let foundToken = null
    for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {
      const layer = layers[layerIdx]

      for (let nodeIdx = 0; nodeIdx < nodesPerLayer; nodeIdx++) {
        const nodeX = layer.x
        const nodeY = networkTop + nodeSpacing * (nodeIdx + 1)

        const distance = Math.sqrt(Math.pow(mouseX - nodeX, 2) + Math.pow(mouseY - nodeY, 2))

        if (distance <= nodeRadius + 5) {
          foundToken = { idx: currentTokenIdx, x: mouseX, y: mouseY }
          break
        }
      }

      if (foundToken) break
    }

    setHoveredToken(foundToken)
  }

  const handleCanvasMouseLeave = () => {
    setHoveredToken(null)
  }

  // Get embedding vector for hovered token
  const getTokenEmbedding = (tokenIdx: number): number[] | null => {
    const embeddings = activationData.activations.combined_embeddings || activationData.activations.token_embeddings
    if (!embeddings) return null

    const values = embeddings.values as number[][][]
    if (!values || !values[0] || !values[0][tokenIdx]) return null

    return values[0][tokenIdx]
  }

  return (
    <div className={`generation-flow ${className}`}>
      <div className="controls-panel">
        <div className="playback-controls">
          <button onClick={handlePlayPause} className="btn-control primary">
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={handleReset} className="btn-control">
            ↺ Reset
          </button>
        </div>

        <div className="settings-controls">
          <div className="control-group">
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
            onClick={() => setShowLabels(!showLabels)}
            className="btn-control"
          >
            {showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels'}
          </button>
        </div>
      </div>

      <div className="visualization-container">
        <div className="canvas-container" ref={canvasContainerRef}>
          <canvas
            ref={canvasRef}
            className="flow-canvas"
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
          />
          {hoveredToken && (
            <div
              className="embedding-tooltip"
              style={{
                left: `${hoveredToken.x + 15}px`,
                top: `${hoveredToken.y + 15}px`
              }}
            >
              <div className="tooltip-header">
                Token: <strong>{decodedTokens[hoveredToken.idx] || `[${activationData.tokens[hoveredToken.idx]}]`}</strong>
              </div>
              <div className="tooltip-content">
                <div className="tooltip-label">Embedding Vector (first 10 dims):</div>
                {(() => {
                  const embedding = getTokenEmbedding(hoveredToken.idx)
                  if (!embedding) return <div className="tooltip-error">No embedding available</div>
                  return (
                    <div className="embedding-values">
                      {embedding.slice(0, 10).map((val, i) => (
                        <div key={i} className="embedding-value">
                          <span className="dim-label">[{i}]</span>
                          <span className="dim-value">{val.toFixed(4)}</span>
                        </div>
                      ))}
                      {embedding.length > 10 && (
                        <div className="embedding-more">... and {embedding.length - 10} more dimensions</div>
                      )}
                    </div>
                  )
                })()}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="output-panel" ref={outputBoxRef}>
        <div className="output-header">
          <span className="output-title">Generated Output</span>
          <span className="output-count">{currentOutputTokens.length} tokens</span>
        </div>
        <div className="output-content">
          {currentOutputTokens.length === 0 ? (
            <div className="output-placeholder">Tokens will appear here as they are generated...</div>
          ) : (
            <div className="output-text">
              {currentOutputTokens.map((token, idx) => (
                <span key={idx} className="output-token" style={{ animationDelay: `${idx * 50}ms` }}>
                  {token}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <style>{`
        .generation-flow {
          background: linear-gradient(to bottom, #0a0a0a, #1a1a1a);
          border: 1px solid rgba(100, 116, 139, 0.3);
          border-radius: 12px;
          padding: 1.5rem;
        }

        .controls-panel {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .playback-controls,
        .settings-controls {
          display: flex;
          gap: 0.75rem;
          align-items: center;
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
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .btn-control:hover:not(:disabled) {
          background: rgba(34, 211, 238, 0.2);
          box-shadow: 0 0 15px rgba(34, 211, 238, 0.3);
          transform: translateY(-1px);
        }

        .btn-control:active:not(:disabled) {
          transform: translateY(0);
        }

        .btn-control.primary {
          background: rgba(34, 211, 238, 0.2);
          border-color: #22d3ee;
          box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          min-width: 150px;
        }

        .control-group label {
          font-size: 0.75rem;
          color: #94a3b8;
          font-weight: 600;
        }

        .control-group input[type="range"] {
          width: 100%;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          outline: none;
          -webkit-appearance: none;
        }

        .control-group input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          background: #22d3ee;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(34, 211, 238, 0.6);
        }

        .visualization-container {
          margin-bottom: 1.5rem;
        }

        .canvas-container {
          position: relative;
          overflow: auto;
          border-radius: 8px;
          background: #0a0a0a;
          box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.5);
        }

        .flow-canvas {
          display: block;
          cursor: crosshair;
        }

        .embedding-tooltip {
          position: absolute;
          background: rgba(15, 23, 42, 0.98);
          border: 1px solid rgba(34, 211, 238, 0.5);
          border-radius: 8px;
          padding: 0.75rem;
          pointer-events: none;
          z-index: 1000;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px rgba(34, 211, 238, 0.2);
          max-width: 300px;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.75rem;
        }

        .tooltip-header {
          color: #22d3ee;
          font-weight: 600;
          margin-bottom: 0.5rem;
          padding-bottom: 0.5rem;
          border-bottom: 1px solid rgba(34, 211, 238, 0.2);
        }

        .tooltip-header strong {
          color: #fff;
        }

        .tooltip-content {
          color: #94a3b8;
        }

        .tooltip-label {
          font-size: 0.7rem;
          color: #64748b;
          margin-bottom: 0.25rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .embedding-values {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 0.25rem;
          margin-top: 0.25rem;
        }

        .embedding-value {
          display: flex;
          justify-content: space-between;
          padding: 0.15rem 0.25rem;
          background: rgba(34, 211, 238, 0.05);
          border-radius: 3px;
        }

        .dim-label {
          color: #64748b;
          font-size: 0.7rem;
        }

        .dim-value {
          color: #22d3ee;
          font-weight: 600;
        }

        .embedding-more {
          grid-column: 1 / -1;
          text-align: center;
          color: #64748b;
          font-size: 0.7rem;
          font-style: italic;
          margin-top: 0.25rem;
        }

        .tooltip-error {
          color: #f87171;
          font-size: 0.7rem;
          font-style: italic;
        }

        .output-panel {
          background: rgba(15, 23, 42, 0.8);
          border: 1px solid rgba(167, 139, 250, 0.3);
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(167, 139, 250, 0.15);
        }

        .output-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          background: rgba(167, 139, 250, 0.1);
          border-bottom: 1px solid rgba(167, 139, 250, 0.2);
        }

        .output-title {
          font-weight: 700;
          font-size: 0.95rem;
          color: #a78bfa;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .output-count {
          font-size: 0.75rem;
          color: #94a3b8;
          font-family: 'Monaco', 'Courier New', monospace;
          background: rgba(167, 139, 250, 0.1);
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
        }

        .output-content {
          padding: 1rem;
          max-height: 200px;
          overflow-y: auto;
        }

        .output-placeholder {
          color: #64748b;
          font-size: 0.85rem;
          font-style: italic;
          text-align: center;
          padding: 2rem 1rem;
        }

        .output-text {
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.9rem;
          line-height: 1.8;
          color: #e2e8f0;
          word-wrap: break-word;
        }

        .output-token {
          display: inline;
          animation: tokenFadeIn 0.3s ease-out;
          background: rgba(167, 139, 250, 0.1);
          padding: 0.15rem 0.25rem;
          border-radius: 3px;
          margin: 0 1px;
        }

        @keyframes tokenFadeIn {
          from {
            opacity: 0;
            transform: translateY(-3px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .output-content::-webkit-scrollbar {
          width: 8px;
        }

        .output-content::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 4px;
        }

        .output-content::-webkit-scrollbar-thumb {
          background: rgba(167, 139, 250, 0.3);
          border-radius: 4px;
        }

        .output-content::-webkit-scrollbar-thumb:hover {
          background: rgba(167, 139, 250, 0.5);
        }
      `}</style>
    </div>
  )
}
