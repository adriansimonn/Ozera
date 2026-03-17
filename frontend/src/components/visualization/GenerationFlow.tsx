/**
 * Generation Flow Visualization
 * Complete animated view of autoregressive generation process.
 * Shows all tokens flowing through the model in a single continuous animation.
 */

import { memo, useEffect, useRef, useState } from 'react'
import type { ActivationData } from '../../types/model'

interface GenerationFlowProps {
  activationData: ActivationData
  className?: string
}

export const GenerationFlow = memo(function GenerationFlow({
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
  const canvasContainerRef = useRef<HTMLDivElement>(null)
  const [, setHoveredTopToken] = useState<{token: string, probability: number} | null>(null)
  const [topTokenChoices, setTopTokenChoices] = useState<Array<{token: string, tokenId: number, probability: number, isSelected: boolean}>>([])
  const [tokenDecodeCache, setTokenDecodeCache] = useState<Map<number, string>>(new Map())

  // Animation refs: these track values during rAF without triggering re-renders.
  // They sync to React state every SYNC_INTERVAL frames so the DOM updates.
  const SYNC_INTERVAL = 10
  const frameCountRef = useRef(0)
  const currentOutputTokensRef = useRef<string[]>([])
  const lastGeneratedTokenRef = useRef<{text: string, index: number} | null>(null)
  const topTokenChoicesRef = useRef<Array<{token: string, tokenId: number, probability: number, isSelected: boolean}>>([])
  const needsSyncRef = useRef(false)
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1400, height: 550 })

  const width = canvasDimensions.width
  const height = canvasDimensions.height
  const numLayers = activationData.activations.layers?.length || 4
  const promptTokens = activationData.metadata.prompt_tokens || 0
  const generatedTokens = activationData.metadata.generated_tokens || 0
  const totalTokens = activationData.tokens.length
  const decodedTokens = activationData.metadata.decoded_tokens || []
  const topK = activationData.metadata.top_k || 10

  // Track activation ID to detect changes and avoid stale cache
  const lastActivationIdRef = useRef<string>('')

  // Helper to get decoded token text - checks metadata first, then cache
  const getDecodedToken = (idx: number): string => {
    // First try metadata decoded_tokens
    if (decodedTokens[idx]) {
      return decodedTokens[idx]
    }
    // Then try cache using the token ID
    const tokenId = activationData.tokens[idx]
    if (tokenDecodeCache.has(tokenId)) {
      return tokenDecodeCache.get(tokenId)!
    }
    // Fallback to token ID display
    return `[${tokenId}]`
  }

  // Decode token IDs to text via API.
  // If forceClean is true, starts with an empty cache (used on activation change).
  const decodeTokenIds = async (tokenIds: number[], forceClean: boolean = false): Promise<Map<number, string>> => {
    const baseCache = forceClean ? new Map<number, string>() : new Map(tokenDecodeCache)
    const toFetch = tokenIds.filter(id => !baseCache.has(id))

    if (toFetch.length === 0) return baseCache

    try {
      // Use the open-source endpoint for non-Ozera models (they have different tokenizers)
      const modelFamily = activationData.metadata?.model_family
      const isOpenSourceModel = modelFamily && modelFamily !== 'ozera'
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const endpoint = isOpenSourceModel
        ? `${apiBase}/open-source/decode-tokens`
        : `${apiBase}/decode-tokens`

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token_ids: toFetch,
          model: activationData.model
        })
      })

      if (response.ok) {
        const data = await response.json()
        toFetch.forEach((id, idx) => {
          baseCache.set(id, data.decoded_tokens[idx])
        })
        setTokenDecodeCache(baseCache)
      }
    } catch (error) {
      console.error('Error decoding tokens:', error)
    }

    return baseCache
  }

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

  // Get top token probabilities from pre-computed top-K logits
  const getTopTokenProbabilities = (tokenIdx: number, topKCount: number = 10): Array<{token: string, tokenId: number, probability: number, isSelected: boolean}> => {
    const topKLogits = activationData.activations.top_k_logits
    if (!topKLogits || tokenIdx < 0 || tokenIdx >= topKLogits.seq_len) return []

    const indices = topKLogits.indices[tokenIdx]
    const probabilities = topKLogits.probabilities[tokenIdx]
    const decodedToks = topKLogits.decoded_tokens[tokenIdx]
    if (!indices || !probabilities || !decodedToks) return []

    // Determine which token was actually selected
    const selectedTokenId = tokenIdx + 1 < activationData.tokens.length ? activationData.tokens[tokenIdx + 1] : -1

    const count = Math.min(topKCount, indices.length)
    const result: Array<{token: string, tokenId: number, probability: number, isSelected: boolean}> = []

    for (let i = 0; i < count; i++) {
      const decoded = decodedToks[i]
      let displayToken: string
      if (decoded === '') {
        displayToken = '\u2205'
      } else if (decoded.trim() === '') {
        displayToken = JSON.stringify(decoded)
      } else {
        displayToken = decoded
      }
      result.push({
        token: displayToken,
        tokenId: indices[i],
        probability: probabilities[i],
        isSelected: indices[i] === selectedTokenId
      })
    }

    return result
  }

  // Pre-fetch sequence token IDs when activation data loads
  // (top-K logit tokens are now pre-decoded by the backend)
  useEffect(() => {
    // Detect if this is a new activation (different model/run = different tokenizer)
    const isNewActivation = lastActivationIdRef.current !== activationData.id
    if (isNewActivation) {
      lastActivationIdRef.current = activationData.id
      setTopTokenChoices([])
    }

    const prefetchSequenceTokens = async () => {
      // Only need to decode sequence tokens (for generated output display)
      const allTokenIds = new Set<number>()
      activationData.tokens.forEach(tokenId => allTokenIds.add(tokenId))

      if (allTokenIds.size > 0) {
        await decodeTokenIds(Array.from(allTokenIds), isNewActivation)
      }
    }

    prefetchSequenceTokens()
  }, [activationData])

  // Update canvas dimensions based on container size
  useEffect(() => {
    const updateDimensions = () => {
      if (canvasContainerRef.current) {
        const containerWidth = canvasContainerRef.current.clientWidth
        // Maintain aspect ratio while filling container
        const aspectRatio = 550 / 1400
        const newHeight = Math.max(550, containerWidth * aspectRatio)
        setCanvasDimensions({ width: containerWidth, height: newHeight })
      }
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (canvasContainerRef.current) {
      resizeObserver.observe(canvasContainerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

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

      // Update current output tokens in ref (no re-render)
      if (tokensInCurrentStep > promptTokens) {
        const outputTokenIndices = Array.from(
          { length: tokensInCurrentStep - promptTokens },
          (_, i) => promptTokens + i
        )
        const newOutputTokens = outputTokenIndices.map(idx => {
          return getDecodedToken(idx)
        })
        if (JSON.stringify(newOutputTokens) !== JSON.stringify(currentOutputTokensRef.current)) {
          currentOutputTokensRef.current = newOutputTokens
          needsSyncRef.current = true
        }
      } else {
        if (currentOutputTokensRef.current.length > 0) {
          currentOutputTokensRef.current = []
          needsSyncRef.current = true
        }
      }

      // Draw network structure
      drawNetwork(ctx, layers, progress, tokensInCurrentStep, stepProgress, flowingTokenIdx, isPromptToken)

      // Draw generation visualization
      drawGenerationProcess(ctx, layers, progress, tokenHeight, maxVisibleTokens, stepProgress, flowingTokenIdx, isPromptToken, tokensInCurrentStep)

      // Draw info
      drawInfo(ctx, progress)

      // Throttle React state syncs to every SYNC_INTERVAL frames
      frameCountRef.current++
      if (needsSyncRef.current && frameCountRef.current % SYNC_INTERVAL === 0) {
        needsSyncRef.current = false
        setCurrentOutputTokens(currentOutputTokensRef.current)
        setTopTokenChoices(topTokenChoicesRef.current)
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    const drawNetwork = (
      ctx: CanvasRenderingContext2D,
      layers: Array<{ x: number; label: string; nodes: Array<{x: number, y: number, active: number}> }>,
      _progress: number,
      tokensInCurrentStep: number,
      stepProgress: number,
      flowingTokenIdx: number,
      _isPrompt: boolean
    ) => {

      // Calculate layer progress for connections
      const layerProgress = stepProgress * layers.length
      const activeLayerIdx = Math.floor(layerProgress)
      void (layerProgress - activeLayerIdx)

      // Draw all connections between layers based on actual activation patterns
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i]
        const nextLayer = layers[i + 1]

        // Calculate activation progress for this connection layer
        const layerActivationProgress = (stepProgress * layers.length) - i
        const isProcessing = layerActivationProgress > 0 && layerActivationProgress <= 1
        const hasProcessed = layerActivationProgress > 1

        // Connect each node to all nodes in next layer
        for (let fromIdx = 0; fromIdx < currentLayer.nodes.length; fromIdx++) {
          const fromNode = currentLayer.nodes[fromIdx]
          for (let toIdx = 0; toIdx < nextLayer.nodes.length; toIdx++) {
            const toNode = nextLayer.nodes[toIdx]

            // Base opacity for inactive connections
            let opacity = 0.02
            let lineWidth = 0.4
            let color = '255, 255, 255' // White for all connections

            if (tokensInCurrentStep > 0 && flowingTokenIdx < totalTokens) {
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

                // Much brighter and more visible activations
                opacity = 0.1 + boostedStrength * 0.6
                lineWidth = 1.0 + boostedStrength * 2.5
              } else if (hasProcessed && stepProgress < 0.95) {
                // Keep connections activated (persistent) after processing until token generation completes
                opacity = 0.08 + connectionStrength * 0.4
                lineWidth = 0.8 + connectionStrength * 1.5
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
          // Only show real activation data once the animation pass has reached this layer
          let baseIntensity = 0.15

          if (tokensInCurrentStep > 0 && flowingTokenIdx < totalTokens && (isProcessing || hasProcessed)) {
            if (layerIdx === 0 || layerIdx === 1) {
              // Input or Embedding layer
              baseIntensity = getEmbeddingActivation(flowingTokenIdx, nodeIdx)
            } else if (layerIdx < layers.length - 1) {
              // Transformer layer
              const transformerLayerIdx = layerIdx - 2
              baseIntensity = getNodeActivations(transformerLayerIdx, flowingTokenIdx, nodeIdx)
            } else {
              // Output layer - use top-K logit values for intensity
              const topKLogits = activationData.activations.top_k_logits
              if (topKLogits && flowingTokenIdx < topKLogits.seq_len) {
                const topValues = topKLogits.values[flowingTokenIdx]
                if (topValues && topValues.length > 0) {
                  // Use the spread of top-K values to drive node intensity
                  const idx = nodeIdx % topValues.length
                  const rawValue = Math.abs(topValues[idx])
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
            const waveBoost = Math.max(0, activationWave) * 0.8

            // Much brighter during processing
            intensity = Math.min(1, baseIntensity * 1.5 + waveBoost)
          } else if (hasProcessed && stepProgress < 0.95) {
            // Keep nodes bright and activated after processing until token generation completes
            intensity = baseIntensity * 1.3
          } else if (hasProcessed) {
            // Final fade after token completes
            const fadeAmount = Math.max(0, 1 - (layerActivationProgress - 1) * 0.5)
            intensity = baseIntensity * (0.7 + fadeAmount * 0.3)
          }

          node.active = intensity

          // White color for all nodes
          let nodeColor = '255, 255, 255'

          // Enhanced glow during activation - much brighter
          const glowSize = nodeRadius * (2.5 + intensity * 3)
          const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowSize)
          gradient.addColorStop(0, `rgba(${nodeColor}, ${intensity * 0.8})`)
          gradient.addColorStop(0.5, `rgba(${nodeColor}, ${intensity * 0.4})`)
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(node.x, node.y, glowSize, 0, Math.PI * 2)
          ctx.fill()

          // Node core with intensity-based sizing - brighter
          const coreSize = nodeRadius * (0.8 + intensity * 0.4)
          ctx.fillStyle = `rgba(${nodeColor}, ${0.4 + intensity * 0.6})`
          ctx.beginPath()
          ctx.arc(node.x, node.y, coreSize, 0, Math.PI * 2)
          ctx.fill()

          // Bright center for high activation - lower threshold, brighter core
          if (intensity > 0.4) {
            ctx.fillStyle = `rgba(255, 255, 255, ${(intensity - 0.4) * 1.8})`
            ctx.beginPath()
            ctx.arc(node.x, node.y, coreSize * 0.5, 0, Math.PI * 2)
            ctx.fill()
          }

          // Node border - brighter and thicker during activation
          ctx.strokeStyle = `rgba(${nodeColor}, ${0.6 + intensity * 0.4})`
          ctx.lineWidth = intensity > 0.5 ? 2 : 1.2
          ctx.beginPath()
          ctx.arc(node.x, node.y, coreSize, 0, Math.PI * 2)
          ctx.stroke()
        })

        // Layer label with highlight when active
        if (showLabels) {
          ctx.font = 'bold 13px Monaco'

          if (isProcessing) {
            ctx.fillStyle = '#ffffff'
            ctx.shadowColor = '#ffffff'
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
      _progress: number,
      _tokenHeight: number,
      _maxVisibleTokens: number,
      stepProgress: number,
      flowingTokenIdx: number,
      _isPrompt: boolean,
      tokensInCurrentStep: number
    ) => {
      // Show generated token appearing next to output layer
      if (tokensInCurrentStep > 0) {
        const color = '#ffffff' // White for all tokens

        // Calculate layer progress
        const layerProgress = stepProgress * layers.length
        const currentLayerIdx = Math.floor(layerProgress)

        // Update top token choices for the current token immediately.
        // logits[0][i] predicts tokens[i+1], so to show predictions for the
        // token at position X, we read logits at position X-1.
        if (flowingTokenIdx >= promptTokens && flowingTokenIdx < totalTokens) {
          const topChoices = getTopTokenProbabilities(flowingTokenIdx - 1, topK)
          if (topChoices.length > 0 && JSON.stringify(topChoices) !== JSON.stringify(topTokenChoicesRef.current)) {
            topTokenChoicesRef.current = topChoices
            needsSyncRef.current = true
          }
        }

        if (stepProgress > 0.95 || currentLayerIdx >= layers.length - 1) {
          const tokenText = getDecodedToken(flowingTokenIdx)
          const tokenToDisplay = { text: tokenText, index: flowingTokenIdx }
          if (!lastGeneratedTokenRef.current || lastGeneratedTokenRef.current.index !== flowingTokenIdx) {
            lastGeneratedTokenRef.current = tokenToDisplay
            needsSyncRef.current = true
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
      // Processing token label above progress bar
      const totalSteps = generatedTokens + 1
      const currentStep = Math.floor(progress * totalSteps)
      const tokensInCurrentStep = Math.min(promptTokens + currentStep, totalTokens)
      const currentTokenIdx = tokensInCurrentStep > 0 ? tokensInCurrentStep - 1 : 0

      if (showLabels && progress > 0 && progress < 1) {
        const labelText = getDecodedToken(currentTokenIdx)
        const displayLabel = labelText.length > 12 ? labelText.substring(0, 11) + '…' : labelText

        ctx.font = 'bold 14px Monaco'
        ctx.fillStyle = '#ffffff'
        ctx.textAlign = 'center'
        const pulseAlpha = 0.6 + Math.sin(progress * Math.PI * 8) * 0.4
        ctx.globalAlpha = pulseAlpha
        ctx.fillText(`Processing: ${displayLabel}`, width / 2, height - 60)
        ctx.globalAlpha = 1
      }

      // Progress bar at bottom
      const barWidth = 500
      const barHeight = 6
      const barX = (width - barWidth) / 2
      const barY = height - 30

      ctx.fillStyle = 'rgba(100, 116, 139, 0.3)'
      ctx.fillRect(barX, barY, barWidth, barHeight)

      ctx.fillStyle = '#ffffff'
      ctx.fillRect(barX, barY, barWidth * progress, barHeight)

      // Progress percentage
      ctx.font = '11px Monaco'
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.fillText(`${Math.floor(progress * 100)}%`, width / 2, barY + barHeight + 15)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [activationData, isPlaying, animationSpeed, showLabels, numLayers, promptTokens, generatedTokens, totalTokens, decodedTokens, tokenDecodeCache, width, height, canvasDimensions])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    progressRef.current = 0
    lastGeneratedTokenRef.current = null
    topTokenChoicesRef.current = []
    currentOutputTokensRef.current = []
    setTopTokenChoices([])
    setCurrentOutputTokens([])
  }

  const handlePrevious = () => {
    // Move backward by one layer step
    const totalSteps = generatedTokens + 1
    const layerStep = 1 / (numLayers + 3) // +3 for input, embed, output layers
    const newProgress = Math.max(0, progressRef.current - layerStep / totalSteps)
    progressRef.current = newProgress
  }

  const handleNext = () => {
    // Move forward by one layer step
    const totalSteps = generatedTokens + 1
    const layerStep = 1 / (numLayers + 3) // +3 for input, embed, output layers
    const newProgress = Math.min(1, progressRef.current + layerStep / totalSteps)
    progressRef.current = newProgress
  }


  return (
    <div className={`generation-flow ${className}`}>
      <div className="controls-panel">
        <div className="controls-section controls-left">
          <div className="control-group speed-control">
            <label>Speed: {animationSpeed.toFixed(2)}x</label>
            <input
              type="range"
              min="0.01"
              max="3"
              step="0.01"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="controls-section controls-center">
          <button onClick={handlePrevious} className="btn-control">
            ⏮ Previous
          </button>
          <button onClick={handlePlayPause} className="btn-control primary">
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={handleNext} className="btn-control">
            Next ⏭
          </button>
          <button onClick={handleReset} className="btn-control">
            ↺ Reset
          </button>
        </div>

        <div className="controls-section controls-right">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className="btn-control"
          >
            {showLabels ? 'Hide Labels' : 'Show Labels'}
          </button>
        </div>
      </div>

      <div className="visualization-container">
        <div className="visualization-content">
          <div className="canvas-container" ref={canvasContainerRef}>
            <canvas
              ref={canvasRef}
              className="flow-canvas"
            />
          </div>

          <div className="top-tokens-panel" style={{ maxHeight: `${canvasDimensions.height}px` }}>
            <div className="top-tokens-header">
              <span className="top-tokens-title">Top-{topK} Token Choices</span>
            </div>
            <div className="top-tokens-list">
              {topTokenChoices.length === 0 ? (
                <div className="top-tokens-placeholder">
                  Token probabilities will appear here...
                </div>
              ) : (
                topTokenChoices.map((choice, idx) => (
                  <div
                    key={idx}
                    className={`top-token-item ${choice.isSelected ? 'selected' : ''}`}
                    onMouseEnter={() => setHoveredTopToken({ token: choice.token, probability: choice.probability })}
                    onMouseLeave={() => setHoveredTopToken(null)}
                  >
                    <div className="top-token-rank">#{idx + 1}</div>
                    <div className="top-token-text">{choice.token}</div>
                    <div className="top-token-probability">
                      {(choice.probability * 100).toFixed(2)}%
                    </div>
                    {choice.isSelected && (
                      <div className="selected-indicator">✓</div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
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

      <div className="prompt-tokens-panel">
        <div className="prompt-tokens-header">
          <span className="prompt-tokens-title">Tokenized Prompt</span>
          <span className="prompt-tokens-count">{promptTokens} tokens</span>
        </div>
        <div className="prompt-tokens-content">
          {promptTokens === 0 ? (
            <div className="prompt-tokens-placeholder">Prompt tokens will appear here...</div>
          ) : (
            <div className="prompt-tokens-list">
              {Array.from({ length: promptTokens }, (_, idx) => {
                const tokenText = getDecodedToken(idx)
                return (
                  <span key={idx} className="prompt-token">
                    <span className="prompt-token-index">{idx}</span>
                    <span className="prompt-token-text">{tokenText}</span>
                  </span>
                )
              })}
            </div>
          )}
        </div>
      </div>

      <style>{`
        .generation-flow {
          padding: 0;
        }

        .controls-panel {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem;
          margin-bottom: 1.5rem;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05);
          flex-wrap: wrap;
          gap: 0.75rem;
        }

        .controls-section {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .controls-left {
          flex: 1;
          justify-content: flex-start;
        }

        .controls-center {
          flex: 0 0 auto;
        }

        .controls-right {
          flex: 1;
          justify-content: flex-end;
        }

        .btn-control {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.15);
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
          color: #ffffff;
        }

        .btn-control:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.12);
          box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
          transform: translateY(-1px);
        }

        .btn-control:active:not(:disabled) {
          transform: translateY(0);
        }

        .btn-control.primary {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.3);
          box-shadow: 0 0 10px rgba(255, 255, 255, 0.15);
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          min-width: 150px;
        }

        .control-group.speed-control {
          min-width: 300px;
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
          background: #ffffff;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(255, 255, 255, 0.6);
        }

        .visualization-container {
          margin-bottom: 1.5rem;
        }

        .visualization-content {
          display: flex;
          gap: 1rem;
          align-items: flex-start;
        }

        .canvas-container {
          position: relative;
          flex: 1;
          min-width: 0;
        }

        .flow-canvas {
          display: block;
          cursor: crosshair;
          width: 100%;
          height: auto;
        }

        .top-tokens-panel {
          width: 280px;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05);
          display: flex;
          flex-direction: column;
          flex-shrink: 0;
        }

        .top-tokens-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          background: transparent;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .top-tokens-title {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 400;
        }

        .top-tokens-count {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.5);
          font-weight: 400;
        }

        .top-tokens-list {
          flex: 1;
          overflow-y: auto;
          padding: 0.5rem;
        }

        .top-tokens-placeholder {
          color: #64748b;
          font-size: 0.8rem;
          font-style: italic;
          text-align: center;
          padding: 2rem 1rem;
        }

        .top-token-item {
          display: grid;
          grid-template-columns: 35px 1fr 70px 20px;
          gap: 0.5rem;
          align-items: center;
          padding: 0.6rem 0.75rem;
          margin-bottom: 0.4rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 0;
          transition: all 0.2s ease;
          cursor: pointer;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.75rem;
        }

        .top-token-item:hover {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
          transform: translateX(3px);
        }

        .top-token-item.selected {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.3);
          box-shadow: 0 0 20px rgba(255, 255, 255, 0.15);
          font-weight: 600;
        }

        .top-token-item.selected .top-token-text {
          color: #ffffff;
          font-weight: 700;
        }

        .top-token-rank {
          color: #64748b;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .top-token-item.selected .top-token-rank {
          color: #94a3b8;
        }

        .top-token-text {
          color: #e2e8f0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .top-token-probability {
          color: #94a3b8;
          font-size: 0.7rem;
          text-align: right;
          font-weight: 600;
        }

        .top-token-item.selected .top-token-probability {
          color: #ffffff;
        }

        .selected-indicator {
          color: #ffffff;
          font-size: 0.9rem;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .top-tokens-list::-webkit-scrollbar {
          width: 6px;
        }

        .top-tokens-list::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 3px;
        }

        .top-tokens-list::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.15);
          border-radius: 3px;
        }

        .top-tokens-list::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.25);
        }

        .output-panel {
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05);
          overflow: hidden;
        }

        .output-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          background: transparent;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .output-title {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 400;
        }

        .output-count {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.5);
          font-weight: 400;
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
          background: rgba(255, 255, 255, 0.05);
          padding: 0.15rem 0.25rem;
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
          background: rgba(255, 255, 255, 0.15);
        }

        .output-content::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.25);
        }

        .prompt-tokens-panel {
          margin-top: 1.5rem;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05);
          overflow: hidden;
        }

        .prompt-tokens-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          background: transparent;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .prompt-tokens-title {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 400;
        }

        .prompt-tokens-count {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.5);
          font-weight: 400;
        }

        .prompt-tokens-content {
          padding: 1rem;
          max-height: 200px;
          overflow-y: auto;
        }

        .prompt-tokens-placeholder {
          color: #64748b;
          font-size: 0.85rem;
          font-style: italic;
          text-align: center;
          padding: 2rem 1rem;
        }

        .prompt-tokens-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }

        .prompt-token {
          display: inline-flex;
          align-items: center;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.8rem;
          transition: all 0.2s ease;
        }

        .prompt-token:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .prompt-token-index {
          padding: 0.3rem 0.5rem;
          background: rgba(255, 255, 255, 0.08);
          color: #64748b;
          font-size: 0.65rem;
          font-weight: 600;
          border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        .prompt-token-text {
          padding: 0.3rem 0.5rem;
          color: #e2e8f0;
        }

        .prompt-tokens-content::-webkit-scrollbar {
          width: 8px;
        }

        .prompt-tokens-content::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 4px;
        }

        .prompt-tokens-content::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.15);
        }

        .prompt-tokens-content::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.25);
        }

        /* Light mode */
        [data-bg="light"] .controls-panel {
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid rgba(0, 0, 0, 0.1);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8), 0 0 0 1px rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .btn-control {
          background: rgba(0, 0, 0, 0.04);
          border: 1px solid rgba(0, 0, 0, 0.12);
          color: #1d1d1f;
        }

        [data-bg="light"] .btn-control:hover:not(:disabled) {
          background: rgba(0, 0, 0, 0.06);
          box-shadow: 0 0 15px rgba(0, 0, 0, 0.08);
        }

        [data-bg="light"] .btn-control.primary {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.2);
          box-shadow: 0 0 10px rgba(0, 0, 0, 0.06);
        }

        [data-bg="light"] .control-group input[type="range"] {
          background: rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .control-group input[type="range"]::-webkit-slider-thumb {
          background: #1d1d1f;
          box-shadow: 0 0 8px rgba(0, 0, 0, 0.2);
        }

        [data-bg="light"] .top-tokens-panel {
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid rgba(0, 0, 0, 0.1);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8), 0 0 0 1px rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .top-tokens-header {
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .top-tokens-title {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .top-tokens-count {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .top-token-item {
          background: rgba(0, 0, 0, 0.03);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .top-token-item:hover {
          background: rgba(0, 0, 0, 0.05);
          border-color: rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .top-token-item.selected {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.2);
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.06);
        }

        [data-bg="light"] .top-token-item.selected .top-token-text {
          color: #1d1d1f;
        }

        [data-bg="light"] .top-token-text {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .top-token-item.selected .top-token-probability {
          color: #1d1d1f;
        }

        [data-bg="light"] .selected-indicator {
          color: #1d1d1f;
        }

        [data-bg="light"] .top-tokens-list::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.03);
        }

        [data-bg="light"] .top-tokens-list::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.12);
        }

        [data-bg="light"] .top-tokens-list::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.2);
        }

        [data-bg="light"] .output-panel {
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid rgba(0, 0, 0, 0.1);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8), 0 0 0 1px rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .output-header {
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .output-title {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .output-count {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .output-text {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .output-token {
          background: rgba(0, 0, 0, 0.04);
        }

        [data-bg="light"] .output-content::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.03);
        }

        [data-bg="light"] .output-content::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.12);
        }

        [data-bg="light"] .output-content::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.2);
        }

        [data-bg="light"] .prompt-tokens-panel {
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid rgba(0, 0, 0, 0.1);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8), 0 0 0 1px rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .prompt-tokens-header {
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .prompt-tokens-title {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .prompt-tokens-count {
          color: rgba(0, 0, 0, 0.45);
        }

        [data-bg="light"] .prompt-token {
          background: rgba(0, 0, 0, 0.04);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .prompt-token:hover {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .prompt-token-index {
          background: rgba(0, 0, 0, 0.05);
          border-right: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-bg="light"] .prompt-token-text {
          color: rgba(0, 0, 0, 0.7);
        }

        [data-bg="light"] .prompt-tokens-content::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.03);
        }

        [data-bg="light"] .prompt-tokens-content::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.12);
        }

        [data-bg="light"] .prompt-tokens-content::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.2);
        }

        /* Default interface: remove gap below controls panel */
        [data-interface="default"] .controls-panel {
          margin-bottom: 0;
        }
      `}</style>
    </div>
  )
})
