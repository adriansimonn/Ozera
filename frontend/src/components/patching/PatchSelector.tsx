/**
 * Visual layer selector for activation patching.
 * Displays a network diagram where users can click layers to add patches.
 * Reuses patterns from GenerationFlow visualization.
 */

import { useRef, useEffect, useState, useCallback } from 'react'
import { Network, Plus, Eye, Zap, CircleOff } from 'lucide-react'
import type { PatchSpec, PatchType, InterventionType, ModelLayerInfo } from '../../types/patching'
import { useTheme } from '../../hooks/useTheme'

interface PatchSelectorProps {
  modelInfo: ModelLayerInfo | null
  patches: PatchSpec[]
  onAddPatch: (patch: PatchSpec) => void
  disabled?: boolean
}

// Color scheme for different patch types
const PATCH_TYPE_COLORS: Record<PatchType, { primary: string; secondary: string; label: string }> = {
  residual: { primary: '147, 51, 234', secondary: '168, 85, 247', label: 'Residual' },  // Purple
  attention: { primary: '59, 130, 246', secondary: '96, 165, 250', label: 'Attention' }, // Blue
  mlp: { primary: '34, 197, 94', secondary: '74, 222, 128', label: 'MLP' },              // Green
  attn_output: { primary: '6, 182, 212', secondary: '34, 211, 238', label: 'Attn Out' },  // Cyan
  ff_output: { primary: '245, 158, 11', secondary: '251, 191, 36', label: 'FF Out' },     // Amber
  post_attn: { primary: '236, 72, 153', secondary: '244, 114, 182', label: 'Post-Attn' }, // Pink
  post_ff: { primary: '249, 115, 22', secondary: '251, 146, 60', label: 'Post-FF' },      // Orange
}

// Color scheme for ablation types
const ABLATION_TYPE_COLORS: Record<string, { primary: string; secondary: string; label: string }> = {
  zero_ablate: { primary: '239, 68, 68', secondary: '248, 113, 113', label: 'Zero' },
  mean_ablate: { primary: '168, 85, 247', secondary: '192, 132, 252', label: 'Mean' },
  noise_ablate: { primary: '251, 191, 36', secondary: '253, 224, 71', label: 'Noise' },
}

// Helper to get colors and label for a patch based on its intervention type
function getPatchColors(patchInfo: { patchType: PatchType; interventionType: InterventionType }) {
  const isAblation = patchInfo.interventionType !== 'patch'
  if (isAblation) {
    const ablationColors = ABLATION_TYPE_COLORS[patchInfo.interventionType]
    return {
      ...ablationColors,
      label: `${ablationColors.label} ${PATCH_TYPE_COLORS[patchInfo.patchType].label}`,
    }
  }
  return PATCH_TYPE_COLORS[patchInfo.patchType]
}

interface LayerNode {
  x: number
  y: number
  layerIdx: number
  label: string
  isTransformerLayer: boolean
}

interface PatchInfo {
  patchType: PatchType
  interventionType: InterventionType
}

interface HoverInfo {
  layerIdx: number
  x: number
  y: number
  label: string
  patchCount: number
  activePatches: PatchInfo[]
}

export function PatchSelector({
  modelInfo,
  patches,
  onAddPatch,
  disabled = false,
}: PatchSelectorProps) {
  const { isLight } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 300 })
  const [hoveredLayer, setHoveredLayer] = useState<HoverInfo | null>(null)
  const [showPatchMenu, setShowPatchMenu] = useState<{ layerIdx: number; x: number; y: number } | null>(null)
  const layerNodesRef = useRef<LayerNode[]>([])

  const numLayers = modelInfo?.num_layers ?? 6
  const numHeads = modelInfo?.num_heads ?? 8

  // Get patches for a specific layer
  const getLayerPatches = useCallback((layerIdx: number): PatchSpec[] => {
    return patches.filter(p => p.layer === layerIdx)
  }, [patches])

  // Get unique patch info (type + intervention) for a layer
  const getLayerPatchInfo = useCallback((layerIdx: number): PatchInfo[] => {
    const layerPatches = getLayerPatches(layerIdx)
    // Create unique key for each patch type + intervention combination
    const seen = new Set<string>()
    return layerPatches.filter(p => {
      const key = `${p.patch_type}-${p.intervention_type}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    }).map(p => ({ patchType: p.patch_type, interventionType: p.intervention_type }))
  }, [getLayerPatches])

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth
        setDimensions({ width, height: Math.max(280, Math.min(350, width * 0.4)) })
      }
    }

    updateDimensions()
    const resizeObserver = new ResizeObserver(updateDimensions)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => resizeObserver.disconnect()
  }, [])

  // Draw the network visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = dimensions
    canvas.width = width
    canvas.height = height

    // Theme-aware canvas colors
    const bgColor = isLight ? '#f5f5f7' : 'rgba(10, 10, 10, 1)'
    const lineColor = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.06)'
    const defaultNodeColor = isLight ? '160, 160, 170' : '100, 100, 100'
    const inactiveNodeColor = isLight ? '190, 190, 200' : '60, 60, 60'
    const borderDefault = (hovered: boolean) => isLight
      ? `rgba(0, 0, 0, ${hovered ? 0.35 : 0.15})`
      : `rgba(255, 255, 255, ${hovered ? 0.5 : 0.2})`
    const borderInactive = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.1)'
    const textPrimary = (opacity: number) => isLight
      ? `rgba(0, 0, 0, ${opacity})`
      : `rgba(255, 255, 255, ${opacity})`

    // Clear canvas
    ctx.fillStyle = bgColor
    ctx.fillRect(0, 0, width, height)

    // Layout calculations
    const padding = { left: 60, right: 60, top: 50, bottom: 40 }
    const networkWidth = width - padding.left - padding.right

    // Total layers: Input -> Embed -> [L0...Ln-1] -> Output
    const totalVisualLayers = numLayers + 3 // Input, Embed, n transformer layers, Output
    const layerSpacing = networkWidth / (totalVisualLayers - 1)

    const nodeRadius = 18
    const nodesPerLayer = 4

    // Create layer positions
    const layers: LayerNode[] = []

    // Input layer
    layers.push({
      x: padding.left,
      y: height / 2,
      layerIdx: -2,
      label: 'Input',
      isTransformerLayer: false,
    })

    // Embed layer
    layers.push({
      x: padding.left + layerSpacing,
      y: height / 2,
      layerIdx: -1,
      label: 'Embed',
      isTransformerLayer: false,
    })

    // Transformer layers
    for (let i = 0; i < numLayers; i++) {
      layers.push({
        x: padding.left + layerSpacing * (i + 2),
        y: height / 2,
        layerIdx: i,
        label: `L${i}`,
        isTransformerLayer: true,
      })
    }

    // Output layer
    layers.push({
      x: padding.left + layerSpacing * (numLayers + 2),
      y: height / 2,
      layerIdx: -3,
      label: 'Output',
      isTransformerLayer: false,
    })

    layerNodesRef.current = layers

    // Draw connections between layers
    for (let i = 0; i < layers.length - 1; i++) {
      const from = layers[i]
      const to = layers[i + 1]

      // Multiple connection lines for visual depth
      for (let n = 0; n < nodesPerLayer; n++) {
        const yOffset = (n - (nodesPerLayer - 1) / 2) * 15

        ctx.beginPath()
        ctx.moveTo(from.x, from.y + yOffset)
        ctx.lineTo(to.x, to.y + yOffset)
        ctx.strokeStyle = lineColor
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    // Draw layer nodes
    layers.forEach(layer => {
      const isHovered = hoveredLayer?.layerIdx === layer.layerIdx
      const layerPatchInfo = layer.isTransformerLayer ? getLayerPatchInfo(layer.layerIdx) : []
      const hasPatch = layerPatchInfo.length > 0
      const isClickable = layer.isTransformerLayer && !disabled
      const firstPatchColors = hasPatch ? getPatchColors(layerPatchInfo[0]) : null

      // Draw outer glow for patched layers
      if (hasPatch && firstPatchColors) {
        const primaryColor = firstPatchColors.primary
        const gradient = ctx.createRadialGradient(
          layer.x, layer.y, nodeRadius * 0.5,
          layer.x, layer.y, nodeRadius * 2.5
        )
        gradient.addColorStop(0, `rgba(${primaryColor}, ${isLight ? 0.25 : 0.4})`)
        gradient.addColorStop(1, `rgba(${primaryColor}, 0)`)

        ctx.beginPath()
        ctx.arc(layer.x, layer.y, nodeRadius * 2.5, 0, Math.PI * 2)
        ctx.fillStyle = gradient
        ctx.fill()
      }

      // Draw main node
      const baseColor = hasPatch && firstPatchColors
        ? firstPatchColors.primary
        : isClickable
          ? defaultNodeColor
          : inactiveNodeColor

      const nodeGradient = ctx.createRadialGradient(
        layer.x - nodeRadius * 0.3, layer.y - nodeRadius * 0.3, 0,
        layer.x, layer.y, nodeRadius
      )
      nodeGradient.addColorStop(0, `rgba(${baseColor}, ${isHovered ? 1 : 0.9})`)
      nodeGradient.addColorStop(1, `rgba(${baseColor}, ${isHovered ? 0.8 : 0.6})`)

      ctx.beginPath()
      ctx.arc(layer.x, layer.y, isHovered ? nodeRadius * 1.15 : nodeRadius, 0, Math.PI * 2)
      ctx.fillStyle = nodeGradient
      ctx.fill()

      // Draw border
      ctx.strokeStyle = hasPatch && firstPatchColors
        ? `rgba(${firstPatchColors.secondary}, ${isHovered ? 1 : 0.8})`
        : isClickable
          ? borderDefault(isHovered)
          : borderInactive
      ctx.lineWidth = isHovered ? 2.5 : 1.5
      ctx.stroke()

      // Draw patch count badge if multiple patches
      if (layerPatchInfo.length > 1) {
        const badgeX = layer.x + nodeRadius * 0.7
        const badgeY = layer.y - nodeRadius * 0.7
        const badgeRadius = 8

        ctx.beginPath()
        ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'
        ctx.fill()

        ctx.fillStyle = 'rgba(255, 255, 255, 1)'
        ctx.font = 'bold 10px Inter, system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(layerPatchInfo.length.toString(), badgeX, badgeY)
      }

      // Draw label below node
      ctx.fillStyle = isHovered
        ? textPrimary(0.95)
        : hasPatch && firstPatchColors
          ? `rgba(${firstPatchColors.secondary}, 0.9)`
          : textPrimary(0.6)
      ctx.font = `${isHovered ? '600' : '500'} 11px Inter, system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(layer.label, layer.x, layer.y + nodeRadius + 8)

      // Draw clickable indicator for transformer layers
      if (isClickable && isHovered && !hasPatch) {
        ctx.fillStyle = textPrimary(0.7)
        ctx.font = 'bold 16px Inter, system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText('+', layer.x, layer.y)
      }
    })

    // Draw title
    ctx.fillStyle = textPrimary(0.7)
    ctx.font = '500 12px Inter, system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText('Click a layer to add a patch', padding.left, 15)

    // Draw legend
    const legendX = width - padding.right
    const legendY = 12
    ctx.textAlign = 'right'
    ctx.fillStyle = textPrimary(0.4)
    ctx.font = '400 10px Inter, system-ui, sans-serif'
    ctx.fillText('Hover for details', legendX, legendY)

  }, [dimensions, numLayers, hoveredLayer, patches, getLayerPatchInfo, disabled, isLight])

  // Handle mouse movement for hover detection
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    // getBoundingClientRect is affected by CSS zoom, so scale to canvas coordinates
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    // Check if hovering over any layer node
    const nodeRadius = 18
    let foundLayer: HoverInfo | null = null

    for (const layer of layerNodesRef.current) {
      if (!layer.isTransformerLayer) continue

      const dx = x - layer.x
      const dy = y - layer.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      if (distance <= nodeRadius * 1.5) {
        const layerPatches = getLayerPatches(layer.layerIdx)
        foundLayer = {
          layerIdx: layer.layerIdx,
          x: layer.x,
          y: layer.y,
          label: layer.label,
          patchCount: layerPatches.length,
          activePatches: getLayerPatchInfo(layer.layerIdx),
        }
        break
      }
    }

    setHoveredLayer(foundLayer)

    // Update cursor
    if (canvas) {
      canvas.style.cursor = foundLayer ? 'pointer' : 'default'
    }
  }, [disabled, getLayerPatches, getLayerPatchInfo])

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setHoveredLayer(null)
    // Don't dismiss the patch menu on mouse leave - let the backdrop handle closing it
  }, [])

  // Handle click to show patch menu
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled || !hoveredLayer) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    setShowPatchMenu({
      layerIdx: hoveredLayer.layerIdx,
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    })
  }, [disabled, hoveredLayer])

  // Handle adding a patch from menu
  const handleAddPatchType = useCallback((patchType: PatchType, interventionType: InterventionType = 'patch') => {
    if (!showPatchMenu) return

    const patch: PatchSpec = {
      layer: showPatchMenu.layerIdx,
      patch_type: patchType,
      intervention_type: interventionType,
      blend_factor: 1.0,
      positions: null,
      heads: null,
    }

    onAddPatch(patch)
    setShowPatchMenu(null)
  }, [showPatchMenu, onAddPatch])

  return (
    <div className="patch-selector" ref={containerRef}>
      <div className="selector-header">
        <Network className="header-icon" />
        <h3>Visual Layer Selector</h3>
      </div>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
        />

        {/* Hover tooltip */}
        {hoveredLayer && !showPatchMenu && (
          <div
            className="layer-tooltip"
            style={{
              left: hoveredLayer.x,
              top: hoveredLayer.y - 70,
            }}
          >
            <div className="tooltip-header">
              <span className="tooltip-layer">{hoveredLayer.label}</span>
              <span className="tooltip-heads">{numHeads} heads</span>
            </div>
            {hoveredLayer.patchCount > 0 ? (
              <div className="tooltip-patches">
                {hoveredLayer.activePatches.map((patchInfo, idx) => {
                  const colors = getPatchColors(patchInfo)
                  return (
                    <span
                      key={`${patchInfo.patchType}-${patchInfo.interventionType}-${idx}`}
                      className="patch-badge"
                      style={{
                        backgroundColor: `rgba(${colors.primary}, 0.2)`,
                        borderColor: `rgba(${colors.primary}, 0.5)`,
                        color: `rgba(${colors.secondary}, 1)`,
                      }}
                    >
                      {colors.label}
                    </span>
                  )
                })}
              </div>
            ) : (
              <div className="tooltip-hint">Click to add patch</div>
            )}
          </div>
        )}

        {/* Patch type selection menu */}
        {showPatchMenu && (
          <>
            <div className="menu-backdrop" onClick={() => setShowPatchMenu(null)} />
            <div
              className="patch-menu"
              style={{
                left: Math.min(showPatchMenu.x, dimensions.width - 200),
                top: showPatchMenu.y + 10,
              }}
            >
              <div className="menu-header">
                <Plus className="menu-icon" />
                <span>Add Intervention to L{showPatchMenu.layerIdx}</span>
              </div>

              {/* Patch section */}
              <div className="menu-section">
                <div className="section-label">
                  <Zap className="section-icon" />
                  <span>Patch from Source</span>
                </div>
                <div className="menu-items">
                  {Object.entries(PATCH_TYPE_COLORS).map(([type, colors]) => (
                    <button
                      key={type}
                      className="menu-item"
                      onClick={() => handleAddPatchType(type as PatchType, 'patch')}
                      style={{
                        '--patch-color': `rgb(${colors.primary})`,
                        '--patch-color-light': `rgb(${colors.secondary})`,
                      } as React.CSSProperties}
                    >
                      <span className="item-dot" />
                      <span className="item-label">{colors.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Ablation section */}
              <div className="menu-section">
                <div className="section-label">
                  <CircleOff className="section-icon" />
                  <span>Ablate</span>
                </div>
                <div className="menu-items">
                  {Object.entries(ABLATION_TYPE_COLORS).map(([ablationType, colors]) => (
                    <button
                      key={ablationType}
                      className="menu-item"
                      onClick={() => handleAddPatchType('residual' as PatchType, ablationType as InterventionType)}
                      style={{
                        '--patch-color': `rgb(${colors.primary})`,
                        '--patch-color-light': `rgb(${colors.secondary})`,
                      } as React.CSSProperties}
                    >
                      <span className="item-dot" />
                      <span className="item-label">{colors.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Active patches legend */}
      {patches.length > 0 && (
        <div className="patches-legend">
          <Eye className="legend-icon" />
          <span className="legend-label">Active:</span>
          <div className="legend-badges">
            {patches.map((patch, idx) => {
              const isAblation = patch.intervention_type !== 'patch'
              const colors = isAblation
                ? ABLATION_TYPE_COLORS[patch.intervention_type]
                : PATCH_TYPE_COLORS[patch.patch_type]
              const label = isAblation
                ? `${ABLATION_TYPE_COLORS[patch.intervention_type]?.label || patch.intervention_type}`
                : PATCH_TYPE_COLORS[patch.patch_type].label

              return (
                <span
                  key={idx}
                  className="legend-badge"
                  style={{
                    backgroundColor: `rgba(${colors.primary}, 0.15)`,
                    borderColor: `rgba(${colors.primary}, 0.4)`,
                    color: `rgba(${colors.secondary}, 1)`,
                  }}
                >
                  L{patch.layer} {label}
                </span>
              )
            })}
          </div>
        </div>
      )}

      <style>{`
        .patch-selector {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.1);
          margin-bottom: 1rem;
        }

        .selector-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.875rem 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .selector-header h3 {
          margin: 0;
          font-size: 0.85rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
          letter-spacing: 0.025em;
        }

        .header-icon {
          width: 16px;
          height: 16px;
          color: rgba(255, 255, 255, 0.5);
        }

        .canvas-container {
          position: relative;
          width: 100%;
        }

        .canvas-container canvas {
          display: block;
          width: 100%;
        }

        .layer-tooltip {
          position: absolute;
          transform: translateX(-50%);
          background: rgba(20, 20, 20, 0.98);
          border: 1px solid rgba(255, 255, 255, 0.15);
          padding: 0.625rem 0.875rem;
          pointer-events: none;
          z-index: 10;
          min-width: 120px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }

        .tooltip-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: 0.375rem;
        }

        .tooltip-layer {
          font-weight: 600;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.95);
        }

        .tooltip-heads {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .tooltip-patches {
          display: flex;
          flex-wrap: wrap;
          gap: 0.375rem;
        }

        .patch-badge {
          font-size: 0.7rem;
          padding: 0.125rem 0.375rem;
          border: 1px solid;
          font-weight: 500;
        }

        .tooltip-hint {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .menu-backdrop {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 15;
        }

        .patch-menu {
          position: absolute;
          background: rgba(20, 20, 20, 0.98);
          border: 1px solid rgba(255, 255, 255, 0.15);
          z-index: 20;
          min-width: 160px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        }

        .menu-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
        }

        .menu-icon {
          width: 14px;
          height: 14px;
          color: rgba(59, 130, 246, 0.8);
        }

        .menu-section {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .menu-section:first-of-type {
          border-top: none;
        }

        .section-label {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.5rem 0.75rem 0.25rem;
          font-size: 0.65rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.4);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .section-icon {
          width: 10px;
          height: 10px;
        }

        .menu-items {
          padding: 0.25rem 0;
        }

        .menu-item {
          display: flex;
          align-items: center;
          gap: 0.625rem;
          width: 100%;
          padding: 0.5rem 0.75rem;
          background: transparent;
          border: none;
          cursor: pointer;
          font-size: 0.8rem;
          color: rgba(255, 255, 255, 0.8);
          text-align: left;
          transition: all 0.15s;
        }

        .menu-item:hover {
          background: rgba(255, 255, 255, 0.05);
          color: rgba(255, 255, 255, 1);
        }

        .item-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--patch-color);
          box-shadow: 0 0 6px var(--patch-color);
        }

        .item-label {
          flex: 1;
        }

        .item-hint {
          font-size: 0.65rem;
          color: rgba(255, 255, 255, 0.35);
          margin-left: auto;
        }

        .patches-legend {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          flex-wrap: wrap;
        }

        .legend-icon {
          width: 14px;
          height: 14px;
          color: rgba(255, 255, 255, 0.4);
        }

        .legend-label {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.5);
          margin-right: 0.25rem;
        }

        .legend-badges {
          display: flex;
          flex-wrap: wrap;
          gap: 0.375rem;
        }

        .legend-badge {
          font-size: 0.7rem;
          padding: 0.125rem 0.5rem;
          border: 1px solid;
          font-weight: 500;
        }

        /* Light mode */
        [data-bg="light"] .patch-selector { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.1); }
        [data-bg="light"] .selector-header { border-bottom-color: rgba(0,0,0,0.08); }
        [data-bg="light"] .selector-header h3 { color: #1d1d1f; }
        [data-bg="light"] .header-icon { color: rgba(0,0,0,0.45); }
        [data-bg="light"] .layer-tooltip { background: rgba(255,255,255,0.97); border-color: rgba(0,0,0,0.12); box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        [data-bg="light"] .tooltip-layer { color: #1d1d1f; }
        [data-bg="light"] .tooltip-heads { color: rgba(0,0,0,0.45); }
        [data-bg="light"] .tooltip-hint { color: rgba(0,0,0,0.5); }
        [data-bg="light"] .patch-menu { background: rgba(255,255,255,0.98); border-color: rgba(0,0,0,0.12); box-shadow: 0 8px 32px rgba(0,0,0,0.12); }
        [data-bg="light"] .menu-header { color: #1d1d1f; border-bottom-color: rgba(0,0,0,0.08); }
        [data-bg="light"] .menu-section { border-top-color: rgba(0,0,0,0.06); }
        [data-bg="light"] .section-label { color: rgba(0,0,0,0.45); }
        [data-bg="light"] .menu-item { color: rgba(0,0,0,0.7); }
        [data-bg="light"] .menu-item:hover { background: rgba(0,0,0,0.04); color: #1d1d1f; }
        [data-bg="light"] .legend-icon { color: rgba(0,0,0,0.45); }
        [data-bg="light"] .legend-label { color: rgba(0,0,0,0.55); }
        [data-bg="light"] .patches-legend { border-top-color: rgba(0,0,0,0.08); }
      `}</style>
    </div>
  )
}

export default PatchSelector
