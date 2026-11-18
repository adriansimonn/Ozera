/**
 * Visual layer selector for activation patching.
 * Displays a network diagram where users can click layers to add patches.
 * Reuses patterns from GenerationFlow visualization.
 */

import { useRef, useEffect, useState, useCallback } from 'react'
import type { PatchSpec, PatchType, InterventionType, ModelLayerInfo } from '../../types/patching'
import { useAppPalette } from '../../hooks/useTheme'
import { MONO } from '../home/homeKit'

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
  const palette = useAppPalette()
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

    // The canvas is drawn in the app's colors (styles/appTheme.ts)
    const bgColor = palette.bg
    const lineColor = palette.rule
    const textPrimary = palette.fg

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

      // A layer is a square cell: outlined where a patch can go, filled in its patch's color
      // once it has one, and a small solid cell where nothing can be patched
      const side = isClickable || hasPatch ? nodeRadius * 2 : nodeRadius
      const grown = isHovered && isClickable ? side + 4 : side
      const left = Math.round(layer.x - grown / 2)
      const top = Math.round(layer.y - grown / 2)

      ctx.fillStyle = hasPatch && firstPatchColors
        ? `rgb(${firstPatchColors.primary})`
        : isClickable
          ? (isHovered ? palette.fillStrong : bgColor)
          : palette.fgFaint
      if (isClickable && !hasPatch) {
        // Cover the lines running behind an empty cell
        ctx.fillStyle = bgColor
        ctx.fillRect(left, top, grown, grown)
        ctx.fillStyle = isHovered ? palette.fillStrong : bgColor
      }
      ctx.fillRect(left, top, grown, grown)

      if (isClickable || hasPatch) {
        ctx.strokeStyle = isHovered ? palette.fg : hasPatch ? 'transparent' : palette.ruleStrong
        ctx.lineWidth = 1
        ctx.strokeRect(left + 0.5, top + 0.5, grown - 1, grown - 1)
      }

      // How many patches, when a layer has more than one
      if (layerPatchInfo.length > 1) {
        const badge = 16
        const badgeX = left + grown - badge / 2
        const badgeY = top - badge / 2
        ctx.fillStyle = palette.fg
        ctx.fillRect(badgeX, badgeY, badge, badge)
        ctx.fillStyle = palette.bg
        ctx.font = `500 11px ${MONO}`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(layerPatchInfo.length.toString(), badgeX + badge / 2, badgeY + badge / 2 + 0.5)
      }

      // Draw label below node
      ctx.fillStyle = isHovered || hasPatch ? textPrimary : palette.fgDim
      ctx.font = `500 12px ${MONO}`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(layer.label, layer.x, layer.y + nodeRadius + 10)

      // Draw clickable indicator for transformer layers
      if (isClickable && isHovered && !hasPatch) {
        ctx.fillStyle = textPrimary
        ctx.font = '400 18px Inter, system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText('+', layer.x, layer.y)
      }
    })

    // Draw title
    ctx.fillStyle = palette.fgDim
    ctx.font = '400 13.5px Inter, system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText('Click a layer to add a patch', padding.left, 15)

    // Draw legend
    const legendX = width - padding.right
    const legendY = 12
    ctx.textAlign = 'right'
    ctx.fillStyle = palette.fgDim
    ctx.font = '400 13.5px Inter, system-ui, sans-serif'
    ctx.fillText('Hover for details', legendX, legendY)

  }, [dimensions, numLayers, hoveredLayer, patches, getLayerPatchInfo, disabled, palette])

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
      <h2 className="oz-title">Visual layer selector</h2>

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
                    <span key={`${patchInfo.patchType}-${patchInfo.interventionType}-${idx}`} className="oz-tag">
                      <i className="patch-key" style={{ background: `rgb(${colors.primary})` }} />
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
              className="patch-menu oz-menu"
              style={{
                left: Math.min(showPatchMenu.x, dimensions.width - 200),
                top: showPatchMenu.y + 10,
              }}
            >
              <div className="menu-header">Add intervention to L{showPatchMenu.layerIdx}</div>

              {/* Patch section */}
              <div className="menu-section">
                <div className="section-label">Patch from source</div>
                <div className="menu-items">
                  {Object.entries(PATCH_TYPE_COLORS).map(([type, colors]) => (
                    <button
                      key={type}
                      className="oz-menu__item"
                      onClick={() => handleAddPatchType(type as PatchType, 'patch')}
                    >
                      <i className="patch-key" style={{ background: `rgb(${colors.primary})` }} />
                      <span className="item-label">{colors.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Ablation section */}
              <div className="menu-section">
                <div className="section-label">Ablate</div>
                <div className="menu-items">
                  {Object.entries(ABLATION_TYPE_COLORS).map(([ablationType, colors]) => (
                    <button
                      key={ablationType}
                      className="oz-menu__item"
                      onClick={() => handleAddPatchType('residual' as PatchType, ablationType as InterventionType)}
                    >
                      <i className="patch-key" style={{ background: `rgb(${colors.primary})` }} />
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
          <span className="legend-label">Active</span>
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
                <span key={idx} className="oz-tag">
                  <i className="patch-key" style={{ background: `rgb(${colors.primary})` }} />
                  L{patch.layer} {label}
                </span>
              )
            })}
          </div>
        </div>
      )}

      <style>{`
        .patch-selector .canvas-container {
          position: relative;
        }

        .patch-selector canvas {
          display: block;
          width: 100%;
          cursor: pointer;
        }

        /* A patch's kind, keyed by a cell of its color */
        .patch-selector .patch-key {
          flex: none;
          width: 8px;
          height: 8px;
        }

        .patch-selector .layer-tooltip {
          position: absolute;
          z-index: 10;
          transform: translateX(-50%);
          padding: 0.5rem 0.7rem;
          background: var(--oz-bg);
          border: 1px solid var(--oz-rule-strong);
          font-size: 0.8rem;
          white-space: nowrap;
          pointer-events: none;
        }

        .patch-selector .tooltip-header {
          display: flex;
          align-items: baseline;
          gap: 0.75rem;
        }

        .patch-selector .tooltip-layer {
          font-weight: 500;
        }

        .patch-selector .tooltip-heads,
        .patch-selector .tooltip-hint {
          color: var(--oz-fg-dim);
        }

        .patch-selector .tooltip-patches {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-top: 0.4rem;
        }

        .patch-selector .menu-backdrop {
          position: fixed;
          inset: 0;
          z-index: 40;
        }

        /* What to add at the layer that was clicked */
        .patch-selector .patch-menu {
          position: absolute;
          z-index: 50;
          min-width: 210px;
        }

        .patch-selector .menu-header {
          padding: 0.45rem 0.75rem 0.6rem;
          font-size: 0.9375rem;
          font-weight: 500;
          letter-spacing: -0.015em;
        }

        .patch-selector .menu-section {
          padding-top: 6px;
          margin-top: 6px;
          border-top: 1px solid var(--oz-rule);
        }

        .patch-selector .section-label {
          padding: 0.3rem 0.75rem 0.25rem;
          color: var(--oz-fg-dim);
          font-size: 0.78rem;
          font-weight: 500;
        }

        .patch-selector .oz-menu__item {
          padding: 0.4rem 0.75rem;
        }

        .patch-selector .patches-legend {
          display: flex;
          align-items: baseline;
          gap: 0.75rem;
          margin-top: 0.9rem;
        }

        .patch-selector .legend-label {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }

        .patch-selector .legend-badges {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
      `}</style>
    </div>
  )
}

export default PatchSelector
