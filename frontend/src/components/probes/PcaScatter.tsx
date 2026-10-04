/**
 * The examples' activations at the selected layer, projected onto their top two principal
 * components and colored by label. Out-of-distribution examples (projected onto the same
 * components) are rings, so a shift in where they land is visible at a glance.
 */

import { useState, type PointerEvent } from 'react'
import type { Pooling, ProbeExample, ProbeRunResult } from '../../types/probes'
import { ChartTooltip } from './chartKit'
import { useElementWidth, useProbeColors } from './probeTheme'

interface PcaScatterProps {
  run: ProbeRunResult
  pooling: Pooling
  position: number
  showOod: boolean
}

interface Point {
  x: number
  y: number
  example: ProbeExample
  ood: boolean
}

const HEIGHT = 300
const MARGIN = { top: 12, right: 12, bottom: 34, left: 40 }
// Pointer distance (px) within which the nearest point is hovered
const HOVER_RADIUS = 16

export function PcaScatter({ run, pooling, position, showOod }: PcaScatterProps) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const [hover, setHover] = useState<{ point: Point; px: number; py: number } | null>(null)

  const result = run.poolings[pooling]
  const coords = result.pca.values[position] ?? []
  const oodCoords = showOod && result.pca_ood ? result.pca_ood.values[position] ?? [] : []
  const variance = result.pca_variance.values[position] ?? [0, 0]

  const points: Point[] = [
    ...coords.map(([x, y], i) => ({ x, y, example: run.pca_examples[i], ood: false })),
    ...oodCoords.map(([x, y], i) => ({ x, y, example: run.pca_ood_examples[i], ood: true })),
  ]

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 50)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const [x0, x1] = [Math.min(...xs, 0), Math.max(...xs, 0)]
  const [y0, y1] = [Math.min(...ys, 0), Math.max(...ys, 0)]
  // One scale for both axes, so distances mean the same in every direction
  const scale = Math.min(innerW / ((x1 - x0) * 1.08 || 1), innerH / ((y1 - y0) * 1.08 || 1))
  const cx = (x0 + x1) / 2
  const cy = (y0 + y1) / 2
  const px = (x: number) => MARGIN.left + innerW / 2 + (x - cx) * scale
  const py = (y: number) => MARGIN.top + innerH / 2 - (y - cy) * scale

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const mx = e.clientX - box.left
    const my = e.clientY - box.top
    let best: Point | null = null
    let bestDist = HOVER_RADIUS ** 2
    for (const p of points) {
      const d = (px(p.x) - mx) ** 2 + (py(p.y) - my) ** 2
      if (d < bestDist) {
        bestDist = d
        best = p
      }
    }
    setHover(best ? { point: best, px: mx, py: my } : null)
  }

  const names = run.dataset.label_names

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={HEIGHT} role="img" aria-label={`PCA of activations at ${position === 0 ? 'the embeddings' : `layer ${position - 1}`}, colored by label`}>
        <rect x={MARGIN.left} y={MARGIN.top} width={innerW} height={innerH} fill="none" stroke={colors.grid} />
        <line x1={px(0)} x2={px(0)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.grid} />
        <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={py(0)} y2={py(0)} stroke={colors.grid} />
        {points.map((p, i) =>
          p.ood ? (
            <circle key={i} cx={px(p.x)} cy={py(p.y)} r={4} fill={colors.surface} stroke={colors.classes[p.example.label]} strokeWidth={2} />
          ) : (
            <circle key={i} cx={px(p.x)} cy={py(p.y)} r={4} fill={colors.classes[p.example.label]} stroke={colors.surface} strokeWidth={1.5} />
          ),
        )}
        {hover && (
          <circle cx={px(hover.point.x)} cy={py(hover.point.y)} r={7} fill="none" stroke={colors.text} strokeWidth={1.5} />
        )}
        <text x={MARGIN.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
          PC1 ({(variance[0] * 100).toFixed(1)}% of variance)
        </text>
        <text x={12} y={MARGIN.top + innerH / 2} textAnchor="middle" fontSize={11} fill={colors.text} transform={`rotate(-90 12 ${MARGIN.top + innerH / 2})`}>
          PC2 ({(variance[1] * 100).toFixed(1)}%)
        </text>
        <rect
          x={MARGIN.left}
          y={MARGIN.top}
          width={innerW}
          height={innerH}
          fill="transparent"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          rows={[
            {
              key: colors.classes[hover.point.example.label],
              label: hover.point.ood ? 'OOD example' : `${hover.point.example.split ?? ''} example`,
              value: names[hover.point.example.label],
            },
          ]}
        >
          <div style={{ marginTop: '0.3rem', whiteSpace: 'normal', wordBreak: 'break-word' }}>{hover.point.example.text}</div>
        </ChartTooltip>
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.classes[1] }} />
          {names[1]}
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-dot" style={{ background: colors.classes[0] }} />
          {names[0]}
        </span>
        {oodCoords.length > 0 && (
          <span className="pl-legend-item">
            <span className="pl-legend-ring" style={{ borderColor: colors.text }} />
            OOD
          </span>
        )}
      </div>
    </div>
  )
}
