/**
 * How the selected probe's scores separate the classes: a mirrored histogram (positive class
 * above the axis, negative below) and the ROC curve.
 */

import { useState, type PointerEvent } from 'react'
import { ChartTooltip } from './chartKit'
import { useElementWidth, useProbeColors } from './probeTheme'
import { classHistogram, formatPercent, formatScore, rocCurve } from './probeUtils'

interface ScoreViewProps {
  scores: number[]
  labels: number[]
  labelNames: [string, string]
  color: string
}

/** A bar path with a 4px rounded data end (top when growing up, bottom when growing down). */
function barPath(x: number, base: number, w: number, h: number, up: boolean): string {
  const r = Math.min(4, w / 2, h)
  if (up) {
    const top = base - h
    return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`
  }
  const bottom = base + h
  return `M${x},${base}V${bottom - r}Q${x},${bottom} ${x + r},${bottom}H${x + w - r}Q${x + w},${bottom} ${x + w},${bottom - r}V${base}Z`
}

export function ScoreHistogram({ scores, labels, labelNames }: ScoreViewProps) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const [hover, setHover] = useState<{ bin: number; px: number; py: number } | null>(null)

  const height = 240
  const margin = { top: 12, right: 12, bottom: 34, left: 36 }
  const innerW = Math.max(width - margin.left - margin.right, 50)
  const innerH = height - margin.top - margin.bottom
  const numBins = Math.max(10, Math.min(36, Math.round(Math.sqrt(scores.length) * 2.5)))
  const bins = classHistogram(scores, labels, numBins)
  if (bins.length === 0) return <div style={{ fontSize: '0.8rem' }}>No scores to show.</div>

  const lo = bins[0].x0
  const hi = bins[bins.length - 1].x1
  const x = (v: number) => margin.left + ((v - lo) / (hi - lo)) * innerW
  const base = margin.top + innerH / 2
  const maxCount = Math.max(1, ...bins.flatMap((b) => b.counts))
  const h = (count: number) => (count / maxCount) * (innerH / 2 - 4)
  const binW = innerW / bins.length
  const barW = Math.min(24, Math.max(binW - 2, 1))
  const ticks = [lo, lo / 2, 0, hi / 2, hi].filter((v, i, arr) => arr.indexOf(v) === i)

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    const bin = Math.max(0, Math.min(bins.length - 1, Math.floor((px - margin.left) / binW)))
    setHover({ bin, px, py: e.clientY - box.top })
  }

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={height} role="img" aria-label={`Score histogram: ${labelNames[1]} above the axis, ${labelNames[0]} below`}>
        {bins.map((b, i) => {
          const bx = margin.left + i * binW + (binW - barW) / 2
          return (
            <g key={i} opacity={hover && hover.bin !== i ? 0.75 : 1}>
              {b.counts[1] > 0 && <path d={barPath(bx, base - 1, barW, h(b.counts[1]), true)} fill={colors.classes[1]} />}
              {b.counts[0] > 0 && <path d={barPath(bx, base + 1, barW, h(b.counts[0]), false)} fill={colors.classes[0]} />}
            </g>
          )
        })}
        <line x1={margin.left} x2={margin.left + innerW} y1={base} y2={base} stroke={colors.axis} />
        <line x1={x(0)} x2={x(0)} y1={margin.top} y2={margin.top + innerH} stroke={colors.baseline} strokeWidth={1} />
        <text x={x(0) + 4} y={margin.top + 10} fontSize={10} fill={colors.text}>
          threshold
        </text>
        <text x={margin.left - 6} y={margin.top + 10} textAnchor="end" fontSize={10} fill={colors.text}>
          {maxCount}
        </text>
        <text x={margin.left - 6} y={margin.top + innerH} textAnchor="end" fontSize={10} fill={colors.text}>
          {maxCount}
        </text>
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={height - 18} textAnchor="middle" fontSize={10} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
            {formatScore(t)}
          </text>
        ))}
        <text x={margin.left + innerW / 2} y={height - 3} textAnchor="middle" fontSize={11} fill={colors.text}>
          Probe score
        </text>
        <rect
          x={margin.left}
          y={margin.top}
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
          title={`Score ${formatScore(bins[hover.bin].x0)} to ${formatScore(bins[hover.bin].x1)}`}
          rows={[
            { key: colors.classes[1], label: labelNames[1], value: String(bins[hover.bin].counts[1]) },
            { key: colors.classes[0], label: labelNames[0], value: String(bins[hover.bin].counts[0]) },
          ]}
        />
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.classes[1] }} />
          {labelNames[1]}
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.classes[0] }} />
          {labelNames[0]}
        </span>
      </div>
    </div>
  )
}

export function RocCurve({ scores, labels, color }: ScoreViewProps) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)

  const points = rocCurve(scores, labels)
  if (points.length === 0) return <div style={{ fontSize: '0.8rem' }}>Both classes are needed for a ROC curve.</div>

  const height = 240
  const margin = { top: 12, right: 12, bottom: 34, left: 40 }
  const innerH = height - margin.top - margin.bottom
  const side = Math.min(innerH, Math.max(width - margin.left - margin.right, 50))
  const left = margin.left + Math.max(0, (width - margin.left - margin.right - side) / 2)
  const x = (v: number) => left + v * side
  const y = (v: number) => margin.top + (1 - v) * side
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.fpr).toFixed(1)},${y(p.tpr).toFixed(1)}`).join('')

  // Where the probe operates: positive when its score is above 0
  const positives = labels.filter((l) => l === 1).length
  const negatives = labels.length - positives
  const tpr0 = scores.filter((s, i) => s > 0 && labels[i] === 1).length / positives
  const fpr0 = scores.filter((s, i) => s > 0 && labels[i] === 0).length / negatives

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    const fpr = Math.max(0, Math.min(1, (px - margin.left) / side))
    let index = 0
    points.forEach((p, i) => {
      if (Math.abs(p.fpr - fpr) < Math.abs(points[index].fpr - fpr)) index = i
    })
    setHover({ index, px, py: e.clientY - box.top })
  }

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={height} role="img" aria-label="ROC curve">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke={colors.grid} />
            <text x={x(0) - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
              {t}
            </text>
            <text x={x(t)} y={y(0) + 14} textAnchor="middle" fontSize={10} fill={colors.text}>
              {t}
            </text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={colors.baseline} strokeWidth={1} />
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={x(fpr0)} cy={y(tpr0)} r={4} fill={color} stroke={colors.surface} strokeWidth={2} />
        <text x={x(fpr0) + 8} y={y(tpr0) + 14} fontSize={10} fill={colors.text} stroke={colors.surface} strokeWidth={3} paintOrder="stroke">
          threshold 0
        </text>
        <text x={x(0.5)} y={height - 3} textAnchor="middle" fontSize={11} fill={colors.text}>
          False positive rate
        </text>
        <text x={left - 28} y={y(0.5)} textAnchor="middle" fontSize={11} fill={colors.text} transform={`rotate(-90 ${left - 28} ${y(0.5)})`}>
          True positive rate
        </text>
        {hover && <circle cx={x(points[hover.index].fpr)} cy={y(points[hover.index].tpr)} r={4} fill="none" stroke={colors.text} strokeWidth={1.5} />}
        <rect
          x={x(0)}
          y={y(1)}
          width={side}
          height={side}
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
            { key: color, label: 'true positive rate', value: formatPercent(points[hover.index].tpr) },
            { label: 'false positive rate', value: formatPercent(points[hover.index].fpr) },
            {
              label: 'threshold',
              value: Number.isFinite(points[hover.index].threshold) ? formatScore(points[hover.index].threshold) : '∞',
            },
          ]}
        />
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          At threshold 0: TPR {formatPercent(tpr0)} · FPR {formatPercent(fpr0)}
        </span>
      </div>
    </div>
  )
}
