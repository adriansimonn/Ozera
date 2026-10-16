/**
 * The Similarity view: cosine similarity between the centered class means at the selected
 * layer, in value order. A circle or helix shows as bands that fall off and come back
 * ("ringing"); the profile beside it averages the cosine at each distance in the order.
 */

import { useMemo, useRef, useState, type PointerEvent } from 'react'
import { Download, Grid3x3 } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRun } from '../../types/geometry'
import { ChartTooltip, Placeholder } from '../lab/chartKit'
import { divergingColor, useElementWidth, useLabColors } from '../lab/labTheme'
import { cosineMatrix, downloadSvg, layerName } from './geometryUtils'

const MARGIN = { top: 12, right: 12, bottom: 56, left: 64 }
const MAX_SIZE = 560
const PROFILE_HEIGHT = 220
const PROFILE_MARGIN = { top: 16, right: 16, bottom: 36, left: 40 }

/** Mean cosine between values `offset` apart in the order (round the cycle for cyclic concepts). */
function cosineProfile(cos: number[][], cyclic: boolean): { offset: number; value: number }[] {
  const k = cos.length
  const sums = new Array(k).fill(0)
  const counts = new Array(k).fill(0)
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const d = cyclic ? Math.min(j - i, k - (j - i)) : j - i
      sums[d] += cos[i][j]
      counts[d]++
    }
  }
  return sums.map((sum, d) => ({ offset: d, value: sum / counts[d] })).filter((p) => p.offset > 0 && counts[p.offset] > 0)
}

export function SimilarityMatrix({ lab }: { lab: GeometryLab }) {
  if (!lab.run) {
    return (
      <section className="pl-panel">
        <Placeholder icon={Grid3x3}>{lab.running ? 'Reading every layer…' : 'Run a concept to compare its class means.'}</Placeholder>
      </section>
    )
  }
  return <SimilarityBody lab={lab} run={lab.run} />
}

function SimilarityBody({ lab, run }: { lab: GeometryLab; run: GeometryRun }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(520)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ i: number; j: number; px: number; py: number } | null>(null)
  const { position } = lab
  const values = run.concept.values
  const k = values.length
  const cos = useMemo(() => cosineMatrix(run.means.values[position]), [run, position])
  const cyclic = run.concept.structure === 'cyclic'
  const profile = run.concept.structure === 'categorical' ? [] : cosineProfile(cos, cyclic)

  const size = Math.max(Math.min(width - MARGIN.left - MARGIN.right, MAX_SIZE), 100)
  const cell = size / k
  const labelEvery = Math.max(1, Math.ceil(k / (size / 14)))

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const mx = e.clientX - box.left - MARGIN.left
    const my = e.clientY - box.top - MARGIN.top
    const i = Math.floor(my / cell)
    const j = Math.floor(mx / cell)
    setHover(i >= 0 && j >= 0 && i < k && j < k ? { i, j, px: e.clientX - box.left, py: e.clientY - box.top } : null)
  }
  const filename = `${run.concept.name}-${run.model}-${layerName(position)}-cosine`.replace(/[^\w.-]+/g, '_')

  return (
    <div className="gl-similarity" style={{ opacity: lab.running ? 0.55 : 1 }}>
      <section className="pl-panel">
        <div className="pl-controls-row">
          <div className="gl-slider">
            <div className="pl-control-label">{layerName(position)}</div>
            <input
              type="range"
              min={0}
              max={run.num_layers}
              value={position}
              onChange={(e) => lab.setPosition(Number(e.target.value))}
              aria-label="Layer"
              aria-valuetext={layerName(position)}
            />
          </div>
        </div>
      </section>

      <div className="pl-results-grid" style={{ alignItems: 'start' }}>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title="Cosine similarity between the class means after removing their mean">Cosine · {layerName(position)}</h3>
            <button
              type="button"
              className="pl-icon-button"
              title="Download SVG"
              aria-label="Download SVG"
              onClick={() => svgRef.current && downloadSvg(svgRef.current, `${filename}.svg`, colors.surface)}
              style={{ color: colors.text }}
            >
              <Download className="pl-icon" />
            </button>
          </div>
          <div ref={ref} className="pl-chart">
            <svg ref={svgRef} width={size + MARGIN.left + MARGIN.right} height={size + MARGIN.top + MARGIN.bottom} role="img" aria-label={`Cosine similarity of ${k} class means at ${layerName(position)}`}>
              <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                {cos.map((row, i) =>
                  row.map((value, j) => (
                    <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell + 0.5} height={cell + 0.5} fill={divergingColor(colors, value)} />
                  )),
                )}
                {hover && <rect x={hover.j * cell} y={hover.i * cell} width={cell} height={cell} fill="none" stroke={colors.text} strokeWidth={1.5} />}
                {values.map((v, i) =>
                  i % labelEvery === 0 ? (
                    <g key={v}>
                      <text x={-6} y={(i + 0.5) * cell} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
                        {v}
                      </text>
                      <text x={(i + 0.5) * cell} y={size + 8} textAnchor="end" fontSize={10} fill={colors.text} transform={`rotate(-45 ${(i + 0.5) * cell} ${size + 8})`}>
                        {v}
                      </text>
                    </g>
                  ) : null,
                )}
              </g>
              <rect x={MARGIN.left} y={MARGIN.top} width={size} height={size} fill="transparent" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)} />
            </svg>
            {hover && (
              <ChartTooltip
                x={hover.px}
                y={hover.py}
                containerWidth={width}
                title={`${values[hover.i]} · ${values[hover.j]}`}
                rows={[{ label: 'cosine', value: cos[hover.i][hover.j].toFixed(3) }]}
              />
            )}
          </div>
          <div className="gl-ramp-legend">
            <span>−1</span>
            <span
              className="gl-ramp"
              style={{
                background: `linear-gradient(to right, ${divergingColor(colors, -1)}, ${divergingColor(colors, 0)}, ${divergingColor(colors, 1)})`,
                borderColor: colors.border,
              }}
            />
            <span>1</span>
          </div>
        </section>

        {profile.length > 1 && (
          <section className="pl-panel">
            <div className="pl-card-title">
              <h3 title={cyclic ? 'Mean cosine between values this far apart round the cycle' : 'Mean cosine between values this far apart in the order'}>
                Cosine by distance
              </h3>
            </div>
            <Profile profile={profile} color={colors.series[0]} />
          </section>
        )}
      </div>

    </div>
  )
}

function Profile({ profile, color }: { profile: { offset: number; value: number }[]; color: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(360)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const innerW = Math.max(width - PROFILE_MARGIN.left - PROFILE_MARGIN.right, 50)
  const innerH = PROFILE_HEIGHT - PROFILE_MARGIN.top - PROFILE_MARGIN.bottom
  const n = profile.length
  const x = (i: number) => PROFILE_MARGIN.left + (n > 1 ? (i / (n - 1)) * innerW : innerW / 2)
  const y = (v: number) => PROFILE_MARGIN.top + ((1 - v) / 2) * innerH
  const path = profile.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join('')
  const labelEvery = Math.max(1, Math.ceil(n / (innerW / 28)))

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={PROFILE_HEIGHT} role="img" aria-label="Mean cosine by distance in the order">
        {[-1, -0.5, 0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PROFILE_MARGIN.left} x2={PROFILE_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
            <text x={PROFILE_MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
              {t}
            </text>
          </g>
        ))}
        {profile.map((p, i) =>
          i % labelEvery === 0 || i === n - 1 ? (
            <text key={p.offset} x={x(i)} y={PROFILE_MARGIN.top + innerH + 14} textAnchor="middle" fontSize={10} fill={colors.text}>
              {p.offset}
            </text>
          ) : null,
        )}
        <text x={PROFILE_MARGIN.left + innerW / 2} y={PROFILE_HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
          Distance in order
        </text>
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
        {hover && <circle cx={x(hover.index)} cy={y(profile[hover.index].value)} r={4} fill={color} stroke={colors.surface} strokeWidth={2} />}
        <rect
          x={PROFILE_MARGIN.left}
          y={PROFILE_MARGIN.top}
          width={innerW}
          height={innerH}
          fill="transparent"
          onPointerMove={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
            const mx = e.clientX - box.left
            const index = Math.max(0, Math.min(n - 1, Math.round(((mx - PROFILE_MARGIN.left) / innerW) * (n - 1))))
            setHover({ index, px: mx, py: e.clientY - box.top })
          }}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          title={`${profile[hover.index].offset} apart`}
          rows={[{ key: color, label: 'mean cosine', value: profile[hover.index].value.toFixed(3) }]}
        />
      )}
    </div>
  )
}
