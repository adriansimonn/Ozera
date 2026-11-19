/**
 * Logit attribution: what each part of the model wrote toward one token's logit.
 *
 * For the selected position, the logit of the token that came next is split into the
 * embeddings' share and each layer's attention and MLP share (bars by layer), and each
 * attention output into its heads (the layer × head grid). The shares are direct effects, read
 * with the final norm's scale frozen, so they add up to the logit.
 */

import { memo, useMemo, useState, type PointerEvent } from 'react'
import * as d3 from 'd3'
import type { LogitAttribution as LogitAttributionData } from '../../types/model'
import { ExportButton } from '../export'
import { ChartTooltip, LabKitStyles, StatTiles } from '../lab/chartKit'
import { divergingColor, inkOn, pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import { percent, showToken, signed } from './tokenText'

interface LogitAttributionProps {
  attribution: LogitAttributionData
  /** The sequence's decoded tokens */
  tokens: string[]
  position: number
  activationId?: string
}

/** One position's shares. */
interface Shares {
  embed: number
  bias: number
  attn: number[]
  mlp: number[]
  heads: number[][]
  attnBias: number[]
  logit: number
  unexplained: number
}

const BARS_HEIGHT = 250
const BARS_MARGIN = { top: 12, right: 12, bottom: 34, left: 44 }
const HEADS_MARGIN = { top: 22, right: 12, bottom: 8, left: 44 }
const LARGEST = 6

function sharesAt(attribution: LogitAttributionData, position: number): Shares {
  const embed = (attribution.embed.values as number[])[position]
  const bias = (attribution.bias.values as number[])[position]
  const attn = (attribution.attn.values as number[][])[position]
  const mlp = (attribution.mlp.values as number[][])[position]
  const logit = (attribution.logit.values as number[])[position]
  const total = embed + bias + d3.sum(attn) + d3.sum(mlp)
  return {
    embed,
    bias,
    attn,
    mlp,
    heads: (attribution.heads.values as number[][][])[position],
    attnBias: (attribution.attn_bias.values as number[][])[position],
    logit,
    unexplained: logit - total,
  }
}

export const LogitAttribution = memo(function LogitAttribution({ attribution, tokens, position, activationId }: LogitAttributionProps) {
  const at = Math.max(0, Math.min(attribution.seq_len - 1, position))
  const shares = useMemo(() => sharesAt(attribution, at), [attribution, at])
  // Only models whose output projections have a bias (Ozera's) carry a share no head wrote
  const hasAttnBias = useMemo(
    () => (attribution.attn_bias.values as number[][]).some((layers) => layers.some((v) => Math.abs(v) > 1e-9)),
    [attribution],
  )
  const hasBias = useMemo(() => (attribution.bias.values as number[]).some((v) => Math.abs(v) > 1e-9), [attribution])

  const target = attribution.target_ids[at]
  const isLast = at === attribution.seq_len - 1
  const largest = useMemo(() => {
    const parts: { name: string; value: number }[] = [{ name: 'Emb', value: shares.embed }]
    shares.mlp.forEach((value, layer) => parts.push({ name: `L${layer} MLP`, value }))
    shares.heads.forEach((heads, layer) => heads.forEach((value, head) => parts.push({ name: `L${layer}H${head}`, value })))
    return parts.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, LARGEST)
  }, [shares])

  return (
    <div className="logit-attribution lab-kit">
      <div className="la-head">
        <div className="la-pair">
          <span className="la-index">{at}</span>
          <span className="la-token">{showToken(tokens[at])}</span>
          <span className="la-arrow" aria-label="predicts">
            →
          </span>
          <span className="la-token la-target">{showToken(attribution.tokens[String(target)])}</span>
          {isLast && <span className="pl-note">top prediction</span>}
        </div>
        {activationId && <ExportButton exportType="logit-attribution" activationId={activationId} position={at} positions={attribution.seq_len} />}
      </div>

      <div className="la-summary">
        <StatTiles
          tiles={[
            { label: 'Logit', value: shares.logit.toFixed(2) },
            { label: 'Probability', value: percent((attribution.prob.values as number[])[at]) },
            { label: 'Rank', value: String((attribution.rank.values as number[])[at]) },
            {
              label: 'Unexplained',
              value: signed(shares.unexplained),
              title: 'The logit minus the sum of the parts: rounding in half-precision models',
            },
          ]}
        />
        <div className="la-largest">
          <span className="la-largest-label">Largest</span>
          {largest.map((part) => (
            <span key={part.name} className="oz-tag la-part">
              {part.name}
              <strong>{signed(part.value)}</strong>
            </span>
          ))}
        </div>
      </div>

      <div className="la-panels">
        <section className="la-panel">
          <LayerBars shares={shares} showBias={hasBias} />
        </section>
        <section className="la-panel">
          <HeadGrid shares={shares} numHeads={attribution.num_heads} showBias={hasAttnBias} />
        </section>
      </div>

      <LabKitStyles />
      <style>{`
        .logit-attribution { padding: 0; }

        .logit-attribution .la-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.9rem var(--oz-gutter);
          border-bottom: 1px solid var(--oz-rule);
        }
        .logit-attribution .la-pair { display: flex; align-items: center; flex-wrap: wrap; gap: 0.6rem; min-width: 0; }
        .logit-attribution .la-index { color: var(--oz-fg-faint); font-family: var(--oz-mono); font-size: 0.75rem; }
        .logit-attribution .la-token {
          max-width: 18rem;
          padding: 0.2rem 0.5rem;
          border: 1px solid var(--oz-rule-strong);
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .logit-attribution .la-target { background: var(--oz-fg); border-color: var(--oz-fg); color: var(--oz-bg); }
        .logit-attribution .la-arrow { color: var(--oz-fg-dim); }

        .logit-attribution .la-summary { padding: 1.2rem var(--oz-gutter) 1.1rem; border-bottom: 1px solid var(--oz-rule); }
        .logit-attribution .pl-tiles { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 1rem; }
        .logit-attribution .la-largest { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
        .logit-attribution .la-largest-label { margin-right: 0.4rem; color: var(--oz-fg-dim); font-size: 0.85rem; font-weight: 500; }
        .logit-attribution .la-part { font-family: var(--oz-mono); font-size: 0.78rem; gap: 0.5rem; }
        .logit-attribution .la-part strong { font-weight: 500; font-variant-numeric: tabular-nums; }

        .logit-attribution .la-panels { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); }
        .logit-attribution .la-panel { min-width: 0; padding: 1.2rem var(--oz-gutter) 1.5rem; }
        .logit-attribution .la-panel + .la-panel { border-left: 1px solid var(--oz-rule); }
        .logit-attribution .pl-card-title { margin-bottom: 0.6rem; }
        .logit-attribution .la-ramp-legend {
          display: flex; align-items: center; gap: 0.5rem; margin-top: 0.7rem;
          color: var(--oz-fg-dim); font-family: var(--oz-mono); font-size: 0.75rem;
        }
        .logit-attribution .la-ramp { width: 140px; height: 8px; border: 1px solid var(--oz-rule-strong); }

        @media (max-width: 1250px) {
          .logit-attribution .la-panels { grid-template-columns: 1fr; }
          .logit-attribution .la-panel + .la-panel { border-left: none; border-top: 1px solid var(--oz-rule); }
        }
      `}</style>
    </div>
  )
})

/** Each layer's attention and MLP share, after the embeddings' (and the output bias's, if any). */
function LayerBars({ shares, showBias }: { shares: Shares; showBias: boolean }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(620)
  const [hover, setHover] = useState<{ band: number; px: number; py: number } | null>(null)

  const layers = shares.attn.length
  // Bands: the embeddings, the bias (when the model has one), then every layer
  const lead: { label: string; name: string; value: number }[] = [{ label: 'Emb', name: 'Embeddings', value: shares.embed }]
  if (showBias) lead.push({ label: 'Bias', name: 'Output bias', value: shares.bias })
  const bands = lead.length + layers

  const innerW = Math.max(width - BARS_MARGIN.left - BARS_MARGIN.right, 80)
  const innerH = BARS_HEIGHT - BARS_MARGIN.top - BARS_MARGIN.bottom
  const band = innerW / bands
  const values = [...lead.map((l) => l.value), ...shares.attn, ...shares.mlp]
  const y = d3
    .scaleLinear()
    .domain([Math.min(0, d3.min(values) ?? 0), Math.max(0, d3.max(values) ?? 0)])
    .nice(5)
    .range([BARS_MARGIN.top + innerH, BARS_MARGIN.top])
  if (y.domain()[0] === y.domain()[1]) y.domain([-1, 1])
  const zero = y(0)
  const bar = (x: number, w: number, value: number, fill: string, key: string) => (
    <rect key={key} x={x} y={Math.min(zero, y(value))} width={Math.max(w, 1)} height={Math.max(Math.abs(y(value) - zero), value === 0 ? 0 : 1)} fill={fill} />
  )
  const half = Math.max((band * 0.8) / 2, 1)
  const labelEvery = Math.max(1, Math.ceil(layers / (innerW / 34)))

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const pointer = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const index = Math.floor((pointer.x - BARS_MARGIN.left) / band)
    setHover(index >= 0 && index < bands ? { band: index, px: pointer.x, py: pointer.y } : null)
  }
  const hovered = hover == null ? null : hover.band < lead.length ? lead[hover.band] : null
  const hoveredLayer = hover != null && hover.band >= lead.length ? hover.band - lead.length : null

  return (
    <>
      <div className="pl-card-title">
        <h3 title="What each layer wrote directly toward the token's logit">By layer</h3>
        <div className="pl-legend" style={{ marginTop: 0 }}>
          <span className="pl-legend-item">
            <span className="pl-legend-rect" style={{ background: colors.series[0] }} />
            Attention
          </span>
          <span className="pl-legend-item">
            <span className="pl-legend-rect" style={{ background: colors.series[1] }} />
            MLP
          </span>
        </div>
      </div>
      <div ref={ref} className="pl-chart">
        <svg width={width} height={BARS_HEIGHT} role="img" aria-label="Each layer's attention and MLP share of the logit">
          {y.ticks(5).map((t) => (
            <g key={t}>
              <line x1={BARS_MARGIN.left} x2={BARS_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={t === 0 ? colors.axis : colors.grid} />
              <text x={BARS_MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
                {t}
              </text>
            </g>
          ))}
          {hover && <rect x={BARS_MARGIN.left + hover.band * band} y={BARS_MARGIN.top} width={band} height={innerH} fill={colors.hover} />}
          {lead.map((l, i) => bar(BARS_MARGIN.left + i * band + band * 0.2, band * 0.6, l.value, colors.baseline, l.label))}
          {shares.attn.map((value, layer) => {
            const x0 = BARS_MARGIN.left + (lead.length + layer) * band + band * 0.1
            return (
              <g key={layer}>
                {bar(x0, half, value, colors.series[0], 'a')}
                {bar(x0 + half, half, shares.mlp[layer], colors.series[1], 'm')}
              </g>
            )
          })}
          {lead.map((l, i) => (
            <text key={l.label} x={BARS_MARGIN.left + (i + 0.5) * band} y={BARS_MARGIN.top + innerH + 14} textAnchor="middle" fontSize={10} fill={colors.text}>
              {band >= 26 ? l.label : l.label[0]}
            </text>
          ))}
          {shares.attn.map((_, layer) =>
            layer % labelEvery === 0 ? (
              <text
                key={layer}
                x={BARS_MARGIN.left + (lead.length + layer + 0.5) * band}
                y={BARS_MARGIN.top + innerH + 14}
                textAnchor="middle"
                fontSize={10}
                fill={colors.text}
              >
                {layer}
              </text>
            ) : null,
          )}
          <text x={BARS_MARGIN.left + innerW / 2} y={BARS_HEIGHT - 3} textAnchor="middle" fontSize={11} fill={colors.dim}>
            Layer
          </text>
          <rect
            x={BARS_MARGIN.left}
            y={BARS_MARGIN.top}
            width={innerW}
            height={innerH}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hover && hovered && (
          <ChartTooltip x={hover.px} y={hover.py} containerWidth={width} title={hovered.name} rows={[{ key: colors.baseline, value: signed(hovered.value), label: 'logit' }]} />
        )}
        {hover && hoveredLayer != null && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={`Layer ${hoveredLayer}`}
            rows={[
              { key: colors.series[0], value: signed(shares.attn[hoveredLayer]), label: 'attention' },
              { key: colors.series[1], value: signed(shares.mlp[hoveredLayer]), label: 'MLP' },
            ]}
          />
        )}
      </div>
    </>
  )
}

/** Every head's share: layers down, heads across (and the output projection's bias, if any). */
function HeadGrid({ shares, numHeads, showBias }: { shares: Shares; numHeads: number; showBias: boolean }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(420)
  const [hover, setHover] = useState<{ layer: number; column: number; px: number; py: number } | null>(null)

  const layers = shares.heads.length
  const columns = numHeads + (showBias ? 1 : 0)
  const innerW = Math.max(width - HEADS_MARGIN.left - HEADS_MARGIN.right, 60)
  const cellW = Math.min(46, innerW / columns)
  const cellH = Math.max(12, Math.min(24, 520 / layers))
  const height = HEADS_MARGIN.top + layers * cellH + HEADS_MARGIN.bottom
  const value = (layer: number, column: number) => (column < numHeads ? shares.heads[layer][column] : shares.attnBias[layer])
  const limit = Math.max(1e-6, ...shares.heads.flat().map(Math.abs), ...(showBias ? shares.attnBias.map(Math.abs) : []))
  const layerEvery = Math.max(1, Math.ceil(14 / cellH))
  const headEvery = Math.max(1, Math.ceil(24 / cellW))

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const pointer = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    const column = Math.floor((pointer.x - HEADS_MARGIN.left) / cellW)
    const layer = Math.floor((pointer.y - HEADS_MARGIN.top) / cellH)
    setHover(column >= 0 && column < columns && layer >= 0 && layer < layers ? { layer, column, px: pointer.x, py: pointer.y } : null)
  }

  return (
    <>
      <div className="pl-card-title">
        <h3 title="What each attention head wrote directly toward the token's logit">Heads</h3>
      </div>
      <div ref={ref} className="pl-chart">
        <svg width={width} height={height} role="img" aria-label={`Each head's share of the logit, ${layers} layers by ${numHeads} heads`}>
          <g transform={`translate(${HEADS_MARGIN.left},${HEADS_MARGIN.top})`}>
            {shares.heads.map((_, layer) =>
              Array.from({ length: columns }, (_, column) => {
                const fill = divergingColor(colors, value(layer, column) / limit)
                return (
                  <g key={`${layer}-${column}`}>
                    <rect x={column * cellW} y={layer * cellH} width={cellW - 1} height={cellH - 1} fill={fill} />
                    {cellW >= 38 && cellH >= 16 && Math.abs(value(layer, column)) >= limit * 0.25 && (
                      <text x={(column + 0.5) * cellW} y={(layer + 0.5) * cellH} dy="0.32em" textAnchor="middle" fontSize={9.5} fill={inkOn(fill)}>
                        {signed(value(layer, column), 1)}
                      </text>
                    )}
                  </g>
                )
              }),
            )}
            {hover && <rect x={hover.column * cellW} y={hover.layer * cellH} width={cellW - 1} height={cellH - 1} fill="none" stroke={colors.text} strokeWidth={1.5} />}
            {shares.heads.map((_, layer) =>
              layer % layerEvery === 0 ? (
                <text key={layer} x={-6} y={(layer + 0.5) * cellH} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
                  L{layer}
                </text>
              ) : null,
            )}
            {Array.from({ length: columns }, (_, column) =>
              column >= numHeads || column % headEvery === 0 ? (
                <text key={column} x={(column + 0.5) * cellW} y={-7} textAnchor="middle" fontSize={10} fill={colors.text}>
                  {column < numHeads ? `H${column}` : 'bias'}
                </text>
              ) : null,
            )}
          </g>
          <rect
            x={HEADS_MARGIN.left}
            y={HEADS_MARGIN.top}
            width={columns * cellW}
            height={layers * cellH}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hover && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={hover.column < numHeads ? `L${hover.layer}H${hover.column}` : `Layer ${hover.layer} output bias`}
            rows={[{ value: signed(value(hover.layer, hover.column)), label: 'logit' }]}
          />
        )}
      </div>
      <div className="la-ramp-legend">
        <span>{signed(-limit, 1)}</span>
        <span
          className="la-ramp"
          style={{ background: `linear-gradient(to right, ${divergingColor(colors, -1)}, ${divergingColor(colors, 0)}, ${divergingColor(colors, 1)})` }}
        />
        <span>{signed(limit, 1)}</span>
      </div>
    </>
  )
}
