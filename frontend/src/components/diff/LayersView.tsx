/**
 * The Layers view: where two models part. Per residual stream position, the distance between
 * their activations, linear CKA and Procrustes, each also on the two halves of the texts; and
 * per token, the KL between their next-token predictions (or its distance at the chosen layer).
 */

import { useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { DiffLab, TokenColor } from '../../hooks/useDiffLab'
import type { LayerMeasure, LayersRun } from '../../types/diff'
import { LAYER_MEASURES } from '../../types/diff'
import { FigureExport } from '../attribution/FigureExport'
import { ChartTooltip, Findings, Placeholder, Segmented, StatTiles, type Finding, type TooltipRow } from '../lab/chartKit'
import { divergingColor, inkOn, pointerIn, useElementWidth, useLabColors } from '../lab/labTheme'
import {
  MEASURE_LABELS,
  MEASURE_TITLES,
  halvesGap,
  massiveText,
  mostDivergent,
  nats,
  pairName,
  peak,
  percent,
  positionLabel,
  positionName,
  profileRange,
  tokenLabel,
  tokenOffsets,
  value,
} from './diffUtils'

type Profile = (number | null)[]

/** How far apart the halves of the texts may be before a profile is flagged: this share of its range, at least this much. */
const HALVES_SHARE = 0.25
const HALVES_FLOOR = 0.05
// Texts shown in the token map before "Show all"
const SHOWN_TEXTS = 6
const DIVERGENT_TOKENS = 10
// Characters of a text shown before one of its tokens
const CONTEXT_CHARS = 26

function findings(run: LayersRun, measure: LayerMeasure): Finding[] {
  const out: Finding[] = []
  const profile = run[measure]
  const [low, high] = profileRange(profile)
  // Distance grows as models part; the other three fall
  const at = peak(profile, measure !== 'distance')
  out.push({
    level: 'info',
    text: `${MEASURE_LABELS[measure]} is ${measure === 'distance' ? 'largest' : 'lowest'} at ${positionName(at).toLowerCase()} (${value(profile[at])}).`,
  })
  const gap = halvesGap(run, measure)
  if (gap == null) {
    out.push({ level: 'info', text: `No halves for ${MEASURE_LABELS[measure]} on a model this wide: check the distance’s.` })
  } else if (gap.gap > Math.max(HALVES_FLOOR, HALVES_SHARE * (high - low))) {
    out.push({
      level: 'warning',
      text: `The two halves of the texts are ${value(gap.gap)} apart at ${positionName(gap.position).toLowerCase()}: the profile depends on the sample.`,
    })
  } else {
    out.push({ level: 'good', text: `The two halves of the texts agree within ${value(gap.gap)}.` })
  }
  if (measure === 'cka') {
    const chance = Math.max(...run.cka_chance.filter((v): v is number => v != null), 0)
    if (chance > 0.2) out.push({ level: 'warning', text: `Randomly paired tokens score up to ${value(chance, 2)}: too few tokens for a model this wide.` })
  }
  const massive = massiveText(run)
  if (massive) out.push({ level: 'info', text: massive })
  if (run.dropped.length > 0) {
    out.push({ level: 'info', text: `${run.dropped.length} text${run.dropped.length === 1 ? '' : 's'} dropped: ${run.dropped[0].reason}.` })
  }
  if (run.cut > 0) out.push({ level: 'info', text: `${run.cut} text${run.cut === 1 ? '' : 's'} cut to what both models can read.` })
  return out
}

const HEIGHT = 300
const MARGIN = { top: 28, right: 76, bottom: 40, left: 48 }

function linePath(values: Profile, x: (i: number) => number, y: (v: number) => number): string {
  let path = ''
  let pen = false
  values.forEach((v, i) => {
    if (v == null) {
      pen = false
      return
    }
    path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
    pen = true
  })
  return path
}

/** A measure at every residual stream position: all texts, each half, and CKA's chance level. */
function ProfileChart({ lab, run }: { lab: DiffLab; run: LayersRun }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const measure = lab.measure
  const whole = run[measure]
  const halves = run.halves[measure]
  const chance = measure === 'cka' ? run.cka_chance : null
  const positions = run.num_layers + 1
  const position = Math.min(lab.position, positions - 1)

  const all = [...whole, ...(halves ?? []).flat(), ...(chance ?? [])].filter((v): v is number => v != null)
  const top = measure === 'distance' ? Math.max(0.5, Math.ceil(Math.max(...all, 0) * 10) / 10) : 1
  const bottom = measure === 'distance' ? 0 : Math.max(0, Math.min(0.5, Math.floor(Math.min(...all, 1) * 10) / 10))
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 50)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => MARGIN.left + (positions > 1 ? i * step : innerW / 2)
  const y = (v: number) => MARGIN.top + (1 - (Math.max(bottom, Math.min(top, v)) - bottom) / (top - bottom || 1)) * innerH
  // Round steps, so every gridline is labeled with the value it's drawn at
  const tickStep = [0.05, 0.1, 0.2, 0.25, 0.5, 1].find((size) => (top - bottom) / size <= 6.001) ?? 1
  const ticks = Array.from({ length: Math.floor((top - bottom) / tickStep + 0.001) + 1 }, (_, i) => bottom + i * tickStep)
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))
  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - MARGIN.left) / step)))
  const color = colors.series[0]
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const { x: px, y: py } = pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement)
    setHover({ index: indexAt(px), px, py })
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft' && position > 0) lab.setPosition(position - 1)
    else if (e.key === 'ArrowRight' && position < positions - 1) lab.setPosition(position + 1)
    else return
    e.preventDefault()
  }
  const rows = (i: number): TooltipRow[] => [
    { key: color, label: 'all texts', value: value(whole[i]) },
    ...(halves ? halves.map((half, h) => ({ label: `half ${h === 0 ? 'A' : 'B'}`, value: value(half[i]) })) : []),
    ...(chance ? [{ key: colors.baseline, label: 'chance', value: value(chance[i]) }] : []),
    ...(run.massive[i] > 0 ? [{ label: 'tokens left out', value: String(run.massive[i]) }] : []),
  ]

  return (
    <div ref={ref}>
      <div
        className="pl-chart"
        tabIndex={0}
        role="group"
        aria-label={`${MEASURE_LABELS[measure]} by layer. Selected: ${positionName(position)}. Use arrow keys to change layer.`}
        onKeyDown={onKeyDown}
        data-assist="diff.layer"
      >
        <svg width={width} height={HEIGHT} role="img" aria-hidden="true">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
              <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                {t.toFixed(tickStep === 0.05 || tickStep === 0.25 ? 2 : 1)}
              </text>
            </g>
          ))}
          <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={MARGIN.top + innerH} y2={MARGIN.top + innerH} stroke={colors.axis} />
          {Array.from({ length: positions }, (_, i) =>
            i % labelEvery === 0 || i === positions - 1 ? (
              <text key={i} x={x(i)} y={MARGIN.top + innerH + 16} textAnchor="middle" fontSize={11} fill={colors.text}>
                {positionLabel(i)}
              </text>
            ) : null,
          )}
          <text x={MARGIN.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
            Layer
          </text>

          {/* Selected layer */}
          <rect x={x(position) - Math.max(step, 8) / 2} y={MARGIN.top} width={Math.max(step, 8)} height={innerH} fill={colors.hover} />
          <text x={x(position)} y={MARGIN.top - 10} textAnchor="middle" fontSize={11} fontWeight={500} fill={colors.text} {...halo}>
            {positionName(position).toLowerCase()}
          </text>

          {chance && <path d={linePath(chance, x, y)} fill="none" stroke={colors.baseline} strokeWidth={1.5} strokeDasharray="5 4" />}
          {halves?.map((half, h) => (
            <path key={h} d={linePath(half, x, y)} fill="none" stroke={color} strokeWidth={1.25} opacity={0.5} strokeLinejoin="round" />
          ))}
          <path d={linePath(whole, x, y)} fill="none" stroke={color} strokeWidth={2.25} strokeLinejoin="round" strokeLinecap="round" />
          {whole[position] != null && <circle cx={x(position)} cy={y(whole[position] as number)} r={4} fill={color} stroke={colors.surface} strokeWidth={2} />}
          {whole[positions - 1] != null && (
            <text x={MARGIN.left + innerW + 6} y={y(whole[positions - 1] as number)} dy="0.32em" fontSize={10} fontWeight={500} fill={colors.text}>
              {MEASURE_LABELS[measure]}
            </text>
          )}

          {hover && <line x1={x(hover.index)} x2={x(hover.index)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} />}
          <rect
            x={MARGIN.left - step / 2}
            y={MARGIN.top}
            width={innerW + step}
            height={innerH}
            fill="transparent"
            style={{ cursor: 'pointer' }}
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
            onClick={(e) => lab.setPosition(indexAt(pointerIn(e, e.currentTarget.ownerSVGElement as SVGSVGElement).x))}
          />
        </svg>
        {hover && <ChartTooltip x={hover.px} y={hover.py} containerWidth={width} title={positionName(hover.index)} rows={rows(hover.index)} />}
      </div>
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-line" style={{ background: color }} />
          All texts
        </span>
        {halves && (
          <span className="pl-legend-item" title="Every other text: where the two lines part, the profile depends on the sample">
            <span className="pl-legend-line" style={{ background: color, opacity: 0.5 }} />
            Each half of the texts
          </span>
        )}
        {chance && (
          <span className="pl-legend-item" title="What randomly paired tokens score">
            <span className="pl-legend-line" style={{ background: `repeating-linear-gradient(to right, ${colors.baseline} 0 5px, transparent 5px 9px)` }} />
            Chance
          </span>
        )}
      </div>
    </div>
  )
}

function ProfileTable({ lab, run }: { lab: DiffLab; run: LayersRun }) {
  const colors = useLabColors()
  return (
    <div className="df-table-scroll" style={{ border: `1px solid ${colors.border}` }}>
      <table className="pl-table" aria-label="Every measure per layer">
        <thead>
          <tr>
            <th>Layer</th>
            {LAYER_MEASURES.map((m) => (
              <th key={m}>{MEASURE_LABELS[m]}</th>
            ))}
            <th title="What randomly paired tokens score on linear CKA">CKA chance</th>
            <th title="Tokens left out for their massive activations">Left out</th>
          </tr>
        </thead>
        <tbody>
          {run.distance.map((_, p) => (
            <tr key={p} onClick={() => lab.setPosition(p)} className={p === lab.position ? 'is-active' : undefined}>
              <td>{positionLabel(p)}</td>
              {LAYER_MEASURES.map((m) => (
                <td key={m}>{value(run[m][p])}</td>
              ))}
              <td>{value(run.cka_chance[p])}</td>
              <td>{run.massive[p] || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The texts, token by token, colored by the KL of the two models' predictions or by their distance at the chosen layer. */
function TokenMap({ lab, run }: { lab: DiffLab; run: LayersRun }) {
  const colors = useLabColors()
  const [hovered, setHovered] = useState<{ text: number; index: number } | null>(null)
  const [all, setAll] = useState(false)
  const offsets = useMemo(() => tokenOffsets(run.texts), [run])
  const position = Math.min(lab.position, run.num_layers)
  const byDistance = lab.tokenColor === 'distance'
  const distances = useMemo(() => run.token_distance[position] ?? [], [run, position])

  const scale = useMemo(() => {
    const values = byDistance ? distances.filter((v) => v >= 0) : run.texts.flatMap((t) => t.kl)
    if (values.length === 0) return 1
    // The 98th percentile, so a few extreme tokens don't wash the rest out
    const sorted = [...values].sort((a, b) => a - b)
    return Math.max(sorted[Math.floor(0.98 * (sorted.length - 1))], 1e-6)
  }, [run, byDistance, distances])

  const shown = all ? run.texts : run.texts.slice(0, SHOWN_TEXTS)
  const at = hovered ? run.texts[hovered.text] : null
  const readout = (() => {
    if (!hovered || !at) return byDistance ? 'Hover a token for its distance.' : 'Hover a token for both models’ predictions after it.'
    const i = hovered.index
    const token = `“${tokenLabel(at.tokens[i])}”`
    if (byDistance) {
      const d = distances[offsets[hovered.text] + i]
      return d == null || d < 0 ? `${token} · left out: massive activation` : `${token} · distance ${value(d)}`
    }
    return `${token} · KL ${nats(at.kl[i])} · ${run.base_name} → “${tokenLabel(at.base_top[i])}” ${percent(at.base_prob[i], 0)} · ${run.other_name} → “${tokenLabel(at.other_top[i])}” ${percent(at.other_prob[i], 0)}`
  })()

  return (
    <div>
      <div className="df-readout" aria-live="polite">
        {readout}
      </div>
      <div className="df-tokens">
        {shown.map((text, t) => (
          <p key={t} className="df-text">
            {(text.lead ?? []).map((token, i) => (
              <span key={`lead-${i}`} className="pl-token df-token is-left-out" title="Not compared: a text’s first token" style={{ color: colors.dim }}>
                {token}
              </span>
            ))}
            {text.tokens.map((token, i) => {
              const raw = byDistance ? (distances[offsets[t] + i] ?? -1) : text.kl[i]
              const left = raw < 0
              const background = left ? 'transparent' : divergingColor(colors, Math.min(raw / scale, 1))
              const active = hovered?.text === t && hovered.index === i
              return (
                <span
                  key={i}
                  className={`pl-token df-token${left ? ' is-left-out' : ''}`}
                  onPointerEnter={() => setHovered({ text: t, index: i })}
                  onPointerLeave={() => setHovered(null)}
                  style={{ background, color: left ? colors.dim : inkOn(background), outline: active ? `1.5px solid ${colors.text}` : undefined }}
                >
                  {token.replace(/\n/g, '↵\n').replace(/\t/g, '→\t')}
                </span>
              )
            })}
          </p>
        ))}
      </div>
      {run.texts.length > SHOWN_TEXTS && (
        <button type="button" className="pl-link" onClick={() => setAll(!all)}>
          {all ? `Show ${SHOWN_TEXTS}` : `Show all ${run.texts.length} texts`}
        </button>
      )}
      <div className="df-scale">
        <span>0</span>
        <span className="df-scale__bar" style={{ background: `linear-gradient(to right, ${divergingColor(colors, 0)}, ${divergingColor(colors, 0.5)}, ${divergingColor(colors, 1)})` }} />
        <span>{byDistance ? value(scale, 2) : `${nats(scale)} nats`}</span>
        <span className="pl-note">{byDistance ? `distance at ${positionName(position).toLowerCase()}` : `KL(${run.other_name} ‖ ${run.base_name})`}</span>
      </div>
    </div>
  )
}

function Divergent({ run }: { run: LayersRun }) {
  const tokens = useMemo(() => mostDivergent(run, DIVERGENT_TOKENS), [run])
  return (
    <table className="pl-table df-divergent">
      <thead>
        <tr>
          <th>After</th>
          <th>KL</th>
          <th title={run.base_name}>Base</th>
          <th title={run.other_name}>Other</th>
        </tr>
      </thead>
      <tbody>
        {tokens.map(({ text, index, kl }) => {
          const t = run.texts[text]
          const before = [...(t.lead ?? []), ...t.tokens.slice(0, index)].join('')
          return (
            <tr key={`${text}-${index}`}>
              <td className="df-context" title={before + t.tokens[index]}>
                <span className="df-context__before">{before.length > CONTEXT_CHARS ? `…${before.slice(-CONTEXT_CHARS)}` : before}</span>
                <b>{tokenLabel(t.tokens[index])}</b>
              </td>
              <td>{nats(kl)}</td>
              <td className="df-mono" title={percent(t.base_prob[index], 0)}>
                {tokenLabel(t.base_top[index])}
              </td>
              <td className="df-mono" title={percent(t.other_prob[index], 0)}>
                {tokenLabel(t.other_top[index])}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function LayersView({ lab }: { lab: DiffLab }) {
  const [table, setTable] = useState<'chart' | 'table'>('chart')
  const run = lab.run
  if (!run) {
    return (
      <section className="pl-panel">
        <Placeholder glyph="diff">{lab.running ? 'Reading both models…' : 'Choose a pair and texts, then compare their layers.'}</Placeholder>
      </section>
    )
  }

  const name = pairName(run)
  const top = peak(run.distance)
  const gap = halvesGap(run, 'distance')
  const halves = run.halves[lab.measure]
  const labels = run.distance.map((_, p) => positionLabel(p))

  return (
    <div style={{ opacity: lab.running ? 0.55 : 1 }} aria-busy={lab.running}>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{name}</strong> · {run.texts_name} · {run.num_texts} texts · {run.num_tokens.toLocaleString()} tokens ·{' '}
          {run.chat_template ? 'chat template' : 'plain text'} · ${run.cost.toFixed(2)}
        </div>
        <StatTiles
          tiles={[
            {
              label: 'Next-token KL',
              value: nats(run.kl),
              hint: 'nats per token',
              title: `KL(${run.other_name} ‖ ${run.base_name}) between the two models’ next-token predictions, mean over tokens`,
            },
            { label: 'Same top token', value: percent(run.top1_agreement, 0), title: 'Tokens after which both models predict the same next token' },
            { label: 'Largest distance', value: value(run.distance[top], 2), hint: positionName(top).toLowerCase(), title: MEASURE_TITLES.distance },
            {
              label: 'Halves apart',
              value: gap ? value(gap.gap, 2) : '–',
              hint: 'distance, at most',
              title: 'The largest difference between the distance profiles of the two halves of the texts',
            },
          ]}
        />
        <Findings items={findings(run, lab.measure)} />
      </section>

      <section className="pl-panel">
        <div className="pl-card-title">
          <h3 title={MEASURE_TITLES[lab.measure]}>By layer</h3>
          <div className="df-title-tools">
            <Segmented
              ariaLabel="Chart or table"
              value={table}
              onChange={setTable}
              options={[
                { value: 'chart', label: 'Chart' },
                { value: 'table', label: 'Table' },
              ]}
            />
            <FigureExport
              filename={`diff-${lab.measure}-${run.base}`}
              figure={() => ({
                kind: 'lines',
                title: `${MEASURE_LABELS[lab.measure]} by layer: ${name} (${run.texts_name})`,
                x_label: 'Layer',
                y_label: MEASURE_LABELS[lab.measure],
                column_labels: labels,
                series: [
                  { label: 'All texts', values: run[lab.measure] },
                  ...(halves ? halves.map((values, h) => ({ label: `Half ${h === 0 ? 'A' : 'B'}`, values, style: 'faint' as const })) : []),
                  ...(lab.measure === 'cka' ? [{ label: 'Chance', values: run.cka_chance, style: 'dashed' as const }] : []),
                ],
              })}
            />
          </div>
        </div>
        <div className="pl-controls-row">
          <div data-assist="diff.metric">
            <div className="pl-control-label">Measure</div>
            <Segmented
              ariaLabel="Measure"
              value={lab.measure}
              onChange={lab.setMeasure}
              options={LAYER_MEASURES.map((m) => ({ value: m, label: MEASURE_LABELS[m], title: MEASURE_TITLES[m] }))}
            />
          </div>
        </div>
        {table === 'chart' ? <ProfileChart lab={lab} run={run} /> : <ProfileTable lab={lab} run={run} />}
        <button type="button" className="pl-link df-open" onClick={() => lab.setView('directions')}>
          Directions at {positionName(Math.min(lab.position, run.num_layers)).toLowerCase()}
        </button>
      </section>

      <div className="pl-results-grid df-token-grid">
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>Tokens</h3>
            <Segmented
              ariaLabel="Color tokens by"
              value={lab.tokenColor}
              onChange={(v: TokenColor) => lab.setTokenColor(v)}
              options={[
                { value: 'kl', label: 'KL', title: 'How far apart the two models’ next-token predictions are after each token' },
                { value: 'distance', label: `Distance · ${positionLabel(Math.min(lab.position, run.num_layers))}`, title: 'Each token’s distance between the two models at the chosen layer' },
              ]}
            />
          </div>
          <TokenMap lab={lab} run={run} />
        </section>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3 title="The tokens after which the two models’ next-token predictions are furthest apart">Largest KL</h3>
          </div>
          <Divergent run={run} />
        </section>
      </div>
    </div>
  )
}
