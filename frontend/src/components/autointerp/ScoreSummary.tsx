/**
 * The Autointerp tab's Summary view: an atlas's scores with one pair of models, against its
 * random-direction controls (explained and scored the same way; on many SAEs they score nearly
 * as well, so no SAE-level number is shown without them), the token baseline, and scores by
 * frequency band.
 */

import { useMemo, useState, type PointerEvent } from 'react'
import { BarChart3 } from 'lucide-react'
import { useAutointerpSummary, type AutointerpAtlasRef } from '../../hooks/useAutointerp'
import type { SummaryResponse } from '../../types/autointerp'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, Placeholder, StatTiles, type Finding, type Tile } from '../lab/chartKit'
import { pointerIn, useElementWidth, useLabColors, type LabColors } from '../lab/labTheme'
import { formatScore, modelName } from './labels'

const BINS = 20
const HEIGHT = 200
const MARGIN = { top: 20, right: 12, bottom: 30, left: 40 }
// Medians this close count as features scoring like random directions; controls needed for a
// baseline (the backend's MIN_CONTROLS)
const SAME_AS_CONTROLS = 0.03
const MIN_CONTROLS = 8

function pct(value: number | null | undefined): string {
  return value == null ? '–' : `${Math.round(value * 100)}%`
}

function formatFreq(freq: number): string {
  return freq >= 0.001 ? `${(freq * 100).toFixed(2)}%` : `${(freq * 1e6).toFixed(0)}/M`
}

function shares(values: number[]): { counts: number[]; shares: number[] } {
  const counts = new Array(BINS).fill(0)
  values.forEach((v) => {
    counts[Math.min(BINS - 1, Math.max(0, Math.floor(v * BINS)))] += 1
  })
  return { counts, shares: counts.map((c) => (values.length ? c / values.length : 0)) }
}

function Histogram({ summary, colors }: { summary: SummaryResponse; colors: LabColors }) {
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ bin: number; x: number; y: number } | null>(null)
  const features = useMemo(() => shares(summary.scores), [summary.scores])
  const controls = useMemo(() => shares(summary.control_scores), [summary.control_scores])
  const showControls = summary.control_scores.length > 0
  const plotW = Math.max(100, width - MARGIN.left - MARGIN.right)
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom
  const top = Math.max(...features.shares, ...(showControls ? controls.shares : [0]), 0.01)
  const ceiling = Math.ceil(top * 10) / 10
  const x = (v: number) => MARGIN.left + v * plotW
  const y = (share: number) => MARGIN.top + plotH * (1 - share / ceiling)
  const binW = plotW / BINS
  const ticks = [0, ceiling / 2, ceiling]

  // The controls' distribution as a step line over the bins
  let step = ''
  controls.shares.forEach((s, i) => {
    step += `${i === 0 ? `M${x(0)},${y(0)} ` : ''}L${x(i / BINS)},${y(s)} L${x((i + 1) / BINS)},${y(s)} `
  })
  step += `L${x(1)},${y(0)}`

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const p = pointerIn(e, e.currentTarget)
    const bin = Math.floor((p.x - MARGIN.left) / binW)
    setHover(bin >= 0 && bin < BINS && p.y >= MARGIN.top && p.y <= MARGIN.top + plotH ? { bin, x: p.x, y: p.y } : null)
  }

  return (
    <div className="pl-chart" ref={ref}>
      <svg width={width} height={HEIGHT} role="img" aria-label="Scores of features and of random-direction controls" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + plotW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <text x={MARGIN.left - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill={colors.text} opacity={0.7}>
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {features.shares.map((s, i) =>
          s > 0 ? (
            <path
              key={i}
              d={roundedTop(x(i / BINS) + 1, y(s), binW - 2, y(0) - y(s), Math.min(4, (binW - 2) / 2))}
              fill={colors.series[0]}
              opacity={hover && hover.bin !== i ? 0.55 : 1}
            />
          ) : null,
        )}
        {showControls && <path d={step} fill="none" stroke={colors.baseline} strokeWidth={2} />}
        <line x1={x(0.5)} x2={x(0.5)} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke={colors.axis} strokeDasharray="3 3" />
        <text x={x(0.5)} y={MARGIN.top - 6} textAnchor="middle" fontSize={10} fill={colors.text} opacity={0.7}>
          chance
        </text>
        <line x1={MARGIN.left} x2={MARGIN.left + plotW} y1={y(0)} y2={y(0)} stroke={colors.axis} />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={x(t)} y={y(0) + 16} textAnchor="middle" fontSize={10} fill={colors.text} opacity={0.7}>
            {t.toFixed(2)}
          </text>
        ))}
        <text x={MARGIN.left + plotW / 2} y={HEIGHT - 2} textAnchor="middle" fontSize={10} fill={colors.text} opacity={0.7}>
          Score
        </text>
        {hover && <rect x={x(hover.bin / BINS)} y={MARGIN.top} width={binW} height={plotH} fill={colors.hover} pointerEvents="none" />}
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.x}
          y={hover.y}
          containerWidth={width}
          title={`Score ${(hover.bin / BINS).toFixed(2)}–${((hover.bin + 1) / BINS).toFixed(2)}`}
          rows={[
            { key: colors.series[0], label: `features (${features.counts[hover.bin]})`, value: pct(features.shares[hover.bin]) },
            ...(showControls ? [{ key: colors.baseline, label: `controls (${controls.counts[hover.bin]})`, value: pct(controls.shares[hover.bin]) }] : []),
          ]}
        />
      )}
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.series[0] }} /> Features ({summary.scores.length})
        </span>
        {showControls && (
          <span className="pl-legend-item">
            <span className="pl-legend-line" style={{ background: colors.baseline }} /> Random-direction controls ({summary.control_scores.length})
          </span>
        )}
      </div>
    </div>
  )
}

/** A bar with rounded top corners, anchored square to its baseline. */
function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  const radius = Math.min(r, h)
  return `M${x},${y + h} L${x},${y + radius} Q${x},${y} ${x + radius},${y} L${x + w - radius},${y} Q${x + w},${y} ${x + w},${y + radius} L${x + w},${y + h} Z`
}

function findings(summary: SummaryResponse): Finding[] {
  const out: Finding[] = []
  const f = summary.features
  const c = summary.controls
  if (f && c && c.n >= MIN_CONTROLS) {
    const gap = f.median - c.median
    if (gap < SAME_AS_CONTROLS) {
      out.push({ level: 'warning', text: `Features score like random directions: median ${formatScore(f.median)} vs ${formatScore(c.median)}.` })
    } else {
      out.push({ level: 'good', text: `Median feature ${formatScore(f.median)} vs ${formatScore(c.median)} for random directions; ${pct(summary.above_controls)} beat the controls’ 90th percentile.` })
    }
  } else if (f) {
    out.push({ level: 'info', text: 'Too few controls scored for a baseline.' })
  }
  if (f && summary.token_features > 0) {
    out.push({ level: 'info', text: `${summary.token_features} of ${f.n} explanations don’t beat their top tokens (token features).` })
  }
  return out
}

export function ScoreSummary({ atlasRef }: { atlasRef: AutointerpAtlasRef }) {
  const colors = useLabColors()
  const [pair, setPair] = useState<{ explainer: string; scorer: string } | null>(null)
  const query = useAutointerpSummary(atlasRef, pair)
  const summary = query.data
  if (!summary) {
    return <div className="pl-note">{query.isLoading ? 'Loading…' : (query.error as Error | null)?.message ?? ''}</div>
  }
  if (!summary.features) {
    return <Placeholder icon={BarChart3}>No scored explanations on this atlas yet.</Placeholder>
  }

  const tiles: Tile[] = [
    { label: 'Scored', value: summary.features.n.toLocaleString(), hint: 'features' },
    { label: 'Median score', value: formatScore(summary.features.median), hint: `p10 ${formatScore(summary.features.p10)} · p90 ${formatScore(summary.features.p90)}`, title: 'Mean of detection and fuzzing; 0.5 is chance' },
    {
      label: 'Controls',
      value: summary.controls ? formatScore(summary.controls.median) : '–',
      hint: summary.controls ? `median of ${summary.controls.n}` : 'none scored',
      title: 'Random residual directions, explained and scored the same way',
    },
    { label: 'Above controls', value: pct(summary.above_controls), hint: 'over their p90', title: 'Share of features scoring above the controls’ 90th percentile' },
    {
      label: 'Token baseline',
      value: summary.token_baseline ? formatScore(summary.token_baseline.median) : '–',
      hint: 'median detection',
      title: 'Detection with each feature’s top 3 firing tokens as its explanation',
    },
    { label: 'Token features', value: String(summary.token_features), hint: `of ${summary.features.n}` },
  ]
  const pairKey = `${summary.explainer}|${summary.scorer}`

  return (
    <div className="ai-summary">
      {summary.pairs.length > 1 && (
        <div style={{ maxWidth: 420, marginBottom: '0.75rem' }}>
          <div className="pl-control-label">Models</div>
          <Dropdown
            value={pairKey}
            onChange={(v) => {
              const [explainer, scorer] = v.split('|')
              setPair({ explainer, scorer })
            }}
            options={summary.pairs.map((p) => ({
              value: `${p.explainer}|${p.scorer}`,
              label: `${modelName(p.explainer)} explains · ${modelName(p.scorer)} scores (${p.features})`,
            }))}
          />
        </div>
      )}
      <StatTiles tiles={tiles} />
      <Findings items={findings(summary)} />
      <Histogram summary={summary} colors={colors} />
      {summary.bands.length > 0 && (
        <table className="pl-table ai-bands" style={{ marginTop: '1rem' }}>
          <thead>
            <tr>
              <th title="Bands of equal size, by the share of corpus tokens a feature is active on">Frequency</th>
              <th>Features</th>
              <th>Mean</th>
              <th>Median</th>
              <th title="Share scoring above the controls’ 90th percentile">Above controls</th>
            </tr>
          </thead>
          <tbody>
            {summary.bands.map((band, i) => (
              <tr key={i} style={{ cursor: 'default', borderTop: `1px solid ${colors.grid}` }}>
                <td>
                  {formatFreq(band.lo)}–{formatFreq(band.hi)}
                </td>
                <td>{band.n}</td>
                <td>{formatScore(band.mean)}</td>
                <td>{formatScore(band.median)}</td>
                <td>{pct(band.above_controls)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
