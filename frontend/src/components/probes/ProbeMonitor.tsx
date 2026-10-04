/**
 * A probe monitoring a generation on the Generate page, the way production activation monitors
 * work: the probe's score of every token as the model reads it, as a trace over the sequence
 * and on the tokens themselves. Scores stream in with the text, one token behind it (a token
 * is scored when the model reads it to predict the next one).
 *
 * Carries the probe-lab class so the Probe Lab's shared styles (tooltip, tokens, legend) apply.
 */

import { useId, useState, type PointerEvent } from 'react'
import type { MonitorProbeInfo, MonitorTrace } from '../../types/probes'
import { ChartTooltip, ProbeLabStyles } from './chartKit'
import { divergingColor, inkOn, useElementWidth, useProbeColors } from './probeTheme'
import { formatScore } from './probeUtils'

interface ProbeMonitorProps {
  trace: MonitorTrace
  probe: MonitorProbeInfo
  /** Still generating */
  live?: boolean
}

const HEIGHT = 130
const MARGIN = { top: 14, right: 12, bottom: 22, left: 40 }

function visibleToken(token: string): string {
  return token.replace(/\n/g, '↵\n').replace(/\t/g, '→\t')
}

/** The scale scores are drawn and colored on: the probe's training spread, or the generated scores' (robust) range. */
function scoreScale(trace: MonitorTrace, probe: MonitorProbeInfo): number {
  if (probe.scale) return probe.scale
  const generated = trace.scores.slice(trace.promptTokens).filter((s): s is number => s != null).map(Math.abs).sort((a, b) => a - b)
  if (generated.length === 0) return 1
  return Math.max(generated[Math.floor(0.98 * (generated.length - 1))], 1e-6)
}

function ScoreTrace({ trace, probe }: { trace: MonitorTrace; probe: MonitorProbeInfo }) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const clipId = useId().replace(/:/g, '')

  const n = trace.tokens.length
  const scale = scoreScale(trace, probe)
  const range = Math.max(scale * 1.6, ...trace.scores.slice(trace.promptTokens).map((s) => (s == null ? 0 : Math.min(Math.abs(s), scale * 3))))
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 60)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const x = (i: number) => MARGIN.left + (n > 1 ? (i / (n - 1)) * innerW : innerW / 2)
  const clamp = (v: number) => Math.max(-range, Math.min(range, v))
  const y = (v: number) => MARGIN.top + (1 - (clamp(v) + range) / (2 * range)) * innerH
  const zero = y(0)

  // Line and area, with gaps at unscored positions
  let line = ''
  let area = ''
  let run: number[] = []
  const flush = () => {
    if (run.length === 0) return
    const pts = run.map((i) => `${x(i).toFixed(1)},${y(trace.scores[i] as number).toFixed(1)}`)
    line += `M${pts.join('L')}`
    area += `M${x(run[0]).toFixed(1)},${zero.toFixed(1)}L${pts.join('L')}L${x(run[run.length - 1]).toFixed(1)},${zero.toFixed(1)}Z`
    run = []
  }
  trace.scores.forEach((s, i) => (s == null ? flush() : run.push(i)))
  flush()

  const promptEnd = Math.min(trace.promptTokens, n)
  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    const index = Math.max(0, Math.min(n - 1, Math.round(((px - MARGIN.left) / innerW) * (n - 1))))
    setHover({ index, px, py: e.clientY - box.top })
  }
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }
  const hovered = hover ? trace.scores[hover.index] : null

  return (
    <div ref={ref} className="pl-chart">
      <svg width={width} height={HEIGHT} role="img" aria-label={`Probe score at each of ${n} tokens`}>
        <defs>
          <clipPath id={`${clipId}-above`}>
            <rect x={0} y={0} width={width} height={zero} />
          </clipPath>
          <clipPath id={`${clipId}-below`}>
            <rect x={0} y={zero} width={width} height={HEIGHT - zero} />
          </clipPath>
        </defs>
        {promptEnd > 0 && (
          <g>
            <rect x={MARGIN.left} y={MARGIN.top} width={Math.max(x(Math.max(promptEnd - 1, 0)) - MARGIN.left, 2)} height={innerH} fill={colors.hover} />
            <text x={MARGIN.left + 4} y={MARGIN.top + 10} fontSize={10} fill={colors.text} {...halo}>
              prompt
            </text>
          </g>
        )}
        {[range, range / 2, -range / 2, -range].map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <text x={MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
              {formatScore(t)}
            </text>
          </g>
        ))}
        <path d={area} fill={colors.classes[1]} opacity={0.28} clipPath={`url(#${clipId}-above)`} />
        <path d={area} fill={colors.classes[0]} opacity={0.28} clipPath={`url(#${clipId}-below)`} />
        <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={zero} y2={zero} stroke={colors.baseline} />
        <path d={line} fill="none" stroke={colors.text} strokeWidth={1.5} strokeLinejoin="round" opacity={0.85} />
        <text x={MARGIN.left + innerW} y={MARGIN.top + 10} textAnchor="end" fontSize={10} fill={colors.text} {...halo}>
          ↑ {probe.labelNames[1]}
        </text>
        <text x={MARGIN.left + innerW} y={MARGIN.top + innerH - 4} textAnchor="end" fontSize={10} fill={colors.text} {...halo}>
          ↓ {probe.labelNames[0]}
        </text>
        <text x={MARGIN.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" fontSize={10} fill={colors.text}>
          Token
        </text>
        {hover && (
          <g>
            <line x1={x(hover.index)} x2={x(hover.index)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} />
            {hovered != null && (
              <circle cx={x(hover.index)} cy={y(hovered)} r={4} fill={divergingColor(colors, hovered / scale)} stroke={colors.surface} strokeWidth={2} />
            )}
          </g>
        )}
        <rect
          x={MARGIN.left - 6}
          y={MARGIN.top}
          width={innerW + 12}
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
          title={`“${trace.tokens[hover.index]}” · token ${hover.index + 1}${hover.index < trace.promptTokens ? ' (prompt)' : ''}`}
          rows={[
            {
              label: hovered == null ? 'not read yet' : `→ ${probe.labelNames[hovered > 0 ? 1 : 0]}`,
              value: hovered == null ? '–' : formatScore(hovered),
            },
          ]}
        />
      )}
    </div>
  )
}

export function ProbeMonitor({ trace, probe, live }: ProbeMonitorProps) {
  const colors = useProbeColors()
  const [showPrompt, setShowPrompt] = useState(false)
  const scale = scoreScale(trace, probe)

  const generated = trace.scores
    .map((s, i) => ({ s, i }))
    .filter((p): p is { s: number; i: number } => p.i >= trace.promptTokens && p.s != null)
  const peak = generated.reduce<{ s: number; i: number } | null>((m, p) => (m == null || p.s > m.s ? p : m), null)
  const positive = generated.filter((p) => p.s > 0).length
  const unscored = trace.tokens.length - trace.scores.filter((s) => s != null).length

  const tokens = trace.tokens
    .map((token, i) => ({ token, i, s: trace.scores[i] }))
    .filter((t) => showPrompt || t.i >= trace.promptTokens)

  return (
    <div className="probe-lab probe-monitor" style={{ borderColor: colors.border, color: colors.text }}>
      <div className="pm-head">
        <span className="pm-title">Probe monitor</span>
        <span className="pl-note">
          {probe.name} · layer {probe.layer}
        </span>
        {live && <span className="pm-live" style={{ color: colors.classes[1] }}>● live</span>}
      </div>
      <div className="pm-summary">
        {generated.length === 0 ? (
          <span className="pl-note">{live ? 'Waiting for the first generated token to be read…' : 'No generated tokens were scored.'}</span>
        ) : (
          <>
            <span>
              <strong>{positive}</strong> of {generated.length} generated tokens read as <strong>{probe.labelNames[1]}</strong>
            </span>
            {peak && (
              <span>
                highest <strong>{formatScore(peak.s)}</strong> at “{trace.tokens[peak.i].trim() || trace.tokens[peak.i]}”
              </span>
            )}
          </>
        )}
      </div>

      {trace.tokens.length > 1 && <ScoreTrace trace={trace} probe={probe} />}

      <div className="pm-tokens-head">
        <button type="button" className="pm-link" onClick={() => setShowPrompt((v) => !v)} style={{ color: colors.text }}>
          {showPrompt ? 'Hide' : 'Show'} the prompt ({trace.promptTokens} tokens)
        </button>
        {unscored > 0 && !live && <span className="pl-note">Faded tokens weren&apos;t read after being generated, so have no score.</span>}
      </div>
      <div className="pl-tokens pm-tokens">
        {tokens.map(({ token, i, s }) => {
          const background = s == null ? 'transparent' : divergingColor(colors, s / scale)
          return (
            <span
              key={i}
              className="pl-token"
              title={s == null ? 'not read yet' : formatScore(s)}
              style={{
                background,
                color: s == null ? colors.text : inkOn(background),
                opacity: s == null ? 0.45 : i < trace.promptTokens ? 0.6 : 1,
              }}
            >
              {visibleToken(token)}
            </span>
          )
        })}
      </div>
      <div className="pl-scale">
        <span>{probe.labelNames[0]}</span>
        <span
          className="pl-scale-bar"
          style={{
            background: `linear-gradient(to right, ${divergingColor(colors, -1)}, ${divergingColor(colors, 0)}, ${divergingColor(colors, 1)})`,
          }}
        />
        <span>{probe.labelNames[1]}</span>
        <span className="pl-note">
          ±{formatScore(scale)} {probe.scale ? '(2 standard deviations of training scores)' : ''}
        </span>
      </div>

      <ProbeLabStyles />
      <style>{`
        .probe-monitor { border: 1px solid; padding: 0.75rem 0.875rem; margin-top: 0.75rem; text-align: left; }
        .probe-monitor .pm-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem; margin-bottom: 0.3rem; }
        .probe-monitor .pm-title { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
        .probe-monitor .pm-live { font-size: 0.7rem; margin-left: auto; }
        .probe-monitor .pm-summary { display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.78rem; margin-bottom: 0.4rem; }
        .probe-monitor .pm-tokens-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.75rem; margin: 0.35rem 0; }
        .probe-monitor .pm-link { background: none; border: none; padding: 0; font-size: 0.72rem; text-decoration: underline; cursor: pointer; }
        .probe-monitor .pl-tokens {
          font-family: 'JetBrains Mono', ui-monospace, monospace;
          font-size: 0.78rem;
          line-height: 1.85;
          white-space: pre-wrap;
          word-break: break-word;
          max-height: 220px;
          overflow-y: auto;
        }
        .probe-monitor .pl-scale { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.72rem; }
        .probe-monitor .pl-scale-bar { width: 120px; height: 8px; border-radius: 4px; }
      `}</style>
    </div>
  )
}
