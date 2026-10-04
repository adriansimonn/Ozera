/**
 * Shared components of the Probe Lab's charts and controls: the hover tooltip, segmented
 * controls, findings, stat tiles, run bars, placeholders, and the styles every Probe Lab
 * component uses (scoped under .probe-lab).
 */

import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Info, Play, type LucideIcon } from 'lucide-react'
import { useProbeColors } from './probeTheme'

export interface Finding {
  level: 'warning' | 'info' | 'good'
  text: ReactNode
}

/** Short findings about a result, each marked by its level's icon. */
export function Findings({ items }: { items: Finding[] }) {
  const colors = useProbeColors()
  if (items.length === 0) return null
  return (
    <ul className="pl-checks">
      {items.map((f, i) => (
        <li key={i} className="pl-check">
          {f.level === 'warning' ? (
            <AlertTriangle className="pl-icon" aria-label="Warning" style={{ color: colors.warning }} />
          ) : f.level === 'good' ? (
            <CheckCircle2 className="pl-icon" aria-label="Good" style={{ color: colors.methods.diff_means }} />
          ) : (
            <Info className="pl-icon" aria-label="Note" style={{ opacity: 0.6 }} />
          )}
          <span>{f.text}</span>
        </li>
      ))}
    </ul>
  )
}

export interface Tile {
  label: string
  value: string
  hint?: string
}

export function StatTiles({ tiles }: { tiles: Tile[] }) {
  const colors = useProbeColors()
  return (
    <div className="pl-tiles">
      {tiles.map((tile) => (
        <div key={tile.label} className="pl-tile" style={{ borderColor: colors.border, background: colors.panel }}>
          <div className="pl-tile-label">{tile.label}</div>
          <div className="pl-tile-value">{tile.value}</div>
          {tile.hint && <div className="pl-tile-hint">{tile.hint}</div>}
        </div>
      ))}
    </div>
  )
}

/** A tool's action button, with its cost estimate (or why it can't run) beside it. */
export function RunBar({
  note,
  warn,
  label,
  running,
  disabled,
  onRun,
}: {
  note?: string | null
  warn?: boolean
  label: string
  running: boolean
  disabled?: boolean
  onRun: () => void
}) {
  const colors = useProbeColors()
  return (
    <div className="pl-run-bar">
      <span className="pl-note" style={warn ? { color: colors.warning } : undefined}>
        {note}
      </span>
      <button
        type="button"
        className="pl-button"
        onClick={onRun}
        disabled={running || disabled}
        style={{ borderColor: colors.border, color: colors.text }}
      >
        {running ? (
          <span className="pl-spinner" style={{ borderColor: colors.border, borderTopColor: colors.text }} />
        ) : (
          <Play className="pl-icon" aria-hidden="true" />
        )}
        {label}
      </button>
    </div>
  )
}

/** What a view shows before it has anything to show. */
export function Placeholder({ icon: Icon, children }: { icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="pl-placeholder">
      <Icon className="pl-placeholder-icon" aria-hidden="true" />
      <p>{children}</p>
    </div>
  )
}

export interface TooltipRow {
  key?: string
  label: string
  value: string
}

/**
 * A hover readout positioned inside a relatively positioned chart container. Values lead
 * (strong), labels follow; series rows carry a short line key in the series color.
 */
export function ChartTooltip({
  x,
  y,
  containerWidth,
  title,
  rows,
  children,
}: {
  x: number
  y: number
  containerWidth: number
  title?: string
  rows?: TooltipRow[]
  children?: ReactNode
}) {
  const colors = useProbeColors()
  const width = 240
  const left = x + 14 + width > containerWidth ? Math.max(0, x - width - 14) : x + 14
  return (
    <div
      className="pl-tooltip"
      role="status"
      style={{
        left,
        top: Math.max(0, y - 10),
        maxWidth: width,
        background: colors.isLight ? 'rgba(255,255,255,0.97)' : 'rgba(20,20,20,0.97)',
        borderColor: colors.border,
        color: colors.text,
      }}
    >
      {title && <div className="pl-tooltip-title">{title}</div>}
      {rows?.map((row, i) => (
        <div key={i} className="pl-tooltip-row">
          {row.key ? <span className="pl-tooltip-key" style={{ background: row.key }} /> : <span className="pl-tooltip-key-space" />}
          <span className="pl-tooltip-value">{row.value}</span>
          <span className="pl-tooltip-label">{row.label}</span>
        </div>
      ))}
      {children}
    </div>
  )
}

/** Styles shared by the Probe Lab's components, all scoped under .probe-lab. */
export function ProbeLabStyles() {
  return (
    <style>{`
      .probe-lab .pl-tooltip {
        position: absolute;
        z-index: 20;
        pointer-events: none;
        padding: 0.5rem 0.625rem;
        border: 1px solid;
        font-size: 0.75rem;
        line-height: 1.35;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
      }
      .probe-lab .pl-tooltip-title { font-weight: 600; margin-bottom: 0.25rem; }
      .probe-lab .pl-tooltip-row { display: flex; align-items: center; gap: 0.4rem; white-space: nowrap; }
      .probe-lab .pl-tooltip-key { width: 12px; height: 2px; border-radius: 1px; flex-shrink: 0; }
      .probe-lab .pl-tooltip-key-space { width: 12px; flex-shrink: 0; }
      .probe-lab .pl-tooltip-value { font-weight: 600; font-variant-numeric: tabular-nums; }
      .probe-lab .pl-tooltip-label { opacity: 0.85; }

      .probe-lab .pl-chart { position: relative; width: 100%; }
      .probe-lab .pl-chart svg { display: block; overflow: visible; }
      .probe-lab .pl-chart svg text { font-family: inherit; }
      .probe-lab .pl-chart:focus-visible { outline: 2px solid rgba(59, 130, 246, 0.6); outline-offset: 2px; }

      .probe-lab .pl-card-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem 0.75rem;
        min-height: 1.75rem;
        margin-bottom: 0.625rem;
      }
      .probe-lab .pl-card-title h3 {
        margin: 0;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .probe-lab .pl-legend { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; font-size: 0.75rem; margin-top: 0.5rem; }
      .probe-lab .pl-legend-item { display: inline-flex; align-items: center; gap: 0.4rem; }
      .probe-lab .pl-legend-line { width: 16px; height: 2px; border-radius: 1px; }
      .probe-lab .pl-legend-dot { width: 8px; height: 8px; border-radius: 50%; }
      .probe-lab .pl-legend-ring { width: 8px; height: 8px; border-radius: 50%; border: 2px solid; box-sizing: border-box; }
      .probe-lab .pl-legend-rect { width: 10px; height: 10px; border-radius: 2px; }

      .probe-lab .pl-segmented { display: inline-flex; border: 1px solid; }
      .probe-lab .pl-segmented button {
        padding: 0.35rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 500;
        background: transparent;
        border: none;
        cursor: pointer;
        color: inherit;
        white-space: nowrap;
      }
      .probe-lab .pl-segmented button:disabled { opacity: 0.4; cursor: not-allowed; }
      .probe-lab .pl-control-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
      }

      .probe-lab .pl-input, .probe-lab .pl-textarea {
        width: 100%;
        padding: 0.5rem 0.625rem;
        border: 1px solid;
        font-size: 0.85rem;
        font-family: inherit;
        outline: none;
        box-sizing: border-box;
      }
      .probe-lab .pl-textarea { resize: vertical; min-height: 84px; }
      .probe-lab .pl-input:focus, .probe-lab .pl-textarea:focus { border-color: rgba(59, 130, 246, 0.5) !important; }

      .probe-lab .pl-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        padding: 0.5rem 0.875rem;
        border: 1px solid;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        background: transparent;
        color: inherit;
        white-space: nowrap;
      }
      .probe-lab .pl-button:disabled { opacity: 0.45; cursor: not-allowed; }
      .probe-lab .pl-icon { width: 14px; height: 14px; flex-shrink: 0; }

      .probe-lab .pl-spinner {
        width: 14px;
        height: 14px;
        border: 2px solid;
        border-radius: 50%;
        animation: pl-spin 0.8s linear infinite;
      }
      @keyframes pl-spin { to { transform: rotate(360deg); } }

      .probe-lab .pl-hint { font-size: 0.72rem; line-height: 1.45; }
      .probe-lab .pl-hint code { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; }
      .probe-lab .pl-note { font-size: 0.72rem; }
      .probe-lab .pl-empty { font-size: 0.8rem; padding: 1rem; border: 1px dashed; text-align: center; }
      .probe-lab .pl-check-row { display: flex; gap: 0.4rem; align-items: flex-start; cursor: pointer; font-size: 0.8rem; }
      .probe-lab .pl-check-row input { margin-top: 0.15rem; }
      .probe-lab .pl-icon-button { background: none; border: none; padding: 0.3rem; cursor: pointer; display: inline-flex; }
      .probe-lab .pl-icon-button:disabled { opacity: 0.35; cursor: not-allowed; }
      .probe-lab .pl-link { background: none; border: none; padding: 0; font-size: 0.72rem; cursor: pointer; text-align: left; }

      .probe-lab .pl-controls-row { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; margin-bottom: 1rem; }
      .probe-lab .pl-run-bar { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-top: 0.75rem; }
      .probe-lab .pl-token { padding: 0.12rem 0.05rem; border-radius: 3px; margin-right: 1px; cursor: default; }
      .probe-lab .pl-run-meta { font-size: 0.75rem; margin-bottom: 0.875rem; }
      .probe-lab .pl-result { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid; }
      .probe-lab .pl-error-note { color: #ef4444; font-size: 0.75rem; margin-top: 0.5rem; }
      .probe-lab .pl-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
      }
      .probe-lab .pl-placeholder-icon { width: 36px; height: 36px; opacity: 0.3; margin-bottom: 0.75rem; }
      .probe-lab .pl-placeholder p { margin: 0; font-size: 0.8rem; max-width: 360px; line-height: 1.5; }

      .probe-lab .pl-tiles {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 0.5rem;
        margin-bottom: 0.875rem;
      }
      .probe-lab .pl-tile { border: 1px solid; padding: 0.5rem 0.65rem; }
      .probe-lab .pl-tile-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
      .probe-lab .pl-tile-value { font-size: 1.15rem; font-weight: 600; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
      .probe-lab .pl-tile-hint { font-size: 0.68rem; margin-top: 0.1rem; }
      .probe-lab .pl-checks { list-style: none; margin: 0 0 0.875rem; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
      .probe-lab .pl-check { display: flex; gap: 0.5rem; align-items: flex-start; font-size: 0.78rem; line-height: 1.4; }
      .probe-lab .pl-check .pl-icon { margin-top: 0.1rem; }

      .probe-lab .pl-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; font-variant-numeric: tabular-nums; }
      .probe-lab .pl-table th { text-align: right; font-weight: 600; padding: 0.35rem 0.5rem; position: sticky; top: 0; }
      .probe-lab .pl-table th:first-child, .probe-lab .pl-table td:first-child { text-align: left; }
      .probe-lab .pl-table td { text-align: right; padding: 0.3rem 0.5rem; }
      .probe-lab .pl-table tbody tr { cursor: pointer; }
    `}</style>
  )
}

/** A row of mutually exclusive options. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
  ariaLabel,
}: {
  value: T
  options: { value: T; label: string; disabled?: boolean; title?: string }[]
  onChange: (value: T) => void
  disabled?: boolean
  ariaLabel: string
}) {
  const colors = useProbeColors()
  return (
    <div className="pl-segmented" role="radiogroup" aria-label={ariaLabel} style={{ borderColor: colors.border }}>
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.title}
            disabled={disabled || option.disabled}
            onClick={() => onChange(option.value)}
            style={{
              background: active ? (colors.isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.12)') : 'transparent',
              color: colors.text,
              fontWeight: active ? 600 : 500,
            }}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
