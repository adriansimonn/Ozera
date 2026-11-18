/**
 * Shared components of the labs' charts and controls (Probe Lab, Geometry Lab): the hover
 * tooltip, segmented controls, findings, stat tiles, run bars, placeholders, and the styles
 * they use, scoped under .lab-kit (the root class of every page or widget that uses the kit).
 * Colors come from the app's custom properties (styles/app.css); buttons and chips are the app's.
 */

import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import { EmptyState, type EmptyGlyph } from '../common/EmptyState'
import { Spinner } from '../common/Spinner'
import { useLabColors } from './labTheme'

export interface Finding {
  level: 'warning' | 'info' | 'good'
  text: ReactNode
}

/** Short findings about a result, each marked by its level's icon. */
export function Findings({ items }: { items: Finding[] }) {
  const colors = useLabColors()
  if (items.length === 0) return null
  return (
    <ul className="pl-checks">
      {items.map((f, i) => (
        <li key={i} className="pl-check">
          {f.level === 'warning' ? (
            <AlertTriangle className="pl-icon" aria-label="Warning" style={{ color: colors.warning }} />
          ) : f.level === 'good' ? (
            <CheckCircle2 className="pl-icon" aria-label="Good" style={{ color: colors.good }} />
          ) : (
            <Info className="pl-icon" aria-label="Note" style={{ color: colors.dim }} />
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
  /** What the number means, on hover */
  title?: string
}

export function StatTiles({ tiles }: { tiles: Tile[] }) {
  return (
    <div className="pl-tiles">
      {tiles.map((tile) => (
        <div key={tile.label} className="pl-tile" title={tile.title}>
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
  assist,
}: {
  note?: string | null
  warn?: boolean
  label: string
  running: boolean
  disabled?: boolean
  onRun: () => void
  /** The assistant's highlight target for the button ("page.control") */
  assist?: string
}) {
  const colors = useLabColors()
  return (
    <div className="pl-run-bar">
      <span className="pl-note" style={warn ? { color: colors.warning } : undefined}>
        {note}
      </span>
      <button
        type="button"
        className={`oz-btn oz-btn--small${running ? '' : ' oz-btn--arrow'}`}
        data-assist={assist}
        onClick={onRun}
        disabled={running || disabled}
      >
        {running && <Spinner />}
        {label}
      </button>
    </div>
  )
}

/** What a view shows before it has anything to show. */
export function Placeholder({ glyph, children }: { glyph: EmptyGlyph; children: ReactNode }) {
  return <EmptyState glyph={glyph}>{children}</EmptyState>
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

/** Styles shared by the labs' components, all scoped under .lab-kit. */
export function LabKitStyles() {
  return (
    <style>{`
      .lab-kit .pl-tooltip {
        position: absolute;
        z-index: 20;
        pointer-events: none;
        padding: 0.55rem 0.7rem;
        background: var(--oz-bg);
        border: 1px solid var(--oz-rule-strong);
        color: var(--oz-fg);
        font-size: 0.8rem;
        line-height: 1.4;
      }
      .lab-kit .pl-tooltip-title { font-weight: 500; margin-bottom: 0.25rem; }
      .lab-kit .pl-tooltip-row { display: flex; align-items: center; gap: 0.45rem; white-space: nowrap; }
      .lab-kit .pl-tooltip-key { width: 8px; height: 8px; flex-shrink: 0; }
      .lab-kit .pl-tooltip-key-space { width: 8px; flex-shrink: 0; }
      .lab-kit .pl-tooltip-value { font-weight: 500; font-variant-numeric: tabular-nums; }
      .lab-kit .pl-tooltip-label { color: var(--oz-fg-dim); }

      .lab-kit .pl-chart { position: relative; width: 100%; }
      .lab-kit .pl-chart svg { display: block; overflow: visible; }
      .lab-kit .pl-chart svg text { font-family: inherit; }
      .lab-kit .pl-chart:focus-visible { outline: 1.5px solid var(--oz-mark); outline-offset: 3px; }

      .lab-kit .pl-card-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem 0.75rem;
        min-height: 2rem;
        margin-bottom: 0.9rem;
      }
      .lab-kit .pl-card-title h3 {
        margin: 0;
        font-size: 1.0625rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        text-transform: none;
      }

      .lab-kit .pl-legend { display: flex; flex-wrap: wrap; gap: 0.35rem 1.1rem; font-size: 0.8rem; color: var(--oz-fg-dim); margin-top: 0.6rem; }
      .lab-kit .pl-legend-item { display: inline-flex; align-items: center; gap: 0.45rem; }
      .lab-kit .pl-legend-line { width: 16px; height: 2px; }
      .lab-kit .pl-legend-dot { width: 8px; height: 8px; border-radius: 50%; }
      .lab-kit .pl-legend-ring { width: 8px; height: 8px; border-radius: 50%; border: 2px solid; box-sizing: border-box; }
      .lab-kit .pl-legend-rect { width: 9px; height: 9px; }

      .lab-kit .pl-segmented { display: inline-flex; flex-wrap: wrap; gap: 6px; }
      .lab-kit .pl-segmented .oz-chip { padding: 0.42rem 0.7rem; }
      .lab-kit .pl-control-label {
        margin-bottom: 0.45rem;
        color: var(--oz-fg-dim);
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0;
        text-transform: none;
      }

      .lab-kit .pl-input, .lab-kit .pl-textarea {
        width: 100%;
        padding: 0.55rem 0.7rem;
        background: transparent;
        border: 1px solid var(--oz-rule-strong);
        color: var(--oz-fg);
        font-size: 0.9rem;
        font-family: inherit;
        outline: none;
        box-sizing: border-box;
        transition: border-color 0.2s ease;
      }
      .lab-kit .pl-textarea { resize: vertical; min-height: 84px; line-height: 1.5; }
      .lab-kit .pl-input:hover:not(:disabled):not(:focus), .lab-kit .pl-textarea:hover:not(:disabled):not(:focus) { border-color: var(--oz-fg-faint); }
      .lab-kit .pl-input:focus, .lab-kit .pl-textarea:focus { border-color: var(--oz-fg); }
      .lab-kit .pl-input::placeholder, .lab-kit .pl-textarea::placeholder { color: var(--oz-fg-faint); }
      .lab-kit .pl-input:disabled, .lab-kit .pl-textarea:disabled { border-color: var(--oz-rule); color: var(--oz-fg-faint); cursor: not-allowed; }
      .lab-kit .pl-input[type="number"] { font-variant-numeric: tabular-nums; }

      /* The second button of a tool: outlined */
      .lab-kit .pl-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        padding: 0.5rem 0.8rem;
        background: transparent;
        border: 1px solid var(--oz-rule-strong);
        color: var(--oz-fg);
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        line-height: 1.2;
        cursor: pointer;
        white-space: nowrap;
        transition: border-color 0.2s ease, color 0.2s ease;
      }
      .lab-kit .pl-button:hover:not(:disabled) { border-color: var(--oz-mark); color: var(--oz-mark); }
      .lab-kit .pl-button:disabled { border-color: var(--oz-rule); color: var(--oz-fg-faint); cursor: not-allowed; }
      .lab-kit .pl-icon { width: 16px; height: 16px; flex-shrink: 0; stroke-width: 2.25; }

      .lab-kit .pl-hint { color: var(--oz-fg-dim); font-size: 0.85rem; line-height: 1.45; }
      .lab-kit .pl-hint code { font-family: var(--oz-mono); font-size: 0.8rem; }
      .lab-kit .pl-note { color: var(--oz-fg-dim); font-size: 0.85rem; font-variant-numeric: tabular-nums; }
      .lab-kit .pl-empty { padding: 1.1rem; border: 1px dashed var(--oz-rule-strong); color: var(--oz-fg-dim); font-size: 0.875rem; text-align: center; }
      .lab-kit .pl-check-row { display: flex; gap: 0.55rem; align-items: flex-start; cursor: pointer; font-size: 0.875rem; }
      .lab-kit .pl-check-row input { margin-top: 0.2rem; }
      .lab-kit .pl-icon-button {
        display: inline-flex; padding: 0.35rem; background: none; border: 1px solid transparent; color: var(--oz-fg); cursor: pointer;
        transition: border-color 0.2s ease, color 0.2s ease;
      }
      .lab-kit .pl-icon-button:hover:not(:disabled) { border-color: var(--oz-rule-strong); color: var(--oz-mark); }
      .lab-kit .pl-icon-button:disabled { color: var(--oz-fg-faint); cursor: not-allowed; }
      .lab-kit .pl-link {
        padding: 0 0 1px; background: none; border: none; border-bottom: 1px solid var(--oz-fg-faint); color: var(--oz-fg);
        font-size: 0.85rem; font-weight: 500; text-align: left; cursor: pointer; transition: color 0.2s ease, border-color 0.2s ease;
      }
      .lab-kit .pl-link:hover { color: var(--oz-mark); border-color: var(--oz-mark); }
      .lab-kit a.pl-inline-link { color: inherit; text-decoration: underline; text-decoration-color: var(--oz-fg-faint); text-underline-offset: 3px; transition: color 0.2s ease; }
      .lab-kit a.pl-inline-link:hover { color: var(--oz-mark); text-decoration-color: currentColor; }

      .lab-kit .pl-controls-row { display: flex; flex-wrap: wrap; gap: 0.9rem 1.5rem; margin-bottom: 1.1rem; }
      .lab-kit .pl-run-bar { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-top: 0.9rem; }
      .lab-kit .pl-token { padding: 0.12rem 0.05rem; margin-right: 1px; cursor: default; }
      .lab-kit .pl-run-meta { color: var(--oz-fg-dim); font-size: 0.85rem; margin-bottom: 1rem; }
      .lab-kit .pl-result { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--oz-rule); }
      .lab-kit .pl-error-note { color: var(--oz-danger); font-size: 0.85rem; margin-top: 0.5rem; }

      /* Headline numbers, each under a rule */
      .lab-kit .pl-tiles {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
        gap: 1rem 1.25rem;
        margin-bottom: 1.25rem;
      }
      .lab-kit .pl-tile { padding-top: 0.6rem; border-top: 1px solid var(--oz-rule-strong); }
      .lab-kit .pl-tile-label { color: var(--oz-fg-dim); font-size: 0.8rem; font-weight: 500; letter-spacing: 0; text-transform: none; }
      .lab-kit .pl-tile-value { margin-top: 0.2rem; font-size: 1.6rem; font-weight: 500; letter-spacing: -0.03em; line-height: 1.15; font-variant-numeric: tabular-nums; }
      .lab-kit .pl-tile-hint { margin-top: 0.2rem; color: var(--oz-fg-dim); font-size: 0.78rem; }
      .lab-kit .pl-checks { list-style: none; margin: 0 0 1.1rem; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
      .lab-kit .pl-check { display: flex; gap: 0.55rem; align-items: flex-start; font-size: 0.875rem; line-height: 1.45; }
      .lab-kit .pl-check .pl-icon { margin-top: 0.12rem; }

      .lab-kit .pl-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
      .lab-kit .pl-table th { text-align: right; font-size: 0.8rem; font-weight: 500; color: var(--oz-fg-dim); padding: 0.45rem 0.6rem; position: sticky; top: 0; background: var(--oz-bg); }
      .lab-kit .pl-table th:first-child, .lab-kit .pl-table td:first-child { text-align: left; }
      .lab-kit .pl-table td { text-align: right; padding: 0.4rem 0.6rem; }
      .lab-kit .pl-table tbody tr { cursor: pointer; }
      .lab-kit .pl-table tbody tr:hover { background: var(--oz-fill); }
      .lab-kit .pl-feature-link { display: inline-flex; align-items: center; gap: 0.2rem; font-weight: 500; text-decoration: none; font-variant-numeric: tabular-nums; transition: color 0.2s ease; }
      .lab-kit .pl-feature-link:hover { color: var(--oz-mark); }
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
  return (
    <div className="pl-segmented" role="radiogroup" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          className="oz-chip"
          aria-checked={option.value === value}
          title={option.title}
          disabled={disabled || option.disabled}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
