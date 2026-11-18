/**
 * The app's colors, in the homepage's manner (components/home/homeTheme.ts): the dark app sits on
 * the theme's ink with white text, the light app is black on white, rules are hairlines, and the
 * one color is the theme's accent, which marks the hovered and the active.
 *
 * The same values reach CSS as custom properties (--oz-*, see styles/app.css) and TypeScript as
 * literals, for canvases, three.js and exported SVGs, which can't read custom properties.
 */

import type { CSSProperties } from 'react'
import { THEME, mix, readableOn } from '../components/home/homeTheme'

export interface AppPalette {
  isLight: boolean
  /** The page */
  bg: string
  /** Text, icons, solid buttons */
  fg: string
  /** Supporting text */
  fgDim: string
  /** Placeholders, the inactive and the disabled */
  fgFaint: string
  /** Hairlines between sections */
  rule: string
  /** Outlines of controls */
  ruleStrong: string
  /** A wash under the hovered */
  fill: string
  /** A wash under the selected */
  fillStrong: string
  /** The theme's accent, as a fill */
  accent: string
  /** Text on the accent */
  onAccent: string
  /** The accent as text or a line: the accent on ink, black on white (where the accent can't be read) */
  mark: string
  danger: string
  warn: string
  ok: string
}

const DARK: AppPalette = {
  isLight: false,
  bg: THEME.ink,
  fg: '#ffffff',
  fgDim: 'rgba(255, 255, 255, 0.7)',
  fgFaint: 'rgba(255, 255, 255, 0.45)',
  rule: 'rgba(255, 255, 255, 0.13)',
  ruleStrong: 'rgba(255, 255, 255, 0.28)',
  fill: 'rgba(255, 255, 255, 0.05)',
  fillStrong: 'rgba(255, 255, 255, 0.1)',
  accent: THEME.accent,
  onAccent: readableOn(THEME.accent),
  mark: THEME.accent,
  danger: '#ff6b6b',
  warn: '#fab219',
  ok: '#7fe38c',
}

const LIGHT: AppPalette = {
  isLight: true,
  bg: '#ffffff',
  fg: '#000000',
  fgDim: 'rgba(0, 0, 0, 0.68)',
  fgFaint: 'rgba(0, 0, 0, 0.45)',
  rule: 'rgba(0, 0, 0, 0.14)',
  ruleStrong: 'rgba(0, 0, 0, 0.32)',
  fill: 'rgba(0, 0, 0, 0.04)',
  fillStrong: 'rgba(0, 0, 0, 0.08)',
  accent: THEME.figureAccent,
  onAccent: readableOn(THEME.figureAccent),
  mark: '#000000',
  danger: '#c62828',
  warn: '#b77900',
  ok: '#1a7f37',
}

export function appPalette(isLight: boolean): AppPalette {
  return isLight ? LIGHT : DARK
}

/** The palette as the custom properties styles/app.css reads. */
export function paletteVars(palette: AppPalette): Record<string, string> {
  return {
    '--oz-bg': palette.bg,
    '--oz-fg': palette.fg,
    '--oz-fg-dim': palette.fgDim,
    '--oz-fg-faint': palette.fgFaint,
    '--oz-rule': palette.rule,
    '--oz-rule-strong': palette.ruleStrong,
    '--oz-fill': palette.fill,
    '--oz-fill-strong': palette.fillStrong,
    '--oz-accent': palette.accent,
    '--oz-on-accent': palette.onAccent,
    '--oz-mark': palette.mark,
    '--oz-danger': palette.danger,
    '--oz-warn': palette.warn,
    '--oz-ok': palette.ok,
  }
}

/**
 * The same colors by reference, for inline styles: they follow the theme without a re-render.
 * Not for canvases or anything serialized out of the page.
 */
export const VAR = {
  bg: 'var(--oz-bg)',
  fg: 'var(--oz-fg)',
  fgDim: 'var(--oz-fg-dim)',
  fgFaint: 'var(--oz-fg-faint)',
  rule: 'var(--oz-rule)',
  ruleStrong: 'var(--oz-rule-strong)',
  fill: 'var(--oz-fill)',
  fillStrong: 'var(--oz-fill-strong)',
  accent: 'var(--oz-accent)',
  onAccent: 'var(--oz-on-accent)',
  mark: 'var(--oz-mark)',
  danger: 'var(--oz-danger)',
  warn: 'var(--oz-warn)',
  ok: 'var(--oz-ok)',
} as const

export type InlineVars = CSSProperties & Record<`--${string}`, string>

/** A hex color's channels as "r, g, b", for building rgba() strings on a canvas. */
export function rgbChannels(hex: string): string {
  return [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16)).join(', ')
}

/**
 * A sequential scale for heatmaps (t from 0 to 1), after the homepage: on ink, the theme's ramp
 * from the page's dark up to its palest, as the hero's mosaic is colored; on white, grays from
 * white to black, as the light section draws its cells.
 */
export function sequentialScale(isLight: boolean): (t: number) => string {
  const stops = isLight ? ['#ffffff', '#d4d4d4', '#8a8a8a', '#3a3a3a', '#000000'] : [THEME.ink, ...THEME.ramps[0]]
  return (t) => {
    const at = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1)
    const i = Math.min(stops.length - 2, Math.floor(at))
    return mix(stops[i], stops[i + 1], at - i)
  }
}

/** Black or white, whichever reads on a color of the sequential scale at t. */
export function inkOnSequential(isLight: boolean, t: number): string {
  return isLight ? (t > 0.45 ? '#ffffff' : '#000000') : t > 0.5 ? '#000000' : '#ffffff'
}
