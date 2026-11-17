/**
 * The homepage's colors. The dark sections sit on the theme's ink with white text, the light
 * section is black on white, and everything colored (the hero's animation, the figure's accent,
 * the token field's mark, the hover and active states) comes from the theme.
 */

export interface Theme {
  /** The dark sections' background */
  ink: string
  /** The accent on the dark sections */
  accent: string
  /** The accent of the figure in the light section */
  figureAccent: string
  /** The streaks' colors; the earlier ones are drawn more often */
  streaks: string[]
  /** The mosaic's ribbons, each as stops from its fringe to its core */
  ramps: string[][]
  /** Stray cells among the ribbons */
  sprinkles: string[]
  /** The token field's mark, from its top to its bottom */
  mark: [string, string]
}

export const THEMES: Record<string, Theme> = {
  volt: {
    ink: '#0a0b09',
    accent: '#c8f53a',
    figureAccent: '#c8f53a',
    streaks: ['#c8f53a', '#ffffff', '#6fd67a', '#eaff9c', '#3aa05c'],
    ramps: [
      ['#141f0a', '#4f7a14', '#c8f53a', '#f6ffd6'],
      ['#0c2112', '#27833f', '#7fe38c', '#edffee'],
      ['#1f220a', '#8f9a1a', '#e9f25a', '#fdffe0'],
    ],
    sprinkles: ['#ffffff', '#3aa05c'],
    mark: ['#c8f53a', '#c8f53a'],
  },
  cobalt: {
    ink: '#05060b',
    accent: '#4f7dff',
    figureAccent: '#2b55f5',
    streaks: ['#4f7dff', '#a9c4ff', '#ffffff', '#7a5cff', '#22b8f0'],
    ramps: [
      ['#0a1238', '#2b55f5', '#9dbbff', '#f2f6ff'],
      ['#190f40', '#6a4cf0', '#c0b2ff', '#f6f3ff'],
      ['#062838', '#1899d6', '#a6e2ff', '#effaff'],
    ],
    sprinkles: ['#ffffff', '#7a5cff'],
    mark: ['#a9c4ff', '#4f7dff'],
  },
  viridis: {
    ink: '#07050c',
    accent: '#dfe318',
    figureAccent: '#1f9e89',
    streaks: ['#fde725', '#5ec962', '#21918c', '#ffffff', '#3b528b', '#7e4fc4'],
    ramps: [
      ['#2a1150', '#3b528b', '#21918c', '#5ec962', '#fde725', '#fdf7b8'],
      ['#1c0c36', '#472d7b', '#3b528b', '#21918c', '#5ec962', '#d9f5b8'],
    ],
    sprinkles: ['#ffffff', '#7e4fc4'],
    mark: ['#fde725', '#21918c'],
  },
  crimson: {
    ink: '#0b0707',
    accent: '#ff3d5a',
    figureAccent: '#f0203f',
    streaks: ['#ff3d5a', '#ffb0bb', '#ffffff', '#c02a8a', '#ff7a8c'],
    ramps: [
      ['#33060f', '#b3122c', '#ff3d5a', '#ffe0e4'],
      ['#2a0a2c', '#a02a86', '#ee8ed0', '#fdeefa'],
      ['#2e0d14', '#d9516a', '#ffb0bb', '#fff3f5'],
    ],
    sprinkles: ['#ffffff', '#c02a8a'],
    mark: ['#ffb0bb', '#ff3d5a'],
  },
}

// For comparing themes: "?theme=<name>" on the address the app is loaded from picks one
const requested = new URLSearchParams(window.location.search).get('theme')
export const THEME: Theme = (requested && THEMES[requested]) || THEMES.volt

export const INK = THEME.ink
export const PAPER = '#ffffff'

const channels = (hex: string) => [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16))

/** The color a fraction t of the way from one hex color to another. */
export function mix(from: string, to: string, t: number): string {
  const a = channels(from)
  const b = channels(to)
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * Math.min(1, Math.max(0, t)))).join(', ')})`
}

/** A number of colors spread evenly along a list of stops. */
export function ramp(stops: string[], steps: number): string[] {
  return Array.from({ length: steps }, (_, n) => {
    const at = (n / (steps - 1)) * (stops.length - 1)
    const i = Math.min(stops.length - 2, Math.floor(at))
    return mix(stops[i], stops[i + 1], at - i)
  })
}

/** Black or white, whichever reads better on a hex color. */
export function readableOn(hex: string): string {
  const [r, g, b] = channels(hex)
  return r * 0.299 + g * 0.587 + b * 0.114 > 150 ? '#000000' : '#ffffff'
}
