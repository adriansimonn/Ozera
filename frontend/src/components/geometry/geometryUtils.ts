/**
 * Pure helpers for the Geometry Lab: value colors, cosine matrices, picking a layer, labels,
 * chart paths, and encoding coordinates to send back for comparisons.
 */

import type { LabColors } from '../lab/labTheme'
import type { ConceptStructure, GeometryEstimate, GeometryRead, GeometryRun, PerPosition, RunConcept, WireTensor } from '../../types/geometry'

export const READ_LABELS: Record<GeometryRead, string> = {
  value_last: 'Value',
  value_mean: 'Value mean',
  template_last: 'End',
}
export const READ_TITLES: Record<GeometryRead, string> = {
  value_last: "The value's last token",
  value_mean: "The mean over the value's tokens",
  template_last: "The template's last token, where the model has read the whole context",
}
export const STRUCTURE_LABELS: Record<ConceptStructure, string> = {
  cyclic: 'Cyclic',
  ordinal: 'Ordinal',
  categorical: 'Categorical',
}

export function positionLabel(position: number): string {
  return position === 0 ? 'emb' : String(position - 1)
}

export function layerName(position: number): string {
  return position === 0 ? 'Embeddings' : `Layer ${position - 1}`
}

export function formatValue(value: number | null | undefined, digits = 2): string {
  return value == null || Number.isNaN(value) ? '–' : value.toFixed(digits)
}

export function formatPercent(value: number | null | undefined): string {
  return value == null || Number.isNaN(value) ? '–' : `${Math.round(value * 100)}%`
}

export function estimateText(estimate: GeometryEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

export function runLabel(run: GeometryRun, modelName: string): string {
  return `${run.concept.name} · ${modelName} · ${READ_LABELS[run.read]}`
}

/** Example i's text: template floor(i / k) with value i % k. */
export function exampleText(concept: RunConcept, i: number): string {
  const k = concept.values.length
  return concept.templates[Math.floor(i / k)].replace('{x}', concept.values[i % k])
}

export function sameValues(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

// Value colors

/** OKLCH (L 0–1, C, h degrees) to an sRGB hex, clipped to the gamut. */
function oklch(l: number, c: number, h: number): string {
  const rad = (h * Math.PI) / 180
  const a = c * Math.cos(rad)
  const b = c * Math.sin(rad)
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3
  const linear = [
    4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
  ]
  return (
    '#' +
    linear
      .map((x) => {
        const v = Math.max(0, Math.min(1, x))
        const srgb = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055
        return Math.round(srgb * 255)
          .toString(16)
          .padStart(2, '0')
      })
      .join('')
  )
}

function parseHex(hex: string): number[] {
  const h = hex.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16))
}

function mixHex(a: string, b: string, t: number): string {
  const ca = parseHex(a)
  const cb = parseHex(b)
  return '#' + ca.map((v, i) => Math.round(v + (cb[i] - v) * t).toString(16).padStart(2, '0')).join('')
}

// The sequential blue ramp, steps 100–700
const BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

function rampAt(steps: string[], t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (steps.length - 1)
  const i = Math.min(Math.floor(x), steps.length - 2)
  return mixHex(steps[i], steps[i + 1], x - i)
}

/**
 * A color per value, by the job the color does:
 * - cyclic: an angle, so a hue wheel at constant OKLCH lightness (the values go round; red sits
 *   at a hue concept's red); labels carry identity
 * - ordinal: an order, so one blue ramp, light to dark (anchored at the dark end in dark mode)
 * - categorical: identity, so the categorical slots in order, by group when there are groups
 *   (more than eight groups or ungrouped values fall back to one color and their labels)
 */
export function valueColors(
  concept: Pick<RunConcept, 'structure' | 'values' | 'positions' | 'groups'>,
  colors: LabColors,
): string[] {
  const k = concept.values.length
  if (concept.structure === 'cyclic') {
    const lightness = colors.isLight ? 0.6 : 0.7
    return concept.values.map((_, i) => {
      const degrees = concept.positions ? concept.positions[i] : (360 * i) / k
      return oklch(lightness, 0.13, degrees + 29)
    })
  }
  if (concept.structure === 'ordinal') {
    // Light: steps 250–700; dark: 600 (near the surface) up to 100
    const steps = colors.isLight ? BLUE_RAMP.slice(3) : BLUE_RAMP.slice(0, 11).reverse()
    return concept.values.map((_, i) => rampAt(steps, k > 1 ? i / (k - 1) : 0))
  }
  const keys = concept.groups ?? concept.values
  const distinct = Array.from(new Set(keys))
  if (distinct.length > colors.series.length) return concept.values.map(() => colors.series[0])
  return keys.map((key) => colors.series[distinct.indexOf(key)])
}

/**
 * A probability's color (t in 0–1, of the chart's largest): from near the surface at 0 to the
 * dark end of the blue ramp in light mode, the light end in dark mode.
 */
export function sequentialColor(colors: LabColors, t: number): string {
  const steps = colors.isLight ? ['#f2f1ee', ...BLUE_RAMP] : ['#1c1c1b', ...BLUE_RAMP.slice().reverse()]
  return rampAt(steps, t)
}

/** Whether a concept's class means are joined in a closed loop. */
export function closedPath(structure: ConceptStructure): boolean {
  return structure === 'cyclic'
}

// Geometry

/** An SVG path through a series' points, broken where values are missing. */
export function linePath(values: PerPosition, x: (i: number) => number, y: (v: number) => number): string {
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

/** Cosine similarity between every pair of (centered) class means. [k, k] */
export function cosineMatrix(means: number[][]): number[][] {
  const norms = means.map((row) => Math.sqrt(row.reduce((sum, v) => sum + v * v, 0)))
  return means.map((a, i) =>
    means.map((b, j) => {
      if (norms[i] === 0 || norms[j] === 0) return 0
      let dot = 0
      for (let c = 0; c < a.length; c++) dot += a[c] * b[c]
      return dot / (norms[i] * norms[j])
    }),
  )
}

/** A hypothesis's margin over its shuffled baseline (best of the shuffles) at a position. */
export function margin(r2: PerPosition, shuffledMax: PerPosition, position: number): number | null {
  const a = r2[position]
  const b = shuffledMax[position]
  return a == null || b == null ? null : a - b
}

/**
 * The layer (position >= 1) where some hypothesis beats its best shuffled fit by the most, at
 * subspace size index dimsIndex; ties go to the earliest. Falls back to the layer whose class
 * means hold the most of the examples' variance.
 */
export function bestPosition(run: GeometryRun, dimsIndex: number): number {
  const positions = run.num_layers + 1
  let best = 1
  let bestMargin = -Infinity
  for (const h of run.shapes?.hypotheses ?? []) {
    for (let p = 1; p < positions; p++) {
      const m = margin(h.r2[dimsIndex], h.shuffled_max[dimsIndex], p)
      if (m != null && m > bestMargin) {
        bestMargin = m
        best = p
      }
    }
  }
  if (bestMargin > -Infinity) return best
  const between = run.between_fraction
  for (let p = 1; p < positions; p++) if ((between[p] ?? -1) > (between[best] ?? -1)) best = p
  return best
}

/** The default subspace size: 3 (the components the Shape view plots) when it's tested, else the largest. */
export function defaultDims(dims: number[]): number {
  return dims.includes(3) ? 3 : dims[dims.length - 1] ?? 3
}

/** Nested arrays as a base64 float32 tensor (the API's wire format). */
export function encodeWire(values: number[][] | number[][][], shape: number[]): WireTensor {
  const flat = new Float32Array(shape.reduce((a, b) => a * b, 1))
  let i = 0
  const walk = (v: unknown) => {
    if (Array.isArray(v)) v.forEach(walk)
    else flat[i++] = v as number
  }
  walk(values)
  const bytes = new Uint8Array(flat.buffer)
  let binary = ''
  const chunk = 0x8000
  for (let j = 0; j < bytes.length; j += chunk) binary += String.fromCharCode(...bytes.subarray(j, j + chunk))
  return { encoding: 'base64_float32', values: btoa(binary), shape }
}

/** Save an on-screen SVG chart as a standalone .svg file (its fills and strokes are attributes). */
export function downloadSvg(svg: SVGSVGElement, filename: string, background: string): void {
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('font-family', 'system-ui, -apple-system, "Segoe UI", sans-serif')
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  rect.setAttribute('width', '100%')
  rect.setAttribute('height', '100%')
  rect.setAttribute('fill', background)
  clone.insertBefore(rect, clone.firstChild)
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
