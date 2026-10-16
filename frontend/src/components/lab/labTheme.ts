/**
 * Colors and sizing for the labs' charts (Probe Lab and Geometry Lab).
 *
 * Colors (validated for CVD separation and contrast on the app's light and dark surfaces):
 * the two probe methods take violet and aqua; class 0 / negative scores take blue and class 1 /
 * positive scores red, the poles of the diverging scale the token heatmap uses, so "red means
 * the probe says positive" holds on every chart. Baselines are muted gray reference lines.
 * Results with the probe's direction ablated take orange (validated against both methods).
 * Probes on a few SAE features take magenta, next to the logistic regression violet (validated
 * as a pair in both modes; below 3:1 on light surfaces, so their charts carry direct labels
 * and a table view). Transfer AUROC uses its own diverging scale around chance (0.5): toward
 * blue as a probe transfers, toward orange as it ranks the other data backwards.
 *
 * series: the categorical slots in their fixed order (validated as adjacent series in both
 * modes; slots 3–5 sit below 3:1 on light surfaces, so charts using them carry direct labels
 * and a table view). The Geometry Lab's hypotheses take them in order.
 */

import { useEffect, useRef, useState } from 'react'
import type { ProbeMethod } from '../../types/probes'
import { useTheme } from '../../hooks/useTheme'

export interface LabColors {
  isLight: boolean
  methods: Record<ProbeMethod, string>
  classes: [string, string]
  ablated: string
  sae: string
  /** Categorical slots, in their fixed order */
  series: string[]
  /** Poles of the transfer AUROC scale: [AUROC 0, AUROC 1] */
  transfer: [string, string]
  midpoint: string
  baseline: string
  text: string
  surface: string
  grid: string
  axis: string
  border: string
  panel: string
  hover: string
  accent: string
  warning: string
}

const LIGHT: LabColors = {
  isLight: true,
  methods: { logreg: '#4a3aa7', diff_means: '#1baf7a' },
  classes: ['#2a78d6', '#e34948'],
  ablated: '#eb6834',
  sae: '#e87ba4',
  series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  transfer: ['#eb6834', '#184f95'],
  midpoint: '#f0efec',
  baseline: '#898781',
  text: '#1d1d1f',
  surface: '#ffffff',
  grid: 'rgba(0, 0, 0, 0.07)',
  axis: 'rgba(0, 0, 0, 0.22)',
  border: 'rgba(0, 0, 0, 0.1)',
  panel: 'rgba(0, 0, 0, 0.025)',
  hover: 'rgba(0, 0, 0, 0.05)',
  accent: '#2563eb',
  warning: '#b77900',
}

const DARK: LabColors = {
  isLight: false,
  methods: { logreg: '#9085e9', diff_means: '#199e70' },
  classes: ['#3987e5', '#e66767'],
  ablated: '#d95926',
  sae: '#d55181',
  series: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
  transfer: ['#d95926', '#6da7ec'],
  midpoint: '#383835',
  baseline: '#898781',
  text: '#ffffff',
  surface: '#0a0a0a',
  grid: 'rgba(255, 255, 255, 0.07)',
  axis: 'rgba(255, 255, 255, 0.22)',
  border: 'rgba(255, 255, 255, 0.1)',
  panel: 'rgba(255, 255, 255, 0.025)',
  hover: 'rgba(255, 255, 255, 0.06)',
  accent: 'rgba(96, 165, 250, 0.95)',
  warning: '#fab219',
}

export function useLabColors(): LabColors {
  const { isLight } = useTheme()
  return isLight ? LIGHT : DARK
}

/** A diverging color for a score: midpoint gray at 0, toward blue (negative) or red (positive). */
export function divergingColor(colors: LabColors, t: number): string {
  const clamped = Math.max(-1, Math.min(1, t))
  const pole = clamped < 0 ? colors.classes[0] : colors.classes[1]
  return mix(colors.midpoint, pole, Math.abs(clamped))
}

/** A transfer AUROC's color: midpoint gray at chance (0.5), toward blue at 1, toward orange at 0. */
export function transferColor(colors: LabColors, auroc: number): string {
  const t = Math.max(-1, Math.min(1, (auroc - 0.5) * 2))
  return mix(colors.midpoint, colors.transfer[t < 0 ? 0 : 1], Math.abs(t))
}

function parseHex(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number]
}

function mix(a: string, b: string, t: number): string {
  const ca = parseHex(a)
  const cb = parseHex(b)
  const c = ca.map((v, i) => Math.round(v + (cb[i] - v) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}

/** Ink (near-black or white) that reads on a background color. */
export function inkOn(background: string): string {
  const match = background.match(/\d+/g)
  if (!match) return '#000'
  const [r, g, b] = match.map(Number).map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
  return luminance > 0.4 ? '#111111' : '#ffffff'
}

/**
 * The pointer's position in an element's own CSS pixels, the units its chart draws in. The
 * page's CSS zoom (index.css) scales clientX/Y and getBoundingClientRect but not layout sizes,
 * so the offset from the box is rescaled by layout size over rendered size.
 */
export function pointerIn(e: { clientX: number; clientY: number }, element: Element) {
  const box = element.getBoundingClientRect()
  const [width, height] =
    element instanceof SVGSVGElement
      ? [element.width.baseVal.value, element.height.baseVal.value]
      : element instanceof HTMLElement
        ? [element.offsetWidth, element.offsetHeight]
        : [box.width, box.height]
  return {
    x: (e.clientX - box.left) * (box.width > 0 && width > 0 ? width / box.width : 1),
    y: (e.clientY - box.top) * (box.height > 0 && height > 0 ? height / box.height : 1),
  }
}

/** The rendered width of an element, kept up to date. */
export function useElementWidth<T extends HTMLElement>(fallback = 600) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0].contentRect.width)
      if (next > 0) setWidth(next)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return { ref, width }
}
