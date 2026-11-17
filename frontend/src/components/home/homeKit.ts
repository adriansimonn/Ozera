/**
 * Shared pieces of the homepage's canvases: deterministic noise, and a render loop that sizes a
 * canvas to its box and only runs while it's on screen. The colors are in homeTheme.ts.
 */

export const MONO = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace'

/** A stable pseudo-random number in [0, 1) for a pair of integers. */
export function hash2(x: number, y: number): number {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

/** Smooth value noise in [0, 1). */
export function noise2(x: number, y: number): number {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * (3 - 2 * xf)
  const v = yf * yf * (3 - 2 * yf)
  const a = hash2(xi, yi)
  const b = hash2(xi + 1, yi)
  const c = hash2(xi, yi + 1)
  const d = hash2(xi + 1, yi + 1)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export interface Scene {
  /**
   * The canvas's box changed (also called once before the first frame). Sizes are CSS pixels;
   * dpr is the backing store's scale.
   */
  resize: (width: number, height: number, dpr: number) => void
  frame: (ctx: CanvasRenderingContext2D, time: number, dt: number) => void
  /** The moment to draw, once, for visitors who prefer reduced motion */
  stillAt?: number
  /** CSS font shorthands the scene draws text with; it's laid out again once they've loaded */
  fonts?: string[]
  /** A cap on the backing store's pixels, for scenes that repaint a large canvas every frame */
  maxPixels?: number
  /** A cap on the frame rate, for scenes with nothing to gain from a faster display */
  maxFps?: number
}

export interface CanvasRun {
  stop: () => void
  /** Draws the still frame again, for a scene whose inputs changed under reduced motion */
  redraw: () => void
}

/**
 * Drives a scene on a canvas: keeps the backing store matched to the element's box, and runs
 * frames only while the canvas is on screen and the tab is visible.
 */
export function runCanvas(canvas: HTMLCanvasElement, scene: Scene): CanvasRun {
  const ctx = canvas.getContext('2d')
  if (!ctx) return { stop: () => {}, redraw: () => {} }

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let raf = 0
  let onScreen = false
  let sized = false
  let last = 0
  let time = 0

  // A little under the frame's length, so a display at exactly the cap isn't halved by jitter
  const minGap = scene.maxFps ? 1000 / scene.maxFps - 2 : 0

  const tick = (now: number) => {
    raf = 0
    if (!onScreen || document.hidden || !sized) return
    if (last && now - last < minGap) {
      raf = requestAnimationFrame(tick)
      return
    }
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60
    last = now
    time += dt
    scene.frame(ctx, time, dt)
    raf = requestAnimationFrame(tick)
  }

  const start = () => {
    if (reduced || raf || !onScreen || document.hidden) return
    last = 0
    raf = requestAnimationFrame(tick)
  }

  const fit = () => {
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    if (!width || !height) return
    let dpr = Math.min(window.devicePixelRatio || 1, 2)
    if (scene.maxPixels) dpr = Math.min(dpr, Math.sqrt(scene.maxPixels / (width * height)))
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    scene.resize(width, height, dpr)
    sized = true
    if (reduced) scene.frame(ctx, scene.stillAt ?? 0, 0)
    else start()
  }

  const resizeObserver = new ResizeObserver(fit)
  resizeObserver.observe(canvas)

  const viewObserver = new IntersectionObserver(
    (entries) => {
      onScreen = entries[entries.length - 1].isIntersecting
      start()
    },
    { threshold: 0.05 },
  )
  viewObserver.observe(canvas)

  document.addEventListener('visibilitychange', start)

  // Text measured before the webfonts arrived is laid out again once they're in
  let disposed = false
  if (scene.fonts && document.fonts) {
    Promise.all(scene.fonts.map((font) => document.fonts.load(font)))
      .then(() => {
        if (!disposed) fit()
      })
      .catch(() => {})
  }

  return {
    stop: () => {
      disposed = true
      cancelAnimationFrame(raf)
      resizeObserver.disconnect()
      viewObserver.disconnect()
      document.removeEventListener('visibilitychange', start)
    },
    redraw: () => {
      if (reduced && sized) scene.frame(ctx, scene.stillAt ?? 0, 0)
    },
  }
}
