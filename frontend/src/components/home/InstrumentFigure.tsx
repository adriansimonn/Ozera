/**
 * The tools section's figure: a grid of glyph tiles that redraws itself as a small diagram of
 * whichever instrument is active (a forward pass, a patch, a sparse code, …). Choosing
 * an instrument sweeps across the grid tile by tile and plays its diagram from the beginning.
 */

import { useEffect, useRef } from 'react'
import { MONO, clamp01, hash2, noise2, runCanvas, type CanvasRun } from './homeKit'
import { PAPER, THEME, readableOn } from './homeTheme'

const GRID = 24
const ACCENT = 2 // a figure returns this for a tile drawn in the accent color

// The figure is drawn for a white page. Tile tones from the faintest value to the strongest:
// grays down to black, so the accent is the only color
const TONES = ['#f0f0f0', '#e3e3e3', '#d1d1d1', '#b6b6b6', '#949494', '#6d6d6d', '#434343', '#171717']
const ACCENT_GLYPH = readableOn(THEME.figureAccent) === PAPER ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.82)'
const GLYPHS = '0123456789ABCDEF#%&*+<>=/\\|{}[]()~^:$?@!-'

const ease = (t: number) => 1 - Math.pow(1 - clamp01(t), 3)

/** A tile's value in [0, 1] (0: no tile) or ACCENT, at column c and row r (row 0 at the top). */
type Figure = (c: number, r: number, time: number) => number

/**
 * A forward pass: after the prompt, each new token's activations climb the layers, then the
 * next token is emitted.
 */
const forwardPass: Figure = (c, r, time) => {
  const token = c - 2
  if (token < 0 || token > 19 || r < 2 || r > 21) return 0
  const prompt = 7
  const step = 0.5
  const k = time % ((20 - prompt) * step + 2.2)
  const current = prompt + Math.floor(k / step)
  const within = k / step - (current - prompt)
  if (token > current) return 0
  if (r === 21) return token === current ? 0.9 : token < prompt ? 0.5 : 0.7 // the token row
  if (r === 20 || r === 3) return 0
  // The row of predictions: one above every generated token
  if (r === 2) return token < prompt ? 0 : token < current ? 0.95 : within > 0.9 ? ACCENT : 0
  const depth = (19 - r) / 15
  if (token < current) return 0.16 + 0.6 * noise2(token * 0.45 + 9, r * 0.38)
  if (depth > within * 1.12) return 0
  return Math.abs(depth - within) < 0.05 ? ACCENT : 0.82
}

/** Patching: one activation from the clean run (top) is written into the corrupted run (bottom). */
const patching: Figure = (c, r, time) => {
  if (c < 2 || c > 21) return 0
  const period = 7.5
  const run = Math.floor(time / period)
  const k = time - run * period
  const column = 5 + Math.floor(hash2(run, 1) * 9)
  const layer = 2 + Math.floor(hash2(run, 2) * 5) // rows down from the top of a block
  const sourceRow = 2 + layer
  const targetRow = 13 + layer

  if (r >= 2 && r <= 10) {
    if (c === column && r === sourceRow && k < 2.3) return ACCENT
    return 0.46 + 0.42 * noise2(c * 0.42, r * 0.5)
  }
  // The patch travels down the column between the runs
  const travel = (k - 1.1) / 1.2
  const head = sourceRow + (targetRow - sourceRow) * clamp01(travel)
  if (c === column && travel > 0 && travel < 1 && Math.abs(r - head) < 0.6) return ACCENT
  if (r === 11 || r === 12) return 0
  if (r >= 13 && r <= 21) {
    if (k > 2.3 && c === column && r === targetRow) return ACCENT
    // Its effect reaches later positions and deeper layers (upward here)
    const reach = (k - 2.5) * 6.5
    const hops = c - column + (targetRow - r)
    if (k > 2.5 && c >= column && r <= targetRow && hops <= reach) return 0.46 + 0.42 * noise2(c * 0.42, (r - 11) * 0.5)
    return 0.12 + 0.12 * noise2(c * 0.6 + 4, r * 0.6)
  }
  return 0
}

/** A sparse autoencoder: a dense band of activations above, a few features firing below. */
const sparse: Figure = (c, r, time) => {
  if (c < 2 || c > 21) return 0
  if (r >= 2 && r <= 4) return 0.34 + 0.56 * noise2(c * 0.55 + Math.floor(time / 1.1) * 3.1, r * 0.8)
  if (r < 7 || r > 21) return 0
  const fired = hash2(c * 31 + Math.floor((time + hash2(c, r) * 6) / 2.6), r * 17)
  if (fired < 0.007) return ACCENT
  return fired < 0.05 ? 0.95 : 0.06
}

/** A linear probe: a boundary swinging into place between two classes. */
const probe: Figure = (c, r, time) => {
  if (c < 2 || c > 21 || r < 2 || r > 21) return 0
  const x = c - 11.5
  const y = r - 11.5
  const truth = x * Math.cos(0.62) + y * Math.sin(0.62)
  const k = time % 8.5
  const angle = 0.62 + 1.25 * (1 - ease(k / 3.4))
  const side = x * Math.cos(angle) + y * Math.sin(angle)
  if (Math.abs(side) < 0.72) return ACCENT
  if (hash2(c + 40, r + 40) > 0.5 || Math.abs(truth) < 1.3) return 0
  if (truth > 0 !== side > 0) return 0.5 // on the wrong side of the probe, for now
  return truth > 0 ? 0.95 : 0.24
}

/** A concept on a circle: seven clusters around a ring, and a point steered along it. */
const circle: Figure = (c, r, time) => {
  const x = c - 11.5
  const y = r - 11.5
  const radius = 7.7
  const lap = time / 1.25
  const at = Math.floor(lap) + ease((lap % 1) / 0.55)
  const angle = (at / 7) * Math.PI * 2 - Math.PI / 2
  if (Math.hypot(x - radius * Math.cos(angle), y - radius * Math.sin(angle)) < 1.25) return ACCENT
  for (let n = 0; n < 7; n++) {
    const a = (n / 7) * Math.PI * 2 - Math.PI / 2
    if (Math.hypot(x - radius * Math.cos(a), y - radius * Math.sin(a)) < 1.5) return 0.94
  }
  const off = Math.abs(Math.hypot(x, y) - radius)
  if (off < 0.72) return 0.3
  return Math.hypot(x, y) < 4.6 && hash2(c + 3, r + 8) < 0.16 ? 0.1 : 0
}

/** Training: a loss curve filling in, step by step. */
const training: Figure = (c, r, time) => {
  const step = c - 2
  if (step < 0 || step > 19 || r < 2 || r > 21) return 0
  const period = 9.5
  const run = Math.floor(time / period)
  const current = Math.floor(Math.min(1, (time - run * period) / 6.4) * 20)
  if (step > current) return 0
  const loss = 3 + 15.5 * Math.exp(-step / 4.3) + 1.6 * (hash2(step, run + 60) - 0.5)
  const top = 21 - Math.max(1, Math.round(loss)) + 1
  if (r < top) return 0
  if (r === top) return step === current && current < 20 ? ACCENT : 0.95
  return 0.4 - 0.24 * ((r - top) / (21 - top + 1))
}

const FIGURES: Figure[] = [forwardPass, patching, sparse, probe, circle, training]

interface Props {
  /** Index of the figure to draw */
  active: number
  /** Changes whenever the figure should play from its beginning again, as when it's chosen twice */
  restart?: number
  className?: string
}

export function InstrumentFigure({ active, restart = 0, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const runRef = useRef<CanvasRun | null>(null)
  const activeRef = useRef(active)
  activeRef.current = active
  const restartRef = useRef(restart)
  restartRef.current = restart

  // With motion reduced there's no loop to pick the new figure up
  useEffect(() => {
    runRef.current?.redraw()
  }, [active])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let size = 0
    let shown = -1 // the figure the tiles are drawing or sweeping toward
    let restarted = restartRef.current
    let startedAt = 0 // when that figure was chosen: its sweep and its own clock run from here
    const value = new Float32Array(GRID * GRID)
    const accent = new Float32Array(GRID * GRID)
    const glyph = new Uint8Array(GRID * GRID)
    for (let i = 0; i < glyph.length; i++) glyph[i] = Math.floor(hash2(i, 77) * GLYPHS.length)

    const frame = (ctx: CanvasRenderingContext2D, time: number, dt: number) => {
      const still = dt === 0
      if (activeRef.current !== shown || restartRef.current !== restarted) {
        shown = activeRef.current
        restarted = restartRef.current
        startedAt = time
      }
      const figure = FIGURES[shown] ?? FIGURES[0]
      // The still frame is a moment well into the diagram, not its first
      const playing = still ? time : time - startedAt
      const pitch = size / GRID
      const gap = Math.max(1.5, pitch * 0.09)

      // A few tiles change glyph now and then
      if (!still && Math.random() < dt * 14) glyph[Math.floor(Math.random() * glyph.length)] = Math.floor(Math.random() * GLYPHS.length)

      ctx.fillStyle = PAPER
      ctx.fillRect(0, 0, size, size)
      ctx.font = `500 ${Math.max(7, pitch * 0.44).toFixed(1)}px ${MONO}`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      for (let r = 0; r < GRID; r++) {
        for (let c = 0; c < GRID; c++) {
          const i = r * GRID + c
          const target = figure(c, r, playing)
          const targetValue = target === ACCENT ? 1 : target
          const targetAccent = target === ACCENT ? 1 : 0
          if (still) {
            value[i] = targetValue
            accent[i] = targetAccent
          } else if (playing >= ((c + r) / (GRID * 2)) * 0.5 + hash2(c, r) * 0.14) {
            // The sweep has reached this tile: ease it toward the new figure
            const rate = Math.min(1, dt * 13)
            value[i] += (targetValue - value[i]) * rate
            accent[i] += (targetAccent - accent[i]) * Math.min(1, dt * 18)
          }

          const v = value[i]
          const x = (c + 0.5) * pitch
          const y = (r + 0.5) * pitch
          const char = GLYPHS[glyph[i]]
          if (v < 0.035) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.17)'
            ctx.fillText(char, x, y + 0.5)
            continue
          }
          const lit = accent[i] > 0.5
          const tone = Math.min(TONES.length - 1, Math.floor(v * TONES.length))
          const side = (pitch - gap) * Math.min(1, 0.35 + v * 5)
          ctx.fillStyle = lit ? THEME.figureAccent : TONES[tone]
          ctx.fillRect(x - side / 2, y - side / 2, side, side)
          ctx.fillStyle = lit ? ACCENT_GLYPH : tone >= 4 ? 'rgba(255, 255, 255, 0.9)' : tone >= 2 ? 'rgba(0, 0, 0, 0.62)' : 'rgba(0, 0, 0, 0.4)'
          ctx.fillText(char, x, y + 0.5)
        }
      }
    }

    const run = runCanvas(canvas, {
      resize: (width) => {
        size = width
      },
      frame,
      stillAt: 6.6,
      fonts: [`500 12px ${MONO}`],
    })
    runRef.current = run
    return run.stop
  }, [])

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />
}

export default InstrumentFigure
