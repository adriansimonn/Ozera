/**
 * The sign-in page's animation: a relief map drawn in contour lines, its ground rolling slowly,
 * on which a mosaic of the Ozera mark resolves (two rings and a chevron, drawn the way the
 * homepage's hero draws its figures: home/FlowMosaic.tsx). As the mark comes out the ground rises
 * into a peak under it, so the lines around it close into rings, and they keep spreading from
 * it. While the mark is held, highlights run round its rings, a pulse runs down the chevron and
 * leaves the outer ring as a ripple, and the pointer raises a hill of its own and lights the
 * cells around it. Then the mark dissolves and the ground is left to itself. While the visitor
 * is being signed in the mark stays, and everything moves faster.
 */

import { useEffect, useRef } from 'react'
import { MONO, clamp01, easeInOut, hash2, noise2, runCanvas, smoothstep } from '../home/homeKit'
import { INK, THEME, ramp } from '../home/homeTheme'

// Seconds: the relief alone on load, then the cycle: the mark resolves on it, it's held, it
// dissolves, and the relief is alone again
const INTRO = 0.8
const RESOLVE = 2.6
const HOLD = 16
const DISSOLVE = 2.2
const RAW = 2.4
const CYCLE = RESOLVE + HOLD + DISSOLVE + RAW

/** How far the mosaic has resolved at a moment of the timeline, 0 (the relief alone) to 1 (the mark). */
function resolvedAt(time: number): number {
  let t = time - INTRO
  if (t < 0) return 0
  t %= CYCLE
  if (t < RESOLVE) return easeInOut(t / RESOLVE)
  if (t < RESOLVE + HOLD) return 1
  if (t < RESOLVE + HOLD + DISSOLVE) return 1 - easeInOut((t - RESOLVE - HOLD) / DISSOLVE)
  return 0
}

// The mark: its outer ring's radius as a fraction of the box's shorter side, the inner ring's as
// a fraction of the outer's, and how far the mosaic reaches around it
const MARK = 0.29
const INNER = 0.62
const REACH = 1.26

const BEAT = 5.4 // seconds from one pulse down the chevron to the next
const PULSE = 1.1 // seconds the pulse takes from the apex to the outer ring
const RIPPLE = 1.7 // seconds its ripple takes from the ring to the mosaic's edge
const TORCH_RADIUS = 150
const TAU = Math.PI * 2

// A stroke's colors from its fringe to its core, one per step of value; the outer ring, the
// inner ring and the chevron take the theme's ramps in turn
const STEPS = 7
const RAMPS = THEME.ramps.map((stops) => ramp(stops, STEPS))
const SPRINKLES = THEME.sprinkles
const PALETTE = [...new Set([...RAMPS.flat(), ...SPRINKLES])]
const RAMP_INDEX = RAMPS.map((colors) => colors.map((color) => PALETTE.indexOf(color)))
const SPRINKLE_INDEX = SPRINKLES.map((color) => PALETTE.indexOf(color))
const LATTICE = 'rgba(255, 255, 255, 0.09)'

// The relief
const LEVELS = 16 // contour lines to a unit of height
const INDEX = 4 // every fourth line is drawn heavier, as a map's index contours are
const GAP = 0.03 // between the rings round the mark, as a fraction of the box's shorter side
const DRIFT = 0.3 // lines a second that leave a peak
const BUCKETS = 30 // more lines than a frame can hold: the ground's range, and the peak's above it
// A line's color by its height: the theme's first ramp, from its fringe to its core
const HEIGHT_COLORS = ramp(THEME.ramps[0], 24)

// The probes: markers that climb the relief, each trailing where it has been and reading its height
const PROBES = 5 // on a small canvas, two, without their readings
const TRAIL = 20 // points a probe trails, one every few pixels

const GRADIENTS = 64
const GRADIENT_X = Float32Array.from({ length: GRADIENTS }, (_, n) => Math.cos((n / GRADIENTS) * Math.PI * 2))
const GRADIENT_Y = Float32Array.from({ length: GRADIENTS }, (_, n) => Math.sin((n / GRADIENTS) * Math.PI * 2))

/** A lattice point's pull on a point (dx, dy) away from it. */
function pull(xi: number, yi: number, dx: number, dy: number): number {
  const g = Math.floor(hash2(xi, yi) * GRADIENTS)
  return GRADIENT_X[g] * dx + GRADIENT_Y[g] * dy
}

/** Smooth gradient noise, in about [-0.7, 0.7]: rolling ground, with no grain of the lattice in it. */
function terrain(x: number, y: number): number {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const xf = x - xi
  const yf = y - yi
  const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
  const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
  const a = pull(xi, yi, xf, yf)
  const b = pull(xi + 1, yi, xf - 1, yf)
  const c = pull(xi, yi + 1, xf, yf - 1)
  const d = pull(xi + 1, yi + 1, xf - 1, yf - 1)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

interface Probe {
  x: number
  y: number
  /** 0 hidden to 1 shown: a probe fades in where it lands and out where it ends */
  shown: number
  ending: boolean
  life: number
  /** Where it has been, oldest first, as [x, y, ...] */
  trail: number[]
}

const PART_FIELD = 0 // around the mark: lit only by the ripple and the pointer
const PART_OUTER = 1
const PART_INNER = 2
const PART_CHEVRON = 3

const SHAPE_CIRCLE = 0
const SHAPE_SQUARE = 1
const SHAPE_EYE = 2 // a square with a dot punched out
const SHAPE_PLUS = 3

function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const t = clamp01(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))
  return Math.hypot(px - ax - dx * t, py - ay - dy * t)
}

interface Props {
  className?: string
  /** The visitor is being signed in */
  busy?: boolean
}

export function ReliefMark({ className, busy = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const busyRef = useRef(busy)
  busyRef.current = busy

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let W = 0
    let H = 0
    let unit = 1 // the box's shorter side
    let centerX = 0
    let centerY = 0
    let radius = 1 // the outer ring's
    let reach = 1 // the mosaic's, from the center

    // The grid. The mark doesn't move, so what a cell is part of is worked out once.
    let cell = 13
    let cols = 0
    let rows = 0
    let originX = 0
    let originY = 0
    let cellRand = new Float32Array(0)
    let cellReveal = new Float32Array(0) // the progress at which a cell resolves
    let cellEdge = new Float32Array(0) // 1 inside the mosaic, falling to 0 across its edge
    let cellPart = new Uint8Array(0)
    let cellAcross = new Float32Array(0) // 1 on a stroke's core, falling to 0 at its fringe
    let cellAt = new Float32Array(0) // where along its stroke: a ring's angle, or the chevron's 0 (apex) to 1
    let cellRho = new Float32Array(0) // distance from the center
    let cellColor = new Int8Array(0)
    let cellSize = new Float32Array(0)
    let cellShape = new Uint8Array(0)
    let cellCover = new Float32Array(0)

    // The relief: its height sampled on a grid of its own, and each line's segments, a list to
    // a level, as [x1, y1, x2, y2, ...]
    let sample = 10
    let fieldCols = 0
    let fieldRows = 0
    let heights = new Float32Array(0)
    const lines: number[][] = Array.from({ length: BUCKETS }, () => [])

    const probes: Probe[] = Array.from({ length: PROBES }, () => ({ x: 0, y: 0, shown: 0, ending: true, life: 0, trail: [] }))
    let seed = 7
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) | 0
      return (seed >>> 0) / 4294967296
    }

    let clock = 0
    let spin = 0 // the clock of everything that moves, which runs faster while busy
    let hurry = busyRef.current ? 1 : 0 // 0 at rest, 1 while the visitor is being signed in
    const pointer = { x: -9999, y: -9999, seen: -Infinity }

    const resize = (width: number, height: number) => {
      W = width
      H = height
      unit = Math.min(W, H)
      centerX = W / 2
      centerY = H / 2
      radius = unit * MARK
      reach = radius * REACH

      sample = unit < 520 ? 8 : W * H > 1.6e6 ? 12 : 10
      fieldCols = Math.ceil(W / sample)
      fieldRows = Math.ceil(H / sample)
      heights = new Float32Array((fieldCols + 1) * (fieldRows + 1))

      cell = unit < 520 ? 10 : 13
      cols = Math.ceil(W / cell) + 1
      rows = Math.ceil(H / cell) + 1
      originX = (W - cols * cell) / 2
      originY = (H - rows * cell) / 2
      const cells = cols * rows
      cellRand = new Float32Array(cells)
      cellReveal = new Float32Array(cells)
      cellEdge = new Float32Array(cells)
      cellPart = new Uint8Array(cells)
      cellAcross = new Float32Array(cells)
      cellAt = new Float32Array(cells)
      cellRho = new Float32Array(cells)
      cellColor = new Int8Array(cells)
      cellSize = new Float32Array(cells)
      cellShape = new Uint8Array(cells)
      cellCover = new Float32Array(cells)

      // The mark's strokes, a few cells wide: the chevron runs from the center to the outer ring
      const half = Math.max(radius * 0.085, cell * 1.15)
      const foot = radius * Math.SQRT1_2
      const across = (distance: number) => 1 - smoothstep(0.25, 1, distance / half)
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const x = originX + (c + 0.5) * cell - centerX
          const y = originY + (r + 0.5) * cell - centerY
          const rho = Math.hypot(x, y)
          const rand = hash2(c, r)
          cellRand[i] = rand
          cellRho[i] = rho
          // The mosaic resolves from the center outward. Toward its edge the squares that cover
          // the relief shrink, so its lines show between them: a halftone, not an edge.
          const edge = 1 - smoothstep(0.86, 1.12, rho / reach + (rand - 0.5) * 0.06)
          cellEdge[i] = edge
          cellReveal[i] = edge > 0.1 ? Math.min(1, rho / reach) * 0.78 + hash2(c + 91, r + 17) * 0.22 : 9

          const onOuter = across(Math.abs(rho - radius))
          const onInner = across(Math.abs(rho - radius * INNER))
          const onChevron = across(Math.min(distanceToSegment(x, y, 0, 0, -foot, foot), distanceToSegment(x, y, 0, 0, foot, foot)))
          if (onChevron > 0 && onChevron >= onOuter && onChevron >= onInner) {
            cellPart[i] = PART_CHEVRON
            cellAcross[i] = onChevron
            cellAt[i] = clamp01((Math.abs(x) + y) / (2 * foot))
          } else if (onOuter > 0 || onInner > 0) {
            cellPart[i] = onOuter >= onInner ? PART_OUTER : PART_INNER
            cellAcross[i] = Math.max(onOuter, onInner)
            cellAt[i] = Math.atan2(y, x)
          }
        }
      }
    }

    /**
     * The ground's height at a point: two octaves of noise, drifting apart; a peak under the mark, as
     * far risen as the mark has resolved; and a hill under the pointer.
     */
    const heightAt = (x: number, y: number, rise: number, torch: number) => {
      const u = x / unit
      const v = y / unit
      let height =
        0.5 +
        0.62 * terrain(u * 1.9 + spin * 0.016, v * 1.9 - spin * 0.011) +
        0.24 * terrain((u * 0.8 - v * 0.6) * 4.3 + 31.7, (u * 0.6 + v * 0.8) * 4.3 + 11.3 + spin * 0.022)
      if (rise > 0) {
        // A cone, whose lines are rings a fixed gap apart; past the mosaic it gives way to the ground
        const rho = Math.hypot(x - centerX, y - centerY)
        const peak = 0.78 + (reach - rho) / (LEVELS * GAP * unit)
        height += (peak - height) * rise * (1 - smoothstep(reach, reach * 2.3, rho))
      }
      if (torch > 0) {
        const from = Math.hypot(x - pointer.x, y - pointer.y) / (unit * 0.09)
        height += 0.2 * torch * Math.exp(-from * from * 0.5)
      }
      return height
    }

    /** Samples the ground and draws its contour lines, found square by square. */
    const drawRelief = (ctx: CanvasRenderingContext2D, time: number, resolved: number, still: boolean) => {
      const torch = clamp01((1.6 - (time - pointer.seen)) / 0.9)
      // Adding to every height moves each line downhill: the lines spread from the peaks
      const drift = spin * DRIFT
      let n = 0
      for (let j = 0; j <= fieldRows; j++) {
        for (let i = 0; i <= fieldCols; i++) heights[n++] = heightAt(i * sample, j * sample, resolved, torch) * LEVELS + drift
      }

      const first = Math.floor(drift)
      for (const list of lines) list.length = 0
      const stride = fieldCols + 1
      for (let j = 0; j < fieldRows; j++) {
        const y = j * sample
        for (let i = 0; i < fieldCols; i++) {
          const at = j * stride + i
          const a = heights[at] // the square's corners, clockwise from its top left
          const b = heights[at + 1]
          const c = heights[at + stride + 1]
          const d = heights[at + stride]
          const low = Math.min(a, b, c, d)
          const high = Math.max(a, b, c, d)
          const x = i * sample
          for (let level = Math.ceil(low); level <= high; level++) {
            const list = lines[level - first]
            if (!list) continue
            // Where the line crosses the square's sides
            const top = a < level !== b < level
            const right = b < level !== c < level
            const bottom = d < level !== c < level
            const left = a < level !== d < level
            const tx = x + (sample * (level - a)) / (b - a)
            const ry = y + (sample * (level - b)) / (c - b)
            const bx = x + (sample * (level - d)) / (c - d)
            const ly = y + (sample * (level - a)) / (d - a)
            if (top && right && bottom && left) {
              // A saddle: which pair of corners the line cuts off depends on the square's middle
              if (a + b + c + d >= level * 4 === a >= level) list.push(tx, y, x + sample, ry, bx, y + sample, x, ly)
              else list.push(tx, y, x, ly, x + sample, ry, bx, y + sample)
            } else if (top && right) list.push(tx, y, x + sample, ry)
            else if (top && bottom) list.push(tx, y, bx, y + sample)
            else if (top && left) list.push(tx, y, x, ly)
            else if (right && bottom) list.push(x + sample, ry, bx, y + sample)
            else if (right && left) list.push(x + sample, ry, x, ly)
            else if (bottom && left) list.push(bx, y + sample, x, ly)
          }
        }
      }

      // On load the lines come up out of the dark
      const shown = still ? 1 : smoothstep(0, 1.4, time)
      ctx.lineCap = 'round'
      for (let k = 0; k < BUCKETS; k++) {
        const list = lines[k]
        if (!list.length) continue
        const level = first + k
        const height = clamp01((level - drift) / LEVELS)
        const index = ((level % INDEX) + INDEX) % INDEX === 0
        ctx.strokeStyle = HEIGHT_COLORS[Math.round((0.27 + 0.56 * height) * (HEIGHT_COLORS.length - 1))]
        ctx.globalAlpha = shown * (index ? 1 : 0.7)
        ctx.lineWidth = index ? 1.5 : 1
        ctx.beginPath()
        for (let s = 0; s < list.length; s += 4) {
          ctx.moveTo(list[s], list[s + 1])
          ctx.lineTo(list[s + 2], list[s + 3])
        }
        ctx.stroke()
      }
      ctx.globalAlpha = 1
    }

    /** Moves the probes uphill, and starts each again somewhere else once it has nowhere to climb. */
    const stepProbes = (time: number, dt: number, resolved: number) => {
      const torch = clamp01((1.6 - (time - pointer.seen)) / 0.9)
      const keepOut = resolved > 0.3 ? reach * 1.2 : 0 // the mosaic, and the room around it
      const active = unit < 520 ? 2 : PROBES
      for (let n = 0; n < active; n++) {
        const probe = probes[n]
        if (probe.ending) {
          probe.shown -= dt * 1.6
          if (probe.shown > 0) continue
          // Lands anywhere clear of the mark and of the other probes
          for (let attempt = 0; attempt < 8; attempt++) {
            probe.x = (0.06 + 0.88 * random()) * W
            probe.y = (0.08 + 0.84 * random()) * H
            const x = probe.x
            const y = probe.y
            const crowded = probes.some((other) => other !== probe && other.shown > 0 && Math.hypot(other.x - x, other.y - y) < unit * 0.16)
            if (!crowded && Math.hypot(x - centerX, y - centerY) > keepOut * 1.25) break
          }
          probe.shown = 0
          probe.ending = false
          probe.life = 9 + random() * 10
          probe.trail.length = 0
          continue
        }
        probe.shown = Math.min(1, probe.shown + dt * 1.6)
        probe.life -= dt

        // Up the slope, at a walk
        const reachOut = 6
        const slopeX = heightAt(probe.x + reachOut, probe.y, resolved, torch) - heightAt(probe.x - reachOut, probe.y, resolved, torch)
        const slopeY = heightAt(probe.x, probe.y + reachOut, resolved, torch) - heightAt(probe.x, probe.y - reachOut, resolved, torch)
        const slope = Math.hypot(slopeX, slopeY)
        const pace = unit * 0.032 * dt
        probe.x += (slopeX / (slope || 1)) * pace
        probe.y += (slopeY / (slope || 1)) * pace

        const trail = probe.trail
        const last = trail.length - 2
        if (last < 0 || Math.hypot(probe.x - trail[last], probe.y - trail[last + 1]) > 7) {
          trail.push(probe.x, probe.y)
          if (trail.length > TRAIL * 2) trail.splice(0, 2)
        }

        // It ends on a summit, at the mark's edge, off the map, or when its time is up
        const out = probe.x < 14 || probe.x > W - 14 || probe.y < 14 || probe.y > H - 14
        if (slope < 0.0016 || out || probe.life <= 0 || Math.hypot(probe.x - centerX, probe.y - centerY) < keepOut) probe.ending = true
      }
    }

    const drawProbes = (ctx: CanvasRenderingContext2D, time: number, resolved: number) => {
      const torch = clamp01((1.6 - (time - pointer.seen)) / 0.9)
      ctx.font = `500 10px ${MONO}`
      ctx.textBaseline = 'middle'
      ctx.fillStyle = '#ffffff'
      for (const probe of probes) {
        if (probe.shown <= 0) continue
        const trail = probe.trail
        for (let n = 0; n < trail.length; n += 2) {
          ctx.globalAlpha = probe.shown * 0.55 * ((n + 2) / trail.length)
          ctx.fillRect(trail[n] - 0.9, trail[n + 1] - 0.9, 1.8, 1.8)
        }
        ctx.globalAlpha = probe.shown
        ctx.fillRect(Math.round(probe.x) - 2.5, Math.round(probe.y) - 2.5, 5, 5)
        if (unit < 520) continue
        ctx.globalAlpha = probe.shown * 0.72
        // Its reading beside it, on whichever side has the room
        const flip = probe.x > W - 56
        ctx.textAlign = flip ? 'right' : 'left'
        ctx.fillText(heightAt(probe.x, probe.y, resolved, torch).toFixed(3), Math.round(probe.x) + (flip ? -9 : 9), Math.round(probe.y) + 0.5)
      }
      ctx.globalAlpha = 1
    }

    const drawMosaic = (ctx: CanvasRenderingContext2D, time: number, resolved: number) => {
      if (resolved <= 0) return

      // The highlights: two round the outer ring, one the other way round the inner
      const outerHead = spin * 0.45
      const innerHead = -spin * 0.7
      // The pulse down the chevron, and the ripple it leaves the outer ring as
      const beat = spin % BEAT
      const pulseAt = beat / PULSE
      const rippleAt = radius + ((beat - PULSE) / RIPPLE) * (reach - radius)
      const rippleFade = beat > PULSE ? Math.pow(clamp01(1 - (rippleAt - radius) / (reach - radius)), 0.7) : 0
      const torch = clamp01((1.6 - (time - pointer.seen)) / 0.9)
      let anyCover = false

      for (let r = 0; r < rows; r++) {
        const y = originY + (r + 0.5) * cell
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const edge = cellReveal[i]
          const cover = smoothstep(edge - 0.07, edge + 0.07, resolved * 1.14 - 0.07)
          cellCover[i] = cover
          cellColor[i] = -1
          if (cover < 0.02) continue
          anyCover = true

          const x = originX + (c + 0.5) * cell
          const part = cellPart[i]
          const across = cellAcross[i]
          let value = 0
          let band = 1
          if (part === PART_CHEVRON) {
            const off = (cellAt[i] - pulseAt) / 0.16
            value = across * (0.6 + 0.4 * Math.exp(-off * off))
            band = 2
          } else if (part !== PART_FIELD) {
            // How far round its ring the cell is behind a highlight
            const outer = part === PART_OUTER
            const gap = outer ? TAU / 2 : TAU
            let behind = outer ? outerHead - cellAt[i] : cellAt[i] - innerHead
            behind = ((behind % gap) + gap) % gap
            value = across * (0.56 + 0.44 * Math.exp(-behind / 0.7)) * (0.9 + 0.2 * noise2(x * 0.021, y * 0.021 + spin * 0.13))
            band = outer ? 0 : 1
          }
          if (rippleFade > 0 && cellRho[i] > radius) {
            const off = (cellRho[i] - rippleAt) / (cell * 1.25)
            const lift = rippleFade * Math.exp(-off * off)
            value = part === PART_FIELD ? 0.46 * lift : value + 0.3 * lift * across
          }
          if (torch > 0) {
            const from = Math.hypot(x - pointer.x, y - pointer.y)
            if (from < TORCH_RADIUS) {
              const lift = torch * Math.pow(1 - from / TORCH_RADIUS, 1.4)
              if (part !== PART_FIELD) value += 0.35 * lift * across
              else if (0.5 * lift > value) {
                value = 0.5 * lift
                band = 0
              }
            }
          }
          const step = Math.min(STEPS, Math.round(value * STEPS)) // stepped sizes, as a readout would have
          if (step < 1) {
            cellSize[i] = 0
            continue
          }
          value = step / STEPS

          const rand = cellRand[i]
          const sprinkle = part !== PART_FIELD && rand > 0.95 && step > 1 && step < 6
          cellColor[i] = sprinkle ? SPRINKLE_INDEX[Math.floor(rand * 997) % SPRINKLES.length] : RAMP_INDEX[band % RAMPS.length][step - 1]

          // The rings in circles, the chevron in squares, and a few strays
          const pick = hash2(c + 5, r + 3)
          const shape =
            pick > 0.965 ? SHAPE_PLUS : part === PART_CHEVRON ? (pick > 0.7 && value > 0.45 ? SHAPE_EYE : SHAPE_SQUARE) : SHAPE_CIRCLE
          cellShape[i] = shape
          cellSize[i] = cell * (shape === SHAPE_CIRCLE ? 0.26 + 0.7 * value : 0.3 + 0.58 * value) * smoothstep(0.15, 1, cover)
        }
      }
      if (!anyCover) return

      // Each resolved cell first covers the relief beneath it
      ctx.fillStyle = INK
      ctx.beginPath()
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const cover = cellCover[i]
          if (cover < 0.02) continue
          const size = cover > 0.66 && cellEdge[i] >= 1 ? cell + 0.6 : cell * Math.min(1, cover * 1.5) * cellEdge[i]
          ctx.rect(originX + (c + 0.5) * cell - size / 2, originY + (r + 0.5) * cell - size / 2, size, size)
        }
      }
      ctx.fill()

      // The empty lattice
      ctx.fillStyle = LATTICE
      ctx.beginPath()
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          if (cellColor[i] >= 0 || cellCover[i] < 0.5 || cellEdge[i] < 0.5) continue
          ctx.rect(originX + (c + 0.5) * cell - 0.6, originY + (r + 0.5) * cell - 0.6, 1.2, 1.2)
        }
      }
      ctx.fill()

      for (let color = 0; color < PALETTE.length; color++) {
        ctx.fillStyle = PALETTE[color]
        ctx.beginPath()
        let any = false
        for (let i = 0; i < cellColor.length; i++) {
          if (cellColor[i] !== color) continue
          const size = cellSize[i]
          if (size < 0.8) continue
          any = true
          const x = originX + ((i % cols) + 0.5) * cell
          const y = originY + (Math.floor(i / cols) + 0.5) * cell
          const shape = cellShape[i]
          if (shape === SHAPE_CIRCLE) {
            ctx.moveTo(x + size / 2, y)
            ctx.arc(x, y, size / 2, 0, TAU)
          } else if (shape === SHAPE_PLUS) {
            const arm = size * 0.3
            ctx.rect(x - size / 2, y - arm / 2, size, arm)
            ctx.rect(x - arm / 2, y - size / 2, arm, size)
          } else {
            ctx.rect(x - size / 2, y - size / 2, size, size)
          }
        }
        if (any) ctx.fill()
      }

      // The dots punched out of the "eye" squares
      ctx.fillStyle = INK
      ctx.beginPath()
      for (let i = 0; i < cellColor.length; i++) {
        if (cellColor[i] < 0 || cellShape[i] !== SHAPE_EYE || cellSize[i] < 5) continue
        const x = originX + ((i % cols) + 0.5) * cell
        const y = originY + (Math.floor(i / cols) + 0.5) * cell
        const hole = cellSize[i] * 0.3
        ctx.moveTo(x + hole / 2, y)
        ctx.arc(x, y, hole / 2, 0, TAU)
      }
      ctx.fill()
    }

    const frame = (ctx: CanvasRenderingContext2D, time: number, dt: number) => {
      const still = dt === 0
      clock = time
      hurry += ((busyRef.current ? 1 : 0) - hurry) * Math.min(1, dt * 3)
      spin = still ? time : spin + dt * (1 + 1.6 * hurry)

      // Signing in brings the mark out, wherever the cycle is
      const resolved = Math.max(resolvedAt(time), easeInOut(clamp01(hurry * 1.15)))

      ctx.clearRect(0, 0, W, H)
      drawRelief(ctx, time, resolved, still)
      if (!still) {
        stepProbes(time, dt, resolved)
        drawProbes(ctx, time, resolved)
      }
      drawMosaic(ctx, time, resolved)
    }

    const onPointerMove = (e: PointerEvent) => {
      const box = canvas.getBoundingClientRect()
      if (!box.width || !box.height) return
      pointer.x = ((e.clientX - box.left) * canvas.clientWidth) / box.width
      pointer.y = ((e.clientY - box.top) * canvas.clientHeight) / box.height
      pointer.seen = clock
    }
    const host = canvas.parentElement ?? canvas
    host.addEventListener('pointermove', onPointerMove)

    const run = runCanvas(canvas, {
      resize,
      frame,
      stillAt: INTRO + RESOLVE + 3,
      maxPixels: 5.2e6,
      maxFps: 60,
    })

    return () => {
      run.stop()
      host.removeEventListener('pointermove', onPointerMove)
    }
  }, [])

  return (
    <div className={className} aria-hidden="true">
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, display: 'block', width: '100%', height: '100%' }} />
    </div>
  )
}

export default ReliefMark
