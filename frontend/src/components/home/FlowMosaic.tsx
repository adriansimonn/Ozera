/**
 * The hero's animation: streaks swirling along a flow field resolve into a quantized mosaic of
 * dots and squares, which turns through a few figures and then dissolves back. The first figure
 * reads the streaks' own stream function, so its ribbons run where the streaks were flowing; the
 * others are interference rings, an attention pattern, a patching heat map and the Golden Gate
 * Bridge.
 *
 * Three stacked canvases. The streaks leave fading trails on the lowest (each frame draws only
 * their newest segments); the middle one holds soft patches of background that quiet the trails
 * behind anything laid over the animation, painted once; the top one is redrawn whole with the
 * mosaic.
 */

import { useEffect, useRef, type CSSProperties } from 'react'
import { easeInOut, hash2, noise2, runCanvas, smoothstep } from './homeKit'
import { INK, THEME, ramp } from './homeTheme'

// Seconds: streaks alone on load, then the cycle: the streaks resolve into the first figure, each
// figure is shown and turns into the next, the last dissolves, and the streaks run alone again
const INTRO = 1.4
const RESOLVE = 3.2
const SHOW = 3
const TURN = 0.9
const DISSOLVE = 2.8
const RAW = 2.2
const FIGURES = 5
// A figure is on screen from about halfway through arriving until halfway through leaving. The
// first arrives with the resolve and the last leaves with the dissolve, both slower than a turn,
// so those two are held for that much less, and every figure gets the same time on screen.
const FIRST_SHOW = SHOW - (RESOLVE - TURN) / 2
const LAST_SHOW = SHOW - (DISSOLVE - TURN) / 2
const HOLD = FIRST_SHOW + (FIGURES - 2) * SHOW + LAST_SHOW + (FIGURES - 1) * TURN
const CYCLE = RESOLVE + HOLD + DISSOLVE + RAW

/** How far the mosaic has resolved at a moment of the timeline, 0 (streaks) to 1 (mosaic). */
function resolvedAt(time: number): number {
  let t = time - INTRO
  if (t < 0) return 0
  t %= CYCLE
  if (t < RESOLVE) return easeInOut(t / RESOLVE)
  if (t < RESOLVE + HOLD) return 1
  if (t < RESOLVE + HOLD + DISSOLVE) return 1 - easeInOut((t - RESOLVE - HOLD) / DISSOLVE)
  return 0
}

/** Seconds into the figures' rotation, counted as if the first were held as long as the rest. */
function shownFor(time: number): number {
  const t = time - INTRO
  return t < 0 ? -1 : (t % CYCLE) - RESOLVE + SHOW - FIRST_SHOW
}

const STREAK_COLORS = THEME.streaks

// A ribbon's colors from its fringe to its core, one per step of value; the ribbons take the
// theme's ramps in turn
const STEPS = 7
const RAMPS = THEME.ramps.map((stops) => ramp(stops, STEPS))
const SPRINKLES = THEME.sprinkles
const PALETTE = [...new Set([...RAMPS.flat(), ...SPRINKLES])]
const RAMP_INDEX = RAMPS.map((ramp) => ramp.map((color) => PALETTE.indexOf(color)))
const SPRINKLE_INDEX = SPRINKLES.map((color) => PALETTE.indexOf(color))
const LATTICE = 'rgba(255, 255, 255, 0.09)'

const TRAIL_FADE = 5 // per second: a trail is down to a tenth after about half a second
// Seconds between fades. Fading in fewer, stronger steps leaves less behind: a fade can't take
// a pixel's last few levels of alpha, and the weaker the step, the more of them it leaves.
const FADE_EVERY = 1 / 34
const CLEARANCE = 78 // pixels of calm the mosaic leaves around what's laid over it

const SHAPE_CIRCLE = 0
const SHAPE_SQUARE = 1
const SHAPE_EYE = 2 // a square with a dot punched out
const SHAPE_PLUS = 3

interface Props {
  className?: string
  /** Selector for the elements laid over the canvas that the mosaic should flow around */
  avoid?: string
}

const LAYER: CSSProperties = { position: 'absolute', inset: 0, display: 'block', width: '100%', height: '100%' }

export function FlowMosaic({ className, avoid }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trailsRef = useRef<HTMLCanvasElement>(null)
  const calmRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const trails = trailsRef.current?.getContext('2d')
    const calm = calmRef.current?.getContext('2d')
    if (!canvas || !trails || !calm) return

    let W = 0
    let H = 0
    let unit = 1 // the flow's length scale in pixels

    // The grid
    let cell = 13
    let cols = 0
    let rows = 0
    let originX = 0
    let originY = 0
    let cellRand = new Float32Array(0)
    let cellReveal = new Float32Array(0) // the progress at which a cell resolves
    let cellTurn = new Float32Array(0) // the progress at which a cell turns to the next figure
    let cellClear = new Float32Array(0) // 0 under what's laid over the canvas, 1 away from it
    let cellColor = new Int8Array(0)
    let cellSize = new Float32Array(0)
    let cellShape = new Uint8Array(0)
    let cellCover = new Float32Array(0)

    // The streaks: where each head is, and where it was a frame ago
    let count = 0
    let headX = new Float32Array(0)
    let headY = new Float32Array(0)
    let prevX = new Float32Array(0)
    let prevY = new Float32Array(0)
    let speed = new Float32Array(0)
    let life = new Float32Array(0)
    let wait = new Float32Array(0)
    let tint = new Uint8Array(0)
    let trailsDrawn = false
    let fadeDue = 0

    // Two counter-rotating vortices and a weaker third; positions as fractions of the box
    const vortices = [
      { x: 0.33, y: 0.4, gamma: 1, sway: 0.03, rate: 0.11, at: 0, cx: 0, cy: 0 },
      { x: 0.69, y: 0.6, gamma: -1, sway: 0.028, rate: 0.09, at: 1.7, cx: 0, cy: 0 },
      { x: 1.04, y: 0.04, gamma: 0.5, sway: 0.02, rate: 0.07, at: 3.1, cx: 0, cy: 0 },
    ]
    let focusX = 0
    let focusY = 0

    // The attention pattern's matrix (a mosaic cell per entry) and the heat map's blocks of cells
    let matrix = 0
    let matrixCol = 0
    let matrixRow = 0
    let blockW = 4
    let blockH = 3
    let gridCols = 0
    let gridRows = 0
    let gridCol = 0
    let gridRow = 0

    // The bridge: its towers' columns, how far its cables run on past them, and its rows
    let towerA = 0
    let towerB = 0
    let sideSpan = 0
    let towerRise = 0 // rows from the deck up to a tower's top
    let deckRow = 0
    let waterRow = 0
    let seaRows = 0

    let seed = 1

    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) | 0
      return (seed >>> 0) / 4294967296
    }

    const placeVortices = (time: number) => {
      // The pair sits on a diagonal of the box, steeper in a tall one
      const portrait = W < H * 0.9
      vortices[0].x = portrait ? 0.36 : 0.33
      vortices[0].y = portrait ? 0.33 : 0.4
      vortices[1].x = portrait ? 0.64 : 0.69
      vortices[1].y = portrait ? 0.65 : 0.6
      for (const v of vortices) {
        v.cx = (v.x + v.sway * Math.sin(time * v.rate + v.at)) * W
        v.cy = (v.y + v.sway * Math.cos(time * v.rate * 0.8 + v.at * 1.3)) * H
      }
      focusX = (vortices[0].cx + vortices[1].cx) / 2
      focusY = (vortices[0].cy + vortices[1].cy) / 2
    }

    // A slow standing wave on top of the vortices, so the orbits aren't perfect circles
    const WAVE = 0.16
    const WAVE_X = 2.3
    const WAVE_Y = 2.9

    /** The stream function at a point; the flow runs along its level sets. */
    const stream = (x: number, y: number, time: number) => {
      let psi = 0
      for (const v of vortices) {
        const dx = (x - v.cx) / unit
        const dy = (y - v.cy) / unit
        psi -= 0.5 * v.gamma * Math.log(dx * dx + dy * dy + 0.0025)
      }
      return psi + WAVE * Math.sin((x / unit) * WAVE_X + time * 0.11) * Math.cos((y / unit) * WAVE_Y - time * 0.07)
    }

    /** How much of the mosaic lives at a point: full between the vortices, none at their cores. */
    const envelope = (x: number, y: number, time: number) => {
      const ex = (x - focusX) / (W * 0.52)
      const ey = (y - focusY) / (H * 0.5)
      let e = Math.exp(-(ex * ex + ey * ey) * 1.5)
      e *= 0.55 + 0.9 * noise2(x * 0.0042 + 3, y * 0.0042 + time * 0.03)
      e = smoothstep(0.1, 0.46, e)
      for (let i = 0; i < 2; i++) {
        const dx = (x - vortices[i].cx) / unit
        const dy = (y - vortices[i].cy) / unit
        e *= 1 - Math.exp(-(dx * dx + dy * dy) / 0.016)
      }
      return e
    }

    // What a figure reports for a cell: how strongly it's lit (0 to 1), and which ribbon or part
    // of the figure it belongs to, which picks its ramp
    let value = 0
    let band = 0

    /** The flow: ribbons along the stream function's level sets, where the streaks ran. */
    const flow = (x: number, y: number, _c: number, _r: number, time: number) => {
      const s = stream(x, y, time) * 1.6 + time * 0.035
      band = Math.floor(s)
      const across = Math.abs(s - band - 0.5) * 2
      value = (1 - smoothstep(0.16, 0.9, across)) * envelope(x, y, time)
      value *= 0.74 + 0.34 * noise2(x * 0.021, y * 0.021 + time * 0.13)
    }

    /** Superposition: rings spreading from the two vortices, lit where their crests meet. */
    const rings = (x: number, y: number, _c: number, _r: number, time: number) => {
      const wavelength = unit * 0.1
      const a = Math.hypot(x - vortices[0].cx, y - vortices[0].cy) / wavelength - time * 0.3
      const b = Math.hypot(x - vortices[1].cx, y - vortices[1].cy) / wavelength - time * 0.3
      const crest = (Math.cos(a * Math.PI * 2) + Math.cos(b * Math.PI * 2)) / 2
      band = Math.round(a) + Math.round(b)
      const fromFocus = Math.hypot((x - focusX) / (W * 0.5), (y - focusY) / (H * 0.5))
      value = crest > 0 ? Math.pow(crest, 0.85) * (1 - smoothstep(0.42, 1.12, fromFocus)) : 0
    }

    /**
     * An attention pattern: queries down, keys across, nothing above the diagonal. The diagonal,
     * an induction stripe with its echo, and the first token's column stand out of the rest.
     */
    const attention = (_x: number, _y: number, c: number, r: number, time: number) => {
      value = 0
      band = 3
      const query = r - matrixRow
      const key = c - matrixCol
      if (query < 0 || query >= matrix || key < 0 || key > query) return
      const back = query - key
      const offset = matrix * (0.34 + 0.04 * Math.sin(time * 0.7))
      value = 0.2 + 0.2 * hash2(key + 11, query + 5)
      const diagonal = 1 - back / 5.5
      if (diagonal > value) {
        value = diagonal
        band = 0
      }
      const stripe = 1 - Math.abs(back - offset) / 4.2
      if (stripe > value) {
        value = stripe
        band = 1
      }
      const echo = Math.max(0.78 * (1 - Math.abs(back - offset * 1.9) / 3.2), 0.9 * (1 - key / 3.6))
      if (echo > value) {
        value = echo
        band = 2
      }
      // The row being read
      const reading = ((time * 0.4) % 1.25) * matrix
      value *= 0.84 + 0.24 * Math.exp(-((query - reading) * (query - reading)) / 16)
    }

    /**
     * A patching heat map: tokens across, layers up, in blocks of cells. The path stands out: up
     * the subject's column, across to the last token, and on up, with a pulse climbing it.
     */
    const heatmap = (_x: number, _y: number, c: number, r: number, time: number) => {
      value = 0
      band = 0
      const dc = c - gridCol
      const dr = r - gridRow
      if (dc < 0 || dr < 0) return
      const i = Math.floor(dc / (blockW + 1))
      const j = Math.floor(dr / (blockH + 1))
      // Outside the grid, or in a gap between blocks
      if (i >= gridCols || j >= gridRows || dc % (blockW + 1) === blockW || dr % (blockH + 1) === blockH) return
      const layer = 1 - j / (gridRows - 1)
      const subject = Math.round(gridCols * 0.28)
      const path = subject + (gridCols - 1 - subject) * smoothstep(0.36, 0.74, layer)
      const on = (1 - Math.abs(i - path) / 1.9) * (0.82 + 0.18 * Math.cos((layer - time * 0.3) * Math.PI * 2))
      value = 0.16 + 0.42 * hash2(i + 3, j + 9) * noise2(i * 0.8 + time * 0.25, j * 0.8)
      band = 1 + Math.floor(hash2(i, j + 40) * 2)
      if (on > value) {
        value = on
        band = 0
      }
    }

    /** Rows from the deck up to the cable at a column; negative past the anchorages. */
    const cableOver = (col: number) => {
      const half = (towerB - towerA) / 2
      const out = Math.abs(col - (towerA + towerB) / 2) - half
      if (out <= 0) return 1 + (towerRise - 1) * Math.pow(1 + out / half, 2)
      return out > sideSpan ? -1 : towerRise * Math.pow(1 - out / sideSpan, 1.35)
    }

    /**
     * The Golden Gate Bridge, side on: two towers, the cable slung between them and down to the
     * anchorages, the suspenders, the deck with its traffic, and the strait under it.
     */
    const bridge = (_x: number, _y: number, c: number, r: number, time: number) => {
      value = 0
      band = 0
      const over = deckRow - r
      const fromTower = Math.min(Math.abs(c - towerA), Math.abs(c - towerB))
      // Columns past the anchorage, onto the headland
      const ashore = Math.abs(c - (towerA + towerB) / 2) - (towerB - towerA) / 2 - sideSpan

      if (r > waterRow) {
        const depth = (r - waterRow) / seaRows
        if (depth > 1) return
        // Ripples, fewer toward the bottom, and the towers' broken reflections
        band = 1
        value = smoothstep(0.5, 0.78, noise2(c * 0.17 - time * 0.3 + r * 7.3, r * 0.61)) * (0.66 - 0.44 * depth)
        const glint = (0.8 - 0.7 * depth) * smoothstep(0.3, 0.6, noise2(r * 1.3 - time * 0.9, c * 0.5))
        if (fromTower <= 1 && glint > value) {
          value = glint
          band = 0
        }
        return
      }

      // A tower: two legs, the struts between them, and the pier it stands on
      if (over <= towerRise) {
        const strut =
          over === towerRise ||
          over === Math.round(towerRise * 0.78) ||
          over === Math.round(towerRise * 0.54) ||
          over === Math.round(towerRise * 0.28) ||
          over === -1
        if (fromTower === 1 || (fromTower === 0 && strut) || (fromTower <= 2 && r === waterRow)) {
          value = 0.86
          return
        }
      }

      // The roadway, brighter where the traffic is, on its truss
      const fade = 1 - smoothstep(2, 10, ashore)
      if (over === 0) {
        value = (0.72 + 0.28 * smoothstep(0.42, 0.7, noise2(c * 0.3 - time * 1.1, 5))) * fade
        return
      }
      if (over === -1 && ashore < 0) {
        value = 0.44
        return
      }

      // The cable, a cell or more per column so it holds together where it's steep
      const cable = Math.round(cableOver(c))
      if (over > 0 && cable > 0) {
        const steep = Math.round(Math.max(cableOver(c - 1), cableOver(c + 1)))
        if (over >= cable && over < Math.max(cable + 1, steep)) {
          value = 0.72
          return
        }
        // The suspenders, on every other column; a tower's own columns stay open between struts
        if (over < cable && fromTower > 1 && fromTower % 2 === 0) {
          value = 0.42
          return
        }
      }

      // The headlands: the left one climbs above the road, the right one stays under it
      if (ashore > -4) {
        const inland = ashore + 4
        const left = c < towerA
        const top = Math.min(-(waterRow - deckRow) + inland * 1.7, left ? inland * 0.7 - 2 : -1)
        if (over <= top + 2 * noise2(c * 0.4, 9)) {
          value = (0.3 + 0.3 * noise2(c * 0.33, r * 0.33)) * fade
          band = 1
        }
      }
    }

    // In the order they're shown
    const figures = [flow, rings, attention, heatmap, bridge]

    /**
     * Makes room for what's laid over the canvas: marks the cells under (and near) it, which the
     * mosaic leaves clear, and paints the patches that quiet the streaks behind it.
     */
    const measureClear = () => {
      cellClear.fill(1)
      // Half resolution is plenty for something this soft; sizing the canvas also clears it
      calm.canvas.width = Math.ceil(W / 2)
      calm.canvas.height = Math.ceil(H / 2)
      if (!avoid || !cols) return
      const box = canvas.getBoundingClientRect()
      if (!box.width || !box.height) return
      const scaleX = W / box.width
      const scaleY = H / box.height
      calm.scale(0.5, 0.5)
      calm.fillStyle = INK
      calm.shadowColor = INK
      calm.shadowBlur = 26
      calm.globalAlpha = 0.74
      for (const element of document.querySelectorAll(avoid)) {
        const rect = element.getBoundingClientRect()
        const x0 = (rect.left - box.left) * scaleX
        const x1 = (rect.right - box.left) * scaleX
        const y0 = (rect.top - box.top) * scaleY
        const y1 = (rect.bottom - box.top) * scaleY
        if (x1 < 0 || x0 > W || y1 < 0 || y0 > H) continue
        calm.fillRect(x0 - 16, y0 - 10, x1 - x0 + 32, y1 - y0 + 20)
        for (let r = 0; r < rows; r++) {
          const y = originY + (r + 0.5) * cell
          const dy = Math.max(y0 - y, 0, y - y1)
          if (dy > CLEARANCE) continue
          for (let c = 0; c < cols; c++) {
            const x = originX + (c + 0.5) * cell
            const dx = Math.max(x0 - x, 0, x - x1)
            cellClear[r * cols + c] *= smoothstep(10, CLEARANCE, Math.hypot(dx, dy))
          }
        }
      }
    }

    const spawn = (i: number, delay: number) => {
      for (let attempt = 0; attempt < 8; attempt++) {
        const x = -40 + random() * (W + 80)
        const y = -40 + random() * (H + 80)
        const ex = (x - focusX) / (W * 0.62)
        const ey = (y - focusY) / (H * 0.7)
        // Denser around the vortices, sparse toward the edges
        if (random() > 0.22 + 0.78 * Math.exp(-(ex * ex + ey * ey) * 1.2)) continue
        headX[i] = x
        headY[i] = y
        break
      }
      prevX[i] = headX[i]
      prevY[i] = headY[i]
      wait[i] = delay
      life[i] = 3.5 + random() * 5.5
      speed[i] = 80 + random() * 110
      // Skewed toward the first colors
      tint[i] = Math.floor(Math.pow(random(), 1.5) * STREAK_COLORS.length)
    }

    const resize = (width: number, height: number, dpr: number) => {
      const firstLayout = W === 0
      W = width
      H = height
      unit = Math.min(W, H)
      placeVortices(0)

      // The trails' canvas matches the mosaic's; resizing it clears it, and the streaks redraw
      trails.canvas.width = Math.round(W * dpr)
      trails.canvas.height = Math.round(H * dpr)
      trails.setTransform(dpr, 0, 0, dpr, 0, 0)
      trails.lineCap = 'round'
      trails.lineWidth = 1.15
      trailsDrawn = false

      cell = W < 720 ? 10 : 13
      cols = Math.ceil(W / cell) + 1
      rows = Math.ceil(H / cell) + 1
      originX = (W - cols * cell) / 2
      originY = (H - rows * cell) / 2
      const cells = cols * rows
      cellRand = new Float32Array(cells)
      cellReveal = new Float32Array(cells)
      cellTurn = new Float32Array(cells)
      cellClear = new Float32Array(cells)
      cellColor = new Int8Array(cells)
      cellSize = new Float32Array(cells)
      cellShape = new Uint8Array(cells)
      cellCover = new Float32Array(cells)
      const reach = Math.hypot(Math.max(focusX, W - focusX), Math.max(focusY, H - focusY))
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const x = originX + (c + 0.5) * cell
          const y = originY + (r + 0.5) * cell
          cellRand[i] = hash2(c, r)
          cellReveal[i] = (Math.hypot(x - focusX, y - focusY) / reach) * 0.78 + hash2(c + 91, r + 17) * 0.22
          // Figures turn in a sweep from the left, leaning down
          cellTurn[i] = ((c / cols) * 0.7 + (r / rows) * 0.3) * 0.8 + hash2(c + 7, r + 131) * 0.2
        }
      }
      measureClear()

      // The matrix sits a little right of center: its weight is in its lower left
      matrix = Math.floor(Math.min(cols, rows) * 0.8)
      matrixCol = Math.round((cols - matrix) / 2 + matrix * 0.06)
      matrixRow = Math.round((rows - matrix) / 2)
      blockW = cols < 56 ? 3 : 4
      blockH = cols < 56 ? 2 : 3
      gridCols = Math.floor((cols * 0.84 + 1) / (blockW + 1))
      gridRows = Math.floor((rows * 0.78 + 1) / (blockH + 1))
      gridCol = Math.round((cols - gridCols * (blockW + 1) + 1) / 2)
      gridRow = Math.round((rows - gridRows * (blockH + 1) + 1) / 2)

      // The bridge's main span takes a little over half the width, and its towers stand half as
      // tall as the span is long: far taller than the real ones, which a mosaic would lose
      const span = Math.round(cols * 0.27) * 2
      towerA = Math.round((cols - 1 - span) / 2)
      towerB = towerA + span
      sideSpan = Math.round(span * 0.32)
      towerRise = Math.min(Math.round(span * 0.52), Math.floor(rows * 0.4))
      const piers = Math.max(3, Math.round(towerRise * 0.36))
      seaRows = Math.max(5, Math.round(rows * 0.2))
      deckRow = Math.round((rows - towerRise - piers - seaRows) / 2) + towerRise
      waterRow = deckRow + piers

      const wanted = Math.round(Math.max(320, Math.min(1250, (W * H) / 1050)))
      if (wanted !== count) {
        count = wanted
        headX = new Float32Array(count)
        headY = new Float32Array(count)
        prevX = new Float32Array(count)
        prevY = new Float32Array(count)
        speed = new Float32Array(count)
        life = new Float32Array(count)
        wait = new Float32Array(count)
        tint = new Uint8Array(count)
        // On load the streaks arrive over a second or so
        for (let i = 0; i < count; i++) spawn(i, firstLayout ? random() * random() * 1.2 : random() * 0.6)
      }
    }

    const stepStreaks = (time: number, dt: number) => {
      for (let i = 0; i < count; i++) {
        if (wait[i] > 0) {
          wait[i] -= dt
          continue
        }
        const x = headX[i]
        const y = headY[i]
        let vx = 0
        let vy = 0
        for (const v of vortices) {
          const dx = (x - v.cx) / unit
          const dy = (y - v.cy) / unit
          const k = v.gamma / (dx * dx + dy * dy + 0.0025)
          vx -= dy * k
          vy += dx * k
        }
        const px = (x / unit) * WAVE_X + time * 0.11
        const py = (y / unit) * WAVE_Y - time * 0.07
        vx -= WAVE * WAVE_Y * Math.sin(px) * Math.sin(py)
        vy -= WAVE * WAVE_X * Math.cos(px) * Math.cos(py)
        const norm = Math.hypot(vx, vy) || 1
        prevX[i] = x
        prevY[i] = y
        headX[i] = x + (vx / norm) * speed[i] * dt
        headY[i] = y + (vy / norm) * speed[i] * dt
        life[i] -= dt
        // A streak that ends just stops; its trail fades by itself
        if (life[i] <= 0 || x < -70 || x > W + 70 || y < -70 || y > H + 70) spawn(i, random() * 0.5)
      }
    }

    /** Fades what's on the trails' canvas and adds each streak's newest segment. */
    const drawTrails = (dt: number) => {
      fadeDue += dt
      if (fadeDue >= FADE_EVERY) {
        trails.globalCompositeOperation = 'destination-out'
        trails.globalAlpha = 1 - Math.exp(-fadeDue * TRAIL_FADE)
        trails.fillRect(0, 0, W, H)
        trails.globalCompositeOperation = 'source-over'
        fadeDue = 0
      }
      trails.globalAlpha = 0.92
      for (let color = 0; color < STREAK_COLORS.length; color++) {
        trails.strokeStyle = STREAK_COLORS[color]
        trails.beginPath()
        for (let i = 0; i < count; i++) {
          if (tint[i] !== color || wait[i] > 0) continue
          trails.moveTo(prevX[i], prevY[i])
          trails.lineTo(headX[i], headY[i])
        }
        trails.stroke()
      }
      trailsDrawn = true
    }

    /** A bright bead at each streak's head, except behind what's laid over the canvas. */
    const drawBeads = (ctx: CanvasRenderingContext2D) => {
      ctx.globalAlpha = 0.95
      for (let color = 0; color < STREAK_COLORS.length; color++) {
        ctx.fillStyle = STREAK_COLORS[color]
        ctx.beginPath()
        for (let i = 0; i < count; i++) {
          if (tint[i] !== color || wait[i] > 0) continue
          const c = Math.floor((headX[i] - originX) / cell)
          const r = Math.floor((headY[i] - originY) / cell)
          if (c < 0 || c >= cols || r < 0 || r >= rows || cellClear[r * cols + c] < 0.6) continue
          ctx.rect(headX[i] - 1.1, headY[i] - 1.1, 2.2, 2.2)
        }
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    const drawMosaic = (ctx: CanvasRenderingContext2D, time: number, resolved: number) => {
      if (resolved <= 0) return
      // The figure on show, and how far it has turned into the next (0 while it's shown)
      const shown = shownFor(time)
      const showing = Math.min(FIGURES - 1, Math.max(0, Math.floor(shown / (SHOW + TURN))))
      const into = shown - showing * (SHOW + TURN) - SHOW
      const turning = showing < FIGURES - 1 && into > 0 ? into / TURN : 0
      let anyCover = false

      for (let r = 0; r < rows; r++) {
        const y = originY + (r + 0.5) * cell
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const x = originX + (c + 0.5) * cell
          const rand = cellRand[i]

          const edge = cellReveal[i]
          const cover = smoothstep(edge - 0.07, edge + 0.07, resolved * 1.14 - 0.07)
          cellCover[i] = cover
          cellColor[i] = -1
          if (cover < 0.02) continue
          anyCover = true

          const clear = cellClear[i]
          if (clear < 0.02) {
            cellSize[i] = 0
            continue
          }
          // A cell turning to the next figure shrinks away, then grows back as its new self
          let figure = showing
          let turned = 1
          if (turning > 0) {
            const q = smoothstep(cellTurn[i] - 0.12, cellTurn[i] + 0.12, turning * 1.24 - 0.12)
            if (q >= 0.5) figure++
            turned = Math.abs(1 - 2 * q)
          }
          figures[figure](x, y, c, r, time)
          if (value > 1) value = 1
          const step = Math.round(value * clear * STEPS) // stepped sizes, as a readout would have
          if (step < 1) {
            cellSize[i] = 0
            continue
          }
          value = step / STEPS

          const ramp = ((band % RAMPS.length) + RAMPS.length) % RAMPS.length
          const sprinkle = rand > 0.95 && step > 1 && step < 6
          cellColor[i] = sprinkle ? SPRINKLE_INDEX[Math.floor(rand * 997) % SPRINKLES.length] : RAMP_INDEX[ramp][step - 1]

          // Runs of squares among the circles, by region, with a few strays
          const region = noise2(c * 0.085 + band * 5.7, r * 0.085)
          const pick = hash2(c + 5, r + 3)
          const shape =
            pick > 0.965 ? SHAPE_PLUS : region > 0.56 ? (pick > 0.72 && value > 0.45 ? SHAPE_EYE : SHAPE_SQUARE) : SHAPE_CIRCLE
          cellShape[i] = shape
          const grown = smoothstep(0.15, 1, cover)
          cellSize[i] = cell * (shape === SHAPE_CIRCLE ? 0.26 + 0.7 * value : 0.3 + 0.58 * value) * grown * turned
        }
      }
      if (!anyCover) return

      // Each resolved cell first covers the streaks beneath it
      ctx.fillStyle = INK
      ctx.beginPath()
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cover = cellCover[r * cols + c]
          if (cover < 0.02) continue
          const size = cover > 0.66 ? cell + 0.6 : cell * cover * 1.5
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
          if (cellColor[i] >= 0 || cellCover[i] < 0.5) continue
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
            ctx.arc(x, y, size / 2, 0, Math.PI * 2)
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
        ctx.arc(x, y, hole / 2, 0, Math.PI * 2)
      }
      ctx.fill()
    }

    const frame = (ctx: CanvasRenderingContext2D, time: number, dt: number) => {
      const still = dt === 0
      const resolved = resolvedAt(time)
      placeVortices(time)

      ctx.clearRect(0, 0, W, H)

      if (resolved < 0.999 && !still) {
        stepStreaks(time, dt)
        drawTrails(dt)
        drawBeads(ctx)
      } else if (trailsDrawn) {
        // Fully resolved, the mosaic covers every streak: they rest until it starts to dissolve
        trails.clearRect(0, 0, W, H)
        trailsDrawn = false
      }
      drawMosaic(ctx, time, resolved)
    }

    const textObserver = new ResizeObserver(measureClear)
    if (avoid) document.querySelectorAll(avoid).forEach((element) => textObserver.observe(element))
    const run = runCanvas(canvas, {
      resize,
      frame,
      stillAt: INTRO + RESOLVE + FIRST_SHOW / 2,
      maxPixels: 5.2e6,
      maxFps: 60,
    })

    return () => {
      run.stop()
      textObserver.disconnect()
    }
  }, [avoid])

  return (
    <div className={className} aria-hidden="true">
      <canvas ref={trailsRef} style={LAYER} />
      <canvas ref={calmRef} style={LAYER} />
      <canvas ref={canvasRef} style={LAYER} />
    </div>
  )
}

export default FlowMosaic
