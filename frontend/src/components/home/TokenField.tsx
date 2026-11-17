/**
 * The closing section's backdrop: a field of monospaced tokens, most of them dim, with the
 * lit ones drawing the Ozera mark (two rings and a chevron) in the theme's colors. The mark
 * develops when the section scrolls into view, tokens re-sample now and then, and the pointer
 * lights the tokens around it.
 */

import { useEffect, useRef } from 'react'
import { MONO, clamp01, hash2, runCanvas, smoothstep } from './homeKit'
import { INK, PAPER, THEME, mix } from './homeTheme'

const VOCABULARY = (
  'LOGIT ATTN HEAD MLP RESID LAYER TOKEN EMBED QUERY KEY VALUE NORM PROBE PATCH FEATURE SAE STEER ' +
  'CIRCUIT NEURON SOFTMAX GRAD BASIS LATENT SPARSE DENSE HOOK CACHE PROMPT ABLATE VECTOR MANIFOLD ' +
  'ATLAS UNEMBED BIAS WEIGHT TOPK STREAM INDUCT COPY BOS EOS L0 L1 MSE AUROC PCA OV QK DIM RANK'
).split(' ')

const BY_LENGTH = new Map<number, string[]>()
for (const word of VOCABULARY) BY_LENGTH.set(word.length, [...(BY_LENGTH.get(word.length) ?? []), word])

const DIM = 0.17
const TORCH_RADIUS = 150

const LEVEL_DIM = 0
const LEVEL_LIT = 1 // on a ring of the mark
const LEVEL_ACCENT = 2 // on its chevron, which a pulse runs down
const LEVEL_ALPHA = [DIM, 0.97, 1]
const CLEARANCE = 44 // pixels of calm the field leaves around the page's text

interface Word {
  text: string
  x: number
  y: number
  width: number
  level: number
  /** The color a word of the mark lights in */
  color: string
  /** Seconds after the section enters view at which the word lights */
  delay: number
  /** Distance along the chevron from its apex, in radii (the chevron's words only) */
  along: number
  /** 0 under the page's text, 1 away from it */
  clear: number
  glow: number
}

function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const t = clamp01(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))
  return Math.hypot(px - ax - dx * t, py - ay - dy * t)
}

interface Props {
  className?: string
  /** Selector for the elements laid over the canvas (in the same positioned box) that the field makes room for */
  avoid?: string
}

export function TokenField({ className, avoid }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let W = 0
    let H = 0
    let fontSize = 13
    let words: Word[] = []
    let enteredAt: number | null = reduced ? -100 : null
    let entered = reduced
    let clock = 0
    let drawnAt = -1
    let swapClock = 0
    const pointer = { x: -9999, y: -9999, seen: -Infinity }

    const layout = (ctx: CanvasRenderingContext2D) => {
      // The mark: beside the headline on wide screens, above it on narrow ones
      const wide = W >= 900
      const cx = wide ? W * 0.735 : W * 0.5
      const cy = wide ? H * 0.47 : H * 0.27
      const radius = wide ? Math.min(H * 0.4, W * 0.245) : Math.min(W * 0.44, H * 0.23)

      // Smaller type under a smaller mark, so its strokes stay a few words wide
      fontSize = Math.max(9, Math.min(13, Math.round(radius / 26)))
      const lineHeight = Math.round(fontSize * 1.85)
      ctx.font = `500 ${fontSize}px ${MONO}`
      const advance = ctx.measureText('M').width
      const half = Math.max(radius * 0.07, lineHeight * 0.66) // half the stroke's width
      const outer = radius * 0.93
      const inner = radius * 0.53
      const foot = outer * Math.SQRT1_2

      // The text laid over the field, in the canvas's own box (offsets ignore reveal transforms)
      const rects = avoid
        ? Array.from((canvas.parentElement ?? document).querySelectorAll<HTMLElement>(avoid), (element) => ({
            // Text near the left edge clears through to it, so no stray half-words are left outside
            x0: element.offsetLeft < CLEARANCE * 2 ? -Infinity : element.offsetLeft,
            y0: element.offsetTop,
            x1: element.offsetLeft + element.offsetWidth,
            y1: element.offsetTop + element.offsetHeight,
          }))
        : []

      words = []

      /** Adds a word at a place in a row; "edge" is where that row's calm around the text ends. */
      const place = (text: string, x: number, y: number, edge: number) => {
        const width = text.length * advance

        // How much of the word sits on the mark's strokes
        let onRing = 0
        let onChevron = 0
        for (let k = 0; k < text.length; k++) {
          const px = x + (k + 0.5) * advance
          const py = y - fontSize * 0.36
          const rho = Math.hypot(px - cx, py - cy)
          const left = distanceToSegment(px, py, cx, cy, cx - foot, cy + foot)
          const right = distanceToSegment(px, py, cx, cy, cx + foot, cy + foot)
          if (Math.min(left, right) < half) onChevron++
          else if (Math.abs(rho - outer) < half || Math.abs(rho - inner) < half) onRing++
        }

        const mx = x + width / 2 - cx
        const my = y - fontSize * 0.36 - cy
        const rho = Math.hypot(mx, my)
        const turn = ((Math.atan2(my, mx) + Math.PI / 2 + Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2)
        let level = LEVEL_DIM
        let delay = 0
        if (onChevron / text.length >= 0.5) {
          level = LEVEL_ACCENT
          delay = 1.5 + (rho / radius) * 0.7
        } else if ((onRing + onChevron) / text.length >= 0.5) {
          level = LEVEL_LIT
          // Each ring is drawn clockwise from the top, the inner one a beat behind
          delay = turn * 1.3 + (rho < (outer + inner) / 2 ? 0.35 : 0)
        }

        // Words before the edge fade with their distance from the text; from it on they're whole
        let clear = 1
        if (x < edge) {
          for (const rect of rects) {
            const dx = Math.max(rect.x0 - (x + width), 0)
            const dy = Math.max(rect.y0 - y, 0, y - fontSize - rect.y1)
            clear *= smoothstep(2, CLEARANCE, Math.hypot(dx, dy))
          }
        }
        // The mark runs between the theme's two colors, from its top to its bottom
        const color = level === LEVEL_DIM ? PAPER : mix(THEME.mark[0], THEME.mark[1], (my + outer) / (outer * 2))
        if (clear > 0.02) words.push({ text, x, y, width, level, color, delay, along: rho / radius, clear, glow: reduced ? 1 : 0 })
      }

      const rows = Math.ceil(H / lineHeight) + 1
      for (let row = 0; row < rows; row++) {
        const y = row * lineHeight + fontSize
        const word = (n: number) => VOCABULARY[Math.floor(hash2(row * 131 + n, 29) * VOCABULARY.length)]

        // A row passing the page's text has a word start exactly where the calm around that text
        // ends, so the zone left clear is the same width beside every piece of that text
        let edge = -Infinity
        for (const rect of rects) {
          if (Math.max(rect.y0 - y, y - fontSize - rect.y1) < CLEARANCE) edge = Math.max(edge, rect.x1 + CLEARANCE)
        }

        // Laid out from that edge in both directions; other rows start a little off the left side
        const start = edge > -Infinity ? edge : -hash2(row, 5) * advance * 7
        for (let n = 0, x = start; x < W; n++) {
          place(word(n), x, y, edge)
          x += (word(n).length + 1) * advance
        }
        for (let n = 1, x = start; x > 0; n++) {
          x -= (word(-n).length + 1) * advance
          place(word(-n), x, y, edge)
        }
      }
    }

    const frame = (ctx: CanvasRenderingContext2D, time: number, dt: number) => {
      clock = time
      const still = dt === 0
      if (entered && enteredAt === null) enteredAt = time
      const since = enteredAt === null ? -1 : time - enteredAt

      // Swap a token for another of its length now and then
      swapClock += dt
      let dirty = still || drawnAt < 0
      if (swapClock > 0.11 && words.length) {
        swapClock = 0
        const word = words[Math.floor(Math.random() * words.length)]
        const pool = BY_LENGTH.get(word.text.length)
        if (pool) word.text = pool[Math.floor(Math.random() * pool.length)]
        dirty = true
      }
      const torch = time - pointer.seen < 1.6
      const developing = since >= 0 && since < 7
      // The mark's chevron keeps pulsing, so redraw steadily, but no faster than the eye needs
      if (!dirty && !torch && !developing && time - drawnAt < 1 / 20) return
      drawnAt = time

      ctx.fillStyle = INK
      ctx.fillRect(0, 0, W, H)
      ctx.font = `500 ${fontSize}px ${MONO}`
      ctx.textBaseline = 'alphabetic'

      const torchFade = torch ? clamp01((1.6 - (time - pointer.seen)) / 0.9) : 0
      // The plain dim words go down in one pass; the rest need a style each
      const special: Word[] = []
      ctx.fillStyle = PAPER
      ctx.globalAlpha = DIM
      for (const word of words) {
        if (word.level !== LEVEL_DIM) {
          if (!still && since >= word.delay) word.glow += (1 - word.glow) * Math.min(1, dt * 6)
          special.push(word)
        } else if (word.clear < 1) {
          special.push(word)
        } else if (torchFade > 0 && Math.hypot(word.x + word.width / 2 - pointer.x, word.y - fontSize * 0.36 - pointer.y) < TORCH_RADIUS) {
          special.push(word)
        } else {
          ctx.fillText(word.text, word.x, word.y)
        }
      }

      for (const word of special) {
        let alpha = DIM + (LEVEL_ALPHA[word.level] - DIM) * word.glow
        if (word.level === LEVEL_DIM && torchFade > 0) {
          const d = Math.hypot(word.x + word.width / 2 - pointer.x, word.y - fontSize * 0.36 - pointer.y)
          alpha = Math.max(alpha, DIM + 0.62 * torchFade * (1 - clamp01(d / TORCH_RADIUS)) ** 1.4)
        }
        alpha *= word.clear
        if (word.level === LEVEL_ACCENT && word.glow > 0.35) {
          // A pulse runs down the chevron from its apex
          const wave = (((time * 0.28 - word.along * 0.55) % 1) + 1) % 1
          ctx.fillStyle = !still && wave < 0.09 ? PAPER : word.color
        } else {
          ctx.fillStyle = word.color
        }
        ctx.globalAlpha = alpha
        ctx.fillText(word.text, word.x, word.y)
      }
      ctx.globalAlpha = 1
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

    const context = canvas.getContext('2d')
    const relayout = () => {
      if (!context || !W) return
      layout(context)
      // Laid out again mid-view (a resize, the font arriving): keep what has already developed
      if (enteredAt !== null) for (const word of words) word.glow = clock - enteredAt >= word.delay ? 1 : 0
      drawnAt = -1
    }
    const run = runCanvas(canvas, {
      resize: (width, height) => {
        W = width
        H = height
        relayout()
      },
      frame,
      fonts: [`500 13px ${MONO}`],
    })
    const textObserver = new ResizeObserver(() => {
      relayout()
      run.redraw()
    })
    if (avoid) (canvas.parentElement ?? document).querySelectorAll(avoid).forEach((element) => textObserver.observe(element))

    // The mark develops once a good part of the section is in view
    const viewObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          entered = true
          viewObserver.disconnect()
        }
      },
      { threshold: 0.35 },
    )
    viewObserver.observe(canvas)

    return () => {
      run.stop()
      textObserver.disconnect()
      viewObserver.disconnect()
      host.removeEventListener('pointermove', onPointerMove)
    }
  }, [avoid])

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />
}

export default TokenField
