/**
 * Token text as the Generate page's views show it: whitespace made visible, so " Paris" and
 * "Paris" read apart and a newline token isn't a blank.
 */

// Characters that draw as nothing: controls, no-break and zero-width spaces, direction marks
// eslint-disable-next-line no-control-regex
const INVISIBLE = /[\u0000-\u001f\u007f-\u00a0\u00ad\u1680\u2000-\u200f\u2028-\u202f\u205f-\u2064\u3000\ufeff]/g

export function showToken(token: string | undefined | null): string {
  if (token == null) return '?'
  if (token === '') return '\u25AF'
  return token
    .replace(/ /g, '\u00B7')
    .replace(/\n/g, '\u21B5')
    .replace(/\t/g, '\u2192')
    .replace(INVISIBLE, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`)
}

/** A signed number with its sign always written (+1.20, -0.35 with a true minus). */
export function signed(value: number, digits = 2): string {
  const text = Math.abs(value).toFixed(digits)
  if (Number(text) === 0) return text
  return `${value < 0 ? '\u2212' : '+'}${text}`
}

export function percent(value: number): string {
  if (!Number.isFinite(value)) return '\u2013'
  const pct = value * 100
  if (pct > 0 && pct < 0.05) return '<0.1%'
  return `${pct >= 99.95 || pct === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`
}

/** A lens row's name: the embeddings, then each layer. */
export function lensRowLabel(row: number): string {
  return row === 0 ? 'Emb' : `L${row - 1}`
}
