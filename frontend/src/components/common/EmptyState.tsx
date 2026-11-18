/**
 * What a results area shows before it has results: a pictogram of square cells, as on the
 * homepage (CellGlyph), over one line of text.
 */

import type { ReactNode } from 'react'
import { CellGlyph } from '../home/CellGlyph'

// "#" a lit cell, a digit a lit cell that pulses in that order, "." an empty one
const GLYPHS = {
  // Lines of text, the last one still being written
  generate: ['.........', '###.##.#.', '.........', '##.###.12', '.........'],
  // One prompt's activations dropping into another's
  patching: ['#.#.1.#.#', '....2....', '....3....', '#.#.4.#.#'],
  // A causal attention pattern: every token looks back
  attention: ['#......', '##.....', '###....', '####...', '#####..', '######1'],
  // A narrow input, a wide and mostly dark code, a narrow output
  sae: ['..###..', '.......', '1...2.3', '.......', '..###..'],
  // Two classes and the line a probe draws between them
  probes: ['##...1.', '#...1..', '...1..#', '..1..##'],
  // A concept's values in a ring
  geometry: ['..###..', '.#...#.', '#.....#', '#..1..#', '#.....#', '.#...#.', '..###..'],
  // A loss coming down
  training: ['#......', '#......', '.#.....', '..#....', '...##..', '.....12'],
  // A feature map: communities of cells
  atlas: ['##...#.', '##..###', '.....#.', '.1.....', '111..##', '.1...##'],
  // Bars of a chart
  chart: ['......#', '....#.#', '..#.#.#', '#.#.#.1'],
} as const

export type EmptyGlyph = keyof typeof GLYPHS

interface Props {
  glyph?: EmptyGlyph
  /** Fill the parent and center in it */
  fill?: boolean
  className?: string
  children: ReactNode
}

export function EmptyState({ glyph, fill, className, children }: Props) {
  return (
    <div className={`oz-empty${fill ? ' oz-empty--fill' : ''}${className ? ` ${className}` : ''}`}>
      {glyph && <CellGlyph rows={[...GLYPHS[glyph]]} />}
      <p>{children}</p>
    </div>
  )
}

export default EmptyState
