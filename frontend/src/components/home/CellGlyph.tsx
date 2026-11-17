/**
 * A pictogram on a grid of square cells, in the manner of the hero's mosaic. Each row is a
 * string: "#" for a lit cell, a digit for a lit cell that pulses in that order, "." for an empty
 * one.
 */

// A cell's side as a fraction of the grid's pitch. The glyph is drawn 12 pixels to a cell
// (HomePage.css), so these land on whole pixels.
const LIT = 8 / 12
const EMPTY = 2 / 12

interface Props {
  rows: string[]
  className?: string
}

export function CellGlyph({ rows, className }: Props) {
  return (
    <svg
      className={`ozh-glyph${className ? ` ${className}` : ''}`}
      viewBox={`0 0 ${rows[0].length} ${rows.length}`}
      aria-hidden="true"
    >
      {rows.flatMap((row, y) =>
        [...row].map((cell, x) => {
          const side = cell === '.' ? EMPTY : LIT
          const pulse = cell !== '.' && cell !== '#'
          return (
            <rect
              key={`${x}-${y}`}
              className={cell === '.' ? 'ozh-glyph__empty' : pulse ? 'ozh-glyph__lit ozh-glyph__lit--pulse' : 'ozh-glyph__lit'}
              style={pulse ? { animationDelay: `${Number(cell) * 0.18}s` } : undefined}
              x={x + (1 - side) / 2}
              y={y + (1 - side) / 2}
              width={side}
              height={side}
            />
          )
        }),
      )}
    </svg>
  )
}

export default CellGlyph
