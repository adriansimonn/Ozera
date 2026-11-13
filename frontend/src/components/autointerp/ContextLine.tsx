/**
 * One context of a feature's example sets: its tokens, tinted by the feature's activation on each
 * (as a share of its maximum over the corpus), with fuzzing's marks underlined when given. Tokens
 * the atlas didn't read through the SAE (the first positions, massive-norm tokens) are dimmed.
 */

import type { ExampleContext } from '../../types/autointerp'
import type { LabColors } from '../lab/labTheme'

function tint(color: string, alpha: number): string {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255)
  return `${color}${a.toString(16).padStart(2, '0')}`
}

export function ContextLine({ context, colors, tag }: { context: ExampleContext; colors: LabColors; tag?: string }) {
  const highlight = colors.series[1]
  const marked = new Set(context.marks ?? [])
  return (
    <div className="ai-context" style={{ borderTop: `1px solid ${colors.grid}` }}>
      {tag && <span className="ai-context-meta">{tag}</span>}
      <span className="ai-context-meta" title="Activation on its strongest token">
        {context.act > 0 ? context.act.toFixed(2) : '0'}
      </span>
      {context.tokens.map((t, i) => {
        const level = (context.levels[i] ?? 0) / 100
        return (
          <span
            key={i}
            title={level > 0 ? `${Math.round(level * 100)}% of max` : context.read[i] ? undefined : 'Not read through the SAE'}
            style={{
              background: level > 0 ? tint(highlight, 0.15 + 0.6 * level) : undefined,
              boxShadow: marked.has(i) ? `inset 0 -2px 0 ${colors.accent}` : undefined,
              opacity: context.read[i] || level > 0 ? undefined : 0.5,
            }}
          >
            {t.replace(/\n/g, '⏎ ')}
          </span>
        )
      })}
    </div>
  )
}
