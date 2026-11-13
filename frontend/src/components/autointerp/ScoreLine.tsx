/**
 * An explanation's scores on one line (an autointerp explanation's, or a user's hypothesis's):
 * detection, fuzzing, the token baseline, its percentile among the random-direction controls,
 * and recall by activation decile.
 */

import { formatScore } from './labels'

export interface Scored {
  detection: number | null
  detection_hard: number | null
  fuzzing: number | null
  token_detection: number | null
  control_percentile: number | null
  decile_recall: (number | null)[] | null
}

export function DecileRecall({ recall, color, faint }: { recall: (number | null)[]; color: string; faint: string }) {
  const w = 6
  const h = 18
  return (
    <svg
      width={recall.length * (w + 2)}
      height={h}
      role="img"
      aria-label="Recall by activation decile"
      style={{ verticalAlign: 'middle' }}
    >
      {recall.map((r, i) => (
        <g key={i}>
          <title>{`Decile ${i + 1}: ${r == null ? 'no examples' : `${Math.round(r * 100)}% detected`}`}</title>
          <rect x={i * (w + 2)} y={0} width={w} height={h} fill={faint} />
          {r != null && <rect x={i * (w + 2)} y={h * (1 - r)} width={w} height={h * r} fill={color} />}
        </g>
      ))}
    </svg>
  )
}

export function Scores({ e, accent, faint }: { e: Scored; accent: string; faint: string }) {
  const parts: { label: string; value: string; title: string }[] = [
    {
      label: 'Detection',
      value: formatScore(e.detection) + (e.detection_hard != null ? ` (hard ${formatScore(e.detection_hard)})` : ''),
      title: 'Balanced accuracy of telling held-out contexts where it fires from ones where it doesn’t (0.5 is chance); hard: against contexts with its top token where it doesn’t fire',
    },
    { label: 'Fuzzing', value: formatScore(e.fuzzing), title: 'Balanced accuracy of telling its true firing tokens from as many random ones (0.5 is chance)' },
    { label: 'Token baseline', value: formatScore(e.token_detection), title: 'Detection with its top 3 firing tokens as the explanation, on the same contexts' },
  ]
  if (e.control_percentile != null) {
    parts.push({ label: 'Controls', value: `p${Math.round(e.control_percentile)}`, title: 'Its score’s percentile among random-direction controls explained and scored the same way' })
  }
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: '0.75rem', alignItems: 'center' }}>
      {parts.map((p) => (
        <span key={p.label} title={p.title}>
          <span style={{ opacity: 0.7 }}>{p.label}</span> <span style={{ fontFamily: 'monospace' }}>{p.value}</span>
        </span>
      ))}
      {e.decile_recall && (
        <span title="Share of held-out contexts detected, by activation decile (weakest to strongest)" style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ opacity: 0.7 }}>By decile</span>
          <DecileRecall recall={e.decile_recall} color={accent} faint={faint} />
        </span>
      )}
    </div>
  )
}
