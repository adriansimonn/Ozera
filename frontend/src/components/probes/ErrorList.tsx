/**
 * The selected probe's worst mistakes: the most confident false positives and false negatives.
 * Surface-feature shortcuts (a word, a length, a template) tend to show up here first.
 */

import { useProbeColors } from './probeTheme'
import { formatScore } from './probeUtils'

interface ErrorListProps {
  scores: number[]
  labels: number[]
  texts: string[]
  labelNames: [string, string]
  limit?: number
}

export function ErrorList({ scores, labels, texts, labelNames, limit = 6 }: ErrorListProps) {
  const colors = useProbeColors()
  const indices = scores.map((_, i) => i)
  const falsePositives = indices.filter((i) => labels[i] === 0 && scores[i] > 0).sort((a, b) => scores[b] - scores[a])
  const falseNegatives = indices.filter((i) => labels[i] === 1 && scores[i] <= 0).sort((a, b) => scores[a] - scores[b])

  const column = (title: string, hint: string, items: number[], predicted: 0 | 1) => (
    <div className="pl-errors-col">
      <div className="pl-errors-head">
        <span>{title}</span>
        <span className="pl-errors-count">{items.length}</span>
      </div>
      <div className="pl-errors-hint">{hint}</div>
      {items.length === 0 ? (
        <div className="pl-errors-empty" style={{ borderColor: colors.border }}>
          None
        </div>
      ) : (
        items.slice(0, limit).map((i) => (
          <div key={i} className="pl-errors-item" style={{ borderColor: colors.border }} title={texts[i]}>
            <span className="pl-errors-score" style={{ borderColor: colors.classes[predicted] }}>
              {formatScore(scores[i])}
            </span>
            <span className="pl-errors-text">{texts[i]}</span>
          </div>
        ))
      )}
    </div>
  )

  return (
    <div className="pl-errors">
      {column('False positives', `${labelNames[0]}, scored as ${labelNames[1]}`, falsePositives, 1)}
      {column('False negatives', `${labelNames[1]}, scored as ${labelNames[0]}`, falseNegatives, 0)}
      <style>{`
        .probe-lab .pl-errors { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .probe-lab .pl-errors-head { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; }
        .probe-lab .pl-errors-count { font-variant-numeric: tabular-nums; }
        .probe-lab .pl-errors-hint { font-size: 0.7rem; margin-bottom: 0.5rem; }
        .probe-lab .pl-errors-empty { font-size: 0.75rem; padding: 0.75rem; border: 1px dashed; text-align: center; }
        .probe-lab .pl-errors-item {
          display: flex;
          gap: 0.5rem;
          align-items: flex-start;
          padding: 0.4rem 0;
          border-top: 1px solid;
          font-size: 0.78rem;
          line-height: 1.35;
        }
        .probe-lab .pl-errors-score {
          flex-shrink: 0;
          min-width: 3.2rem;
          text-align: center;
          padding: 0.05rem 0.3rem;
          border-left: 3px solid;
          font-variant-numeric: tabular-nums;
          font-weight: 600;
          font-size: 0.72rem;
        }
        .probe-lab .pl-errors-text {
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
          word-break: break-word;
        }
        @media (max-width: 700px) {
          .probe-lab .pl-errors { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  )
}
