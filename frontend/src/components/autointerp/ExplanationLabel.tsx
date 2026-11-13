/**
 * A feature's autointerp explanation as an inline label (truncated, with the full text, its
 * score and the model that wrote it in the title). Explanations are plain text, never markup.
 */

import type { CSSProperties } from 'react'
import type { FeatureLabel } from '../../types/autointerp'
import { labelTitle } from './labels'

export function ExplanationLabel({
  label,
  maxWidth = '28ch',
  style,
  className,
}: {
  label: FeatureLabel
  maxWidth?: string | number
  style?: CSSProperties
  className?: string
}) {
  return (
    <span
      className={className}
      title={labelTitle(label)}
      style={{
        display: 'inline-block',
        maxWidth,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        verticalAlign: 'bottom',
        ...style,
      }}
    >
      {label.explanation}
    </span>
  )
}
