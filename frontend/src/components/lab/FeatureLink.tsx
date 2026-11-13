/**
 * A feature's ID, linked into the SAE browser with the SAE, the feature, and an example text
 * filled in, followed by its autointerp explanation when it has one (looked up unless given).
 */

import { ExternalLink } from 'lucide-react'
import { useFeatureLabel } from '../../hooks/useFeatureLabels'
import type { FeatureLabel } from '../../types/autointerp'
import type { SaeRef } from '../../types/probes'
import { ExplanationLabel } from '../autointerp/ExplanationLabel'
import { useLabColors } from './labTheme'
import { saeBrowserUrl } from './saeLinks'

export function FeatureLink({
  sae,
  feature,
  text,
  model,
  label,
}: {
  sae: SaeRef
  feature: number
  text?: string
  /** The model the SAE reads here (labels come from its atlases) */
  model?: string
  /** The label, when the caller already has it (null: none) */
  label?: FeatureLabel | null
}) {
  const colors = useLabColors()
  const looked = useFeatureLabel(label === undefined ? sae : null, feature, model)
  const shown = label === undefined ? looked : label
  return (
    <span style={{ whiteSpace: 'nowrap' }}>
      <a
        href={saeBrowserUrl(sae, feature, text)}
        target="_blank"
        rel="noopener noreferrer"
        className="pl-feature-link"
        style={{ color: colors.accent }}
        title="Open in the SAE browser"
      >
        #{feature}
        <ExternalLink className="pl-icon" aria-hidden="true" style={{ width: 11, height: 11 }} />
      </a>
      {shown && (
        <>
          <span style={{ color: colors.axis }}> · </span>
          <ExplanationLabel label={shown} maxWidth="24ch" style={{ color: colors.text }} />
        </>
      )}
    </span>
  )
}
