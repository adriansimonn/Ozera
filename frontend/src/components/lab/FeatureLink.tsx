/** A feature's ID, linked into the SAE browser with the SAE, the feature, and an example text filled in. */

import { ExternalLink } from 'lucide-react'
import type { SaeRef } from '../../types/probes'
import { useLabColors } from './labTheme'
import { saeBrowserUrl } from './saeLinks'

export function FeatureLink({ sae, feature, text }: { sae: SaeRef; feature: number; text?: string }) {
  const colors = useLabColors()
  return (
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
  )
}
