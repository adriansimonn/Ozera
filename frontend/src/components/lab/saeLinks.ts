/**
 * SAEs in the labs (Probe Lab, Geometry Lab): a stable key per SAE, short widths, and links into
 * the SAE browser with an SAE, a feature, and an example text filled in (see useSaeDeepLink).
 */

import type { SaeRef } from '../../types/probes'

export function refKey(ref: SaeRef): string {
  return ref.kind === 'ozera' ? `ozera:${ref.model}:${ref.layer}` : `external:${ref.sae_id}`
}

/** A link into the SAE browser, with the SAE, a feature, and an example text to analyze filled in. */
export function saeBrowserUrl(ref: SaeRef, feature: number, text?: string): string {
  const params = new URLSearchParams()
  if (ref.kind === 'ozera') {
    params.set('model', ref.model)
    params.set('layer', String(ref.layer))
    params.set('type', 'residual')
  } else {
    params.set('external', ref.sae_id)
  }
  params.set('feature', String(feature))
  if (text) params.set('text', text)
  return `/sae?${params.toString()}`
}

export function formatWidth(width: number | null): string {
  if (width == null) return '?'
  return width >= 1000 ? `${Math.round(width / 1024)}k` : String(width)
}
