/**
 * Links into the SAE browser from elsewhere (the Probe Lab's SAE features): which SAE to select,
 * a feature to show once the text is analyzed, and the text, read from the query string once.
 *
 *   /sae?model=nano&layer=3&type=residual&feature=12&text=...
 *   /sae?external=<sae_id>&feature=12&text=...
 *
 * Nothing runs by itself: analyzing is charged, so the page waits for the user to press Analyze.
 */

import { useCallback, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { SAESelection } from '../components/sae/SAESelector'

export interface SaeDeepLink {
  selection: SAESelection | null
  feature: number | null
  text: string
}

function parseLink(params: URLSearchParams): SaeDeepLink | null {
  const external = params.get('external')
  const model = params.get('model')
  const layer = Number(params.get('layer'))
  const type = params.get('type')
  let selection: SAESelection | null = null
  if (external) {
    selection = { model: 'nano', layer: 0, activationType: 'residual', externalId: external }
  } else if ((model === 'nano' || model === 'mini') && Number.isInteger(layer) && layer >= 0) {
    selection = { model, layer, activationType: type === 'mlp_output' ? 'mlp_output' : 'residual' }
  }
  const featureParam = params.get('feature')
  const feature = featureParam != null && /^\d+$/.test(featureParam) ? Number(featureParam) : null
  const text = params.get('text') ?? ''
  return selection || feature != null || text ? { selection, feature, text } : null
}

export function useSaeDeepLink() {
  const [params] = useSearchParams()
  const [link] = useState(() => parseLink(params))
  const [pendingFeature, setPendingFeature] = useState<number | null>(link?.feature ?? null)
  const clearPending = useCallback(() => setPendingFeature(null), [])
  return { link, pendingFeature, clearPending }
}
