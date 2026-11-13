/**
 * Autointerp labels for SAE features, wherever features appear (lab FeatureLinks, Browse, the
 * Analyze heatmap). Lookups for one SAE made close together go out as one request, and each
 * SAE's answers (including "no explanation") are kept, so a page of FeatureLinks costs one call.
 */

import { useMemo } from 'react'
import { useQuery, type QueryClient } from '@tanstack/react-query'
import { apiClient } from '../api/client'
import { refKey } from '../components/lab/saeLinks'
import { useAuthStore } from '../stores/authStore'
import type { FeatureLabel } from '../types/autointerp'
import type { SaeRef } from '../types/probes'

export type FeatureLabels = Map<number, FeatureLabel>

const EMPTY: FeatureLabels = new Map()
// Features per request (the backend's MAX_LABEL_FEATURES), and how long lookups gather
const MAX_PER_REQUEST = 5000
const GATHER_MS = 10

// Per SAE and model: each feature looked up (null: no explanation), and the lookup gathering now
const known = new Map<string, Map<number, FeatureLabel | null>>()
const gathering = new Map<string, { features: Set<number>; done: Promise<void> }>()

function scope(sae: SaeRef, model?: string): string {
  return `${refKey(sae)}|${model ?? ''}`
}

function fetchGathered(key: string, sae: SaeRef, model: string | undefined, features: number[]): Promise<void> {
  let batch = gathering.get(key)
  if (!batch) {
    const ids = new Set<number>()
    const done = new Promise<void>((resolve, reject) => {
      setTimeout(async () => {
        gathering.delete(key)
        const cache = known.get(key) ?? new Map<number, FeatureLabel | null>()
        known.set(key, cache)
        try {
          const all = [...ids]
          for (let i = 0; i < all.length; i += MAX_PER_REQUEST) {
            const chunk = all.slice(i, i + MAX_PER_REQUEST)
            const result = await apiClient.featureLabels({ sae, model, features: chunk })
            chunk.forEach((f) => cache.set(f, null))
            result.labels.forEach((label) => cache.set(label.feature, label))
          }
          resolve()
        } catch (err) {
          reject(err)
        }
      }, GATHER_MS)
    })
    batch = { features: ids, done }
    gathering.set(key, batch)
  }
  features.forEach((f) => batch!.features.add(f))
  return batch.done
}

async function lookup(sae: SaeRef, model: string | undefined, features: number[]): Promise<FeatureLabels> {
  const key = scope(sae, model)
  const missing = features.filter((f) => !known.get(key)?.has(f))
  if (missing.length) await fetchGathered(key, sae, model, missing)
  const cache = known.get(key)
  const out: FeatureLabels = new Map()
  features.forEach((f) => {
    const label = cache?.get(f)
    if (label) out.set(f, label)
  })
  return out
}

/** Labels of features of an SAE (from its atlases on model, if given). */
export function useFeatureLabels(sae: SaeRef | null | undefined, features: number[], model?: string): FeatureLabels {
  const { isAuthenticated } = useAuthStore()
  const idsKey = useMemo(() => [...new Set(features)].sort((a, b) => a - b).join(','), [features])
  const ids = useMemo(() => (idsKey ? idsKey.split(',').map(Number) : []), [idsKey])
  const query = useQuery({
    queryKey: ['feature-labels', sae ? refKey(sae) : '', model ?? '', idsKey],
    queryFn: () => lookup(sae as SaeRef, model, ids),
    enabled: !!sae && isAuthenticated && ids.length > 0,
    staleTime: 10 * 60_000,
  })
  return query.data ?? EMPTY
}

export function useFeatureLabel(sae: SaeRef | null | undefined, feature: number, model?: string): FeatureLabel | null {
  const features = useMemo(() => [feature], [feature])
  return useFeatureLabels(sae, features, model).get(feature) ?? null
}

/** Drop an SAE's labels (one of its features was just explained), here and in react-query. */
export function forgetLabels(queryClient: QueryClient, sae: SaeRef): void {
  const prefix = `${refKey(sae)}|`
  ;[...known.keys()].filter((key) => key.startsWith(prefix)).forEach((key) => known.delete(key))
  queryClient.invalidateQueries({ queryKey: ['feature-labels', refKey(sae)] })
}
