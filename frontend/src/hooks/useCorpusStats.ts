/**
 * Corpus statistics for the SAE browser, from a Geometry Lab atlas of the selected SAE when one
 * has been computed: a feature catalog with real firing frequencies, and each feature's top
 * contexts over the corpus (instead of over the one text analyzed).
 */

import { useCallback, useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { SAESelection } from '../components/sae/SAESelector'
import type { AtlasCorpusId, AtlasMeta, AtlasRef } from '../types/geometry'
import type { FeatureCatalog, FeatureInterpretation } from '../types/model'
import type { SaeRef } from '../types/probes'

// Features listed in the catalog (the most frequent), and tokens shown each side of a context's firing token
const CATALOG_FEATURES = 2000
const CONTEXT_SIDE = 8

function saeRef(selection: SAESelection | null): SaeRef | null {
  if (!selection) return null
  if (selection.externalId) return { kind: 'external', sae_id: selection.externalId }
  return selection.activationType === 'residual' ? { kind: 'ozera', model: selection.model, layer: selection.layer } : null
}

/** The atlas to read: OpenWebText's if there is one, else the newest. */
function pick(atlases: AtlasMeta[]): AtlasMeta | null {
  return atlases.find((a) => a.corpus.id === 'openwebtext') ?? atlases[0] ?? null
}

export function useCorpusStats(selection: SAESelection | null, enabled: boolean) {
  const [atlas, setAtlas] = useState<AtlasMeta | null>(null)
  const [catalog, setCatalog] = useState<FeatureCatalog | null>(null)
  const ref = saeRef(selection)
  const key = ref ? JSON.stringify(ref) : ''

  useEffect(() => {
    setAtlas(null)
    setCatalog(null)
    if (!ref || !enabled) return
    let cancelled = false
    apiClient
      .atlasLookup(ref)
      .then(async (atlases) => {
        const chosen = pick(atlases)
        if (!chosen || cancelled) return
        const page = await apiClient.atlasFeatures({
          model: chosen.model,
          sae: ref,
          corpus: chosen.corpus.id as AtlasCorpusId,
          sort: 'tokens',
          descending: true,
          offset: 0,
          limit: CATALOG_FEATURES,
        })
        if (cancelled) return
        setAtlas(chosen)
        setCatalog({
          sae_id: key,
          num_features: page.width,
          features: page.rows.map((row) => ({
            feature_idx: row.id,
            activation_frequency: row.freq,
            mean_activation: row.mean,
            max_activation: row.max,
            polysemanticity: row.diversity,
            suggested_label: `Feature ${row.id}`,
            confidence: 0,
            unique_tokens: row.tokens,
          })),
          dead_features: page.dead,
          polysemantic_features: [],
          monosemantic_features: [],
        })
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
    // key stands for ref
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled])

  /**
   * A feature's top corpus contexts as the browser shows interpretations: labeled by the token it
   * fires on most among them (confidence: that token's share), polysemanticity the diversity of
   * their firing tokens (as in the catalog).
   */
  const loadFeature = useCallback(
    async (feature: number): Promise<FeatureInterpretation | null> => {
      if (!atlas || !ref) return null
      const atlasRef: AtlasRef = { model: atlas.model, sae: ref, corpus: atlas.corpus.id as AtlasCorpusId }
      const data = await apiClient.atlasFeature(atlasRef, feature)
      const firing = data.examples.map((e) => e.tokens[e.at] ?? '')
      const counts: Record<string, number> = {}
      firing.forEach((t) => (counts[t] = (counts[t] ?? 0) + 1))
      const [top, topCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] ?? ['', 0]
      const distinct = Object.keys(counts).length
      return {
        feature_idx: feature,
        top_activating_tokens: data.examples.map((e) => {
          const side = Math.min(e.at, e.tokens.length - 1 - e.at, CONTEXT_SIDE)
          return {
            token: e.tokens[e.at] ?? '',
            token_id: 0,
            activation_value: e.act,
            position: e.at,
            context: e.tokens.slice(e.at - side, e.at + side + 1),
            prompt: e.tokens.join(''),
          }
        }),
        token_frequency_distribution: Object.fromEntries(Object.entries(counts).map(([t, n]) => [t, n / Math.max(firing.length, 1)])),
        suggested_label: data.tokens > 0 ? `Fires on “${top.trim() || top}”` : 'Never active over the corpus',
        confidence: firing.length ? topCount / firing.length : 0,
        activation_statistics: {
          total_activations: data.tokens,
          mean_activation: data.mean,
          max_activation: data.max,
          unique_tokens: distinct,
          polysemanticity: data.diversity,
        },
      }
    },
    // key stands for ref
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [atlas, key],
  )

  return { atlas, catalog, loadFeature }
}
