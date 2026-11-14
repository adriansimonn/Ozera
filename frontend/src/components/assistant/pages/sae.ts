/**
 * The SAE page's assistant bridge: what's on screen (the view, the SAE, the text and the
 * analysis's strongest features), set_inputs, and an analyze_sae run's result opening in the
 * Analyze view.
 */

import type { SAEAnalyzeResponse } from '../../../api/client'
import type { SAESelection } from '../../sae/SAESelector'
import type { BridgeParts } from '../../../hooks/useAssistantPage'
import { asNumber, asString, cut, parseSaeKey, round, type ControlSpec, type Fields } from './common'

export const SAE_CONTROLS: ControlSpec[] = [
  { id: 'mode' },
  { id: 'sae' },
  { id: 'text' },
  { id: 'feature' },
  { id: 'analyze_button' },
]

const MODES = ['analyze', 'browse', 'dashboard', 'compare', 'autointerp', 'load'] as const
type Mode = (typeof MODES)[number]

export interface SaeBridgeArgs {
  mode: Mode
  setMode: (mode: Mode) => void
  selection: SAESelection | null
  setSelection: (selection: SAESelection) => void
  text: string
  setText: (text: string) => void
  analyzing: boolean
  analyzeResponse: SAEAnalyzeResponse | null
  selectedFeature: number | null
  selectFeature: (feature: number) => void
  /** Shows a feature once the text is analyzed (the deep link's pending feature) */
  setPendingFeature: (feature: number) => void
  corpusAtlas: { model: string; corpus: { id: string }; tokens: number } | null
  /** Show an analysis (as the Analyze button's own result) */
  loadAnalysis: (response: SAEAnalyzeResponse, selection: SAESelection) => void
}

function selectionOf(key: string): SAESelection | null {
  const ref = parseSaeKey(key)
  if (!ref) return null
  return ref.kind === 'ozera'
    ? { model: ref.model, layer: ref.layer, activationType: 'residual' }
    : { model: 'nano', layer: 0, activationType: 'residual', externalId: ref.sae_id }
}

function selectionKey(selection: SAESelection | null): string | null {
  if (!selection) return null
  if (selection.externalId) return `external:${selection.externalId}`
  return `ozera:${selection.model}:${selection.layer}`
}

export function saeBridge(a: SaeBridgeArgs): BridgeParts {
  return {
    snapshot: () => {
      const r = a.analyzeResponse
      const strongest = r
        ? [...r.top_features].sort((x, y) => y.activation - x.activation).slice(0, 10).map((f) => ({
            feature: f.feature_id, token: f.token, position: f.token_idx, activation: round(f.activation),
          }))
        : null
      return {
        page: 'sae',
        mode: a.mode,
        sae: selectionKey(a.selection),
        sae_type: a.selection && !a.selection.externalId ? a.selection.activationType : undefined,
        text: cut(a.text),
        analyzing: a.analyzing || undefined,
        analysis: r
          ? {
              tokens: r.num_tokens,
              avg_l0: round(r.metrics.avg_l0, 1),
              active_features: r.metrics.num_active_features,
              width: r.metrics.total_features,
              max_activation: round(r.metrics.max_activation),
              strongest,
            }
          : null,
        selected_feature: a.selectedFeature,
        corpus_atlas: a.corpusAtlas ? { model: a.corpusAtlas.model, corpus: a.corpusAtlas.corpus.id, tokens: a.corpusAtlas.tokens } : null,
      }
    },
    apply: (fields: Fields) => {
      const mode = asString(fields.mode)
      if (mode && (MODES as readonly string[]).includes(mode)) a.setMode(mode as Mode)
      const key = asString(fields.sae)
      const ref = key ? selectionOf(key) : null
      if (ref) a.setSelection(ref)
      const text = asString(fields.text)
      if (text != null) a.setText(text)
      const feature = asNumber(fields.feature)
      if (feature != null) {
        const browsing = (mode ?? a.mode) === 'browse'
        // A new SAE or text clears the analysis, so the feature waits for the next one
        if (browsing || (a.analyzeResponse && !ref && text == null)) a.selectFeature(feature)
        else a.setPendingFeature(feature)
      }
    },
    load: (tool, data) => {
      if (tool !== 'analyze_sae') return
      const { sae, text, response } = data as { sae: string; text: string; response: SAEAnalyzeResponse }
      const selection = selectionOf(sae)
      if (!selection || !response || response.error) return
      a.setSelection(selection)
      a.setText(text)
      a.setMode('analyze')
      a.loadAnalysis(response, selection)
    },
  }
}
