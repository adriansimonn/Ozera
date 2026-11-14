/**
 * The Attention Analysis page's assistant bridge: the mode, model and captures, the results on
 * screen (head types, comparisons, mined patterns), set_inputs, and classify_heads,
 * compare_attention and mine_patterns runs' results opening on the page.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { ClassifyHeadsResponse, CompareAttentionResponse, MinePatternResponse } from '../../../types/analysis'
import type { CapturedActivationSummary } from '../../../types/patching'
import { asString, cut, round, type ControlSpec, type Fields } from './common'

export const ANALYSIS_CONTROLS: ControlSpec[] = [
  { id: 'mode' },
  { id: 'model' },
  { id: 'prompt' },
  { id: 'activation' },
  { id: 'capture_button' },
  { id: 'run_button' },
]

const MODES = ['classify', 'compare', 'mine'] as const
type Mode = (typeof MODES)[number]

export interface AnalysisBridgeArgs {
  mode: Mode
  setMode: (mode: Mode) => void
  model: string
  /** Switch models, as the page's model picker does (its selections are cleared) */
  setModel: (model: string) => void
  /** Switch models keeping the selections (a run's result) */
  setModelOnly: (model: string) => void
  capturePrompt: string
  setCapturePrompt: (text: string) => void
  captured: CapturedActivationSummary[]
  refreshCaptured: () => void
  selected: [string, string]
  setSelected: (first: string, second: string) => void
  busy: boolean
  classification: ClassifyHeadsResponse | null
  setClassification: (result: ClassifyHeadsResponse) => void
  comparison: CompareAttentionResponse | null
  setComparison: (result: CompareAttentionResponse) => void
  mining: MinePatternResponse | null
  setMining: (result: MinePatternResponse) => void
}

function classificationSummary(r: ClassifyHeadsResponse) {
  const notable = r.classifications
    .filter((c) => c.primary_type !== 'mixed' && c.primary_type !== 'distributed')
    .sort((x, y) => y.confidence - x.confidence)
    .slice(0, 12)
    .map((c) => ({ layer: c.layer, head: c.head, type: c.primary_type, confidence: round(c.confidence) }))
  const counts = Object.fromEntries(Object.entries(r.summary).filter(([, n]) => n > 0))
  return { activation: r.activation_id, prompt: cut(r.prompt, 200), layers: r.num_layers, heads_per_layer: r.num_heads, counts, most_confident: notable }
}

function comparisonSummary(r: CompareAttentionResponse) {
  return {
    prompts: [cut(r.prompt1, 120), cut(r.prompt2, 120)],
    layer_similarity: r.layer_similarities.map((l) => ({ layer: l.layer, mean: round(l.mean_similarity) })),
    most_different_heads: r.head_differences.slice(0, 8).map((h) => ({ layer: h.layer, head: h.head, similarity: round(h.similarity) })),
    common: r.common_patterns.slice(0, 5),
    divergent: r.divergent_patterns.slice(0, 5),
    method: r.comparison_method,
    warning: r.length_warning,
  }
}

function miningSummary(r: MinePatternResponse) {
  return {
    counts: Object.fromEntries(Object.entries(r.classification_summary).filter(([, n]) => n > 0)),
    top_heads: r.top_heads.slice(0, 8).map(([layer, head, importance]) => ({ layer, head, importance: round(importance) })),
    cross_layer: r.cross_layer_patterns.slice(0, 6).map((p) => ({ type: p.pattern_type, description: cut(p.description, 160), confidence: round(p.confidence) })),
    circuits: r.circuit_candidates.slice(0, 6).map((c) => ({ type: c.type, description: cut(c.description, 160), confidence: round(c.confidence) })),
  }
}

export function analysisBridge(a: AnalysisBridgeArgs): BridgeParts {
  return {
    snapshot: () => {
      const captures = a.captured.filter((c) => c.model_id === a.model)
      return {
        page: 'analysis',
        mode: a.mode,
        model: a.model || null,
        capture_prompt: cut(a.capturePrompt, 300),
        captures: captures.slice(0, 10).map((c) => ({ id: c.id, prompt: cut(c.prompt, 80), tokens: c.num_tokens })),
        more_captures: captures.length > 10 ? captures.length - 10 : undefined,
        selected: a.selected.filter(Boolean),
        busy: a.busy || undefined,
        classification: a.mode === 'classify' && a.classification ? classificationSummary(a.classification) : null,
        comparison: a.mode === 'compare' && a.comparison ? comparisonSummary(a.comparison) : null,
        mining: a.mode === 'mine' && a.mining ? miningSummary(a.mining) : null,
      }
    },
    apply: (fields: Fields) => {
      const mode = asString(fields.mode)
      if (mode && (MODES as readonly string[]).includes(mode)) a.setMode(mode as Mode)
      const model = asString(fields.model)
      if (model && model !== a.model) a.setModel(model)
      const prompt = asString(fields.prompt)
      if (prompt != null) a.setCapturePrompt(prompt)
      const activation = asString(fields.activation)?.trim()
      if (activation) {
        // Compare mode takes a second capture after the first
        const [first] = a.selected
        if ((mode ?? a.mode) === 'compare' && first && first !== activation) a.setSelected(first, activation)
        else a.setSelected(activation, '')
      }
    },
    load: (tool, data) => {
      if (tool === 'classify_heads') {
        const { model, activation_id, result } = data as { model: string; activation_id: string; result: ClassifyHeadsResponse }
        a.setMode('classify')
        a.setModelOnly(model)
        a.refreshCaptured()
        a.setSelected(activation_id, '')
        a.setClassification(result)
      } else if (tool === 'compare_attention') {
        const { model, activation_ids, result } = data as { model: string; activation_ids: [string, string]; result: CompareAttentionResponse }
        a.setMode('compare')
        a.setModelOnly(model)
        a.refreshCaptured()
        a.setSelected(activation_ids[0], activation_ids[1])
        a.setComparison(result)
      } else if (tool === 'mine_patterns') {
        const { model, activation_id, result } = data as { model: string; activation_id: string; result: MinePatternResponse }
        a.setMode('mine')
        a.setModelOnly(model)
        a.refreshCaptured()
        a.setSelected(activation_id, '')
        a.setMining(result)
      }
    },
  }
}
