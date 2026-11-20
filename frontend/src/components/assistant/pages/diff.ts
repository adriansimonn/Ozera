/**
 * The Model Diff page's assistant bridge (registered by useDiffLab): the pair and texts, the
 * layers run on screen in brief (its profiles, how far the halves of the texts are apart, the
 * tokens the models disagree on most), the directions and the patch last run; set_inputs; and
 * diff_layers, diff_directions and diff_patch runs opening on the page.
 */

import { decodeLayersRun } from '../../../api/client'
import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { DiffLab, DiffViewId } from '../../../hooks/useDiffLab'
import type { DirectionsRun, LayerMeasure, LayersRun, PatchKind, PatchRun, TextSpec } from '../../../types/diff'
import { LAYER_MEASURES, PATCH_KINDS } from '../../../types/diff'
import { halvesGap, mostDivergent, peak } from '../../diff/diffUtils'
import { asBool, asNumber, asString, cut, round, type ControlSpec, type Fields } from './common'

export const DIFF_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'sample' },
  { id: 'count' },
  { id: 'seed' },
  { id: 'texts' },
  { id: 'chat_template' },
  { id: 'run_button' },
  { id: 'metric' },
  { id: 'layer' },
  { id: 'directions_button' },
  { id: 'component' },
  { id: 'prompt' },
  { id: 'donor' },
  { id: 'sites' },
  { id: 'patch_button' },
]

const VIEWS: DiffViewId[] = ['layers', 'directions', 'patch']
const LISTED_PAIRS = 12

const rounded = (values: (number | null)[]) => values.map((v) => round(v))

function layersSummary(run: LayersRun) {
  const top = peak(run.distance)
  return {
    pair: { base: run.base, other: run.other },
    texts: run.texts_name,
    num_texts: run.num_texts,
    tokens_compared: run.num_tokens,
    chat_template: run.chat_template,
    positions: `0 to ${run.num_layers} (0 the embeddings, i layer i-1's output)`,
    distance: rounded(run.distance),
    cka: rounded(run.cka),
    cka_chance: rounded(run.cka_chance),
    procrustes: rounded(run.procrustes),
    largest_distance: { position: top, distance: round(run.distance[top]) },
    halves_differ_by_at_most: {
      distance: round(halvesGap(run, 'distance')?.gap),
      cka: round(halvesGap(run, 'cka')?.gap),
      procrustes: round(halvesGap(run, 'procrustes')?.gap),
    },
    mean_kl: round(run.kl, 4),
    top_prediction_agrees: round(run.top1_agreement),
    most_divergent_tokens: mostDivergent(run, 5).map(({ text, index, kl }) => {
      const t = run.texts[text]
      return {
        after: cut(t.tokens.slice(Math.max(0, index - 10), index + 1).join(''), 80),
        kl: round(kl),
        base_predicts: t.base_top[index],
        other_predicts: t.other_top[index],
      }
    }),
  }
}

function directionsSummary(run: DirectionsRun) {
  const tested = new Map(run.test.components.map((t) => [t.index, t]))
  return {
    pair: { base: run.base, other: run.other },
    position: run.layer,
    texts: run.texts_name,
    mean_share: round(run.mean_share),
    components: run.components.slice(0, 6).map((c) => {
      const t = tested.get(c.index)
      return {
        component: c.index,
        share: round(c.share),
        highest_tokens: c.high.slice(0, 3).map((x) => x.token),
        closed: t ? round(t.closed) : undefined,
        controls_closed: t ? t.controls.map((x) => round(x.closed)) : undefined,
      }
    }),
    held_out_kl: round(run.test.kl_base, 4),
    full_closed: round(run.test.full_closed),
    joint_closed: round(run.test.joint.closed),
  }
}

function patchSummary(run: PatchRun) {
  const cells = run.grid
    .flatMap((row, layer) => row.map((kl, position) => ({ layer, position, kl })))
    .sort((a, b) => a.kl - b.kl)
    .slice(0, 5)
  return {
    pair: { base: run.base, other: run.other },
    donor: run.donor,
    patched: run.kind,
    prompt: cut(run.prompt, 200),
    kl_unpatched: round(run.kl_base, 4),
    donor_predicts: run.predictions.donor[0]?.token,
    recipient_predicts: run.predictions.recipient[0]?.token,
    lowest_kl_cells: cells.map((c) => ({ layer: c.layer, token: run.tokens[c.position], kl: round(c.kl, 4) })),
    all_positions_by_layer: run.all_positions.map((v) => round(v, 3)),
  }
}

export function diffBridge(lab: DiffLab): BridgeParts {
  return {
    controls: DIFF_CONTROLS.map((c) => c.id),
    tools: ['diff_layers', 'diff_directions', 'diff_patch'],
    snapshot: () => ({
      page: 'diff',
      view: lab.view,
      pair: lab.pair ? { base: lab.pair.base, other: lab.pair.other, kind: lab.pair.kind, layers: lab.pair.num_layers, width: lab.pair.hidden_dim } : null,
      pairs: lab.pairs.slice(0, LISTED_PAIRS).map((p) => `${p.base} vs ${p.other}`),
      texts: lab.source === 'builtin' ? { sample: lab.sampleId, count: lab.count, seed: lab.seed } : { own_texts: lab.ownCount },
      chat_template: lab.chatTemplate,
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      running: lab.running || lab.finding || lab.patching || undefined,
      error: lab.error ?? undefined,
      measure: lab.measure,
      layer: lab.position,
      layers_run: lab.run ? layersSummary(lab.run) : null,
      directions_view:
        lab.view === 'directions' ? { estimate_usd: lab.directionsEstimate ? round(lab.directionsEstimate.estimated_cost, 4) : null } : undefined,
      directions: lab.directions ? directionsSummary(lab.directions) : null,
      component_shown: lab.directions ? lab.component : undefined,
      patch_view:
        lab.view === 'patch'
          ? { prompt: cut(lab.prompt, 300), donor: lab.donor, patched: lab.kind, estimate_usd: lab.patchEstimate ? round(lab.patchEstimate.estimated_cost, 4) : null }
          : undefined,
      patch: lab.patched ? patchSummary(lab.patched) : null,
    }),
    apply: (fields: Fields) => {
      const view = asString(fields.view)
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as DiffViewId)
      // The backend checked the model has a pair (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModel(model.trim())
      const sample = asString(fields.sample)
      if (sample) {
        lab.setSampleId(sample)
        lab.setSource('builtin')
      }
      const count = asNumber(fields.count)
      if (count != null) lab.setCount(Math.max(4, Math.min(200, Math.round(count))))
      const seed = asNumber(fields.seed)
      if (seed != null) lab.setSeed(Math.max(0, Math.round(seed)))
      const texts = asString(fields.texts)
      if (texts != null) {
        lab.setOwn(texts)
        lab.setSource('own')
      }
      const template = asBool(fields.chat_template)
      if (template != null) lab.setChatTemplate(template)
      const metric = asString(fields.metric)
      if (metric && (LAYER_MEASURES as string[]).includes(metric)) {
        lab.setMeasure(metric as LayerMeasure)
        if (!view) lab.setView('layers')
      }
      const layer = asNumber(fields.layer)
      if (layer != null) lab.setPosition(layer)
      const component = asNumber(fields.component)
      if (component != null) {
        lab.setComponent(Math.max(1, Math.round(component)))
        if (!view) lab.setView('directions')
      }
      const prompt = asString(fields.prompt)
      if (prompt != null) lab.setPrompt(prompt)
      const donor = asString(fields.donor)
      if (donor === 'other' || donor === 'base') lab.setDonor(donor)
      const sites = asString(fields.sites)
      if (sites && (PATCH_KINDS as string[]).includes(sites)) lab.setKind(sites as PatchKind)
      if (!view && ['prompt', 'donor', 'sites'].some((name) => name in fields)) lab.setView('patch')
    },
    load: (tool, data) => {
      if (tool === 'diff_layers') {
        const { run, texts } = data as { run: unknown; texts?: TextSpec }
        lab.loadRun(decodeLayersRun(run), texts)
      } else if (tool === 'diff_directions') {
        const { response, texts } = data as { response: DirectionsRun; texts?: TextSpec }
        lab.showDirections(response, texts)
      } else if (tool === 'diff_patch') {
        lab.showPatched((data as { response: PatchRun }).response)
      }
    },
  }
}
