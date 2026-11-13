/**
 * The Geometry Lab's assistant bridge (registered by useGeometryLab, so both variants have it):
 * the settings, and the run on screen with each shape hypothesis's held-out R² against its
 * shuffled baseline, set_inputs, and a run_geometry run opening in the lab.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { GeometryLab } from '../../../hooks/useGeometryLab'
import type { GeometryRead, GeometryRun } from '../../../types/geometry'
import type { GeometryViewId } from '../../geometry/GeometryViews'
import { asBool, asNumber, asString, round, type ControlSpec, type Fields } from './common'

export const GEOMETRY_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'concept' },
  { id: 'read' },
  { id: 'chat_template' },
  { id: 'seed' },
  { id: 'layer' },
  { id: 'run_button' },
]

const VIEWS: GeometryViewId[] = ['shape', 'layers', 'similarity', 'compare', 'steer', 'sae', 'atlas']
const READS: GeometryRead[] = ['value_last', 'value_mean', 'template_last']

function runSummary(run: GeometryRun, position: number, dims: number) {
  const tested = run.shapes?.dims ?? []
  const d = Math.max(tested.indexOf(dims), 0)
  const hypotheses = (run.shapes?.hypotheses ?? []).map((h) => {
    let best = 1
    let margin = -Infinity
    for (let p = 1; p <= run.num_layers; p++) {
      const r2 = h.r2[d]?.[p]
      const shuffled = h.shuffled_max[d]?.[p]
      if (r2 != null && shuffled != null && r2 - shuffled > margin) {
        margin = r2 - shuffled
        best = p
      }
    }
    return {
      hypothesis: h.label,
      r2: round(h.r2[d]?.[position] ?? null),
      shuffled_max: round(h.shuffled_max[d]?.[position] ?? null),
      best_layer: best,
      best_r2: round(h.r2[d]?.[best] ?? null),
      best_shuffled_max: round(h.shuffled_max[d]?.[best] ?? null),
    }
  })
  return {
    concept: run.concept.name,
    builtin: run.concept.builtin_id,
    values: run.concept.values.length,
    model: run.model,
    read: run.read,
    layers: run.num_layers,
    shown_layer: position,
    subspace_dims: tested[d] ?? null,
    participation_ratio: round(run.participation_ratio[position] ?? null, 2),
    components_90: run.components_90[position] ?? null,
    between_fraction: round(run.between_fraction[position] ?? null),
    hypotheses,
  }
}

export function geometryBridge(lab: GeometryLab): BridgeParts {
  return {
    snapshot: () => ({
      page: 'geometry',
      view: lab.view,
      model: lab.modelId || null,
      concept: lab.source === 'builtin' ? lab.builtinId : `custom: ${lab.draft.name || 'unnamed'}`,
      read: lab.read,
      chat_template: lab.chatTemplate,
      seed: lab.seed,
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      running: lab.running || undefined,
      run: lab.run ? runSummary(lab.run, lab.position, lab.dims) : null,
      runs_kept: lab.runs.length,
    }),
    apply: (fields: Fields) => {
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModelId(model)
      const concept = asString(fields.concept)
      if (concept) {
        lab.setSource('builtin')
        lab.setBuiltinId(concept)
      }
      const read = asString(fields.read)
      if (read && (READS as string[]).includes(read)) lab.setRead(read as GeometryRead)
      const chat = asBool(fields.chat_template)
      if (chat != null) lab.setChatTemplate(chat)
      const seed = asNumber(fields.seed)
      if (seed != null) lab.setSeed(Math.max(0, Math.floor(seed)))
      const layer = asNumber(fields.layer)
      if (layer != null) lab.setPosition(lab.run ? Math.min(layer, lab.run.num_layers) : layer)
      const view = asString(fields.view)
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as GeometryViewId)
    },
    load: (tool, data) => {
      if (tool !== 'run_geometry') return
      const { run } = data as { run: GeometryRun }
      if (run) lab.loadRun(run)
    },
  }
}
