/**
 * The Geometry Lab's assistant bridge, in parts. The lab's part (registered by useGeometryLab, so
 * both variants have it): the settings, the concept (built-in, or the user's own from its
 * fields), the kept runs with their run_ids and the run on screen with each shape hypothesis's
 * held-out R² against its shuffled baseline, set_inputs, and a run_geometry run opening in the
 * lab. It also shows the run and view a field or another run's result is for. The compare,
 * steer, sae and atlas views each register their own part (inputs, result on screen, result).
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { GeometryLab } from '../../../hooks/useGeometryLab'
import type {
  AtlasCorpusId,
  AtlasTokens,
  CompareResult,
  ConceptStructure,
  GeometryRead,
  GeometryRun,
  GeometrySaeResult,
  SteerResult,
} from '../../../types/geometry'
import type { GeometryViewId } from '../../geometry/GeometryViews'
import { asBool, asNumber, asString, cut, round, type ControlSpec, type Fields } from './common'

export const GEOMETRY_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'run' },
  { id: 'concept' },
  { id: 'concept_name' },
  { id: 'structure' },
  { id: 'values' },
  { id: 'templates' },
  { id: 'eval_templates' },
  { id: 'read' },
  { id: 'chat_template' },
  { id: 'seed' },
  { id: 'layer' },
  { id: 'run_button' },
]

// Each view's own controls (its part fills them)
export const COMPARE_CONTROLS: ControlSpec[] = [{ id: 'compare_with' }, { id: 'compare_dims' }]
export const STEER_CONTROLS: ControlSpec[] = [
  { id: 'steer_source' },
  { id: 'steer_target' },
  { id: 'steer_steps' },
  { id: 'steer_dims' },
  { id: 'steer_template' },
  { id: 'steer_button' },
]
export const SAE_CONTROLS: ControlSpec[] = [{ id: 'sae' }, { id: 'sae_button' }]
export const ATLAS_CONTROLS: ControlSpec[] = [{ id: 'atlas_sae' }, { id: 'atlas_corpus' }, { id: 'atlas_tokens' }, { id: 'atlas_button' }]

const VIEWS: GeometryViewId[] = ['shape', 'layers', 'similarity', 'compare', 'steer', 'sae', 'atlas']
const READS: GeometryRead[] = ['value_last', 'value_mean', 'template_last']
const STRUCTURES: ConceptStructure[] = ['cyclic', 'ordinal', 'categorical']
const CORPORA: AtlasCorpusId[] = ['openwebtext', 'wikitext', 'tinystories']
const TOKENS: AtlasTokens[] = [1_000_000, 2_000_000, 5_000_000]

const VIEW_CONTROLS: [GeometryViewId, ControlSpec[]][] = [
  ['compare', COMPARE_CONTROLS],
  ['steer', STEER_CONTROLS],
  ['sae', SAE_CONTROLS],
  ['atlas', ATLAS_CONTROLS],
]
const VIEW_TOOLS: Record<string, GeometryViewId> = {
  compare_geometry: 'compare',
  steer_geometry: 'steer',
  geometry_sae: 'sae',
  build_atlas: 'atlas',
}

const ids = (specs: ControlSpec[]) => specs.map((c) => c.id)

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
    run_id: run.run_id ?? null,
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

/** A newline-separated list from set_inputs, as the concept panel's text (one item per line). */
function lines(value: string): string {
  return value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .join('\n')
}

export function geometryBridge(lab: GeometryLab): BridgeParts {
  const localRun = (runId: string | null | undefined) => (runId ? lab.runs.find((r) => r.run.run_id === runId) ?? null : null)
  return {
    controls: ids(GEOMETRY_CONTROLS),
    tools: ['run_geometry'],
    snapshot: () => ({
      page: 'geometry',
      view: lab.view,
      model: lab.modelId || null,
      concept: lab.source === 'builtin' ? lab.builtinId : `custom: ${lab.draft.name || 'unnamed'}`,
      custom:
        lab.source === 'custom'
          ? {
              name: lab.draft.name || null,
              structure: lab.draft.structure,
              values: cut(lab.custom.values.join(', '), 400),
              templates: lab.custom.templates.length,
              eval_templates: lab.custom.eval_templates?.length ?? 0,
            }
          : undefined,
      read: lab.read,
      chat_template: lab.chatTemplate,
      seed: lab.seed,
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      running: lab.running || undefined,
      run: lab.run ? runSummary(lab.run, lab.position, lab.dims) : null,
      runs: lab.runs.map((r) => ({ run_id: r.run.run_id ?? null, label: r.label, shown: r.id === lab.runId || undefined })),
    }),
    apply: (fields: Fields) => {
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModelId(model)
      const shown = localRun(asString(fields.run)?.trim())
      if (shown) lab.selectRun(shown.id)
      const concept = asString(fields.concept)
      if (concept) {
        lab.setSource('builtin')
        lab.setBuiltinId(concept)
      }
      // The user's own concept: any of its fields switches to it
      const name = asString(fields.concept_name)
      const structure = asString(fields.structure)
      const values = asString(fields.values)
      const templates = asString(fields.templates)
      const evalTemplates = asString(fields.eval_templates)
      if (name != null || structure || values != null || templates != null || evalTemplates != null) {
        lab.setSource('custom')
        lab.setDraft((draft) => ({
          name: name != null ? name.trim() : draft.name,
          structure: structure && (STRUCTURES as string[]).includes(structure) ? (structure as ConceptStructure) : draft.structure,
          values: values != null ? lines(values) : draft.values,
          templates: templates != null ? lines(templates) : draft.templates,
          evalTemplates: evalTemplates != null ? lines(evalTemplates) : draft.evalTemplates,
        }))
      }
      const read = asString(fields.read)
      if (read && (READS as string[]).includes(read)) lab.setRead(read as GeometryRead)
      const chat = asBool(fields.chat_template)
      if (chat != null) lab.setChatTemplate(chat)
      const seed = asNumber(fields.seed)
      if (seed != null) lab.setSeed(Math.max(0, Math.floor(seed)))
      const layer = asNumber(fields.layer)
      const run = shown?.run ?? lab.run
      if (layer != null) lab.setPosition(run ? Math.min(layer, run.num_layers) : layer)
      const view = asString(fields.view)
      const implied = VIEW_CONTROLS.find(([, specs]) => specs.some((c) => c.id in fields))?.[0]
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as GeometryViewId)
      else if (implied) lab.setView(implied)
    },
    load: (tool, data) => {
      const d = data as Record<string, unknown>
      if (tool === 'run_geometry') {
        const { run } = d as { run: GeometryRun }
        if (run) lab.loadRun(run)
        return
      }
      // Another view's run: show its run, layer and view (the view's part opens the result)
      const view = VIEW_TOOLS[tool]
      if (!view) return
      lab.setView(view)
      if (tool === 'build_atlas') {
        const model = typeof d.model === 'string' ? d.model : null
        if (model && model !== lab.modelId) lab.setModelId(model)
        return
      }
      const shown = localRun(d.run_id as string)
      if (shown) lab.selectRun(shown.id)
      const response = d.response as { position?: number } | undefined
      const position = tool === 'compare_geometry' ? (d.layer as number) : response?.position
      if (typeof position === 'number') lab.setPosition(position)
    },
  }
}

// The compare view (CompareBody)

export function comparePart(a: {
  runId: string | null
  runs: { id: number; runId: string | null; label: string }[]
  otherId: number
  setOtherId: (id: number) => void
  dims: number | 'all'
  setDims: (dims: number | 'all') => void
  result: CompareResult | null
}): BridgeParts {
  return {
    controls: ids(COMPARE_CONTROLS),
    tools: ['compare_geometry'],
    snapshot: () => {
      const r = a.result
      const at = r?.position ?? 0
      return {
        compare_view: {
          run_id: a.runId,
          compare_with: a.runs.find((x) => x.id === a.otherId)?.runId ?? null,
          dims: a.dims,
          result: r
            ? {
                best_aligned_layer: at, r2: round(r.r2[at]), shuffled_max_r2: round(r.shuffled_r2.max[at]),
                cka: round(r.cka[at]), shuffled_max_cka: round(r.shuffled_cka.max[at]),
              }
            : null,
        },
      }
    },
    apply: (fields) => {
      const other = asString(fields.compare_with)?.trim()
      const match = other ? a.runs.find((x) => x.runId === other) : null
      if (match) a.setOtherId(match.id)
      const dims = asString(fields.compare_dims)
      if (dims) a.setDims(dims === 'all' ? 'all' : Number(dims))
    },
    load: (tool, data) => {
      if (tool !== 'compare_geometry') return
      const { other_run_id, dims } = data as { other_run_id: string; dims: number }
      const match = a.runs.find((x) => x.runId === other_run_id)
      if (match) a.setOtherId(match.id)
      a.setDims(dims ? dims : 'all')
    },
  }
}

// The steer view (SteerBody, one per run)

export function steerPart(a: {
  runId: string | null
  values: string[]
  source: number
  setSource: (i: number) => void
  target: number
  setTarget: (i: number) => void
  steps: number
  setSteps: (n: number) => void
  dims: number
  setDims: (n: number) => void
  evalTemplate: string
  setEvalTemplate: (t: string) => void
  estimate: number | null
  result: SteerResult | null
  onResult: (r: SteerResult) => void
}): BridgeParts {
  const index = (value: string | null) => {
    if (value == null) return -1
    const text = value.trim().toLowerCase()
    return a.values.findIndex((v) => v.toLowerCase() === text)
  }
  return {
    controls: ids(STEER_CONTROLS),
    tools: ['steer_geometry'],
    snapshot: () => {
      const r = a.result
      const walk = (method: 'manifold' | 'linear') => {
        const m = r?.methods[method]
        return m ? { nearest_by_step: m.nearest.map((i) => r!.values[i]), final_to_target: round(m.to_target[m.to_target.length - 1] ?? null) } : null
      }
      return {
        steer_view: {
          run_id: a.runId,
          from: a.values[a.source] ?? null,
          to: a.values[a.target] ?? null,
          steps: a.steps,
          dims: a.dims,
          eval_prompt: cut(a.evalTemplate, 200),
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r ? { position: r.position, manifold: walk('manifold'), linear: walk('linear') } : null,
        },
      }
    },
    apply: (fields) => {
      // Fields for another run wait for its view
      const run = asString(fields.run)?.trim()
      if (run && run !== a.runId) return false
      const source = index(asString(fields.steer_source))
      if (source >= 0) a.setSource(source)
      const target = index(asString(fields.steer_target))
      if (target >= 0) a.setTarget(target)
      const steps = asNumber(fields.steer_steps)
      if (steps != null) a.setSteps(Math.max(4, Math.min(32, Math.round(steps))))
      const dims = asNumber(fields.steer_dims)
      if (dims != null) a.setDims(Math.max(1, Math.min(a.values.length - 1, Math.round(dims))))
      const template = asString(fields.steer_template)
      if (template != null) a.setEvalTemplate(template)
    },
    load: (tool, data) => {
      if (tool !== 'steer_geometry') return
      const { run_id, response } = data as { run_id: string; response: SteerResult }
      // A result for another run waits for its view
      if (run_id !== a.runId) return false
      a.setSource(response.source)
      a.setTarget(response.target)
      a.setSteps(response.steps)
      a.setDims(response.dims)
      a.setEvalTemplate(response.eval_template)
      a.onResult(response)
    },
  }
}

// The sae view (SaeBody, one per run)

export function saePart(a: {
  runId: string | null
  sae: string | null
  pick: (key: string) => void
  estimate: number | null
  result: GeometrySaeResult | null
  onResult: (result: GeometrySaeResult, key: string) => void
}): BridgeParts {
  return {
    controls: ids(SAE_CONTROLS),
    tools: ['geometry_sae'],
    snapshot: () => {
      const r = a.result
      return {
        sae_view: {
          run_id: a.runId,
          sae: a.sae,
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r
            ? {
                sae: r.sae.info.name, l0: round(r.sae.l0, 1), fvu: round(r.sae.fvu),
                fits: r.fits.map((f) => ({ dims: f.dims ?? 'all', verdict: f.verdict, features_for_90: f.n90 })),
              }
            : null,
        },
      }
    },
    apply: (fields) => {
      const run = asString(fields.run)?.trim()
      if (run && run !== a.runId) return false
      const sae = asString(fields.sae)
      if (sae) a.pick(sae)
    },
    load: (tool, data) => {
      if (tool !== 'geometry_sae') return
      const { run_id, sae, response } = data as { run_id: string; sae: string; response: GeometrySaeResult }
      if (run_id !== a.runId) return false
      a.pick(sae)
      a.onResult(response, sae)
    },
  }
}

// The atlas view (AtlasBody, one per model)

export function atlasPart(a: {
  model: string
  sae: string
  setChoice: (key: string) => void
  corpus: AtlasCorpusId
  setCorpus: (c: AtlasCorpusId) => void
  tokens: AtlasTokens
  setTokens: (t: AtlasTokens) => void
  estimate: number | null
  built: string | null
  job: string | null
  refresh: () => void
}): BridgeParts {
  return {
    controls: ids(ATLAS_CONTROLS),
    tools: ['build_atlas'],
    snapshot: () => ({
      atlas_view: {
        model: a.model, sae: a.sae || null, corpus: a.corpus, tokens: a.tokens,
        estimate_usd: a.estimate != null ? round(a.estimate, 2) : null, atlas_built: a.built, job: a.job,
      },
    }),
    apply: (fields) => {
      const model = asString(fields.model)
      if (model && model !== a.model) return false
      const sae = asString(fields.atlas_sae)
      if (sae) a.setChoice(sae)
      const corpus = asString(fields.atlas_corpus)
      if (corpus && (CORPORA as string[]).includes(corpus)) a.setCorpus(corpus as AtlasCorpusId)
      const tokens = asNumber(fields.atlas_tokens)
      if (tokens != null && (TOKENS as number[]).includes(tokens)) a.setTokens(tokens as AtlasTokens)
    },
    load: (tool, data) => {
      if (tool !== 'build_atlas') return
      const { model, sae, corpus, tokens } = data as { model: string; sae: string; corpus: AtlasCorpusId; tokens: AtlasTokens }
      if (model !== a.model) return false
      a.setChoice(sae)
      a.setCorpus(corpus)
      a.setTokens(tokens)
      a.refresh()
    },
  }
}
