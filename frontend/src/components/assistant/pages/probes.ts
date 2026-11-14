/**
 * The Probe Lab's assistant bridge, in parts. The lab's part (registered by useProbeLab, so both
 * variants have it): the settings, the dataset (built-in, or rows the assistant gives), the
 * probe the views use, a sweep's results with their baselines and its run_id, set_inputs, and a
 * train_probe or save_probe run opening in the lab. It also switches to the view a field or a
 * result is for. Each view registers its own part (its inputs, its result on screen, its run's
 * result): score (TokenHeatmap), steer, ablate, sae and generalize.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { ProbeLab } from '../../../hooks/useProbeLab'
import type {
  AblateProbeResponse,
  GeneralizeRequest,
  GeneralizeResponse,
  ParsedProbeDataset,
  Pooling,
  ProbeMethod,
  ProbeRow,
  ProbeRunResult,
  SaeProbeResponse,
  SavedProbe,
  ScoreProbeResponse,
  SteerProbeRequest,
  SteerProbeResponse,
} from '../../../types/probes'
import type { ProbeViewId } from '../../probes/ProbeViews'
import { asBool, asNumber, asString, cut, round, type ControlSpec, type Fields } from './common'

export const PROBE_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'dataset' },
  { id: 'rows' },
  { id: 'label_names' },
  { id: 'dataset_name' },
  { id: 'chat_template' },
  { id: 'test_fraction' },
  { id: 'seed' },
  { id: 'pooling' },
  { id: 'method' },
  { id: 'layer' },
  { id: 'train_button' },
  { id: 'probe' },
  { id: 'direction' },
]

// Each view's own controls (its part fills them)
export const SCORE_CONTROLS: ControlSpec[] = [{ id: 'score_text' }, { id: 'score_button' }]
export const STEER_CONTROLS: ControlSpec[] = [
  { id: 'steer_prompt' },
  { id: 'steer_alphas' },
  { id: 'steer_max_tokens' },
  { id: 'steer_temperature' },
  { id: 'steer_generated_only' },
  { id: 'steer_ablate' },
  { id: 'steer_button' },
]
export const ABLATE_CONTROLS: ControlSpec[] = [{ id: 'ablate_button' }]
export const SAE_CONTROLS: ControlSpec[] = [{ id: 'sae' }, { id: 'sae_button' }]
export const GENERALIZE_CONTROLS: ControlSpec[] = [
  { id: 'gen_datasets' },
  { id: 'gen_transfer' },
  { id: 'gen_pooling' },
  { id: 'gen_test_fraction' },
  { id: 'gen_chat_template' },
  { id: 'gen_button' },
]

const VIEWS: ProbeViewId[] = ['sweep', 'score', 'generalize', 'steer', 'ablate', 'sae']
const POOLINGS: Pooling[] = ['mean', 'last', 'max']
const METHODS: ProbeMethod[] = ['logreg', 'diff_means']

// The view each part's fields and run results are for
const VIEW_CONTROLS: [ProbeViewId, ControlSpec[]][] = [
  ['score', SCORE_CONTROLS],
  ['steer', STEER_CONTROLS],
  ['ablate', ABLATE_CONTROLS],
  ['sae', SAE_CONTROLS],
  ['generalize', GENERALIZE_CONTROLS],
]
const VIEW_TOOLS: Record<string, ProbeViewId> = {
  score_probe: 'score',
  steer_probe: 'steer',
  ablate_probe: 'ablate',
  compare_probe_sae: 'sae',
  generalize_probes: 'generalize',
}

const ids = (specs: ControlSpec[]) => specs.map((c) => c.id)

function at(values: (number | null)[] | undefined, i: number): number | null {
  return values ? round(values[i] ?? null) : null
}

/** Two label names from set_inputs (the backend sends a checked pair). */
function labelPair(value: unknown): [string, string] | null {
  return Array.isArray(value) && value.length === 2 ? [String(value[0]), String(value[1])] : null
}

function sweepSummary(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod, position: number) {
  const best = POOLINGS.flatMap((p) =>
    METHODS.map((m) => {
      const metrics = run.poolings[p]?.methods[m]?.metrics
      const auroc = metrics?.test_auroc ?? []
      let layer = 1
      for (let i = 1; i < auroc.length; i++) if ((auroc[i] ?? -1) > (auroc[layer] ?? -1)) layer = i
      return {
        pooling: p, method: m, best_layer: layer, test_auroc: at(auroc, layer),
        control_auroc: at(metrics?.control_test_auroc, layer), embeddings_auroc: at(auroc, 0),
        ood_auroc: at(metrics?.ood_auroc, layer),
      }
    }),
  )
  const metrics = run.poolings[pooling]?.methods[method]?.metrics
  return {
    run_id: run.run_id ?? null,
    model: run.model,
    dataset: run.dataset.name,
    labels: run.dataset.label_names,
    examples: { train: run.dataset.n_train, test: run.dataset.n_test, ood: run.dataset.n_ood },
    majority_test_acc: round(run.majority.test_acc),
    layers: run.num_layers,
    best,
    shown: {
      pooling, method, layer: position,
      test_auroc: at(metrics?.test_auroc, position), control_auroc: at(metrics?.control_test_auroc, position),
      test_acc: at(metrics?.test_acc, position), selectivity: at(metrics?.selectivity, position),
      ood_auroc: at(metrics?.ood_auroc, position),
    },
  }
}

export function probeBridge(lab: ProbeLab): BridgeParts {
  return {
    controls: ids(PROBE_CONTROLS),
    tools: ['train_probe', 'save_probe'],
    snapshot: () => ({
      page: 'probes',
      view: lab.view,
      model: lab.modelId || null,
      dataset: lab.source === 'builtin' ? lab.builtinId : lab.upload ? `upload: ${lab.upload.filename} (${lab.upload.rows.length} rows)` : null,
      chat_template: lab.chatTemplate,
      test_fraction: lab.testFraction,
      seed: lab.seed,
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      training: lab.training || undefined,
      run: lab.run ? sweepSummary(lab.run, lab.pooling, lab.method, lab.position) : null,
      probe: lab.view === 'score' ? lab.heatmapSource : lab.causalSource,
      direction: lab.causalMethod,
      saved_probes: lab.savedProbes.slice(0, 12).map((p) => ({ id: p.id, name: cut(p.name, 60), model: p.model_id, layer: p.layer })),
      more_saved: lab.savedProbes.length > 12 ? lab.savedProbes.length - 12 : undefined,
    }),
    apply: (fields: Fields) => {
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModelId(model)
      const dataset = asString(fields.dataset)
      if (dataset) {
        lab.setSource('builtin')
        lab.setBuiltinId(dataset)
      }
      const rows = fields.rows as unknown
      const names = labelPair(fields.label_names)
      const name = asString(fields.dataset_name)?.trim()
      if (Array.isArray(rows)) {
        lab.setUploadRows({ filename: name || 'Your rows', rows: rows as ProbeRow[], label_names: names ?? ['0', '1'], warnings: [] })
      } else if ((names || name) && lab.upload) {
        lab.setUploadRows({ ...lab.upload, filename: name || lab.upload.filename, label_names: names ?? lab.upload.label_names })
      }
      const chat = asBool(fields.chat_template)
      if (chat != null) lab.setChatTemplate(chat)
      const fraction = asNumber(fields.test_fraction)
      if (fraction != null) lab.setTestFraction(fraction)
      const seed = asNumber(fields.seed)
      if (seed != null) lab.setSeed(Math.max(0, Math.floor(seed)))
      const pooling = asString(fields.pooling)
      if (pooling && (POOLINGS as string[]).includes(pooling)) lab.setPooling(pooling as Pooling)
      const method = asString(fields.method)
      if (method && (METHODS as string[]).includes(method)) lab.setMethod(method as ProbeMethod)
      const layer = asNumber(fields.layer)
      if (layer != null) lab.setPosition(lab.run ? Math.min(layer, lab.run.num_layers) : layer)
      const probe = asString(fields.probe)?.trim()
      if (probe) {
        const source = probe === 'sweep' ? 'sweep' : `saved:${Number(probe)}`
        lab.setHeatmapSource(source)
        lab.setCausalSource(source)
      }
      const direction = asString(fields.direction)
      if (direction && (METHODS as string[]).includes(direction)) lab.setCausalMethod(direction as ProbeMethod)
      // The view: the one asked for, or the one whose controls were set
      const view = asString(fields.view)
      const implied = VIEW_CONTROLS.find(([, specs]) => specs.some((c) => c.id in fields))?.[0]
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as ProbeViewId)
      else if (implied) lab.setView(implied)
    },
    load: (tool, data) => {
      const d = data as Record<string, unknown>
      if (tool === 'train_probe') {
        const { run, test_fraction, upload } = d as { run: ProbeRunResult; test_fraction: number; upload: ParsedProbeDataset | null }
        if (run) lab.loadRun(run, test_fraction, upload)
        return
      }
      if (tool === 'save_probe') {
        if (d.probe) lab.addSaved(d.probe as SavedProbe)
        return
      }
      // A view's run: show the view and its probe (the view's part opens the result)
      const view = VIEW_TOOLS[tool]
      if (!view) return
      lab.setView(view)
      const probeId = typeof d.probe_id === 'number' ? d.probe_id : null
      if (probeId == null) return
      if (tool === 'score_probe') lab.setHeatmapSource(`saved:${probeId}`)
      else lab.setCausalSource(`saved:${probeId}`)
      if (!lab.savedProbes.some((p) => p.id === probeId)) lab.refreshSaved()
    },
  }
}

// The score view (TokenHeatmap)

export interface ScoredResult {
  response: ScoreProbeResponse
  scale: number
  labelNames: [string, string]
  description: string
}

export function scorePart(a: {
  source: string
  text: string
  setText: (text: string) => void
  scored: ScoredResult | null
  setScored: (scored: ScoredResult) => void
}): BridgeParts {
  return {
    controls: ids(SCORE_CONTROLS),
    tools: ['score_probe'],
    snapshot: () => {
      const r = a.scored?.response.results[0]
      return {
        score_view: {
          probe: a.source,
          text: cut(a.text, 300),
          result: r && a.scored
            ? { probe: a.scored.description, score: round(r.score), reads_as: a.scored.labelNames[r.score > 0 ? 1 : 0], tokens: r.tokens.length }
            : null,
        },
      }
    },
    apply: (fields) => {
      const text = asString(fields.score_text)
      if (text != null) a.setText(text)
    },
    load: (tool, data) => {
      if (tool !== 'score_probe') return
      const { texts, response, labels, scale, description } = data as {
        texts: string[]; response: ScoreProbeResponse; labels: [string, string]; scale: number | null; description: string
      }
      a.setText(texts[0] ?? '')
      a.setScored({ response, scale: scale ?? 0, labelNames: labels, description })
    },
  }
}

// The steer view (SteeringPanel)

export function steerPart(a: {
  probeKey: string
  prompt: string
  setPrompt: (v: string) => void
  alphaText: string
  setAlphaText: (v: string) => void
  maxTokens: number
  setMaxTokens: (v: number) => void
  temperature: number
  setTemperature: (v: number) => void
  generatedOnly: boolean
  setGeneratedOnly: (v: boolean) => void
  ablate: boolean
  setAblate: (v: boolean) => void
  estimate: number | null
  result: { key: string; response: SteerProbeResponse } | null
  setResult: (r: { key: string; response: SteerProbeResponse }) => void
}): BridgeParts {
  return {
    controls: ids(STEER_CONTROLS),
    tools: ['steer_probe'],
    snapshot: () => {
      const r = a.result && a.result.key === a.probeKey ? a.result.response : null
      return {
        steer_view: {
          probe: a.probeKey,
          prompt: cut(a.prompt, 300),
          alphas: a.alphaText,
          max_tokens: a.maxTokens,
          temperature: a.temperature,
          generated_only: a.generatedOnly,
          ablate: a.ablate,
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r
            ? r.generations.map((g) => ({
                kind: g.kind, alpha: g.alpha, probe_score: round(g.probe_score), perplexity: round(g.perplexity, 2),
                tokens_changed: g.tokens_changed, text: cut(g.text, 160),
              }))
            : null,
        },
      }
    },
    apply: (fields) => {
      const prompt = asString(fields.steer_prompt)
      if (prompt != null) a.setPrompt(prompt)
      const alphas = asString(fields.steer_alphas)
      if (alphas != null) a.setAlphaText(alphas)
      const max = asNumber(fields.steer_max_tokens)
      if (max != null) a.setMaxTokens(Math.max(1, Math.min(200, Math.round(max))))
      const temperature = asNumber(fields.steer_temperature)
      if (temperature != null) a.setTemperature(Math.max(0, Math.min(2, temperature)))
      const generatedOnly = asBool(fields.steer_generated_only)
      if (generatedOnly != null) a.setGeneratedOnly(generatedOnly)
      const ablate = asBool(fields.steer_ablate)
      if (ablate != null) a.setAblate(ablate)
    },
    load: (tool, data) => {
      if (tool !== 'steer_probe') return
      const { probe_id, request, response } = data as { probe_id: number; request: SteerProbeRequest; response: SteerProbeResponse }
      a.setPrompt(request.prompt)
      a.setAlphaText(request.alphas.filter((x) => x !== 0).join(', '))
      a.setMaxTokens(request.max_tokens ?? 40)
      a.setTemperature(request.temperature ?? 0)
      a.setGeneratedOnly(!!request.generated_only)
      a.setAblate(!!request.ablate)
      a.setResult({ key: `saved:${probe_id}`, response })
    },
  }
}

// The ablate view (AblationPanel)

export function ablatePart(a: {
  probeKey: string
  estimate: number | null
  result: { key: string; response: AblateProbeResponse } | null
  setResult: (r: { key: string; response: AblateProbeResponse }) => void
}): BridgeParts {
  return {
    controls: ids(ABLATE_CONTROLS),
    tools: ['ablate_probe'],
    snapshot: () => {
      const r = a.result && a.result.key === a.probeKey ? a.result.response : null
      const p = r ? r.layer + 1 : 0
      return {
        ablate_view: {
          probe: a.probeKey,
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r
            ? {
                dataset: r.dataset.name,
                behaviour: {
                  probe: { kl: round(r.behaviour.probe.kl), top1_agreement: round(r.behaviour.probe.top1_agreement) },
                  control: { kl: round(r.behaviour.control.kl), top1_agreement: round(r.behaviour.control.top1_agreement) },
                },
                test_auroc_at_probe_layer: Object.fromEntries(Object.entries(r.curves).map(([k, v]) => [k, round(v[p] ?? null)])),
              }
            : null,
        },
      }
    },
    apply: () => {},
    load: (tool, data) => {
      if (tool !== 'ablate_probe') return
      const { probe_id, response } = data as { probe_id: number; response: AblateProbeResponse }
      a.setResult({ key: `saved:${probe_id}`, response })
    },
  }
}

// The sae view (SaePanel)

export function saePart(a: {
  probeKey: string
  sae: string | null
  setChoice: (key: string) => void
  estimate: number | null
  result: { key: string; response: SaeProbeResponse } | null
  setResult: (r: { key: string; response: SaeProbeResponse }) => void
}): BridgeParts {
  return {
    controls: ids(SAE_CONTROLS),
    tools: ['compare_probe_sae'],
    snapshot: () => {
      const r = a.result && a.result.key === `${a.probeKey}|${a.sae}` ? a.result.response : null
      return {
        sae_view: {
          probe: a.probeKey,
          sae: a.sae,
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r
            ? {
                nearest: r.alignment.features.slice(0, 5).map((f) => ({ feature: f.feature, cos: round(f.cos) })),
                random_max_cos: round(r.alignment.random_max_cos),
                by_contribution: r.alignment.contributions.features.slice(0, 5).map((f) => ({ feature: f.feature, contribution: round(f.contribution) })),
                dense_test_auroc: round(r.dense.test_auroc ?? null),
              }
            : null,
        },
      }
    },
    apply: (fields) => {
      const sae = asString(fields.sae)
      if (sae) a.setChoice(sae)
    },
    load: (tool, data) => {
      if (tool !== 'compare_probe_sae') return
      const { probe_id, sae, response } = data as { probe_id: number; sae: string; response: SaeProbeResponse }
      a.setChoice(sae)
      a.setResult({ key: `saved:${probe_id}|${sae}`, response })
    },
  }
}

// The generalize view (GeneralizationPanel)

export function generalizePart(a: {
  datasets: string[]
  setDatasets: (items: { id: string; part: 'main' | 'ood' }[]) => void
  transfer: boolean
  setTransfer: (v: boolean) => void
  pooling: Pooling
  setPooling: (v: Pooling) => void
  testFraction: number
  setTestFraction: (v: number) => void
  chatTemplate: boolean
  setChatTemplate: (v: boolean) => void
  estimate: number | null
  result: GeneralizeResponse | null
  show: (request: GeneralizeRequest, response: GeneralizeResponse) => void
}): BridgeParts {
  return {
    controls: ids(GENERALIZE_CONTROLS),
    tools: ['generalize_probes'],
    snapshot: () => {
      const r = a.result
      return {
        generalize_view: {
          datasets: a.datasets,
          transfer: a.transfer,
          pooling: a.pooling,
          test_fraction: a.testFraction,
          chat_template: a.chatTemplate,
          estimate_usd: a.estimate != null ? round(a.estimate, 4) : null,
          result: r ? { models: r.models, datasets: r.datasets.map((d) => d.name), layers: r.num_layers } : null,
        },
      }
    },
    apply: (fields) => {
      const list = asString(fields.gen_datasets)
      if (list != null) {
        a.setDatasets(
          list
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean)
            .map((item) => {
              const [id, part] = item.split(':')
              return { id: id.trim(), part: part?.trim() === 'ood' ? 'ood' : 'main' }
            }),
        )
      }
      const transfer = asBool(fields.gen_transfer)
      if (transfer != null) a.setTransfer(transfer)
      const pooling = asString(fields.gen_pooling)
      if (pooling && (POOLINGS as string[]).includes(pooling)) a.setPooling(pooling as Pooling)
      const fraction = asNumber(fields.gen_test_fraction)
      if (fraction != null) a.setTestFraction(Math.max(0.1, Math.min(0.5, fraction)))
      const chat = asBool(fields.gen_chat_template)
      if (chat != null) a.setChatTemplate(chat)
    },
    load: (tool, data) => {
      if (tool !== 'generalize_probes') return
      const { request, response } = data as { request: GeneralizeRequest; response: GeneralizeResponse }
      a.show(request, response)
    },
  }
}
