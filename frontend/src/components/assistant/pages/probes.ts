/**
 * The Probe Lab's assistant bridge (registered by useProbeLab, so both variants have it): the
 * settings, and a sweep's results with their baselines (each pooling and method's best layer
 * against its control task, and the layer on screen), set_inputs, and a train_probe run's
 * sweep opening in the lab.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { ProbeLab } from '../../../hooks/useProbeLab'
import type { Pooling, ProbeMethod, ProbeRunResult } from '../../../types/probes'
import type { ProbeViewId } from '../../probes/ProbeViews'
import { asBool, asNumber, asString, round, type ControlSpec, type Fields } from './common'

export const PROBE_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'dataset' },
  { id: 'chat_template' },
  { id: 'test_fraction' },
  { id: 'seed' },
  { id: 'pooling' },
  { id: 'method' },
  { id: 'layer' },
  { id: 'train_button' },
]

const VIEWS: ProbeViewId[] = ['sweep', 'score', 'generalize', 'steer', 'ablate', 'sae']
const POOLINGS: Pooling[] = ['mean', 'last', 'max']
const METHODS: ProbeMethod[] = ['logreg', 'diff_means']

function at(values: (number | null)[] | undefined, i: number): number | null {
  return values ? round(values[i] ?? null) : null
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
      saved_probes: lab.savedProbes.length,
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
      const view = asString(fields.view)
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as ProbeViewId)
    },
    load: (tool, data) => {
      if (tool !== 'train_probe') return
      const { run, test_fraction } = data as { run: ProbeRunResult; test_fraction: number }
      if (run) lab.loadRun(run, test_fraction)
    },
  }
}
