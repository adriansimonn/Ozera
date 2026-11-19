/**
 * The Attribution page's assistant bridge, in two parts. The lab's part (registered by
 * useAttributionLab): the model and pairs, the sweep on screen with its largest effects and
 * its run_id, what the verify and ablate views measured, set_inputs, and run_attribution,
 * verify_attribution and ablate_components runs opening on the page. The continuation view
 * registers its own (ContinuationView): its prompts and patches, the two continuations, and a
 * run_patching run's result.
 */

import { decodeAttributionRun } from '../../../api/client'
import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { AttributionLab, AttributionViewId, VerifyMode } from '../../../hooks/useAttributionLab'
import type { AblateResponse, PairRow, PairSpec, PositionKind, VerifyResponse } from '../../../types/attribution'
import { POSITION_KINDS } from '../../../types/attribution'
import type { InterventionType, PatchSpec, PatchType, PatchingResult } from '../../../types/patching'
import { cellName, componentName, headName, positionGrid, scopeOptions, tokenLabel, topComponents } from '../../attribution/attributionUtils'
import { asNumber, asString, cut, round, type ControlSpec, type Fields } from './common'

export const ATTRIBUTION_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'task' },
  { id: 'pairs' },
  { id: 'rows' },
  { id: 'seed' },
  { id: 'metric' },
  { id: 'run_button' },
  { id: 'sites' },
  { id: 'scope' },
  { id: 'verify_cells' },
  { id: 'verify_count' },
  { id: 'verify_button' },
  { id: 'head' },
  { id: 'components' },
  { id: 'ablate_button' },
]

// The continuation view's own controls (its part fills them)
export const CONTINUATION_CONTROLS: ControlSpec[] = [
  { id: 'source_prompt' },
  { id: 'target_prompt' },
  { id: 'patches' },
  { id: 'max_tokens' },
  { id: 'temperature' },
  { id: 'patch_button' },
]

const VIEWS: AttributionViewId[] = ['sweep', 'verify', 'heads', 'ablate', 'continuation']
const VERIFY_MODES: VerifyMode[] = ['top', 'positions', 'heads', 'resid']

const ids = (specs: ControlSpec[]) => specs.map((c) => c.id)

function sweepSummary(lab: AttributionLab) {
  const run = lab.run
  if (!run) return null
  const scope = lab.scopeInfo
  const grid = scope ? positionGrid(run, scope, lab.sites) : []
  const cells = grid
    .flatMap((row, layer) => row.map((estimate, position) => ({ layer, position, estimate })))
    .sort((a, b) => Math.abs(b.estimate) - Math.abs(a.estimate))
    .slice(0, 6)
  return {
    run_id: run.run_id,
    model: run.model,
    task: run.task_name,
    pairs: run.num_pairs,
    dropped_pairs: run.dropped.length || undefined,
    metric: run.metric,
    clean: round(run.clean_mean),
    corrupted: round(run.corrupted_mean),
    scale: run.normalized ? 'corrupted run 0, clean run 1' : "the metric's own units (no gap between the prompts)",
    pairs_solved: round(run.solved, 2),
    top_components: topComponents(run, 8).map((cell) => ({
      cell: cellName(cell),
      estimate: round(cell.estimate),
      exact: cell.kind === 'head' ? round(lab.exactHeads[headName(cell.layer, cell.head ?? 0)]?.effect) : undefined,
    })),
    position_map: scope
      ? {
          shows: lab.sites,
          of: lab.scope,
          top_cells: cells.map((c) => ({ layer: c.layer, position: c.position, token: tokenLabel(scope.labels[c.position]), estimate: round(c.estimate) })),
        }
      : null,
  }
}

function verifySummary(result: { mode: VerifyMode; response: VerifyResponse }) {
  const cells = [...result.response.cells].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, 10)
  return {
    cells: result.mode,
    patched: result.response.cells.length,
    rank_correlation: round(result.response.rank_correlation),
    largest_exact: cells.map((c) => ({ cell: cellName(c), exact: round(c.effect), se: round(c.se), estimate: round(c.estimate) })),
  }
}

function ablateSummary(response: AblateResponse) {
  return {
    components: response.components.map(componentName),
    metric_ablated: Object.fromEntries(
      Object.entries(response.methods).map(([method, r]) => [
        method,
        { value: round(r.value), se: round(r.se), random_sets: r.random.map((v) => round(v)) },
      ]),
    ),
    scale: response.normalized ? 'clean run 1, corrupted run 0' : "the metric's own units",
  }
}

export function attributionBridge(lab: AttributionLab): BridgeParts {
  return {
    controls: ids(ATTRIBUTION_CONTROLS),
    tools: ['run_attribution', 'verify_attribution', 'ablate_components'],
    snapshot: () => ({
      page: 'attribution',
      view: lab.view,
      model: lab.modelId || null,
      layers: lab.model?.num_layers ?? null,
      heads_per_layer: lab.model?.num_heads ?? null,
      pairs: lab.source === 'builtin' ? { task: lab.taskId, pairs: lab.pairCount, seed: lab.seed } : lab.upload ? { upload: lab.upload.filename, pairs: lab.upload.rows.length } : null,
      metric: lab.metric,
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      running: lab.running || lab.verifying || lab.ablating || undefined,
      error: lab.error ?? undefined,
      sweep: sweepSummary(lab),
      verify_view: lab.view === 'verify' ? { cells: lab.verifyMode, count: lab.verifyCount, estimate_usd: lab.verifyEstimate ? round(lab.verifyEstimate.estimated_cost, 4) : null } : undefined,
      verified: lab.verified ? verifySummary(lab.verified) : null,
      selected_head: lab.selectedHead ? headName(lab.selectedHead.layer, lab.selectedHead.head) : undefined,
      ablate_view: lab.view === 'ablate' ? { components: lab.componentsText, estimate_usd: lab.ablateEstimate ? round(lab.ablateEstimate.estimated_cost, 4) : null } : undefined,
      ablated: lab.ablated ? ablateSummary(lab.ablated.response) : null,
    }),
    apply: (fields: Fields) => {
      const view = asString(fields.view)
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as AttributionViewId)
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModelId(model)
      const task = asString(fields.task)
      if (task) {
        lab.setTaskId(task)
        lab.setSource('builtin')
      }
      const pairs = asNumber(fields.pairs)
      if (pairs != null) lab.setPairCount(Math.max(1, Math.min(200, Math.round(pairs))))
      if (Array.isArray(fields.rows)) {
        const rows = fields.rows as unknown as PairRow[]
        lab.setUpload({ filename: 'Your pairs', rows, has_answers: rows.every((r) => r.correct != null), warnings: [] })
        lab.setSource('upload')
      }
      const seed = asNumber(fields.seed)
      if (seed != null) lab.setSeed(Math.max(0, Math.round(seed)))
      const metric = asString(fields.metric)
      if (metric === 'logit_diff' || metric === 'kl') lab.setMetric(metric)

      const sites = asString(fields.sites)
      if (sites && (POSITION_KINDS as string[]).includes(sites)) lab.setSites(sites as PositionKind)
      const scope = asString(fields.scope)?.trim().toLowerCase()
      if (scope && lab.run) {
        const options = scopeOptions(lab.run)
        if (scope === 'mean') {
          const mean = options.find((o) => o.value.startsWith('cls:'))
          if (mean) lab.setScope(mean.value)
        } else if (/^\d+$/.test(scope) && Number(scope) >= 1 && Number(scope) <= lab.run.num_pairs) {
          lab.setScope(`pair:${Number(scope) - 1}`)
        }
      }
      const cells = asString(fields.verify_cells)
      if (cells && (VERIFY_MODES as string[]).includes(cells)) {
        lab.setVerifyMode(cells as VerifyMode)
        if (!view) lab.setView('verify')
      }
      const count = asNumber(fields.verify_count)
      if (count != null) lab.setVerifyCount(count)
      const head = /^\s*(\d+)\s*[.:]\s*(\d+)\s*$/.exec(asString(fields.head) ?? '')
      if (head) {
        lab.setSelectedHead({ layer: Number(head[1]), head: Number(head[2]) })
        if (!view) lab.setView('heads')
      }
      const components = asString(fields.components)
      if (components != null) {
        lab.setComponentsText(components)
        if (!view) lab.setView('ablate')
      }
    },
    load: (tool, data) => {
      if (tool === 'run_attribution') {
        const { run, pairs } = data as { run: unknown; pairs: PairSpec }
        lab.loadRun(decodeAttributionRun(run), pairs)
      } else if (tool === 'verify_attribution') {
        const { run_id, what, response } = data as { run_id: string | null; what: VerifyMode; response: VerifyResponse }
        lab.showVerified({ mode: what, runId: run_id, pair: null, response })
        lab.setView('verify')
      } else if (tool === 'ablate_components') {
        const { run_id, response } = data as { run_id: string | null; response: AblateResponse }
        lab.showAblated({ runId: run_id, response })
        lab.setComponentsText(response.components.map(componentName).join(', '))
        lab.setView('ablate')
      }
    },
  }
}

// The continuation view (ContinuationView)

export interface ContinuationBridgeArgs {
  model: string
  setModel: (model: string) => void
  /** Show the continuation view (a run's result opens in it) */
  show: () => void
  sourcePrompt: string
  setSourcePrompt: (text: string) => void
  targetPrompt: string
  setTargetPrompt: (text: string) => void
  patches: PatchSpec[]
  setPatches: (patches: PatchSpec[]) => void
  maxTokens: number
  setMaxTokens: (n: number) => void
  temperature: number
  setTemperature: (t: number) => void
  running: boolean
  result: PatchingResult | null
  setResult: (result: PatchingResult | null) => void
}

function patchText(p: PatchSpec): string {
  const parts = [`L${p.layer} ${p.patch_type} ${p.intervention_type}`]
  if (p.heads?.length) parts.push(`heads ${p.heads.join(',')}`)
  if (p.positions?.length) parts.push(`positions ${p.positions.join(',')}`)
  if (p.blend_factor !== 1) parts.push(`blend ${p.blend_factor}`)
  return parts.join(' ')
}

/** Patches from set_inputs or a run (checked by the backend), as the view keeps them. */
function asPatches(value: unknown): PatchSpec[] | null {
  if (!Array.isArray(value)) return null
  return value.map((p) => {
    const patch = p as Record<string, unknown>
    const list = (v: unknown) => (Array.isArray(v) && v.length ? (v as number[]) : null)
    return {
      layer: Number(patch.layer),
      patch_type: patch.patch_type as PatchType,
      intervention_type: (patch.intervention_type ?? 'patch') as InterventionType,
      positions: list(patch.positions),
      heads: list(patch.heads),
      neurons: list(patch.neurons),
      blend_factor: Number(patch.blend_factor ?? 1),
    }
  })
}

export function continuationPart(a: ContinuationBridgeArgs): BridgeParts {
  return {
    controls: ids(CONTINUATION_CONTROLS),
    tools: ['run_patching'],
    snapshot: () => {
      const r = a.result
      return {
        continuation_view: {
          source_prompt: cut(a.sourcePrompt, 400),
          target_prompt: cut(a.targetPrompt, 400),
          patches: a.patches.slice(0, 16).map(patchText),
          max_tokens: a.maxTokens,
          temperature: a.temperature,
          running: a.running || undefined,
          result: r
            ? {
                baseline_output: cut(r.baseline_output, 400),
                patched_output: cut(r.patched_output, 400),
                first_divergence_position: r.effect_summary.first_divergence_position,
                token_changes: r.effect_summary.token_changes,
                changed_tokens: r.effect_summary.changed_tokens.slice(0, 8),
              }
            : null,
        },
      }
    },
    apply: (fields: Fields) => {
      const source = asString(fields.source_prompt)
      if (source != null) a.setSourcePrompt(source)
      const target = asString(fields.target_prompt)
      if (target != null) a.setTargetPrompt(target)
      const patches = asPatches(fields.patches)
      if (patches) a.setPatches(patches)
      const max = asNumber(fields.max_tokens)
      if (max != null) a.setMaxTokens(Math.max(1, Math.min(100, Math.round(max))))
      const temperature = asNumber(fields.temperature)
      if (temperature != null) a.setTemperature(round(Math.max(0, Math.min(2, temperature)), 2) ?? 0)
      const mine = ['source_prompt', 'target_prompt', 'patches', 'max_tokens', 'temperature'].some((name) => name in fields)
      if (mine && !('view' in fields)) a.show()
    },
    load: (tool, data) => {
      if (tool !== 'run_patching') return
      const { request, result } = data as {
        request: { model: string; source_prompt: string | null; target_prompt: string; patches: unknown; max_tokens: number; temperature: number }
        result: PatchingResult
      }
      a.setModel(request.model)
      a.setSourcePrompt(request.source_prompt ?? '')
      a.setTargetPrompt(request.target_prompt)
      a.setPatches(asPatches(request.patches) ?? [])
      a.setMaxTokens(request.max_tokens)
      a.setTemperature(request.temperature)
      a.setResult(result)
      a.show()
    },
  }
}
