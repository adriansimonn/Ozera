/**
 * The Circuit Tracing page's assistant bridge (registered by useCircuitLab): the model, prompt
 * and target, the user's graphs, the graph on screen in brief (its scores and most influential
 * feature nodes, with their IDs), the pinned nodes and the last intervention; set_inputs; and
 * trace_circuit and intervene_circuit runs opening on the page.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { CircuitLab, CircuitViewId } from '../../../hooks/useCircuitLab'
import type { CircuitGraph, CircuitJob, InterveneResponse } from '../../../types/circuits'
import { CONTROL_KINDS } from '../../../types/circuits'
import { SCALES, errorShare, supportingFeatures } from '../../circuits/circuitUtils'
import { asString, cut, round, type ControlSpec, type Fields } from './common'

export const CIRCUITS_CONTROLS: ControlSpec[] = [
  { id: 'view' },
  { id: 'model' },
  { id: 'transcoders' },
  { id: 'prompt' },
  { id: 'target' },
  { id: 'trace_button' },
  { id: 'graph' },
  { id: 'node' },
  { id: 'nodes' },
  { id: 'scale' },
  { id: 'intervene_button' },
]

const VIEWS: CircuitViewId[] = ['graph', 'intervene']
const LISTED_FEATURES = 12
const LISTED_GRAPHS = 8

function graphSummary(lab: CircuitLab, graph: CircuitGraph) {
  return {
    graph_id: graph.job_id,
    model: graph.model,
    transcoders: graph.transcoders.id,
    prompt: cut(graph.prompt, 300),
    tokens: graph.tokens.length,
    target: { token: graph.target.token, prob: round(graph.target.prob), rank: graph.target.rank + 1 },
    replacement_score: round(graph.scores.replacement),
    completeness_score: round(graph.scores.completeness),
    error_share: round(errorShare(graph)),
    target_inputs: {
      features: round(graph.scores.target_inputs.absolute.feature),
      errors: round(graph.scores.target_inputs.absolute.error),
      tokens: round(graph.scores.target_inputs.absolute.token),
    },
    feature_nodes: lab.features.length,
    edges: graph.edges.source.length,
    top_features: lab.features.slice(0, LISTED_FEATURES).map((node) => ({
      node: node.id,
      at: graph.tokens[node.position],
      activation: round(node.activation, 2),
      influence: round(node.influence),
      support: round(node.support),
      promotes: (node.promotes ?? []).slice(0, 4),
    })),
  }
}

function interventionSummary(response: InterveneResponse) {
  return {
    graph_id: response.job_id,
    nodes: response.nodes.map((n) => n.id),
    scale: response.scale,
    target: response.target.token,
    logit_change: round(response.chosen.change),
    prob: { before: round(response.base.prob), after: round(response.chosen.prob) },
    controls: Object.fromEntries(CONTROL_KINDS.map((kind) => [kind, (response.controls[kind] ?? []).map((c) => round(c.change))])),
    top_after: response.predictions.chosen[0]?.token,
  }
}

/** Pinned nodes from text: node IDs separated by commas, or top:<n>. */
function parseNodes(lab: CircuitLab, text: string): string[] | null {
  const top = /^\s*top\s*:?\s*(\d+)\s*$/i.exec(text)
  if (top) return lab.graph ? supportingFeatures(lab.graph).slice(0, Math.max(1, Number(top[1]))).map((n) => n.id) : null
  const known = new Set(lab.features.map((n) => n.id))
  const ids = text.split(',').map((part) => part.trim()).filter(Boolean)
  return ids.every((id) => known.has(id)) ? ids : null
}

export function circuitsBridge(lab: CircuitLab): BridgeParts {
  return {
    controls: CIRCUITS_CONTROLS.map((c) => c.id),
    tools: ['trace_circuit', 'intervene_circuit'],
    snapshot: () => ({
      page: 'circuits',
      view: lab.view,
      model: lab.modelId || null,
      models_with_transcoders: lab.models.map((m) => m.model_id),
      transcoders: lab.setId || null,
      transcoder_sets: lab.model?.transcoders.map((t) => t.id),
      prompt: cut(lab.prompt, 600),
      target: lab.target || '(top prediction)',
      estimate_usd: lab.estimate ? round(lab.estimate.estimated_cost, 4) : null,
      building: lab.building ? { graph_id: lab.building.job_id, stage: lab.building.progress?.stage ?? lab.building.status } : undefined,
      error: lab.error ?? undefined,
      graphs: lab.jobs.slice(0, LISTED_GRAPHS).map((job) => ({
        graph_id: job.job_id,
        status: job.status,
        model: job.model,
        prompt: cut(job.prompt, 80),
        target: job.summary?.target.token,
      })),
      graph: lab.graph ? graphSummary(lab, lab.graph) : null,
      selected_node: lab.selected ?? undefined,
      pinned_nodes: lab.pinned,
      intervene_view: lab.view === 'intervene' ? { scale: lab.scale, estimate_usd: lab.interveneEstimate ? round(lab.interveneEstimate.estimated_cost, 4) : null } : undefined,
      intervening: lab.intervening || undefined,
      intervention: lab.intervention ? interventionSummary(lab.intervention) : null,
    }),
    apply: (fields: Fields) => {
      const view = asString(fields.view)
      if (view && (VIEWS as string[]).includes(view)) lab.setView(view as CircuitViewId)
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) lab.setModelId(model)
      const transcoders = asString(fields.transcoders)
      if (transcoders) lab.setSetId(transcoders)
      const prompt = asString(fields.prompt)
      if (prompt != null) lab.setPrompt(prompt)
      const target = asString(fields.target)
      if (target != null) lab.setTarget(target)
      const graph = asString(fields.graph)?.trim()
      if (graph) {
        lab.setView('graph')
        lab.openJob(graph)
      }
      const node = asString(fields.node)?.trim()
      if (node && lab.graph?.nodes.some((n) => n.id === node)) {
        lab.setSelected(node)
        if (!view) lab.setView('graph')
      }
      const nodes = asString(fields.nodes)
      if (nodes != null && lab.graph) {
        const ids = parseNodes(lab, nodes)
        if (ids) {
          lab.setPinned(ids)
          if (!view) lab.setView('intervene')
        }
      }
      const scale = asString(fields.scale)
      if (scale != null && SCALES.some((s) => s.value === Number(scale))) {
        lab.setScale(Number(scale))
        if (!view) lab.setView('intervene')
      }
    },
    load: (tool, data) => {
      if (tool === 'trace_circuit') {
        const { job } = data as { job: CircuitJob }
        lab.setModelId(job.model)
        lab.setSetId(job.transcoders)
        lab.setPrompt(job.prompt)
        lab.setTarget(job.target ?? '')
        lab.setView('graph')
        lab.watchJob(job)
      } else if (tool === 'intervene_circuit') {
        lab.showIntervention((data as { response: InterveneResponse }).response)
      }
    },
  }
}
