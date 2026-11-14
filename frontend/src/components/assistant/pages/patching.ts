/**
 * The Activation Patching page's assistant bridge: the experiment's settings and patches, the
 * result on screen (where the outputs diverge), set_inputs, and a run_patching run's result
 * opening on the page.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { InterventionType, PatchSpec, PatchType, PatchingResult } from '../../../types/patching'
import { asNumber, asString, cut, round, type ControlSpec, type Fields } from './common'

export const PATCHING_CONTROLS: ControlSpec[] = [
  { id: 'model' },
  { id: 'source_prompt' },
  { id: 'target_prompt' },
  { id: 'patches' },
  { id: 'max_tokens' },
  { id: 'temperature' },
  { id: 'run_button' },
]

export interface PatchingBridgeArgs {
  model: string
  setModel: (model: string) => void
  layers: number | null
  heads: number | null
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
  captured: number
}

function patchText(p: PatchSpec): string {
  const parts = [`L${p.layer} ${p.patch_type} ${p.intervention_type}`]
  if (p.heads?.length) parts.push(`heads ${p.heads.join(',')}`)
  if (p.positions?.length) parts.push(`positions ${p.positions.join(',')}`)
  if (p.blend_factor !== 1) parts.push(`blend ${p.blend_factor}`)
  return parts.join(' ')
}

/** Patches from set_inputs or a run (checked by the backend), as the page keeps them. */
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

export function patchingBridge(a: PatchingBridgeArgs): BridgeParts {
  return {
    snapshot: () => {
      const r = a.result
      return {
        page: 'patching',
        model: a.model || null,
        layers: a.layers,
        heads_per_layer: a.heads,
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
        captured_activations: a.captured,
      }
    },
    apply: (fields: Fields) => {
      // The backend checked the model exists (the list may still be loading on a page just opened)
      const model = asString(fields.model)
      if (model) a.setModel(model)
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
    },
  }
}
