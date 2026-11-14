/**
 * The Generate page's assistant bridge, in two parts: the generator's inputs (TextGenerator) and
 * the page's visualizations (which generation is open, and which visualization). Their snapshots
 * merge. A generate run's result opens by its route (/?id=<activation id>); the generator shows
 * the prompt and model it ran with.
 */

import type { BridgeParts } from '../../../hooks/useAssistantPage'
import type { VisualizationType } from '../../../hooks/useActivationData'
import { asBool, asNumber, asString, cut, type ControlSpec, type Fields } from './common'

export const GENERATE_CONTROLS: ControlSpec[] = [
  { id: 'prompt' },
  { id: 'model' },
  { id: 'max_tokens' },
  { id: 'until_eos' },
  { id: 'temperature' },
  { id: 'top_k' },
  { id: 'mode' },
  { id: 'visualization' },
  { id: 'activation' },
  { id: 'generate_button' },
]

const VISUALIZATIONS: VisualizationType[] = ['network', 'attention', 'activations', 'journey', 'flow']

export interface GeneratorBridgeArgs {
  prompt: string
  setPrompt: (prompt: string) => void
  model: string
  setModel: (model: string) => void
  maxTokens: number
  setMaxTokens: (n: number) => void
  untilEos: boolean
  setUntilEos: (on: boolean) => void
  temperature: number
  setTemperature: (t: number) => void
  topK: number
  setTopK: (k: number) => void
  mode: 'visualize' | 'tokens'
  setMode: (mode: 'visualize' | 'tokens') => void
  busy: boolean
}

/** The generator's inputs. */
export function generatorBridge(a: GeneratorBridgeArgs): BridgeParts {
  return {
    snapshot: () => ({
      page: 'generate',
      prompt: cut(a.prompt),
      model: a.model,
      max_tokens: a.untilEos ? null : a.maxTokens,
      until_eos: a.untilEos,
      temperature: a.temperature,
      top_k: a.topK,
      mode: a.mode,
      generating: a.busy || undefined,
    }),
    apply: (fields: Fields) => {
      const prompt = asString(fields.prompt)
      if (prompt != null) a.setPrompt(prompt)
      const model = asString(fields.model)
      if (model) a.setModel(model)
      const until = asBool(fields.until_eos)
      if (until != null) a.setUntilEos(until)
      const max = asNumber(fields.max_tokens)
      if (max != null) a.setMaxTokens(Math.max(1, Math.min(300, Math.round(max))))
      const temperature = asNumber(fields.temperature)
      if (temperature != null) a.setTemperature(Math.max(0, Math.min(2, temperature)))
      const topK = asNumber(fields.top_k)
      if (topK != null) a.setTopK(Math.max(1, Math.min(100, Math.round(topK))))
      const mode = asString(fields.mode)
      if (mode === 'visualize' || mode === 'tokens') a.setMode(mode)
    },
    load: (tool, data) => {
      if (tool !== 'generate') return
      const { prompt, model } = data as { prompt?: string; model?: string }
      if (model) a.setModel(model)
      if (prompt != null) a.setPrompt(prompt)
      a.setMode('visualize')
    },
  }
}

export interface VisualizationBridgeArgs {
  activationId: string | null
  openActivation: (id: string) => void
  loaded: { model?: string; prompt?: string; tokens?: number } | null
  visualization: VisualizationType
  setVisualization: (v: VisualizationType) => void
  layer: number
  head: number
  tokenIndex: number
}

/** The page's visualizations: which stored generation is open, and how it's shown. */
export function visualizationBridge(a: VisualizationBridgeArgs): BridgeParts {
  return {
    snapshot: () => ({
      activation: a.activationId,
      loaded: a.loaded,
      visualization: a.visualization,
      selected: { layer: a.layer, head: a.head, token: a.tokenIndex },
    }),
    apply: (fields: Fields) => {
      const visualization = asString(fields.visualization)
      if (visualization && (VISUALIZATIONS as string[]).includes(visualization)) a.setVisualization(visualization as VisualizationType)
      const activation = asString(fields.activation)
      if (activation) a.openActivation(activation.trim())
    },
  }
}
