/** Text for autointerp labels: model names, scores, and a label's title. */

import type { FeatureLabel } from '../../types/autointerp'

const MODEL_NAMES: Record<string, string> = {
  'claude-haiku-4-5': 'Claude Haiku 4.5',
  'claude-sonnet-5-5': 'Claude Sonnet 5.5',
  'claude-opus-5-5': 'Claude Opus 5.5',
}

export function modelName(id: string): string {
  return MODEL_NAMES[id] ?? id
}

export function formatScore(value: number | null | undefined): string {
  return value == null ? '–' : value.toFixed(2)
}

/** The label's title: the full explanation, its scores (0.5 is chance), and who wrote it. */
export function labelTitle(label: FeatureLabel): string {
  const scores =
    label.status === 'scored'
      ? `Score ${formatScore(label.score)} (detection ${formatScore(label.detection)}, fuzzing ${formatScore(label.fuzzing)}; 0.5 is chance)` +
        (label.token_feature ? ' · token feature' : '')
      : 'Not scored'
  return `${label.explanation}\n${scores}\nExplained by ${modelName(label.explainer)}`
}
