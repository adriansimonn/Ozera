/**
 * Pure helpers for the Probe Lab: positions, metric lookups, ROC curves, and request builders.
 */

import type {
  InlineProbe,
  MetricName,
  MonitorProbeInfo,
  MonitorTrace,
  PerPosition,
  Pooling,
  ProbeEstimate,
  ProbeMethod,
  ProbeRunResult,
  SaveProbeRequest,
  SavedProbe,
} from '../../types/probes'

export type EvalSet = 'test' | 'ood'
export type SweepMetric = 'auroc' | 'acc'

export const METHOD_LABELS: Record<ProbeMethod, string> = {
  logreg: 'Logistic regression',
  diff_means: 'Difference in means',
}
export const METHOD_SHORT: Record<ProbeMethod, string> = { logreg: 'LR', diff_means: 'DiM' }
export const POOLING_LABELS: Record<Pooling, string> = { last: 'Last token', mean: 'Mean', max: 'Max' }

/** A sweep position's label: "emb" for the embeddings, else the layer whose output it is. */
export function positionLabel(position: number): string {
  return position === 0 ? 'emb' : String(position - 1)
}

/** The metric plotted for a metric kind on an evaluation set. */
export function metricKey(metric: SweepMetric, evalSet: EvalSet): MetricName {
  return `${evalSet}_${metric}` as MetricName
}

export function metricValues(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod, key: MetricName): PerPosition {
  return run.poolings[pooling].methods[method].metrics[key] ?? []
}

/** The layer (position >= 1) with the best test AUROC; ties go to the earliest. */
export function bestPosition(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod): number {
  const values = metricValues(run, pooling, method, 'test_auroc')
  let best = 1
  for (let i = 1; i < values.length; i++) {
    if ((values[i] ?? -1) > (values[best] ?? -1)) best = i
  }
  return best
}

/** Scores and labels of the selected probe on an evaluation set. */
export function evalScores(
  run: ProbeRunResult,
  pooling: Pooling,
  method: ProbeMethod,
  position: number,
  evalSet: EvalSet,
): { scores: number[]; labels: number[]; texts: string[] } {
  const result = run.poolings[pooling].methods[method]
  const tensor = evalSet === 'test' ? result.test_scores : result.ood_scores
  const examples = evalSet === 'test' ? run.test_examples : run.ood_examples
  return {
    scores: tensor ? tensor.values[position] : [],
    labels: examples.map((e) => e.label),
    texts: examples.map((e) => e.text),
  }
}

export interface RocPoint {
  fpr: number
  tpr: number
  threshold: number
}

/** ROC curve points (from (0,0) to (1,1)), stepping through thresholds from high to low. */
export function rocCurve(scores: number[], labels: number[]): RocPoint[] {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
  const positives = labels.filter((l) => l === 1).length
  const negatives = labels.length - positives
  if (positives === 0 || negatives === 0) return []
  const points: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: Infinity }]
  let tp = 0
  let fp = 0
  for (let k = 0; k < order.length; k++) {
    const i = order[k]
    if (labels[i] === 1) tp++
    else fp++
    // Tied scores move together
    if (k + 1 < order.length && scores[order[k + 1]] === scores[i]) continue
    points.push({ fpr: fp / negatives, tpr: tp / positives, threshold: scores[i] })
  }
  return points
}

export interface HistogramBin {
  x0: number
  x1: number
  counts: [number, number]
}

/** Per-class counts of scores in equal-width bins spanning the scores (and the threshold 0). */
export function classHistogram(scores: number[], labels: number[], numBins: number): HistogramBin[] {
  if (scores.length === 0) return []
  let lo = Math.min(0, ...scores)
  let hi = Math.max(0, ...scores)
  if (hi - lo < 1e-9) {
    lo -= 1
    hi += 1
  }
  const width = (hi - lo) / numBins
  const bins: HistogramBin[] = Array.from({ length: numBins }, (_, i) => ({
    x0: lo + i * width,
    x1: lo + (i + 1) * width,
    counts: [0, 0],
  }))
  scores.forEach((s, i) => {
    const b = Math.min(numBins - 1, Math.max(0, Math.floor((s - lo) / width)))
    bins[b].counts[labels[i] === 1 ? 1 : 0]++
  })
  return bins
}

/**
 * How far apart the class means are along a sweep probe's direction: the difference-in-means
 * vector projected on the probe's unit weights (the difference in means's own norm, for a
 * difference-in-means probe). Steering adds α times this along the direction.
 */
export function classGap(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod, position: number): number | null {
  const weights = run.poolings[pooling].methods[method].weights.values[position]
  const meanDiff = run.poolings[pooling].methods.diff_means.weights.values[position]
  if (!weights || !meanDiff) return null
  let dot = 0
  let norm = 0
  for (let i = 0; i < weights.length; i++) {
    dot += weights[i] * meanDiff[i]
    norm += weights[i] * weights[i]
  }
  return norm > 0 ? dot / Math.sqrt(norm) : null
}

/** The probe at a sweep position, ready to score text with (or steer and ablate with). */
export function inlineProbe(run: ProbeRunResult, pooling: Pooling, method: ProbeMethod, position: number): InlineProbe {
  const result = run.poolings[pooling].methods[method]
  return {
    layer: position - 1,
    pooling,
    weights: result.weights.values[position],
    bias: result.biases[position] ?? 0,
    chat_template: run.chat_template,
    read_span: run.read_span,
    method,
    class_gap: classGap(run, pooling, method, position),
  }
}

/** A save request for the probe at a sweep position, with its metrics and the sweep it came from. */
export function saveRequest(
  run: ProbeRunResult,
  pooling: Pooling,
  method: ProbeMethod,
  position: number,
  name: string,
  testFraction: number,
): SaveProbeRequest {
  const result = run.poolings[pooling].methods[method]
  const at = (key: MetricName) => result.metrics[key]?.[position] ?? null
  const probe = inlineProbe(run, pooling, method, position)
  return {
    layer: probe.layer,
    pooling,
    weights: probe.weights,
    bias: probe.bias,
    chat_template: probe.chat_template,
    read_span: probe.read_span,
    name,
    model: run.model,
    method,
    normalization: {
      score_mean: at('score_mean'),
      score_std: at('score_std'),
      act_norm: run.poolings[pooling].act_norms[position] ?? null,
      class_gap: probe.class_gap ?? null,
    },
    metrics: {
      test_auroc: at('test_auroc'),
      test_acc: at('test_acc'),
      train_acc: at('train_acc'),
      ood_auroc: at('ood_auroc'),
      ood_acc: at('ood_acc'),
      control_test_auroc: at('control_test_auroc'),
      control_test_acc: at('control_test_acc'),
      control_train_acc: at('control_train_acc'),
      selectivity: at('selectivity'),
      embedding_test_auroc: result.metrics.test_auroc?.[0] ?? null,
      l2: result.l2?.[position] ?? null,
      sweep: {
        test_auroc: result.metrics.test_auroc ?? [],
        control_test_auroc: result.metrics.control_test_auroc ?? [],
        ood_auroc: result.metrics.ood_auroc ?? null,
      },
    },
    dataset: {
      ...run.dataset,
      seed: run.seed,
      test_fraction: testFraction,
    },
  }
}

export function formatMetric(value: number | null | undefined, digits = 3): string {
  return value == null || Number.isNaN(value) ? '–' : value.toFixed(digits)
}

export function formatPercent(value: number | null | undefined): string {
  return value == null || Number.isNaN(value) ? '–' : `${(value * 100).toFixed(1)}%`
}

export function formatScore(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

/** A GPU run's cost estimate, or why it can't run. */
export function estimateText(estimate: ProbeEstimate): string {
  const seconds = Math.ceil(estimate.estimated_seconds)
  return estimate.within_limit
    ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${seconds}s GPU`
    : `Too large: ~${seconds}s GPU (limit ${estimate.max_seconds}s)`
}

/** Download a value as a JSON file. */
export function downloadJson(filename: string, value: unknown): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/** A monitor trace with a streamed chunk of probe scores (positions start..) written in. */
export function appendMonitorScores(
  trace: MonitorTrace | null,
  chunk: { start: number; tokens: string[]; scores: (number | null)[]; prompt_tokens: number },
): MonitorTrace {
  const tokens = trace ? [...trace.tokens] : []
  const scores = trace ? [...trace.scores] : []
  chunk.scores.forEach((score, i) => {
    const position = chunk.start + i
    while (tokens.length < position) {
      tokens.push('')
      scores.push(null)
    }
    tokens[position] = chunk.tokens[i] ?? ''
    scores[position] = score
  })
  return { tokens, scores, promptTokens: trace?.promptTokens ?? chunk.prompt_tokens }
}

/** What the Generate page shows about a saved probe monitoring a generation. */
export function monitorProbeInfo(probe: SavedProbe): MonitorProbeInfo {
  const names = probe.dataset.label_names
  const std = probe.normalization.score_std
  return {
    id: probe.id,
    name: probe.name,
    layer: probe.layer,
    labelNames: Array.isArray(names) && names.length === 2 ? [String(names[0]), String(names[1])] : ['Negative', 'Positive'],
    scale: std != null && std > 0 ? 2 * std : null,
  }
}
