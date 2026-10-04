/**
 * Generalization matrix: probes trained on one dataset, tested on the others, and for an
 * open-source model, across its base and instruct versions. A probe that reads the concept keeps
 * ranking new data correctly; one that learned something particular to its training set doesn't.
 */

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { apiClient } from '../../api/client'
import type { ProbeLab } from '../../hooks/useProbeLab'
import type {
  GeneralizeResponse,
  MatrixDatasetSpec,
  ParsedProbeDataset,
  Pooling,
  ProbeDatasetSummary,
  ProbeEstimate,
  ProbeMethod,
} from '../../types/probes'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, RunBar, Segmented, type Finding, type TooltipRow } from './chartKit'
import { FileButton } from './DatasetPanel'
import { inkOn, transferColor, useElementWidth, useProbeColors, type ProbeColors } from './probeTheme'
import { METHOD_LABELS, POOLING_LABELS, estimateText, formatMetric, formatPercent, positionLabel } from './probeUtils'

const MAX_DATASETS = 6

interface MatrixEntry {
  key: string
  spec: MatrixDatasetSpec
  name: string
  rows: number
  chars: number
  labelNames: [string, string]
}

type MatrixMetric = 'auroc' | 'acc'

function builtinEntry(d: ProbeDatasetSummary, part: 'main' | 'ood'): MatrixEntry {
  const ood = part === 'ood' && d.ood
  return {
    key: `${d.id}:${part}`,
    spec: { builtin_id: d.id, part },
    name: ood ? `${d.name} · OOD` : d.name,
    rows: ood ? ood.num_rows : d.num_rows,
    chars: ood ? ood.total_chars : d.total_chars,
    labelNames: d.label_names,
  }
}

function uploadEntry(parsed: ParsedProbeDataset, key: string): MatrixEntry {
  return {
    key,
    spec: { rows: parsed.rows, name: parsed.filename, label_names: parsed.label_names },
    name: parsed.filename,
    rows: parsed.rows.length,
    chars: parsed.rows.reduce((sum, row) => sum + row.text.length, 0),
    labelNames: parsed.label_names,
  }
}

/** The datasets chosen on the left: the training set, then its OOD set. */
function entriesFromLab(lab: ProbeLab): MatrixEntry[] {
  const entries: MatrixEntry[] = []
  if (lab.source === 'builtin' && lab.builtin) {
    entries.push(builtinEntry(lab.builtin, 'main'))
    if (!lab.oodUpload && lab.builtin.ood) entries.push(builtinEntry(lab.builtin, 'ood'))
  } else if (lab.source === 'upload' && lab.upload) {
    entries.push(uploadEntry(lab.upload, `upload:${lab.upload.filename}`))
  }
  if (lab.oodUpload) entries.push(uploadEntry(lab.oodUpload, `upload-ood:${lab.oodUpload.filename}`))
  return entries
}

/** Hanley & McNeil's standard error of an AUROC. */
function aurocError(auroc: number, positives: number, negatives: number): number {
  const q1 = auroc / (2 - auroc)
  const q2 = (2 * auroc * auroc) / (1 + auroc)
  const variance =
    (auroc * (1 - auroc) + (positives - 1) * (q1 - auroc * auroc) + (negatives - 1) * (q2 - auroc * auroc)) /
    (positives * negatives)
  return Math.sqrt(Math.max(variance, 0))
}

function mean(values: (number | null | undefined)[]): number | null {
  const kept = values.filter((v): v is number => v != null)
  return kept.length ? kept.reduce((a, b) => a + b, 0) / kept.length : null
}

interface Condition {
  model: number
  dataset: number
}

function conditionsOf(result: GeneralizeResponse): Condition[] {
  return result.models.flatMap((_, model) => result.datasets.map((_, dataset) => ({ model, dataset })))
}

/** Per position: the mean over the diagonal, over other datasets (same model), and over the other model (same dataset). */
function layerSummaries(result: GeneralizeResponse, method: ProbeMethod, metric: MatrixMetric) {
  const values = result[metric][method]
  const conditions = conditionsOf(result)
  const positions = result.num_layers + 1
  const series = { diagonal: [] as (number | null)[], datasets: [] as (number | null)[], models: [] as (number | null)[] }
  for (let p = 0; p < positions; p++) {
    const diagonal: (number | null)[] = []
    const datasets: (number | null)[] = []
    const models: (number | null)[] = []
    conditions.forEach((a, i) =>
      conditions.forEach((b, j) => {
        const v = values[i][j][p]
        if (i === j) diagonal.push(v)
        else if (a.model === b.model) datasets.push(v)
        else if (a.dataset === b.dataset) models.push(v)
      }),
    )
    series.diagonal.push(mean(diagonal))
    series.datasets.push(mean(datasets))
    series.models.push(result.models.length > 1 ? mean(models) : null)
  }
  return series
}

/** The layer to show first: the one picked in the sweep (same model), else the best in-distribution layer. */
function defaultPosition(result: GeneralizeResponse, method: ProbeMethod, lab: ProbeLab): number {
  if (lab.run && lab.run.model === result.model && lab.position > 0 && lab.position <= result.num_layers) return lab.position
  const diagonal = layerSummaries(result, method, 'auroc').diagonal
  let best = Math.min(1, diagonal.length - 1)
  diagonal.forEach((v, p) => {
    if (p >= 1 && v != null && v > (diagonal[best] ?? -1)) best = p
  })
  return best
}

function findings(result: GeneralizeResponse, method: ProbeMethod, position: number, modelName: (id: string) => string): Finding[] {
  const out: Finding[] = []
  const values = result.auroc[method]
  const conditions = conditionsOf(result)
  const D = result.datasets.length
  const name = (c: Condition) =>
    result.models.length > 1 ? `${result.datasets[c.dataset].name} (${modelName(result.models[c.model])})` : result.datasets[c.dataset].name
  const summary = layerSummaries(result, method, 'auroc')
  const diag = summary.diagonal[position]
  const across = summary.datasets[position]
  if (diag != null && across != null) {
    const drops = across < diag - 0.15
    out.push({
      level: drops ? 'warning' : 'info',
      text: `Mean AUROC ${formatMetric(diag, 2)} on own dataset, ${formatMetric(across, 2)} on others${drops ? ': doesn’t carry over.' : '.'}`,
    })
  }

  const flipped: { a: Condition; b: Condition; v: number }[] = []
  conditions.forEach((a, i) =>
    conditions.forEach((b, j) => {
      const v = values[i][j][position]
      if (i !== j && a.model === b.model && v != null && v < 0.35) flipped.push({ a, b, v })
    }),
  )
  flipped.sort((x, y) => x.v - y.v)
  if (flipped.length > 0) {
    const shown = flipped.slice(0, 3).map((f) => `${name(f.a)} → ${name(f.b)} (${formatMetric(f.v, 2)})`)
    out.push({
      level: 'warning',
      text: `Ranked backwards: ${shown.join(', ')}${flipped.length > 3 ? `, +${flipped.length - 3} more` : ''}.`,
    })
  }

  if (result.models.length > 1) {
    const [first, second] = result.models
    const toOther = (from: number) =>
      mean(result.datasets.map((_, d) => values[from * D + d][(1 - from) * D + d][position]))
    const own = (from: number) => mean(result.datasets.map((_, d) => values[from * D + d][from * D + d][position]))
    const a = toOther(0)
    const b = toOther(1)
    if (a != null && b != null) {
      out.push({
        level: Math.min(a - (own(0) ?? a), b - (own(1) ?? b)) < -0.1 ? 'warning' : 'info',
        text: `${modelName(first)} → ${modelName(second)}: ${formatMetric(a, 2)} (own ${formatMetric(own(0), 2)}) · ${modelName(second)} → ${modelName(first)}: ${formatMetric(b, 2)} (own ${formatMetric(own(1), 2)}).`,
      })
    }
  }

  if (result.overlap.some((row) => row.some((n) => n > 0))) {
    out.push({ level: 'warning', text: '† Cells share test texts with the training split.' })
  }
  const smallest = Math.min(...result.datasets.map((d) => d.n_test))
  if (smallest < 40) {
    out.push({ level: 'info', text: `Smallest test split: ${smallest} examples.` })
  }
  const labelSets = new Set(result.datasets.map((d) => d.label_names.join(' / ')))
  if (labelSets.size > 1) {
    out.push({ level: 'info', text: `Class names differ (${[...labelSets].join('; ')}); the second label is positive.` })
  }
  return out
}

const CHART_HEIGHT = 220
const CHART_MARGIN = { top: 18, right: 112, bottom: 36, left: 44 }

function linePath(values: (number | null)[], x: (i: number) => number, y: (v: number) => number): string {
  let path = ''
  let pen = false
  values.forEach((v, i) => {
    if (v == null) {
      pen = false
      return
    }
    path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
    pen = true
  })
  return path
}

const SERIES_LABELS = {
  diagonal: 'Same dataset',
  datasets: 'Other datasets',
  models: 'Other model, same dataset',
} as const
// Direct labels at the lines' ends
const SERIES_SHORT = { diagonal: 'same dataset', datasets: 'other datasets', models: 'other model' } as const
type SeriesKey = keyof typeof SERIES_LABELS

function seriesStyles(colors: ProbeColors, method: ProbeMethod): Record<SeriesKey, { color: string; dash?: string }> {
  return {
    diagonal: { color: colors.baseline },
    datasets: { color: colors.methods[method] },
    models: { color: colors.methods[method], dash: '5 4' },
  }
}

/** Mean AUROC (or accuracy) per layer: in distribution, across datasets, and across models. Click a layer to show its matrix. */
function TransferByLayer({
  result,
  method,
  metric,
  position,
  onSelect,
}: {
  result: GeneralizeResponse
  method: ProbeMethod
  metric: MatrixMetric
  position: number
  onSelect: (position: number) => void
}) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const summary = layerSummaries(result, method, metric)
  const styles = seriesStyles(colors, method)
  const shown = (Object.keys(SERIES_LABELS) as SeriesKey[]).filter((k) => summary[k].some((v) => v != null))

  const positions = result.num_layers + 1
  const innerW = Math.max(width - CHART_MARGIN.left - CHART_MARGIN.right, 60)
  const innerH = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom
  const step = positions > 1 ? innerW / (positions - 1) : innerW
  const x = (i: number) => CHART_MARGIN.left + (positions > 1 ? i * step : innerW / 2)
  const all = shown.flatMap((k) => summary[k]).filter((v): v is number => v != null)
  const yMin = Math.min(0.4, Math.floor(Math.min(...all, 1) * 10) / 10)
  const y = (v: number) => CHART_MARGIN.top + (1 - (v - yMin) / (1 - yMin)) * innerH
  const tickStep = 1 - yMin > 0.6 ? 0.2 : 0.1
  const ticks: number[] = []
  for (let t = 1; t >= yMin - 1e-9; t -= tickStep) ticks.push(Math.round(t * 100) / 100)
  const labelEvery = Math.max(1, Math.ceil(positions / (innerW / 30)))
  const halo = { stroke: colors.surface, strokeWidth: 3, paintOrder: 'stroke' as const, strokeLinejoin: 'round' as const }

  const ends = shown
    .map((k) => {
      const values = summary[k]
      let last = values.length - 1
      while (last >= 0 && values[last] == null) last--
      return { k, y: last >= 0 ? y(values[last] as number) : null }
    })
    .filter((e): e is { k: SeriesKey; y: number } => e.y != null)
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12)

  const indexAt = (px: number) => Math.max(0, Math.min(positions - 1, Math.round((px - CHART_MARGIN.left) / step)))
  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    setHover({ index: indexAt(px), px, py: e.clientY - box.top })
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft' && position > 0) onSelect(position - 1)
    else if (e.key === 'ArrowRight' && position < positions - 1) onSelect(position + 1)
    else return
    e.preventDefault()
  }
  const metricName = metric === 'auroc' ? 'AUROC' : 'accuracy'
  const tooltipRows = (i: number): TooltipRow[] =>
    shown.map((k) => ({ key: styles[k].color, label: SERIES_LABELS[k], value: formatMetric(summary[k][i]) }))

  return (
    <div>
      <div
        ref={ref}
        className="pl-chart"
        tabIndex={0}
        role="group"
        aria-label={`Mean ${metricName} per layer. Selected: ${position === 0 ? 'embeddings' : `layer ${position - 1}`}. Use arrow keys to change layer.`}
        onKeyDown={onKeyDown}
      >
        <svg width={width} height={CHART_HEIGHT} role="img" aria-hidden="true">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
              <text x={CHART_MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={CHART_MARGIN.top + innerH} y2={CHART_MARGIN.top + innerH} stroke={colors.axis} />
          {Array.from({ length: positions }, (_, i) =>
            i % labelEvery === 0 || i === positions - 1 ? (
              <text key={i} x={x(i)} y={CHART_MARGIN.top + innerH + 16} textAnchor="middle" fontSize={11} fill={colors.text}>
                {positionLabel(i)}
              </text>
            ) : null,
          )}
          <text x={CHART_MARGIN.left + innerW / 2} y={CHART_HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
            Layer
          </text>
          <rect x={x(position) - Math.max(step, 8) / 2} y={CHART_MARGIN.top} width={Math.max(step, 8)} height={innerH} fill={colors.hover} />
          {yMin < 0.5 && (
            <g>
              <line x1={CHART_MARGIN.left} x2={CHART_MARGIN.left + innerW} y1={y(0.5)} y2={y(0.5)} stroke={colors.baseline} opacity={0.7} />
              <text x={CHART_MARGIN.left + 4} y={y(0.5) - 4} fontSize={10} fill={colors.text} {...halo}>
                chance
              </text>
            </g>
          )}
          {shown.map((k) => (
            <path
              key={k}
              d={linePath(summary[k], x, y)}
              fill="none"
              stroke={styles[k].color}
              strokeWidth={2}
              strokeDasharray={styles[k].dash}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {shown.map((k) =>
            summary[k][position] != null ? (
              <circle key={k} cx={x(position)} cy={y(summary[k][position] as number)} r={4} fill={styles[k].color} stroke={colors.surface} strokeWidth={2} />
            ) : null,
          )}
          {ends.map((e) => (
            <text key={e.k} x={CHART_MARGIN.left + innerW + 8} y={e.y} dy="0.32em" fontSize={10.5} fill={colors.text}>
              <tspan fill={styles[e.k].color} fontWeight={700}>
                —{' '}
              </tspan>
              {SERIES_SHORT[e.k]}
            </text>
          ))}
          {hover && <line x1={x(hover.index)} x2={x(hover.index)} y1={CHART_MARGIN.top} y2={CHART_MARGIN.top + innerH} stroke={colors.axis} />}
          <rect
            x={CHART_MARGIN.left - step / 2}
            y={CHART_MARGIN.top}
            width={innerW + step}
            height={innerH}
            fill="transparent"
            style={{ cursor: 'pointer' }}
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
            onClick={() => hover && onSelect(hover.index)}
          />
        </svg>
        {hover && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={`${hover.index === 0 ? 'Embeddings' : `Layer ${hover.index - 1}`} · mean ${metricName}`}
            rows={tooltipRows(hover.index)}
          />
        )}
      </div>
      <div className="pl-legend">
        {shown.map((k) => (
          <span key={k} className="pl-legend-item">
            <span
              className="pl-legend-line"
              style={{
                background: styles[k].dash
                  ? `repeating-linear-gradient(to right, ${styles[k].color} 0 5px, transparent 5px 9px)`
                  : styles[k].color,
              }}
            />
            {SERIES_LABELS[k]}
          </span>
        ))}
      </div>
    </div>
  )
}

/** The matrix at one layer: rows are what probes were trained on, columns what they're tested on. */
function TransferMatrix({
  result,
  method,
  metric,
  position,
  modelName,
}: {
  result: GeneralizeResponse
  method: ProbeMethod
  metric: MatrixMetric
  position: number
  modelName: (id: string) => string
}) {
  const colors = useProbeColors()
  const wrap = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ i: number; j: number; x: number; y: number } | null>(null)
  const conditions = conditionsOf(result)
  const D = result.datasets.length
  const multi = result.models.length > 1
  const values = result[metric][method]

  const color = (v: number | null) => (v == null ? colors.panel : transferColor(colors, v))
  const onCell = (e: PointerEvent<HTMLTableCellElement>, i: number, j: number) => {
    const box = wrap.current?.getBoundingClientRect()
    if (box) setHover({ i, j, x: e.clientX - box.left + (wrap.current?.scrollLeft ?? 0), y: e.clientY - box.top })
  }

  const tooltip = (i: number, j: number): TooltipRow[] => {
    const test = result.datasets[conditions[j].dataset]
    const auroc = result.auroc[method][i][j][position]
    const acc = result.acc[method][i][j][position]
    const rows: TooltipRow[] = [
      {
        label: 'AUROC',
        value: auroc == null ? '–' : `${formatMetric(auroc)} ± ${formatMetric(aurocError(auroc, test.n_test_positive, test.n_test - test.n_test_positive), 2)}`,
      },
      { label: 'accuracy at the probe’s threshold', value: formatPercent(acc) },
      { label: 'test examples', value: String(test.n_test) },
    ]
    const overlap = result.overlap[conditions[i].dataset][conditions[j].dataset]
    if (overlap > 0) rows.push({ label: 'also in the training split', value: String(overlap) })
    return rows
  }
  const conditionName = (c: Condition) =>
    multi ? `${result.datasets[c.dataset].name} on ${modelName(result.models[c.model])}` : result.datasets[c.dataset].name

  return (
    <div ref={wrap} className="pl-matrix-wrap" onPointerLeave={() => setHover(null)}>
      <table className="pl-matrix" aria-label={`Test ${metric === 'auroc' ? 'AUROC' : 'accuracy'} of probes trained on each row's dataset and tested on each column's`}>
        <thead>
          {multi && (
            <tr>
              <th colSpan={2} />
              {result.models.map((m) => (
                <th key={m} colSpan={D} scope="colgroup" className="pl-matrix-group" style={{ borderColor: colors.border }}>
                  Tested on {modelName(m)}
                </th>
              ))}
            </tr>
          )}
          <tr>
            <th colSpan={multi ? 2 : 1} className="pl-matrix-corner">
              Tested on →
              <br />
              Trained on ↓
            </th>
            {conditions.map((c, j) => (
              <th key={j} scope="col" title={result.datasets[c.dataset].description ?? undefined}>
                {result.datasets[c.dataset].name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {conditions.map((a, i) => (
            <tr key={i}>
              {multi && a.dataset === 0 && (
                <th rowSpan={D} scope="rowgroup" className="pl-matrix-group-row" style={{ borderColor: colors.border }}>
                  <span>{modelName(result.models[a.model])}</span>
                </th>
              )}
              <th scope="row" title={result.datasets[a.dataset].description ?? undefined}>
                {result.datasets[a.dataset].name}
              </th>
              {conditions.map((b, j) => {
                const v = values[i][j][position]
                const bg = color(v)
                const overlap = result.overlap[a.dataset][b.dataset] > 0
                return (
                  <td
                    key={j}
                    onPointerMove={(e) => onCell(e, i, j)}
                    aria-label={`${conditionName(a)} to ${conditionName(b)}: ${formatMetric(v)}`}
                    style={{
                      background: bg,
                      color: v == null ? colors.text : inkOn(bg),
                      boxShadow: i === j ? `inset 0 0 0 2px ${colors.text}` : undefined,
                    }}
                  >
                    {v == null ? '–' : v.toFixed(2)}
                    {overlap && <sup aria-label="shares texts with the training split">†</sup>}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {hover && (
        <ChartTooltip
          x={hover.x}
          y={hover.y}
          containerWidth={wrap.current?.scrollWidth ?? 600}
          title={`${conditionName(conditions[hover.i])} → ${conditionName(conditions[hover.j])}`}
          rows={tooltip(hover.i, hover.j)}
        />
      )}
    </div>
  )
}

function ScaleLegend() {
  const colors = useProbeColors()
  const stops = [0, 0.25, 0.5, 0.75, 1].map((v) => `${transferColor(colors, v)} ${v * 100}%`).join(', ')
  return (
    <div className="pl-scale" aria-hidden="true">
      <span>0</span>
      <span className="pl-scale-bar" style={{ background: `linear-gradient(to right, ${stops})`, borderColor: colors.border }} />
      <span>1</span>
      <span style={{ marginLeft: '0.75rem' }}>Outlined: same dataset</span>
    </div>
  )
}

export function ProbeGeneralization({ lab, onShowPurchaseCredits }: { lab: ProbeLab; onShowPurchaseCredits?: () => void }) {
  const colors = useProbeColors()
  const [edited, setEdited] = useState<MatrixEntry[] | null>(null)
  const [parsing, setParsing] = useState(false)
  const [transfer, setTransfer] = useState(false)
  const [pooling, setPooling] = useState<Pooling>(lab.pooling)
  const [testFraction, setTestFraction] = useState(0.3)
  const [chatTemplate, setChatTemplate] = useState(true)
  const [estimate, setEstimate] = useState<ProbeEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<GeneralizeResponse | null>(null)
  const [method, setMethod] = useState<ProbeMethod>('diff_means')
  const [metric, setMetric] = useState<MatrixMetric>('auroc')
  const [position, setPosition] = useState(1)
  const uploads = useRef(0)

  const fromLab = useMemo(() => entriesFromLab(lab), [lab.source, lab.builtin, lab.upload, lab.oodUpload]) // eslint-disable-line react-hooks/exhaustive-deps
  const entries = edited ?? fromLab
  const model = lab.model
  const sibling = model?.sibling ? lab.models.find((m) => m.model_id === model.sibling) ?? null : null
  const useTransfer = transfer && sibling != null
  const anyInstruct = (model?.is_instruct ?? false) || (useTransfer && (sibling?.is_instruct ?? false))
  const modelName = (id: string) => lab.models.find((m) => m.model_id === id)?.display_name ?? id

  useEffect(() => {
    if (!sibling) setTransfer(false)
  }, [sibling])

  useEffect(() => {
    if (!lab.isAuthenticated || !lab.modelId || entries.length < 2) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateGeneralization({
          model: lab.modelId,
          transfer_model: useTransfer ? sibling!.model_id : null,
          datasets: entries.map((e) => ({ num_examples: e.rows, total_chars: e.chars })),
          chat_template: chatTemplate && anyInstruct,
          test_fraction: testFraction,
        })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [lab.isAuthenticated, lab.modelId, entries, useTransfer, sibling, chatTemplate, anyInstruct, testFraction])

  const edit = (next: MatrixEntry[]) => setEdited(next)
  const addBuiltin = (value: string) => {
    const [id, part] = value.split(':') as [string, 'main' | 'ood']
    const d = lab.datasets.find((x) => x.id === id)
    if (d && entries.length < MAX_DATASETS) edit([...entries, builtinEntry(d, part)])
  }
  const addUpload = async (file: File) => {
    setParsing(true)
    setError(null)
    try {
      const parsed = await apiClient.parseProbeDataset(file)
      uploads.current += 1
      edit([...entries, uploadEntry(parsed, `file:${uploads.current}`)])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to read the file')
    } finally {
      setParsing(false)
    }
  }

  const present = new Set(entries.map((e) => e.key))
  const addOptions = lab.datasets.flatMap((d) => [
    { value: `${d.id}:main`, label: `${d.name} · ${d.num_rows} examples`, disabled: present.has(`${d.id}:main`) },
    ...(d.ood
      ? [{ value: `${d.id}:ood`, label: `${d.name} · OOD · ${d.ood.num_rows} examples`, disabled: present.has(`${d.id}:ood`) }]
      : []),
  ])

  const run = async () => {
    if (!lab.isAuthenticated || !lab.modelId) return
    setRunning(true)
    setError(null)
    try {
      const response = await apiClient.generalizeProbes({
        model: lab.modelId,
        transfer_model: useTransfer ? sibling!.model_id : null,
        datasets: entries.map((e) => e.spec),
        pooling,
        chat_template: chatTemplate && anyInstruct,
        read_span: lab.readSpan,
        test_fraction: testFraction,
        seed: lab.seed,
      })
      setResult(response)
      setPosition(defaultPosition(response, method, lab))
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Generalization run failed')
    } finally {
      setRunning(false)
    }
  }

  const positions = result ? result.num_layers + 1 : 0
  const metricName = metric === 'auroc' ? 'AUROC' : 'Accuracy'

  return (
    <section className="pl-panel pl-generalize">
      <div className="pl-card-title">
        <h3>Generalization matrix</h3>
      </div>

      <div className="pl-generalize-setup">
        <div style={{ minWidth: 0 }}>
          <div className="pl-control-label">
            Datasets ({entries.length} of {MAX_DATASETS})
          </div>
          <div className="pl-matrix-datasets">
            {entries.map((e, i) => (
              <div key={e.key} className="pl-matrix-dataset" style={{ borderColor: colors.border, background: colors.panel }}>
                <span className="pl-matrix-dataset-name" title={e.name}>
                  {e.name}
                </span>
                <span className="pl-hint">
                  {e.rows} · {e.labelNames[0]} / {e.labelNames[1]}
                </span>
                <button
                  type="button"
                  className="pl-icon-button"
                  aria-label={`Remove ${e.name}`}
                  onClick={() => edit(entries.filter((_, k) => k !== i))}
                  disabled={running}
                  style={{ color: colors.text }}
                >
                  <X className="pl-icon" />
                </button>
              </div>
            ))}
          </div>
          {entries.length < MAX_DATASETS && (
            <div className="pl-matrix-add">
              <div style={{ flex: '1 1 220px', minWidth: 0 }}>
                <Dropdown value="" onChange={addBuiltin} options={addOptions} placeholder="Add a built-in set" disabled={running} />
              </div>
              <div style={{ flex: '0 1 200px' }}>
                <FileButton label="Add a file" busy={parsing} disabled={running || !lab.isAuthenticated} onFile={addUpload} />
              </div>
            </div>
          )}
          {edited && (
            <button type="button" className="pl-link" onClick={() => setEdited(null)} style={{ color: colors.accent, marginTop: '0.5rem' }}>
              Reset to the training dataset
            </button>
          )}
        </div>

        <div className="pl-generalize-settings">
          {sibling && (
            <label className="pl-check-row">
              <input type="checkbox" checked={transfer} disabled={running} onChange={(e) => setTransfer(e.target.checked)} />
              Include {sibling.display_name}
            </label>
          )}
          {anyInstruct && (
            <label className="pl-check-row">
              <input type="checkbox" checked={chatTemplate} disabled={running} onChange={(e) => setChatTemplate(e.target.checked)} />
              Use chat template (instruct only)
            </label>
          )}
          <div className="pl-setting-row">
            <div>
              <div className="pl-control-label">Pooling</div>
              <Segmented
                ariaLabel="Pooling"
                value={pooling}
                onChange={setPooling}
                disabled={running}
                options={(['last', 'mean', 'max'] as Pooling[]).map((p) => ({ value: p, label: POOLING_LABELS[p] }))}
              />
            </div>
            <div>
              <div className="pl-control-label">Test split</div>
              <Segmented
                ariaLabel="Test split"
                value={String(testFraction)}
                onChange={(v) => setTestFraction(Number(v))}
                disabled={running}
                options={['0.2', '0.3', '0.4', '0.5'].map((v) => ({ value: v, label: `${Number(v) * 100}%` }))}
              />
            </div>
          </div>
          <RunBar
            note={
              !lab.isAuthenticated
                ? 'Sign in to run'
                : entries.length < 2
                  ? 'Add at least two datasets'
                  : estimate
                    ? estimateText(estimate)
                    : null
            }
            warn={estimate != null && !estimate.within_limit}
            label="Run matrix"
            running={running}
            disabled={!lab.isAuthenticated || !lab.modelId || entries.length < 2 || (estimate != null && !estimate.within_limit)}
            onRun={run}
          />
        </div>
      </div>

      {error && <div className="pl-error-note">{error}</div>}

      {result && (
        <div className="pl-result" style={{ borderColor: colors.border, opacity: running ? 0.55 : 1 }}>
          <div className="pl-run-meta">
            <strong>{result.models.map(modelName).join(' + ')}</strong> · {result.datasets.length} datasets ·{' '}
            {POOLING_LABELS[result.pooling].toLowerCase()} pooling{result.chat_template ? ' · chat template' : ''} · $
            {result.cost.toFixed(2)}
          </div>

          <div className="pl-controls-row" style={{ alignItems: 'flex-end' }}>
            <div>
              <div className="pl-control-label">Method</div>
              <Segmented
                ariaLabel="Method"
                value={method}
                onChange={setMethod}
                options={[
                  { value: 'diff_means', label: 'Diff. in means', title: METHOD_LABELS.diff_means },
                  { value: 'logreg', label: 'Logistic regression', title: METHOD_LABELS.logreg },
                ]}
              />
            </div>
            <div>
              <div className="pl-control-label">Metric</div>
              <Segmented
                ariaLabel="Metric"
                value={metric}
                onChange={setMetric}
                options={[
                  { value: 'auroc', label: 'AUROC' },
                  { value: 'acc', label: 'Accuracy', title: 'At each probe’s own threshold: shows when scores shift on new data' },
                ]}
              />
            </div>
            <div>
              <div className="pl-control-label">Layer</div>
              <div className="pl-stepper" style={{ borderColor: colors.border }}>
                <button type="button" className="pl-icon-button" aria-label="Previous layer" disabled={position <= 0} onClick={() => setPosition(position - 1)} style={{ color: colors.text }}>
                  <ChevronLeft className="pl-icon" />
                </button>
                <span>{position === 0 ? 'Embeddings' : `Layer ${position - 1}`}</span>
                <button type="button" className="pl-icon-button" aria-label="Next layer" disabled={position >= positions - 1} onClick={() => setPosition(position + 1)} style={{ color: colors.text }}>
                  <ChevronRight className="pl-icon" />
                </button>
              </div>
            </div>
          </div>

          <Findings items={findings(result, method, position, modelName)} />

          <div className="pl-card-title">
            <h3>
              Test {metricName} · {position === 0 ? 'embeddings' : `layer ${position - 1}`}
            </h3>
          </div>
          <TransferMatrix result={result} method={method} metric={metric} position={position} modelName={modelName} />
          <ScaleLegend />

          <div className="pl-card-title" style={{ marginTop: '1rem' }}>
            <h3>Mean {metricName} by layer</h3>
          </div>
          <TransferByLayer result={result} method={method} metric={metric} position={position} onSelect={setPosition} />
        </div>
      )}

      <style>{`
        .probe-lab .pl-generalize-setup { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 1rem 1.5rem; margin-top: 0.5rem; }
        .probe-lab .pl-generalize-settings { display: flex; flex-direction: column; gap: 0.75rem; }
        .probe-lab .pl-matrix-datasets { display: flex; flex-direction: column; gap: 0.3rem; }
        .probe-lab .pl-matrix-dataset { display: flex; align-items: center; gap: 0.5rem; border: 1px solid; padding: 0.3rem 0.3rem 0.3rem 0.6rem; }
        .probe-lab .pl-matrix-dataset-name { font-size: 0.8rem; font-weight: 600; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .probe-lab .pl-matrix-add { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
        .probe-lab .pl-stepper { display: inline-flex; align-items: center; gap: 0.25rem; border: 1px solid; padding: 0 0.15rem; font-size: 0.75rem; font-weight: 600; min-width: 9rem; justify-content: space-between; }
        .probe-lab .pl-matrix-wrap { position: relative; overflow-x: auto; }
        .probe-lab .pl-matrix { border-collapse: separate; border-spacing: 2px; font-size: 0.75rem; font-variant-numeric: tabular-nums; }
        .probe-lab .pl-matrix th { font-weight: 600; font-size: 0.7rem; padding: 0.25rem 0.4rem; text-align: left; max-width: 9rem; vertical-align: bottom; line-height: 1.25; }
        .probe-lab .pl-matrix thead th { text-align: center; }
        .probe-lab .pl-matrix tbody th { vertical-align: middle; }
        .probe-lab .pl-matrix .pl-matrix-corner { text-align: left; font-weight: 500; white-space: nowrap; vertical-align: bottom; }
        .probe-lab .pl-matrix .pl-matrix-group { border-bottom: 1px solid; }
        .probe-lab .pl-matrix .pl-matrix-group-row { border-right: 1px solid; width: 1.6rem; padding: 0; }
        .probe-lab .pl-matrix .pl-matrix-group-row span { display: inline-block; writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; padding: 0.3rem 0.2rem; }
        .probe-lab .pl-matrix td { min-width: 3.4rem; height: 2.1rem; text-align: center; font-weight: 600; border-radius: 2px; cursor: default; }
        .probe-lab .pl-matrix td sup { font-size: 0.6rem; margin-left: 1px; }
        .probe-lab .pl-scale { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; font-size: 0.7rem; margin-top: 0.5rem; }
        .probe-lab .pl-scale-bar { width: 140px; height: 8px; border: 1px solid; border-radius: 2px; }
        @media (max-width: 900px) {
          .probe-lab .pl-generalize-setup { grid-template-columns: 1fr; }
        }
      `}</style>
    </section>
  )
}
