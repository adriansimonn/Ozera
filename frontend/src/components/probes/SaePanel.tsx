/**
 * A probe next to a sparse autoencoder at its layer: which SAE features the probe reads (and
 * which lie along its direction), linked into the SAE browser, and whether a probe on a few SAE
 * features does as well as the dense probe, in and out of distribution.
 */

import { useEffect, useState, type PointerEvent } from 'react'
import { ExternalLink } from 'lucide-react'
import { apiClient } from '../../api/client'
import type { ProbeLab } from '../../hooks/useProbeLab'
import type { ProbeEstimate, ProbeSaeInfo, SaeFeature, SaeProbeResponse, SaeRef } from '../../types/probes'
import { Dropdown } from '../common/Dropdown'
import type { CausalProbe } from './CausalPanel'
import { ChartTooltip, Findings, RunBar, Segmented, StatTiles, type Finding, type TooltipRow } from './chartKit'
import { useElementWidth, useProbeColors } from './probeTheme'
import { estimateText, formatMetric, formatPercent, formatScore } from './probeUtils'

function refKey(ref: SaeRef): string {
  return ref.kind === 'ozera' ? `ozera:${ref.model}:${ref.layer}` : `external:${ref.sae_id}`
}

/** A link into the SAE browser, with the SAE, a feature, and an example text to analyze filled in. */
function saeBrowserUrl(ref: SaeRef, feature: number, text?: string): string {
  const params = new URLSearchParams()
  if (ref.kind === 'ozera') {
    params.set('model', ref.model)
    params.set('layer', String(ref.layer))
    params.set('type', 'residual')
  } else {
    params.set('external', ref.sae_id)
  }
  params.set('feature', String(feature))
  if (text) params.set('text', text)
  return `/sae?${params.toString()}`
}

function formatWidth(width: number | null): string {
  if (width == null) return '?'
  return width >= 1000 ? `${Math.round(width / 1024)}k` : String(width)
}

function FeatureLink({ sae, feature, text }: { sae: SaeRef; feature: number; text?: string }) {
  const colors = useProbeColors()
  return (
    <a
      href={saeBrowserUrl(sae, feature, text)}
      target="_blank"
      rel="noopener noreferrer"
      className="pl-feature-link"
      style={{ color: colors.accent }}
      title="Open in the SAE browser"
    >
      #{feature}
      <ExternalLink className="pl-icon" aria-hidden="true" style={{ width: 11, height: 11 }} />
    </a>
  )
}

type FeatureOrder = 'contribution' | 'cosine'

function FeatureTable({ result, order, labelNames }: { result: SaeProbeResponse; order: FeatureOrder; labelNames: [string, string] }) {
  const colors = useProbeColors()
  const { contributions } = result.alignment
  const features = order === 'contribution' ? contributions.features : result.alignment.features
  const gap = contributions.gap
  const maxShare = Math.max(...features.map((f) => Math.abs(gap ? f.contribution / gap : 0)), 1e-9)
  const saeRef = result.sae.info.ref
  const exampleText = (f: SaeFeature) => (f.top_example != null ? result.examples[f.top_example]?.text : undefined)

  return (
    <div style={{ maxHeight: 360, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
      <table className="pl-table pl-feature-table" aria-label="SAE features">
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th>Feature</th>
            <th title="(probe weights · decoder row) × class mean difference, over the probe's class gap">Share of gap</th>
            <th title="Cosine between the feature's decoder direction and the probe's direction">Cosine</th>
            <th title="Mean pooled activation over training examples">
              Mean {labelNames[0]} / {labelNames[1]}
            </th>
            <th title="Share of training examples it's active on">Active on</th>
            <th style={{ textAlign: 'left' }}>Strongest on</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => {
            const share = gap ? f.contribution / gap : 0
            const text = exampleText(f)
            const inactive = f.freq_pos === 0 && f.freq_neg === 0
            return (
              <tr key={f.feature} style={{ borderTop: `1px solid ${colors.grid}`, cursor: 'default' }}>
                <td>
                  <FeatureLink sae={saeRef} feature={f.feature} text={text} />
                </td>
                <td>
                  <span className="pl-share">
                    <span className="pl-share-track" style={{ background: colors.grid }}>
                      <span
                        className="pl-share-bar"
                        style={{
                          width: `${(Math.abs(share) / maxShare) * 50}%`,
                          left: share >= 0 ? '50%' : `${50 - (Math.abs(share) / maxShare) * 50}%`,
                          background: colors.methods[result.method],
                        }}
                      />
                    </span>
                    {formatPercent(share)}
                  </span>
                </td>
                <td>
                  <span className="pl-legend-dot" style={{ background: colors.classes[f.cos >= 0 ? 1 : 0], marginRight: 4 }} />
                  {f.cos >= 0 ? '+' : '−'}
                  {Math.abs(f.cos).toFixed(2)}
                </td>
                <td>
                  {formatScore(f.mean_neg)} / {formatScore(f.mean_pos)}
                </td>
                <td>{inactive ? <span style={{ color: colors.warning }}>never</span> : `${formatPercent(f.freq_neg)} / ${formatPercent(f.freq_pos)}`}</td>
                <td className="pl-feature-example" title={text}>
                  {text ?? '–'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

const HEIGHT = 240
const MARGIN = { top: 16, right: 118, bottom: 38, left: 44 }

/** Test (and OOD) AUROC of probes on the k SAE features with the largest class difference, against the dense probe. */
function SparseChart({ result }: { result: SaeProbeResponse }) {
  const colors = useProbeColors()
  const { ref, width } = useElementWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ index: number; px: number; py: number } | null>(null)
  const sparse = result.sparse!
  const ks = sparse.ks
  const hasOod = sparse.metrics.ood_auroc != null && result.dense.ood_auroc != null
  const series = [
    { key: 'sae_test', label: 'SAE features, test', values: sparse.metrics.test_auroc ?? [], color: colors.sae },
    ...(hasOod ? [{ key: 'sae_ood', label: 'SAE features, OOD', values: sparse.metrics.ood_auroc ?? [], color: colors.sae, dash: '5 4' }] : []),
  ]
  const dense = [
    { key: 'dense_test', label: 'dense probe, test', value: result.dense.test_auroc ?? null, color: colors.methods.logreg },
    ...(hasOod ? [{ key: 'dense_ood', label: 'dense probe, OOD', value: result.dense.ood_auroc ?? null, color: colors.methods.logreg, dash: '5 4' }] : []),
  ]

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 60)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const maxLog = Math.max(Math.log2(ks[ks.length - 1]), 1)
  const x = (k: number) => MARGIN.left + (Math.log2(k) / maxLog) * innerW
  const all = [...series.flatMap((s) => s.values), ...dense.map((d) => d.value)].filter((v): v is number => v != null)
  const yMin = Math.min(0.5, Math.floor(Math.min(...all, 1) * 10) / 10)
  const y = (v: number) => MARGIN.top + (1 - (v - yMin) / (1 - yMin)) * innerH
  const tickStep = 1 - yMin > 0.6 ? 0.2 : 0.1
  const ticks: number[] = []
  for (let t = 1; t >= yMin - 1e-9; t -= tickStep) ticks.push(Math.round(t * 100) / 100)

  const ends = [
    ...series.map((s) => ({ key: s.key, label: s.label, color: s.color, dash: s.dash, y: s.values[s.values.length - 1] })),
    ...dense.map((d) => ({ key: d.key, label: d.label, color: d.color, dash: d.dash, y: d.value })),
  ]
    .filter((e): e is typeof e & { y: number } => e.y != null)
    .map((e) => ({ ...e, py: y(e.y) }))
    .sort((a, b) => a.py - b.py)
  for (let i = 1; i < ends.length; i++) ends[i].py = Math.max(ends[i].py, ends[i - 1].py + 12)

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    let index = 0
    ks.forEach((k, i) => {
      if (Math.abs(x(k) - px) < Math.abs(x(ks[index]) - px)) index = i
    })
    setHover({ index, px, py: e.clientY - box.top })
  }
  const tooltipRows = (i: number): TooltipRow[] => [
    ...series.map((s) => ({ key: s.color, label: s.label, value: formatMetric(s.values[i]) })),
    ...dense.map((d) => ({ key: d.color, label: d.label, value: formatMetric(d.value) })),
  ]

  return (
    <div ref={ref} className="pl-chart" role="group" aria-label="AUROC of probes on k SAE features against the dense probe; the table view lists every value">
      <svg width={width} height={HEIGHT} role="img" aria-hidden="true">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
            <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={colors.text} style={{ fontVariantNumeric: 'tabular-nums' }}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={MARGIN.left} x2={MARGIN.left + innerW} y1={MARGIN.top + innerH} y2={MARGIN.top + innerH} stroke={colors.axis} />
        {ks.map((k) => (
          <text key={k} x={x(k)} y={MARGIN.top + innerH + 16} textAnchor="middle" fontSize={11} fill={colors.text}>
            {k}
          </text>
        ))}
        <text x={MARGIN.left + innerW / 2} y={HEIGHT - 4} textAnchor="middle" fontSize={11} fill={colors.text}>
          SAE features used (k)
        </text>
        {dense.map((d) =>
          d.value != null ? (
            <line key={d.key} x1={MARGIN.left} x2={MARGIN.left + innerW} y1={y(d.value)} y2={y(d.value)} stroke={d.color} strokeWidth={2} strokeDasharray={d.dash} />
          ) : null,
        )}
        {series.map((s) => {
          let path = ''
          let pen = false
          s.values.forEach((v, i) => {
            if (v == null) {
              pen = false
              return
            }
            path += `${pen ? 'L' : 'M'}${x(ks[i]).toFixed(1)},${y(v).toFixed(1)}`
            pen = true
          })
          return <path key={s.key} d={path} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round" />
        })}
        {series.map((s) =>
          s.values.map((v, i) =>
            v == null ? null : (
              <circle key={`${s.key}${i}`} cx={x(ks[i])} cy={y(v)} r={4} fill={s.dash ? colors.surface : s.color} stroke={s.dash ? s.color : colors.surface} strokeWidth={2} />
            ),
          ),
        )}
        {ends.map((e) => (
          <text key={e.key} x={MARGIN.left + innerW + 8} y={e.py} dy="0.32em" fontSize={10.5} fill={colors.text}>
            <tspan fill={e.color} fontWeight={700}>
              —{' '}
            </tspan>
            {e.label}
          </text>
        ))}
        {hover && <line x1={x(ks[hover.index])} x2={x(ks[hover.index])} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke={colors.axis} />}
        <rect
          x={MARGIN.left - 12}
          y={MARGIN.top}
          width={innerW + 24}
          height={innerH}
          fill="transparent"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && (
        <ChartTooltip
          x={hover.px}
          y={hover.py}
          containerWidth={width}
          title={`${ks[hover.index]} SAE feature${ks[hover.index] === 1 ? '' : 's'}`}
          rows={tooltipRows(hover.index)}
        />
      )}
    </div>
  )
}

function SparseTable({ result }: { result: SaeProbeResponse }) {
  const colors = useProbeColors()
  const sparse = result.sparse!
  const hasOod = sparse.metrics.ood_auroc != null
  const row = (label: string, m: Partial<Record<string, number | null | undefined>>, key: string) => (
    <tr key={key} style={{ borderTop: `1px solid ${colors.grid}`, cursor: 'default' }}>
      <td>{label}</td>
      <td>{formatMetric(m.test_auroc)}</td>
      {hasOod && <td>{formatMetric(m.ood_auroc)}</td>}
      <td>{formatPercent(m.test_acc)}</td>
      <td>{formatPercent(m.train_acc)}</td>
    </tr>
  )
  return (
    <div style={{ border: `1px solid ${colors.border}` }}>
      <table className="pl-table" aria-label="Sparse and dense probes">
        <thead>
          <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
            <th>Probe</th>
            <th>Test AUROC</th>
            {hasOod && <th>OOD AUROC</th>}
            <th>Test accuracy</th>
            <th>Train accuracy</th>
          </tr>
        </thead>
        <tbody>
          {sparse.ks.map((k, i) =>
            row(
              `${k} SAE feature${k === 1 ? '' : 's'}`,
              Object.fromEntries(Object.entries(sparse.metrics).map(([name, values]) => [name, values?.[i]])),
              String(k),
            ),
          )}
          {row(`Dense (${result.hidden_dim} dimensions)`, result.dense, 'dense')}
        </tbody>
      </table>
    </div>
  )
}

function saeFindings(result: SaeProbeResponse): Finding[] {
  const out: Finding[] = []
  const { fvu, info } = result.sae
  if (fvu != null && fvu > 0.5) {
    out.push({ level: 'warning', text: `Poor fit: ${formatPercent(fvu)} of variance unexplained. Read features with care.` })
  }
  if (info.match === 'sibling') {
    out.push({ level: 'info', text: `Trained on ${info.trained_on}, not ${result.model}.` })
  }
  const { contributions, features, random_max_cos, control_max_cos } = result.alignment
  const top8 = contributions.cumulative.find((c) => c.k === 8)?.share
  if (top8 != null && contributions.gap > 0) {
    out.push({ level: 'info', text: `Top 8 features: ${formatPercent(top8)} of the class gap.` })
  } else if (contributions.gap <= 0) {
    out.push({ level: 'warning', text: 'Negative class gap on this dataset.' })
  }
  const nearest = Math.abs(features[0]?.cos ?? 0)
  const controlMax = Math.max(...control_max_cos, 0)
  if (nearest <= controlMax) {
    out.push({
      level: 'warning',
      text: `Nearest feature by cosine (${nearest.toFixed(2)}) is no nearer than shuffled-label directions' (${Math.min(...control_max_cos).toFixed(2)}–${controlMax.toFixed(2)}; random ${random_max_cos.toFixed(2)}). Rank by contribution.`,
    })
  }

  const sparse = result.sparse
  const denseTest = result.dense.test_auroc
  if (sparse && denseTest != null) {
    const tests = sparse.metrics.test_auroc ?? []
    const close = sparse.ks.find((_, i) => tests[i] != null && (tests[i] as number) >= denseTest - 0.02)
    const best = Math.max(...tests.filter((v): v is number => v != null))
    const maxK = sparse.ks[sparse.ks.length - 1]
    out.push(
      close != null
        ? { level: 'good', text: `${close} SAE feature${close === 1 ? ' matches' : 's match'} the dense probe within 0.02 AUROC (${formatMetric(denseTest)}).` }
        : { level: 'info', text: `${maxK} SAE features reach AUROC ${formatMetric(best)} vs dense ${formatMetric(denseTest)}.` },
    )
    const oods = sparse.metrics.ood_auroc
    const denseOod = result.dense.ood_auroc
    const last = oods?.[oods.length - 1]
    if (last != null && denseOod != null) {
      out.push({
        level: last < denseOod - 0.05 ? 'warning' : 'info',
        text: `OOD: ${maxK} SAE features reach ${formatMetric(last)} vs dense ${formatMetric(denseOod)}.`,
      })
    }
  }
  return out
}

export function SaePanel({ probe, lab, onShowPurchaseCredits }: { probe: CausalProbe; lab: ProbeLab; onShowPurchaseCredits?: () => void }) {
  const colors = useProbeColors()
  const [saes, setSaes] = useState<ProbeSaeInfo[] | null>(null)
  const [choice, setChoice] = useState('')
  const [estimate, setEstimate] = useState<ProbeEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ key: string; response: SaeProbeResponse } | null>(null)
  const [order, setOrder] = useState<FeatureOrder>('contribution')
  const [view, setView] = useState<'chart' | 'table'>('chart')

  useEffect(() => {
    if (!lab.isAuthenticated) {
      setSaes([])
      return
    }
    let cancelled = false
    setSaes(null)
    apiClient
      .listProbeSaes(probe.model)
      .then((list) => !cancelled && setSaes(list))
      .catch(() => !cancelled && setSaes([]))
    return () => {
      cancelled = true
    }
  }, [lab.isAuthenticated, probe.model])

  const atLayer = (saes ?? []).filter((s) => s.layer === probe.layer)
  const usable = atLayer.filter((s) => !s.unusable_reason)
  const sae = usable.find((s) => refKey(s.ref) === choice) ?? usable[0] ?? null
  const otherLayers = [...new Set((saes ?? []).filter((s) => !s.unusable_reason).map((s) => s.layer))].filter((l) => l !== probe.layer)
  const stats = lab.stats
  const instruct = lab.models.find((m) => m.model_id === probe.model)?.is_instruct ?? false
  const chatMismatch = probe.chatTemplate && !instruct

  useEffect(() => {
    if (!lab.isAuthenticated || !stats || !sae || chatMismatch) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateProbeRun({
          model: probe.model,
          num_examples: stats.numRows + (stats.ood?.rows ?? 0),
          total_chars: stats.chars + (stats.ood?.chars ?? 0),
          chat_template: probe.chatTemplate,
          kind: 'sae',
          sae: sae.ref,
        })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [lab.isAuthenticated, stats, sae, probe.model, probe.chatTemplate, chatMismatch])

  const run = async () => {
    const dataset = lab.datasetSpec()
    if (!dataset || !sae) return
    setRunning(true)
    setError(null)
    try {
      const response = await apiClient.saeProbe({ ...probe.ref, sae: sae.ref, dataset, test_fraction: lab.testFraction, seed: lab.seed })
      setResult({ key: `${probe.key}|${refKey(sae.ref)}`, response })
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'SAE comparison failed')
    } finally {
      setRunning(false)
    }
  }

  const response = result?.response ?? null
  const stale = result != null && sae != null && result.key !== `${probe.key}|${refKey(sae.ref)}`
  const options = atLayer.map((s) => ({
    value: refKey(s.ref),
    label: `${s.name} · ${formatWidth(s.width)} features${s.match === 'sibling' ? ` · trained on ${s.trained_on}` : ''}${s.unusable_reason ? ` · ${s.unusable_reason}` : ''}`,
    disabled: !!s.unusable_reason,
  }))
  const isGemma = probe.model.startsWith('gemma-3')

  return (
    <div className="pl-sae">
      <div className="pl-card-title">
        <h3>SAE features</h3>
      </div>

      {!lab.isAuthenticated ? (
        <div className="pl-note">Sign in to compare with SAEs</div>
      ) : saes == null ? (
        <div className="pl-note">Looking for SAEs…</div>
      ) : atLayer.length === 0 ? (
        <div className="pl-empty" style={{ borderColor: colors.border }}>
          No SAE at layer {probe.layer}
          {otherLayers.length > 0 && ` (available at layer${otherLayers.length > 1 ? 's' : ''} ${otherLayers.join(', ')})`}.{' '}
          {isGemma ? `Load Gemma Scope 2 resid_post layer ${probe.layer} on the ` : 'Load one on the '}
          <a href="/sae" target="_blank" rel="noopener noreferrer" style={{ color: colors.accent }}>
            SAE page
          </a>
          .
        </div>
      ) : (
        <>
          <div className="pl-control-label">SAE</div>
          <Dropdown value={sae ? refKey(sae.ref) : ''} onChange={setChoice} options={options} placeholder="Choose an SAE" disabled={running} />
          <div className="pl-note" style={{ marginTop: '0.6rem' }}>
            Dataset: {stats ? `${stats.name} · ${stats.numRows} examples${stats.ood ? ` + ${stats.ood.rows} OOD` : ''}` : 'none chosen'}
          </div>
          {chatMismatch && <div className="pl-error-note">Needs a chat template, which {probe.model} doesn&apos;t have</div>}
          <RunBar
            note={estimate ? estimateText(estimate) : null}
            warn={estimate != null && !estimate.within_limit}
            label="Compare with SAE"
            running={running}
            disabled={!sae || !stats || chatMismatch || (estimate != null && !estimate.within_limit)}
            onRun={run}
          />
        </>
      )}

      {error && <div className="pl-error-note">{error}</div>}

      {response && (
        <div className="pl-result" style={{ borderColor: colors.border, opacity: running ? 0.55 : 1 }}>
          <div className="pl-run-meta">
            {stale && <strong style={{ color: colors.warning }}>From a different probe or SAE · </strong>}
            <strong>{response.sae.info.name}</strong> · {response.dataset.name} · {response.dataset.n_train} train / {response.dataset.n_test} test
            {response.dataset.n_ood > 0 ? ` / ${response.dataset.n_ood} OOD` : ''} · ${response.cost.toFixed(2)}
          </div>

          <StatTiles
            tiles={[
              { label: 'Features', value: response.sae.info.width?.toLocaleString() ?? '–' },
              { label: 'L0', value: response.sae.l0.toFixed(1), hint: 'active per token' },
              { label: 'Unexplained', value: formatPercent(response.sae.fvu), hint: 'variance' },
              {
                label: 'Gap from SAE',
                value: response.alignment.contributions.gap ? formatPercent(response.alignment.contributions.reconstructed / response.alignment.contributions.gap) : '–',
                hint: 'of class gap',
              },
            ]}
          />
          <Findings items={saeFindings(response)} />

          <div className="pl-card-title">
            <h3>Features</h3>
            <Segmented
              ariaLabel="Order features by"
              value={order}
              onChange={setOrder}
              options={[
                { value: 'contribution', label: 'Contribution' },
                { value: 'cosine', label: 'Cosine' },
              ]}
            />
          </div>
          <FeatureTable result={response} order={order} labelNames={response.dataset.label_names} />
          {order === 'cosine' && (
            <div className="pl-note" style={{ marginTop: '0.4rem' }}>
              Nearest-feature baselines: random {response.alignment.random_max_cos.toFixed(2)} · shuffled labels{' '}
              {Math.min(...response.alignment.control_max_cos).toFixed(2)}–{Math.max(...response.alignment.control_max_cos).toFixed(2)}
            </div>
          )}

          {response.sparse ? (
            <>
              <div className="pl-card-title" style={{ marginTop: '1rem' }}>
                <h3>Sparse vs dense probe · AUROC</h3>
                <Segmented
                  ariaLabel="Chart or table"
                  value={view}
                  onChange={setView}
                  options={[
                    { value: 'chart', label: 'Chart' },
                    { value: 'table', label: 'Table' },
                  ]}
                />
              </div>
              {view === 'chart' ? <SparseChart result={response} /> : <SparseTable result={response} />}
              <div className="pl-legend">
                <span className="pl-legend-item">
                  <span className="pl-legend-line" style={{ background: colors.sae }} />
                  SAE features
                </span>
                <span className="pl-legend-item">
                  <span className="pl-legend-line" style={{ background: colors.methods.logreg }} />
                  Dense probe
                </span>
                {response.dataset.n_ood > 0 && (
                  <span className="pl-legend-item">
                    <span className="pl-legend-line" style={{ background: `repeating-linear-gradient(to right, ${colors.text} 0 5px, transparent 5px 9px)` }} />
                    OOD
                  </span>
                )}
              </div>
              <div className="pl-hint" style={{ marginTop: '0.5rem' }}>
                Feature order:{' '}
                {response.sparse.features.slice(0, 8).map((f, i) => (
                  <span key={f.feature}>
                    {i > 0 && ', '}
                    <FeatureLink sae={response.sae.info.ref} feature={f.feature} text={f.top_example != null ? response.examples[f.top_example]?.text : undefined} /> (
                    {f.mean_pos - f.mean_neg >= 0 ? '+' : '−'}
                    {formatScore(Math.abs(f.mean_pos - f.mean_neg))})
                  </span>
                ))}
                {response.sparse.features.length > 8 ? ', …' : ''}
              </div>
            </>
          ) : (
            <div className="pl-note" style={{ marginTop: '0.75rem' }}>
              No feature differs between the classes, so no sparse probes.
            </div>
          )}
        </div>
      )}

      <style>{`
        .probe-lab .pl-feature-link { display: inline-flex; align-items: center; gap: 0.2rem; font-weight: 600; text-decoration: none; font-variant-numeric: tabular-nums; }
        .probe-lab .pl-feature-link:hover { text-decoration: underline; }
        .probe-lab .pl-feature-table td { white-space: nowrap; }
        .probe-lab .pl-feature-table .pl-feature-example { text-align: left; max-width: 22rem; overflow: hidden; text-overflow: ellipsis; font-size: 0.72rem; }
        .probe-lab .pl-share { display: inline-flex; align-items: center; gap: 0.4rem; justify-content: flex-end; }
        .probe-lab .pl-share-track { position: relative; display: inline-block; width: 56px; height: 6px; border-radius: 3px; }
        .probe-lab .pl-share-bar { position: absolute; top: 0; height: 6px; border-radius: 3px; }
      `}</style>
    </div>
  )
}
