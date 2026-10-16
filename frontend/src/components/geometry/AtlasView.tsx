/**
 * The Atlas view: shapes without labels. An atlas job reads a corpus through an SAE (each
 * feature's statistics and contexts, which features are active together, Louvain communities of
 * co-active features, each community's geometry against frequency-matched random groups, and a
 * map of the decoder rows). The view shows the map, the communities ranked by how much of their
 * variance three components hold beyond random groups, a community's tokens in 3D with its
 * features and contexts, and opens a community in the Shape view as a concept.
 */

import { lazy, Suspense, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Download, Map as MapIcon, X } from 'lucide-react'
import { apiClient } from '../../api/client'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { ProbeSaeInfo } from '../../types/probes'
import type {
  AtlasCommunity,
  AtlasCommunitySummary,
  AtlasContext,
  AtlasCorpus,
  AtlasCorpusId,
  AtlasEstimate,
  AtlasFeature,
  AtlasJob,
  AtlasRef,
  AtlasStatus,
  AtlasSummary,
  AtlasTokens,
  GeometryEstimate,
} from '../../types/geometry'
import { Dropdown } from '../common/Dropdown'
import { ChartTooltip, Findings, Placeholder, RunBar, Segmented, StatTiles, type Finding, type Tile } from '../lab/chartKit'
import { useElementWidth, useLabColors, type LabColors } from '../lab/labTheme'
import { FeatureLink } from '../lab/FeatureLink'
import { formatWidth, refKey } from '../lab/saeLinks'
import { STRUCTURE_LABELS, downloadSvg, estimateText, formatPercent, formatValue } from './geometryUtils'

const ShapeScene = lazy(() => import('./ShapeScene'))

const TOKEN_CHOICES: AtlasTokens[] = [1_000_000, 2_000_000, 5_000_000]
const JOB_POLL_MS = 8000
// Communities colored on the map (the best ranked), and a community whose excess over random
// groups at 3 components reads as low-dimensional
const MAP_COLORED = 8
const NOTABLE_EXCESS = 0.2
const CONTEXTS_SHOWN = 12

function formatTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 1)}M`
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`
  return String(n)
}

function formatDate(iso: string): string {
  const date = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function share(values: number[] | null | undefined, n: number): number | null {
  return values ? values.slice(0, n).reduce((a, b) => a + b, 0) : null
}

/** A token as text, with line breaks and bare spaces visible. */
function visible(token: string): string {
  if (token.includes('\n')) return token.replace(/\n/g, '⏎').trim() || '⏎'
  return token.trim() || '␣'
}

function words(c: { top_words: [string, number][] }, n = 4): string {
  const shown = c.top_words.slice(0, n).map(([w]) => visible(w))
  return shown.length ? shown.join(', ') : '–'
}

function saeLabel(s: ProbeSaeInfo): string {
  const parts = [`Layer ${s.layer}`, s.name, `${formatWidth(s.width)} features`]
  if (s.match === 'sibling') parts.push(`trained on ${s.trained_on}`)
  if (s.unusable_reason) parts.push(s.unusable_reason)
  return parts.join(' · ')
}

function contextText(c: AtlasContext): string {
  return c.tokens.join('').replace(/\s+/g, ' ').trim()
}

/** A color with an opacity, for highlighting tokens (colors are #rrggbb). */
function tint(color: string, alpha: number): string {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255)
  return `${color}${a.toString(16).padStart(2, '0')}`
}

export function AtlasView({ lab }: { lab: GeometryLab }) {
  if (!lab.isAuthenticated) {
    return (
      <section className="pl-panel">
        <Placeholder icon={MapIcon}>Sign in to read atlases.</Placeholder>
      </section>
    )
  }
  if (!lab.modelId) {
    return (
      <section className="pl-panel">
        <Placeholder icon={MapIcon}>Choose a model.</Placeholder>
      </section>
    )
  }
  return <AtlasBody key={lab.modelId} lab={lab} model={lab.modelId} />
}

function AtlasBody({ lab, model }: { lab: GeometryLab; model: string }) {
  const colors = useLabColors()
  const [saes, setSaes] = useState<ProbeSaeInfo[] | null>(null)
  const [corpora, setCorpora] = useState<AtlasCorpus[]>([])
  const [choice, setChoice] = useState('')
  const [corpus, setCorpus] = useState<AtlasCorpusId>('openwebtext')
  const [tokens, setTokens] = useState<AtlasTokens>(2_000_000)
  const [status, setStatus] = useState<AtlasStatus | null>(null)
  const [estimate, setEstimate] = useState<AtlasEstimate | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<{ key: string; summary: AtlasSummary } | null>(null)
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [statusKey, setStatusKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    Promise.all([apiClient.listProbeSaes(model), apiClient.listAtlasCorpora()])
      .then(([list, corpusList]) => {
        if (cancelled) return
        setSaes(list)
        setCorpora(corpusList)
      })
      .catch(() => !cancelled && setSaes([]))
    return () => {
      cancelled = true
    }
  }, [model])

  const usable = (saes ?? []).filter((s) => !s.unusable_reason)
  const sae = usable.find((s) => refKey(s.ref) === choice) ?? usable[0] ?? null
  const saeKey = sae ? refKey(sae.ref) : ''

  // The SAE's atlases and jobs, polled while a job is running
  useEffect(() => {
    if (!sae) return
    let cancelled = false
    apiClient
      .atlasStatus({ model, sae: sae.ref })
      .then((s) => !cancelled && setStatus(s))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the atlas status'))
    return () => {
      cancelled = true
    }
    // saeKey stands for sae
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, saeKey, statusKey])

  const atlas = status?.atlases.find((a) => a.corpus.id === corpus) ?? null
  const job = status?.jobs.find((j) => j.corpus === corpus) ?? null
  const active = job != null && (job.status === 'queued' || job.status === 'running') ? job : null

  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => {
      apiClient
        .getAtlasJob(active.job_id)
        .then((updated) => {
          if (updated.status === 'completed' || updated.status === 'failed') setStatusKey((k) => k + 1)
          else setStatus((s) => (s ? { ...s, jobs: s.jobs.map((j) => (j.job_id === updated.job_id ? updated : j)) } : s))
        })
        .catch(() => undefined)
    }, JOB_POLL_MS)
    return () => clearInterval(timer)
  }, [active])

  // The stored atlas for the chosen corpus
  const atlasKey = atlas ? `${saeKey}|${corpus}|${atlas.created_at}` : ''
  useEffect(() => {
    if (!atlas || !sae) return
    let cancelled = false
    setLoadingSummary(true)
    apiClient
      .atlasSummary({ model, sae: sae.ref, corpus })
      .then((s) => !cancelled && setLoaded({ key: atlasKey, summary: s }))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the atlas'))
      .finally(() => !cancelled && setLoadingSummary(false))
    return () => {
      cancelled = true
    }
    // atlasKey stands for atlas, sae and corpus
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atlasKey])

  // Cost of computing (or recomputing) it
  const requestKey = sae ? `${saeKey}|${corpus}|${tokens}` : ''
  useEffect(() => {
    if (!sae || active) {
      setEstimate(null)
      return
    }
    let cancelled = false
    apiClient
      .estimateAtlas({ model, sae: sae.ref, corpus, tokens, seed: 0 })
      .then((e) => !cancelled && setEstimate(e))
      .catch(() => !cancelled && setEstimate(null))
    return () => {
      cancelled = true
    }
    // requestKey stands for sae, corpus and tokens
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey, active?.job_id])

  const compute = async () => {
    if (!sae) return
    setSubmitting(true)
    setError(null)
    try {
      const started = await apiClient.submitAtlasJob({ model, sae: sae.ref, corpus, tokens, seed: 0 })
      setStatus((s) => (s ? { ...s, jobs: [started, ...s.jobs.filter((j) => j.job_id !== started.job_id)] } : s))
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') lab.showPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Failed to start the atlas job')
    } finally {
      setSubmitting(false)
    }
  }

  const corpusInfo = corpora.find((c) => c.id === corpus)
  const note = active
    ? null
    : estimate
      ? !estimate.corpus_available
        ? `${corpusInfo?.name ?? corpus} isn't available`
        : estimate.within_limit
          ? `≈ $${estimate.estimated_cost.toFixed(2)} · ~${Math.ceil(estimate.estimated_seconds / 60)} min GPU`
          : `Too large: ~${Math.ceil(estimate.estimated_seconds / 60)} min GPU (limit ${Math.round(estimate.max_seconds / 60)})`
      : null
  const failed = job?.status === 'failed' && job.mine ? job : null
  const summary = loaded && loaded.key === atlasKey ? loaded.summary : null

  return (
    <div className="gl-atlas">
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{lab.modelName(model)}</strong> · features over a corpus, no labels
        </div>
        {saes == null ? (
          <div className="pl-note">Looking for SAEs…</div>
        ) : usable.length === 0 ? (
          <div className="pl-empty" style={{ borderColor: colors.border }}>
            No SAE reads {lab.modelName(model)}&apos;s residual stream. Load one on the{' '}
            <a href="/sae" target="_blank" rel="noopener noreferrer" style={{ color: colors.accent }}>
              SAE page
            </a>
            .
          </div>
        ) : (
          <>
            <div className="gl-atlas-head">
              <div className="gl-atlas-sae">
                <div className="pl-control-label" title="SAEs that read this model's residual stream after a layer">
                  SAE
                </div>
                <Dropdown
                  value={saeKey}
                  onChange={setChoice}
                  options={(saes ?? []).map((s) => ({ value: refKey(s.ref), label: saeLabel(s), disabled: !!s.unusable_reason }))}
                  disabled={submitting}
                />
              </div>
              <div>
                <div className="pl-control-label" title="Text the model reads; its license allows showing it">
                  Corpus
                </div>
                <Segmented
                  ariaLabel="Corpus"
                  value={corpus}
                  onChange={setCorpus}
                  options={(corpora.length ? corpora : [{ id: 'openwebtext', name: 'OpenWebText', license: '', description: '' }]).map((c) => ({
                    value: c.id as AtlasCorpusId,
                    label: `${c.name}${status?.atlases.some((a) => a.corpus.id === c.id) ? ' ●' : ''}`,
                    title: [c.description, c.license].filter(Boolean).join(' · '),
                  }))}
                />
              </div>
              <div>
                <div className="pl-control-label" title="Corpus tokens to read (an SAE with many active features per token reads fewer)">
                  Tokens
                </div>
                <Segmented
                  ariaLabel="Tokens"
                  value={String(tokens)}
                  onChange={(v) => setTokens(Number(v) as AtlasTokens)}
                  options={TOKEN_CHOICES.map((t) => ({ value: String(t), label: formatTokens(t) }))}
                />
              </div>
            </div>

            <div className="pl-hint gl-atlas-state">
              {atlas ? (
                <>
                  {formatTokens(atlas.tokens)} tokens of {atlas.corpus.name}
                  {atlas.corpus.license ? ` (${atlas.corpus.license})` : ''} · computed {formatDate(atlas.created_at)}
                  {atlas.stopped_early ? ` · stopped at ${formatTokens(atlas.tokens)} of ${formatTokens(atlas.requested_tokens)}: the latents filled the budget` : ''}
                </>
              ) : active ? null : (
                'Not computed for this corpus yet.'
              )}
            </div>
            {active && <JobProgress job={active} colors={colors} />}
            {failed && !active && (
              <div className="pl-error-note">
                The last job failed: {failed.error ?? 'unknown error'}
                {failed.refunded != null ? ` · refunded $${failed.refunded.toFixed(2)}` : ''}
              </div>
            )}
            {!active && (
              <RunBar
                note={note}
                warn={estimate != null && (!estimate.within_limit || !estimate.corpus_available)}
                label={atlas ? 'Recompute' : 'Compute atlas'}
                running={submitting}
                disabled={!sae || !estimate || !estimate.within_limit || !estimate.corpus_available}
                onRun={compute}
              />
            )}
          </>
        )}
        {error && <div className="pl-error-note">{error}</div>}
      </section>

      {loadingSummary && !summary && (
        <section className="pl-panel">
          <div className="pl-note">Loading the atlas…</div>
        </section>
      )}
      {summary && <AtlasResults key={atlasKey} lab={lab} summary={summary} />}

      <style>{`
        .geometry-lab .gl-atlas-head { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: flex-end; margin-bottom: 0.75rem; }
        .geometry-lab .gl-atlas-sae { min-width: 260px; max-width: 520px; flex: 1; }
        .geometry-lab .gl-atlas-state { min-height: 1.1rem; }
        .geometry-lab .gl-progress { height: 6px; border: 1px solid; margin-top: 0.4rem; }
        .geometry-lab .gl-progress > div { height: 100%; transition: width 0.4s; }
        .geometry-lab .gl-context { font-size: 0.75rem; line-height: 1.7; white-space: pre-wrap; word-break: break-word; padding: 0.3rem 0; }
        .geometry-lab .gl-context span { border-radius: 2px; }
        .geometry-lab .gl-context-meta { font-size: 0.68rem; opacity: 0.7; margin-right: 0.5rem; font-variant-numeric: tabular-nums; }
        .geometry-lab .gl-chips { display: flex; flex-wrap: wrap; gap: 0.25rem; }
        .geometry-lab .gl-chip { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.1rem 0.4rem; border: 1px solid; font-size: 0.7rem; white-space: nowrap; background: none; color: inherit; }
        .geometry-lab button.gl-chip { cursor: pointer; }
        .geometry-lab .gl-atlas-table td { white-space: nowrap; }
        .geometry-lab .gl-atlas-table .gl-words { text-align: left; max-width: 16rem; overflow: hidden; text-overflow: ellipsis; }
        .geometry-lab .gl-detail-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem; }
      `}</style>
    </div>
  )
}

function JobProgress({ job, colors }: { job: AtlasJob; colors: LabColors }) {
  const fraction = job.progress?.fraction ?? 0
  const minutes = Math.max(1, Math.ceil(job.estimated_seconds / 60))
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div className="pl-hint">
        {job.status === 'queued' ? 'Starting' : job.progress?.stage ?? 'Running'}
        {job.progress ? ` · ${Math.round(fraction * 100)}%` : ''} · {formatTokens(job.tokens)} tokens · about {minutes} min
        {job.mine ? '' : ' · started by another user'}
      </div>
      <div className="gl-progress" style={{ borderColor: colors.border }}>
        <div style={{ width: `${Math.max(2, fraction * 100)}%`, background: colors.accent }} />
      </div>
    </div>
  )
}

// Results

function AtlasResults({ lab, summary }: { lab: GeometryLab; summary: AtlasSummary }) {
  const atlasRef = useMemo((): AtlasRef => ({ model: summary.model, sae: summary.sae, corpus: summary.corpus.id as AtlasCorpusId }), [summary])
  const [selected, setSelected] = useState<number | null>(summary.ranked[0] ?? null)
  const [feature, setFeature] = useState<number | null>(null)
  const byId = useMemo(() => new Map(summary.communities.map((c) => [c.id, c])), [summary])
  const ranked = summary.ranked.map((id) => byId.get(id)).filter((c): c is AtlasCommunitySummary => c != null)
  const notable = ranked.filter((c) => (c.geometry?.score ?? 0) >= NOTABLE_EXCESS)

  const tiles: Tile[] = [
    { label: 'Live', value: `${formatTokens(summary.alive)}`, hint: `of ${formatWidth(summary.width)} features`, title: 'Features active on some corpus token' },
    { label: 'L0', value: formatValue(summary.l0, 1), hint: 'active per token' },
    { label: 'Unexplained', value: formatPercent(summary.fvu), hint: 'variance', title: "Share of the corpus activations' variance the SAE's reconstructions miss" },
    {
      label: 'Communities',
      value: String(summary.communities.length),
      hint: `${summary.ranked.length} measured`,
      title: `Louvain groups of at least 3 co-active features; groups of up to ${summary.geometry.max_features} with enough stored tokens get their geometry measured`,
    },
    {
      label: 'Graph',
      value: formatTokens(summary.graph.nodes),
      hint: `${formatTokens(summary.graph.edges)} edges`,
      title: `Features active at least ${summary.graph.min_firings} times, each joined to up to ${summary.graph.neighbors} others active on the same tokens (at least ${summary.graph.min_pair_count} times and ${summary.graph.min_lift}× chance), weighted by sqrt(P(j|i) P(i|j))`,
    },
    { label: 'Tokens', value: formatTokens(summary.tokens), hint: summary.corpus.name },
  ]

  const findings: Finding[] = []
  if (summary.fvu != null && summary.fvu > 0.5) {
    findings.push({ level: 'warning', text: `Poor fit: ${formatPercent(summary.fvu)} of the corpus activations' variance unexplained.` })
  }
  if (summary.graph.tokens != null && summary.graph.tokens < summary.tokens) {
    findings.push({
      level: 'warning',
      text: `Dense SAE (L0 ${formatValue(summary.l0, 0)}): co-activation counted on ${formatTokens(summary.graph.tokens)} of ${formatTokens(summary.tokens)} tokens.`,
    })
  }
  if (notable.length > 0) {
    const best = notable[0]
    findings.push({
      level: 'good',
      text: `${notable.length} communit${notable.length === 1 ? 'y holds' : 'ies hold'} ≥${Math.round(NOTABLE_EXCESS * 100)} points more variance in 3 PCs than random groups. Best: ${words(best, 3)} (${formatPercent(share(best.geometry?.explained, 3))} vs ${formatPercent(share(best.geometry?.baseline, 3))}).`,
    })
  } else if (ranked.length > 0) {
    findings.push({ level: 'info', text: 'No community is much lower-dimensional than random groups of its size.' })
  }
  if (summary.map.method === 'pca') findings.push({ level: 'info', text: 'Map: PCA of decoder rows (UMAP unavailable).' })

  const pick = (id: number | null) => {
    setSelected(id)
    setFeature(null)
  }

  return (
    <>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{summary.sae_name}</strong> · layer {summary.layer} · {formatTokens(summary.tokens)} {summary.corpus.name} tokens
        </div>
        <StatTiles tiles={tiles} />
        <Findings items={findings} />
      </section>

      <div className="pl-results-grid">
        <section className="pl-panel">
          <AtlasMap summary={summary} byId={byId} selected={selected} onSelect={pick} onFeature={setFeature} />
        </section>
        <section className="pl-panel">
          <CommunityTable summary={summary} ranked={ranked} selected={selected} onSelect={pick} />
        </section>
      </div>

      {selected != null && byId.get(selected) && (
        <CommunityDetail
          key={selected}
          lab={lab}
          atlasRef={atlasRef}
          summary={summary}
          community={byId.get(selected) as AtlasCommunitySummary}
          onClose={() => pick(null)}
          onFeature={setFeature}
        />
      )}
      {feature != null && (
        <FeatureDetail key={feature} atlasRef={atlasRef} feature={feature} byId={byId} onFeature={setFeature} onCommunity={pick} onClose={() => setFeature(null)} />
      )}
    </>
  )
}

// The map

const MAP_MARGIN = 10

function communityColors(summary: AtlasSummary, colors: LabColors): Map<number, string> {
  return new Map(summary.ranked.slice(0, MAP_COLORED).map((id, i) => [id, colors.series[i % colors.series.length]]))
}

function AtlasMap({
  summary,
  byId,
  selected,
  onSelect,
  onFeature,
}: {
  summary: AtlasSummary
  byId: Map<number, AtlasCommunitySummary>
  selected: number | null
  onSelect: (id: number | null) => void
  onFeature: (f: number) => void
}) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(480)
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<{ i: number; px: number; py: number } | null>(null)
  const height = Math.min(Math.max(width * 0.85, 280), 520)
  const { features, xy } = summary.map
  const communityOf = useMemo(() => {
    const out = new Map<number, number>()
    summary.communities.forEach((c) => c.features.forEach((f) => out.set(f, c.id)))
    return out
  }, [summary])
  const palette = useMemo(() => communityColors(summary, colors), [summary, colors])
  const sx = (v: number) => MAP_MARGIN + ((v + 1) / 2) * (width - 2 * MAP_MARGIN)
  const sy = (v: number) => MAP_MARGIN + ((1 - v) / 2) * (height - 2 * MAP_MARGIN)

  // One path of round dots per color: thousands of points stay one SVG element each
  const layers = useMemo(() => {
    const groups = new Map<string, { color: string; width: number; opacity: number; order: number; d: string[] }>()
    features.forEach((f, i) => {
      const c = communityOf.get(f)
      const isSelected = c != null && c === selected
      const color = isSelected ? colors.text : c != null && palette.has(c) ? (palette.get(c) as string) : c != null ? colors.baseline : colors.axis
      const key = `${color}|${isSelected}`
      if (!groups.has(key)) {
        groups.set(key, {
          color,
          width: isSelected ? 5 : c != null && palette.has(c) ? 3.5 : 2.5,
          opacity: isSelected ? 1 : c != null && palette.has(c) ? 0.85 : c != null ? 0.45 : 0.3,
          order: isSelected ? 3 : c != null && palette.has(c) ? 2 : c != null ? 1 : 0,
          d: [],
        })
      }
      const [x, y] = xy[i]
      ;(groups.get(key) as { d: string[] }).d.push(`M${sx(x).toFixed(1)},${sy(y).toFixed(1)}h0`)
    })
    return [...groups.values()].sort((a, b) => a.order - b.order)
    // sx, sy depend on width and height
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [features, xy, communityOf, palette, selected, colors, width, height])

  // Nearest point under the pointer, from a grid of cells
  const grid = useMemo(() => {
    const cell = 12
    const cells = new Map<string, number[]>()
    xy.forEach(([x, y], i) => {
      const key = `${Math.floor(sx(x) / cell)},${Math.floor(sy(y) / cell)}`
      if (!cells.has(key)) cells.set(key, [])
      ;(cells.get(key) as number[]).push(i)
    })
    return { cell, cells }
    // sx, sy depend on width and height
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xy, width, height])

  const nearest = (px: number, py: number): number | null => {
    const cx = Math.floor(px / grid.cell)
    const cy = Math.floor(py / grid.cell)
    let best: number | null = null
    let bestD = 8 * 8
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (const i of grid.cells.get(`${cx + dx},${cy + dy}`) ?? []) {
          const d = (sx(xy[i][0]) - px) ** 2 + (sy(xy[i][1]) - py) ** 2
          if (d < bestD) {
            bestD = d
            best = i
          }
        }
    return best
  }

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - box.left
    const py = e.clientY - box.top
    const i = nearest(px, py)
    setHover(i == null ? null : { i, px, py })
  }
  const onClick = () => {
    if (!hover) return
    const f = features[hover.i]
    const c = communityOf.get(f)
    if (c != null) onSelect(c)
    onFeature(f)
  }

  const hovered = hover ? features[hover.i] : null
  const hoveredCommunity = hovered != null ? byId.get(communityOf.get(hovered) ?? -1) : undefined
  const legend = summary.ranked.slice(0, MAP_COLORED).map((id) => byId.get(id)).filter((c): c is AtlasCommunitySummary => c != null)

  return (
    <div>
      <div className="pl-card-title">
        <h3
          title={`Decoder rows in 2D (${summary.map.method === 'umap' ? 'UMAP, cosine distance, after PCA to 50 dimensions' : 'PCA'}); the ${MAP_COLORED} best-ranked communities in color, other communities grey, features in none faint`}
        >
          Map · decoder rows
        </h3>
        <button
          type="button"
          className="pl-icon-button"
          title="Download SVG"
          aria-label="Download SVG"
          onClick={() => svgRef.current && downloadSvg(svgRef.current, `atlas-map-${summary.model}-L${summary.layer}.svg`, colors.surface)}
          style={{ color: colors.text }}
        >
          <Download className="pl-icon" />
        </button>
      </div>
      <div ref={ref} className="pl-chart">
        <svg ref={svgRef} width={width} height={height} role="img" aria-label="Map of the SAE's decoder rows, colored by community">
          <rect x={0.5} y={0.5} width={width - 1} height={height - 1} fill="none" stroke={colors.grid} />
          {layers.map((layer, i) => (
            <path key={i} d={layer.d.join('')} stroke={layer.color} strokeWidth={layer.width} strokeOpacity={layer.opacity} strokeLinecap="round" fill="none" />
          ))}
          {hover && <circle cx={sx(xy[hover.i][0])} cy={sy(xy[hover.i][1])} r={5} fill="none" stroke={colors.text} strokeWidth={1.5} />}
          <rect
            x={0}
            y={0}
            width={width}
            height={height}
            fill="transparent"
            style={{ cursor: hover ? 'pointer' : 'default' }}
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
            onClick={onClick}
          />
        </svg>
        {hover && hovered != null && (
          <ChartTooltip
            x={hover.px}
            y={hover.py}
            containerWidth={width}
            title={`#${hovered}`}
            rows={
              hoveredCommunity
                ? [
                    { label: 'community', value: `#${hoveredCommunity.id}` },
                    { label: 'fires on', value: words(hoveredCommunity, 3) },
                  ]
                : [{ label: '', value: 'in no community' }]
            }
          />
        )}
      </div>
      {legend.length > 0 && (
        <div className="pl-legend">
          {legend.map((c) => (
            <button key={c.id} type="button" className="pl-legend-item pl-link" onClick={() => onSelect(c.id)} style={{ color: colors.text, fontSize: '0.72rem' }}>
              <span className="pl-legend-dot" style={{ background: palette.get(c.id) }} />
              {words(c, 2)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// Communities

function CommunityTable({
  summary,
  ranked,
  selected,
  onSelect,
}: {
  summary: AtlasSummary
  ranked: AtlasCommunitySummary[]
  selected: number | null
  onSelect: (id: number) => void
}) {
  const colors = useLabColors()
  const [scope, setScope] = useState<'ranked' | 'all'>('ranked')
  const palette = useMemo(() => communityColors(summary, colors), [summary, colors])
  const rows = scope === 'ranked' ? ranked : summary.communities
  return (
    <div>
      <div className="pl-card-title">
        <h3 title="Communities by how much more of their variance 3 principal components hold than for frequency-matched random groups of as many features">
          Communities
        </h3>
        <Segmented
          ariaLabel="Communities shown"
          value={scope}
          onChange={setScope}
          options={[
            { value: 'ranked', label: `Ranked · ${ranked.length}`, title: 'Those whose geometry was measured, best first' },
            { value: 'all', label: `All · ${summary.communities.length}`, title: 'Every community, largest first' },
          ]}
        />
      </div>
      <div style={{ maxHeight: 460, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
        <table className="pl-table gl-atlas-table" aria-label="Communities">
          <thead>
            <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
              <th>#</th>
              <th style={{ textAlign: 'left' }} title="The tokens its features fire on most">
                Fires on
              </th>
              <th title="Features">Size</th>
              <th title="Share of its variance on the top 3 principal components (activations projected onto the span of its decoder rows)">3 PCs</th>
              <th title="The same for frequency-matched random groups of as many features">Random</th>
              <th title="3 PCs minus random">Excess</th>
              <th title="Share of corpus tokens any of its features is active on">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const active = c.id === selected
              return (
                <tr
                  key={c.id}
                  onClick={() => onSelect(c.id)}
                  style={{ borderTop: `1px solid ${colors.grid}`, background: active ? colors.hover : undefined, fontWeight: active ? 600 : undefined }}
                >
                  <td>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span className="pl-legend-dot" style={{ background: palette.get(c.id) ?? 'transparent', border: palette.has(c.id) ? 'none' : `1px solid ${colors.axis}` }} />
                      {c.rank != null ? c.rank + 1 : '–'}
                    </span>
                  </td>
                  <td className="gl-words" title={c.top_words.map(([w, n]) => `${visible(w)} (${n})`).join(', ')}>
                    {words(c)}
                  </td>
                  <td>{c.size}</td>
                  <td>{formatPercent(share(c.geometry?.explained, 3))}</td>
                  <td>{formatPercent(share(c.geometry?.baseline, 3))}</td>
                  <td>{c.geometry?.score != null ? `${c.geometry.score >= 0 ? '+' : ''}${Math.round(c.geometry.score * 100)}` : '–'}</td>
                  <td>{formatPercent(c.tokens / summary.tokens)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function CommunityDetail({
  lab,
  atlasRef,
  summary,
  community,
  onClose,
  onFeature,
}: {
  lab: GeometryLab
  atlasRef: AtlasRef
  summary: AtlasSummary
  community: AtlasCommunitySummary
  onClose: () => void
  onFeature: (f: number) => void
}) {
  const colors = useLabColors()
  const [detail, setDetail] = useState<AtlasCommunity | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [estimate, setEstimate] = useState<GeometryEstimate | null>(null)

  useEffect(() => {
    let cancelled = false
    apiClient
      .atlasCommunity(atlasRef, community.id)
      .then((d) => !cancelled && setDetail(d))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the community'))
    return () => {
      cancelled = true
    }
  }, [atlasRef, community.id])

  const concept = detail?.concept ?? null
  useEffect(() => {
    if (!concept) return
    let cancelled = false
    apiClient
      .estimateGeometryRun({ model: atlasRef.model, concept: { custom: concept }, chat_template: false })
      .then((e) => !cancelled && setEstimate(e))
      .catch(() => !cancelled && setEstimate(null))
    return () => {
      cancelled = true
    }
  }, [concept, atlasRef.model])

  const g = community.geometry
  const tiles: Tile[] = [
    { label: 'Features', value: String(community.size) },
    {
      label: '2 PCs',
      value: formatPercent(share(g?.explained, 2)),
      hint: `random ${formatPercent(share(g?.baseline, 2))}`,
      title: 'Share of its variance on the top 2 principal components, against frequency-matched random groups',
    },
    { label: '3 PCs', value: formatPercent(share(g?.explained, 3)), hint: `random ${formatPercent(share(g?.baseline, 3))}` },
    { label: 'Tokens', value: formatPercent(community.tokens / summary.tokens), hint: `${formatTokens(community.tokens)} active on` },
    { label: 'Measured on', value: g ? formatTokens(g.tokens) : '–', hint: 'stored tokens', title: 'Stored corpus activations where any of its features fires' },
  ]
  const findings: Finding[] = []
  if (!g) {
    findings.push({
      level: 'info',
      text: community.size > summary.geometry.max_features ? `Too large to measure (over ${summary.geometry.max_features} features).` : 'Too few stored tokens to measure its geometry.',
    })
  } else if (community.size <= 3) {
    findings.push({ level: 'info', text: '3 features span at most 3 dimensions: compare 2 PCs.' })
  } else if ((g.score ?? 0) >= NOTABLE_EXCESS) {
    findings.push({ level: 'good', text: `Low-dimensional: 3 PCs hold ${formatPercent(share(g.explained, 3))}, random groups ${formatPercent(share(g.baseline, 3))}.` })
  }

  return (
    <section className="pl-panel">
      <div className="gl-detail-head">
        <div className="pl-run-meta">
          <strong>Community #{community.id}</strong>
          {community.rank != null ? ` · rank ${community.rank + 1}` : ''} · fires on {words(community, 6)}
        </div>
        <button type="button" className="pl-icon-button" aria-label="Close" title="Close" onClick={onClose} style={{ color: colors.text }}>
          <X className="pl-icon" />
        </button>
      </div>
      <StatTiles tiles={tiles} />
      <Findings items={findings} />
      {error && <div className="pl-error-note">{error}</div>}
      {detail && (
        <>
          {detail.geometry && (
            <div className="pl-results-grid" style={{ marginBottom: '1rem' }}>
              <div>
                <div className="pl-card-title">
                  <h3 title="Tokens where any of its features fires, their activations projected onto the span of its decoder rows, on the top three principal components; colored by its most active feature there">
                    Tokens · PC1–3
                  </h3>
                </div>
                <CommunityScene detail={detail} colors={colors} />
              </div>
              <div>
                <ExplainedBars geometry={detail.geometry} filename={`atlas-community-${community.id}-explained`} />
                <PcPlane detail={detail} filename={`atlas-community-${community.id}-pc12`} />
              </div>
            </div>
          )}

          <div className="pl-results-grid" style={{ marginBottom: 0 }}>
            <div>
              <div className="pl-card-title">
                <h3>Features</h3>
              </div>
              <div style={{ maxHeight: 320, overflowY: 'auto', border: `1px solid ${colors.border}` }}>
                <table className="pl-table gl-atlas-table" aria-label="The community's features">
                  <thead>
                    <tr style={{ background: colors.isLight ? '#f5f5f7' : '#151515' }}>
                      <th>Feature</th>
                      <th style={{ textAlign: 'left' }} title="The token it's most active on">
                        Top
                      </th>
                      <th title="Share of corpus tokens it's active on">Freq</th>
                      <th title="Mean activation where active">Mean</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.features.map((f, i) => (
                      <tr key={f.id} onClick={() => onFeature(f.id)} style={{ borderTop: `1px solid ${colors.grid}` }}>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                            <span className="pl-legend-dot" style={{ background: colors.series[i % colors.series.length] }} />
                            <FeatureLink sae={atlasRef.sae} feature={f.id} text={detail.examples.find((e) => e.feature === f.id) ? contextText(detail.examples.find((e) => e.feature === f.id) as AtlasContext) : undefined} />
                          </span>
                        </td>
                        <td className="gl-words">{f.top != null ? visible(f.top) : '–'}</td>
                        <td>{formatPercent(f.tokens / summary.tokens)}</td>
                        <td>{formatValue(f.mean, 2)}</td>
                        <td>{formatValue(f.max, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <ConceptPanel lab={lab} detail={detail} model={atlasRef.model} position={summary.layer + 1} estimate={estimate} />
          </div>

          <div style={{ marginTop: '1rem' }}>
            <div className="pl-card-title">
              <h3 title="Its features' strongest contexts (each feature's own activation shaded)">Contexts</h3>
            </div>
            {detail.examples.slice(0, CONTEXTS_SHOWN).map((c, i) => (
              <ContextLine key={i} context={c} colors={colors} label={`#${c.feature}`} onLabel={() => onFeature(c.feature)} />
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function CommunityScene({ detail, colors }: { detail: AtlasCommunity; colors: LabColors }) {
  const geometry = detail.geometry as NonNullable<AtlasCommunity['geometry']>
  // Each feature's centroid, labeled by the token it's most active on
  const { means, labels, meanColors } = useMemo(() => {
    const sums = new Map<number, [number, number, number, number]>()
    geometry.points.forEach(([x, y, z], i) => {
      const f = geometry.point_feature[i]
      const s = sums.get(f) ?? [0, 0, 0, 0]
      sums.set(f, [s[0] + x, s[1] + y, s[2] + (z ?? 0), s[3] + 1])
    })
    const order = [...sums.keys()].sort((a, b) => a - b)
    return {
      means: order.map((f) => {
        const [x, y, z, n] = sums.get(f) as [number, number, number, number]
        return [x / n, y / n, z / n]
      }),
      labels: order.map((f) => (detail.features[f]?.top != null ? visible(detail.features[f].top as string) : `#${detail.features[f]?.id}`)),
      meanColors: order.map((f) => colors.series[f % colors.series.length]),
    }
  }, [geometry, detail.features, colors])
  const pointColor = useMemo(() => (i: number) => colors.series[geometry.point_feature[i] % colors.series.length], [geometry, colors])

  return (
    <Suspense fallback={<div className="pl-note">Loading 3D…</div>}>
      <ShapeScene
        points={geometry.points}
        means={means}
        colors={meanColors}
        labels={labels}
        closed={false}
        connect={false}
        showPoints
        explained={geometry.explained}
        exampleText={(i) => {
          const [before, token, after] = geometry.point_context[i] ?? ['', '', '']
          return `…${before}[${token}]${after}…`
        }}
        pointColor={pointColor}
        pointTitle={(i) => `#${detail.features[geometry.point_feature[i]]?.id}`}
        ink={colors.text}
        muted={colors.baseline}
        surface={colors.surface}
        height={340}
      />
    </Suspense>
  )
}

const BARS_HEIGHT = 120

function ExplainedBars({ geometry, filename }: { geometry: NonNullable<AtlasCommunity['geometry']>; filename: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(320)
  const svgRef = useRef<SVGSVGElement>(null)
  const margin = { top: 8, right: 8, bottom: 20, left: 34 }
  const innerW = Math.max(width - margin.left - margin.right, 60)
  const innerH = BARS_HEIGHT - margin.top - margin.bottom
  const n = geometry.explained.length
  const slot = innerW / n
  const bar = slot * 0.34
  const y = (v: number) => margin.top + (1 - v) * innerH
  return (
    <div>
      <div className="pl-card-title">
        <h3 title="Explained variance ratio of each principal component, next to frequency-matched random groups of as many features">Explained · vs random</h3>
        <button
          type="button"
          className="pl-icon-button"
          title="Download SVG"
          aria-label="Download SVG"
          onClick={() => svgRef.current && downloadSvg(svgRef.current, `${filename}.svg`, colors.surface)}
          style={{ color: colors.text }}
        >
          <Download className="pl-icon" />
        </button>
      </div>
      <div ref={ref} className="pl-chart">
        <svg ref={svgRef} width={width} height={BARS_HEIGHT} role="img" aria-label="Explained variance per component against random groups">
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={margin.left} x2={margin.left + innerW} y1={y(t)} y2={y(t)} stroke={colors.grid} />
              <text x={margin.left - 5} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={colors.text}>
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          {geometry.explained.map((v, i) => {
            const x = margin.left + i * slot + slot / 2
            const b = geometry.baseline?.[i]
            return (
              <g key={i}>
                <rect x={x - bar} y={y(v)} width={bar} height={Math.max(0, y(0) - y(v))} fill={colors.series[0]}>
                  <title>PC{i + 1}: {formatPercent(v)}</title>
                </rect>
                {b != null && (
                  <rect x={x} y={y(b)} width={bar} height={Math.max(0, y(0) - y(b))} fill={colors.baseline} fillOpacity={0.6}>
                    <title>Random groups, PC{i + 1}: {formatPercent(b)}</title>
                  </rect>
                )}
                <text x={x} y={BARS_HEIGHT - 6} textAnchor="middle" fontSize={10} fill={colors.text}>
                  PC{i + 1}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <div className="pl-legend">
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.series[0] }} />
          Community
        </span>
        <span className="pl-legend-item">
          <span className="pl-legend-rect" style={{ background: colors.baseline, opacity: 0.6 }} />
          Random groups
        </span>
      </div>
    </div>
  )
}

const PLANE_HEIGHT = 220

function PcPlane({ detail, filename }: { detail: AtlasCommunity; filename: string }) {
  const colors = useLabColors()
  const { ref, width } = useElementWidth<HTMLDivElement>(320)
  const svgRef = useRef<SVGSVGElement>(null)
  const geometry = detail.geometry as NonNullable<AtlasCommunity['geometry']>
  const extent = Math.max(...geometry.points.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]), 1e-6) * 1.08
  const size = Math.min(width, PLANE_HEIGHT)
  const ox = width / 2
  const scale = (size / 2 - 8) / extent
  return (
    <div style={{ marginTop: '0.75rem' }}>
      <div className="pl-card-title">
        <h3 title="The same tokens on PC1 × PC2 (the figure export)">PC1 × PC2</h3>
        <button
          type="button"
          className="pl-icon-button"
          title="Download SVG"
          aria-label="Download SVG"
          onClick={() => svgRef.current && downloadSvg(svgRef.current, `${filename}.svg`, colors.surface)}
          style={{ color: colors.text }}
        >
          <Download className="pl-icon" />
        </button>
      </div>
      <div ref={ref} className="pl-chart">
        <svg ref={svgRef} width={width} height={size} role="img" aria-label="The community's tokens on its first two principal components">
          <line x1={ox - size / 2} x2={ox + size / 2} y1={size / 2} y2={size / 2} stroke={colors.grid} />
          <line x1={ox} x2={ox} y1={0} y2={size} stroke={colors.grid} />
          {geometry.points.map(([x, y], i) => (
            <circle key={i} cx={ox + x * scale} cy={size / 2 - y * scale} r={2.2} fill={colors.series[geometry.point_feature[i] % colors.series.length]} fillOpacity={0.7}>
              <title>
                #{detail.features[geometry.point_feature[i]]?.id}: …{geometry.point_context[i]?.[0]}[{geometry.point_context[i]?.[1]}]{geometry.point_context[i]?.[2]}…
              </title>
            </circle>
          ))}
          <text x={ox + size / 2 - 2} y={size / 2 - 4} textAnchor="end" fontSize={10} fill={colors.text} opacity={0.7}>
            PC1 {formatPercent(geometry.explained[0])}
          </text>
          <text x={ox + 4} y={10} fontSize={10} fill={colors.text} opacity={0.7}>
            PC2 {formatPercent(geometry.explained[1])}
          </text>
        </svg>
      </div>
    </div>
  )
}

function ConceptPanel({
  lab,
  detail,
  model,
  position,
  estimate,
}: {
  lab: GeometryLab
  detail: AtlasCommunity
  model: string
  position: number
  estimate: GeometryEstimate | null
}) {
  const colors = useLabColors()
  const concept = detail.concept
  return (
    <div>
      <div className="pl-card-title">
        <h3 title="Its most frequent firing words as values, the contexts it fires in as templates: read at every layer in the Shape view, with the shape tests and their baselines">
          As a concept
        </h3>
      </div>
      {!concept ? (
        <div className="pl-empty" style={{ borderColor: colors.border }}>
          {detail.concept_note ?? 'No concept for this community.'}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div className="gl-chips">
            {concept.values.map((v) => (
              <span key={v} className="gl-chip" style={{ borderColor: colors.border }}>
                {v}
              </span>
            ))}
          </div>
          <div className="pl-hint">
            {STRUCTURE_LABELS[concept.structure]}
            {concept.matched ? ` · order and positions of the built-in ${concept.matched}` : concept.structure === 'categorical' ? ' · no order: set one in Custom to test shapes' : ' · numeric order'}
            {' · '}
            {concept.templates.length} templates
          </div>
          <div className="pl-hint" style={{ opacity: 0.75 }} title={concept.templates.join('\n')}>
            {concept.templates.slice(0, 3).map((t) => `“${t}”`).join(' · ')}
          </div>
          <RunBar
            note={estimate ? `${estimate.num_examples} examples · ${estimateText(estimate)}` : null}
            warn={estimate != null && !estimate.within_limit}
            label="Open in Shape"
            running={lab.running}
            disabled={lab.running || (estimate != null && !estimate.within_limit)}
            onRun={() => lab.openConcept(concept, model, position)}
          />
        </div>
      )}
    </div>
  )
}

// Contexts and features

function ContextLine({ context, colors, label, onLabel }: { context: AtlasContext; colors: LabColors; label?: string; onLabel?: () => void }) {
  const highlight = colors.series[1]
  return (
    <div className="gl-context" style={{ borderTop: `1px solid ${colors.grid}` }}>
      {label && (
        <button type="button" className="pl-link gl-context-meta" onClick={onLabel} style={{ color: colors.accent }}>
          {label}
        </button>
      )}
      <span className="gl-context-meta">{formatValue(context.act, 2)}</span>
      {context.tokens.map((t, i) => {
        const level = (context.levels[i] ?? 0) / 100
        const text = t.replace(/\n/g, '⏎ ')
        return (
          <span
            key={i}
            title={`${formatValue((level * 100) | 0, 0)}% of max`}
            style={{
              background: level > 0 ? tint(highlight, 0.15 + 0.6 * level) : undefined,
              outline: i === context.at ? `1px solid ${highlight}` : undefined,
              fontWeight: i === context.at ? 600 : undefined,
            }}
          >
            {text}
          </span>
        )
      })}
    </div>
  )
}

function FeatureDetail({
  atlasRef,
  feature,
  byId,
  onFeature,
  onCommunity,
  onClose,
}: {
  atlasRef: AtlasRef
  feature: number
  byId: Map<number, AtlasCommunitySummary>
  onFeature: (f: number) => void
  onCommunity: (c: number) => void
  onClose: () => void
}) {
  const colors = useLabColors()
  const [data, setData] = useState<AtlasFeature | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    apiClient
      .atlasFeature(atlasRef, feature)
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the feature'))
    return () => {
      cancelled = true
    }
  }, [atlasRef, feature])

  const community = data?.community != null ? byId.get(data.community) : undefined
  return (
    <section className="pl-panel">
      <div className="gl-detail-head">
        <div className="pl-run-meta">
          <strong>Feature </strong>
          <FeatureLink sae={atlasRef.sae} feature={feature} text={data?.examples[0] ? contextText(data.examples[0]) : undefined} />
          {community && (
            <>
              {' · '}
              <button type="button" className="pl-link" style={{ color: colors.accent, fontSize: '0.75rem' }} onClick={() => onCommunity(community.id)}>
                community #{community.id}
              </button>
            </>
          )}
        </div>
        <button type="button" className="pl-icon-button" aria-label="Close" title="Close" onClick={onClose} style={{ color: colors.text }}>
          <X className="pl-icon" />
        </button>
      </div>
      {error && <div className="pl-error-note">{error}</div>}
      {data && data.tokens === 0 && <div className="pl-note">Never active over {formatTokens(data.corpus_tokens)} tokens.</div>}
      {data && data.tokens > 0 && (
        <>
          <StatTiles
            tiles={[
              { label: 'Active on', value: formatPercent(data.freq), hint: `${formatTokens(data.tokens)} tokens` },
              { label: 'Mean', value: formatValue(data.mean, 2), hint: 'where active' },
              { label: 'Max', value: formatValue(data.max, 2) },
            ]}
          />
          {data.neighbors.length > 0 && (
            <div style={{ marginBottom: '0.75rem' }}>
              <div className="pl-control-label" title="Features most often active on the same tokens (sqrt(P(j|i) P(i|j))); the share is P(j|i): how often each is active where this one is">
                Active with
              </div>
              <div className="gl-chips">
                {data.neighbors.map((n) => (
                  <button key={n.id} type="button" className="gl-chip" onClick={() => onFeature(n.id)} style={{ borderColor: colors.border }} title={`P(#${n.id} | #${feature}) = ${formatPercent(n.conditional)}`}>
                    #{n.id} · {formatPercent(n.conditional)}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="pl-card-title">
            <h3 title="The contexts it's most active on (one per sequence), its activation on each token shaded">Top contexts</h3>
          </div>
          {data.examples.map((c, i) => (
            <ContextLine key={i} context={c} colors={colors} />
          ))}
        </>
      )}
    </section>
  )
}
