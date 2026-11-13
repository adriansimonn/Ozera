/**
 * The SAE page's Autointerp mode: Claude's explanations of an SAE's features, from one of its
 * atlases. Run starts a job over a selection of features (estimated first; it holds its upper
 * bound and is settled on usage) and follows it; Features lists the explanations; Summary sets
 * their scores against random-direction controls; Feature shows one explanation with the
 * contexts it was written from and the scorer's mistakes. Views stay mounted while hidden.
 */

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Map as MapIcon, Sparkles } from 'lucide-react'
import { apiClient } from '../../api/client'
import {
  isActive,
  useAtlasRecompute,
  useAutointerpStatus,
  useJobEndRefresh,
  useJobEstimate,
  useSubmitJob,
  type AutointerpAtlasRef,
} from '../../hooks/useAutointerp'
import type { AutointerpJob, AutointerpStatus, JobRequest, JobSelection, JobSelectionKind } from '../../types/autointerp'
import type { AtlasCorpusId, AtlasMeta } from '../../types/geometry'
import type { SaeRef } from '../../types/probes'
import { Dropdown } from '../common/Dropdown'
import { LabKitStyles, Placeholder, RunBar, Segmented } from '../lab/chartKit'
import { useLabColors, type LabColors } from '../lab/labTheme'
import { refKey } from '../lab/saeLinks'
import { ExplanationTable } from './ExplanationTable'
import { FeatureExplanation } from './FeatureExplanation'
import { modelName } from './labels'
import { ScoreSummary } from './ScoreSummary'

type View = 'run' | 'features' | 'summary' | 'feature'

const ESTIMATE_DEBOUNCE_MS = 400

function atlasChoice(a: AtlasMeta): string {
  return `${a.model}|${a.corpus.id}`
}

function formatTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 1)}M`
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`
  return String(n)
}

function formatDate(iso: string): string {
  const date = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function usd(value: number | null | undefined): string {
  return value == null ? '–' : `$${value.toFixed(2)}`
}

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(timer)
  }, [value, ms])
  return debounced
}

export function AutointerpView({ sae, onShowPurchaseCredits }: { sae: SaeRef | null; onShowPurchaseCredits?: () => void }) {
  const colors = useLabColors()
  const atlases = useQuery({
    queryKey: ['atlas-lookup', sae ? refKey(sae) : ''],
    queryFn: () => apiClient.atlasLookup(sae as SaeRef),
    enabled: !!sae,
    staleTime: 60_000,
  })
  const [choice, setChoice] = useState('')
  const list = atlases.data ?? []
  const atlas = list.find((a) => atlasChoice(a) === choice) ?? list.find((a) => a.corpus.id === 'openwebtext') ?? list[0] ?? null
  const atlasRef = useMemo(
    (): AutointerpAtlasRef | null => (sae && atlas ? { model: atlas.model, sae, corpus: atlas.corpus.id as AtlasCorpusId } : null),
    // refKey and the atlas choice stand for sae and atlas
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sae ? refKey(sae) : '', atlas ? atlasChoice(atlas) : ''],
  )
  const [view, setView] = useState<View>('run')
  const [feature, setFeature] = useState<number | null>(null)
  const status = useAutointerpStatus(atlasRef)
  useJobEndRefresh(atlasRef, status.data)

  const openFeature = (f: number) => {
    setFeature(f)
    setView('feature')
  }

  let body
  if (!sae) {
    body = <Placeholder icon={Sparkles}>Choose an SAE.</Placeholder>
  } else if (atlases.isLoading) {
    body = <div className="pl-note">Looking for atlases…</div>
  } else if (!atlasRef || !atlas) {
    body = (
      <Placeholder icon={MapIcon}>
        No atlas of this SAE yet. Compute one in the{' '}
        <a href="/geometry" style={{ color: colors.accent }}>
          Geometry Lab
        </a>
        ’s Atlas view.
      </Placeholder>
    )
  } else {
    const fileName = `autointerp-${atlas.model}-${atlas.corpus.id}-${atlas.sae_name}`.replace(/[^\w.-]+/g, '_')
    body = (
      <>
        <div className="ai-head">
          <Segmented
            ariaLabel="Autointerp view"
            value={view}
            onChange={setView}
            options={[
              { value: 'run', label: 'Run' },
              { value: 'features', label: `Features${status.data?.explained ? ` (${status.data.explained.toLocaleString()})` : ''}` },
              { value: 'summary', label: 'Summary' },
              { value: 'feature', label: feature != null ? `Feature #${feature}` : 'Feature' },
            ]}
          />
          {list.length > 1 && (
            <div className="ai-atlas-pick">
              <Dropdown
                value={atlasChoice(atlas)}
                onChange={setChoice}
                options={list.map((a) => ({ value: atlasChoice(a), label: `${a.corpus.name} · ${a.model} · ${formatTokens(a.tokens)} tokens` }))}
              />
            </div>
          )}
        </div>
        <div key={atlasChoice(atlas)}>
          <div hidden={view !== 'run'}>
            <RunView atlasRef={atlasRef} status={status.data} statusError={(status.error as Error | null)?.message ?? null} colors={colors} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
          <div hidden={view !== 'features'}>
            <ExplanationTable atlasRef={atlasRef} fileName={fileName} selected={feature} onSelect={openFeature} />
          </div>
          <div hidden={view !== 'summary'}>{view === 'summary' || status.data?.explained ? <ScoreSummary atlasRef={atlasRef} /> : null}</div>
          <div hidden={view !== 'feature'}>
            <FeatureView atlasRef={atlasRef} feature={feature} onFeature={setFeature} width={atlas.width} onShowPurchaseCredits={onShowPurchaseCredits} />
          </div>
        </div>
      </>
    )
  }

  return (
    <div className="autointerp-view lab-kit" style={{ color: colors.text }}>
      <LabKitStyles />
      {body}
      <style>{`
        .autointerp-view .ai-head { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
        .autointerp-view .ai-atlas-pick { min-width: 260px; }
        .autointerp-view .ai-panel { padding: 1rem 1.25rem; border: 1px solid; margin-bottom: 1rem; }
        .autointerp-view .ai-form { display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: flex-end; margin-top: 0.75rem; }
        .autointerp-view .ai-form .ai-models { min-width: 200px; }
        .autointerp-view .ai-number { width: 110px; }
        .autointerp-view .ai-progress { height: 6px; border: 1px solid; margin-top: 0.4rem; }
        .autointerp-view .ai-progress > div { height: 100%; transition: width 0.4s; }
        .autointerp-view .ai-jobs { display: flex; flex-direction: column; gap: 0.6rem; }
        .autointerp-view .ai-job { font-size: 0.75rem; line-height: 1.5; }
        .autointerp-view .ai-filters { align-items: flex-end; }
        .autointerp-view .ai-export { display: flex; gap: 0.4rem; margin-left: auto; }
        .autointerp-view .ai-table-wrap { max-height: 62vh; overflow: auto; }
        .autointerp-view .ai-table td { white-space: nowrap; }
        .autointerp-view .ai-table td.ai-explanation-cell { text-align: left; white-space: nowrap; max-width: 30rem; overflow: hidden; text-overflow: ellipsis; }
        .autointerp-view .ai-sort { background: none; border: none; padding: 0; cursor: pointer; font: inherit; font-weight: 600; display: inline-flex; gap: 2px; align-items: center; }
        .autointerp-view .ai-pager { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; margin-top: 0.75rem; }
        .autointerp-view .ai-bands { max-width: 640px; }
        .autointerp-view .ai-contexts { margin-top: 0.5rem; }
        .autointerp-view .ai-context-group { margin-top: 1rem; }
        .autointerp-view .ai-context { font-size: 0.75rem; line-height: 1.7; white-space: pre-wrap; word-break: break-word; padding: 0.3rem 0; }
        .autointerp-view .ai-context span { border-radius: 2px; }
        .autointerp-view .ai-context-meta { font-size: 0.68rem; opacity: 0.7; margin-right: 0.5rem; font-variant-numeric: tabular-nums; }
        .autointerp-view .ai-section { margin-top: 1.1rem; }
        .autointerp-view .ai-section-head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.9rem; margin-bottom: 0.4rem; }
        .autointerp-view .ai-section-head .pl-control-label { margin-bottom: 0; }
        .autointerp-view .ai-stat { font-size: 0.75rem; font-variant-numeric: tabular-nums; }
        .autointerp-view .ai-small-button { padding: 0.25rem 0.6rem; font-size: 0.72rem; }
        .autointerp-view .ai-generation-texts { margin-top: 0.4rem; }
        .autointerp-view .ai-hypothesis-form { display: flex; gap: 0.5rem; align-items: stretch; max-width: 760px; }
        .autointerp-view .ai-hypothesis-table { margin-top: 0.6rem; }
        .autointerp-view .ai-hypothesis-table td { white-space: nowrap; }
        .autointerp-view .ai-hypothesis-table td.ai-explanation-cell { text-align: left; max-width: 34rem; overflow: hidden; text-overflow: ellipsis; }
        .autointerp-view .ai-row-tag { display: inline-block; min-width: 6.5rem; font-size: 0.68rem; font-weight: 600; opacity: 0.75; }
        .autointerp-view .ai-reexplain { display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: flex-end; }
        .autointerp-view .ai-reexplain .ai-models { min-width: 190px; }
        .autointerp-view .ai-hypothesis-detail { padding: 0.75rem 0.9rem; border: 1px solid; }
      `}</style>
    </div>
  )
}

// Run

function JobLine({ job, colors }: { job: AutointerpJob; colors: LabColors }) {
  const naming = job.selection.kind === 'communities'
  const scope = naming
    ? `${(job.communities ?? 0).toLocaleString()} community names · ${modelName(job.explainer)}`
    : `${job.features.toLocaleString()} features + ${job.controls} controls · ${job.scored.toLocaleString()} scored · ${modelName(job.explainer)} / ${modelName(job.scorer)}`
  if (isActive(job)) {
    const fraction = job.progress?.fraction ?? 0
    return (
      <div className="ai-job">
        <div>
          <strong>{job.status === 'queued' ? 'Starting' : job.progress?.stage ?? 'Running'}</strong>
          {job.progress && job.progress.fraction > 0 ? ` · ${Math.round(fraction * 100)}%` : ''} · {scope}
          {job.mine ? (job.held != null ? ` · holding ${usd(job.held)}` : '') : ' · started by another user'}
        </div>
        <div className="ai-progress" style={{ borderColor: colors.border }}>
          <div style={{ width: `${Math.max(2, fraction * 100)}%`, background: colors.accent }} />
        </div>
      </div>
    )
  }
  const results = job.results ?? {}
  return (
    <div className="ai-job" style={{ opacity: 0.85 }}>
      {job.status === 'completed' ? (
        naming ? (
          <>
            <strong>Completed</strong> {job.finished_at ? formatDate(job.finished_at) : ''} · {(job.names?.named ?? 0).toLocaleString()} named
            {job.names?.refused ? `, ${job.names.refused} refused` : ''}
            {job.names?.failed ? `, ${job.names.failed} failed` : ''}
          </>
        ) : (
          <>
            <strong>Completed</strong> {job.finished_at ? formatDate(job.finished_at) : ''} · {(results.scored ?? 0).toLocaleString()} scored,{' '}
            {(results.unscored ?? 0).toLocaleString()} unscored{results.refused ? `, ${results.refused} refused` : ''}
          </>
        )
      ) : (
        <>
          <strong style={{ color: colors.warning }}>Failed</strong>: {job.error ?? 'unknown error'}
        </>
      )}
      {job.mine && job.settled != null ? ` · cost ${usd(job.settled)}${job.refunded ? ` (refunded ${usd(job.refunded)})` : ''}` : ''}
      <div style={{ opacity: 0.75 }}>{scope}</div>
    </div>
  )
}

function parseList(text: string): number[] {
  return [...new Set(text.split(/[\s,]+/).filter(Boolean).map(Number).filter((n) => Number.isInteger(n) && n >= 0))]
}

function RunView({
  atlasRef,
  status,
  statusError,
  colors,
  onShowPurchaseCredits,
}: {
  atlasRef: AutointerpAtlasRef
  status: AutointerpStatus | undefined
  statusError: string | null
  colors: LabColors
  onShowPurchaseCredits?: () => void
}) {
  const [kind, setKind] = useState<JobSelectionKind>('top')
  const [n, setN] = useState('1000')
  const [start, setStart] = useState('0')
  const [end, setEnd] = useState('999')
  const [listText, setListText] = useState('')
  const [scoring, setScoring] = useState<'all' | 'sample'>('all')
  const [explainer, setExplainer] = useState<string | null>(null)
  const [scorer, setScorer] = useState<string | null>(null)
  const recompute = useAtlasRecompute(atlasRef, status)
  const submit = useSubmitJob(atlasRef)

  const selection = useMemo((): JobSelection | null => {
    if (kind === 'top') return Number(n) >= 1 ? { kind, n: Math.floor(Number(n)), scoring } : null
    if (kind === 'range') return Number(end) >= Number(start) ? { kind, start: Math.floor(Number(start)), end: Math.floor(Number(end)), scoring } : null
    if (kind === 'list') {
      const features = parseList(listText)
      return features.length ? { kind, features, scoring } : null
    }
    return { kind, scoring }
  }, [kind, n, start, end, listText, scoring])
  const ready = status?.state === 'ready' && status.available
  const request = useMemo(
    (): JobRequest | null =>
      ready && selection && status
        ? { ...atlasRef, selection, explainer: explainer ?? status.explainer, scorer: scorer ?? status.scorer }
        : null,
    [ready, selection, status, atlasRef, explainer, scorer],
  )
  const debounced = useDebounced(request, ESTIMATE_DEBOUNCE_MS)
  const estimate = useJobEstimate(debounced)
  // Naming the atlas's communities: a job of its own, from the explanations stored now
  const namingRequest = useMemo(
    (): JobRequest | null =>
      ready && status?.communities?.nameable
        ? { ...atlasRef, selection: { kind: 'communities', scoring: 'all' }, explainer: explainer ?? status.explainer, scorer: scorer ?? status.scorer }
        : null,
    [ready, status, atlasRef, explainer, scorer],
  )
  const namingEstimate = useJobEstimate(useDebounced(namingRequest, ESTIMATE_DEBOUNCE_MS))
  const naming = useSubmitJob(atlasRef)

  if (!status) return <div className="pl-note">{statusError ?? 'Loading…'}</div>
  const a = status.atlas
  const active = status.jobs.find(isActive) ?? null
  const runError = submit.error && (submit.error as Error).message !== 'INSUFFICIENT_CREDITS' ? (submit.error as Error).message : null
  const e = estimate.data
  const nothing = e != null && e.features + e.controls === 0
  const note = !request
    ? null
    : estimate.isFetching && !e
      ? 'Estimating…'
      : estimate.error
        ? (estimate.error as Error).message
        : e
          ? nothing
            ? 'Every selected feature is already explained with these models'
            : `${e.features.toLocaleString()} features + ${e.controls} controls · ${e.scored.toLocaleString()} scored${e.skipped ? ` · ${e.skipped.toLocaleString()} already explained` : ''} · ≈ ${usd(e.estimated_cost)}, holds up to ≈ ${usd(e.hold)}`
          : null

  const run = () => {
    if (!request) return
    submit.mutate(request, {
      onError: (err) => {
        if ((err as Error).message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      },
    })
  }

  const models = status.job_models.map((m) => ({ value: m.id, label: m.name }))
  const input = { borderColor: colors.border, background: 'transparent', color: colors.text }

  const c = status.communities
  const ne = namingEstimate.data
  const namingError = naming.error && (naming.error as Error).message !== 'INSUFFICIENT_CREDITS' ? (naming.error as Error).message : null
  const namingNote = !c
    ? null
    : active
      ? 'A job is running on this atlas'
      : `${c.named} of ${c.total} named · ${c.nameable} nameable` +
        (namingRequest && ne && ne.communities > 0 ? ` · ${ne.communities} to name ≈ ${usd(ne.estimated_cost)}, holds up to ≈ ${usd(ne.hold)}` : '')
  const runNaming = () => {
    if (!namingRequest) return
    naming.mutate(namingRequest, {
      onError: (err) => {
        if ((err as Error).message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      },
    })
  }

  return (
    <>
      <section className="ai-panel" style={{ borderColor: colors.border, background: colors.panel }}>
        <div className="pl-run-meta">
          {a ? (
            <>
              <strong>{a.corpus_name}</strong> · {formatTokens(a.tokens)} tokens · {a.width.toLocaleString()} features
              {status.alive != null ? ` (${status.alive.toLocaleString()} live)` : ''}
              {a.l0 != null ? ` · L0 ${a.l0.toFixed(1)}` : ''} · atlas {formatDate(a.created_at)} · {status.explained.toLocaleString()} explained · {status.controls}{' '}
              controls scored
            </>
          ) : (
            'No atlas'
          )}
        </div>

        {status.state === 'recompute' ? (
          <>
            <div className="pl-hint">{status.reason}</div>
            {recompute.running ? (
              <div className="pl-hint" style={{ marginTop: '0.5rem' }}>
                Recomputing the atlas{recompute.job?.progress ? ` · ${recompute.job.progress.stage} ${Math.round(recompute.job.progress.fraction * 100)}%` : ''}
              </div>
            ) : (
              <RunBar
                note={recompute.estimate ? `≈ ${usd(recompute.estimate.estimated_cost)} · same tokens and seed` : null}
                label="Recompute atlas"
                running={recompute.starting}
                disabled={!recompute.estimate}
                onRun={recompute.start}
              />
            )}
            {recompute.error && <div className="pl-error-note">{recompute.error}</div>}
          </>
        ) : status.state !== 'ready' ? (
          <div className="pl-hint" style={{ color: status.state === 'dense' ? colors.warning : undefined }}>
            {status.reason}
          </div>
        ) : !status.available ? (
          <div className="pl-hint">Autointerp isn’t configured on this server.</div>
        ) : (
          <>
            <div className="ai-form">
              <div>
                <div className="pl-control-label">Features</div>
                <Segmented
                  ariaLabel="Features"
                  value={kind}
                  onChange={setKind}
                  options={[
                    { value: 'top', label: 'Most frequent', title: 'The N features active on the most corpus tokens' },
                    { value: 'range', label: 'Range' },
                    { value: 'list', label: 'List' },
                    { value: 'all', label: 'All', title: `Every live feature (at most ${status.max_job_features.toLocaleString()} a job)` },
                  ]}
                />
              </div>
              {kind === 'top' && (
                <div className="ai-number">
                  <div className="pl-control-label">N</div>
                  <input className="pl-input" type="number" min={1} max={status.max_job_features} value={n} onChange={(ev) => setN(ev.target.value)} style={input} />
                </div>
              )}
              {kind === 'range' && (
                <>
                  <div className="ai-number">
                    <div className="pl-control-label">From</div>
                    <input className="pl-input" type="number" min={0} value={start} onChange={(ev) => setStart(ev.target.value)} style={input} />
                  </div>
                  <div className="ai-number">
                    <div className="pl-control-label">To</div>
                    <input className="pl-input" type="number" min={0} value={end} onChange={(ev) => setEnd(ev.target.value)} style={input} />
                  </div>
                </>
              )}
              <div>
                <div className="pl-control-label" title="Scoring is most of a job's tokens">
                  Score
                </div>
                <Segmented
                  ariaLabel="Score"
                  value={scoring}
                  onChange={setScoring}
                  options={[
                    { value: 'all', label: 'All' },
                    { value: 'sample', label: `Sample of ${status.score_sample}`, title: 'A sample stratified by frequency; the rest are explained only' },
                  ]}
                />
              </div>
              <div className="ai-models">
                <div className="pl-control-label">Explainer</div>
                <Dropdown value={explainer ?? status.explainer} onChange={setExplainer} options={models} />
              </div>
              <div className="ai-models">
                <div className="pl-control-label">Scorer</div>
                <Dropdown value={scorer ?? status.scorer} onChange={setScorer} options={models} />
              </div>
            </div>
            {kind === 'list' && (
              <textarea
                className="pl-textarea"
                value={listText}
                onChange={(ev) => setListText(ev.target.value)}
                placeholder="Feature IDs, separated by commas or spaces"
                style={{ ...input, marginTop: '0.75rem', minHeight: 60 }}
              />
            )}
            <RunBar
              note={active ? 'A job is running on this atlas' : note}
              warn={!!estimate.error}
              label="Run job"
              running={submit.isPending}
              disabled={!request || !e || nothing || !!active || !!estimate.error}
              onRun={run}
            />
            {runError && <div className="pl-error-note">{runError}</div>}
          </>
        )}
      </section>

      {ready && c && c.total > 0 && (
        <section className="ai-panel" style={{ borderColor: colors.border, background: colors.panel }}>
          <div
            className="pl-control-label"
            title="One request per community names what its features share, from their explanations and contexts; the Geometry Lab’s atlas shows the names. A community is nameable once half its features (at least 2) are explained."
          >
            Community names
          </div>
          <RunBar
            note={namingNote}
            label="Name communities"
            running={naming.isPending}
            disabled={!namingRequest || !ne || ne.communities === 0 || !!active || !!namingEstimate.error}
            onRun={runNaming}
          />
          {namingError && <div className="pl-error-note">{namingError}</div>}
        </section>
      )}

      {status.jobs.length > 0 && (
        <section className="ai-panel" style={{ borderColor: colors.border, background: colors.panel }}>
          <div className="pl-control-label">Jobs</div>
          <div className="ai-jobs">
            {status.jobs.map((job) => (
              <JobLine key={job.job_id} job={job} colors={colors} />
            ))}
          </div>
        </section>
      )}
    </>
  )
}

// Feature

function FeatureView({
  atlasRef,
  feature,
  onFeature,
  width,
  onShowPurchaseCredits,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number | null
  onFeature: (feature: number) => void
  width: number
  onShowPurchaseCredits?: () => void
}) {
  const colors = useLabColors()
  const [text, setText] = useState(feature != null ? String(feature) : '')
  useEffect(() => setText(feature != null ? String(feature) : ''), [feature])
  const go = () => {
    const f = Number(text)
    if (Number.isInteger(f) && f >= 0 && f < width) onFeature(f)
  }
  return (
    <>
      <div className="pl-controls-row" style={{ alignItems: 'flex-end' }}>
        <div className="ai-number">
          <div className="pl-control-label">Feature</div>
          <input
            className="pl-input"
            type="number"
            min={0}
            max={width - 1}
            value={text}
            onChange={(ev) => setText(ev.target.value)}
            onKeyDown={(ev) => ev.key === 'Enter' && go()}
            onBlur={go}
            style={{ borderColor: colors.border, background: 'transparent', color: colors.text }}
          />
        </div>
      </div>
      {feature != null ? (
        <FeatureExplanation key={feature} atlasRef={atlasRef} feature={feature} onShowPurchaseCredits={onShowPurchaseCredits} detailed />
      ) : (
        <Placeholder icon={Sparkles}>Pick a feature in Features, or enter its ID.</Placeholder>
      )}
    </>
  )
}
