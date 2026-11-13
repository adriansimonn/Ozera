/**
 * A feature's autointerp explanation (in the SAE browser, and the Autointerp tab's Feature view):
 * the explanation and its scores (next to the token baseline and the random-direction controls),
 * recall by activation decile, and the tokens its decoder row promotes and suppresses; or an
 * Explain button with its price. In detail, also what tests it (a generation test, the user's
 * own hypotheses scored on the same held-out set, and its earlier explanations with
 * re-explaining), the explain set it was written from and the scorer's mistakes on its held-out
 * contexts.
 */

import { useState } from 'react'
import { Loader2, Sparkles } from 'lucide-react'
import { useThemeColors } from '../../hooks/useTheme'
import { useAutointerpStatus, useExplainFeature, useFeatureAutointerp, useFeatureExamples, type AutointerpAtlasRef } from '../../hooks/useAutointerp'
import type { ExampleContext, ExplanationSource, TokenWeight } from '../../types/autointerp'
import { useLabColors } from '../lab/labTheme'
import { ContextLine } from './ContextLine'
import { GenerationTestPanel, HypothesisPanel, Versions } from './ExplanationTests'
import { formatScore, modelName } from './labels'
import { Scores } from './ScoreLine'

function price(value: number): string {
  return `$${value.toFixed(2)}`
}

function Tokens({ label, tokens, title, color, border }: { label: string; tokens: TokenWeight[]; title: string; color: string; border: string }) {
  if (!tokens.length) return null
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'baseline' }}>
      <span title={title} style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.04em', opacity: 0.7, minWidth: 74 }}>
        {label}
      </span>
      {tokens.map((t, i) => (
        <span
          key={i}
          title={t.value.toFixed(3)}
          style={{ fontFamily: 'monospace', fontSize: '0.75rem', padding: '0 4px', border: `1px solid ${border}`, color, whiteSpace: 'pre' }}
        >
          {JSON.stringify(t.token).slice(1, -1)}
        </span>
      ))}
    </div>
  )
}

export function FeatureExplanation({
  atlasRef,
  feature,
  onShowPurchaseCredits,
  detailed = false,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number
  onShowPurchaseCredits?: () => void
  /** Also show the explain set and the scorer's mistakes */
  detailed?: boolean
}) {
  const tc = useThemeColors()
  const status = useAutointerpStatus(atlasRef)
  const ready = status.data?.state === 'ready'
  const featureRef = { ...atlasRef, feature }
  const detail = useFeatureAutointerp(featureRef, !!status.data && status.data.state !== 'no_atlas')
  const explain = useExplainFeature()
  // Detail: the earlier explanation in view (null: the one shown), and the hypothesis opened
  const [viewing, setViewing] = useState<number | null>(null)
  const [hypothesisId, setHypothesisId] = useState<number | null>(null)
  const accent = tc.isLight ? '#1d1d1f' : '#ffffff'
  const faint = tc.isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)'

  const versions = detail.data?.versions ?? []
  const earlier = detailed && viewing != null ? versions.find((v) => v.id === viewing) ?? null : null
  const e = earlier ?? detail.data?.explanation ?? null
  const hypothesis = detailed ? detail.data?.hypotheses.find((h) => h.id === hypothesisId) ?? null : null
  const explaining = explain.isPending && explain.variables?.feature === feature
  const explainError = explain.error && explain.variables?.feature === feature ? (explain.error as Error).message : null
  const runExplain = () =>
    explain.mutate(featureRef, {
      onError: (err) => {
        if (err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      },
    })

  const button = {
    padding: '0.35rem 0.75rem',
    fontSize: '0.8rem',
    border: `1px solid ${tc.borderStrong}`,
    background: tc.surfaceActive,
    color: tc.text,
    cursor: 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
  } as const

  let body
  if (status.isLoading || detail.isLoading) {
    body = <Loader2 size={14} className="animate-spin" style={{ opacity: 0.6 }} />
  } else if (!status.data) {
    body = <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>{(status.error as Error | null)?.message ?? 'Unavailable'}</span>
  } else if (e?.explanation) {
    body = (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontSize: '0.95rem', lineHeight: 1.35 }}>{e.explanation}</div>
        {e.status === 'scored' ? (
          <Scores e={e} accent={accent} faint={faint} />
        ) : (
          <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{e.note ?? 'Not scored'}</div>
        )}
        <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>
          {modelName(e.explainer)}
          {e.status === 'scored' ? ` · scored by ${modelName(e.scorer)}` : ''}
          {earlier && (
            <>
              {' · an earlier explanation · '}
              <button type="button" className="pl-link" style={{ color: 'inherit', textDecoration: 'underline' }} onClick={() => setViewing(null)}>
                show the newest
              </button>
            </>
          )}
        </div>
      </div>
    )
  } else if (e?.status === 'refused') {
    body = <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>{e.note ?? 'Refused'}</span>
  } else if (status.data.state === 'recompute') {
    body = (
      <span style={{ fontSize: '0.8rem', opacity: 0.7 }} title={status.data.reason ?? undefined}>
        Atlas v{status.data.atlas?.version ?? 1} · recompute it under Autointerp
      </span>
    )
  } else if (!ready) {
    body = <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>{status.data.reason}</span>
  } else if (detail.data && !detail.data.has_sets) {
    body = <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Never active over the corpus</span>
  } else if (!status.data.available) {
    body = <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Autointerp isn’t configured</span>
  } else {
    body = (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <button
          type="button"
          style={button}
          disabled={explaining}
          onClick={runExplain}
          title={`${modelName(status.data.explainer)} explains it from its top and decile contexts; ${modelName(status.data.scorer)} scores it on held-out ones. Charged its usage.`}
        >
          {explaining ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
          {explaining ? 'Explaining' : `Explain · ${price(status.data.explain_price)}`}
        </button>
        {detail.data && !detail.data.scorable && <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>{detail.data.unscored_note}</span>}
        {explainError && (
          <span style={{ fontSize: '0.75rem', color: 'rgba(239,68,68,0.9)' }}>
            {explainError === 'INSUFFICIENT_CREDITS' ? 'Not enough credits' : explainError}
          </span>
        )}
      </div>
    )
  }

  const promoted = detail.data?.promoted ?? []
  const suppressed = detail.data?.suppressed ?? []
  return (
    <div className="ai-explanation" style={{ border: `1px solid ${tc.border}`, background: tc.surface, padding: '0.75rem 1rem', marginBottom: '0.75rem', color: tc.text }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
        <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.7 }}>Explanation</span>
        {e?.status === 'scored' && (
          <span style={{ display: 'inline-flex', gap: 8, alignItems: 'baseline' }}>
            {e.token_feature && (
              <span
                title="The explanation doesn’t beat its top tokens by much: what it fires on is mostly which token it is"
                style={{ fontSize: '0.7rem', padding: '0 5px', border: `1px solid ${tc.borderStrong}` }}
              >
                Token feature
              </span>
            )}
            <span title="Mean of detection and fuzzing (0.5 is chance); ± is how much it moves when explained and scored again (95%)" style={{ fontFamily: 'monospace', fontSize: '0.9rem' }}>
              {formatScore(e.score)}
              {e.score_pm != null && (
                <span style={{ opacity: 0.6, fontSize: '0.75rem' }}> ±{e.score_pm.toFixed(2)}</span>
              )}
            </span>
          </span>
        )}
      </div>
      {body}
      {(promoted.length > 0 || suppressed.length > 0) && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 10 }}>
          <Tokens
            label="Promotes"
            tokens={promoted}
            title="Tokens its decoder row raises most through the unembedding (logit change)"
            color={tc.text}
            border={tc.border}
          />
          <Tokens label="Suppresses" tokens={suppressed} title="Tokens its decoder row lowers most" color={tc.text} border={tc.border} />
        </div>
      )}
      {detailed && ready && status.data && detail.data?.has_sets && (
        <>
          {e?.explanation && (
            <GenerationTestPanel
              key={`e${e.id}`}
              atlasRef={atlasRef}
              feature={feature}
              source={{ explanation_id: e.id }}
              tests={detail.data.tests}
              status={status.data}
              onShowPurchaseCredits={onShowPurchaseCredits}
            />
          )}
          {versions.length > 0 && (
            <Versions
              atlasRef={atlasRef}
              feature={feature}
              versions={versions}
              viewing={viewing}
              onView={setViewing}
              status={status.data}
              onShowPurchaseCredits={onShowPurchaseCredits}
            />
          )}
          {detail.data.scorable && (
            <HypothesisPanel
              atlasRef={atlasRef}
              feature={feature}
              explanation={detail.data.explanation}
              hypotheses={detail.data.hypotheses}
              tests={detail.data.tests}
              status={status.data}
              selected={hypothesis?.id ?? null}
              onSelect={setHypothesisId}
              onShowPurchaseCredits={onShowPurchaseCredits}
            />
          )}
          {hypothesis && (
            <div className="ai-section ai-hypothesis-detail" style={{ borderColor: tc.border }}>
              <div style={{ fontSize: '0.9rem', lineHeight: 1.35, marginBottom: 6 }}>{hypothesis.text}</div>
              {hypothesis.status === 'scored' ? (
                <Scores e={hypothesis} accent={accent} faint={faint} />
              ) : (
                <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{hypothesis.note ?? 'Not scored'}</div>
              )}
              <div style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: 4 }}>Your hypothesis · scored by {modelName(hypothesis.scorer)}</div>
              <GenerationTestPanel
                key={`h${hypothesis.id}`}
                atlasRef={atlasRef}
                feature={feature}
                source={{ hypothesis_id: hypothesis.id }}
                tests={detail.data.tests}
                status={status.data}
                onShowPurchaseCredits={onShowPurchaseCredits}
              />
            </div>
          )}
          <FeatureContexts
            atlasRef={atlasRef}
            feature={feature}
            source={hypothesis ? { hypothesis_id: hypothesis.id } : earlier ? { explanation_id: earlier.id } : {}}
            subject={hypothesis ? 'your hypothesis' : earlier ? 'this earlier explanation' : null}
          />
        </>
      )}
    </div>
  )
}

const EXPLAIN_SHOWN = 10

function ContextGroup({ title, hint, contexts, total, tag }: {
  title: string
  hint: string
  contexts: ExampleContext[]
  total?: number
  tag: (c: ExampleContext) => string
}) {
  const colors = useLabColors()
  if (!contexts.length) return null
  return (
    <div className="ai-context-group">
      <div className="pl-control-label" title={hint}>
        {title} · {total ?? contexts.length}
        {total != null && total > contexts.length ? ` (${contexts.length} shown)` : ''}
      </div>
      {contexts.map((c, i) => (
        <ContextLine key={i} context={c} colors={colors} tag={tag(c)} />
      ))}
    </div>
  )
}

function decileTag(c: ExampleContext): string {
  return c.kind === 'top' ? 'top' : c.decile != null ? `d${c.decile + 1}` : ''
}

/**
 * The explain set the explanation was written from, and where the scorer got it wrong (on the
 * explanation shown, or on the explanation or hypothesis that source names).
 */
function FeatureContexts({
  atlasRef,
  feature,
  source,
  subject,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number
  source: ExplanationSource
  /** Whose mistakes these are, when not the shown explanation's */
  subject: string | null
}) {
  const examples = useFeatureExamples({ ...atlasRef, feature }, true, source)
  const [allExplain, setAllExplain] = useState(false)
  const data = examples.data
  if (examples.isLoading) return <div className="pl-note" style={{ marginTop: 12 }}>Loading contexts…</div>
  if (!data) return null
  const mistakes = data.mistakes
  return (
    <div className="ai-contexts">
      {mistakes && (
        <>
          {subject && (
            <div className="pl-note" style={{ marginTop: '1rem' }}>
              The scorer’s mistakes on {subject}
            </div>
          )}
          <ContextGroup
            title="Missed"
            hint="Held-out contexts where it fires that the scorer said it doesn't, strongest first"
            contexts={mistakes.missed}
            total={mistakes.counts.missed}
            tag={decileTag}
          />
          <ContextGroup
            title="False alarms"
            hint="Contexts where it doesn't fire that the scorer said it does (random windows, or ones with its top token)"
            contexts={mistakes.false_alarms}
            total={mistakes.counts.false_alarms}
            tag={(c) => (c.kind === 'token' ? 'same token' : 'random')}
          />
          <ContextGroup
            title="Fuzzing errors"
            hint="Underlined: the tokens the scorer saw marked (correctly or at random); tinted: where it really fires"
            contexts={mistakes.fuzzing}
            total={mistakes.counts.fuzzing}
            tag={decileTag}
          />
          {mistakes.counts.missed + mistakes.counts.false_alarms + mistakes.counts.fuzzing === 0 && (
            <div className="pl-note">No mistakes on the held-out set.</div>
          )}
        </>
      )}
      {data.note && <div className="pl-note">{data.note}</div>}
      <ContextGroup
        title="Explain set"
        hint="The contexts the explainer read: its strongest, then a sample from each activation decile"
        contexts={allExplain ? data.explain : data.explain.slice(0, EXPLAIN_SHOWN)}
        total={data.explain.length}
        tag={decileTag}
      />
      {data.explain.length > EXPLAIN_SHOWN && (
        <button type="button" className="pl-link" onClick={() => setAllExplain((v) => !v)}>
          {allExplain ? 'Show fewer' : `Show all ${data.explain.length}`}
        </button>
      )}
    </div>
  )
}
