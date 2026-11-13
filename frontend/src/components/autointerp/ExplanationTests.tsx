/**
 * Testing a feature's explanations (the Autointerp tab's Feature view):
 *
 * - GenerationTestPanel: Claude writes texts an explanation says the feature fires on, and
 *   near-misses; the model and SAE read them. Shows how many fire, and the texts.
 * - HypothesisPanel: the user's own explanations, scored on the same held-out set, next to the
 *   autointerp explanation and the token baseline.
 * - Versions: every explanation of the feature (the newest is shown, never the best-scoring),
 *   and explaining it again with other models.
 */

import { useState } from 'react'
import { Loader2, Trash2 } from 'lucide-react'
import { useDeleteHypothesis, useExplainFeature, useGenerationTest, useScoreHypothesis, type AutointerpAtlasRef } from '../../hooks/useAutointerp'
import type { AutointerpStatus, ExampleContext, Explanation, ExplanationSource, GenerationTest, GenerationText, Hypothesis } from '../../types/autointerp'
import { Dropdown } from '../common/Dropdown'
import { useLabColors } from '../lab/labTheme'
import { ContextLine } from './ContextLine'
import { formatScore, modelName } from './labels'

// The backend's MAX_HYPOTHESIS_CHARS (core/autointerp/budget.py): a hypothesis is as short as an explanation
const MAX_HYPOTHESIS_CHARS = 200

function usd(value: number): string {
  return `$${value.toFixed(2)}`
}

function percent(rate: number | null): string {
  return rate == null ? '–' : `${Math.round(rate * 100)}%`
}

function formatDate(iso: string): string {
  const date = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function creditsError(message: string | null): string | null {
  return message === 'INSUFFICIENT_CREDITS' ? 'Not enough credits' : message
}

function testOf(tests: GenerationTest[], source: ExplanationSource): GenerationTest | null {
  return (
    tests.find((t) => (source.explanation_id != null ? t.explanation_id === source.explanation_id : t.hypothesis_id === source.hypothesis_id)) ?? null
  )
}

function asContext(t: GenerationText): ExampleContext {
  return { tokens: t.tokens, levels: t.levels, read: t.read, centre: 0, act: t.peak, decile: null, marks: null, kind: null }
}

const GENERATION_HINT =
  'Claude writes 10 texts the explanation says it fires on, and 5 near-misses it says it doesn’t; the model and SAE read them. A text fires when the feature reaches half its corpus maximum on some token.'

export function GenerationTestPanel({
  atlasRef,
  feature,
  source,
  tests,
  status,
  onShowPurchaseCredits,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number
  /** The explanation or hypothesis tested */
  source: ExplanationSource
  tests: GenerationTest[]
  status: AutointerpStatus
  onShowPurchaseCredits?: () => void
}) {
  const colors = useLabColors()
  const run = useGenerationTest()
  const [open, setOpen] = useState(false)
  const test = testOf(tests, source)
  const error = creditsError(run.error ? (run.error as Error).message : null)
  const start = () =>
    run.mutate(
      { ...atlasRef, feature, ...source },
      {
        onSuccess: () => setOpen(true),
        onError: (err) => {
          if ((err as Error).message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
        },
      },
    )

  const firing = test?.texts.filter((t) => t.kind === 'firing') ?? []
  const near = test?.texts.filter((t) => t.kind === 'near_miss') ?? []
  return (
    <div className="ai-section">
      <div className="ai-section-head">
        <span className="pl-control-label" title={GENERATION_HINT}>
          Generation test
        </span>
        {test ? (
          <>
            <span className="ai-stat" title="Texts written to make it fire on which it reaches the threshold">
              <strong>{firing.filter((t) => t.fired).length}/{firing.length}</strong> fire
            </span>
            <span className="ai-stat" title="Near-misses on which it stays below the threshold">
              <strong>{near.filter((t) => !t.fired).length}/{near.length}</strong> near-misses quiet
            </span>
            <span className="ai-stat" style={{ opacity: 0.65 }}>{modelName(test.generator)}</span>
            <button type="button" className="pl-link" style={{ color: colors.accent }} onClick={() => setOpen((v) => !v)}>
              {open ? 'Hide texts' : 'Show texts'}
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="pl-button ai-small-button"
              disabled={run.isPending || !status.available}
              onClick={start}
              title={GENERATION_HINT}
              style={{ borderColor: colors.border, color: colors.text }}
            >
              {run.isPending && <Loader2 size={12} className="animate-spin" />}
              {run.isPending ? 'Testing' : `Test${status.generation_test_price != null ? ` · ${usd(status.generation_test_price)}` : ''}`}
            </button>
            {error && <span className="ai-stat" style={{ color: colors.warning }}>{error}</span>}
          </>
        )}
      </div>
      {test && open && (
        <div className="ai-generation-texts">
          <div className="pl-note" title="Written to make the feature fire">Written to fire</div>
          {firing.map((t, i) => (
            <ContextLine key={`f${i}`} context={asContext(t)} colors={colors} tag={t.fired ? 'fires' : 'silent'} />
          ))}
          <div className="pl-note" style={{ marginTop: '0.6rem' }} title="Close in topic and wording, but lacking what the explanation names">
            Near-misses
          </div>
          {near.map((t, i) => (
            <ContextLine key={`n${i}`} context={asContext(t)} colors={colors} tag={t.fired ? 'fires' : 'quiet'} />
          ))}
        </div>
      )}
    </div>
  )
}

export function HypothesisPanel({
  atlasRef,
  feature,
  explanation,
  hypotheses,
  tests,
  status,
  selected,
  onSelect,
  onShowPurchaseCredits,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number
  /** The autointerp explanation shown, to compare with */
  explanation: Explanation | null
  hypotheses: Hypothesis[]
  tests: GenerationTest[]
  status: AutointerpStatus
  /** The hypothesis whose details and mistakes are shown */
  selected: number | null
  onSelect: (id: number | null) => void
  onShowPurchaseCredits?: () => void
}) {
  const colors = useLabColors()
  const [text, setText] = useState('')
  const score = useScoreHypothesis()
  const remove = useDeleteHypothesis()
  const featureRef = { ...atlasRef, feature }
  const error = creditsError(score.error ? (score.error as Error).message : null)
  const trimmed = text.trim()

  const submit = () => {
    if (!trimmed || score.isPending) return
    score.mutate(
      { ...featureRef, text: trimmed },
      {
        onSuccess: (result) => {
          setText('')
          onSelect(result.hypothesis.id)
        },
        onError: (err) => {
          if ((err as Error).message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
        },
      },
    )
  }

  const baseline = explanation?.token_detection ?? hypotheses.find((h) => h.token_detection != null)?.token_detection ?? null
  const generation = (source: ExplanationSource) => {
    const test = testOf(tests, source)
    return test ? `${percent(test.firing_rate)} · ${percent(test.near_miss_rate)}` : '–'
  }
  const header = { background: colors.surface, borderBottom: `1px solid ${colors.border}` }
  const reference = { borderTop: `1px solid ${colors.grid}`, cursor: 'default', opacity: 0.85 } as const

  return (
    <div className="ai-section">
      <div className="ai-section-head">
        <span className="pl-control-label" title="Your own explanation, scored by the same scorer on the same held-out contexts as the autointerp explanation. Private to you.">
          Hypotheses
        </span>
      </div>
      <div className="ai-hypothesis-form">
        <input
          className="pl-input"
          value={text}
          maxLength={MAX_HYPOTHESIS_CHARS}
          onChange={(ev) => setText(ev.target.value)}
          onKeyDown={(ev) => ev.key === 'Enter' && submit()}
          placeholder="What do you think it fires on?"
          aria-label="Your explanation"
          style={{ borderColor: colors.border, background: 'transparent', color: colors.text }}
        />
        <button
          type="button"
          className="pl-button"
          disabled={!trimmed || score.isPending || !status.available}
          onClick={submit}
          title={`${modelName(status.scorer)} scores it on the feature’s held-out contexts. Charged its usage.`}
          style={{ borderColor: colors.border, color: colors.text }}
        >
          {score.isPending && <Loader2 size={13} className="animate-spin" />}
          {score.isPending ? 'Scoring' : `Score · ${usd(status.hypothesis_price)}`}
        </button>
      </div>
      {error && <div className="pl-error-note">{error}</div>}

      {(hypotheses.length > 0 || explanation?.status === 'scored') && (
        <table className="pl-table ai-hypothesis-table">
          <thead>
            <tr>
              <th style={{ ...header, textAlign: 'left' }}>Explanation</th>
              <th style={header} title="Mean of detection and fuzzing (0.5 is chance)">Score</th>
              <th style={header} title="Detection: telling held-out contexts where it fires from ones where it doesn’t">Det.</th>
              <th style={header} title="Fuzzing: telling its true firing tokens from as many random ones">Fuzz.</th>
              <th style={header} title="Generation test: texts written to fire that do · near-misses that stay quiet">Gen.</th>
              <th style={header} />
            </tr>
          </thead>
          <tbody>
            {explanation?.status === 'scored' && (
              <tr style={reference}>
                <td className="ai-explanation-cell" title={explanation.explanation ?? undefined}>
                  <span className="ai-row-tag">Autointerp</span>
                  {explanation.explanation}
                </td>
                <td>{formatScore(explanation.score)}</td>
                <td>{formatScore(explanation.detection)}</td>
                <td>{formatScore(explanation.fuzzing)}</td>
                <td>{generation({ explanation_id: explanation.id })}</td>
                <td />
              </tr>
            )}
            {baseline != null && (
              <tr style={reference}>
                <td className="ai-explanation-cell" title="Its top 3 firing tokens as the explanation, scored by detection on the same contexts">
                  <span className="ai-row-tag">Token baseline</span>
                </td>
                <td>–</td>
                <td>{formatScore(baseline)}</td>
                <td>–</td>
                <td>–</td>
                <td />
              </tr>
            )}
            {hypotheses.map((h) => (
              <tr
                key={h.id}
                onClick={() => onSelect(h.id === selected ? null : h.id)}
                style={{ borderTop: `1px solid ${colors.grid}`, background: h.id === selected ? colors.hover : undefined }}
              >
                <td className="ai-explanation-cell" title={h.status === 'scored' ? h.text : `${h.text}\n${h.note ?? h.status}`}>
                  <span className="ai-row-tag" style={{ color: colors.accent }}>Yours</span>
                  {h.text}
                </td>
                <td style={{ fontWeight: 600 }}>{h.status === 'scored' ? formatScore(h.score) : h.status}</td>
                <td>{formatScore(h.detection)}</td>
                <td>{formatScore(h.fuzzing)}</td>
                <td>{generation({ hypothesis_id: h.id })}</td>
                <td>
                  <button
                    type="button"
                    className="pl-icon-button"
                    title="Delete"
                    aria-label="Delete hypothesis"
                    disabled={remove.isPending}
                    onClick={(ev) => {
                      ev.stopPropagation()
                      if (h.id === selected) onSelect(null)
                      remove.mutate({ ref: featureRef, id: h.id })
                    }}
                    style={{ color: colors.text }}
                  >
                    <Trash2 className="pl-icon" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export function Versions({
  atlasRef,
  feature,
  versions,
  viewing,
  onView,
  status,
  onShowPurchaseCredits,
}: {
  atlasRef: AutointerpAtlasRef
  feature: number
  /** Every explanation, newest first */
  versions: Explanation[]
  /** The version in view */
  viewing: number | null
  onView: (id: number | null) => void
  status: AutointerpStatus
  onShowPurchaseCredits?: () => void
}) {
  const colors = useLabColors()
  const explain = useExplainFeature()
  const [explainer, setExplainer] = useState<string | null>(null)
  const [scorer, setScorer] = useState<string | null>(null)
  const models = status.models.map((m) => ({ value: m.id, label: m.name }))
  const pair = { explainer: explainer ?? status.explainer, scorer: scorer ?? status.scorer }
  const stored = versions.some(
    (v) => v.explainer === pair.explainer && v.scorer === pair.scorer && v.prompt_version === status.prompt_version && v.examples_version === status.examples_version,
  )
  const error = creditsError(explain.error ? (explain.error as Error).message : null)
  const run = () =>
    explain.mutate(
      { ...atlasRef, feature, ...pair },
      {
        onSuccess: () => onView(null),
        onError: (err) => {
          if ((err as Error).message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
        },
      },
    )

  return (
    <div className="ai-section">
      <div className="ai-section-head">
        <span className="pl-control-label" title="Every explanation is kept. The newest is the one shown, never the best-scoring: choosing by score on the contexts the score is reported on would game it.">
          Explanations · {versions.length}
        </span>
      </div>
      <div className="ai-reexplain">
        <div className="ai-models">
          <div className="pl-control-label">Explainer</div>
          <Dropdown value={pair.explainer} onChange={setExplainer} options={models} />
        </div>
        <div className="ai-models">
          <div className="pl-control-label">Scorer</div>
          <Dropdown value={pair.scorer} onChange={setScorer} options={models} />
        </div>
        <button
          type="button"
          className="pl-button"
          disabled={stored || explain.isPending || !status.available}
          onClick={run}
          title={stored ? 'Already explained with these models' : 'Explain and score it again with these models; the new explanation becomes the one shown'}
          style={{ borderColor: colors.border, color: colors.text }}
        >
          {explain.isPending && <Loader2 size={13} className="animate-spin" />}
          {explain.isPending ? 'Explaining' : stored ? 'Explained' : `Re-explain · ${usd(status.explain_price)}`}
        </button>
      </div>
      {error && <div className="pl-error-note">{error}</div>}
      {versions.length > 1 && (
        <table className="pl-table ai-hypothesis-table">
          <tbody>
            {versions.map((v, i) => {
              const active = viewing === v.id || (viewing == null && i === 0)
              return (
                <tr key={v.id} onClick={() => onView(i === 0 ? null : v.id)} style={{ borderTop: `1px solid ${colors.grid}`, background: active ? colors.hover : undefined }}>
                  <td className="ai-explanation-cell" title={v.explanation ?? v.note ?? undefined}>
                    <span className="ai-row-tag">{i === 0 ? 'Shown' : formatDate(v.created_at)}</span>
                    {v.explanation ?? <span style={{ opacity: 0.6 }}>{v.note ?? v.status}</span>}
                  </td>
                  <td style={{ fontWeight: 600 }}>{v.status === 'scored' ? formatScore(v.score) : v.status}</td>
                  <td style={{ opacity: 0.75 }}>
                    {modelName(v.explainer).replace('Claude ', '')} / {modelName(v.scorer).replace('Claude ', '')}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
