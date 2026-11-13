/**
 * The Autointerp tab's Features view: an atlas's explained features (each one's newest
 * explanation) with their scores, sorted and filtered (score at least, status, token features)
 * and searched (Postgres full-text search of the explanations), a page at a time, with exports.
 */

import { useEffect, useState } from 'react'
import { ArrowDown, ArrowUp, Download, ListTree } from 'lucide-react'
import { apiClient } from '../../api/client'
import { useExplainedFeatures, type AutointerpAtlasRef } from '../../hooks/useAutointerp'
import type { FeatureRow, FeatureSort } from '../../types/autointerp'
import { Placeholder, Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { formatScore } from './labels'

const PAGE = 100
const SEARCH_DEBOUNCE_MS = 300

type StatusFilter = 'any' | 'scored' | 'unscored' | 'refused'
type TokenFilter = 'any' | 'exclude' | 'only'

const COLUMNS: { key: FeatureSort | null; label: string; title?: string; align?: 'left' }[] = [
  { key: 'feature', label: 'ID' },
  { key: null, label: 'Explanation', align: 'left' },
  { key: 'score', label: 'Score', title: 'Mean of detection and fuzzing (0.5 is chance)' },
  { key: 'detection', label: 'Det.', title: 'Detection: telling held-out contexts where it fires from ones where it doesn’t' },
  { key: 'fuzzing', label: 'Fuzz.', title: 'Fuzzing: telling its true firing tokens from as many random ones' },
  { key: 'delta', label: 'Δ token', title: 'Detection minus the token baseline’s (its top 3 firing tokens as the explanation)' },
  { key: 'freq', label: 'Freq', title: 'Share of corpus tokens it’s active on' },
  { key: null, label: 'Tag' },
]

function formatFreq(freq: number | null): string {
  if (freq == null) return '–'
  return freq >= 0.001 ? `${(freq * 100).toFixed(2)}%` : `${(freq * 1e6).toFixed(0)}/M`
}

function formatDelta(delta: number | null): string {
  return delta == null ? '–' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}`
}

function tag(row: FeatureRow): { text: string; title?: string } {
  if (row.status === 'refused') return { text: 'refused', title: row.note ?? undefined }
  if (row.status === 'unscored') return { text: 'unscored', title: row.note ?? undefined }
  if (row.token_feature) return { text: 'token', title: 'Doesn’t beat its top tokens by much: what it fires on is mostly which token it is' }
  return { text: '' }
}

export function ExplanationTable({
  atlasRef,
  fileName,
  selected,
  onSelect,
}: {
  atlasRef: AutointerpAtlasRef
  /** Exports' file name, without extension */
  fileName: string
  selected: number | null
  onSelect: (feature: number) => void
}) {
  const colors = useLabColors()
  const [sort, setSort] = useState<FeatureSort>('score')
  const [descending, setDescending] = useState(true)
  const [status, setStatus] = useState<StatusFilter>('any')
  const [tokens, setTokens] = useState<TokenFilter>('any')
  const [minScore, setMinScore] = useState('')
  const [searchText, setSearchText] = useState('')
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [exportError, setExportError] = useState<string | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => setSearch(searchText.trim()), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [searchText])
  useEffect(() => setOffset(0), [sort, descending, status, tokens, minScore, search])

  const min = minScore.trim() === '' ? null : Math.min(1, Math.max(0, Number(minScore)))
  const query = useExplainedFeatures(atlasRef, {
    sort,
    descending,
    offset,
    limit: PAGE,
    min_score: min != null && !Number.isNaN(min) ? min : null,
    status: status === 'any' ? null : status,
    token_features: tokens === 'any' ? null : tokens,
    search: search || null,
  })
  const data = query.data

  const sortBy = (key: FeatureSort) => {
    if (key === sort) setDescending((d) => !d)
    else {
      setSort(key)
      setDescending(key !== 'feature')
    }
  }

  const download = (format: 'csv' | 'json') => {
    setExportError(null)
    apiClient.exportExplanations({ ...atlasRef, format }, fileName).catch((err: Error) => setExportError(err.message))
  }

  if (data && data.explained === 0) {
    return <Placeholder icon={ListTree}>No explanations on this atlas yet.</Placeholder>
  }

  return (
    <div className="ai-table-view">
      <div className="pl-controls-row ai-filters">
        <div style={{ flex: '1 1 220px' }}>
          <div className="pl-control-label">Search</div>
          <input
            className="pl-input"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Explanations"
            style={{ borderColor: colors.border, background: 'transparent', color: colors.text }}
          />
        </div>
        <div>
          <div className="pl-control-label">Status</div>
          <Segmented
            ariaLabel="Status"
            value={status}
            onChange={setStatus}
            options={[
              { value: 'any', label: 'All' },
              { value: 'scored', label: 'Scored' },
              { value: 'unscored', label: 'Unscored' },
              { value: 'refused', label: 'Refused' },
            ]}
          />
        </div>
        <div>
          <div className="pl-control-label" title="Token features: explanations that don’t beat their top tokens by much">
            Token features
          </div>
          <Segmented
            ariaLabel="Token features"
            value={tokens}
            onChange={setTokens}
            options={[
              { value: 'any', label: 'All' },
              { value: 'exclude', label: 'Hide' },
              { value: 'only', label: 'Only' },
            ]}
          />
        </div>
        <div style={{ width: 90 }}>
          <div className="pl-control-label">Score ≥</div>
          <input
            className="pl-input"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minScore}
            onChange={(e) => setMinScore(e.target.value)}
            placeholder="0.5"
            style={{ borderColor: colors.border, background: 'transparent', color: colors.text }}
          />
        </div>
        <div className="ai-export">
          <button type="button" className="pl-button" onClick={() => download('csv')} style={{ borderColor: colors.border, color: colors.text }} title="Every explanation and its scores, with the atlas, models and prompt version">
            <Download className="pl-icon" aria-hidden="true" /> CSV
          </button>
          <button type="button" className="pl-button" onClick={() => download('json')} style={{ borderColor: colors.border, color: colors.text }}>
            <Download className="pl-icon" aria-hidden="true" /> JSON
          </button>
        </div>
      </div>
      {exportError && <div className="pl-error-note">{exportError}</div>}

      <div className="ai-table-wrap">
        <table className="pl-table ai-table">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.label}
                  title={c.title}
                  style={{ background: colors.surface, borderBottom: `1px solid ${colors.border}`, textAlign: c.align ?? (c.label === 'ID' ? 'left' : 'right') }}
                >
                  {c.key ? (
                    <button type="button" className="ai-sort" onClick={() => sortBy(c.key as FeatureSort)} style={{ color: colors.text }}>
                      {c.label}
                      {sort === c.key && (descending ? <ArrowDown size={11} aria-label="descending" /> : <ArrowUp size={11} aria-label="ascending" />)}
                    </button>
                  ) : (
                    c.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data?.rows.map((row) => {
              const t = tag(row)
              return (
                <tr
                  key={row.feature}
                  onClick={() => onSelect(row.feature)}
                  style={{ background: row.feature === selected ? colors.hover : undefined, borderTop: `1px solid ${colors.grid}` }}
                >
                  <td style={{ color: colors.accent, fontWeight: 600 }}>#{row.feature}</td>
                  <td className="ai-explanation-cell" title={row.explanation ?? row.note ?? undefined}>
                    {row.explanation ?? <span style={{ opacity: 0.6 }}>{row.note ?? '–'}</span>}
                  </td>
                  <td>{formatScore(row.score)}</td>
                  <td title={row.detection_hard != null ? `Against same-token contexts: ${formatScore(row.detection_hard)}` : undefined}>{formatScore(row.detection)}</td>
                  <td>{formatScore(row.fuzzing)}</td>
                  <td>{formatDelta(row.delta)}</td>
                  <td>{formatFreq(row.freq)}</td>
                  <td title={t.title} style={{ opacity: 0.75 }}>{t.text}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="ai-pager">
        <span className="pl-note">
          {data
            ? data.total === 0
              ? 'No explanations match.'
              : `${offset + 1}–${Math.min(offset + PAGE, data.total)} of ${data.total.toLocaleString()}${data.total !== data.explained ? ` (${data.explained.toLocaleString()} explained)` : ''}`
            : query.isLoading
              ? 'Loading…'
              : (query.error as Error | null)?.message ?? ''}
        </span>
        <span style={{ display: 'inline-flex', gap: 6 }}>
          <button type="button" className="pl-button" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE))} style={{ borderColor: colors.border, color: colors.text }}>
            Previous
          </button>
          <button
            type="button"
            className="pl-button"
            disabled={!data || offset + PAGE >= data.total}
            onClick={() => setOffset((o) => o + PAGE)}
            style={{ borderColor: colors.border, color: colors.text }}
          >
            Next
          </button>
        </span>
      </div>
    </div>
  )
}
