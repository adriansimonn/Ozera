/**
 * Token heatmap: paste any text and see a probe's score at every token, with the tokens the
 * probe pools over (its read span) marked and its pooled score for the whole text.
 */

import { useState } from 'react'
import { ScanText } from 'lucide-react'
import type { Pooling, ProbeMethod, ProbeRunResult, SavedProbe, ScoreProbeResponse } from '../../types/probes'
import { apiClient } from '../../api/client'
import { Dropdown } from '../common/Dropdown'
import { Placeholder, RunBar } from './chartKit'
import { divergingColor, inkOn, useProbeColors } from './probeTheme'
import { METHOD_SHORT, POOLING_LABELS, formatScore, inlineProbe } from './probeUtils'

/** "sweep" for the probe selected in the layer sweep, or "saved:<id>" */
export type HeatmapSource = string

interface TokenHeatmapProps {
  run: ProbeRunResult | null
  pooling: Pooling
  method: ProbeMethod
  position: number
  savedProbes: SavedProbe[]
  source: HeatmapSource
  onSourceChange: (source: HeatmapSource) => void
  onShowPurchaseCredits?: () => void
}

interface ScoredView {
  response: ScoreProbeResponse
  scale: number
  labelNames: [string, string]
  description: string
}

function labelNamesOf(dataset: Record<string, unknown>): [string, string] {
  const names = dataset.label_names
  return Array.isArray(names) && names.length === 2 ? [String(names[0]), String(names[1])] : ['Negative', 'Positive']
}

function visibleToken(token: string): string {
  return token.replace(/\n/g, '↵\n').replace(/\t/g, '→\t')
}

export function TokenHeatmap({
  run,
  pooling,
  method,
  position,
  savedProbes,
  source,
  onSourceChange,
  onShowPurchaseCredits,
}: TokenHeatmapProps) {
  const colors = useProbeColors()
  const [text, setText] = useState('')
  const [scoring, setScoring] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<ScoredView | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  const sweepAvailable = run != null && position > 0
  const savedId = source.startsWith('saved:') ? Number(source.slice(6)) : null
  const saved = savedProbes.find((p) => p.id === savedId) ?? null

  const options = [
    ...(sweepAvailable
      ? [
          {
            value: 'sweep',
            label: `Sweep selection · ${run!.model} L${position - 1} ${POOLING_LABELS[pooling].toLowerCase()} ${METHOD_SHORT[method]}`,
          },
        ]
      : []),
    ...savedProbes.map((p) => ({
      value: `saved:${p.id}`,
      label: `${p.name} · ${p.model_id} L${p.layer}`,
      disabled: !p.model_available,
    })),
  ]
  const ready = (source === 'sweep' && sweepAvailable) || saved != null

  const handleScore = async () => {
    if (!text.trim() || !ready) return
    setScoring(true)
    setError(null)
    try {
      if (source === 'sweep' && run) {
        const stds = run.poolings[pooling].methods[method].metrics.score_std
        const response = await apiClient.scoreProbe({
          model: run.model,
          texts: [text],
          probe: inlineProbe(run, pooling, method, position),
        })
        setView({
          response,
          scale: 2 * (stds?.[position] ?? 0),
          labelNames: run.dataset.label_names,
          description: `${run.model}, layer ${position - 1}, ${POOLING_LABELS[pooling].toLowerCase()} pooling, ${METHOD_SHORT[method]}`,
        })
      } else if (saved) {
        const response = await apiClient.scoreProbe({ probe_id: saved.id, texts: [text] })
        setView({
          response,
          scale: 2 * (saved.normalization.score_std ?? 0),
          labelNames: labelNamesOf(saved.dataset),
          description: `${saved.name} (${saved.model_id}, layer ${saved.layer})`,
        })
      }
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Scoring failed')
    } finally {
      setScoring(false)
    }
  }

  const result = view?.response.results[0]
  const scale = result ? view!.scale || Math.max(...result.scores.map(Math.abs), 1e-6) : 1

  return (
    <div className="pl-heatmap">
      <div className="pl-card-title">
        <h3>Token heatmap</h3>
      </div>

      {options.length === 0 ? (
        <Placeholder icon={ScanText}>Train or save a probe to score text with it.</Placeholder>
      ) : (
        <>
          <div className="pl-control-label">Probe</div>
          <Dropdown value={ready ? source : ''} onChange={onSourceChange} options={options} placeholder="Choose a probe" />
          {saved?.model_changed && (
            <div className="pl-note" style={{ color: colors.warning, marginTop: '0.4rem' }}>
              Model changed since this probe was saved
            </div>
          )}
          <textarea
            className="pl-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Text to score…"
            maxLength={2000}
            rows={3}
            style={{ marginTop: '0.625rem', borderColor: colors.border, background: colors.panel, color: colors.text }}
          />
          <RunBar label="Score text" running={scoring} disabled={!text.trim() || !ready} onRun={handleScore} />
        </>
      )}

      {error && <div className="pl-error-note">{error}</div>}

      {result && view && (
        <div className="pl-result" style={{ borderColor: colors.border }}>
          <div className="pl-heatmap-verdict">
            <span className="pl-legend-dot" style={{ background: colors.classes[result.score > 0 ? 1 : 0] }} />
            <span>
              <strong>{view.labelNames[result.score > 0 ? 1 : 0]}</strong> · {formatScore(result.score)}
            </span>
            <span className="pl-note">{view.description}</span>
          </div>

          <div className="pl-heatmap-readout" aria-live="polite">
            {hovered != null
              ? `“${result.tokens[hovered]}” · ${formatScore(result.scores[hovered])}${
                  hovered >= result.span[0] && hovered < result.span[1] ? '' : ' · not pooled'
                }`
              : 'Hover a token for its score. Faded tokens aren’t pooled.'}
          </div>

          <div className="pl-tokens">
            {result.tokens.map((token, i) => {
              const inSpan = i >= result.span[0] && i < result.span[1]
              const background = divergingColor(colors, result.scores[i] / scale)
              return (
                <span
                  key={i}
                  className="pl-token"
                  title={`${formatScore(result.scores[i])}`}
                  onPointerEnter={() => setHovered(i)}
                  onPointerLeave={() => setHovered(null)}
                  style={{
                    background,
                    color: inkOn(background),
                    opacity: inSpan ? 1 : 0.4,
                    outline: hovered === i ? `1.5px solid ${colors.text}` : undefined,
                  }}
                >
                  {visibleToken(token)}
                </span>
              )
            })}
          </div>

          <div className="pl-scale">
            <span>{view.labelNames[0]}</span>
            <span
              className="pl-scale-bar"
              style={{
                background: `linear-gradient(to right, ${divergingColor(colors, -1)}, ${divergingColor(colors, 0)}, ${divergingColor(colors, 1)})`,
              }}
            />
            <span>{view.labelNames[1]}</span>
            <span className="pl-note" title={view.scale ? '2 standard deviations of training scores' : 'Largest score'}>
              ±{formatScore(scale)}
            </span>
          </div>
        </div>
      )}

      <style>{`
        .probe-lab .pl-heatmap-verdict { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
        .probe-lab .pl-heatmap-readout { font-size: 0.75rem; margin: 0.5rem 0; min-height: 1.1rem; font-variant-numeric: tabular-nums; }
        .probe-lab .pl-tokens {
          font-family: 'JetBrains Mono', ui-monospace, monospace;
          font-size: 0.82rem;
          line-height: 1.9;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .probe-lab .pl-scale { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.75rem; font-size: 0.75rem; }
        .probe-lab .pl-scale-bar { width: 140px; height: 8px; border-radius: 4px; }
      `}</style>
    </div>
  )
}
