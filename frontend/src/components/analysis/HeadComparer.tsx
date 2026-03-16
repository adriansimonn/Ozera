/**
 * HeadComparer component for side-by-side comparison of attention patterns.
 * Theme-aware: adapts to light/dark mode and glass interface.
 */

import { useTheme, useThemeColors } from '../../hooks/useTheme'
import type { CompareAttentionResponse } from '../../types/analysis'

interface HeadComparerProps {
  comparison: CompareAttentionResponse
  onHeadSelect?: (layer: number, head: number) => void
}

function similarityColor(value: number, isLight: boolean) {
  if (value > 0.7) return isLight ? '#16a34a' : '#4ade80'
  if (value > 0.4) return isLight ? '#ca8a04' : '#facc15'
  return isLight ? '#dc2626' : '#f87171'
}

function similarityBarBg(value: number) {
  if (value > 0.7) return '#22c55e'
  if (value > 0.4) return '#eab308'
  return '#ef4444'
}

export function HeadComparer({ comparison, onHeadSelect }: HeadComparerProps) {
  const { isGlass, isLight } = useTheme()
  const tc = useThemeColors()

  const {
    prompt1,
    prompt2,
    layer_similarities,
    head_differences,
    common_patterns,
    divergent_patterns,
    length_warning,
    comparison_method,
  } = comparison

  const overallSimilarity = layer_similarities.length > 0
    ? layer_similarities.reduce((sum, l) => sum + l.mean_similarity, 0) / layer_similarities.length
    : 0

  const card: React.CSSProperties = {
    background: tc.surface,
    border: `1px solid ${tc.border}`,
    padding: '1rem',
  }

  const glassClass = isGlass ? ' glass' : ''

  const warningBg = isLight ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.08)'
  const warningBorder = isLight ? 'rgba(245,158,11,0.2)' : 'rgba(245,158,11,0.3)'
  const warningText = isLight ? '#b45309' : '#fbbf24'
  const warningTextSub = isLight ? 'rgba(180,120,0,0.7)' : 'rgba(251,191,36,0.6)'

  const greenAccent = isLight ? '#16a34a' : '#4ade80'
  const greenBorder = isLight ? 'rgba(22,163,74,0.2)' : 'rgba(74,222,128,0.25)'
  const redAccent = isLight ? '#dc2626' : '#f87171'
  const redBorder = isLight ? 'rgba(220,38,38,0.2)' : 'rgba(248,113,113,0.25)'
  const redBg = isLight ? 'rgba(220,38,38,0.04)' : 'rgba(248,113,113,0.06)'

  const trackBg = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'

  return (
    <div className="hc-root">
      {/* Length Mismatch Warning */}
      {length_warning && (
        <div
          className={glassClass}
          style={{
            background: warningBg,
            border: `1px solid ${warningBorder}`,
            padding: '1rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
            <span style={{ color: warningText, fontSize: '1rem', fontWeight: 700, lineHeight: 1 }}>!</span>
            <div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: warningText, marginBottom: '0.25rem' }}>
                Sequence Length Mismatch
              </div>
              <div style={{ fontSize: '0.8rem', color: warningTextSub }}>
                {length_warning}
              </div>
              <div style={{ fontSize: '0.7rem', color: warningTextSub, marginTop: '0.5rem' }}>
                Comparison method: <span style={{ fontFamily: 'monospace' }}>{comparison_method}</span>
                {comparison_method === 'statistical' && (
                  <span> (compares attention statistics instead of raw matrices)</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Prompts being compared */}
      <div className="hc-prompts-grid">
        {[
          { label: 'Prompt 1', text: prompt1 },
          { label: 'Prompt 2', text: prompt2 },
        ].map(({ label, text }) => (
          <div key={label} className={glassClass} style={card}>
            <div style={{ fontSize: '0.65rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: tc.textSub, marginBottom: '0.375rem' }}>
              {label}
            </div>
            <div style={{ fontSize: '0.8rem', color: tc.text, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {text}
            </div>
          </div>
        ))}
      </div>

      {/* Overall Similarity Score */}
      <div className={glassClass} style={card}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: tc.textMid }}>Overall Similarity</div>
            {comparison_method === 'statistical' && (
              <div style={{ fontSize: '0.65rem', color: tc.textSub }}>(statistical comparison)</div>
            )}
          </div>
          <span style={{ fontSize: '1.5rem', fontWeight: 700, color: similarityColor(overallSimilarity, isLight) }}>
            {(overallSimilarity * 100).toFixed(1)}%
          </span>
        </div>
        <div style={{ height: 10, background: trackBg, borderRadius: 5, overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              borderRadius: 5,
              background: similarityBarBg(overallSimilarity),
              width: `${overallSimilarity * 100}%`,
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      </div>

      {/* Layer-by-Layer Similarity */}
      <div className={glassClass} style={card}>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: tc.textMid, marginBottom: '1rem' }}>
          Layer Similarity
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {layer_similarities.map((layer) => (
            <div key={layer.layer} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.7rem', color: tc.textSub, width: 52, flexShrink: 0, fontFamily: 'monospace' }}>
                L{layer.layer}
              </span>
              <div style={{ flex: 1, height: 6, background: trackBg, borderRadius: 3, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    borderRadius: 3,
                    background: similarityBarBg(layer.mean_similarity),
                    width: `${layer.mean_similarity * 100}%`,
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
              <span style={{ fontSize: '0.7rem', color: tc.textMid, width: 36, textAlign: 'right', fontFamily: 'monospace' }}>
                {(layer.mean_similarity * 100).toFixed(0)}%
              </span>
              <span style={{ fontSize: '0.65rem', color: tc.textFaint, width: 64 }}>
                ({(layer.min_similarity * 100).toFixed(0)}-{(layer.max_similarity * 100).toFixed(0)}%)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Most Different Heads */}
      {head_differences.length > 0 && (
        <div className={glassClass} style={card}>
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: tc.textMid, marginBottom: '1rem' }}>
            Most Different Heads
          </div>
          <div className="hc-diff-grid">
            {head_differences.slice(0, 10).map((diff) => (
              <button
                key={`${diff.layer}-${diff.head}`}
                onClick={() => onHeadSelect?.(diff.layer, diff.head)}
                className="hc-diff-item"
                style={{
                  background: redBg,
                  border: `1px solid ${redBorder}`,
                }}
              >
                <div style={{ fontSize: '0.65rem', color: tc.textSub, fontFamily: 'monospace' }}>
                  L{diff.layer}H{diff.head}
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: redAccent, marginTop: '0.125rem' }}>
                  {(diff.difference * 100).toFixed(0)}%
                </div>
                <div style={{ fontSize: '0.6rem', color: tc.textFaint }}>different</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pattern Analysis */}
      <div className="hc-patterns-grid">
        {common_patterns.length > 0 && (
          <div className={glassClass} style={{ ...card, borderColor: greenBorder }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: greenAccent, marginBottom: '0.75rem' }}>
              Common Patterns
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {common_patterns.map((pattern, idx) => (
                <div key={idx} style={{ fontSize: '0.8rem', color: tc.textMid, display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                  <span style={{ color: greenAccent, flexShrink: 0, marginTop: 1 }}>&#10003;</span>
                  <span>{pattern}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {divergent_patterns.length > 0 && (
          <div className={glassClass} style={{ ...card, borderColor: redBorder }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: redAccent, marginBottom: '0.75rem' }}>
              Divergent Patterns
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {divergent_patterns.map((pattern, idx) => (
                <div key={idx} style={{ fontSize: '0.8rem', color: tc.textMid, display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                  <span style={{ color: redAccent, flexShrink: 0, marginTop: 1 }}>&#10007;</span>
                  <span>{pattern}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* No significant differences */}
      {head_differences.length === 0 && (
        <div className={glassClass} style={{ ...card, borderColor: greenBorder, textAlign: 'center', padding: '1.5rem' }}>
          <div style={{ fontSize: '1rem', fontWeight: 600, color: greenAccent, marginBottom: '0.5rem' }}>
            High Similarity Detected
          </div>
          <div style={{ fontSize: '0.8rem', color: tc.textMid }}>
            Attention patterns are very similar between these two prompts.
            This suggests the model processes them in similar ways.
          </div>
        </div>
      )}

      <style>{`
        .hc-root {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          padding: 1rem 1.25rem;
        }
        .hc-prompts-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.75rem;
        }
        .hc-diff-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 0.5rem;
        }
        .hc-diff-item {
          padding: 0.625rem 0.5rem;
          cursor: pointer;
          transition: opacity 0.15s;
          text-align: center;
          background: transparent;
        }
        .hc-diff-item:hover {
          opacity: 0.75;
        }
        .hc-patterns-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.75rem;
        }
        @media (max-width: 800px) {
          .hc-prompts-grid,
          .hc-patterns-grid {
            grid-template-columns: 1fr;
          }
          .hc-diff-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }
      `}</style>
    </div>
  )
}
