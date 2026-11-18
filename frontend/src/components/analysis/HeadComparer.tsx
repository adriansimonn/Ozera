/**
 * HeadComparer component for side-by-side comparison of attention patterns.
 * Colors come from the app's custom properties (styles/app.css), so it follows the theme.
 */

import type { CompareAttentionResponse } from '../../types/analysis'

interface HeadComparerProps {
  comparison: CompareAttentionResponse
  onHeadSelect?: (layer: number, head: number) => void
}

export function HeadComparer({ comparison, onHeadSelect }: HeadComparerProps) {
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

  return (
    <div className="hc-root">
      {/* Length Mismatch Warning */}
      {length_warning && (
        <div className="oz-note oz-note--warn">
          <div>
            <div className="hc-note-title">Sequence length mismatch</div>
            <div>{length_warning}</div>
            <div className="oz-hint">
              Comparison method: <span className="oz-mono">{comparison_method}</span>
              {comparison_method === 'statistical' && (
                <span> (compares attention statistics instead of raw matrices)</span>
              )}
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
          <div key={label} className="hc-prompt">
            <div className="hc-label">{label}</div>
            <div className="hc-prompt-text">{text}</div>
          </div>
        ))}
      </div>

      {/* Overall Similarity Score */}
      <div className="hc-overall">
        <div className="hc-label">
          Overall similarity
          {comparison_method === 'statistical' && <span> (statistical comparison)</span>}
        </div>
        <div className="hc-overall-value">{(overallSimilarity * 100).toFixed(1)}%</div>
        <div className="hc-track">
          <div className="hc-bar" style={{ width: `${overallSimilarity * 100}%` }} />
        </div>
      </div>

      {/* Layer-by-Layer Similarity */}
      <div>
        <h3 className="hc-heading">Layer similarity</h3>
        <div className="hc-layers">
          {layer_similarities.map((layer) => (
            <div key={layer.layer} className="hc-layer">
              <span className="hc-layer-name">L{layer.layer}</span>
              <div className="hc-track">
                <div className="hc-bar" style={{ width: `${layer.mean_similarity * 100}%` }} />
              </div>
              <span className="hc-layer-value">{(layer.mean_similarity * 100).toFixed(0)}%</span>
              <span className="hc-layer-range">
                {(layer.min_similarity * 100).toFixed(0)}–{(layer.max_similarity * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Most Different Heads */}
      {head_differences.length > 0 && (
        <div>
          <h3 className="hc-heading">Most different heads</h3>
          <div className="hc-diff-grid">
            {head_differences.slice(0, 10).map((diff) => (
              <button
                key={`${diff.layer}-${diff.head}`}
                onClick={() => onHeadSelect?.(diff.layer, diff.head)}
                className="hc-diff-item"
              >
                <div className="hc-diff-name">L{diff.layer}H{diff.head}</div>
                <div className="hc-diff-value">{(diff.difference * 100).toFixed(0)}%</div>
                <div className="hc-diff-note">different</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pattern Analysis */}
      {(common_patterns.length > 0 || divergent_patterns.length > 0) && (
        <div className="hc-patterns-grid">
          {common_patterns.length > 0 && (
            <div>
              <h3 className="hc-heading">Common patterns</h3>
              <ul className="hc-patterns">
                {common_patterns.map((pattern, idx) => (
                  <li key={idx}>
                    <span className="hc-mark hc-mark--same" aria-hidden="true">&#10003;</span>
                    <span>{pattern}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {divergent_patterns.length > 0 && (
            <div>
              <h3 className="hc-heading">Divergent patterns</h3>
              <ul className="hc-patterns">
                {divergent_patterns.map((pattern, idx) => (
                  <li key={idx}>
                    <span className="hc-mark hc-mark--different" aria-hidden="true">&#10007;</span>
                    <span>{pattern}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* No significant differences */}
      {head_differences.length === 0 && (
        <div className="oz-note oz-note--ok">
          <div>
            <div className="hc-note-title">High similarity detected</div>
            <div>Attention patterns are very similar between these two prompts.</div>
          </div>
        </div>
      )}

      <style>{`
        .hc-root {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        .hc-root .hc-note-title {
          font-weight: 500;
        }
        .hc-root .hc-label {
          margin-bottom: 0.35rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }
        .hc-root .hc-heading {
          margin: 0 0 1rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }
        /* The two prompts, with a rule between */
        .hc-root .hc-prompts-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          border-top: 1px solid var(--oz-rule);
          border-bottom: 1px solid var(--oz-rule);
        }
        .hc-root .hc-prompt {
          min-width: 0;
          padding: 0.9rem 1.25rem 1rem 0;
        }
        .hc-root .hc-prompt + .hc-prompt {
          padding-left: 1.25rem;
          padding-right: 0;
          border-left: 1px solid var(--oz-rule);
        }
        .hc-root .hc-prompt-text {
          overflow: hidden;
          font-family: var(--oz-mono);
          font-size: 0.9rem;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .hc-root .hc-overall-value {
          margin-bottom: 0.75rem;
          font-size: 3rem;
          font-weight: 500;
          letter-spacing: -0.04em;
          line-height: 1;
          font-variant-numeric: tabular-nums;
        }
        /* A share: a hairline filled from the left */
        .hc-root .hc-track {
          flex: 1;
          height: 3px;
          background: var(--oz-rule);
        }
        .hc-root .hc-bar {
          height: 100%;
          background: var(--oz-fg);
          transition: width 0.5s cubic-bezier(0.16, 0.8, 0.2, 1);
        }
        .hc-root .hc-layer {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 0.55rem 0;
          border-top: 1px solid var(--oz-rule);
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }
        .hc-root .hc-layer:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }
        .hc-root .hc-layer-name {
          width: 2.5rem;
          flex: none;
          color: var(--oz-fg-dim);
        }
        .hc-root .hc-layer-value {
          width: 2.75rem;
          text-align: right;
        }
        .hc-root .hc-layer-range {
          width: 5.5rem;
          color: var(--oz-fg-dim);
        }
        .hc-root .hc-diff-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 6px;
        }
        /* A head as a cell */
        .hc-root .hc-diff-item {
          padding: 0.7rem 0.5rem 0.75rem;
          border: 1px solid var(--oz-rule);
          text-align: center;
          cursor: pointer;
          transition: border-color 0.2s ease;
        }
        .hc-root .hc-diff-item:hover {
          border-color: var(--oz-fg);
        }
        .hc-root .hc-diff-name {
          color: var(--oz-fg-dim);
          font-family: var(--oz-mono);
          font-size: 0.78rem;
        }
        .hc-root .hc-diff-value {
          margin-top: 0.15rem;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
          font-variant-numeric: tabular-nums;
        }
        .hc-root .hc-diff-note {
          color: var(--oz-fg-dim);
          font-size: 0.78rem;
        }
        .hc-root .hc-patterns-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 2rem;
        }
        .hc-root .hc-patterns {
          margin: 0;
          padding: 0;
          list-style: none;
        }
        .hc-root .hc-patterns li {
          display: flex;
          align-items: flex-start;
          gap: 0.6rem;
          padding: 0.55rem 0;
          border-top: 1px solid var(--oz-rule);
          font-size: 0.9rem;
          line-height: 1.45;
        }
        .hc-root .hc-patterns li:last-child {
          border-bottom: 1px solid var(--oz-rule);
        }
        .hc-root .hc-mark {
          flex: none;
        }
        .hc-root .hc-mark--same {
          color: var(--oz-ok);
        }
        .hc-root .hc-mark--different {
          color: var(--oz-danger);
        }
        @media (max-width: 800px) {
          .hc-root .hc-prompts-grid,
          .hc-root .hc-patterns-grid {
            grid-template-columns: 1fr;
          }
          .hc-root .hc-prompt + .hc-prompt {
            padding-left: 0;
            border-left: none;
            border-top: 1px solid var(--oz-rule);
          }
          .hc-root .hc-diff-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }
      `}</style>
    </div>
  )
}
