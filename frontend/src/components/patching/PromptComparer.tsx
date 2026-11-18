/**
 * Side-by-side comparison of baseline vs patched outputs.
 * Highlights token differences and shows effect summary.
 */

import type { EffectSummary } from '../../types/patching'

type InterventionMode = 'patch' | 'ablation' | 'mixed'

interface PromptComparerProps {
  baselineOutput: string
  patchedOutput: string
  baselineDecoded: string[]
  patchedDecoded: string[]
  effectSummary: EffectSummary
  interventionMode?: InterventionMode
}

const OUTPUT_LABELS: Record<InterventionMode, string> = {
  patch: 'Patched output',
  ablation: 'Ablated output',
  mixed: 'Intervened output',
}

export function PromptComparer({
  baselineOutput,
  patchedOutput,
  baselineDecoded,
  patchedDecoded,
  effectSummary,
  interventionMode = 'patch',
}: PromptComparerProps) {
  const { first_divergence_position, token_changes, changed_tokens, baseline_length, patched_length } = effectSummary

  const hasDifference = token_changes > 0

  // Find positions where tokens differ for highlighting
  const differingPositions = new Set(changed_tokens.map(ct => ct.position))

  return (
    <div className="prompt-comparer">
      <div className="comparer-header">
        {hasDifference ? (
          <span className="oz-tag oz-tag--accent">
            {token_changes} token{token_changes !== 1 ? 's' : ''} changed
          </span>
        ) : (
          <span className="oz-tag">No difference</span>
        )}
        {hasDifference && first_divergence_position !== null && (
          <span className="oz-hint">First divergence at position {first_divergence_position}</span>
        )}
      </div>

      <div className="comparison-grid">
        <div className="output-column baseline">
          <div className="column-header">
            <span className="column-label">Baseline output</span>
            <span className="token-count">{baseline_length} tokens</span>
          </div>
          <div className="output-content">
            <div className="output-text">{baselineOutput}</div>
            <div className="tokens-row">
              {baselineDecoded.map((token, idx) => (
                <span
                  key={idx}
                  className={`token ${differingPositions.has(idx) ? 'different' : ''}`}
                  title={`Position ${idx}`}
                >
                  {token.replace(/\n/g, '\\n').replace(/ /g, '\u00B7')}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="output-column patched">
          <div className="column-header">
            <span className="column-label">{OUTPUT_LABELS[interventionMode]}</span>
            <span className="token-count">{patched_length} tokens</span>
          </div>
          <div className="output-content">
            <div className="output-text">{patchedOutput}</div>
            <div className="tokens-row">
              {patchedDecoded.map((token, idx) => (
                <span
                  key={idx}
                  className={`token ${differingPositions.has(idx) ? 'different' : ''}`}
                  title={`Position ${idx}`}
                >
                  {token.replace(/\n/g, '\\n').replace(/ /g, '\u00B7')}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {hasDifference && (
        <div className="effect-details">
          <div className="changed-tokens-list">
            <span className="detail-label">Changed tokens</span>
            <div className="changes">
              {changed_tokens.slice(0, 10).map((ct, idx) => (
                <div key={idx} className="token-change">
                  <span className="change-position">{String(ct.position).padStart(2, '0')}</span>
                  <span className="change-from">"{ct.baseline_token}"</span>
                  <span className="change-arrow">&rarr;</span>
                  <span className="change-to">"{ct.patched_token}"</span>
                </div>
              ))}
              {changed_tokens.length > 10 && (
                <div className="more-changes">
                  ... and {changed_tokens.length - 10} more
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        .prompt-comparer .comparer-header {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.5rem 0.9rem;
          margin-bottom: 1.25rem;
        }

        /* Baseline and intervened, side by side with a rule between */
        .prompt-comparer .comparison-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          border-top: 1px solid var(--oz-rule);
          border-bottom: 1px solid var(--oz-rule);
        }

        .prompt-comparer .output-column {
          min-width: 0;
          padding: 1rem 1.25rem 1.25rem 0;
        }

        .prompt-comparer .output-column + .output-column {
          padding: 1rem 0 1.25rem 1.25rem;
          border-left: 1px solid var(--oz-rule);
        }

        .prompt-comparer .column-header {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.75rem;
        }

        .prompt-comparer .column-label {
          font-size: 0.9375rem;
          font-weight: 500;
        }

        .prompt-comparer .token-count {
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-variant-numeric: tabular-nums;
        }

        .prompt-comparer .output-text {
          margin-bottom: 0.9rem;
          font-family: var(--oz-mono);
          font-size: 0.875rem;
          line-height: 1.65;
          white-space: pre-wrap;
          word-break: break-word;
        }

        .prompt-comparer .tokens-row {
          display: flex;
          flex-wrap: wrap;
          gap: 3px;
        }

        /* A token as a cell; the ones the intervention changed are lit */
        .prompt-comparer .token {
          padding: 2px 6px;
          border: 1px solid var(--oz-rule);
          font-family: var(--oz-mono);
          font-size: 0.78rem;
          white-space: pre;
        }

        .prompt-comparer .token.different {
          background: var(--oz-accent);
          border-color: var(--oz-accent);
          color: var(--oz-on-accent);
        }

        .prompt-comparer .effect-details {
          margin-top: 1.25rem;
        }

        .prompt-comparer .detail-label {
          display: block;
          margin-bottom: 0.5rem;
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
          font-weight: 500;
        }

        .prompt-comparer .token-change {
          display: flex;
          align-items: baseline;
          gap: 0.75rem;
          padding: 0.4rem 0;
          border-top: 1px solid var(--oz-rule);
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }

        .prompt-comparer .change-position {
          color: var(--oz-fg-faint);
          font-size: 0.75rem;
        }

        .prompt-comparer .change-from {
          color: var(--oz-fg-dim);
        }

        .prompt-comparer .change-arrow {
          color: var(--oz-fg-faint);
        }

        .prompt-comparer .more-changes {
          padding-top: 0.5rem;
          border-top: 1px solid var(--oz-rule);
          color: var(--oz-fg-dim);
          font-size: 0.85rem;
        }

        @media (max-width: 900px) {
          .prompt-comparer .comparison-grid { grid-template-columns: 1fr; }
          .prompt-comparer .output-column, .prompt-comparer .output-column + .output-column { padding: 1rem 0 1.25rem; border-left: none; }
          .prompt-comparer .output-column + .output-column { border-top: 1px solid var(--oz-rule); }
        }
      `}</style>
    </div>
  )
}

export default PromptComparer
