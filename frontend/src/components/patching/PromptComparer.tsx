/**
 * Side-by-side comparison of baseline vs patched outputs.
 * Highlights token differences and shows effect summary.
 */

import { GitCompare, AlertCircle, CheckCircle2 } from 'lucide-react'
import type { EffectSummary } from '../../types/patching'

interface PromptComparerProps {
  baselineOutput: string
  patchedOutput: string
  baselineDecoded: string[]
  patchedDecoded: string[]
  effectSummary: EffectSummary
}

export function PromptComparer({
  baselineOutput,
  patchedOutput,
  baselineDecoded,
  patchedDecoded,
  effectSummary,
}: PromptComparerProps) {
  const { first_divergence_position, token_changes, changed_tokens, baseline_length, patched_length } = effectSummary

  const hasDifference = token_changes > 0

  // Find positions where tokens differ for highlighting
  const differingPositions = new Set(changed_tokens.map(ct => ct.position))

  return (
    <div className="prompt-comparer">
      <div className="comparer-header">
        <GitCompare className="header-icon" />
        <h3>Output Comparison</h3>
        {hasDifference ? (
          <span className="diff-badge different">
            <AlertCircle className="badge-icon" />
            {token_changes} token{token_changes !== 1 ? 's' : ''} changed
          </span>
        ) : (
          <span className="diff-badge same">
            <CheckCircle2 className="badge-icon" />
            No difference
          </span>
        )}
      </div>

      <div className="comparison-grid">
        <div className="output-column baseline">
          <div className="column-header">
            <span className="column-label">Baseline Output</span>
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
            <span className="column-label">Patched Output</span>
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
          <div className="detail-header">Effect Details</div>
          <div className="detail-row">
            <span className="detail-label">First divergence:</span>
            <span className="detail-value">
              {first_divergence_position !== null ? `Position ${first_divergence_position}` : 'N/A'}
            </span>
          </div>
          <div className="changed-tokens-list">
            <span className="detail-label">Changed tokens:</span>
            <div className="changes">
              {changed_tokens.slice(0, 10).map((ct, idx) => (
                <div key={idx} className="token-change">
                  <span className="change-position">[{ct.position}]</span>
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
        .prompt-comparer {
          background: rgba(0, 0, 0, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1.25rem;
        }

        .comparer-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 1rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .comparer-header h3 {
          margin: 0;
          font-size: 0.9rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.9);
          letter-spacing: 0.025em;
        }

        .header-icon {
          width: 16px;
          height: 16px;
          color: rgba(255, 255, 255, 0.5);
        }

        .diff-badge {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.25rem 0.625rem;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .diff-badge.different {
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgba(239, 68, 68, 0.9);
        }

        .diff-badge.same {
          background: rgba(34, 197, 94, 0.15);
          border: 1px solid rgba(34, 197, 94, 0.3);
          color: rgba(34, 197, 94, 0.9);
        }

        .badge-icon {
          width: 12px;
          height: 12px;
        }

        .comparison-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
        }

        .output-column {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .output-column.baseline {
          border-left: 3px solid rgba(156, 163, 175, 0.5);
        }

        .output-column.patched {
          border-left: 3px solid rgba(59, 130, 246, 0.5);
        }

        .column-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.625rem 0.875rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .column-label {
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.8);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .token-count {
          font-size: 0.7rem;
          color: rgba(255, 255, 255, 0.4);
        }

        .output-content {
          padding: 0.875rem;
        }

        .output-text {
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.9);
          line-height: 1.6;
          margin-bottom: 1rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
          white-space: pre-wrap;
          word-break: break-word;
        }

        .tokens-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem;
        }

        .token {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.7rem;
          padding: 0.125rem 0.375rem;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.6);
          white-space: nowrap;
        }

        .token.different {
          background: rgba(239, 68, 68, 0.2);
          border-color: rgba(239, 68, 68, 0.4);
          color: rgba(239, 68, 68, 0.9);
        }

        .effect-details {
          margin-top: 1.25rem;
          padding-top: 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .detail-header {
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.7);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.75rem;
        }

        .detail-row {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
          font-size: 0.85rem;
        }

        .detail-label {
          color: rgba(255, 255, 255, 0.5);
        }

        .detail-value {
          color: rgba(255, 255, 255, 0.9);
          font-family: monospace;
        }

        .changed-tokens-list {
          margin-top: 0.75rem;
        }

        .changes {
          margin-top: 0.5rem;
          display: flex;
          flex-direction: column;
          gap: 0.375rem;
        }

        .token-change {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8rem;
          padding: 0.375rem 0.5rem;
          background: rgba(255, 255, 255, 0.03);
        }

        .change-position {
          color: rgba(255, 255, 255, 0.4);
          font-size: 0.7rem;
        }

        .change-from {
          color: rgba(156, 163, 175, 0.8);
        }

        .change-arrow {
          color: rgba(255, 255, 255, 0.3);
        }

        .change-to {
          color: rgba(59, 130, 246, 0.9);
        }

        .more-changes {
          font-size: 0.75rem;
          color: rgba(255, 255, 255, 0.4);
          font-style: italic;
          padding-left: 0.5rem;
        }

        @media (max-width: 768px) {
          .comparison-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  )
}

export default PromptComparer
