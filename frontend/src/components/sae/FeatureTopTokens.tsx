/**
 * Feature top tokens component.
 * Shows the top activating tokens for a specific SAE feature.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { useThemeColors } from '../../hooks/useTheme'
import { MONO } from '../home/homeKit'

interface TokenActivationExample {
  token: string
  token_id: number
  activation_value: number
  position: number
  context: string[]
  prompt: string
}

interface FeatureInterpretation {
  feature_idx: number
  top_activating_tokens: TokenActivationExample[]
  token_frequency_distribution: Record<string, number>
  suggested_label: string
  confidence: number
  activation_statistics: {
    total_activations: number
    mean_activation: number
    max_activation: number
    unique_tokens: number
    polysemanticity: number
  }
}

interface FeatureTopTokensProps {
  interpretation: FeatureInterpretation
  maxExamples?: number
  className?: string
}

export function FeatureTopTokens({
  interpretation,
  maxExamples = 10,
  className = '',
}: FeatureTopTokensProps) {
  const tc = useThemeColors()
  const chartRef = useRef<SVGSVGElement>(null)
  const [selectedExample, setSelectedExample] = useState<TokenActivationExample | null>(null)

  // Get top tokens by frequency
  const topTokens = Object.entries(interpretation.token_frequency_distribution)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 20)

  // Render token frequency bar chart
  useEffect(() => {
    if (!chartRef.current || topTokens.length === 0) return

    d3.select(chartRef.current).selectAll('*').remove()

    const svg = d3.select(chartRef.current)
    const width = 300
    const height = Math.min(300, topTokens.length * 24 + 20)
    const margin = { top: 10, right: 10, bottom: 10, left: 80 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const maxFreq = Math.max(...topTokens.map(([, f]) => f))

    const xScale = d3.scaleLinear()
      .domain([0, maxFreq])
      .range([0, innerWidth])

    const yScale = d3.scaleBand()
      .domain(topTokens.map(([t]) => t))
      .range([0, innerHeight])
      .padding(0.2)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Bars
    g.selectAll('rect')
      .data(topTokens)
      .enter()
      .append('rect')
      .attr('x', 0)
      .attr('y', ([t]) => yScale(t) || 0)
      .attr('width', ([, f]) => xScale(f))
      .attr('height', yScale.bandwidth())
      .attr('fill', tc.text)

    // Token labels
    g.selectAll('.token-label')
      .data(topTokens)
      .enter()
      .append('text')
      .attr('class', 'token-label')
      .attr('x', -5)
      .attr('y', ([t]) => (yScale(t) || 0) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', tc.text)
      .attr('font-size', '12px')
      .attr('font-family', MONO)
      .text(([t]) => t.length > 10 ? t.slice(0, 10) + '...' : t)

    // Frequency labels
    g.selectAll('.freq-label')
      .data(topTokens)
      .enter()
      .append('text')
      .attr('class', 'freq-label')
      .attr('x', ([, f]) => xScale(f) + 5)
      .attr('y', ([t]) => (yScale(t) || 0) + yScale.bandwidth() / 2)
      .attr('dominant-baseline', 'middle')
      .attr('fill', tc.textMuted)
      .attr('font-size', '11.5px')
      .attr('font-family', MONO)
      .text(([, f]) => `${(f * 100).toFixed(1)}%`)

  }, [topTokens, tc.text, tc.textMuted])

  const { activation_statistics: stats } = interpretation

  return (
    <div className={`feature-top-tokens ${className}`}>
      {/* Header */}
      <div className="ftt-head">
        <h3>Feature {interpretation.feature_idx}</h3>
        <p>
          {interpretation.suggested_label}
          <span> · {(interpretation.confidence * 100).toFixed(0)}% confidence</span>
        </p>
      </div>

      {/* Statistics */}
      <div className="ftt-stats">
        <div className="ftt-stat">
          <div className="ftt-stat-label">Activations</div>
          <div className="ftt-stat-value">{stats.total_activations.toLocaleString()}</div>
        </div>
        <div className="ftt-stat">
          <div className="ftt-stat-label">Mean</div>
          <div className="ftt-stat-value">{stats.mean_activation.toFixed(3)}</div>
        </div>
        <div className="ftt-stat">
          <div className="ftt-stat-label">Max</div>
          <div className="ftt-stat-value">{stats.max_activation.toFixed(3)}</div>
        </div>
        <div className="ftt-stat">
          <div className="ftt-stat-label">Unique tokens</div>
          <div className="ftt-stat-value">{stats.unique_tokens}</div>
        </div>
        <div className="ftt-stat">
          <div className="ftt-stat-label">Polysemanticity</div>
          <div className="ftt-stat-value">{stats.polysemanticity.toFixed(2)}</div>
          <div className="ftt-track" title="From monosemantic (0) to polysemantic (1)">
            <div style={{ width: `${stats.polysemanticity * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="ftt-grid">
        {/* Token frequency chart */}
        <div>
          <div className="ftt-label">Token distribution</div>
          <svg ref={chartRef} width="100%" height={Math.min(300, topTokens.length * 24 + 20)} />
        </div>

        {/* Top activating examples */}
        <div>
          <div className="ftt-label">Top activating examples</div>
          <div className="ftt-examples">
            {interpretation.top_activating_tokens.slice(0, maxExamples).map((example, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedExample(selectedExample === example ? null : example)}
                className={`ftt-example${selectedExample === example ? ' is-open' : ''}`}
                aria-expanded={selectedExample === example}
              >
                <div className="ftt-example-head">
                  <span>"{example.token}"</span>
                  <span className="ftt-example-value">{example.activation_value.toFixed(3)}</span>
                </div>
                {selectedExample === example && (
                  <div className="ftt-context">
                    {example.context.map((t, i) => (
                      <span key={i} className={i === Math.floor(example.context.length / 2) ? 'ftt-context-token' : ''}>
                        {t}{' '}
                      </span>
                    ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .feature-top-tokens .ftt-head {
          margin-bottom: 1.25rem;
        }
        .feature-top-tokens .ftt-head h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }
        .feature-top-tokens .ftt-head p {
          margin: 0.35rem 0 0;
          font-size: 0.9375rem;
        }
        .feature-top-tokens .ftt-head p span {
          color: var(--oz-fg-dim);
        }
        /* Headline numbers, each under a rule */
        .feature-top-tokens .ftt-stats {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 1.25rem 1.5rem;
          margin-bottom: 1.75rem;
        }
        .feature-top-tokens .ftt-stat {
          padding-top: 0.6rem;
          border-top: 1px solid var(--oz-rule-strong);
        }
        .feature-top-tokens .ftt-stat-label,
        .feature-top-tokens .ftt-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
        }
        .feature-top-tokens .ftt-label {
          margin-bottom: 0.6rem;
          font-size: 0.85rem;
        }
        .feature-top-tokens .ftt-stat-value {
          margin-top: 0.2rem;
          font-size: 1.6rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
          font-variant-numeric: tabular-nums;
        }
        .feature-top-tokens .ftt-track {
          height: 3px;
          margin-top: 0.5rem;
          background: var(--oz-rule);
        }
        .feature-top-tokens .ftt-track > div {
          height: 100%;
          background: var(--oz-fg);
        }
        .feature-top-tokens .ftt-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 2rem;
        }
        .feature-top-tokens .ftt-examples {
          max-height: 300px;
          overflow-y: auto;
          border-bottom: 1px solid var(--oz-rule);
        }
        /* An example: its token and how strongly the feature fired; open, its context */
        .feature-top-tokens .ftt-example {
          position: relative;
          width: 100%;
          padding: 0.5rem 0;
          border-top: 1px solid var(--oz-rule);
          font-family: var(--oz-mono);
          font-size: 0.85rem;
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }
        .feature-top-tokens .ftt-example:hover {
          color: var(--oz-mark);
        }
        .feature-top-tokens .ftt-example.is-open:hover {
          color: inherit;
        }
        .feature-top-tokens .ftt-example-head {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
        }
        .feature-top-tokens .ftt-example-value {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
        }
        .feature-top-tokens .ftt-context {
          margin-top: 0.45rem;
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          line-height: 1.6;
        }
        /* The token the feature fired on, lit */
        .feature-top-tokens .ftt-context-token {
          padding: 0 0.2em;
          background: var(--oz-accent);
          color: var(--oz-on-accent);
        }
        @media (max-width: 900px) {
          .feature-top-tokens .ftt-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  )
}
