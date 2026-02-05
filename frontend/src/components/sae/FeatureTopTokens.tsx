/**
 * Feature top tokens component.
 * Shows the top activating tokens for a specific SAE feature.
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Tag, TrendingUp, Hash } from 'lucide-react'

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
      .attr('fill', '#e5e7eb')
      .attr('opacity', 0.7)

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
      .attr('fill', '#e5e7eb')
      .attr('font-size', '11px')
      .attr('font-family', 'monospace')
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
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text(([, f]) => `${(f * 100).toFixed(1)}%`)

  }, [topTokens])

  const { activation_statistics: stats } = interpretation

  return (
    <div className={`feature-top-tokens ${className}`} style={{ background: 'rgba(255,255,255,0.03)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.1)' }}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-2">
          <Hash className="w-4 h-4 text-purple-400" />
          <span className="text-lg font-semibold text-gray-200">
            Feature {interpretation.feature_idx}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Tag className="w-3 h-3 text-gray-500" />
          <span className="text-sm text-gray-400">
            {interpretation.suggested_label}
          </span>
          <span className="text-xs text-gray-600">
            ({(interpretation.confidence * 100).toFixed(0)}% confidence)
          </span>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-px bg-gray-800">
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Activations</div>
          <div className="text-lg font-mono text-white">{stats.total_activations.toLocaleString()}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Mean</div>
          <div className="text-lg font-mono text-purple-300">{stats.mean_activation.toFixed(3)}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Max</div>
          <div className="text-lg font-mono text-purple-200">{stats.max_activation.toFixed(3)}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Unique Tokens</div>
          <div className="text-lg font-mono text-white">{stats.unique_tokens}</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-px bg-gray-800">
        {/* Token frequency chart */}
        <div className="bg-black/40 p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">
            Token Distribution
          </div>
          <svg ref={chartRef} width="100%" height={Math.min(300, topTokens.length * 24 + 20)} />
        </div>

        {/* Top activating examples */}
        <div className="bg-black/40 p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-3 flex items-center gap-2">
            <TrendingUp className="w-3 h-3" />
            Top Activating Examples
          </div>
          <div className="space-y-2 max-h-[280px] overflow-y-auto">
            {interpretation.top_activating_tokens.slice(0, maxExamples).map((example, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedExample(selectedExample === example ? null : example)}
                className={`w-full text-left p-2 border transition-colors ${
                  selectedExample === example
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-gray-800 hover:border-gray-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-white">
                    "{example.token}"
                  </span>
                  <span className="text-xs text-purple-300 font-mono">
                    {example.activation_value.toFixed(3)}
                  </span>
                </div>
                {selectedExample === example && (
                  <div className="mt-2 text-xs text-gray-500">
                    <div className="mb-1">Context:</div>
                    <div className="font-mono text-gray-400 bg-black/40 p-2 rounded">
                      {example.context.map((t, i) => (
                        <span
                          key={i}
                          className={i === Math.floor(example.context.length / 2) ? 'text-purple-300 font-bold' : ''}
                        >
                          {t}{' '}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Polysemanticity indicator */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 uppercase tracking-wide">
            Polysemanticity Score
          </span>
          <span className={`text-sm font-mono ${
            stats.polysemanticity < 0.3 ? 'text-green-400' :
            stats.polysemanticity < 0.7 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {stats.polysemanticity.toFixed(2)}
          </span>
        </div>
        <div className="mt-2 h-2 bg-gray-800 rounded overflow-hidden">
          <div
            className={`h-full transition-all ${
              stats.polysemanticity < 0.3 ? 'bg-green-500' :
              stats.polysemanticity < 0.7 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${stats.polysemanticity * 100}%` }}
          />
        </div>
        <div className="flex justify-between mt-1 text-xs text-gray-600">
          <span>Monosemantic</span>
          <span>Polysemantic</span>
        </div>
      </div>

      <style>{`
        .feature-top-tokens .p-2 { padding: 0.5rem; }
        .feature-top-tokens .p-3 { padding: 0.75rem; }
        .feature-top-tokens .p-4 { padding: 1rem; }
        .feature-top-tokens .border-b { border-bottom: 1px solid rgba(255,255,255,0.1); }
        .feature-top-tokens .border-t { border-top: 1px solid rgba(255,255,255,0.1); }
        .feature-top-tokens .border-gray-800 { border-color: rgba(255,255,255,0.1); }
        .feature-top-tokens .border-gray-700 { border-color: rgba(255,255,255,0.1); }
        .feature-top-tokens .border-purple-500 { border-color: rgba(255,255,255,0.4); }
        .feature-top-tokens .bg-black\\/40 { background: rgba(0,0,0,0.2); }
        .feature-top-tokens .bg-gray-800 { background: rgba(255,255,255,0.1); }
        .feature-top-tokens .bg-purple-500\\/10 { background: rgba(255,255,255,0.08); }
        .feature-top-tokens .bg-green-500 { background: rgba(34,197,94,0.9); }
        .feature-top-tokens .bg-yellow-500 { background: rgba(234,179,8,0.9); }
        .feature-top-tokens .bg-red-500 { background: rgba(239,68,68,0.9); }
        .feature-top-tokens .text-white { color: #fff; }
        .feature-top-tokens .text-gray-200 { color: rgba(255,255,255,0.95); }
        .feature-top-tokens .text-gray-400 { color: rgba(255,255,255,0.4); }
        .feature-top-tokens .text-gray-500 { color: rgba(255,255,255,0.5); }
        .feature-top-tokens .text-gray-600 { color: rgba(255,255,255,0.2); }
        .feature-top-tokens .text-purple-200 { color: rgba(255,255,255,0.7); }
        .feature-top-tokens .text-purple-300 { color: rgba(255,255,255,0.8); }
        .feature-top-tokens .text-purple-400 { color: rgba(255,255,255,0.6); }
        .feature-top-tokens .text-green-400 { color: rgba(34,197,94,0.9); }
        .feature-top-tokens .text-yellow-400 { color: rgba(234,179,8,0.9); }
        .feature-top-tokens .text-red-400 { color: rgba(239,68,68,0.9); }
        .feature-top-tokens button:hover .border-gray-700 { border-color: rgba(255,255,255,0.2); }
      `}</style>
    </div>
  )
}
