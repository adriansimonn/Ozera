/**
 * HeadComparer component for side-by-side comparison of attention patterns.
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
  } = comparison

  // Calculate overall similarity
  const overallSimilarity = layer_similarities.length > 0
    ? layer_similarities.reduce((sum, l) => sum + l.mean_similarity, 0) / layer_similarities.length
    : 0

  return (
    <div className="space-y-6">
      {/* Prompts being compared */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Prompt 1</div>
          <div className="text-sm text-slate-300 font-mono truncate">{prompt1}</div>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Prompt 2</div>
          <div className="text-sm text-slate-300 font-mono truncate">{prompt2}</div>
        </div>
      </div>

      {/* Overall Similarity Score */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-300">Overall Similarity</h3>
          <span
            className={`text-2xl font-bold ${
              overallSimilarity > 0.7 ? 'text-green-400' :
              overallSimilarity > 0.4 ? 'text-yellow-400' : 'text-red-400'
            }`}
          >
            {(overallSimilarity * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              overallSimilarity > 0.7 ? 'bg-green-500' :
              overallSimilarity > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${overallSimilarity * 100}%` }}
          />
        </div>
      </div>

      {/* Layer-by-Layer Similarity */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Layer Similarity</h3>
        <div className="space-y-2">
          {layer_similarities.map((layer) => (
            <div key={layer.layer} className="flex items-center gap-3">
              <span className="text-xs text-slate-500 w-16">Layer {layer.layer}</span>
              <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    layer.mean_similarity > 0.7 ? 'bg-green-500' :
                    layer.mean_similarity > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${layer.mean_similarity * 100}%` }}
                />
              </div>
              <span className="text-xs text-slate-400 w-12 text-right">
                {(layer.mean_similarity * 100).toFixed(0)}%
              </span>
              <span className="text-xs text-slate-600">
                ({(layer.min_similarity * 100).toFixed(0)}-{(layer.max_similarity * 100).toFixed(0)}%)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Most Different Heads */}
      {head_differences.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Most Different Heads</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
            {head_differences.slice(0, 10).map((diff) => (
              <button
                key={`${diff.layer}-${diff.head}`}
                onClick={() => onHeadSelect?.(diff.layer, diff.head)}
                className="bg-slate-800/50 border border-red-500/30 rounded-lg p-3 hover:border-red-500/60 transition-all"
              >
                <div className="text-xs text-slate-400">
                  L{diff.layer} H{diff.head}
                </div>
                <div className="text-lg font-bold text-red-400">
                  {(diff.difference * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500">different</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pattern Analysis */}
      <div className="grid grid-cols-2 gap-4">
        {/* Common Patterns */}
        {common_patterns.length > 0 && (
          <div className="bg-slate-900/50 border border-green-500/30 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-green-400 mb-3">Common Patterns</h3>
            <ul className="space-y-2">
              {common_patterns.map((pattern, idx) => (
                <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                  <span className="text-green-500 mt-0.5">✓</span>
                  <span>{pattern}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Divergent Patterns */}
        {divergent_patterns.length > 0 && (
          <div className="bg-slate-900/50 border border-red-500/30 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-red-400 mb-3">Divergent Patterns</h3>
            <ul className="space-y-2">
              {divergent_patterns.map((pattern, idx) => (
                <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">✗</span>
                  <span>{pattern}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* No significant differences */}
      {head_differences.length === 0 && (
        <div className="bg-slate-900/50 border border-green-500/30 rounded-lg p-6 text-center">
          <div className="text-green-400 text-lg mb-2">High Similarity Detected</div>
          <div className="text-slate-400 text-sm">
            Attention patterns are very similar between these two prompts.
            This suggests the model processes them in similar ways.
          </div>
        </div>
      )}
    </div>
  )
}
