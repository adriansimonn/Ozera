/**
 * Feature browser component for exploring SAE features.
 * Allows browsing, searching, and filtering learned features.
 */

import { useState, useMemo } from 'react'
import { Zap, AlertTriangle } from 'lucide-react'

interface FeatureStats {
  feature_idx: number
  activation_frequency: number
  mean_activation: number
  max_activation: number
  polysemanticity: number
  suggested_label: string
  confidence: number
  unique_tokens: number
  /** Autointerp score of the explanation shown as its label (0.5 is chance), if scored */
  score?: number | null
  /** Title for the label (the full explanation and its scores) */
  label_title?: string
}

interface FeatureCatalog {
  sae_id: string
  num_features: number
  features: FeatureStats[]
  dead_features: number[]
  polysemantic_features: number[]
  monosemantic_features: number[]
}

interface FeatureBrowserProps {
  catalog: FeatureCatalog
  selectedFeature?: number
  onFeatureSelect: (featureIdx: number) => void
  className?: string
}

type SortField = 'feature_idx' | 'activation_frequency' | 'mean_activation' | 'polysemanticity' | 'confidence' | 'score'
type FilterType = 'all' | 'dead' | 'monosemantic' | 'polysemantic' | 'active'

const FILTER_LABELS: Record<FilterType, string> = {
  all: 'All',
  active: 'Active',
  dead: 'Dead',
  monosemantic: 'Monosemantic',
  polysemantic: 'Polysemantic',
}

export function FeatureBrowser({
  catalog,
  selectedFeature,
  onFeatureSelect,
  className = '',
}: FeatureBrowserProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [sortField, setSortField] = useState<SortField>('activation_frequency')
  const [sortAscending, setSortAscending] = useState(false)
  const [filterType, setFilterType] = useState<FilterType>('active')

  // Filter and sort features
  const filteredFeatures = useMemo(() => {
    let features = [...catalog.features]

    // Apply filter
    switch (filterType) {
      case 'dead':
        features = features.filter(f => catalog.dead_features.includes(f.feature_idx))
        break
      case 'monosemantic':
        features = features.filter(f => catalog.monosemantic_features.includes(f.feature_idx))
        break
      case 'polysemantic':
        features = features.filter(f => catalog.polysemantic_features.includes(f.feature_idx))
        break
      case 'active':
        features = features.filter(f => !catalog.dead_features.includes(f.feature_idx))
        break
    }

    // Apply search
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      features = features.filter(f =>
        f.feature_idx.toString().includes(query) ||
        f.suggested_label.toLowerCase().includes(query)
      )
    }

    // Apply sort (features without a score last, either way)
    features.sort((a, b) => {
      if (sortField === 'score' && (a.score == null || b.score == null)) {
        return (a.score == null ? 1 : 0) - (b.score == null ? 1 : 0)
      }
      const aVal = a[sortField]
      const bVal = b[sortField]
      const cmp = typeof aVal === 'number' && typeof bVal === 'number'
        ? aVal - bVal
        : String(aVal).localeCompare(String(bVal))
      return sortAscending ? cmp : -cmp
    })

    return features
  }, [catalog, searchQuery, sortField, sortAscending, filterType])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAscending(!sortAscending)
    } else {
      setSortField(field)
      setSortAscending(false)
    }
  }

  // The sorted column's direction
  const sortMark = (field: SortField) => (sortField === field ? (sortAscending ? ' ↑' : ' ↓') : '')

  const getPolysemanticity = (score: number) => {
    if (score < 0.3) return 'Mono'
    if (score < 0.7) return 'Mixed'
    return 'Poly'
  }

  return (
    <div className={`feature-browser ${className}`}>
      <div className="fb-head">
        <h3>Feature browser</h3>
        <span className="fb-count">
          {filteredFeatures.length} / {catalog.num_features} features
        </span>
      </div>

      {/* Search and filter */}
      <div className="fb-controls">
        <input
          type="text"
          placeholder="Search by index or label..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="oz-input"
          aria-label="Search features"
        />
        <div className="oz-chips" role="radiogroup" aria-label="Filter features">
          {(['all', 'active', 'dead', 'monosemantic', 'polysemantic'] as FilterType[]).map(type => (
            <button
              key={type}
              role="radio"
              aria-checked={filterType === type}
              onClick={() => setFilterType(type)}
              className="oz-chip"
            >
              {FILTER_LABELS[type]}
            </button>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <div className="fb-stats">
        <div className="fb-stat">
          <div className="fb-stat-label">Total</div>
          <div className="fb-stat-value">{catalog.num_features}</div>
        </div>
        <div className="fb-stat">
          <div className="fb-stat-label">Dead</div>
          <div className="fb-stat-value">{catalog.dead_features.length}</div>
        </div>
        <div className="fb-stat">
          <div className="fb-stat-label">Monosemantic</div>
          <div className="fb-stat-value">{catalog.monosemantic_features.length}</div>
        </div>
        <div className="fb-stat">
          <div className="fb-stat-label">Polysemantic</div>
          <div className="fb-stat-value">{catalog.polysemantic_features.length}</div>
        </div>
      </div>

      {/* Table header */}
      <div className="fb-row fb-row--head">
        <button onClick={() => handleSort('feature_idx')}>Index{sortMark('feature_idx')}</button>
        <div>Label</div>
        <button onClick={() => handleSort('activation_frequency')}>Freq{sortMark('activation_frequency')}</button>
        <button onClick={() => handleSort('mean_activation')}>Mean{sortMark('mean_activation')}</button>
        <button onClick={() => handleSort('score')} title="Autointerp score of the label's explanation (0.5 is chance)">
          Score{sortMark('score')}
        </button>
        <button onClick={() => handleSort('polysemanticity')}>Type{sortMark('polysemanticity')}</button>
        <div></div>
      </div>

      {/* Feature list */}
      <div className="fb-rows">
        {filteredFeatures.length === 0 ? (
          <div className="fb-empty">No features match your criteria</div>
        ) : (
          filteredFeatures.map(feature => {
            const isDead = catalog.dead_features.includes(feature.feature_idx)
            const isSelected = selectedFeature === feature.feature_idx

            return (
              <button
                key={feature.feature_idx}
                onClick={() => onFeatureSelect(feature.feature_idx)}
                className={`fb-row${isSelected ? ' is-active' : ''}${isDead ? ' is-dead' : ''}`}
              >
                <div className="fb-id">F{feature.feature_idx}</div>
                <div className="fb-label" title={feature.label_title ?? feature.suggested_label}>
                  {feature.suggested_label}
                </div>
                <div className="fb-num">{(feature.activation_frequency * 100).toFixed(1)}%</div>
                <div className="fb-num">{feature.mean_activation.toFixed(2)}</div>
                <div className="fb-num">{feature.score != null ? feature.score.toFixed(2) : '–'}</div>
                <div className="fb-type">{getPolysemanticity(feature.polysemanticity)}</div>
                <div className="fb-flags">
                  {isDead && <AlertTriangle aria-label="Dead feature" />}
                  {feature.activation_frequency > 0.5 && <Zap aria-label="Fires on more than half of tokens" />}
                </div>
              </button>
            )
          })
        )}
      </div>

      <style>{`
        .feature-browser .fb-head {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: 1rem;
        }
        .feature-browser .fb-head h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }
        .feature-browser .fb-count {
          color: var(--oz-fg-dim);
          font-size: 0.875rem;
          font-variant-numeric: tabular-nums;
        }
        .feature-browser .fb-controls {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.75rem 1rem;
          margin-bottom: 1.25rem;
        }
        .feature-browser .fb-controls .oz-input {
          flex: 1 1 16rem;
          width: auto;
        }
        /* Headline numbers, each under a rule */
        .feature-browser .fb-stats {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }
        .feature-browser .fb-stat {
          padding-top: 0.6rem;
          border-top: 1px solid var(--oz-rule-strong);
        }
        .feature-browser .fb-stat-label {
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
        }
        .feature-browser .fb-stat-value {
          margin-top: 0.2rem;
          font-size: 1.6rem;
          font-weight: 500;
          letter-spacing: -0.03em;
          line-height: 1.15;
          font-variant-numeric: tabular-nums;
        }
        /* The features: a ruled table, the chosen row marked */
        .feature-browser .fb-row {
          position: relative;
          display: grid;
          grid-template-columns: 5.5rem minmax(0, 1fr) 5rem 4.5rem 4.5rem 4.5rem 2.5rem;
          align-items: baseline;
          gap: 0.75rem;
          width: 100%;
          padding: 0.6rem 0;
          border-top: 1px solid var(--oz-rule);
          text-align: left;
          font-size: 0.875rem;
        }
        .feature-browser button.fb-row {
          cursor: pointer;
          transition: background 0.15s ease;
        }
        .feature-browser button.fb-row:hover {
          background: var(--oz-fill);
        }
        .feature-browser .fb-row.is-active::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: -1px;
          width: 100%;
          height: 1px;
          background: var(--oz-mark);
        }
        .feature-browser .fb-row.is-dead {
          color: var(--oz-fg-faint);
        }
        .feature-browser .fb-row--head {
          border-top: none;
          color: var(--oz-fg-dim);
          font-size: 0.8rem;
          font-weight: 500;
        }
        .feature-browser .fb-row--head button {
          color: inherit;
          font: inherit;
          text-align: left;
          cursor: pointer;
          transition: color 0.2s ease;
        }
        .feature-browser .fb-row--head button:hover {
          color: var(--oz-fg);
        }
        .feature-browser .fb-rows {
          max-height: 420px;
          overflow-y: auto;
          border-bottom: 1px solid var(--oz-rule);
        }
        .feature-browser .fb-id,
        .feature-browser .fb-num {
          font-family: var(--oz-mono);
          font-size: 0.82rem;
        }
        .feature-browser .fb-id {
          font-weight: 500;
        }
        .feature-browser .fb-label {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .feature-browser .fb-type {
          color: var(--oz-fg-dim);
        }
        .feature-browser .fb-flags {
          display: flex;
          justify-content: flex-end;
          gap: 0.25rem;
        }
        .feature-browser .fb-flags svg {
          width: 14px;
          height: 14px;
          stroke-width: 2.25;
        }
        .feature-browser .fb-flags .lucide-alert-triangle {
          color: var(--oz-danger);
        }
        .feature-browser .fb-flags .lucide-zap {
          color: var(--oz-warn);
        }
        .feature-browser .fb-empty {
          padding: 2rem 0;
          color: var(--oz-fg-dim);
          text-align: center;
        }
      `}</style>
    </div>
  )
}
