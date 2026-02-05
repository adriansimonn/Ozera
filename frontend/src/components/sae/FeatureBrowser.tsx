/**
 * Feature browser component for exploring SAE features.
 * Allows browsing, searching, and filtering learned features.
 */

import { useState, useMemo } from 'react'
import { Search, Filter, ChevronDown, ChevronUp, Zap, AlertTriangle } from 'lucide-react'

interface FeatureStats {
  feature_idx: number
  activation_frequency: number
  mean_activation: number
  max_activation: number
  polysemanticity: number
  suggested_label: string
  confidence: number
  unique_tokens: number
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

type SortField = 'feature_idx' | 'activation_frequency' | 'mean_activation' | 'polysemanticity' | 'confidence'
type FilterType = 'all' | 'dead' | 'monosemantic' | 'polysemantic' | 'active'

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
  const [showFilters, setShowFilters] = useState(false)

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

    // Apply sort
    features.sort((a, b) => {
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

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null
    return sortAscending
      ? <ChevronUp className="w-3 h-3" />
      : <ChevronDown className="w-3 h-3" />
  }

  const getFrequencyColor = (freq: number) => {
    if (freq === 0) return 'rgba(255,255,255,0.2)'
    if (freq < 0.01) return 'rgba(255,255,255,0.35)'
    if (freq < 0.1) return 'rgba(255,255,255,0.6)'
    if (freq < 0.5) return 'rgba(255,255,255,0.7)'
    return 'rgba(255,255,255,0.85)'
  }

  const getPolysemanticity = (score: number) => {
    if (score < 0.3) return { label: 'Mono', color: 'rgba(34,197,94,0.9)' }
    if (score < 0.7) return { label: 'Mixed', color: 'rgba(234,179,8,0.9)' }
    return { label: 'Poly', color: 'rgba(239,68,68,0.9)' }
  }

  return (
    <div className={`feature-browser ${className}`} style={{ background: 'rgba(255,255,255,0.03)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.1)' }}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-200 tracking-tight">
            Feature Browser
          </h3>
          <div className="text-sm text-gray-500">
            {filteredFeatures.length} / {catalog.num_features} features
          </div>
        </div>

        {/* Search and filter */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search by index or label..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-black/50 border border-gray-700 text-white text-sm focus:outline-none focus:border-purple-500"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`px-3 py-2 border ${showFilters ? 'border-purple-500 text-purple-400' : 'border-gray-700 text-gray-400'} hover:border-purple-500 transition-colors`}
          >
            <Filter className="w-4 h-4" />
          </button>
        </div>

        {/* Filter options */}
        {showFilters && (
          <div className="mt-3 flex flex-wrap gap-2">
            {(['all', 'active', 'dead', 'monosemantic', 'polysemantic'] as FilterType[]).map(type => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className={`px-3 py-1 text-xs uppercase tracking-wide border transition-colors ${
                  filterType === type
                    ? 'border-purple-500 text-purple-400 bg-purple-500/10'
                    : 'border-gray-700 text-gray-500 hover:border-gray-600'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-px bg-gray-800 border-b border-gray-800">
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Total</div>
          <div className="text-lg font-mono text-white">{catalog.num_features}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Dead</div>
          <div className="text-lg font-mono text-red-400">{catalog.dead_features.length}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Mono</div>
          <div className="text-lg font-mono text-green-400">{catalog.monosemantic_features.length}</div>
        </div>
        <div className="bg-black/40 p-3 text-center">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Poly</div>
          <div className="text-lg font-mono text-yellow-400">{catalog.polysemantic_features.length}</div>
        </div>
      </div>

      {/* Table header */}
      <div className="grid grid-cols-12 gap-2 px-4 py-2 bg-black/60 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
        <button
          className="col-span-2 flex items-center gap-1 hover:text-gray-300"
          onClick={() => handleSort('feature_idx')}
        >
          Index <SortIcon field="feature_idx" />
        </button>
        <div className="col-span-3">Label</div>
        <button
          className="col-span-2 flex items-center gap-1 hover:text-gray-300"
          onClick={() => handleSort('activation_frequency')}
        >
          Freq <SortIcon field="activation_frequency" />
        </button>
        <button
          className="col-span-2 flex items-center gap-1 hover:text-gray-300"
          onClick={() => handleSort('mean_activation')}
        >
          Mean <SortIcon field="mean_activation" />
        </button>
        <button
          className="col-span-2 flex items-center gap-1 hover:text-gray-300"
          onClick={() => handleSort('polysemanticity')}
        >
          Type <SortIcon field="polysemanticity" />
        </button>
        <div className="col-span-1"></div>
      </div>

      {/* Feature list */}
      <div className="max-h-[400px] overflow-y-auto">
        {filteredFeatures.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            No features match your criteria
          </div>
        ) : (
          filteredFeatures.map(feature => {
            const isDead = catalog.dead_features.includes(feature.feature_idx)
            const polySem = getPolysemanticity(feature.polysemanticity)
            const isSelected = selectedFeature === feature.feature_idx

            return (
              <button
                key={feature.feature_idx}
                onClick={() => onFeatureSelect(feature.feature_idx)}
                className={`w-full grid grid-cols-12 gap-2 px-4 py-3 text-left border-b border-gray-800/50 transition-colors ${
                  isSelected
                    ? 'bg-purple-500/20 border-l-2 border-l-purple-500'
                    : 'hover:bg-gray-800/50'
                } ${isDead ? 'opacity-50' : ''}`}
              >
                <div className="col-span-2 font-mono text-purple-300">
                  F{feature.feature_idx}
                </div>
                <div className="col-span-3 text-sm text-gray-300 truncate" title={feature.suggested_label}>
                  {feature.suggested_label}
                </div>
                <div className={`col-span-2 font-mono text-sm ${getFrequencyColor(feature.activation_frequency)}`}>
                  {(feature.activation_frequency * 100).toFixed(1)}%
                </div>
                <div className="col-span-2 font-mono text-sm text-gray-400">
                  {feature.mean_activation.toFixed(3)}
                </div>
                <div className={`col-span-2 text-xs ${polySem.color}`}>
                  {polySem.label}
                </div>
                <div className="col-span-1 flex items-center justify-end">
                  {isDead && <AlertTriangle className="w-3 h-3 text-red-500" />}
                  {feature.activation_frequency > 0.5 && <Zap className="w-3 h-3 text-yellow-500" />}
                </div>
              </button>
            )
          })
        )}
      </div>

      {/* Footer with pagination hint */}
      {filteredFeatures.length > 50 && (
        <div className="p-3 text-center text-xs text-gray-500 border-t border-gray-800">
          Scroll to see more features
        </div>
      )}

      <style>{`
        .feature-browser .p-4 { padding: 1rem; }
        .feature-browser .p-3 { padding: 0.75rem; }
        .feature-browser .p-8 { padding: 2rem; }
        .feature-browser .px-3 { padding-left: 0.75rem; padding-right: 0.75rem; }
        .feature-browser .px-4 { padding-left: 1rem; padding-right: 1rem; }
        .feature-browser .py-1 { padding-top: 0.25rem; padding-bottom: 0.25rem; }
        .feature-browser .py-2 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        .feature-browser .py-3 { padding-top: 0.75rem; padding-bottom: 0.75rem; }
        .feature-browser .border-b { border-bottom: 1px solid rgba(255,255,255,0.1); }
        .feature-browser .border-t { border-top: 1px solid rgba(255,255,255,0.1); }
        .feature-browser .bg-black\\/40 { background: rgba(0,0,0,0.2); }
        .feature-browser .bg-black\\/50 { background: rgba(255,255,255,0.03); }
        .feature-browser .bg-black\\/60 { background: rgba(0,0,0,0.3); }
        .feature-browser .bg-gray-800 { background: rgba(255,255,255,0.1); }
        .feature-browser .bg-gray-800\\/50 { background: rgba(255,255,255,0.03); }
        .feature-browser .bg-purple-500\\/10 { background: rgba(255,255,255,0.08); }
        .feature-browser .bg-purple-500\\/20 { background: rgba(255,255,255,0.12); }
        .feature-browser .border-gray-700 { border-color: rgba(255,255,255,0.1); }
        .feature-browser .border-gray-800 { border-color: rgba(255,255,255,0.1); }
        .feature-browser .border-gray-800\\/50 { border-color: rgba(255,255,255,0.05); }
        .feature-browser .border-purple-500 { border-color: rgba(255,255,255,0.4); }
        .feature-browser .border-l-purple-500 { border-left-color: rgba(255,255,255,0.4); }
        .feature-browser .text-white { color: #fff; }
        .feature-browser .text-gray-200 { color: rgba(255,255,255,0.95); }
        .feature-browser .text-gray-300 { color: rgba(255,255,255,0.7); }
        .feature-browser .text-gray-400 { color: rgba(255,255,255,0.4); }
        .feature-browser .text-gray-500 { color: rgba(255,255,255,0.5); }
        .feature-browser .text-gray-600 { color: rgba(255,255,255,0.2); }
        .feature-browser .text-purple-300 { color: rgba(255,255,255,0.7); }
        .feature-browser .text-purple-400 { color: rgba(255,255,255,0.6); }
        .feature-browser .text-red-400 { color: rgba(239,68,68,0.9); }
        .feature-browser .text-red-500 { color: rgba(239,68,68,0.9); }
        .feature-browser .text-green-400 { color: rgba(34,197,94,0.9); }
        .feature-browser .text-yellow-400 { color: rgba(234,179,8,0.9); }
        .feature-browser .text-yellow-500 { color: rgba(234,179,8,0.9); }
        .feature-browser input:focus { border-color: rgba(255,255,255,0.25); }
        .feature-browser button:hover .text-gray-300 { color: rgba(255,255,255,0.7); }
        .feature-browser button:hover.border-gray-700 { border-color: rgba(255,255,255,0.2); }
        .feature-browser button:hover.border-gray-600 { border-color: rgba(255,255,255,0.2); }
        .feature-browser button:hover.border-purple-500 { border-color: rgba(255,255,255,0.5); }
      `}</style>
    </div>
  )
}
