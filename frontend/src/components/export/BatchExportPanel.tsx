/**
 * Panel for batch exporting multiple figures as a ZIP archive.
 */

import { useState } from 'react'
import { Download, Loader2, Plus, Trash2, Package } from 'lucide-react'
import { apiClient } from '../../api/client'
import { downloadBlob } from '../../utils/svgExport'
import type {
  ExportFormat,
  PresetName,
  BatchExportItem,
  ActivationType,
} from '../../types/export'
import { COLORMAP_OPTIONS, PRESET_DISPLAY_INFO } from '../../types/export'

interface BatchExportPanelProps {
  activationId: string
  numLayers: number
  numHeads: number
  className?: string
}

export function BatchExportPanel({
  activationId,
  numLayers,
  numHeads,
  className = '',
}: BatchExportPanelProps) {
  const [items, setItems] = useState<BatchExportItem[]>([])
  const [preset, setPreset] = useState<PresetName>('default')
  const [format, setFormat] = useState<ExportFormat>('png')
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const addHeatmapItem = () => {
    setItems([
      ...items,
      {
        type: 'attention-heatmap',
        activation_id: activationId,
        layer: 0,
        head: 0,
        config: { preset, format },
        colormap: 'inferno',
      },
    ])
  }

  const addHistogramItem = () => {
    setItems([
      ...items,
      {
        type: 'activation-histogram',
        activation_id: activationId,
        layer: 0,
        activation_type: 'all',
        config: { preset, format },
      },
    ])
  }

  const addAllHeads = (layer: number) => {
    const newItems: BatchExportItem[] = []
    for (let h = 0; h < numHeads; h++) {
      newItems.push({
        type: 'attention-heatmap',
        activation_id: activationId,
        layer,
        head: h,
        config: { preset, format },
        colormap: 'inferno',
      })
    }
    setItems([...items, ...newItems])
  }

  const addAllLayers = () => {
    const newItems: BatchExportItem[] = []
    for (let l = 0; l < numLayers; l++) {
      newItems.push({
        type: 'activation-histogram',
        activation_id: activationId,
        layer: l,
        activation_type: 'all',
        config: { preset, format },
      })
    }
    setItems([...items, ...newItems])
  }

  const removeItem = (index: number) => {
    setItems(items.filter((_, i) => i !== index))
  }

  const updateItem = (index: number, updates: Partial<BatchExportItem>) => {
    setItems(items.map((item, i) => (i === index ? { ...item, ...updates } : item)))
  }

  const clearAll = () => {
    setItems([])
    setError(null)
  }

  const handleExport = async () => {
    if (items.length === 0) {
      setError('Add at least one item to export')
      return
    }

    setIsExporting(true)
    setError(null)

    try {
      // Update all items with current preset and format
      const exportItems = items.map((item) => ({
        ...item,
        config: { ...item.config, preset, format },
      }))

      const blob = await apiClient.exportBatch({ exports: exportItems })
      downloadBlob(blob, `figures_export.zip`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className={`bg-black/40 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Package className="w-5 h-5 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-200 tracking-tight">Batch Export</h3>
          {items.length > 0 && (
            <span className="px-2 py-0.5 bg-white/10 text-white text-xs border border-gray-700">
              {items.length} items
            </span>
          )}
        </div>
        {items.length > 0 && (
          <button
            onClick={clearAll}
            className="text-xs text-gray-400 hover:text-red-400 transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Global settings */}
      <div className="px-4 py-3 border-b border-gray-800 space-y-3">
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wide">Preset</label>
            <select
              value={preset}
              onChange={(e) => setPreset(e.target.value as PresetName)}
              className="w-full px-3 py-2 bg-black/40 border border-gray-700 text-sm text-gray-200 focus:border-gray-500 focus:outline-none transition-colors"
            >
              {Object.entries(PRESET_DISPLAY_INFO).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wide">Format</label>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as ExportFormat)}
              className="w-full px-3 py-2 bg-black/40 border border-gray-700 text-sm text-gray-200 focus:border-gray-500 focus:outline-none transition-colors"
            >
              <option value="png">PNG</option>
              <option value="pdf">PDF</option>
              <option value="svg">SVG</option>
            </select>
          </div>
        </div>
      </div>

      {/* Quick add buttons */}
      <div className="px-4 py-3 border-b border-gray-800">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={addHeatmapItem}
            className="flex items-center gap-1.5 px-2 py-1 text-xs
              bg-white/5 text-gray-300 hover:text-white hover:bg-white/10
              border border-gray-700 hover:border-gray-600 transition-all"
          >
            <Plus className="w-3 h-3" />
            Add Heatmap
          </button>
          <button
            onClick={addHistogramItem}
            className="flex items-center gap-1.5 px-2 py-1 text-xs
              bg-white/5 text-gray-300 hover:text-white hover:bg-white/10
              border border-gray-700 hover:border-gray-600 transition-all"
          >
            <Plus className="w-3 h-3" />
            Add Histogram
          </button>
          <div className="h-6 w-px bg-gray-700" />
          <select
            onChange={(e) => {
              if (e.target.value) {
                addAllHeads(parseInt(e.target.value))
                e.target.value = ''
              }
            }}
            className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none focus:border-gray-500"
            defaultValue=""
          >
            <option value="" disabled>
              Add all heads for layer...
            </option>
            {Array.from({ length: numLayers }, (_, i) => (
              <option key={i} value={i}>
                Layer {i}
              </option>
            ))}
          </select>
          <button
            onClick={addAllLayers}
            className="flex items-center gap-1.5 px-2 py-1 text-xs
              bg-white/5 text-gray-300 hover:text-white hover:bg-white/10
              border border-gray-700 hover:border-gray-600 transition-all"
          >
            <Plus className="w-3 h-3" />
            All Layer Histograms
          </button>
        </div>
      </div>

      {/* Items list */}
      <div className="max-h-64 overflow-y-auto">
        {items.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 text-sm">
            No items added yet. Use the buttons above to add figures to export.
          </div>
        ) : (
          <div className="divide-y divide-gray-800/50">
            {items.map((item, index) => (
              <div key={index} className="px-4 py-2 flex items-center gap-3">
                <span className="text-xs text-gray-500 w-6">{index + 1}.</span>

                {item.type === 'attention-heatmap' ? (
                  <>
                    <span className="text-xs text-gray-300 bg-white/10 px-2 py-0.5 border border-gray-700">
                      Heatmap
                    </span>
                    <select
                      value={item.layer}
                      onChange={(e) => updateItem(index, { layer: parseInt(e.target.value) })}
                      className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none"
                    >
                      {Array.from({ length: numLayers }, (_, i) => (
                        <option key={i} value={i}>
                          L{i}
                        </option>
                      ))}
                    </select>
                    <select
                      value={item.head}
                      onChange={(e) => updateItem(index, { head: parseInt(e.target.value) })}
                      className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none"
                    >
                      {Array.from({ length: numHeads }, (_, i) => (
                        <option key={i} value={i}>
                          H{i}
                        </option>
                      ))}
                    </select>
                    <select
                      value={item.colormap || 'inferno'}
                      onChange={(e) => updateItem(index, { colormap: e.target.value })}
                      className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none"
                    >
                      {COLORMAP_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </>
                ) : (
                  <>
                    <span className="text-xs text-purple-300 bg-purple-500/10 px-2 py-0.5 border border-purple-500/30">
                      Histogram
                    </span>
                    <select
                      value={item.layer ?? ''}
                      onChange={(e) =>
                        updateItem(index, {
                          layer: e.target.value ? parseInt(e.target.value) : undefined,
                        })
                      }
                      className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none"
                    >
                      <option value="">All</option>
                      {Array.from({ length: numLayers }, (_, i) => (
                        <option key={i} value={i}>
                          L{i}
                        </option>
                      ))}
                    </select>
                    <select
                      value={item.activation_type || 'all'}
                      onChange={(e) =>
                        updateItem(index, { activation_type: e.target.value as ActivationType })
                      }
                      className="px-2 py-1 text-xs bg-black/40 text-gray-300 border border-gray-700 focus:outline-none"
                    >
                      <option value="all">All</option>
                      <option value="attn_output">Attn</option>
                      <option value="ff_output">FF</option>
                    </select>
                  </>
                )}

                <div className="flex-1" />

                <button
                  onClick={() => removeItem(index)}
                  className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-2 text-sm text-red-400 bg-red-500/10 border-t border-red-500/30">
          {error}
        </div>
      )}

      {/* Export button */}
      <div className="px-4 py-3 border-t border-gray-800">
        <button
          onClick={handleExport}
          disabled={isExporting || items.length === 0}
          className="w-full flex items-center justify-center gap-2 px-4 py-2
            text-sm font-medium bg-white/10 hover:bg-white/15 disabled:bg-gray-800 disabled:cursor-not-allowed text-white border border-gray-700 hover:border-gray-600 transition-all"
        >
          {isExporting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Exporting {items.length} figures...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Export as ZIP ({items.length} figures)
            </>
          )}
        </button>
      </div>
    </div>
  )
}
