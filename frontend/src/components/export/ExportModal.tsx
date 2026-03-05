/**
 * Modal for configuring and exporting publication-ready figures.
 */

import { useState, useEffect } from 'react'
import { X, Download, Loader2, Settings, FileImage } from 'lucide-react'
import { Dropdown } from '../common/Dropdown'
import { apiClient } from '../../api/client'
import { downloadBlob } from '../../utils/svgExport'
import type {
  ExportFormat,
  PresetName,
  ExportConfig,
  AttentionHeatmapExportRequest,
  ActivationHistogramExportRequest,
  MultiHeadHeatmapExportRequest,
  ActivationType,
} from '../../types/export'
import { COLORMAP_OPTIONS, PRESET_DISPLAY_INFO } from '../../types/export'
import type { ExportType } from './ExportButton'

interface ExportModalProps {
  isOpen: boolean
  onClose: () => void
  exportType: ExportType
  activationId: string
  layer?: number
  head?: number
  initialTokenLabels?: 'text' | 'number'
}

export function ExportModal({
  isOpen,
  onClose,
  exportType,
  activationId,
  layer = 0,
  head = 0,
  initialTokenLabels,
}: ExportModalProps) {
  // Common config state
  const [format, setFormat] = useState<ExportFormat>('png')
  const [preset, setPreset] = useState<PresetName>('default')
  const [transparent, setTransparent] = useState(false)
  const [customWidth, setCustomWidth] = useState<string>('')
  const [customHeight, setCustomHeight] = useState<string>('')
  const [customDpi, setCustomDpi] = useState<string>('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Attention heatmap specific
  const [colormap, setColormap] = useState('inferno')
  const [showColorbar, setShowColorbar] = useState(true)
  const [tokenLabels, setTokenLabels] = useState<'text' | 'number'>('text')
  const [customTitle, setCustomTitle] = useState('')

  // Histogram specific
  const [activationType, setActivationType] = useState<ActivationType>('all')
  const [bins, setBins] = useState(50)
  const [showStats, setShowStats] = useState(true)
  const [logScale, setLogScale] = useState(false)

  // Multi-head specific
  const [cols, setCols] = useState(4)

  // Export state
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      setError(null)
      if (initialTokenLabels) {
        setTokenLabels(initialTokenLabels)
      }
    }
  }, [isOpen, initialTokenLabels])

  if (!isOpen) return null

  const buildConfig = (): ExportConfig => {
    const config: ExportConfig = {
      preset,
      format,
      transparent,
    }

    if (customWidth) config.width = parseFloat(customWidth)
    if (customHeight) config.height = parseFloat(customHeight)
    if (customDpi) config.dpi = parseInt(customDpi)

    return config
  }

  const handleExport = async () => {
    setIsExporting(true)
    setError(null)

    try {
      let blob: Blob
      let filename: string

      const config = buildConfig()

      if (exportType === 'attention-heatmap') {
        const request: AttentionHeatmapExportRequest = {
          activation_id: activationId,
          layer,
          head,
          config,
          colormap,
          show_colorbar: showColorbar,
          token_labels: tokenLabels,
        }
        if (customTitle) request.title = customTitle

        blob = await apiClient.exportAttentionHeatmap(request)
        filename = `attention_L${layer}_H${head}.${format}`
      } else if (exportType === 'multi-head-heatmap') {
        const request: MultiHeadHeatmapExportRequest = {
          activation_id: activationId,
          layer,
          config,
          colormap,
          cols,
        }
        if (customTitle) request.title = customTitle

        blob = await apiClient.exportMultiHeadHeatmap(request)
        filename = `attention_L${layer}_multihead.${format}`
      } else {
        const request: ActivationHistogramExportRequest = {
          activation_id: activationId,
          layer: layer,
          activation_type: activationType,
          config,
          bins,
          show_stats: showStats,
          log_scale: logScale,
        }
        if (customTitle) request.title = customTitle

        blob = await apiClient.exportActivationHistogram(request)
        filename = `histogram_L${layer}_${activationType}.${format}`
      }

      downloadBlob(blob, filename)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const getTitle = () => {
    switch (exportType) {
      case 'attention-heatmap':
        return 'Export Attention Heatmap'
      case 'multi-head-heatmap':
        return 'Export Multi-Head Heatmap'
      case 'activation-histogram':
        return 'Export Activation Histogram'
      default:
        return 'Export Figure'
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-lg mx-4 bg-black/60 backdrop-filter backdrop-blur-xl border border-gray-800">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <FileImage className="w-5 h-5 text-gray-400" />
            <h2 className="text-lg font-semibold text-white tracking-tight">{getTitle()}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-500 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-5 max-h-[70vh] overflow-y-auto">
          {/* Format Selection */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
              Output Format
            </label>
            <div className="flex gap-2">
              {(['png', 'pdf', 'svg'] as ExportFormat[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`px-4 py-2 text-sm font-medium transition-all border ${
                    format === f
                      ? 'bg-white/10 text-white border-gray-600'
                      : 'bg-white/5 text-gray-400 border-gray-700 hover:text-white hover:bg-white/8 hover:border-gray-600'
                  }`}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Preset Selection */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
              Publication Preset
            </label>
            <Dropdown
              value={preset}
              onChange={(v) => setPreset(v as PresetName)}
              options={Object.entries(PRESET_DISPLAY_INFO).map(([key, info]) => ({
                value: key,
                label: `${info.label} - ${info.description}`,
              }))}
            />
          </div>

          {/* Type-specific options */}
          {(exportType === 'attention-heatmap' || exportType === 'multi-head-heatmap') && (
            <>
              {/* Colormap */}
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                  Colormap
                </label>
                <Dropdown
                  value={colormap}
                  onChange={(v) => setColormap(v)}
                  options={COLORMAP_OPTIONS.map((opt) => ({ value: opt.value, label: opt.label }))}
                />
              </div>

              {/* Heatmap options */}
              {exportType === 'attention-heatmap' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                      Labels
                    </label>
                    <Dropdown
                      value={tokenLabels}
                      onChange={(v) => setTokenLabels(v as 'text' | 'number')}
                      options={[
                        { value: 'text', label: 'Token Text' },
                        { value: 'number', label: 'Token Number' },
                      ]}
                    />
                  </div>
                  <label className="flex items-center gap-3 text-sm text-gray-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showColorbar}
                      onChange={(e) => setShowColorbar(e.target.checked)}
                      className="w-4 h-4 border border-gray-600 bg-black/40 text-white focus:ring-0 focus:ring-offset-0"
                    />
                    Show colorbar
                  </label>
                </>
              )}

              {/* Multi-head columns */}
              {exportType === 'multi-head-heatmap' && (
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                    Columns: {cols}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    value={cols}
                    onChange={(e) => setCols(parseInt(e.target.value))}
                    className="w-full h-1 bg-gray-700 appearance-none cursor-pointer"
                  />
                </div>
              )}
            </>
          )}

          {exportType === 'activation-histogram' && (
            <>
              {/* Activation type */}
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                  Activation Type
                </label>
                <Dropdown
                  value={activationType}
                  onChange={(v) => setActivationType(v as ActivationType)}
                  options={[
                    { value: 'all', label: 'All Activations' },
                    { value: 'attn_output', label: 'Attention Output' },
                    { value: 'ff_output', label: 'Feed-Forward Output' },
                  ]}
                />
              </div>

              {/* Bins */}
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                  Bins: {bins}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={bins}
                  onChange={(e) => setBins(parseInt(e.target.value))}
                  className="w-full h-1 bg-gray-700 appearance-none cursor-pointer"
                />
              </div>

              {/* Histogram options */}
              <div className="flex gap-6">
                <label className="flex items-center gap-3 text-sm text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showStats}
                    onChange={(e) => setShowStats(e.target.checked)}
                    className="w-4 h-4 border border-gray-600 bg-black/40 text-white focus:ring-0 focus:ring-offset-0"
                  />
                  Show statistics
                </label>
                <label className="flex items-center gap-3 text-sm text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={logScale}
                    onChange={(e) => setLogScale(e.target.checked)}
                    className="w-4 h-4 border border-gray-600 bg-black/40 text-white focus:ring-0 focus:ring-offset-0"
                  />
                  Log scale
                </label>
              </div>
            </>
          )}

          {/* Custom title */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
              Custom Title (optional)
            </label>
            <input
              type="text"
              value={customTitle}
              onChange={(e) => setCustomTitle(e.target.value)}
              placeholder={
                exportType === 'attention-heatmap'
                  ? `Layer ${layer}, Head ${head}`
                  : 'Activation Distribution'
              }
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-gray-200 placeholder-gray-600 focus:border-gray-500 focus:outline-none transition-colors"
            />
          </div>

          {/* Advanced options toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            <Settings className="w-4 h-4" />
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </button>

          {/* Advanced options */}
          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-gray-800">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wide">
                    Width (inches)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={customWidth}
                    onChange={(e) => setCustomWidth(e.target.value)}
                    placeholder="Auto"
                    className="w-full px-3 py-2 bg-black/40 border border-gray-700 text-sm text-gray-200 placeholder-gray-600 focus:border-gray-500 focus:outline-none transition-colors"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wide">
                    Height (inches)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={customHeight}
                    onChange={(e) => setCustomHeight(e.target.value)}
                    placeholder="Auto"
                    className="w-full px-3 py-2 bg-black/40 border border-gray-700 text-sm text-gray-200 placeholder-gray-600 focus:border-gray-500 focus:outline-none transition-colors"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wide">
                    DPI
                  </label>
                  <input
                    type="number"
                    step="50"
                    value={customDpi}
                    onChange={(e) => setCustomDpi(e.target.value)}
                    placeholder="Auto"
                    className="w-full px-3 py-2 bg-black/40 border border-gray-700 text-sm text-gray-200 placeholder-gray-600 focus:border-gray-500 focus:outline-none transition-colors"
                  />
                </div>
              </div>

              <label className="flex items-center gap-3 text-sm text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={transparent}
                  onChange={(e) => setTransparent(e.target.checked)}
                  className="w-4 h-4 border border-gray-600 bg-black/40 text-white focus:ring-0 focus:ring-offset-0"
                />
                Transparent background
              </label>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 border border-gray-700 hover:border-gray-600 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-white/10 hover:bg-white/15 disabled:bg-gray-800 disabled:cursor-not-allowed text-white border border-gray-700 hover:border-gray-600 transition-all"
          >
            {isExporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Export {format.toUpperCase()}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
