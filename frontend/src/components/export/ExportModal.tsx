/**
 * Modal for configuring and exporting publication-ready figures.
 */

import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { Dropdown } from '../common/Dropdown'
import { apiClient } from '../../api/client'
import { downloadBlob } from '../../utils/svgExport'
import type {
  ExportFormat,
  PresetName,
  ExportConfig,
  AttentionHeatmapExportRequest,
  LogitAttributionExportRequest,
  LogitLensExportRequest,
  MultiHeadHeatmapExportRequest,
} from '../../types/export'
import { COLORMAP_OPTIONS, PRESET_DISPLAY_INFO } from '../../types/export'
import type { ExportType } from './ExportButton'
import { Spinner } from '../common/Spinner'

interface ExportModalProps {
  isOpen: boolean
  onClose: () => void
  exportType: ExportType
  activationId: string
  layer?: number
  head?: number
  initialTokenLabels?: 'text' | 'number'
  /** Logit lens: what the view's cells are colored by */
  initialColorBy?: 'target' | 'final'
  /** Logit lens and attribution: the selected position, and how many the sequence has */
  position?: number
  positions?: number
}

// A lens figure wider than this many positions doesn't fit its tokens in the cells
const LENS_WINDOW = 24

export function ExportModal({
  isOpen,
  onClose,
  exportType,
  activationId,
  layer = 0,
  head = 0,
  initialTokenLabels,
  initialColorBy,
  position = 0,
  positions,
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

  // Logit lens specific: what colors the cells, and the positions shown (first to last)
  const [colorBy, setColorBy] = useState<'target' | 'final'>('target')
  const [showTokens, setShowTokens] = useState(true)
  const [firstPosition, setFirstPosition] = useState('0')
  const [lastPosition, setLastPosition] = useState('0')

  // Logit attribution specific
  const [attributionPosition, setAttributionPosition] = useState('0')

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
      if (initialColorBy) setColorBy(initialColorBy)
      // The window of positions ending at the selected one, or the whole sequence if it's short
      const count = positions ?? position + 1
      const end = count <= LENS_WINDOW ? count - 1 : Math.min(count - 1, Math.max(position, LENS_WINDOW - 1))
      setLastPosition(String(end))
      setFirstPosition(String(count <= LENS_WINDOW ? 0 : Math.max(0, end - LENS_WINDOW + 1)))
      setAttributionPosition(String(position))
    }
  }, [isOpen, initialTokenLabels, initialColorBy, position, positions])

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
      } else if (exportType === 'logit-lens') {
        const last = (positions ?? Infinity) - 1
        const start = Math.max(0, Math.min(last, parseInt(firstPosition) || 0))
        const end = Math.max(start, Math.min(last, parseInt(lastPosition) || 0))
        const request: LogitLensExportRequest = {
          activation_id: activationId,
          color_by: colorBy,
          start,
          end: end + 1,
          config,
          colormap,
          show_colorbar: showColorbar,
          show_tokens: showTokens,
          token_labels: tokenLabels,
        }
        if (customTitle) request.title = customTitle

        blob = await apiClient.exportLogitLens(request)
        filename = `logit_lens_${start}-${end}.${format}`
      } else {
        const at = Math.max(0, Math.min((positions ?? Infinity) - 1, parseInt(attributionPosition) || 0))
        const request: LogitAttributionExportRequest = {
          activation_id: activationId,
          position: at,
          config,
        }
        if (customTitle) request.title = customTitle

        blob = await apiClient.exportLogitAttribution(request)
        filename = `logit_attribution_${at}.${format}`
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
      case 'logit-lens':
        return 'Export Logit Lens'
      case 'logit-attribution':
        return 'Export Logit Attribution'
      default:
        return 'Export Figure'
    }
  }

  return (
    <div className="oz-overlay" onClick={onClose}>
      {/* Modal */}
      <div className="oz-modal export-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} className="oz-icon-btn oz-modal__close" aria-label="Close">
          <X />
        </button>
        <h2 className="oz-modal__title">{getTitle()}</h2>

        {/* Content */}
        <div className="space-y-5">
          {/* Format Selection */}
          <div>
            <label className="oz-label">
              Output format
            </label>
            <div className="oz-chips" role="radiogroup" aria-label="Output format">
              {(['png', 'pdf', 'svg'] as ExportFormat[]).map((f) => (
                <button key={f} role="radio" aria-checked={format === f} onClick={() => setFormat(f)} className="oz-chip">
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Preset Selection */}
          <div>
            <label className="oz-label">
              Publication preset
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
          {(exportType === 'attention-heatmap' || exportType === 'multi-head-heatmap' || exportType === 'logit-lens') && (
            <>
              {/* Colormap */}
              <div>
                <label className="oz-label">
                  Colormap
                </label>
                <Dropdown
                  value={colormap}
                  onChange={(v) => setColormap(v)}
                  options={COLORMAP_OPTIONS.map((opt) => ({ value: opt.value, label: opt.label }))}
                />
              </div>

              {/* Logit lens options */}
              {exportType === 'logit-lens' && (
                <>
                  <div>
                    <label className="oz-label">
                      Color by
                    </label>
                    <Dropdown
                      value={colorBy}
                      onChange={(v) => setColorBy(v as 'target' | 'final')}
                      options={[
                        { value: 'target', label: 'Next token' },
                        { value: 'final', label: 'Final prediction' },
                      ]}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="oz-label">
                        First token
                      </label>
                      <input
                        type="number"
                        min={0}
                        max={positions != null ? positions - 1 : undefined}
                        value={firstPosition}
                        onChange={(e) => setFirstPosition(e.target.value)}
                        className="oz-input oz-input--number"
                      />
                    </div>
                    <div>
                      <label className="oz-label">
                        Last token
                      </label>
                      <input
                        type="number"
                        min={0}
                        max={positions != null ? positions - 1 : undefined}
                        value={lastPosition}
                        onChange={(e) => setLastPosition(e.target.value)}
                        className="oz-input oz-input--number"
                      />
                    </div>
                  </div>
                  <label className="flex items-center gap-3 text-sm cursor-pointer" title={`Written when the figure is up to ${LENS_WINDOW} tokens wide`}>
                    <input
                      type="checkbox"
                      checked={showTokens}
                      onChange={(e) => setShowTokens(e.target.checked)}
                    />
                    Top tokens in cells
                  </label>
                </>
              )}

              {/* Heatmap options */}
              {(exportType === 'attention-heatmap' || exportType === 'logit-lens') && (
                <>
                  <div>
                    <label className="oz-label">
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
                  <label className="flex items-center gap-3 text-sm cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showColorbar}
                      onChange={(e) => setShowColorbar(e.target.checked)}
                    />
                    Show colorbar
                  </label>
                </>
              )}

              {/* Multi-head columns */}
              {exportType === 'multi-head-heatmap' && (
                <div>
                  <label className="oz-label">
                    Columns: {cols}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    value={cols}
                    onChange={(e) => setCols(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}
            </>
          )}

          {exportType === 'logit-attribution' && (
            <div>
              <label className="oz-label">
                Token
              </label>
              <input
                type="number"
                min={0}
                max={positions != null ? positions - 1 : undefined}
                value={attributionPosition}
                onChange={(e) => setAttributionPosition(e.target.value)}
                className="oz-input oz-input--number"
              />
            </div>
          )}

          {/* Custom title */}
          <div>
            <label className="oz-label">
              Custom title (optional)
            </label>
            <input
              type="text"
              value={customTitle}
              onChange={(e) => setCustomTitle(e.target.value)}
              placeholder={
                exportType === 'logit-lens'
                  ? 'Logit lens'
                  : exportType === 'logit-attribution'
                    ? 'Logit attribution'
                    : `Layer ${layer}, Head ${head}`
              }
              className="oz-input"
            />
          </div>

          {/* Advanced options toggle */}
          <button onClick={() => setShowAdvanced(!showAdvanced)} className="oz-link text-sm" aria-expanded={showAdvanced}>
            {showAdvanced ? 'Hide' : 'Show'} advanced options
          </button>

          {/* Advanced options */}
          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-rule">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="oz-label">
                    Width (inches)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={customWidth}
                    onChange={(e) => setCustomWidth(e.target.value)}
                    placeholder="Auto"
                    className="oz-input oz-input--number"
                  />
                </div>
                <div>
                  <label className="oz-label">
                    Height (inches)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={customHeight}
                    onChange={(e) => setCustomHeight(e.target.value)}
                    placeholder="Auto"
                    className="oz-input oz-input--number"
                  />
                </div>
                <div>
                  <label className="oz-label">
                    DPI
                  </label>
                  <input
                    type="number"
                    step="50"
                    value={customDpi}
                    onChange={(e) => setCustomDpi(e.target.value)}
                    placeholder="Auto"
                    className="oz-input oz-input--number"
                  />
                </div>
              </div>

              <label className="flex items-center gap-3 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={transparent}
                  onChange={(e) => setTransparent(e.target.checked)}
                />
                Transparent background
              </label>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="oz-note oz-note--error" role="alert">
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="oz-modal__actions export-modal__actions">
          <button onClick={onClose} className="oz-btn oz-btn--ghost">
            Cancel
          </button>
          <button onClick={handleExport} disabled={isExporting} className={`oz-btn${isExporting ? '' : ' oz-btn--arrow'}`}>
            {isExporting ? (
              <>
                <Spinner />
                Exporting...
              </>
            ) : (
              `Export ${format.toUpperCase()}`
            )}
          </button>
        </div>

        <style>{`
          .oz-app .export-modal { max-width: 34rem; }
          .export-modal .export-modal__actions { margin-top: 1.75rem; }
        `}</style>
      </div>
    </div>
  )
}
