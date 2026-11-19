/**
 * Type definitions for paper-ready figure export.
 */

export type ExportFormat = 'png' | 'pdf' | 'svg'

/**
 * Publication preset names.
 */
export type PresetName = 'nature' | 'neurips' | 'iclr' | 'icml' | 'presentation' | 'default'

/**
 * Information about a publication preset.
 */
export interface PresetInfo {
  name: string
  width_inches: number
  height_inches: number
  dpi: number
  font_family: string
  font_size: number
}

/**
 * Common export configuration.
 */
export interface ExportConfig {
  preset?: PresetName
  format?: ExportFormat
  width?: number
  height?: number
  dpi?: number
  transparent?: boolean
}

/**
 * Request to export attention heatmap.
 */
export interface AttentionHeatmapExportRequest {
  activation_id: string
  layer: number
  head: number
  config?: ExportConfig
  colormap?: string
  title?: string
  show_colorbar?: boolean
  token_labels?: 'text' | 'number'
}

/**
 * Request to export multi-head attention heatmap.
 */
export interface MultiHeadHeatmapExportRequest {
  activation_id: string
  layer: number
  heads?: number[]
  config?: ExportConfig
  colormap?: string
  title?: string
  cols?: number
}

/**
 * Request to export a generation's logit lens.
 */
export interface LogitLensExportRequest {
  activation_id: string
  /** Color by the next token's probability (target) or the final prediction's */
  color_by?: 'target' | 'final'
  /** First position shown, and one past the last */
  start?: number
  end?: number
  config?: ExportConfig
  colormap?: string
  title?: string
  show_colorbar?: boolean
  show_tokens?: boolean
  token_labels?: 'text' | 'number'
}

/**
 * Request to export one position's logit attribution.
 */
export interface LogitAttributionExportRequest {
  activation_id: string
  position: number
  config?: ExportConfig
  title?: string
}

/**
 * Single export item for batch export.
 */
export interface BatchExportItem {
  type: 'attention-heatmap'
  activation_id: string
  layer?: number
  head?: number
  config?: ExportConfig
  colormap?: string
  title?: string
}

/**
 * Request for batch export.
 */
export interface BatchExportRequest {
  exports: BatchExportItem[]
}

/**
 * Export modal state.
 */
export interface ExportModalState {
  isOpen: boolean
  exportType: 'attention-heatmap' | 'logit-lens' | 'logit-attribution' | null
  activationId?: string
  layer?: number
  head?: number
}

/**
 * Colormap options for heatmaps.
 */
export const COLORMAP_OPTIONS = [
  { value: 'inferno', label: 'Inferno' },
  { value: 'viridis', label: 'Viridis' },
  { value: 'plasma', label: 'Plasma' },
  { value: 'magma', label: 'Magma' },
  { value: 'cividis', label: 'Cividis' },
  { value: 'Blues', label: 'Blues' },
  { value: 'Reds', label: 'Reds' },
  { value: 'RdBu', label: 'Red-Blue' },
  { value: 'coolwarm', label: 'Coolwarm' },
] as const

/**
 * Preset display information.
 */
export const PRESET_DISPLAY_INFO: Record<PresetName, { label: string; description: string }> = {
  nature: { label: 'Nature', description: '89mm width, 300 DPI, Helvetica' },
  neurips: { label: 'NeurIPS', description: '5.5in width, 300 DPI, Times' },
  iclr: { label: 'ICLR', description: '6in width, 300 DPI, Times' },
  icml: { label: 'ICML', description: '6.75in width, 300 DPI, Times' },
  presentation: { label: 'Presentation', description: '10in width, 150 DPI, Arial' },
  default: { label: 'Default', description: '8in width, 150 DPI' },
}
