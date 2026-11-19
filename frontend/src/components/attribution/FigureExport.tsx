/**
 * Export a chart of the Attribution page as a figure: the page sends the chart's data and the
 * backend draws it (PNG, PDF or SVG, in a publication preset). The dialog opens at the app's
 * root, clear of the typography of whatever holds the button.
 */

import { useState } from 'react'
import { createPortal } from 'react-dom'
import { Download, X } from 'lucide-react'
import { apiClient } from '../../api/client'
import type { DataFigureRequest } from '../../types/attribution'
import type { ExportFormat, PresetName } from '../../types/export'
import { PRESET_DISPLAY_INFO } from '../../types/export'
import { downloadBlob } from '../../utils/svgExport'
import { Dropdown } from '../common/Dropdown'
import { Spinner } from '../common/Spinner'

interface Props {
  /** The figure as shown now (its title can be changed in the dialog) */
  figure: () => DataFigureRequest
  filename: string
  label?: string
}

export function FigureExport({ figure, filename, label = 'Export figure' }: Props) {
  const [open, setOpen] = useState(false)
  const [format, setFormat] = useState<ExportFormat>('png')
  const [preset, setPreset] = useState<PresetName>('default')
  const [title, setTitle] = useState('')
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = () => {
    setTitle(figure().title ?? '')
    setError(null)
    setOpen(true)
  }

  const run = async () => {
    setExporting(true)
    setError(null)
    try {
      const blob = await apiClient.exportDataFigure({ ...figure(), title, config: { preset, format } })
      downloadBlob(blob, `${filename.replace(/[^\w.-]+/g, '_')}.${format}`)
      setOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  const root = typeof document !== 'undefined' ? (document.querySelector('.oz-app') ?? document.body) : null

  return (
    <>
      <button type="button" className="pl-icon-button" title={label} aria-label={label} onClick={start}>
        <Download className="pl-icon" />
      </button>
      {open &&
        root &&
        createPortal(
          <div className="oz-overlay" onClick={() => setOpen(false)}>
            <div className="oz-modal at-export" role="dialog" aria-modal="true" aria-label={label} onClick={(e) => e.stopPropagation()}>
              <button type="button" onClick={() => setOpen(false)} className="oz-icon-btn oz-modal__close" aria-label="Close">
                <X />
              </button>
              <h2 className="oz-modal__title">Export figure</h2>
              <div className="at-export__fields">
                <div>
                  <label className="oz-label">Output format</label>
                  <div className="oz-chips" role="radiogroup" aria-label="Output format">
                    {(['png', 'pdf', 'svg'] as ExportFormat[]).map((f) => (
                      <button key={f} type="button" role="radio" aria-checked={format === f} onClick={() => setFormat(f)} className="oz-chip">
                        {f.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="oz-label">Publication preset</label>
                  <Dropdown
                    value={preset}
                    onChange={(v) => setPreset(v as PresetName)}
                    options={Object.entries(PRESET_DISPLAY_INFO).map(([key, info]) => ({ value: key, label: `${info.label} - ${info.description}` }))}
                  />
                </div>
                <div>
                  <label className="oz-label">Title</label>
                  <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} className="oz-input" maxLength={200} />
                </div>
                {error && (
                  <div className="oz-note oz-note--error" role="alert">
                    <span>{error}</span>
                  </div>
                )}
              </div>
              <div className="oz-modal__actions">
                <button type="button" onClick={() => setOpen(false)} className="oz-btn oz-btn--ghost">
                  Cancel
                </button>
                <button type="button" onClick={run} disabled={exporting} className={`oz-btn${exporting ? '' : ' oz-btn--arrow'}`}>
                  {exporting && <Spinner />}
                  {exporting ? 'Exporting…' : `Export ${format.toUpperCase()}`}
                </button>
              </div>
              <style>{`
                .oz-app .at-export { max-width: 34rem; }
                .at-export .at-export__fields { display: flex; flex-direction: column; gap: 1.25rem; }
                .at-export .oz-modal__actions { margin-top: 1.75rem; }
              `}</style>
            </div>
          </div>,
          root,
        )}
    </>
  )
}
