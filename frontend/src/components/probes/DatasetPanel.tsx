/**
 * Choosing what to train probes on: a built-in contrastive set or minimal pairs, or an uploaded
 * CSV/JSONL file, plus an optional out-of-distribution (OOD) test set.
 */

import { useRef, useState } from 'react'
import { ArrowLeftRight, Upload, X } from 'lucide-react'
import type { ParsedProbeDataset, ProbeDatasetSummary } from '../../types/probes'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { Segmented } from './chartKit'
import { useProbeColors } from './probeTheme'

const CATEGORY_LABELS: Record<ProbeDatasetSummary['category'], string> = {
  contrastive: 'Contrastive sets',
  minimal_pairs: 'Minimal pairs',
}

export function FileButton({
  label,
  busy,
  disabled,
  onFile,
}: {
  label: string
  busy: boolean
  disabled?: boolean
  onFile: (file: File) => void
}) {
  const colors = useProbeColors()
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <input
        ref={input}
        type="file"
        accept=".csv,.tsv,.jsonl,.json"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        className="pl-button"
        disabled={busy || disabled}
        onClick={() => input.current?.click()}
        style={{ borderColor: colors.border, color: colors.text, width: '100%' }}
      >
        {busy ? (
          <span className="pl-spinner" style={{ borderColor: colors.border, borderTopColor: colors.text }} />
        ) : (
          <Upload className="pl-icon" aria-hidden="true" />
        )}
        {label}
      </button>
    </>
  )
}

function ParsedSummary({
  parsed,
  onSwap,
  onClear,
  disabled,
}: {
  parsed: ParsedProbeDataset
  onSwap: () => void
  onClear: () => void
  disabled?: boolean
}) {
  const colors = useProbeColors()
  const positives = parsed.rows.filter((r) => r.label === 1).length
  return (
    <div className="pl-parsed" style={{ borderColor: colors.border, background: colors.panel }}>
      <div className="pl-parsed-head">
        <span className="pl-parsed-name" title={parsed.filename}>
          {parsed.filename}
        </span>
        <button type="button" className="pl-icon-button" onClick={onClear} disabled={disabled} aria-label="Remove file" style={{ color: colors.text }}>
          <X className="pl-icon" />
        </button>
      </div>
      <div className="pl-parsed-counts">
        <span>
          <span className="pl-legend-dot" style={{ background: colors.classes[0] }} /> {parsed.label_names[0]}:{' '}
          {parsed.rows.length - positives}
        </span>
        <span>
          <span className="pl-legend-dot" style={{ background: colors.classes[1] }} /> {parsed.label_names[1]}: {positives}
        </span>
        <button
          type="button"
          className="pl-icon-button"
          onClick={onSwap}
          disabled={disabled}
          title="Swap which label is the positive class"
          aria-label="Swap positive and negative labels"
          style={{ color: colors.text }}
        >
          <ArrowLeftRight className="pl-icon" />
        </button>
      </div>
      {parsed.warnings.map((warning) => (
        <div key={warning} className="pl-hint" style={{ color: colors.warning }}>
          {warning}
        </div>
      ))}
    </div>
  )
}

export function DatasetPanel({ lab, disabled }: { lab: ProbeLab; disabled?: boolean }) {
  const colors = useProbeColors()
  const [preview, setPreview] = useState<string | null>(null)

  const togglePreview = async (id: string) => {
    if (preview === id) {
      setPreview(null)
      return
    }
    setPreview(id)
    try {
      await lab.loadDatasetDetail(id)
    } catch {
      setPreview(null)
    }
  }

  const detail = preview ? lab.datasetDetails[preview] : null

  return (
    <div className="pl-dataset">
      <Segmented
        ariaLabel="Dataset source"
        value={lab.source}
        onChange={lab.setSource}
        disabled={disabled}
        options={[
          { value: 'builtin', label: 'Built-in' },
          { value: 'upload', label: 'Upload' },
        ]}
      />

      {lab.source === 'builtin' ? (
        <div className="pl-builtin">
          {(['contrastive', 'minimal_pairs'] as const).map((category) => (
            <div key={category}>
              <div className="pl-control-label" style={{ marginTop: '0.75rem' }}>
                {CATEGORY_LABELS[category]}
              </div>
              {lab.datasets
                .filter((d) => d.category === category)
                .map((d) => {
                  const active = d.id === lab.builtinId
                  return (
                    <div key={d.id} className="pl-dataset-item" style={{ borderColor: active ? colors.text : colors.border, background: active ? colors.hover : 'transparent' }}>
                      <button
                        type="button"
                        className="pl-dataset-pick"
                        onClick={() => lab.setBuiltinId(d.id)}
                        disabled={disabled}
                        aria-pressed={active}
                        style={{ color: colors.text }}
                      >
                        <span className="pl-dataset-name">{d.name}</span>
                        <span className="pl-hint">
                          {d.num_rows} examples{d.ood ? ` · ${d.ood.num_rows} OOD` : ''}
                        </span>
                      </button>
                      {active && (
                        <div className="pl-dataset-detail">
                          <div className="pl-hint">{d.description}</div>
                          <div className="pl-parsed-counts">
                            <span>
                              <span className="pl-legend-dot" style={{ background: colors.classes[0] }} /> {d.label_names[0]}
                            </span>
                            <span>
                              <span className="pl-legend-dot" style={{ background: colors.classes[1] }} /> {d.label_names[1]}
                            </span>
                          </div>
                          {d.ood && (
                            <label className="pl-check-row" title={d.ood.description}>
                              <input
                                type="checkbox"
                                checked={lab.useBuiltinOod && !lab.oodUpload}
                                disabled={disabled || lab.oodUpload != null}
                                onChange={(e) => lab.setUseBuiltinOod(e.target.checked)}
                              />
                              <span className="pl-hint">Include its OOD test set</span>
                            </label>
                          )}
                          <button type="button" className="pl-link" onClick={() => togglePreview(d.id)} style={{ color: colors.accent }}>
                            {preview === d.id ? 'Hide examples' : 'Show examples'}
                          </button>
                          {preview === d.id && detail && (
                            <ul className="pl-preview">
                              {[0, 1, 2, 3, 4, 5].map((i) => {
                                const row = detail.rows[Math.floor((i * detail.rows.length) / 6)]
                                return (
                                  <li key={i}>
                                    <span className="pl-legend-dot" style={{ background: colors.classes[row.label] }} />
                                    <span>{row.text}</span>
                                  </li>
                                )
                              })}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
            </div>
          ))}
        </div>
      ) : (
        <div className="pl-upload">
          <div className="pl-hint" style={{ margin: '0.75rem 0 0.5rem' }}>
            CSV, TSV or JSONL with <code>text</code> and <code>label</code> columns (<code>group</code> optional)
          </div>
          {lab.upload ? (
            <ParsedSummary parsed={lab.upload} onSwap={lab.swapUploadLabels} onClear={lab.clearUpload} disabled={disabled} />
          ) : (
            <FileButton label="Choose a file" busy={lab.parsing === 'train'} disabled={disabled} onFile={(f) => lab.parseFile(f, 'train')} />
          )}
        </div>
      )}

      <div className="pl-control-label" style={{ marginTop: '1rem' }}>
        OOD test set (optional)
      </div>
      {lab.oodUpload ? (
        <ParsedSummary parsed={lab.oodUpload} onSwap={lab.swapOodLabels} onClear={lab.clearOodUpload} disabled={disabled} />
      ) : (
        <FileButton label="Upload OOD set" busy={lab.parsing === 'ood'} disabled={disabled} onFile={(f) => lab.parseFile(f, 'ood')} />
      )}

      <style>{`
        .probe-lab .pl-dataset-item { border: 1px solid; margin-bottom: 0.35rem; }
        .probe-lab .pl-dataset-pick {
          display: flex;
          width: 100%;
          justify-content: space-between;
          align-items: baseline;
          gap: 0.5rem;
          padding: 0.5rem 0.65rem;
          background: none;
          border: none;
          cursor: pointer;
          text-align: left;
        }
        .probe-lab .pl-dataset-pick:disabled { cursor: not-allowed; }
        .probe-lab .pl-dataset-name { font-size: 0.82rem; font-weight: 600; }
        .probe-lab .pl-dataset-detail { padding: 0 0.65rem 0.6rem; display: flex; flex-direction: column; gap: 0.4rem; }
        .probe-lab .pl-preview { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
        .probe-lab .pl-preview li { display: flex; gap: 0.4rem; align-items: baseline; font-size: 0.75rem; }
        .probe-lab .pl-preview .pl-legend-dot { flex-shrink: 0; transform: translateY(-1px); }
        .probe-lab .pl-parsed { border: 1px solid; padding: 0.5rem 0.65rem; }
        .probe-lab .pl-parsed-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
        .probe-lab .pl-parsed-name { font-size: 0.8rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .probe-lab .pl-parsed-counts { display: flex; align-items: center; gap: 0.75rem; font-size: 0.75rem; }
        .probe-lab .pl-parsed-counts > span { display: inline-flex; align-items: center; gap: 0.3rem; }
      `}</style>
    </div>
  )
}
