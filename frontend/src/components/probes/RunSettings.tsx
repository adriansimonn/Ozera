/**
 * Run settings for a probe training run, with its cost estimate and the button that starts it.
 */

import { FlaskConical } from 'lucide-react'
import type { ProbeLab } from '../../hooks/useProbeLab'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { estimateText } from './probeUtils'

export function RunSettings({ lab }: { lab: ProbeLab }) {
  const colors = useLabColors()
  const disabled = lab.training
  const instruct = lab.model?.is_instruct ?? false
  const estimate = lab.estimate

  return (
    <div className="pl-settings">
      {instruct && (
        <div className="pl-setting">
          <label className="pl-check-row">
            <input
              type="checkbox"
              checked={lab.chatTemplate}
              disabled={disabled}
              onChange={(e) => lab.setChatTemplate(e.target.checked)}
            />
            Use chat template
          </label>
          {lab.chatTemplate && (
            <div style={{ marginTop: '0.5rem' }}>
              <div className="pl-control-label">Read</div>
              <Segmented
                ariaLabel="Read position"
                value={lab.readSpan}
                onChange={lab.setReadSpan}
                disabled={disabled}
                options={[
                  { value: 'text', label: 'User text', title: "Pool over the text's own tokens; last token = the text's last word" },
                  { value: 'prompt', label: 'Whole prompt', title: 'Pool from the text to the end of the template; last token = where the reply starts' },
                ]}
              />
            </div>
          )}
        </div>
      )}

      <div className="pl-setting-row">
        <div>
          <div className="pl-control-label">Test split</div>
          <Segmented
            ariaLabel="Test split"
            value={String(lab.testFraction)}
            onChange={(v) => lab.setTestFraction(Number(v))}
            disabled={disabled}
            options={['0.1', '0.2', '0.3', '0.4'].map((v) => ({ value: v, label: `${Number(v) * 100}%` }))}
          />
        </div>
        <div>
          <div className="pl-control-label">Seed</div>
          <input
            className="pl-input"
            type="number"
            min={0}
            value={lab.seed}
            disabled={disabled}
            onChange={(e) => lab.setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
            style={{ width: '5.5rem', borderColor: colors.border, background: colors.panel, color: colors.text }}
          />
        </div>
      </div>

      <div className="pl-estimate">
        <span>Estimate</span>
        <span style={estimate && !estimate.within_limit ? { color: colors.warning } : undefined}>
          {!lab.isAuthenticated ? 'Sign in to train' : estimate ? estimateText(estimate) : '–'}
        </span>
      </div>

      <button
        type="button"
        className="pl-train-button"
        onClick={lab.train}
        disabled={lab.training || !lab.modelId || !lab.stats || (estimate != null && !estimate.within_limit)}
      >
        {lab.training ? (
          <>
            <span className="pl-spinner" />
            Training probes…
          </>
        ) : (
          <>
            <FlaskConical className="pl-icon" aria-hidden="true" />
            Train probes
          </>
        )}
      </button>

      <style>{`
        .probe-lab .pl-setting { margin-bottom: 0.875rem; }
        .probe-lab .pl-setting-row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-end; }
        .probe-lab .pl-estimate { display: flex; justify-content: space-between; gap: 0.75rem; margin: 0.875rem 0 0.75rem; font-size: 0.75rem; }
        .probe-lab .pl-estimate span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
      `}</style>
    </div>
  )
}
