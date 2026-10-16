/**
 * Where a run reads each example, its cost estimate and the button that starts it, and the
 * runs made so far (kept for comparing).
 */

import { Shapes } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRead } from '../../types/geometry'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { READ_LABELS, READ_TITLES, estimateText } from './geometryUtils'

export function GeometryRunSettings({ lab }: { lab: GeometryLab }) {
  const colors = useLabColors()
  const disabled = lab.running
  const estimate = lab.estimate

  return (
    <div className="gl-settings">
      <div className="gl-setting">
        <div className="pl-control-label">Read at</div>
        <Segmented
          ariaLabel="Read position"
          value={lab.read}
          onChange={lab.setRead}
          disabled={disabled}
          options={(['value_last', 'value_mean', 'template_last'] as GeometryRead[]).map((r) => ({
            value: r,
            label: READ_LABELS[r],
            title: READ_TITLES[r],
          }))}
        />
      </div>

      <div className="gl-setting-row">
        {lab.model?.is_instruct && (
          <label className="pl-check-row" title="Read each example as a user message in the model's chat template">
            <input type="checkbox" checked={lab.chatTemplate} disabled={disabled} onChange={(e) => lab.setChatTemplate(e.target.checked)} />
            Chat template
          </label>
        )}
        <div>
          <div className="pl-control-label" title="Seed for the held-out template folds and the shuffled baselines">
            Seed
          </div>
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

      <div className="gl-estimate">
        <span>Estimate</span>
        <span style={estimate && !estimate.within_limit ? { color: colors.warning } : undefined}>
          {!lab.isAuthenticated ? 'Sign in to run' : estimate ? `${estimate.num_examples} examples · ${estimateText(estimate)}` : '–'}
        </span>
      </div>

      <button
        type="button"
        className="pl-train-button"
        onClick={lab.start}
        disabled={lab.running || !lab.modelId || !lab.conceptSpec || (estimate != null && !estimate.within_limit)}
      >
        {lab.running ? (
          <>
            <span className="pl-spinner" />
            Reading every layer…
          </>
        ) : (
          <>
            <Shapes className="pl-icon" aria-hidden="true" />
            Run
          </>
        )}
      </button>

      {lab.runs.length > 0 && (
        <div className="gl-runs">
          <div className="pl-control-label">Runs</div>
          {lab.runs.map((r) => {
            const active = r.id === lab.runId
            return (
              <button
                key={r.id}
                type="button"
                className="gl-run"
                aria-pressed={active}
                onClick={() => lab.selectRun(r.id)}
                style={{ borderColor: active ? colors.text : colors.border, background: active ? colors.hover : 'transparent', color: colors.text }}
              >
                <span>{r.label}</span>
                <span className="pl-hint">${r.run.cost.toFixed(2)}</span>
              </button>
            )
          })}
        </div>
      )}

      <style>{`
        .geometry-lab .gl-setting { margin-bottom: 0.875rem; }
        .geometry-lab .gl-setting-row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-end; }
        .geometry-lab .gl-estimate { display: flex; justify-content: space-between; gap: 0.75rem; margin: 0.875rem 0 0.75rem; font-size: 0.75rem; }
        .geometry-lab .gl-estimate span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
        .geometry-lab .gl-runs { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.3rem; }
        .geometry-lab .gl-run {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 0.5rem;
          padding: 0.4rem 0.6rem;
          border: 1px solid;
          font-size: 0.75rem;
          cursor: pointer;
          text-align: left;
        }
      `}</style>
    </div>
  )
}
