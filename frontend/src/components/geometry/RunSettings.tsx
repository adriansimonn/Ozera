/**
 * Where a run reads each example, its cost estimate and the button that starts it, and the
 * runs made so far (kept for comparing).
 */

import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { GeometryRead } from '../../types/geometry'
import { Spinner } from '../common/Spinner'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { READ_LABELS, READ_TITLES, estimateText } from './geometryUtils'

export function GeometryRunSettings({ lab }: { lab: GeometryLab }) {
  const colors = useLabColors()
  const disabled = lab.running
  const estimate = lab.estimate

  return (
    <div className="gl-settings">
      <div className="gl-setting" data-assist="geometry.read">
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
          <label className="pl-check-row" title="Read each example as a user message in the model's chat template" data-assist="geometry.chat_template">
            <input type="checkbox" checked={lab.chatTemplate} disabled={disabled} onChange={(e) => lab.setChatTemplate(e.target.checked)} />
            Chat template
          </label>
        )}
        <div data-assist="geometry.seed">
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
            style={{ width: '5.5rem' }}
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
        className={`oz-btn oz-btn--block${lab.running ? '' : ' oz-btn--arrow'}`}
        data-assist="geometry.run_button"
        onClick={lab.start}
        disabled={lab.running || !lab.modelId || !lab.conceptSpec || (estimate != null && !estimate.within_limit)}
      >
        {lab.running ? (
          <>
            <Spinner />
            Reading every layer…
          </>
        ) : (
          'Run'
        )}
      </button>

      {lab.runs.length > 0 && (
        <div className="gl-runs" data-assist="geometry.run">
          <div className="pl-control-label">Runs</div>
          {lab.runs.map((r) => {
            const active = r.id === lab.runId
            return (
              <button
                key={r.id}
                type="button"
                className={`gl-run${active ? ' is-active' : ''}`}
                aria-pressed={active}
                onClick={() => lab.selectRun(r.id)}
              >
                <span>{r.label}</span>
                <span className="pl-hint">${r.run.cost.toFixed(2)}</span>
              </button>
            )
          })}
        </div>
      )}

      <style>{`
        .geometry-lab .gl-setting { margin-bottom: 1rem; }
        .geometry-lab .gl-setting-row { display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: flex-end; }
        .geometry-lab .gl-estimate {
          display: flex; justify-content: space-between; gap: 0.75rem; margin: 1.25rem 0 0.9rem; padding-top: 0.8rem;
          border-top: 1px solid var(--oz-rule); color: var(--oz-fg-dim); font-size: 0.85rem;
        }
        .geometry-lab .gl-estimate span:last-child { color: var(--oz-fg); text-align: right; font-variant-numeric: tabular-nums; }
        /* Runs so far: a ruled list, the one on screen marked */
        .geometry-lab .gl-runs { margin-top: 1.5rem; display: flex; flex-direction: column; }
        .geometry-lab .gl-run {
          position: relative;
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 0.5rem;
          padding: 0.6rem 0;
          border-top: 1px solid var(--oz-rule);
          color: var(--oz-fg-dim);
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          text-align: left;
          transition: color 0.2s ease;
        }
        .geometry-lab .gl-run:last-child { border-bottom: 1px solid var(--oz-rule); }
        .geometry-lab .gl-run:hover, .geometry-lab .gl-run.is-active { color: var(--oz-fg); }
        .geometry-lab .gl-run.is-active::after { content: ''; position: absolute; left: 0; bottom: -1px; width: 100%; height: 1px; background: var(--oz-mark); }
      `}</style>
    </div>
  )
}
