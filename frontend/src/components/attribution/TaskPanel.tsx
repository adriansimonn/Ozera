/**
 * Choosing the prompt pairs a sweep reads (a built-in task, or an uploaded file of pairs), and
 * the sweep's settings, estimate and button.
 */

import { useRef, useState } from 'react'
import { X } from 'lucide-react'
import type { AttributionLab } from '../../hooks/useAttributionLab'
import type { PairRow } from '../../types/attribution'
import { Spinner } from '../common/Spinner'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { estimateText } from './attributionUtils'

function PairExample({ pair }: { pair: PairRow }) {
  return (
    <li>
      <span className="at-pair__prompt">
        {pair.clean}
        {pair.correct != null && <b>{pair.correct}</b>}
      </span>
      <span className="at-pair__prompt at-pair__prompt--corrupted">
        {pair.corrupted}
        {pair.incorrect != null && <b>{pair.incorrect}</b>}
      </span>
    </li>
  )
}

export function TaskPanel({ lab, disabled }: { lab: AttributionLab; disabled?: boolean }) {
  const colors = useLabColors()
  const input = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState(false)
  const upload = lab.upload

  return (
    <div className="at-task" data-assist="attribution.task">
      {/* The switch stands for the user's own pairs (the upload panel only shows once it's chosen) */}
      <div data-assist="attribution.rows" style={{ display: 'inline-block' }}>
        <Segmented
          ariaLabel="Pairs source"
          value={lab.source}
          onChange={lab.setSource}
          disabled={disabled}
          options={[
            { value: 'builtin', label: 'Built-in' },
            { value: 'upload', label: 'Upload' },
          ]}
        />
      </div>

      {lab.source === 'builtin' ? (
        <>
          <div className="at-tasks">
            {lab.tasks.map((task) => {
              const active = task.id === lab.taskId
              return (
                <div key={task.id} className={`at-task-item${active ? ' is-active' : ''}`}>
                  <button type="button" className="at-task-pick" onClick={() => lab.setTaskId(task.id)} disabled={disabled} aria-pressed={active}>
                    {task.name}
                  </button>
                  {active && (
                    <div className="at-task-detail">
                      <div className="pl-hint">{task.description}</div>
                      {task.examples.length > 0 && (
                        <button type="button" className="pl-link" onClick={() => setPreview(!preview)}>
                          {preview ? 'Hide examples' : 'Show examples'}
                        </button>
                      )}
                      {preview && task.examples.length > 0 && (
                        <ul className="at-pairs">
                          {task.examples.map((pair) => (
                            <PairExample key={pair.clean + pair.corrupted} pair={pair} />
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          <div className="at-setting-row">
            <div data-assist="attribution.pairs">
              <div className="pl-control-label">Pairs</div>
              <input
                className="pl-input"
                type="number"
                min={1}
                max={lab.task?.max_pairs ?? 200}
                value={lab.pairCount}
                disabled={disabled}
                onChange={(e) => lab.setPairCount(Math.max(1, Math.min(lab.task?.max_pairs ?? 200, Math.floor(Number(e.target.value) || 1))))}
                style={{ width: '5.5rem' }}
              />
            </div>
            <div data-assist="attribution.seed">
              <div className="pl-control-label">Seed</div>
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
        </>
      ) : (
        <div className="at-upload">
          <div className="pl-hint" style={{ margin: '0.75rem 0 0.5rem' }}>
            CSV, TSV or JSONL with <code>clean</code> and <code>corrupted</code> columns (<code>correct</code>, <code>incorrect</code>, <code>group</code>{' '}
            optional)
          </div>
          <input
            ref={input}
            type="file"
            accept=".csv,.tsv,.jsonl,.json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) lab.parseFile(file)
              e.target.value = ''
            }}
          />
          {upload ? (
            <div className="at-parsed">
              <div className="at-parsed-head">
                <span className="at-parsed-name" title={upload.filename}>
                  {upload.filename}
                </span>
                <button type="button" className="pl-icon-button" onClick={() => lab.setUpload(null)} disabled={disabled} aria-label="Remove pairs">
                  <X className="pl-icon" />
                </button>
              </div>
              <div className="pl-hint">
                {upload.rows.length} pairs · {upload.has_answers ? 'with answers' : 'no answers (KL)'}
              </div>
              {upload.warnings.map((warning) => (
                <div key={warning} className="pl-hint" style={{ color: colors.warning }}>
                  {warning}
                </div>
              ))}
              <ul className="at-pairs">
                {upload.rows.slice(0, 2).map((pair) => (
                  <PairExample key={pair.clean + pair.corrupted} pair={pair} />
                ))}
              </ul>
            </div>
          ) : (
            <button type="button" className="pl-button" disabled={lab.parsing || disabled} onClick={() => input.current?.click()} style={{ width: '100%' }}>
              {lab.parsing && <Spinner />}
              Choose a file
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export function SweepSettings({ lab }: { lab: AttributionLab }) {
  const colors = useLabColors()
  const estimate = lab.estimate
  const dropped = estimate?.dropped.length ?? 0
  const ready = lab.modelId && lab.pairSpec() != null

  return (
    <div className="at-settings">
      <div data-assist="attribution.metric">
        <div className="pl-control-label">Metric</div>
        <Segmented
          ariaLabel="Metric"
          value={lab.metric}
          onChange={lab.setMetric}
          disabled={lab.running}
          options={[
            {
              value: 'logit_diff',
              label: 'Logit diff',
              disabled: !lab.hasAnswers,
              title: 'The correct answer’s logit minus the incorrect answer’s, at the last position',
            },
            { value: 'kl', label: 'KL', title: 'How far the next-token prediction is from the clean run’s' },
          ]}
        />
      </div>

      <div className="at-estimate">
        <span>Estimate</span>
        <span style={estimate && !estimate.within_limit ? { color: colors.warning } : undefined}>
          {!lab.isAuthenticated ? 'Sign in to run' : estimate ? estimateText(estimate) : '–'}
        </span>
      </div>
      {estimate && dropped > 0 && (
        <div className="pl-hint at-dropped" style={{ color: colors.warning }} title={estimate.dropped[0].reason}>
          {dropped} of {estimate.num_pairs + dropped} pairs can’t be used on this model
        </div>
      )}

      <button
        type="button"
        className={`oz-btn oz-btn--block${lab.running ? '' : ' oz-btn--arrow'}`}
        data-assist="attribution.run_button"
        onClick={lab.runSweep}
        disabled={lab.running || !ready || (estimate != null && !estimate.within_limit)}
      >
        {lab.running ? (
          <>
            <Spinner />
            Running sweep…
          </>
        ) : (
          'Run sweep'
        )}
      </button>
    </div>
  )
}
