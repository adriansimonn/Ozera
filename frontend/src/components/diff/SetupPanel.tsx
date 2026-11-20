/**
 * The Model Diff page's side panel: the pair of models, the texts both read (a built-in sample
 * or the user's own), the layers run's estimate and button, and where else pairs are compared.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { DiffLab } from '../../hooks/useDiffLab'
import { MAX_TEXTS, MIN_TEXTS } from '../../hooks/useDiffLab'
import { Dropdown, type DropdownGroup } from '../common/Dropdown'
import { Spinner } from '../common/Spinner'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { estimateText, pairLabel } from './diffUtils'

const KIND_LABELS = { custom: 'Your models', instruct: 'Base and instruct' } as const

export function PairPicker({ lab, disabled }: { lab: DiffLab; disabled?: boolean }) {
  const groups: DropdownGroup[] = lab.loadingPairs
    ? [{ label: 'Pairs', options: [{ value: '', label: 'Loading pairs…' }] }]
    : (['custom', 'instruct'] as const)
        .map((kind) => ({
          label: KIND_LABELS[kind],
          options: lab.pairs.filter((p) => p.kind === kind).map((p) => ({ value: `${p.base}:${p.other}`, label: pairLabel(p) })),
        }))
        .filter((group) => group.options.length > 0)
  return (
    <div data-assist="diff.model">
      <Dropdown value={lab.pairId} onChange={lab.setPairId} groups={groups} disabled={disabled || lab.loadingPairs} />
      {lab.pair && (
        <div className="pl-hint df-pair">
          <span title="The reference: what the other model is measured against">Base · {lab.pair.base_name}</span>
          <span>Other · {lab.pair.other_name}</span>
        </div>
      )}
    </div>
  )
}

export function TextsPanel({ lab, disabled }: { lab: DiffLab; disabled?: boolean }) {
  const colors = useLabColors()
  const [preview, setPreview] = useState(false)
  const few = lab.ownCount > 0 && lab.ownCount < MIN_TEXTS
  const many = lab.ownCount > MAX_TEXTS

  return (
    <div className="df-texts">
      <Segmented
        ariaLabel="Texts source"
        value={lab.source}
        onChange={lab.setSource}
        disabled={disabled}
        options={[
          { value: 'builtin', label: 'Built-in' },
          { value: 'own', label: 'Your own' },
        ]}
      />

      {lab.source === 'builtin' ? (
        <>
          <div className="df-samples" data-assist="diff.sample">
            {lab.samples.map((sample) => {
              const active = sample.id === lab.sampleId
              return (
                <div key={sample.id} className={`df-sample${active ? ' is-active' : ''}`}>
                  <button type="button" className="df-sample__pick" onClick={() => lab.setSampleId(sample.id)} disabled={disabled} aria-pressed={active}>
                    {sample.name}
                  </button>
                  {active && (
                    <div className="df-sample__detail">
                      <div className="pl-hint">{sample.description}</div>
                      <button type="button" className="pl-link" onClick={() => setPreview(!preview)}>
                        {preview ? 'Hide examples' : 'Show examples'}
                      </button>
                      {preview && (
                        <ul className="df-examples">
                          {sample.examples.map((text) => (
                            <li key={text}>{text}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          <div className="df-setting-row">
            <div data-assist="diff.count">
              <div className="pl-control-label">Texts</div>
              <input
                className="pl-input"
                type="number"
                min={MIN_TEXTS}
                max={lab.sample?.num_texts ?? MAX_TEXTS}
                value={lab.count}
                disabled={disabled}
                onChange={(e) => lab.setCount(Math.max(MIN_TEXTS, Math.min(lab.sample?.num_texts ?? MAX_TEXTS, Math.floor(Number(e.target.value) || MIN_TEXTS))))}
                style={{ width: '5.5rem' }}
              />
            </div>
            <div data-assist="diff.seed">
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
        <div className="df-own" data-assist="diff.texts">
          <textarea
            className="pl-textarea"
            value={lab.own}
            onChange={(e) => lab.setOwn(e.target.value)}
            placeholder="One text per line"
            rows={7}
            disabled={disabled}
            aria-label="Your texts, one per line"
          />
          <div className="pl-hint" style={few || many ? { color: colors.warning } : undefined}>
            {few ? `${lab.ownCount} texts: at least ${MIN_TEXTS}` : many ? `${lab.ownCount} texts: at most ${MAX_TEXTS}` : `${lab.ownCount} texts`}
          </div>
        </div>
      )}

      {lab.pair?.chat_template && (
        <label
          className="pl-check-row df-template"
          title="Both models read each text as a user message in the instruct model's chat template"
          data-assist="diff.chat_template"
        >
          <input type="checkbox" checked={lab.chatTemplate} disabled={disabled} onChange={(e) => lab.setChatTemplate(e.target.checked)} />
          Chat template
        </label>
      )}
    </div>
  )
}

export function CompareSettings({ lab }: { lab: DiffLab }) {
  const colors = useLabColors()
  const estimate = lab.estimate
  const ready = lab.pair != null && lab.textSpec() != null

  return (
    <div className="df-settings">
      <div className="df-estimate">
        <span>Estimate</span>
        <span style={estimate && !estimate.within_limit ? { color: colors.warning } : undefined}>
          {!lab.isAuthenticated ? 'Sign in to run' : estimate ? estimateText(estimate) : '–'}
        </span>
      </div>
      <button
        type="button"
        className={`oz-btn oz-btn--block${lab.running ? '' : ' oz-btn--arrow'}`}
        data-assist="diff.run_button"
        onClick={lab.runLayers}
        disabled={lab.running || !ready || (estimate != null && !estimate.within_limit)}
      >
        {lab.running ? (
          <>
            <Spinner />
            Comparing…
          </>
        ) : (
          'Compare layers'
        )}
      </button>
    </div>
  )
}

/** Where else a pair's two models are compared: the labs that already do it. */
export function AlsoLinks() {
  return (
    <ul className="df-also">
      <li>
        <Link className="pl-inline-link" to="/probes" title="The Generalize view trains probes on one model of a pair and tests them on the other">
          Probes across a pair
        </Link>
        <span className="pl-hint">Probe Lab · Generalize</span>
      </li>
      <li>
        <Link className="pl-inline-link" to="/geometry" title="The Compare view aligns a concept’s shape in one model onto its shape in another">
          Concept shapes across a pair
        </Link>
        <span className="pl-hint">Geometry Lab · Compare</span>
      </li>
    </ul>
  )
}
