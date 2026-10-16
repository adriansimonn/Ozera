/**
 * Choosing a concept: a built-in set (cyclic or ordinal values with templates) or a custom one
 * (values, optional groups, templates and a structure type).
 */

import { Copy } from 'lucide-react'
import type { GeometryLab } from '../../hooks/useGeometryLab'
import type { ConceptStructure, ConceptSummary } from '../../types/geometry'
import { Segmented } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { STRUCTURE_LABELS, valueColors } from './geometryUtils'

// Values shown as chips before the list is cut short
const CHIPS = 14

const STRUCTURE_TITLES: Record<ConceptStructure, string> = {
  cyclic: 'The values go round (days, months): tests a circle and a line',
  ordinal: 'The values have an order (numbers, years): tests a line and a closed loop',
  categorical: 'No order: only groups (value | group) give a shape to test',
}

function ValueChips({ concept }: { concept: Pick<ConceptSummary, 'structure' | 'values' | 'positions' | 'groups'> }) {
  const colors = useLabColors()
  const fills = valueColors(concept, colors)
  const shown = concept.values.slice(0, CHIPS)
  return (
    <div className="gl-chips">
      {shown.map((value, i) => (
        <span key={value} className="gl-chip" style={{ borderColor: colors.border }} title={concept.groups?.[i]}>
          <span className="pl-legend-dot" style={{ background: fills[i] }} />
          {value}
        </span>
      ))}
      {concept.values.length > CHIPS && <span className="pl-hint">+{concept.values.length - CHIPS} more</span>}
    </div>
  )
}

function Builtin({ lab, disabled }: { lab: GeometryLab; disabled?: boolean }) {
  const colors = useLabColors()
  return (
    <div>
      {(['cyclic', 'ordinal', 'categorical'] as const).map((structure) => {
        const concepts = lab.concepts.filter((c) => c.structure === structure)
        if (concepts.length === 0) return null
        return (
          <div key={structure}>
            <div className="pl-control-label" style={{ marginTop: '0.75rem' }} title={STRUCTURE_TITLES[structure]}>
              {STRUCTURE_LABELS[structure]}
            </div>
            {concepts.map((c) => {
              const active = c.id === lab.builtinId
              return (
                <div key={c.id} className="gl-item" style={{ borderColor: active ? colors.text : colors.border, background: active ? colors.hover : 'transparent' }}>
                  <button
                    type="button"
                    className="gl-pick"
                    onClick={() => lab.setBuiltinId(c.id)}
                    disabled={disabled}
                    aria-pressed={active}
                    title={c.description}
                    style={{ color: colors.text }}
                  >
                    <span className="gl-name">{c.name}</span>
                    <span className="pl-hint">
                      {c.values.length} × {c.num_templates}
                    </span>
                  </button>
                  {active && (
                    <div className="gl-detail">
                      <ValueChips concept={c} />
                      <div className="pl-hint">
                        Tests: {c.hypotheses.map((h) => h.label).join(', ')}
                      </div>
                      <button
                        type="button"
                        className="pl-link"
                        onClick={() => lab.copyToCustom(c.id)}
                        disabled={disabled}
                        style={{ color: colors.accent, display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
                      >
                        <Copy className="pl-icon" aria-hidden="true" style={{ width: 12, height: 12 }} />
                        Edit a copy
                      </button>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

function Custom({ lab, disabled }: { lab: GeometryLab; disabled?: boolean }) {
  const colors = useLabColors()
  const { draft, setDraft, custom } = lab
  const field = { borderColor: colors.border, background: colors.panel, color: colors.text }
  const badTemplates = [...custom.templates, ...(custom.eval_templates ?? [])].filter((t) => t.split('{x}').length !== 2).length
  const missingGroups = custom.groups ? custom.groups.filter((g) => !g).length : 0

  return (
    <div className="gl-custom">
      <input
        className="pl-input"
        placeholder="Name"
        value={draft.name}
        disabled={disabled}
        onChange={(e) => setDraft({ ...draft, name: e.target.value })}
        style={field}
        aria-label="Concept name"
      />
      <Segmented
        ariaLabel="Structure"
        value={draft.structure}
        onChange={(structure) => setDraft({ ...draft, structure })}
        disabled={disabled}
        options={(['cyclic', 'ordinal', 'categorical'] as const).map((s) => ({ value: s, label: STRUCTURE_LABELS[s], title: STRUCTURE_TITLES[s] }))}
      />
      <div>
        <div className="pl-control-label" title="One per line, in order. Optional group after a bar: Monday | weekday">
          Values
        </div>
        <textarea
          className="pl-textarea"
          rows={6}
          placeholder={'Monday | weekday\nTuesday | weekday\n…\nSunday | weekend'}
          value={draft.values}
          disabled={disabled}
          onChange={(e) => setDraft({ ...draft, values: e.target.value })}
          style={field}
          aria-label="Values"
        />
      </div>
      <div>
        <div className="pl-control-label" title="One per line, each with {x} where the value goes. Put a space before {x}.">
          Templates
        </div>
        <textarea
          className="pl-textarea"
          rows={6}
          placeholder={'Today is {x}.\nThe meeting is on {x}.\n…'}
          value={draft.templates}
          disabled={disabled}
          onChange={(e) => setDraft({ ...draft, templates: e.target.value })}
          style={field}
          aria-label="Templates"
        />
      </div>
      <div>
        <div className="pl-control-label" title="Optional, for steering: one per line, each ending where the model would say a value">
          Eval prompts
        </div>
        <textarea
          className="pl-textarea"
          rows={2}
          placeholder={'Today is {x}. Tomorrow is'}
          value={draft.evalTemplates}
          disabled={disabled}
          onChange={(e) => setDraft({ ...draft, evalTemplates: e.target.value })}
          style={{ ...field, minHeight: 52 }}
          aria-label="Eval prompts"
        />
      </div>
      <div className="pl-hint">
        {custom.values.length} values × {custom.templates.length} templates = {custom.values.length * custom.templates.length} examples
      </div>
      {badTemplates > 0 && (
        <div className="pl-hint" style={{ color: colors.warning }}>
          {badTemplates} template{badTemplates > 1 ? 's need' : ' needs'} exactly one {'{x}'}
        </div>
      )}
      {missingGroups > 0 && (
        <div className="pl-hint" style={{ color: colors.warning }}>
          {missingGroups} value{missingGroups > 1 ? 's have' : ' has'} no group
        </div>
      )}
      {custom.values.length > 0 && <ValueChips concept={{ ...custom, positions: null, groups: custom.groups ?? null }} />}
    </div>
  )
}

export function ConceptPanel({ lab, disabled }: { lab: GeometryLab; disabled?: boolean }) {
  return (
    <div className="gl-concept">
      <Segmented
        ariaLabel="Concept source"
        value={lab.source}
        onChange={lab.setSource}
        disabled={disabled}
        options={[
          { value: 'builtin', label: 'Built-in' },
          { value: 'custom', label: 'Custom' },
        ]}
      />
      {lab.source === 'builtin' ? <Builtin lab={lab} disabled={disabled} /> : <Custom lab={lab} disabled={disabled} />}

      <style>{`
        .geometry-lab .gl-item { border: 1px solid; margin-bottom: 0.35rem; }
        .geometry-lab .gl-pick {
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
        .geometry-lab .gl-pick:disabled { cursor: not-allowed; }
        .geometry-lab .gl-name { font-size: 0.82rem; font-weight: 600; }
        .geometry-lab .gl-detail { padding: 0 0.65rem 0.6rem; display: flex; flex-direction: column; gap: 0.45rem; align-items: flex-start; }
        .geometry-lab .gl-chips { display: flex; flex-wrap: wrap; gap: 0.25rem; }
        .geometry-lab .gl-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          padding: 0.1rem 0.4rem;
          border: 1px solid;
          font-size: 0.7rem;
          white-space: nowrap;
        }
        .geometry-lab .gl-custom { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.75rem; }
      `}</style>
    </div>
  )
}
