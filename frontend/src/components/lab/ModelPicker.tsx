/**
 * Choosing the model a lab reads: Ozera, open-source, and the user's custom models (from the
 * Probe Lab's model list), each with its depth and width.
 */

import type { ProbeModelInfo } from '../../types/probes'
import { Dropdown, type DropdownGroup } from '../common/Dropdown'

export interface ModelChoice {
  models: ProbeModelInfo[]
  modelId: string
  setModelId: (id: string) => void
  loadingModels: boolean
}

const TYPE_LABELS = { ozera: 'Ozera Models', open_source: 'Open Source', custom: 'Custom Models' } as const

export function ModelPicker({ lab, disabled }: { lab: ModelChoice; disabled?: boolean }) {
  const groups: DropdownGroup[] = lab.loadingModels
    ? [{ label: 'Models', options: [{ value: '', label: 'Loading models…' }] }]
    : (['ozera', 'open_source', 'custom'] as const)
        .map((type) => ({
          label: TYPE_LABELS[type],
          options: lab.models
            .filter((m) => m.model_type === type)
            .map((m) => ({
              value: m.model_id,
              label: `${m.display_name} · ${m.num_layers ?? '?'}L · d=${m.hidden_dim ?? '?'}`,
              disabled: m.num_layers == null,
            })),
        }))
        .filter((group) => group.options.length > 0)
  return <Dropdown value={lab.modelId} onChange={lab.setModelId} groups={groups} disabled={disabled || lab.loadingModels} />
}
