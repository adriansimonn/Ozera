/**
 * Helpers for the pages' assistant bridges: values from set_inputs (checked by the backend
 * against core/assistant/pages.py), and compact snapshot values.
 */

import type { SaeRef } from '../../../types/probes'
import type { FieldValue } from '../../../types/assistant'

export type Fields = Record<string, FieldValue>

/** Ozera's pages (backend/core/assistant/pages.py). */
export const PAGE_ROUTES: Record<string, { route: string; title: string }> = {
  generate: { route: '/', title: 'Generate' },
  training: { route: '/training', title: 'Custom Models' },
  patching: { route: '/patching', title: 'Activation Patching' },
  analysis: { route: '/analysis', title: 'Attention Analysis' },
  sae: { route: '/sae', title: 'SAE Analysis' },
  probes: { route: '/probes', title: 'Probe Lab' },
  geometry: { route: '/geometry', title: 'Geometry Lab' },
  settings: { route: '/settings', title: 'Settings' },
}

/** The control IDs a bridge accepts (they match backend/core/assistant/pages.py, which a test checks). */
export interface ControlSpec {
  id: string
}

export function cut(text: string | null | undefined, chars = 600): string | null {
  if (text == null) return null
  return text.length > chars ? `${text.slice(0, chars)}…` : text
}

export function round(value: number | null | undefined, digits = 3): number | null {
  if (value == null || !Number.isFinite(value)) return null
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

export function asNumber(value: FieldValue | undefined): number | null {
  if (value === undefined || typeof value === 'boolean' || Array.isArray(value)) return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

export function asBool(value: FieldValue | undefined): boolean | null {
  if (value === undefined || Array.isArray(value)) return null
  if (typeof value === 'boolean') return value
  return String(value).toLowerCase() === 'true'
}

export function asString(value: FieldValue | undefined): string | null {
  return value === undefined || Array.isArray(value) ? null : String(value)
}

/** An SAE key (ozera:<model>:<layer> or external:<id>) as a reference. */
export function parseSaeKey(key: string): SaeRef | null {
  const ozera = /^ozera:(nano|mini):(\d+)$/.exec(key)
  if (ozera) return { kind: 'ozera', model: ozera[1] as 'nano' | 'mini', layer: Number(ozera[2]) }
  const external = /^external:([A-Za-z0-9][A-Za-z0-9._-]*)$/.exec(key)
  return external ? { kind: 'external', sae_id: external[1] } : null
}

export function saeKeyOf(ref: SaeRef | null | undefined): string | null {
  if (!ref) return null
  return ref.kind === 'ozera' ? `ozera:${ref.model}:${ref.layer}` : `external:${ref.sae_id}`
}
