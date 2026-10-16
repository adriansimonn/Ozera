/**
 * State and actions of the Geometry Lab page, shared by its glass and default layouts.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import type { ProbeModelInfo } from '../types/probes'
import type {
  ConceptDetail,
  ConceptSpec,
  ConceptStructure,
  ConceptSummary,
  CustomConcept,
  GeometryEstimate,
  GeometryRead,
  GeometryRun,
  RunConcept,
} from '../types/geometry'
import { READ_LABELS, bestPosition, defaultDims } from '../components/geometry/geometryUtils'
import type { GeometryViewId } from '../components/geometry/GeometryViews'

export type ConceptSource = 'builtin' | 'custom'

// Runs kept for comparing (each holds every layer's coordinates)
const MAX_RUNS = 6

export interface StoredRun {
  id: number
  run: GeometryRun
  label: string
}

export interface CustomDraft {
  name: string
  structure: ConceptStructure
  /** One value per line, optionally "value | group" */
  values: string
  /** One template per line, each with {x} */
  templates: string
  /** One per line, each ending where the model would say a value (for steering) */
  evalTemplates: string
}

const EMPTY_DRAFT: CustomDraft = { name: '', structure: 'cyclic', values: '', templates: '', evalTemplates: '' }

function lines(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
}

/** A draft as the API's custom concept (groups only when every value has one). */
export function parseDraft(draft: CustomDraft): CustomConcept {
  const parts = lines(draft.values).map((line) => line.split('|').map((part) => part.trim()))
  const grouped = parts.some((p) => p.length > 1 && p[1])
  return {
    name: draft.name.trim() || 'Custom concept',
    structure: draft.structure,
    values: parts.map((p) => p[0]),
    templates: lines(draft.templates),
    eval_templates: lines(draft.evalTemplates),
    groups: grouped ? parts.map((p) => p[1] ?? '') : null,
  }
}

/** The concept a run read, as a request's concept (its built-in ID, or the whole custom concept). */
export function runConceptSpec(c: RunConcept): ConceptSpec {
  if (c.builtin_id) return { builtin_id: c.builtin_id }
  return {
    custom: {
      name: c.name,
      structure: c.structure,
      values: c.values,
      templates: c.templates,
      eval_templates: c.eval_templates,
      groups: c.groups,
      positions: c.positions,
      helix_periods: c.helix_periods,
    },
  }
}

function draftFrom(concept: ConceptDetail): CustomDraft {
  return {
    name: `${concept.name} (edited)`,
    structure: concept.structure,
    values: concept.values.map((v, i) => (concept.groups ? `${v} | ${concept.groups[i]}` : v)).join('\n'),
    templates: concept.templates.join('\n'),
    evalTemplates: concept.eval_templates.join('\n'),
  }
}

export function useGeometryLab(onShowPurchaseCredits?: () => void) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Models (the Probe Lab's list: the same models can be read)
  const [models, setModels] = useState<ProbeModelInfo[]>([])
  const [modelId, setModelIdState] = useState('')
  const [loadingModels, setLoadingModels] = useState(true)

  // Concepts
  const [concepts, setConcepts] = useState<ConceptSummary[]>([])
  const [source, setSource] = useState<ConceptSource>('builtin')
  const [builtinId, setBuiltinId] = useState('months')
  const [draft, setDraft] = useState<CustomDraft>(EMPTY_DRAFT)
  const [copiedFrom, setCopiedFrom] = useState<ConceptDetail | null>(null)

  // Run settings
  const [read, setRead] = useState<GeometryRead>('value_last')
  const [chatTemplate, setChatTemplate] = useState(false)
  const [seed, setSeed] = useState(0)

  // Runs
  const [estimate, setEstimate] = useState<GeometryEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [runs, setRuns] = useState<StoredRun[]>([])
  const [runId, setRunId] = useState<number | null>(null)
  const [nextId, setNextId] = useState(1)
  const [error, setError] = useState<string | null>(null)

  // What the views show
  const [view, setView] = useState<GeometryViewId>('shape')
  const [position, setPosition] = useState(1)
  const [dims, setDims] = useState(3)
  const [pcPair, setPcPair] = useState<[number, number]>([0, 1])

  const model = models.find((m) => m.model_id === modelId) ?? null
  const current = runs.find((r) => r.id === runId) ?? null
  const run = current?.run ?? null

  const modelName = useCallback((id: string) => models.find((m) => m.model_id === id)?.display_name ?? id, [models])

  const setModelId = useCallback((id: string) => {
    setModelIdState(id)
    setChatTemplate(false)
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoadingModels(true)
    Promise.all([apiClient.getProbeModels(), apiClient.listGeometryConcepts()])
      .then(([modelList, conceptList]) => {
        if (cancelled) return
        setModels(modelList)
        setConcepts(conceptList)
        if (modelList.length > 0) setModelIdState((current) => current || modelList[0].model_id)
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the Geometry Lab'))
      .finally(() => !cancelled && setLoadingModels(false))
    return () => {
      cancelled = true
    }
  }, [isAuthenticated])

  const builtin = concepts.find((c) => c.id === builtinId) ?? null
  const custom = useMemo(() => parseDraft(draft), [draft])

  const conceptSpec = useMemo((): ConceptSpec | null => {
    if (source === 'builtin') return builtin ? { builtin_id: builtin.id } : null
    if (custom.values.length === 0 || custom.templates.length === 0) return null
    // A copied built-in keeps its positions while its values are unchanged
    const keep =
      copiedFrom && copiedFrom.positions && copiedFrom.values.join('\n') === custom.values.join('\n') && copiedFrom.structure === custom.structure
    return {
      custom: keep
        ? { ...custom, positions: copiedFrom.positions, helix_periods: copiedFrom.helix_periods }
        : custom,
    }
  }, [source, builtin, custom, copiedFrom])

  // Cost estimate, refreshed shortly after the settings settle
  const specKey = JSON.stringify(conceptSpec)
  useEffect(() => {
    if (!isAuthenticated || !modelId || !conceptSpec) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateGeometryRun({ model: modelId, concept: conceptSpec, chat_template: chatTemplate })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
    // specKey stands for conceptSpec
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, modelId, specKey, chatTemplate])

  const copyToCustom = async (id: string) => {
    try {
      const detail = await apiClient.getGeometryConcept(id)
      setDraft(draftFrom(detail))
      setCopiedFrom(detail)
      setSource('custom')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load the concept')
    }
  }

  const selectRun = useCallback((id: number) => {
    setRunId(id)
  }, [])

  const execute = async (model: string, concept: ConceptSpec, runRead: GeometryRead, runChatTemplate: boolean, runSeed: number) => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    setRunning(true)
    setError(null)
    try {
      const result = await apiClient.runGeometry({ model, concept, read: runRead, chat_template: runChatTemplate, seed: runSeed })
      const id = nextId
      setNextId(id + 1)
      const label = `${result.concept.name} · ${modelName(result.model)} · ${READ_LABELS[runRead].toLowerCase()}`
      setRuns((current) => [{ id, run: result, label }, ...current].slice(0, MAX_RUNS))
      setRunId(id)
      const tested = result.shapes?.dims ?? []
      const m = tested.includes(dims) ? dims : defaultDims(tested)
      setDims(m)
      setPosition(bestPosition(result, Math.max(tested.indexOf(m), 0)))
      setPcPair([0, 1])
      return id
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Geometry run failed')
    } finally {
      setRunning(false)
    }
  }

  const start = async () => {
    if (!modelId || !conceptSpec) {
      setError('Choose a model and a concept first')
      return
    }
    await execute(modelId, conceptSpec, read, chatTemplate, seed)
    if (view === 'compare') setView('shape')
  }

  /**
   * A concept made elsewhere (an atlas community) as the custom concept, run on a model and
   * shown in the Shape view at a residual stream position (the atlas's). Its positions stay
   * while its values and structure are unedited.
   */
  const openConcept = async (concept: CustomConcept, model: string, at?: number) => {
    const detail: ConceptDetail = {
      id: 'atlas',
      name: concept.name,
      description: '',
      structure: concept.structure,
      values: concept.values,
      groups: concept.groups ?? null,
      positions: concept.positions ?? null,
      num_templates: concept.templates.length,
      total_chars: 0,
      hypotheses: [],
      source: 'Atlas',
      templates: concept.templates,
      eval_templates: concept.eval_templates ?? [],
      helix_periods: concept.helix_periods ?? [],
    }
    setDraft({ ...draftFrom(detail), name: concept.name })
    setCopiedFrom(detail)
    setSource('custom')
    setModelIdState(model)
    setChatTemplate(false)
    const spec: ConceptSpec = {
      custom: {
        name: concept.name,
        structure: concept.structure,
        values: concept.values,
        templates: concept.templates,
        eval_templates: concept.eval_templates ?? [],
        groups: concept.groups ?? null,
        positions: concept.positions ?? null,
        helix_periods: concept.helix_periods ?? [],
      },
    }
    const id = await execute(model, spec, read, false, seed)
    if (id == null) return
    if (at != null) setPosition(at)
    setView('shape')
  }

  /** The current run again on another model (e.g. its base/instruct sibling), to compare them. */
  const rerunOn = async (model: string) => {
    if (!run) return
    const concept = runConceptSpec(run.concept)
    const instruct = models.find((m) => m.model_id === model)?.is_instruct ?? false
    setModelIdState(model)
    return execute(model, concept, run.read, run.chat_template && instruct, run.seed)
  }

  // Keep the selected position inside the current run
  useEffect(() => {
    if (run && position > run.num_layers) setPosition(run.num_layers)
  }, [run, position])

  return {
    isAuthenticated,
    showPurchaseCredits: onShowPurchaseCredits,
    // models
    models,
    model,
    modelId,
    setModelId,
    modelName,
    loadingModels,
    // concepts
    concepts,
    builtin,
    source,
    setSource,
    builtinId,
    setBuiltinId,
    draft,
    setDraft,
    custom,
    copyToCustom,
    conceptSpec,
    // settings
    read,
    setRead,
    chatTemplate,
    setChatTemplate,
    seed,
    setSeed,
    // runs
    estimate,
    running,
    start,
    rerunOn,
    openConcept,
    runs,
    run,
    runId,
    selectRun,
    error,
    setError,
    // views
    view,
    setView,
    position,
    setPosition,
    dims,
    setDims,
    pcPair,
    setPcPair,
  }
}

export type GeometryLab = ReturnType<typeof useGeometryLab>
