/**
 * State and actions of the Attribution page: the model and pair set, the sweep, and what its
 * verify, heads and ablate views run on it. (The continuation view keeps its own.)
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import type {
  AblateResponse,
  AblationMethod,
  AttributionEstimate,
  AttributionModelInfo,
  AttributionRun,
  AttributionTask,
  Cell,
  MetricName,
  PairSpec,
  ParsedPairs,
  PositionKind,
  VerifyResponse,
} from '../types/attribution'
import { ABLATION_METHODS } from '../types/attribution'
import {
  defaultScope,
  headCells,
  headName,
  parseComponents,
  readScope,
  residualCells,
  topComponents,
  topPositions,
} from '../components/attribution/attributionUtils'
import { attributionBridge } from '../components/assistant/pages/attribution'
import { useAssistantPage } from './useAssistantPage'

export type AttributionViewId = 'sweep' | 'verify' | 'heads' | 'ablate' | 'continuation'
export type PairSource = 'builtin' | 'upload'
/** What the verify view patches: the sweep's top components, the top cells of its position map, every head, or one pair's residual stream */
export type VerifyMode = 'top' | 'positions' | 'heads' | 'resid'

export const VERIFY_COUNTS = [10, 20, 40] as const

export interface VerifyResult {
  mode: VerifyMode
  runId: string | null
  /** The pair a residual stream sweep was of */
  pair: number | null
  response: VerifyResponse
}

export interface ExactEffect {
  effect: number
  se: number
}

export function useAttributionLab(onShowPurchaseCredits?: () => void) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Models
  const [models, setModels] = useState<AttributionModelInfo[]>([])
  const [modelId, setModelId] = useState('')
  const [loadingModels, setLoadingModels] = useState(true)

  // Pairs
  const [tasks, setTasks] = useState<AttributionTask[]>([])
  const [source, setSource] = useState<PairSource>('builtin')
  const [taskId, setTaskId] = useState('ioi')
  const [pairCount, setPairCount] = useState(40)
  const [upload, setUpload] = useState<ParsedPairs | null>(null)
  const [parsing, setParsing] = useState(false)
  const [seed, setSeed] = useState(0)
  const [metric, setMetric] = useState<MetricName>('logit_diff')

  // The sweep
  const [estimate, setEstimate] = useState<AttributionEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [run, setRun] = useState<AttributionRun | null>(null)
  // The pairs the sweep on screen ran on: what its verify and ablate runs read
  const [runPairs, setRunPairs] = useState<PairSpec | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Views
  const [view, setView] = useState<AttributionViewId>('sweep')
  const [sites, setSites] = useState<PositionKind>('resid')
  const [scope, setScope] = useState('pair:0')
  const [selectedHead, setSelectedHead] = useState<{ layer: number; head: number } | null>(null)

  // Verify
  const [verifyMode, setVerifyMode] = useState<VerifyMode>('top')
  const [verifyCount, setVerifyCount] = useState<number>(20)
  const [verifying, setVerifying] = useState(false)
  const [verifyEstimate, setVerifyEstimate] = useState<AttributionEstimate | null>(null)
  const [verified, setVerified] = useState<VerifyResult | null>(null)
  // Every head's exact effect measured so far on this sweep, by "layer.head"
  const [exactHeads, setExactHeads] = useState<Record<string, ExactEffect>>({})

  // Ablate
  const [componentsText, setComponentsText] = useState('')
  const [methods, setMethods] = useState<AblationMethod[]>(ABLATION_METHODS)
  const [ablating, setAblating] = useState(false)
  const [ablateEstimate, setAblateEstimate] = useState<AttributionEstimate | null>(null)
  const [ablated, setAblated] = useState<{ runId: string | null; response: AblateResponse } | null>(null)

  const model = models.find((m) => m.model_id === modelId) ?? null
  const task = tasks.find((t) => t.id === taskId) ?? null

  useEffect(() => {
    let cancelled = false
    setLoadingModels(true)
    Promise.all([apiClient.getAttributionModels(), apiClient.listAttributionTasks()])
      .then(([modelList, taskList]) => {
        if (cancelled) return
        setModels(modelList)
        setTasks(taskList)
        if (modelList.length > 0) setModelId((current) => current || modelList[0].model_id)
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load Attribution'))
      .finally(() => !cancelled && setLoadingModels(false))
    return () => {
      cancelled = true
    }
  }, [isAuthenticated])

  const hasAnswers = source === 'builtin' ? true : (upload?.has_answers ?? true)
  // Pairs without answers can only be measured by KL
  const usedMetric: MetricName = hasAnswers ? metric : 'kl'

  const pairSpec = useCallback((): PairSpec | null => {
    if (source === 'builtin') return task ? { task: task.id, n: pairCount, seed } : null
    return upload ? { rows: upload.rows, name: upload.filename } : null
  }, [source, task, pairCount, seed, upload])

  const parseFile = useCallback(
    async (file: File) => {
      if (!isAuthenticated) {
        navigate('/auth')
        return
      }
      setParsing(true)
      setError(null)
      try {
        setUpload(await apiClient.parseAttributionPairs(file))
        setSource('upload')
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to read the file')
      } finally {
        setParsing(false)
      }
    },
    [isAuthenticated, navigate],
  )

  // The sweep's estimate, refreshed shortly after the settings settle
  useEffect(() => {
    const pairs = pairSpec()
    if (!isAuthenticated || !modelId || !pairs) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateAttribution({ kind: 'sweep', model: modelId, pairs })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, modelId, pairSpec])

  const showRun = (result: AttributionRun, pairs: PairSpec) => {
    setRun(result)
    setRunPairs(pairs)
    setScope(defaultScope(result))
    setSelectedHead(null)
    setVerified(null)
    setExactHeads({})
    setAblated(null)
  }

  const runSweep = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const pairs = pairSpec()
    if (!modelId || !pairs) {
      setError('Choose a model and pairs first')
      return
    }
    setRunning(true)
    setError(null)
    setView('sweep')
    try {
      showRun(await apiClient.runAttribution({ model: modelId, pairs, metric: usedMetric }), pairs)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'The sweep failed')
    } finally {
      setRunning(false)
    }
  }

  // An assistant run: its settings and result, as if the page's button had run it
  const loadRun = (result: AttributionRun, pairs: PairSpec) => {
    setModelId(result.model)
    if (pairs.task) {
      setSource('builtin')
      setTaskId(pairs.task)
      setPairCount(pairs.n ?? result.num_pairs)
      setSeed(pairs.seed ?? 0)
    } else if (pairs.rows) {
      setUpload({ filename: pairs.name ?? 'Your pairs', rows: pairs.rows, has_answers: pairs.rows.every((r) => r.correct != null), warnings: [] })
      setSource('upload')
    }
    setMetric(result.metric)
    setError(null)
    setView('sweep')
    showRun(result, pairs)
  }

  // Verify: the cells the chosen mode patches
  const scopeInfo = useMemo(() => (run ? readScope(run, scope) : null), [run, scope])
  const residPair = scopeInfo ? scopeInfo.example : 0
  const verifyCells: Cell[] = useMemo(() => {
    if (!run) return []
    if (verifyMode === 'top') return topComponents(run, verifyCount)
    if (verifyMode === 'heads') return headCells(run)
    if (verifyMode === 'resid') return residualCells(run, residPair)
    return scopeInfo ? topPositions(run, scopeInfo, sites, verifyCount) : []
  }, [run, verifyMode, verifyCount, scopeInfo, sites, residPair])

  useEffect(() => {
    if (!isAuthenticated || !run || !runPairs || verifyCells.length === 0 || view !== 'verify') {
      setVerifyEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateAttribution({ kind: 'verify', model: run.model, pairs: runPairs, cells: verifyCells })
        .then(setVerifyEstimate)
        .catch(() => setVerifyEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, run, runPairs, verifyCells, view])

  const noteHeads = (response: VerifyResponse) => {
    const found: Record<string, ExactEffect> = {}
    for (const cell of response.cells) {
      // A head patched at every position, on every pair
      if (cell.kind === 'head' && cell.head != null && cell.position == null && cell.pair == null && cell.cls == null) {
        found[headName(cell.layer, cell.head)] = { effect: cell.effect, se: cell.se }
      }
    }
    if (Object.keys(found).length > 0) setExactHeads((current) => ({ ...current, ...found }))
  }

  const verify = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    if (!run || !runPairs || verifyCells.length === 0) return
    setVerifying(true)
    setError(null)
    try {
      const response = await apiClient.verifyAttribution({
        model: run.model,
        pairs: runPairs,
        metric: run.metric,
        cells: verifyCells,
        pairs_key: run.pairs_key,
      })
      setVerified({ mode: verifyMode, runId: run.run_id, pair: verifyMode === 'resid' ? residPair : null, response })
      noteHeads(response)
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Patching failed')
    } finally {
      setVerifying(false)
    }
  }

  // Ablate
  const layers = run?.num_layers ?? model?.num_layers ?? 0
  const heads = run?.num_heads ?? model?.num_heads ?? 0
  const parsedComponents = useMemo(() => parseComponents(componentsText, layers, heads), [componentsText, layers, heads])

  useEffect(() => {
    if (!isAuthenticated || !run || !runPairs || view !== 'ablate' || methods.length === 0) {
      setAblateEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateAttribution({ kind: 'ablate', model: run.model, pairs: runPairs, methods })
        .then(setAblateEstimate)
        .catch(() => setAblateEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, run, runPairs, methods, view])

  const ablate = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    if (!run || !runPairs || parsedComponents.components.length === 0 || methods.length === 0) return
    setAblating(true)
    setError(null)
    try {
      const response = await apiClient.ablateComponents({
        model: run.model,
        pairs: runPairs,
        metric: run.metric,
        components: parsedComponents.components,
        methods,
        seed,
        pairs_key: run.pairs_key,
      })
      setAblated({ runId: run.run_id, response })
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'The ablation failed')
    } finally {
      setAblating(false)
    }
  }

  const lab = {
    isAuthenticated,
    // models
    models,
    model,
    modelId,
    setModelId,
    loadingModels,
    // pairs
    tasks,
    task,
    source,
    setSource,
    taskId,
    setTaskId,
    pairCount,
    setPairCount,
    upload,
    setUpload,
    parsing,
    parseFile,
    seed,
    setSeed,
    metric: usedMetric,
    setMetric,
    hasAnswers,
    pairSpec,
    // the sweep
    estimate,
    running,
    runSweep,
    run,
    runPairs,
    loadRun,
    error,
    setError,
    // views
    view,
    setView,
    sites,
    setSites,
    scope,
    setScope,
    scopeInfo,
    selectedHead,
    setSelectedHead,
    // verify
    verifyMode,
    setVerifyMode,
    verifyCount,
    setVerifyCount,
    verifyCells,
    residPair,
    verifyEstimate,
    verifying,
    verify,
    verified,
    showVerified: (result: VerifyResult) => {
      setVerified(result)
      if (!result.runId || result.runId === run?.run_id) noteHeads(result.response)
    },
    exactHeads,
    // ablate
    componentsText,
    setComponentsText,
    parsedComponents,
    methods,
    setMethods,
    ablateEstimate,
    ablating,
    ablate,
    ablated,
    showAblated: setAblated,
  }
  // The assistant reads the lab and fills its settings
  useAssistantPage('attribution', attributionBridge(lab))
  return lab
}

export type AttributionLab = ReturnType<typeof useAttributionLab>
