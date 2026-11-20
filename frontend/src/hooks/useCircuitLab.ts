/**
 * State and actions of the Circuit Tracing page: the model and its transcoders, the prompt and
 * target, the user's graph jobs (polled while one builds), the graph on screen with its
 * selected and pinned nodes, and the intervention run on it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import type {
  CircuitGraph,
  CircuitJob,
  CircuitModelInfo,
  InterveneEstimate,
  InterveneResponse,
  TraceEstimate,
  TraceRequest,
} from '../types/circuits'
import { MAX_INTERVENE_NODES, rankedFeatures, supportingFeatures } from '../components/circuits/circuitUtils'
import { circuitsBridge } from '../components/assistant/pages/circuits'
import { useAssistantPage } from './useAssistantPage'

export type CircuitViewId = 'graph' | 'intervene'

const POLL_MS = 2500
const isActive = (job: CircuitJob) => job.status === 'queued' || job.status === 'running'

export function useCircuitLab(onShowPurchaseCredits?: () => void) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Models and their transcoders
  const [models, setModels] = useState<CircuitModelInfo[]>([])
  const [modelId, setModelId] = useState('')
  const [setId, setSetId] = useState('')
  const [loadingModels, setLoadingModels] = useState(true)

  // What to trace
  const [prompt, setPrompt] = useState('The capital of France is')
  const [target, setTarget] = useState('')
  const [estimate, setEstimate] = useState<TraceEstimate | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // Graph jobs, and the graph on screen
  const [jobs, setJobs] = useState<CircuitJob[]>([])
  const [graph, setGraph] = useState<CircuitGraph | null>(null)
  const [loadingGraph, setLoadingGraph] = useState<string | null>(null)
  // The job whose graph opens when it's built (the one just started here)
  const awaited = useRef<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Views
  const [view, setView] = useState<CircuitViewId>('graph')
  const [selected, setSelected] = useState<string | null>(null)
  const [pinned, setPinned] = useState<string[]>([])
  const [shown, setShown] = useState<number>(80)
  const [showErrors, setShowErrors] = useState(true)

  // Intervention
  const [scale, setScale] = useState(0)
  const [interveneEstimate, setInterveneEstimate] = useState<InterveneEstimate | null>(null)
  const [intervening, setIntervening] = useState(false)
  const [intervention, setIntervention] = useState<InterveneResponse | null>(null)

  const model = models.find((m) => m.model_id === modelId) ?? null
  const transcoders = model?.transcoders.find((t) => t.id === setId) ?? model?.transcoders[0] ?? null

  useEffect(() => {
    let cancelled = false
    setLoadingModels(true)
    apiClient
      .getCircuitModels()
      .then((list) => {
        if (cancelled) return
        setModels(list)
        if (list.length > 0) setModelId((current) => current || list[0].model_id)
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load Circuit Tracing'))
      .finally(() => !cancelled && setLoadingModels(false))
    return () => {
      cancelled = true
    }
  }, [])

  // A model's own sets: keep the chosen one only if the model has it
  useEffect(() => {
    if (model && !model.transcoders.some((t) => t.id === setId)) setSetId(model.transcoders[0]?.id ?? '')
  }, [model, setId])

  const request = useCallback((): TraceRequest | null => {
    if (!modelId || !prompt.trim()) return null
    return { model: modelId, transcoders: transcoders?.id ?? null, prompt, target: target || null }
  }, [modelId, transcoders, prompt, target])

  // The graph's estimate, refreshed shortly after the settings settle
  useEffect(() => {
    const body = request()
    if (!isAuthenticated || !body) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient.estimateCircuit(body).then(setEstimate).catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, request])

  const showGraph = useCallback((loaded: CircuitGraph) => {
    setGraph(loaded)
    setSelected(null)
    setPinned([])
    setIntervention(null)
  }, [])

  const openJob = useCallback(
    async (jobId: string): Promise<CircuitGraph | null> => {
      setLoadingGraph(jobId)
      setError(null)
      try {
        const loaded = await apiClient.getCircuitGraph(jobId)
        showGraph(loaded)
        return loaded
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load the graph')
        return null
      } finally {
        setLoadingGraph(null)
      }
    },
    [showGraph],
  )

  // The user's graphs
  useEffect(() => {
    if (!isAuthenticated) {
      setJobs([])
      return
    }
    let cancelled = false
    apiClient
      .listCircuitJobs()
      .then((list) => !cancelled && setJobs(list))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [isAuthenticated])

  // Jobs being built are polled; the one started here opens when it's done
  const activeIds = jobs.filter(isActive).map((j) => j.job_id).join(',')
  useEffect(() => {
    if (!activeIds) return
    let cancelled = false
    const poll = async () => {
      for (const jobId of activeIds.split(',')) {
        try {
          const updated = await apiClient.getCircuitJob(jobId)
          if (cancelled) return
          setJobs((list) => list.map((j) => (j.job_id === updated.job_id ? updated : j)))
          if (awaited.current === updated.job_id && !isActive(updated)) {
            awaited.current = null
            if (updated.status === 'completed') {
              setView('graph')
              openJob(updated.job_id)
            } else {
              setError(updated.error ?? 'The graph couldn’t be built')
              useAuthStore.getState().refreshUser().catch(() => {})
            }
          }
        } catch {
          // A failed poll only delays the next one
        }
      }
    }
    const timer = setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [activeIds, openJob])

  const building = jobs.find((j) => isActive(j) && j.job_id === awaited.current) ?? jobs.find(isActive) ?? null

  // A job started elsewhere (the assistant's): listed, and opened once it's built
  const watchJob = (job: CircuitJob) => {
    setJobs((list) => [job, ...list.filter((j) => j.job_id !== job.job_id)])
    if (job.status === 'completed') openJob(job.job_id)
    else if (isActive(job)) awaited.current = job.job_id
  }

  const trace = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const body = request()
    if (!body) {
      setError('Choose a model and write a prompt first')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      watchJob(await apiClient.submitCircuitJob(body))
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Failed to start the graph')
    } finally {
      setSubmitting(false)
    }
  }

  // Pinned nodes: the feature nodes an intervention changes
  const features = useMemo(() => (graph ? rankedFeatures(graph) : []), [graph])
  const togglePin = (nodeId: string) =>
    setPinned((list) => (list.includes(nodeId) ? list.filter((id) => id !== nodeId) : list.length < MAX_INTERVENE_NODES ? [...list, nodeId] : list))
  // The nodes that support the target most: ablating them should lower it
  const pinTop = (count: number) => graph && setPinned(supportingFeatures(graph).slice(0, count).map((n) => n.id))

  useEffect(() => {
    if (!isAuthenticated || !graph || pinned.length === 0 || view !== 'intervene') {
      setInterveneEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateIntervention({ job_id: graph.job_id, nodes: pinned, scale })
        .then(setInterveneEstimate)
        .catch(() => setInterveneEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, graph, pinned, scale, view])

  const intervene = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    if (!graph || pinned.length === 0) return
    setIntervening(true)
    setError(null)
    try {
      setIntervention(await apiClient.runIntervention({ job_id: graph.job_id, nodes: pinned, scale }))
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'The intervention failed')
    } finally {
      setIntervening(false)
    }
  }

  // An intervention run elsewhere (the assistant's): its graph, nodes and result, as if run here
  const showIntervention = async (response: InterveneResponse) => {
    if (graph?.job_id !== response.job_id && !(await openJob(response.job_id))) return
    setPinned(response.nodes.map((n) => n.id))
    setScale(response.scale)
    setIntervention(response)
    setView('intervene')
  }

  const lab = {
    isAuthenticated,
    // models
    models,
    model,
    modelId,
    setModelId,
    loadingModels,
    transcoders,
    setId: transcoders?.id ?? '',
    setSetId,
    // what to trace
    prompt,
    setPrompt,
    target,
    setTarget,
    estimate,
    submitting,
    trace,
    // jobs and the graph
    jobs,
    building,
    watchJob,
    graph,
    loadingGraph,
    openJob,
    error,
    setError,
    // views
    view,
    setView,
    selected,
    setSelected,
    pinned,
    setPinned,
    togglePin,
    pinTop,
    features,
    shown,
    setShown,
    showErrors,
    setShowErrors,
    // intervention
    scale,
    setScale,
    interveneEstimate,
    intervening,
    intervene,
    intervention,
    showIntervention,
  }
  // The assistant reads the lab and fills its settings
  useAssistantPage('circuits', circuitsBridge(lab))
  return lab
}

export type CircuitLab = ReturnType<typeof useCircuitLab>
