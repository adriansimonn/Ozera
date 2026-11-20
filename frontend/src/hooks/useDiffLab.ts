/**
 * State and actions of the Model Diff page: the pair and the texts both models read, the
 * layers run, and what the directions and patch views run.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import type {
  DiffEstimate,
  DiffPair,
  DiffSample,
  DirectionsRun,
  Donor,
  LayerMeasure,
  LayersRun,
  PatchKind,
  PatchRun,
  TextSpec,
} from '../types/diff'
import { diffBridge } from '../components/assistant/pages/diff'
import { useAssistantPage } from './useAssistantPage'

export type DiffViewId = 'layers' | 'directions' | 'patch'
export type TextSource = 'builtin' | 'own'
/** What the layers view's tokens are colored by */
export type TokenColor = 'kl' | 'distance'

export const MIN_TEXTS = 4
export const MAX_TEXTS = 200

const pairKey = (pair: Pick<DiffPair, 'base' | 'other'>) => `${pair.base}:${pair.other}`

/** The texts a textarea holds, one per line. */
export function ownTexts(text: string): string[] {
  return [...new Set(text.split('\n').map((line) => line.trim()).filter(Boolean))]
}

export function useDiffLab(onShowPurchaseCredits?: () => void) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()
  // A model named in the address (?model=), as the Custom Models page links here
  const wanted = useRef(new URLSearchParams(location.search).get('model'))
  // Whether the layer shown has been set (by the first pair, or by a layer asked for before it loaded)
  const positioned = useRef(false)

  // The pair
  const [pairs, setPairs] = useState<DiffPair[]>([])
  const [pairId, setPairId] = useState('')
  const [loadingPairs, setLoadingPairs] = useState(true)

  // The texts
  const [samples, setSamples] = useState<DiffSample[]>([])
  const [source, setSource] = useState<TextSource>('builtin')
  const [sampleId, setSampleId] = useState('prose')
  const [count, setCount] = useState(48)
  const [seed, setSeed] = useState(0)
  const [own, setOwn] = useState('')
  const [chatTemplate, setChatTemplate] = useState(false)

  // The layers run
  const [estimate, setEstimate] = useState<DiffEstimate | null>(null)
  const [running, setRunning] = useState(false)
  const [run, setRun] = useState<LayersRun | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Views
  const [view, setView] = useState<DiffViewId>('layers')
  const [measure, setMeasure] = useState<LayerMeasure>('distance')
  const [position, setPosition] = useState(1)
  const [tokenColor, setTokenColor] = useState<TokenColor>('kl')

  // Directions
  const [directionsEstimate, setDirectionsEstimate] = useState<DiffEstimate | null>(null)
  const [finding, setFinding] = useState(false)
  const [directions, setDirections] = useState<DirectionsRun | null>(null)
  const [component, setComponent] = useState(1)

  // Patch
  const [prompt, setPrompt] = useState('')
  const [donor, setDonor] = useState<Donor>('other')
  const [kind, setKind] = useState<PatchKind>('resid')
  const [patchEstimate, setPatchEstimate] = useState<DiffEstimate | null>(null)
  const [patching, setPatching] = useState(false)
  const [patched, setPatched] = useState<PatchRun | null>(null)

  const pair = pairs.find((p) => pairKey(p) === pairId) ?? null
  const shownPair = useRef(pairId)
  shownPair.current = pairId
  const sample = samples.find((s) => s.id === sampleId) ?? null
  // Only a pair with an instruct model has a chat template
  const usedTemplate = chatTemplate && !!pair?.chat_template

  useEffect(() => {
    let cancelled = false
    setLoadingPairs(true)
    Promise.all([apiClient.getDiffPairs(), apiClient.listDiffSamples()])
      .then(([pairList, sampleList]) => {
        if (cancelled) return
        setPairs(pairList)
        setSamples(sampleList)
        const named = wanted.current ? pairList.find((p) => p.other === wanted.current || p.base === wanted.current) : undefined
        if (named) wanted.current = null
        const chosen = named ?? pairList.find((p) => pairKey(p) === shownPair.current) ?? pairList[0]
        setPairId(chosen ? pairKey(chosen) : '')
        // The first pair shown opens at its middle layer, unless a layer was already asked for
        if (chosen && !positioned.current) {
          positioned.current = true
          setPosition(Math.round(chosen.num_layers / 2))
        }
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load Model Diff'))
      .finally(() => !cancelled && setLoadingPairs(false))
    return () => {
      cancelled = true
    }
  }, [isAuthenticated])

  // Choosing a pair opens it at its middle layer
  const choosePair = useCallback(
    (id: string) => {
      setPairId(id)
      const chosen = pairs.find((p) => pairKey(p) === id)
      if (chosen) setPosition(Math.round(chosen.num_layers / 2))
    },
    [pairs],
  )

  // A model's ID picks the pair it's in (the assistant names pairs this way)
  const setModel = useCallback(
    (model: string) => {
      const named = pairs.find((p) => p.other === model || p.base === model)
      if (named) choosePair(pairKey(named))
      else wanted.current = model
    },
    [pairs, choosePair],
  )

  // The layer shown stays inside the pair's model
  const layers = pair?.num_layers ?? run?.num_layers ?? 0
  useEffect(() => {
    if (layers > 0) setPosition((current) => (current > layers ? Math.round(layers / 2) : current))
  }, [layers])

  const chooseLayer = useCallback((next: number) => {
    positioned.current = true
    setPosition(Math.max(0, Math.round(next)))
  }, [])

  const ownList = useMemo(() => ownTexts(own), [own])
  const textSpec = useCallback((): TextSpec | null => {
    if (source === 'builtin') return sample ? { sample: sample.id, n: Math.min(count, sample.num_texts), seed } : null
    return ownList.length >= MIN_TEXTS && ownList.length <= MAX_TEXTS ? { texts: ownList, name: 'Your texts' } : null
  }, [source, sample, count, seed, ownList])

  const pairOf = () => (pair ? { base: pair.base, other: pair.other, chat_template: usedTemplate } : null)

  // The layers run's estimate, refreshed shortly after the settings settle
  useEffect(() => {
    const texts = textSpec()
    if (!isAuthenticated || !pair || !texts) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateDiff({ kind: 'layers', base: pair.base, other: pair.other, chat_template: usedTemplate, texts })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, pair, usedTemplate, textSpec])

  useEffect(() => {
    const texts = textSpec()
    if (!isAuthenticated || !pair || !texts || view !== 'directions') {
      setDirectionsEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateDiff({ kind: 'directions', base: pair.base, other: pair.other, chat_template: usedTemplate, texts })
        .then(setDirectionsEstimate)
        .catch(() => setDirectionsEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, pair, usedTemplate, textSpec, view])

  useEffect(() => {
    if (!isAuthenticated || !pair || !prompt.trim() || view !== 'patch') {
      setPatchEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateDiff({ kind: 'patch', base: pair.base, other: pair.other, chat_template: usedTemplate, prompt })
        .then(setPatchEstimate)
        .catch(() => setPatchEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, pair, usedTemplate, prompt, view])

  const failed = (err: unknown, fallback: string) => {
    if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
    else setError(err instanceof Error ? err.message : fallback)
  }

  const runLayers = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const texts = textSpec()
    const models = pairOf()
    if (!models || !texts) {
      setError('Choose a pair and texts first')
      return
    }
    setRunning(true)
    setError(null)
    setView('layers')
    try {
      setRun(await apiClient.runDiffLayers({ ...models, texts }))
    } catch (err) {
      failed(err, 'The comparison failed')
    } finally {
      setRunning(false)
    }
  }

  const findDirections = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const texts = textSpec()
    const models = pairOf()
    if (!models || !texts) {
      setError('Choose a pair and texts first')
      return
    }
    setFinding(true)
    setError(null)
    try {
      setDirections(await apiClient.runDiffDirections({ ...models, texts, layer: position, seed }))
      setComponent(1)
    } catch (err) {
      failed(err, 'Finding directions failed')
    } finally {
      setFinding(false)
    }
  }

  const runPatch = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const models = pairOf()
    if (!models || !prompt.trim()) return
    setPatching(true)
    setError(null)
    try {
      setPatched(await apiClient.runDiffPatch({ ...models, prompt, donor, kind }))
    } catch (err) {
      failed(err, 'Patching failed')
    } finally {
      setPatching(false)
    }
  }

  // An assistant run's settings, as if the page's own controls had made them
  const adopt = (result: { base: string; other: string; chat_template: boolean }, texts?: TextSpec) => {
    const named = pairs.find((p) => p.base === result.base && p.other === result.other)
    if (named) setPairId(pairKey(named))
    else wanted.current = result.other
    setChatTemplate(result.chat_template)
    if (texts?.sample) {
      setSource('builtin')
      setSampleId(texts.sample)
      if (texts.n) setCount(texts.n)
      setSeed(texts.seed ?? 0)
    } else if (texts?.texts) {
      setSource('own')
      setOwn(texts.texts.join('\n'))
    }
    setError(null)
  }

  const lab = {
    isAuthenticated,
    // the pair
    pairs,
    pair,
    pairId,
    setPairId: choosePair,
    setModel,
    loadingPairs,
    // the texts
    samples,
    sample,
    source,
    setSource,
    sampleId,
    setSampleId,
    count,
    setCount,
    seed,
    setSeed,
    own,
    setOwn,
    ownCount: ownList.length,
    chatTemplate: usedTemplate,
    setChatTemplate,
    textSpec,
    // the layers run
    estimate,
    running,
    runLayers,
    run,
    loadRun: (result: LayersRun, texts?: TextSpec) => {
      adopt(result, texts)
      setRun(result)
      setView('layers')
    },
    error,
    setError,
    // views
    view,
    setView,
    measure,
    setMeasure,
    position,
    setPosition: chooseLayer,
    layers,
    tokenColor,
    setTokenColor,
    // directions
    directionsEstimate,
    finding,
    findDirections,
    directions,
    showDirections: (result: DirectionsRun, texts?: TextSpec) => {
      adopt(result, texts)
      setDirections(result)
      chooseLayer(result.layer)
      setComponent(1)
      setView('directions')
    },
    component,
    setComponent,
    // patch
    prompt,
    setPrompt,
    donor,
    setDonor,
    kind,
    setKind,
    patchEstimate,
    patching,
    runPatch,
    patched,
    showPatched: (result: PatchRun) => {
      adopt(result)
      setPatched(result)
      setPrompt(result.prompt)
      setDonor(result.donor)
      setKind(result.kind)
      setView('patch')
    },
  }
  // The assistant reads the lab and fills its settings
  useAssistantPage('diff', diffBridge(lab))
  return lab
}

export type DiffLab = ReturnType<typeof useDiffLab>
