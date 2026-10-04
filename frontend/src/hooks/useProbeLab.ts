/**
 * State and actions of the Probe Lab page, shared by its glass and default layouts.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import { useAuthStore } from '../stores/authStore'
import type {
  ParsedProbeDataset,
  Pooling,
  ProbeDatasetDetail,
  ProbeDatasetSpec,
  ProbeDatasetSummary,
  ProbeEstimate,
  ProbeMethod,
  ProbeModelInfo,
  ProbeRunResult,
  ReadSpan,
  SavedProbe,
} from '../types/probes'
import { bestPosition, type EvalSet, type SweepMetric } from '../components/probes/probeUtils'
import type { HeatmapSource } from '../components/probes/TokenHeatmap'
import type { ProbeViewId } from '../components/probes/ProbeViews'

export type DatasetSource = 'builtin' | 'upload'

export interface DatasetStats {
  name: string
  numRows: number
  numPositive: number
  chars: number
  labelNames: [string, string]
  ood: { rows: number; chars: number; description: string } | null
}

function swapLabels(parsed: ParsedProbeDataset): ParsedProbeDataset {
  return {
    ...parsed,
    rows: parsed.rows.map((row) => ({ ...row, label: row.label === 1 ? 0 : 1 })),
    label_names: [parsed.label_names[1], parsed.label_names[0]],
  }
}

function charCount(rows: { text: string }[]): number {
  return rows.reduce((sum, row) => sum + row.text.length, 0)
}

export function useProbeLab(onShowPurchaseCredits?: () => void) {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  // Models
  const [models, setModels] = useState<ProbeModelInfo[]>([])
  const [modelId, setModelIdState] = useState('')
  const [loadingModels, setLoadingModels] = useState(true)

  // Datasets
  const [datasets, setDatasets] = useState<ProbeDatasetSummary[]>([])
  const [datasetDetails, setDatasetDetails] = useState<Record<string, ProbeDatasetDetail>>({})
  const [source, setSource] = useState<DatasetSource>('builtin')
  const [builtinId, setBuiltinId] = useState('truth')
  const [useBuiltinOod, setUseBuiltinOod] = useState(true)
  const [upload, setUpload] = useState<ParsedProbeDataset | null>(null)
  const [oodUpload, setOodUpload] = useState<ParsedProbeDataset | null>(null)
  const [parsing, setParsing] = useState<'train' | 'ood' | null>(null)

  // Run settings
  const [chatTemplate, setChatTemplate] = useState(false)
  const [readSpan, setReadSpan] = useState<ReadSpan>('text')
  const [testFraction, setTestFraction] = useState(0.2)
  const [seed, setSeed] = useState(0)

  // Run
  const [estimate, setEstimate] = useState<ProbeEstimate | null>(null)
  const [training, setTraining] = useState(false)
  const [run, setRun] = useState<ProbeRunResult | null>(null)
  const [runTestFraction, setRunTestFraction] = useState(0.2)
  const [error, setError] = useState<string | null>(null)

  // Which tool is on screen, and what the sweep views show
  const [view, setView] = useState<ProbeViewId>('sweep')
  const [pooling, setPooling] = useState<Pooling>('mean')
  const [method, setMethod] = useState<ProbeMethod>('logreg')
  const [metric, setMetric] = useState<SweepMetric>('auroc')
  const [evalSet, setEvalSet] = useState<EvalSet>('test')
  const [position, setPosition] = useState(1)

  // Saved probes and the token heatmap's probe
  const [savedProbes, setSavedProbes] = useState<SavedProbe[]>([])
  const [loadingSaved, setLoadingSaved] = useState(false)
  const [heatmapSource, setHeatmapSource] = useState<HeatmapSource>('sweep')

  // The probe steering and ablation test ("sweep" or "saved:<id>"), and for a sweep probe,
  // which method's direction (difference in means by default: it's the more causal direction
  // in most reports, e.g. Marks & Tegmark's mass-mean probes)
  const [causalSource, setCausalSource] = useState<HeatmapSource>('sweep')
  const [causalMethod, setCausalMethod] = useState<ProbeMethod>('diff_means')

  const model = models.find((m) => m.model_id === modelId) ?? null

  const setModelId = useCallback(
    (id: string) => {
      setModelIdState(id)
      // Instruct models see user input inside their chat template, so read texts that way by default
      setChatTemplate(models.find((m) => m.model_id === id)?.is_instruct ?? false)
    },
    [models],
  )

  useEffect(() => {
    let cancelled = false
    setLoadingModels(true)
    Promise.all([apiClient.getProbeModels(), apiClient.listProbeDatasets()])
      .then(([modelList, datasetList]) => {
        if (cancelled) return
        setModels(modelList)
        setDatasets(datasetList)
        if (modelList.length > 0) setModelIdState((current) => current || modelList[0].model_id)
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : 'Failed to load the Probe Lab'))
      .finally(() => !cancelled && setLoadingModels(false))
    return () => {
      cancelled = true
    }
  }, [isAuthenticated])

  const refreshSaved = useCallback(async () => {
    if (!isAuthenticated) {
      setSavedProbes([])
      return
    }
    setLoadingSaved(true)
    try {
      setSavedProbes(await apiClient.listProbes())
    } catch (err) {
      console.error('Failed to load saved probes:', err)
    } finally {
      setLoadingSaved(false)
    }
  }, [isAuthenticated])

  useEffect(() => {
    refreshSaved()
  }, [refreshSaved])

  const loadDatasetDetail = useCallback(
    async (id: string) => {
      if (datasetDetails[id]) return datasetDetails[id]
      const detail = await apiClient.getProbeDataset(id)
      setDatasetDetails((current) => ({ ...current, [id]: detail }))
      return detail
    },
    [datasetDetails],
  )

  const parseFile = useCallback(
    async (file: File, kind: 'train' | 'ood') => {
      if (!isAuthenticated) {
        navigate('/auth')
        return
      }
      setParsing(kind)
      setError(null)
      try {
        const parsed = await apiClient.parseProbeDataset(file)
        if (kind === 'train') {
          setUpload(parsed)
          setSource('upload')
        } else {
          setOodUpload(parsed)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to read the file')
      } finally {
        setParsing(null)
      }
    },
    [isAuthenticated, navigate],
  )

  const builtin = datasets.find((d) => d.id === builtinId) ?? null

  const stats: DatasetStats | null = useMemo(() => {
    const ood = (description: string | undefined, rows: number, chars: number) =>
      rows > 0 ? { rows, chars, description: description ?? '' } : null
    const uploadedOod = oodUpload ? ood(oodUpload.filename, oodUpload.rows.length, charCount(oodUpload.rows)) : null
    if (source === 'builtin') {
      if (!builtin) return null
      return {
        name: builtin.name,
        numRows: builtin.num_rows,
        numPositive: builtin.num_positive,
        chars: builtin.total_chars,
        labelNames: builtin.label_names,
        ood:
          uploadedOod ??
          (useBuiltinOod && builtin.ood ? ood(builtin.ood.description, builtin.ood.num_rows, builtin.ood.total_chars) : null),
      }
    }
    if (!upload) return null
    return {
      name: upload.filename,
      numRows: upload.rows.length,
      numPositive: upload.rows.filter((r) => r.label === 1).length,
      chars: charCount(upload.rows),
      labelNames: upload.label_names,
      ood: uploadedOod,
    }
  }, [source, builtin, upload, oodUpload, useBuiltinOod])

  // Cost estimate, refreshed shortly after the settings settle
  useEffect(() => {
    if (!isAuthenticated || !modelId || !stats) {
      setEstimate(null)
      return
    }
    const timer = setTimeout(() => {
      apiClient
        .estimateProbeRun({
          model: modelId,
          num_examples: stats.numRows + (stats.ood?.rows ?? 0),
          total_chars: stats.chars + (stats.ood?.chars ?? 0),
          chat_template: chatTemplate,
        })
        .then(setEstimate)
        .catch(() => setEstimate(null))
    }, 350)
    return () => clearTimeout(timer)
  }, [isAuthenticated, modelId, stats, chatTemplate])

  const datasetSpec = (): ProbeDatasetSpec | null => {
    const ood = oodUpload ? { ood_rows: oodUpload.rows, ood_name: oodUpload.filename } : {}
    if (source === 'builtin') {
      return builtin ? { builtin_id: builtin.id, use_builtin_ood: useBuiltinOod, ...ood } : null
    }
    return upload ? { rows: upload.rows, name: upload.filename, label_names: upload.label_names, ...ood } : null
  }

  const train = async () => {
    if (!isAuthenticated) {
      navigate('/auth')
      return
    }
    const dataset = datasetSpec()
    if (!modelId || !dataset) {
      setError('Choose a model and a dataset first')
      return
    }
    setTraining(true)
    setError(null)
    setView('sweep')
    try {
      const result = await apiClient.trainProbes({
        model: modelId,
        dataset,
        chat_template: chatTemplate,
        read_span: chatTemplate ? readSpan : 'text',
        test_fraction: testFraction,
        seed,
      })
      setRun(result)
      setRunTestFraction(testFraction)
      setPosition(bestPosition(result, pooling, method))
      setEvalSet('test')
      setHeatmapSource('sweep')
      setCausalSource('sweep')
    } catch (err) {
      if (err instanceof Error && err.message === 'INSUFFICIENT_CREDITS') onShowPurchaseCredits?.()
      else setError(err instanceof Error ? err.message : 'Probe training failed')
    } finally {
      setTraining(false)
    }
  }

  return {
    isAuthenticated,
    // models
    models,
    model,
    modelId,
    setModelId,
    loadingModels,
    // datasets
    datasets,
    builtin,
    datasetDetails,
    loadDatasetDetail,
    source,
    setSource,
    builtinId,
    setBuiltinId,
    useBuiltinOod,
    setUseBuiltinOod,
    upload,
    oodUpload,
    parsing,
    parseFile,
    clearUpload: () => setUpload(null),
    clearOodUpload: () => setOodUpload(null),
    swapUploadLabels: () => setUpload((current) => (current ? swapLabels(current) : current)),
    swapOodLabels: () => setOodUpload((current) => (current ? swapLabels(current) : current)),
    stats,
    // settings
    chatTemplate,
    setChatTemplate,
    readSpan,
    setReadSpan,
    testFraction,
    setTestFraction,
    seed,
    setSeed,
    // run
    estimate,
    training,
    train,
    run,
    runTestFraction,
    error,
    setError,
    // views
    view,
    setView,
    pooling,
    setPooling,
    method,
    setMethod,
    metric,
    setMetric,
    evalSet,
    setEvalSet,
    position,
    setPosition,
    // saved probes
    savedProbes,
    loadingSaved,
    addSaved: (probe: SavedProbe) => setSavedProbes((current) => [probe, ...current]),
    removeSaved: (id: number) => {
      setSavedProbes((current) => current.filter((p) => p.id !== id))
      setHeatmapSource((current) => (current === `saved:${id}` ? 'sweep' : current))
      setCausalSource((current) => (current === `saved:${id}` ? 'sweep' : current))
    },
    heatmapSource,
    setHeatmapSource,
    // causal validation
    causalSource,
    setCausalSource,
    causalMethod,
    setCausalMethod,
    datasetSpec,
  }
}

export type ProbeLab = ReturnType<typeof useProbeLab>
