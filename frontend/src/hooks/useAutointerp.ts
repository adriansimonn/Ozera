/**
 * Autointerp on the SAE page: whether an atlas's features can be explained, a feature's
 * explanation, scores and the scorer's mistakes, explaining one (charged), jobs over many
 * features (estimated, then charged and polled), the table and summary of an atlas's
 * explanations, and recomputing an atlas that predates autointerp (an atlas job). Testing
 * explanations: scoring the user's own hypotheses, generation tests, and community names.
 */

import { useEffect, useRef, useState } from 'react'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '../api/client'
import { refKey } from '../components/lab/saeLinks'
import type { AutointerpFeatureRef, AutointerpJob, AutointerpStatus, ExplanationSource, FeaturesRequest, JobRequest } from '../types/autointerp'
import type { AtlasCorpusId, AtlasJob, AtlasTokens } from '../types/geometry'
import type { SaeRef } from '../types/probes'
import { forgetLabels } from './useFeatureLabels'

export interface AutointerpAtlasRef {
  model: string
  sae: SaeRef
  corpus: AtlasCorpusId
}

const ATLAS_POLL_MS = 15_000
// A job's status is polled this often while it runs (its batches are checked at most every 30 s)
const JOB_POLL_MS = 10_000

function atlasKey(ref: AutointerpAtlasRef | null) {
  return [ref ? refKey(ref.sae) : '', ref?.model ?? '', ref?.corpus ?? '']
}

function statusKey(ref: AutointerpAtlasRef | null) {
  return ['autointerp-status', ...atlasKey(ref)]
}

function featureKey(ref: AutointerpFeatureRef | null) {
  return ['autointerp-feature', ...atlasKey(ref), ref?.feature ?? -1]
}

export function isActive(job: AutointerpJob): boolean {
  return job.status === 'queued' || job.status === 'running'
}

/** Whether an atlas's features can be explained, and its jobs (polled while one runs). */
export function useAutointerpStatus(ref: AutointerpAtlasRef | null) {
  return useQuery({
    queryKey: statusKey(ref),
    queryFn: () => apiClient.autointerpStatus(ref as AutointerpAtlasRef),
    enabled: !!ref,
    staleTime: 60_000,
    refetchInterval: (query) => (query.state.data?.jobs.some(isActive) ? JOB_POLL_MS : false),
  })
}

/**
 * Reload what an atlas's explanations feed (the table, the summary, features' details and
 * labels) once a job on it ends.
 */
export function useJobEndRefresh(ref: AutointerpAtlasRef | null, status: AutointerpStatus | undefined) {
  const queryClient = useQueryClient()
  const running = useRef(new Set<string>())
  useEffect(() => {
    if (!ref || !status) return
    let ended = false
    status.jobs.forEach((job) => {
      if (isActive(job)) running.current.add(job.job_id)
      else if (running.current.delete(job.job_id)) ended = true
    })
    if (!ended) return
    const scope = atlasKey(ref)
    ;['autointerp-features', 'autointerp-summary', 'autointerp-feature', 'autointerp-examples', 'autointerp-communities'].forEach((name) =>
      queryClient.invalidateQueries({ queryKey: [name, ...scope] }),
    )
    forgetLabels(queryClient, ref.sae)
    // ref is stable for the status it came with
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status])
}

/** What a job would explain, and its price (debounced by the caller's request object). */
export function useJobEstimate(request: JobRequest | null) {
  return useQuery({
    queryKey: ['autointerp-estimate', request],
    queryFn: () => apiClient.estimateAutointerpJob(request as JobRequest),
    enabled: !!request,
    staleTime: 30_000,
    retry: false,
  })
}

export function useSubmitJob(ref: AutointerpAtlasRef | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: JobRequest) => apiClient.submitAutointerpJob(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: statusKey(ref) })
      queryClient.invalidateQueries({ queryKey: ['autointerp-estimate'] })
    },
  })
}

export function useExplainedFeatures(ref: AutointerpAtlasRef | null, request: Omit<FeaturesRequest, 'model' | 'sae' | 'corpus'>) {
  return useQuery({
    queryKey: ['autointerp-features', ...atlasKey(ref), request],
    queryFn: () => apiClient.explainedFeatures({ ...(ref as AutointerpAtlasRef), ...request }),
    enabled: !!ref,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  })
}

export function useAutointerpSummary(ref: AutointerpAtlasRef | null, pair: { explainer: string; scorer: string } | null) {
  return useQuery({
    queryKey: ['autointerp-summary', ...atlasKey(ref), pair?.explainer ?? '', pair?.scorer ?? ''],
    queryFn: () => apiClient.autointerpSummary({ ...(ref as AutointerpAtlasRef), ...(pair ?? {}) }),
    enabled: !!ref,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  })
}

/** A feature's explain set and the scorer's mistakes: on the explanation shown, or on the explanation or hypothesis named. */
export function useFeatureExamples(ref: AutointerpFeatureRef | null, enabled = true, source: ExplanationSource = {}) {
  return useQuery({
    queryKey: ['autointerp-examples', ...atlasKey(ref), ref?.feature ?? -1, source.explanation_id ?? 0, source.hypothesis_id ?? 0],
    queryFn: () => apiClient.featureExamples({ ...(ref as AutointerpFeatureRef), ...source }),
    enabled: !!ref && enabled,
    staleTime: 60_000,
  })
}

export function useFeatureAutointerp(ref: AutointerpFeatureRef | null, enabled = true) {
  return useQuery({
    queryKey: featureKey(ref),
    queryFn: () => apiClient.autointerpFeature(ref as AutointerpFeatureRef),
    enabled: !!ref && enabled,
    staleTime: 60_000,
  })
}

/** Explain a feature, or explain it again with other models (every explanation is kept; the newest is shown). */
export function useExplainFeature() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (ref: AutointerpFeatureRef & { explainer?: string; scorer?: string }) => apiClient.explainFeature(ref),
    onSuccess: (_, ref) => {
      queryClient.invalidateQueries({ queryKey: featureKey(ref) })
      queryClient.invalidateQueries({ queryKey: statusKey(ref) })
      queryClient.invalidateQueries({ queryKey: ['autointerp-examples', ...atlasKey(ref), ref.feature] })
      queryClient.invalidateQueries({ queryKey: ['autointerp-features', ...atlasKey(ref)] })
      queryClient.invalidateQueries({ queryKey: ['autointerp-summary', ...atlasKey(ref)] })
      forgetLabels(queryClient, ref.sae)
    },
  })
}

/** Score the user's own explanation of a feature (charged; private). */
export function useScoreHypothesis() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: AutointerpFeatureRef & { text: string; scorer?: string }) => apiClient.scoreHypothesis(request),
    onSuccess: (_, ref) => queryClient.invalidateQueries({ queryKey: featureKey(ref) }),
  })
}

export function useDeleteHypothesis() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id }: { ref: AutointerpFeatureRef; id: number }) => apiClient.deleteHypothesis(id),
    onSuccess: (_, { ref }) => queryClient.invalidateQueries({ queryKey: featureKey(ref) }),
  })
}

/** Run a generation test of an explanation or a hypothesis (charged; an explanation's is shared). */
export function useGenerationTest() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: AutointerpFeatureRef & ExplanationSource & { generator?: string }) => apiClient.generationTest(request),
    onSuccess: (_, ref) => queryClient.invalidateQueries({ queryKey: featureKey(ref) }),
  })
}

/** Names of an atlas's communities, by community ID (empty until a naming job has run). */
export function useCommunityNames(ref: AutointerpAtlasRef | null, enabled = true) {
  return useQuery({
    queryKey: ['autointerp-communities', ...atlasKey(ref)],
    queryFn: async () => new Map((await apiClient.communityNames(ref as AutointerpAtlasRef)).map((n) => [n.community, n])),
    enabled: !!ref && enabled,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

/**
 * Recomputing an atlas with the same corpus, tokens and seed (so it gets example sets): its
 * estimate, starting the job, and the job until it ends (then the status is reloaded).
 */
export function useAtlasRecompute(ref: AutointerpAtlasRef | null, status: AutointerpStatus | undefined) {
  const queryClient = useQueryClient()
  const [job, setJob] = useState<AtlasJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const request = ref && status?.atlas
    ? { model: ref.model, sae: ref.sae, corpus: ref.corpus, tokens: status.atlas.requested_tokens as AtlasTokens, seed: status.atlas.seed }
    : null
  const estimate = useQuery({
    queryKey: ['atlas-estimate', request],
    queryFn: () => apiClient.estimateAtlas(request!),
    enabled: !!request && status?.state === 'recompute',
    staleTime: 5 * 60_000,
  })
  const submit = useMutation({
    mutationFn: () => apiClient.submitAtlasJob(request!),
    onMutate: () => setError(null),
    onSuccess: (started) => setJob(started),
    onError: (err: Error) => setError(err.message),
  })

  const running = job != null && (job.status === 'queued' || job.status === 'running')
  useEffect(() => {
    if (!running || !job) return
    const timer = setInterval(async () => {
      try {
        const latest = await apiClient.getAtlasJob(job.job_id)
        setJob(latest)
        if (latest.status === 'completed') queryClient.invalidateQueries({ queryKey: statusKey(ref) })
        if (latest.status === 'failed') setError(latest.error ?? 'The atlas job failed')
      } catch {
        // Try again on the next tick
      }
    }, ATLAS_POLL_MS)
    return () => clearInterval(timer)
    // ref is captured for the job it started
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, job?.job_id])

  return { estimate: estimate.data, start: () => submit.mutate(), starting: submit.isPending, job, running, error }
}
